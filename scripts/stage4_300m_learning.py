#!/usr/bin/env python3
"""Bounded 300M learning experiment on STAGE4_SMOKE.

Not Stage 5 production training. Does not freeze 10M. Does not alter
tokenizer, thresholds, or STAGE4_SMOKE manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from tiny_genius.checkpoint import (
    build_optimizer,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from tiny_genius.config import REPO_ROOT
from tiny_genius.data.packing import pack_documents
from tiny_genius.data.pipeline import (
    apply_mixture_cap,
    collect_source_docs,
    load_pipeline_config,
    run_stages,
)
from tiny_genius.data.stage4_smoke import (
    MATH_TOKENS,
    PYTHON_TOKENS,
    STEM_TOKENS,
    TOTAL_TOKENS,
    load_stage4_smoke,
)
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT
from tiny_genius.model import TinyModelConfig, TinyTransformer, expected_parameter_count
from tiny_genius.reproducibility import collect_environment, git_commit, set_global_seed
from tiny_genius.tokenizer import Tokenizer
from tiny_genius.tokenizer.specials import EOS_ID, PAD_ID

OUT_DIR = REPO_ROOT / "artifacts" / "stage4_300m_learning"
PACKED_PATH = OUT_DIR / "packed_sequences.pt"
METRICS_PATH = OUT_DIR / "metrics.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
CKPT_DIR = OUT_DIR / "checkpoints"

PYTHON_SOURCES = ("codecontests_plus",)
MATH_SOURCES = ("openmathinstruct_2",)


class CheckpointedTransformer(TinyTransformer):
    """Same architecture; activation checkpointing for A10 memory only."""

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be (batch, seq), got {tuple(token_ids.shape)}")
        if token_ids.shape[1] > self.config.n_ctx:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds n_ctx {self.config.n_ctx}"
            )
        x = self.drop(self.embed(token_ids))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = self.final_norm(x)
        return self.embed.unembed(x)


def masked_next_token_loss(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    vocab = logits.size(-1)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=PAD_ID,
    )


def verify_prereqs() -> dict[str, Any]:
    identity = load_stage4_smoke()
    tok = Tokenizer.load_frozen()
    if tok.fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            f"tokenizer fingerprint mismatch: {tok.fingerprint} != {EXPECTED_FINGERPRINT}"
        )
    if (REPO_ROOT / "manifests" / "10m" / "FROZEN_10M.json").is_file():
        raise RuntimeError("FROZEN_10M.json must not exist")
    if identity["total_tokens"] != TOTAL_TOKENS:
        raise RuntimeError("STAGE4_SMOKE token count drifted")
    return {"identity": identity, "tokenizer": tok}


def rematerialize_tokens(tok: Tokenizer, max_docs: int) -> dict[str, Any]:
    """Rebuild token_ids from the same authorized sources. Does not rewrite manifests."""
    thresholds, data_cfg, registry = load_pipeline_config()
    sources = {s["source_id"]: s for s in registry["sources"]}
    cap = int(thresholds["thresholds"]["codecontests_max_solutions_per_problem"]["value"])
    docs: list[dict[str, Any]] = []
    fetch_log = {}
    for source_id in (*PYTHON_SOURCES, *MATH_SOURCES):
        source = sources[source_id]
        fetched, err = collect_source_docs(
            source, data_cfg["collection_date"], cap=cap, max_docs=max_docs
        )
        fetch_log[source_id] = {"n_docs": len(fetched), "error": err}
        if err and not fetched:
            raise RuntimeError(f"{source_id}: {err}")
        docs.extend(fetched)
    result = run_stages(
        docs,
        thresholds=thresholds,
        data_cfg=data_cfg,
        refs_dir=REPO_ROOT / data_cfg["contamination_refs_dir"],
        tokenize=True,
    )
    targets = {
        "python": PYTHON_TOKENS,
        "math": MATH_TOKENS,
        "stem": STEM_TOKENS,
    }
    kept = apply_mixture_cap(result["docs"], targets)
    by_domain = {"python": 0, "math": 0, "stem": 0}
    for doc in kept:
        by_domain[doc["domain"]] += int(doc["token_count"])
    return {
        "docs": kept,
        "by_domain": by_domain,
        "fetch_log": fetch_log,
        "packing_from_pipeline": result.get("packing") or {},
        "n_docs": len(kept),
    }


def pack_for_train(docs: list[dict[str, Any]], n_ctx: int) -> tuple[torch.Tensor, dict[str, Any]]:
    sequences, stats = pack_documents(
        docs, n_ctx=n_ctx, shard_target_tokens=max(n_ctx * 8, 1)
    )
    if not sequences:
        raise RuntimeError("packing produced no sequences")
    tensor = torch.tensor(sequences, dtype=torch.long)
    min_id = int(tensor.min())
    max_id = int(tensor.max())
    oob = int(((tensor < 0) | (tensor >= 32768)).sum())
    stats = dict(stats)
    stats.update(
        {
            "min_token_id": min_id,
            "max_token_id": max_id,
            "out_of_range_ids": oob,
            "pad_id": PAD_ID,
            "eos_id": EOS_ID,
            "tensor_shape": list(tensor.shape),
        }
    )
    if oob:
        raise RuntimeError(f"out-of-range token ids: {oob}")
    return tensor, stats


def split_holdout(tensor: torch.Tensor, frac: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    n = tensor.shape[0]
    n_eval = max(1, int(n * frac))
    return tensor[n_eval:], tensor[:n_eval]


def param_update_norm(model: torch.nn.Module, prev: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name not in prev:
            continue
        total += float((param.detach().float().cpu() - prev[name]).pow(2).sum())
    return math.sqrt(total)


def snapshot_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu().clone() for k, v in model.named_parameters()}


def evaluate(model: torch.nn.Module, batch: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        loss = masked_next_token_loss(model, batch)
    model.train()
    return float(loss.detach().float().cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 = one pass over train sequences",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--max-docs-per-source", type=int, default=25000)
    parser.add_argument(
        "--rematerialize",
        action="store_true",
        help="Rebuild packed tokens from HF sources (slow). Default reuses packed.pt.",
    )
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(42)
    pre = verify_prereqs()
    tok = pre["tokenizer"]
    identity = pre["identity"]
    cfg = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "model_300m.yaml")
    if cfg.vocab_size != tok.vocab_size:
        raise RuntimeError("300M vocab_size must match frozen tokenizer")
    expected = expected_parameter_count(cfg)

    if PACKED_PATH.is_file() and not args.rematerialize:
        packed = torch.load(PACKED_PATH, map_location="cpu", weights_only=False)
        sequences = packed["sequences"]
        pack_stats = packed["pack_stats"]
        remat = packed.get("rematerialize") or {}
    else:
        remat = rematerialize_tokens(tok, args.max_docs_per_source)
        sequences, pack_stats = pack_for_train(remat["docs"], n_ctx=args.seq_len)
        torch.save(
            {
                "sequences": sequences,
                "pack_stats": pack_stats,
                "rematerialize": {
                    "by_domain": remat["by_domain"],
                    "fetch_log": remat["fetch_log"],
                    "n_docs": remat["n_docs"],
                },
            },
            PACKED_PATH,
        )
        remat = {
            "by_domain": remat["by_domain"],
            "fetch_log": remat["fetch_log"],
            "n_docs": remat["n_docs"],
        }

    train_seq, eval_seq = split_holdout(sequences)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.no_checkpointing:
        model = TinyTransformer(cfg).to(device)
    else:
        model = CheckpointedTransformer(cfg).to(device)
    actual_params = model.parameter_count()
    if actual_params != expected:
        raise RuntimeError(f"param count {actual_params} != {expected}")
    bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    if bf16:
        model = model.to(dtype=torch.bfloat16)
    optimizer = build_optimizer(model, lr=args.lr)
    # build_optimizer default lr is 3e-3; force documented experiment LR
    for group in optimizer.param_groups:
        group["lr"] = args.lr

    n_train = train_seq.shape[0]
    steps_per_epoch = math.ceil(n_train / (args.microbatch * args.grad_accum))
    max_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch
    eval_batch = eval_seq[: args.microbatch].to(device)

    env = collect_environment(seed=42)
    header = {
        "experiment": "300M-model learning/throughput validation on STAGE4_SMOKE",
        "not_stage5": True,
        "gate_g4": "FAIL",
        "identity": identity["identity"],
        "corpus_hash": identity["corpus_hash"],
        "manifest_tokens": {
            "python": PYTHON_TOKENS,
            "math": MATH_TOKENS,
            "stem": STEM_TOKENS,
            "total": TOTAL_TOKENS,
        },
        "rematerialized_tokens": remat.get("by_domain"),
        "tokenizer_fingerprint": tok.fingerprint,
        "vocab_size": tok.vocab_size,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "pack_stats": pack_stats,
        "model": cfg.to_dict(),
        "parameter_count": actual_params,
        "expected_parameter_count": expected,
        "microbatch": args.microbatch,
        "grad_accum": args.grad_accum,
        "effective_batch_sequences": args.microbatch * args.grad_accum,
        "seq_len": args.seq_len,
        "optimizer": "AdamW",
        "lr": args.lr,
        "scheduler": "constant",
        "bf16": bf16,
        "activation_checkpointing": not args.no_checkpointing,
        "seed": 42,
        "git_commit": git_commit(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "n_train_sequences": n_train,
        "n_eval_sequences": int(eval_seq.shape[0]),
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "environment": env,
    }
    (OUT_DIR / "run_header.json").write_text(
        json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: header[k] for k in (
        "parameter_count", "seq_len", "microbatch", "grad_accum", "max_steps",
        "bf16", "corpus_hash", "gate_g4", "rematerialized_tokens",
    )}, indent=2))

    metrics_handle = METRICS_PATH.open("w", encoding="utf-8")
    nonfinite = 0
    oom = 0
    losses: list[float] = []
    gnoms: list[float] = []
    tokens_seen = 0
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prev_snap = snapshot_cpu(model)
    cursor = 0
    step = 0
    initial_eval = evaluate(model, eval_batch)
    print(
        json.dumps(
            {
                "eval_loss_step0": initial_eval,
                "eval_ppl_step0": math.exp(min(initial_eval, 20)),
            }
        )
    )

    model.train()
    try:
        while step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_tokens = 0
            for _ in range(args.grad_accum):
                if cursor >= n_train:
                    cursor = 0
                end = min(cursor + args.microbatch, n_train)
                batch = train_seq[cursor:end].to(device)
                if batch.shape[0] < args.microbatch:
                    extra = train_seq[: args.microbatch - batch.shape[0]].to(device)
                    batch = torch.cat([batch, extra], dim=0)
                cursor = end
                step_tokens += int((batch != PAD_ID).sum().item())
                try:
                    loss = masked_next_token_loss(model, batch)
                except torch.cuda.OutOfMemoryError:
                    oom += 1
                    torch.cuda.empty_cache()
                    raise
                if not torch.isfinite(loss):
                    nonfinite += 1
                    raise RuntimeError(f"non-finite loss at step {step}")
                (loss / args.grad_accum).backward()
                accum_loss += float(loss.detach().float().cpu())
            grad_sq = 0.0
            for param in model.parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    nonfinite += 1
                    raise RuntimeError(f"non-finite grad at step {step}")
                grad_sq += float(param.grad.detach().float().pow(2).sum().cpu())
            grad_norm = math.sqrt(grad_sq)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            tokens_seen += step_tokens
            mean_loss = accum_loss / args.grad_accum
            losses.append(mean_loss)
            gnoms.append(grad_norm)
            upd = 0.0
            if step == 1 or step % args.log_every == 0 or step == max_steps:
                upd = param_update_norm(model, prev_snap)
                prev_snap = snapshot_cpu(model)
                elapsed = time.perf_counter() - t0
                peak = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                )
                eval_loss = evaluate(model, eval_batch)
                row = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "frac_corpus": tokens_seen / TOTAL_TOKENS,
                    "lr": args.lr,
                    "train_loss": mean_loss,
                    "train_ppl": math.exp(min(mean_loss, 20)),
                    "eval_loss": eval_loss,
                    "eval_ppl": math.exp(min(eval_loss, 20)),
                    "grad_norm": grad_norm,
                    "update_norm": upd,
                    "tokens_per_sec": tokens_seen / elapsed if elapsed else None,
                    "elapsed_sec": elapsed,
                    "peak_vram_mib": peak,
                    "nonfinite_events": nonfinite,
                    "oom_events": oom,
                }
                metrics_handle.write(json.dumps(row) + "\n")
                metrics_handle.flush()
                print(json.dumps(row))
            if step % args.ckpt_every == 0 or step == max_steps:
                ckpt = CKPT_DIR / f"step_{step:05d}.pt"
                save_checkpoint(
                    ckpt,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    metadata={"experiment": "stage4_300m_learning", "gate_g4": "FAIL"},
                )
    finally:
        metrics_handle.close()

    final_ckpt = CKPT_DIR / "final.pt"
    save_checkpoint(
        final_ckpt,
        model=model,
        optimizer=optimizer,
        step=step,
        metadata={"experiment": "stage4_300m_learning", "gate_g4": "FAIL"},
    )
    payload = load_checkpoint(final_ckpt)
    restored, _, rst_step = restore_training_state(final_ckpt, model=None, optimizer=None)
    reload_ok = rst_step == step
    # compare a subset of weights
    a = next(iter(model.state_dict().values())).detach().float().cpu().flatten()[:64]
    b = next(iter(restored.state_dict().values())).detach().float().cpu().flatten()[:64]
    reload_match = bool(torch.allclose(a, b))
    # resume one extra step
    resume_ok = True
    try:
        model2 = restored.to(device)
        if bf16:
            model2 = model2.to(dtype=torch.bfloat16)
        opt2 = build_optimizer(model2, lr=args.lr)
        opt2.load_state_dict(payload["optimizer"])
        batch = train_seq[: args.microbatch].to(device)
        loss = masked_next_token_loss(model2, batch)
        loss.backward()
        opt2.step()
    except Exception as exc:  # noqa: BLE001
        resume_ok = False
        resume_err = str(exc)
    else:
        resume_err = None

    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    util = None
    if device.type == "cuda":
        try:
            import subprocess

            util = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        except OSError:
            util = None

    ppls = [math.exp(min(x, 20)) for x in losses]
    summary = {
        **header,
        "steps_completed": step,
        "tokens_processed": tokens_seen,
        "epochs_approx": tokens_seen / TOTAL_TOKENS,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "initial_ppl": ppls[0] if ppls else None,
        "final_ppl": ppls[-1] if ppls else None,
        "best_ppl": min(ppls) if ppls else None,
        "eval_loss_step0": initial_eval,
        "eval_ppl_step0": math.exp(min(initial_eval, 20)),
        "mean_grad_norm": sum(gnoms) / len(gnoms) if gnoms else None,
        "max_grad_norm": max(gnoms) if gnoms else None,
        "nan_inf_events": nonfinite,
        "oom_events": oom,
        "tokens_per_sec": tokens_seen / elapsed if elapsed else None,
        "peak_vram_mib": peak,
        "gpu_utilization_last": util,
        "runtime_sec": elapsed,
        "checkpoint_sha256": hashlib.sha256(final_ckpt.read_bytes()).hexdigest(),
        "checkpoint_reload_ok": reload_ok and reload_match,
        "resume_extra_step_ok": resume_ok,
        "resume_error": resume_err,
        "packed_sha256": hashlib.sha256(PACKED_PATH.read_bytes()).hexdigest()
        if PACKED_PATH.is_file()
        else None,
        "gate_g4": "FAIL",
        "description": (
            "300M-model learning/throughput validation on the 9.23M-token "
            "STAGE4_SMOKE corpus."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(SUMMARY_PATH),
                "steps": step,
                "final_loss": summary["final_loss"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
