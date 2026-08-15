#!/usr/bin/env python3
"""Bounded Stage 4 GPU smoke. Not a Stage 5 trainer. Not pretraining.

Validates STAGE4_SMOKE → frozen tokenizer → packing → batch → Tiny Transformer
BF16 forward/backward → optimizer → checkpoint on one GPU.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from tiny_genius.checkpoint import (
    build_optimizer,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from tiny_genius.config import REPO_ROOT
from tiny_genius.data.packing import pack_documents
from tiny_genius.data.stage4_smoke import (
    SMOKE_ID,
    TOTAL_TOKENS,
    load_manifest_shards,
    load_stage4_smoke,
)
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT
from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.reproducibility import collect_environment, set_global_seed
from tiny_genius.tokenizer import Tokenizer
from tiny_genius.tokenizer.specials import EOS_ID, PAD_ID
from tiny_genius.training.tiny_loop import next_token_loss

SMOKE_TEXTS = [
    "def add(a: int, b: int) -> int:\n    return a + b\n",
    "def fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n"
    "        a, b = b, a + b\n    return a\n",
    "E = m * c * c\n",
    "The gradient of the loss is back-propagated through the residual stream.\n",
]


def make_batch(tok: Tokenizer, n_ctx: int, batch_size: int) -> tuple[torch.Tensor, dict]:
    docs = []
    for text in SMOKE_TEXTS:
        ids = tok.encode(text, add_bos=False, add_eos=True)
        if ids[-1] != EOS_ID:
            ids.append(EOS_ID)
        docs.append({"token_ids": ids})
    sequences, stats = pack_documents(docs, n_ctx=n_ctx, shard_target_tokens=n_ctx * 8)
    if not sequences:
        raise RuntimeError("packing produced no sequences")
    while len(sequences) < batch_size:
        sequences.extend(sequences)
    batch = torch.tensor(sequences[:batch_size], dtype=torch.long)
    if int(batch.max()) >= tok.vocab_size or int(batch.min()) < 0:
        raise RuntimeError("token ids out of range")
    if batch.shape[1] != n_ctx:
        raise RuntimeError(f"expected seq {n_ctx}, got {batch.shape}")
    return batch, stats


def build_model(tok: Tokenizer, n_ctx: int, device: torch.device) -> tuple[TinyTransformer, str]:
    base = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml")
    config = TinyModelConfig(
        n_layers=base.n_layers,
        d_model=base.d_model,
        n_heads=base.n_heads,
        d_head=base.d_head,
        d_ff=base.d_ff,
        vocab_size=tok.vocab_size,
        n_ctx=n_ctx,
        rms_norm_eps=base.rms_norm_eps,
        rope_theta=base.rope_theta,
        tie_embeddings=True,
        use_bias=False,
        dropout=0.0,
        stage=2,
        name="stage4-gpu-smoke",
        frozen=False,
    )
    model = TinyTransformer(config).to(device)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(dtype=torch.bfloat16)
        dtype = "bf16"
    else:
        dtype = str(next(model.parameters()).dtype)
    return model, dtype


def smoke_run(
    *,
    tok: Tokenizer,
    n_ctx: int,
    batch_size: int,
    steps: int,
    device: torch.device,
    ckpt_path: Path,
) -> dict:
    batch, pack_stats = make_batch(tok, n_ctx, batch_size)
    dataset = torch.utils.data.TensorDataset(batch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loaded = next(iter(loader))[0]
    if not torch.equal(loaded, batch):
        raise RuntimeError("DataLoader did not reproduce packed batch")
    batch = loaded.to(device)
    model, dtype = build_model(tok, n_ctx, device)
    optimizer = build_optimizer(model, lr=1e-3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {loss.item()}")
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError("non-finite gradient")
        optimizer.step()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and not torch.isfinite(p).all():
                    raise RuntimeError("non-finite optimizer parameter")
        losses.append(float(loss.detach().float().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak = 0.0
    elapsed = time.perf_counter() - t0
    tokens = steps * batch.numel()
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        step=steps,
        metadata={"smoke": True, "identity": SMOKE_ID},
    )
    payload = load_checkpoint(ckpt_path)
    if payload["step"] != steps:
        raise RuntimeError("checkpoint step mismatch")
    reloaded, opt2, step = restore_training_state(
        ckpt_path, model=None, optimizer=None, restore_rng=False
    )
    if step != steps:
        raise RuntimeError("restore step mismatch")
    # Compare weights after mapping to CPU float32 for bf16/device differences
    orig = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    rest = {k: v.detach().float().cpu() for k, v in reloaded.state_dict().items()}
    for key in orig:
        if not torch.allclose(orig[key], rest[key], atol=0.0, rtol=0.0):
            raise RuntimeError(f"checkpoint weight mismatch {key}")
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    return {
        "n_ctx": n_ctx,
        "batch_size": batch_size,
        "steps": steps,
        "gradient_accumulation": 1,
        "dtype": dtype,
        "vocab_size": tok.vocab_size,
        "pad_id": PAD_ID,
        "eos_id": EOS_ID,
        "packing_waste": pack_stats.get("waste_ratio"),
        "n_pack_sequences": pack_stats.get("n_sequences"),
        "n_pack_shards": pack_stats.get("n_shards"),
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "finite_loss": all(torch.isfinite(torch.tensor(losses))),
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_sec": tokens / elapsed if elapsed else None,
        "peak_vram_mib": peak,
        "checkpoint": str(ckpt_path),
        "checkpoint_exists": ckpt_path.is_file(),
        "checkpoint_reload_ok": True,
        "checkpoint_sha256": ckpt_hash,
        "tokenizer_fingerprint": tok.fingerprint,
    }


def main() -> int:
    identity = load_stage4_smoke()
    tok = Tokenizer.load_frozen()
    if tok.fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError("tokenizer fingerprint mismatch")
    shards = load_manifest_shards()
    admitted = [s for s in shards if s["status"] == "admitted" and s["token_count"] > 0]
    if sum(s["token_count"] for s in admitted) != TOTAL_TOKENS:
        raise RuntimeError("admitted shard token sum != STAGE4_SMOKE total")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = collect_environment(seed=42)
    env_report = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "python": env.get("python_version"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_mib": (
            torch.cuda.get_device_properties(0).total_memory / (1024**2)
            if torch.cuda.is_available()
            else None
        ),
        "bf16_supported": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        "identity": identity["identity"],
        "corpus_hash": identity["corpus_hash"],
        "total_tokens": identity["total_tokens"],
        "python_tokens": identity["python_tokens"],
        "math_tokens": identity["math_tokens"],
        "stem_tokens": identity["stem_tokens"],
        "gate_g4": identity["gate_g4"],
        "n_manifest_shards": len(shards),
        "n_admitted_positive_shards": len(admitted),
    }
    print(json.dumps({"environment": env_report}, indent=2))

    set_global_seed(42)
    native = smoke_run(
        tok=tok,
        n_ctx=256,
        batch_size=2,
        steps=8,
        device=device,
        ckpt_path=Path("/tmp/stage4_gpu_smoke.pt"),
    )
    print(json.dumps({"native_tiny_ctx": native}, indent=2))

    set_global_seed(42)
    native_rerun = smoke_run(
        tok=tok,
        n_ctx=256,
        batch_size=2,
        steps=8,
        device=device,
        ckpt_path=Path("/tmp/stage4_gpu_smoke_rerun.pt"),
    )
    det_ok = native["losses"] == native_rerun["losses"]
    if device.type == "cpu" and not det_ok:
        det_ok = all(
            abs(a - b) < 1e-5 for a, b in zip(native["losses"], native_rerun["losses"], strict=True)
        )
    if not det_ok:
        raise RuntimeError("seed-42 rerun losses diverged")

    set_global_seed(42)
    long_ctx = smoke_run(
        tok=tok,
        n_ctx=4096,
        batch_size=1,
        steps=2,
        device=device,
        ckpt_path=Path("/tmp/stage4_gpu_smoke_4096.pt"),
    )
    print(json.dumps({"packing_ctx_4096": long_ctx}, indent=2))

    report = {
        "identity": SMOKE_ID,
        "is_10m_milestone": False,
        "gate_g4": "FAIL",
        "gate_g4_reason": identity["gate_g4_reason"],
        "corpus_hash": identity["corpus_hash"],
        "environment": env_report,
        "native_tiny_ctx": native,
        "native_tiny_ctx_rerun": native_rerun,
        "packing_ctx_4096": long_ctx,
        "deterministic_rerun": det_ok,
        "seed": 42,
        "tokenizer_fingerprint": tok.fingerprint,
    }
    out = REPO_ROOT / "artifacts" / "stage4_smoke" / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "deterministic_rerun": det_ok}, indent=2))
    print("stage4_gpu_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
