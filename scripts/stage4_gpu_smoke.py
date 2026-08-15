#!/usr/bin/env python3
"""Bounded Stage 4 GPU smoke test. Not a Stage 5 trainer. Not pretraining.

Proves Stage 4 artifacts (frozen tokenizer, packing, manifests) can feed the
existing Tiny Transformer architecture on one GPU for a few steps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from tiny_genius.checkpoint import build_optimizer, save_checkpoint
from tiny_genius.config import REPO_ROOT
from tiny_genius.data.packing import pack_documents
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT, load_verified_tokenizer
from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.reproducibility import set_global_seed
from tiny_genius.tokenizer.specials import EOS_ID, PAD_ID
from tiny_genius.training.tiny_loop import next_token_loss

SMOKE_TEXTS = [
    "def add(a: int, b: int) -> int:\n    return a + b\n",
    "def fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n"
    "        a, b = b, a + b\n    return a\n",
    "E = m * c * c\n",
    "The gradient of the loss is back-propagated through the residual stream.\n",
]


def load_manifest_summary() -> dict:
    path = REPO_ROOT / "manifests" / "10m" / "mixture_summary.json"
    if not path.is_file():
        return {"missing": True}
    return json.loads(path.read_text(encoding="utf-8"))


def make_batch(tok, n_ctx: int, batch_size: int) -> torch.Tensor:
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
    if batch.max().item() >= tok.vocab_size or batch.min().item() < 0:
        raise RuntimeError("token ids out of range")
    if batch.shape[1] != n_ctx:
        raise RuntimeError(f"expected seq {n_ctx}, got {batch.shape}")
    return batch, stats


def smoke_run(*, n_ctx: int, batch_size: int, steps: int, device: torch.device) -> dict:
    tok = load_verified_tokenizer()
    if tok.fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError("tokenizer fingerprint mismatch")
    batch, pack_stats = make_batch(tok, n_ctx, batch_size)
    batch = batch.to(device)
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
    optimizer = build_optimizer(model, lr=1e-3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # next_token_loss expects long targets; cast inputs if model is bf16
        tokens = batch
        loss = next_token_loss(model, tokens)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {loss.item()}")
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError("non-finite gradient")
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak = 0.0
    elapsed = time.perf_counter() - t0
    tokens = steps * batch.numel()
    ckpt = Path("/tmp/stage4_gpu_smoke.pt")
    save_checkpoint(ckpt, model=model, optimizer=optimizer, step=steps, metadata={"smoke": True})
    return {
        "n_ctx": n_ctx,
        "batch_size": batch_size,
        "steps": steps,
        "dtype": dtype,
        "vocab_size": tok.vocab_size,
        "pad_id": PAD_ID,
        "eos_id": EOS_ID,
        "packing_waste": pack_stats.get("waste_ratio"),
        "losses": losses,
        "finite_loss": all(torch.isfinite(torch.tensor(losses))),
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_sec": tokens / elapsed if elapsed else None,
        "peak_vram_mib": peak,
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.is_file(),
        "tokenizer_fingerprint": tok.fingerprint,
    }


def main() -> int:
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixture = load_manifest_summary()
    print(json.dumps({"device": str(device), "mixture": mixture}, indent=2))
    # Implemented debug model context is 256. Also probe 4096 packing length on GPU.
    native = smoke_run(n_ctx=256, batch_size=2, steps=8, device=device)
    print(json.dumps({"native_tiny_ctx": native}, indent=2))
    long_ctx = smoke_run(n_ctx=4096, batch_size=1, steps=2, device=device)
    print(json.dumps({"packing_ctx_4096": long_ctx}, indent=2))
    print("stage4_gpu_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
