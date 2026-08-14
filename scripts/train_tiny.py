#!/usr/bin/env python3
"""Train the Stage 2 Tiny Transformer on a deterministic debug corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_genius import TinyModelConfig, TinyTransformer, set_global_seed
from tiny_genius.checkpoint import build_optimizer, save_checkpoint
from tiny_genius.config import REPO_ROOT
from tiny_genius.training.tiny_loop import make_tiny_corpus, train_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "tiny.yaml",
        help="Path to Stage 2 tiny.yaml",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tiny_stage2.pt",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    config = TinyModelConfig.from_yaml(args.config)
    model = TinyTransformer(config)
    optimizer = build_optimizer(model, lr=args.lr)
    tokens = make_tiny_corpus(config.vocab_size, args.seq_len, args.batch_size)
    losses = train_steps(model, optimizer, tokens, args.steps)
    dest = save_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        metadata={"losses": losses},
    )
    print(f"config: {config.name} stage={config.stage} frozen={config.frozen}")
    print(f"parameters: {model.parameter_count()}")
    print(f"initial_loss: {losses[0]:.6f}")
    print(f"final_loss: {losses[-1]:.6f}")
    print(f"steps: {args.steps}")
    print(f"checkpoint: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
