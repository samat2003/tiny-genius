"""Single-process checkpoint format for the Stage 2 Tiny Transformer.

A checkpoint is a torch-saved dict with these keys:

- format: "tiny-genius-checkpoint"
- format_version: int
- step: training step completed
- config: TinyModelConfig.to_dict()
- model: model.state_dict()
- optimizer: optimizer.state_dict() or None
- rng: python / torch RNG payloads
- metadata: optional extra mapping

This is not a distributed or sharded checkpoint system.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.reproducibility import collect_environment

CHECKPOINT_FORMAT = "tiny-genius-checkpoint"
CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: TinyTransformer,
    optimizer: Optimizer | None,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "step": int(step),
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "rng": capture_rng_state(),
        "metadata": {
            **(metadata or {}),
            "environment": collect_environment(seed=None),
        },
    }
    torch.save(payload, dest)
    return dest


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unknown checkpoint format: {payload.get('format')}")
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('format_version')}")
    return payload


def restore_training_state(
    path: str | Path,
    *,
    model: TinyTransformer | None = None,
    optimizer: Optimizer | None = None,
    restore_rng: bool = True,
) -> tuple[TinyTransformer, Optimizer | None, int]:
    """Load model/optimizer/step and optionally restore RNG for resume."""
    payload = load_checkpoint(path)
    config = TinyModelConfig.from_dict(payload["config"])
    if model is None:
        model = TinyTransformer(config)
    else:
        if model.config != config:
            raise ValueError("Checkpoint config does not match the provided model")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return model, optimizer, int(payload["step"])


def build_optimizer(model: nn.Module, lr: float = 3e-3) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
