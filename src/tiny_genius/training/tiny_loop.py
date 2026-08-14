"""Deterministic next-token training loop for the Tiny Transformer."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer


def next_token_loss(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on tokens[:, :-1] → tokens[:, 1:]."""
    logits = model(tokens[:, :-1])
    vocab = logits.size(-1)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        tokens[:, 1:].reshape(-1),
    )


def train_steps(
    model: nn.Module,
    optimizer: Optimizer,
    tokens: torch.Tensor,
    n_steps: int,
) -> list[float]:
    """Run `n_steps` updates on a fixed batch. Returns per-step loss values."""
    model.train()
    losses: list[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, tokens)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def make_tiny_corpus(
    vocab_size: int,
    seq_len: int,
    batch_size: int = 4,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Fixed repeating modular sequences (no tokenizer)."""
    rows = []
    for batch_idx in range(batch_size):
        start = (batch_idx * 3) % max(vocab_size - 1, 1)
        seq = [(start + i) % (vocab_size - 1) + 1 for i in range(seq_len)]
        rows.append(seq)
    return torch.tensor(rows, dtype=torch.long, device=device)
