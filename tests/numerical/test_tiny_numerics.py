"""Numerical tests for the Stage 2 Tiny Transformer."""

from __future__ import annotations

from pathlib import Path

import torch

from tiny_genius.checkpoint import build_optimizer
from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.reproducibility import set_global_seed
from tiny_genius.training.tiny_loop import make_tiny_corpus, next_token_loss, train_steps

REPO_ROOT = Path(__file__).resolve().parents[2]


def _model() -> tuple[TinyModelConfig, TinyTransformer]:
    config = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml")
    return config, TinyTransformer(config)


def test_forward_and_backward_are_finite() -> None:
    set_global_seed(11)
    config, model = _model()
    tokens = make_tiny_corpus(config.vocab_size, seq_len=32, batch_size=2)
    loss = next_token_loss(model, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    nonzero = sum(int((g != 0).any()) for g in grads if g is not None)
    assert nonzero >= len(grads) - 1


def test_loss_decreases_on_tiny_corpus() -> None:
    set_global_seed(12)
    config, model = _model()
    optimizer = build_optimizer(model, lr=3e-3)
    tokens = make_tiny_corpus(config.vocab_size, seq_len=32, batch_size=4)
    losses = train_steps(model, optimizer, tokens, n_steps=25)
    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < losses[0]
    assert min(losses) < losses[0]
