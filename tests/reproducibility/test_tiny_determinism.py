"""Deterministic Tiny Transformer training on CPU."""

from __future__ import annotations

from pathlib import Path

from tiny_genius.checkpoint import build_optimizer
from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.reproducibility import set_global_seed
from tiny_genius.training.tiny_loop import make_tiny_corpus, train_steps

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_same_seed_same_loss_trajectory() -> None:
    config = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml")
    tokens = make_tiny_corpus(config.vocab_size, seq_len=24, batch_size=2)

    def run(seed: int) -> list[float]:
        set_global_seed(seed)
        model = TinyTransformer(config)
        optimizer = build_optimizer(model, lr=3e-3)
        return train_steps(model, optimizer, tokens, n_steps=5)

    first = run(33)
    second = run(33)
    third = run(34)
    assert first == second
    assert first != third
