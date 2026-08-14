"""Stage 2 integration: tiny model learns and resumes from checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from tiny_genius.checkpoint import (
    build_optimizer,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.reproducibility import set_global_seed
from tiny_genius.training.tiny_loop import make_tiny_corpus, train_steps

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS = 20
SPLIT = 10


def test_tiny_model_learns_and_resumes(tmp_path: Path) -> None:
    config = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml")
    tokens = make_tiny_corpus(config.vocab_size, seq_len=32, batch_size=4)

    set_global_seed(21)
    continuous = TinyTransformer(config)
    opt_cont = build_optimizer(continuous, lr=3e-3)
    continuous_losses = train_steps(continuous, opt_cont, tokens, STEPS)
    assert continuous_losses[-1] < continuous_losses[0]

    set_global_seed(21)
    first = TinyTransformer(config)
    opt_first = build_optimizer(first, lr=3e-3)
    first_losses = train_steps(first, opt_first, tokens, SPLIT)
    ckpt = tmp_path / "resume.pt"
    save_checkpoint(ckpt, model=first, optimizer=opt_first, step=SPLIT)
    assert ckpt.is_file()

    payload = load_checkpoint(ckpt)
    assert payload["step"] == SPLIT
    assert payload["optimizer"] is not None
    assert payload["config"]["name"] == "tiny"

    set_global_seed(999)
    resumed = TinyTransformer(config)
    opt_resumed = build_optimizer(resumed, lr=3e-3)
    restore_training_state(ckpt, model=resumed, optimizer=opt_resumed, restore_rng=True)
    for key, value in first.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])

    resumed_losses = train_steps(resumed, opt_resumed, tokens, STEPS - SPLIT)
    combined = first_losses + resumed_losses
    assert len(combined) == len(continuous_losses)
    assert combined[0] == continuous_losses[0]
    for left, right in zip(combined, continuous_losses, strict=True):
        assert abs(left - right) < 1e-5
    assert resumed_losses[-1] < first_losses[0]
