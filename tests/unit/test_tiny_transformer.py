"""Unit tests for Stage 2 Tiny Transformer mechanics."""

from __future__ import annotations

from pathlib import Path

import torch

from tiny_genius.checkpoint import build_optimizer, restore_training_state, save_checkpoint
from tiny_genius.model import (
    RMSNorm,
    SwiGLU,
    TinyModelConfig,
    TinyTransformer,
    apply_rope,
    build_rope_cache,
    causal_mask,
    expected_parameter_count,
)
from tiny_genius.reproducibility import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[2]


def official_tiny_config() -> TinyModelConfig:
    return TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml")


def test_tiny_config_matches_project_plan() -> None:
    config = official_tiny_config()
    assert config.n_layers == 4
    assert config.d_model == 256
    assert config.n_heads == 4
    assert config.d_head == 64
    assert 512 <= config.d_ff <= 768
    assert config.n_ctx == 256
    assert config.vocab_size < 1024
    assert config.tie_embeddings is True
    assert config.use_bias is False
    assert config.stage == 2
    assert config.frozen is False


def test_tiny_config_rejects_300m_values() -> None:
    data = official_tiny_config().to_dict()
    data["n_layers"] = 24
    data["d_model"] = 960
    data["n_heads"] = 15
    # 960 != 15*64? 15*64=960 actually. d_ff 2560 is out of range.
    data["d_ff"] = 2560
    try:
        TinyModelConfig.from_dict(data)
    except ValueError as exc:
        assert "512" in str(exc) or "768" in str(exc)
    else:
        raise AssertionError("300M FFN size must be rejected for TinyModelConfig")


def test_causal_mask_blocks_future() -> None:
    mask = causal_mask(4, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (4, 4)
    assert torch.isneginf(mask[0, 1])
    assert mask[1, 0] == 0
    assert torch.isfinite(mask.diag()).all()
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.isneginf(mask[future]).all()
    assert (mask[~future] == 0).all()


def test_causal_attention_ignores_future_tokens() -> None:
    set_global_seed(0)
    config = official_tiny_config()
    model = TinyTransformer(config)
    model.eval()
    tokens = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        base = model(tokens)
        mutated = tokens.clone()
        mutated[0, 10:] = (mutated[0, 10:] + 7) % config.vocab_size
        changed = model(mutated)
    assert torch.allclose(base[0, :10], changed[0, :10], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(base[0, 10:], changed[0, 10:], atol=1e-5, rtol=1e-5)


def test_rope_relative_invariance() -> None:
    head_dim = 64
    q = torch.randn(1, 8, head_dim)
    k = torch.randn(1, 8, head_dim)
    cos, sin = build_rope_cache(16, head_dim)
    q0 = apply_rope(q, cos[:8], sin[:8])
    k0 = apply_rope(k, cos[:8], sin[:8])
    q1 = apply_rope(q, cos[3:11], sin[3:11])
    k1 = apply_rope(k, cos[3:11], sin[3:11])
    dots0 = torch.einsum("bsh,bth->bst", q0, k0)
    dots1 = torch.einsum("bsh,bth->bst", q1, k1)
    assert torch.allclose(dots0, dots1, atol=1e-5, rtol=1e-5)


def test_rmsnorm_unit_rms() -> None:
    torch.manual_seed(1)
    norm = RMSNorm(32, eps=1e-6)
    x = torch.randn(2, 5, 32) * 7.0
    y = norm(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_swiglu_shapes_and_gating() -> None:
    torch.manual_seed(2)
    ffn = SwiGLU(16, 48)
    x = torch.randn(2, 4, 16)
    y = ffn(x)
    assert y.shape == x.shape
    with torch.no_grad():
        ffn.w_gate.weight.zero_()
    gated = ffn(x)
    assert torch.allclose(gated, torch.zeros_like(gated), atol=1e-6)


def test_forward_shapes() -> None:
    set_global_seed(3)
    config = official_tiny_config()
    model = TinyTransformer(config)
    tokens = torch.randint(0, config.vocab_size, (2, 17))
    logits = model(tokens)
    assert logits.shape == (2, 17, config.vocab_size)


def test_tied_embeddings() -> None:
    config = official_tiny_config()
    model = TinyTransformer(config)
    hidden = torch.randn(2, 5, config.d_model)
    tied = model.embed.unembed(hidden)
    manual = hidden @ model.embed.weight.T
    assert torch.allclose(tied, manual)
    # There is no separate lm_head weight.
    names = [name for name, _ in model.named_parameters()]
    assert not any("lm_head" in name for name in names)


def test_no_linear_biases() -> None:
    model = TinyTransformer(official_tiny_config())
    biased = [name for name, param in model.named_parameters() if name.endswith("bias")]
    assert biased == []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None


def test_parameter_count_matches_formula() -> None:
    config = official_tiny_config()
    model = TinyTransformer(config)
    assert model.parameter_count() == expected_parameter_count(config)
    # Sanity: a few million, far below 300M.
    assert 1_000_000 < model.parameter_count() < 10_000_000


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    set_global_seed(4)
    config = official_tiny_config()
    model = TinyTransformer(config)
    optimizer = build_optimizer(model)
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    model.train()
    loss = torch.nn.functional.cross_entropy(
        model(tokens[:, :-1]).reshape(-1, config.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    optimizer.step()
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=7, metadata={"note": "unit"})
    restored, opt2, step = restore_training_state(
        path,
        optimizer=build_optimizer(TinyTransformer(config)),
        restore_rng=False,
    )
    assert step == 7
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert opt2 is not None
    left = optimizer.state_dict()
    right = opt2.state_dict()
    assert left["param_groups"][0]["lr"] == right["param_groups"][0]["lr"]
    assert len(left["state"]) == len(right["state"])
