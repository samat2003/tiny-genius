"""Project-plan 300M config loads; Stage 2 tiny config still rejects 300M FFN."""

from __future__ import annotations

from pathlib import Path

from tiny_genius.model import TinyModelConfig, TinyTransformer, expected_parameter_count

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_300M = 296_925_120


def test_model_300m_yaml_matches_plan_and_closed_form() -> None:
    config = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "model_300m.yaml")
    assert config.name == "model_300m"
    assert config.frozen is False
    assert config.n_layers == 24
    assert config.d_model == 960
    assert config.n_heads == 15
    assert config.d_head == 64
    assert config.d_ff == 2560
    assert config.n_ctx == 4096
    assert config.vocab_size == 32768
    assert config.tie_embeddings is True
    assert config.use_bias is False
    assert expected_parameter_count(config) == EXPECTED_300M


def test_tiny_config_still_rejects_300m_ffn() -> None:
    data = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml").to_dict()
    data["d_ff"] = 2560
    try:
        TinyModelConfig.from_dict(data)
    except ValueError as exc:
        assert "512" in str(exc) or "768" in str(exc)
    else:
        raise AssertionError("Stage 2 tiny config must still reject 300M FFN")


def test_300m_model_parameter_count_on_cpu() -> None:
    config = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "model_300m.yaml")
    model = TinyTransformer(config)
    assert model.parameter_count() == EXPECTED_300M
