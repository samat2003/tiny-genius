"""Stage 2 Tiny Transformer configuration.

This is the debug-scale model from the project plan, not the 300M recipe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT, load_yaml

DEFAULT_TINY_CONFIG = REPO_ROOT / "configs" / "tiny.yaml"


@dataclass(frozen=True)
class TinyModelConfig:
    """Decoder-only Transformer config for the Stage 2 debug model."""

    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    vocab_size: int
    n_ctx: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True
    use_bias: bool = False
    dropout: float = 0.0
    stage: int = 2
    name: str = "tiny"
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads * d_head "
                f"({self.n_heads} * {self.d_head})"
            )
        if self.use_bias:
            raise ValueError("Project plan requires no linear-layer biases")
        if not self.tie_embeddings:
            raise ValueError("Project plan requires tied input/output embeddings")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if not (512 <= self.d_ff <= 768):
            raise ValueError(
                f"Stage 2 FFN hidden size must be in [512, 768], got {self.d_ff}"
            )
        if self.stage != 2:
            raise ValueError("TinyModelConfig is only for Stage 2 (tiny debug model)")
        if self.frozen:
            raise ValueError("Stage 2 tiny config must not be marked frozen")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TinyModelConfig:
        known = {field.name for field in cls.__dataclass_fields__.values()}
        payload = {key: value for key, value in data.items() if key in known}
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> TinyModelConfig:
        return cls.from_dict(load_yaml(path or DEFAULT_TINY_CONFIG))


def expected_parameter_count(config: TinyModelConfig) -> int:
    """Closed-form parameter count for the bias-free tied-embedding tiny model."""
    embed = config.vocab_size * config.d_model
    attn = config.d_model * (3 * config.d_model) + config.d_model * config.d_model
    ffn = 2 * config.d_model * config.d_ff + config.d_ff * config.d_model
    norms = 2 * config.d_model
    per_layer = attn + ffn + norms
    final_norm = config.d_model
    return embed + config.n_layers * per_layer + final_norm
