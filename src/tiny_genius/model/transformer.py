"""Decoder-only Tiny Transformer (Stage 2 debug model)."""

from __future__ import annotations

import torch
from torch import nn

from tiny_genius.model.attention import CausalSelfAttention
from tiny_genius.model.config import TinyModelConfig
from tiny_genius.model.embeddings import TokenEmbedding
from tiny_genius.model.rmsnorm import RMSNorm
from tiny_genius.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            n_ctx=config.n_ctx,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x)))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


class TinyTransformer(nn.Module):
    """Stage 2 debug decoder-only Transformer.

    Tied embeddings, pre-norm RMSNorm, RoPE, SwiGLU, no linear biases.
    """

    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be (batch, seq), got {tuple(token_ids.shape)}")
        if token_ids.shape[1] > self.config.n_ctx:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds n_ctx {self.config.n_ctx}"
            )
        x = self.drop(self.embed(token_ids))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.embed.unembed(x)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
