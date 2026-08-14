"""Causal multi-head self-attention with RoPE."""

from __future__ import annotations

import math

import torch
from torch import nn

from tiny_genius.model.rope import apply_rope, build_rope_cache


def causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask: 0 on/below diagonal, -inf strictly above (future tokens)."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_ctx: int,
        rope_theta: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model != n_heads * d_head:
            raise ValueError("d_model must equal n_heads * d_head")
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_ctx = n_ctx
        self.scale = 1.0 / math.sqrt(d_head)
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        cos, sin = build_rope_cache(n_ctx, d_head, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.w_qkv(x)
        qkv = qkv.view(batch, seq, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # (batch, heads, seq, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + causal_mask(seq, scores.device, scores.dtype)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.w_out(attended)
