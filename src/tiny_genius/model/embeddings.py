"""Token embeddings (output projection is the tied embedding table)."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(token_ids, self.weight)

    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """Tied output projection: logits = hidden @ W^T."""
        return torch.nn.functional.linear(hidden, self.weight)
