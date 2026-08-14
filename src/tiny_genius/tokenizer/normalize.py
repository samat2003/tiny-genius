"""Deterministic normalization policy.

``identity`` is the Stage 3 production policy: no Unicode compatibility
mapping, no tab expansion, no newline rewriting. Python indentation and
identifiers are preserved exactly.
"""

from __future__ import annotations

from typing import Literal

Normalization = Literal["identity"]


def normalize(text: str, policy: Normalization = "identity") -> str:
    if policy != "identity":
        raise ValueError(f"unsupported normalization policy: {policy!r}")
    if not isinstance(text, str):
        raise TypeError("tokenizer input must be str")
    return text
