"""Canonical tokenization (frozen Stage 3 tokenizer only)."""

from __future__ import annotations

from typing import Any

from tiny_genius.tokenizer import Tokenizer

EXPECTED_FINGERPRINT = "219156db6bbe8c573c0f1654ab9f622c0e8bd51519561ac30d2c13fbf3a01a6e"


def load_verified_tokenizer() -> Tokenizer:
    tok = Tokenizer.load_frozen()
    if tok.fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            f"tokenizer fingerprint mismatch: {tok.fingerprint} != {EXPECTED_FINGERPRINT}"
        )
    return tok


def measurement_token_count(tok: Tokenizer, text: str) -> int:
    """Ad hoc count only — must not be packed or written as training ids."""
    return len(tok.encode(text))


def canonical_tokenize(docs: list[dict[str, Any]], tok: Tokenizer) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in docs:
        ids = tok.encode(text=doc["text"], add_bos=False, add_eos=True)
        item = dict(doc)
        item["token_ids"] = ids
        item["token_count"] = len(ids)
        out.append(item)
    return out
