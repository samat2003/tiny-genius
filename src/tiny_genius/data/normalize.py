"""Code-safe document normalization.

Identity-aligned with the frozen tokenizer: no NFKC, no tab expansion,
no newline rewrite. Reject NULs / non-text.
"""

from __future__ import annotations

from tiny_genius.reproducibility import fingerprint


def is_binary_or_nul(text: str) -> bool:
    if "\x00" in text:
        return True
    if not text:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / len(text) < 0.85


def normalize_text(text: str) -> tuple[str | None, str | None]:
    """Return (normalized, reject_reason)."""
    if not isinstance(text, str):
        return None, "not_text"
    if is_binary_or_nul(text):
        return None, "binary"
    return text, None


def raw_hash(text: str) -> str:
    return fingerprint({"raw": text})


def normalized_hash(text: str) -> str:
    return fingerprint({"normalized_utf8_text": text})
