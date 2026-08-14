#!/usr/bin/env python3
"""Verify frozen tokenizer artifact hashes and basic contract."""

from __future__ import annotations

from tiny_genius.tokenizer.api import Tokenizer
from tiny_genius.tokenizer.artifacts import DEFAULT_TOKENIZER_DIR, verify_sha256sums


def main() -> int:
    errors = verify_sha256sums(DEFAULT_TOKENIZER_DIR)
    if errors:
        print("FAIL")
        for item in errors:
            print(item)
        return 1
    tok = Tokenizer.load_frozen()
    sample = "def add(x, y):\n    return x + y\n"
    assert tok.vocab_size == 32768
    assert tok.decode(tok.encode(sample)) == sample
    print(f"ok vocab={tok.vocab_size} algorithm={tok.algorithm} fingerprint={tok.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
