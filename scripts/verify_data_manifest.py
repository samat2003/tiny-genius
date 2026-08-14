#!/usr/bin/env python3
"""Verify frozen 10M data-manifest hashes and tokenizer fingerprint."""

from __future__ import annotations

import json

from tiny_genius.config import REPO_ROOT
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT
from tiny_genius.tokenizer import Tokenizer
from tiny_genius.tokenizer.artifacts import verify_sha256sums


def main() -> int:
    dest = REPO_ROOT / "manifests" / "10m"
    if not (dest / "FROZEN_10M.json").is_file():
        print("FAIL missing FROZEN_10M.json")
        return 1
    errors = verify_sha256sums(dest)
    if errors:
        print("FAIL")
        for item in errors:
            print(item)
        return 1
    tok = Tokenizer.load_frozen()
    if tok.fingerprint != EXPECTED_FINGERPRINT:
        print("FAIL tokenizer fingerprint changed")
        return 1
    frozen = json.loads((dest / "FROZEN_10M.json").read_text(encoding="utf-8"))
    if frozen.get("tokenizer_fingerprint") != tok.fingerprint:
        print("FAIL freeze record fingerprint mismatch")
        return 1
    report = json.loads((dest / "contamination_report.json").read_text(encoding="utf-8"))
    if int(report.get("n_unresolved") or 0) != 0:
        print("FAIL unresolved contamination hits")
        return 1
    print(f"ok milestone=10m fingerprint={tok.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
