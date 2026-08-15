#!/usr/bin/env python3
"""Write STAGE4_SMOKE identity over the existing 9,229,818-token audit.

Does not freeze 10M. Does not rewrite audit files or thresholds.
"""

from __future__ import annotations

import json

from tiny_genius.config import REPO_ROOT
from tiny_genius.data.stage4_smoke import (
    AUDIT_DIR,
    AUDIT_FILES,
    MATH_TOKENS,
    PYTHON_TOKENS,
    SEED,
    SMOKE_DIR,
    SMOKE_ID,
    STEM_TOKENS,
    TOTAL_TOKENS,
    corpus_hash,
)
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT
from tiny_genius.tokenizer.artifacts import sha256_file, write_sha256sums

if (AUDIT_DIR / "FROZEN_10M.json").is_file():
    raise SystemExit("refusing to write STAGE4_SMOKE while FROZEN_10M.json exists")


def main() -> int:
    hashes = {name: sha256_file(AUDIT_DIR / name) for name in AUDIT_FILES}
    payload = {
        "identity": SMOKE_ID,
        "is_10m_milestone": False,
        "gate_g4": "FAIL",
        "gate_g4_reason": (
            "The exact 10M milestone is not frozen because STEM=0 and the "
            "corpus contains 9,229,818 tokens. STAGE4_SMOKE is an engineering "
            "validation corpus only."
        ),
        "frozen_10m_json": False,
        "python_tokens": PYTHON_TOKENS,
        "math_tokens": MATH_TOKENS,
        "stem_tokens": STEM_TOKENS,
        "total_tokens": TOTAL_TOKENS,
        "tokenizer_fingerprint": EXPECTED_FINGERPRINT,
        "seed": SEED,
        "audit_dir": "manifests/10m",
        "audit_sha256": hashes,
        "corpus_hash": corpus_hash(AUDIT_DIR),
        "notes": (
            "Preserved Stage 4 Python/Math audit after exclusive-STEM Outcome B. "
            "Not a production pre-training corpus. Do not call this 10M."
        ),
    }
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    dest = SMOKE_DIR / "STAGE4_SMOKE.json"
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_sha256sums(SMOKE_DIR, ("STAGE4_SMOKE.json",))
    (SMOKE_DIR / "README.md").write_text(
        "# STAGE4_SMOKE\n\n"
        "Engineering/training-path smoke corpus. **Not** the 10M milestone.\n\n"
        f"- total tokens = {TOTAL_TOKENS:,}\n"
        f"- Python = {PYTHON_TOKENS:,}\n"
        f"- Math = {MATH_TOKENS:,}\n"
        f"- STEM = {STEM_TOKENS}\n"
        "- Gate G4 remains **FAIL**\n"
        "- `FROZEN_10M.json` must not be created\n",
        encoding="utf-8",
    )
    print(f"wrote {dest.relative_to(REPO_ROOT)}")
    print(f"corpus_hash {payload['corpus_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
