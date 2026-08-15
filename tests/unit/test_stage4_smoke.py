"""STAGE4_SMOKE is an engineering corpus, not the 10M freeze."""

from __future__ import annotations

from tiny_genius.config import REPO_ROOT
from tiny_genius.data.stage4_smoke import (
    AUDIT_DIR,
    FROZEN_10M,
    MATH_TOKENS,
    PYTHON_TOKENS,
    SMOKE_PATH,
    STEM_TOKENS,
    TOTAL_TOKENS,
    load_manifest_shards,
    load_stage4_smoke,
    validate_stage4_smoke,
)
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT
from tiny_genius.tokenizer import Tokenizer
from tiny_genius.tokenizer.artifacts import verify_sha256sums


def test_stage4_smoke_identity_and_g4_fail() -> None:
    assert not FROZEN_10M.is_file()
    payload = load_stage4_smoke()
    assert payload["identity"] == "STAGE4_SMOKE"
    assert payload["is_10m_milestone"] is False
    assert payload["gate_g4"] == "FAIL"
    assert payload["total_tokens"] == TOTAL_TOKENS == 9_229_818
    assert payload["python_tokens"] == PYTHON_TOKENS
    assert payload["math_tokens"] == MATH_TOKENS
    assert payload["stem_tokens"] == STEM_TOKENS == 0
    assert payload["tokenizer_fingerprint"] == EXPECTED_FINGERPRINT
    assert validate_stage4_smoke(payload) == []
    assert verify_sha256sums(AUDIT_DIR) == []
    assert verify_sha256sums(SMOKE_PATH.parent) == []


def test_tokenizer_load_frozen_fingerprint() -> None:
    tok = Tokenizer.load_frozen()
    assert tok.fingerprint == EXPECTED_FINGERPRINT


def test_manifest_shards_sum_to_smoke_total() -> None:
    shards = load_manifest_shards()
    admitted = [s for s in shards if s["status"] == "admitted" and s["token_count"] > 0]
    assert sum(s["token_count"] for s in admitted) == TOTAL_TOKENS
    assert (REPO_ROOT / "manifests" / "10m" / "FROZEN_10M.json").is_file() is False
