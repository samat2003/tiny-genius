"""Frozen tokenizer artifacts satisfy Gate G3 when present."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_genius.tokenizer.api import Tokenizer
from tiny_genius.tokenizer.artifacts import DEFAULT_TOKENIZER_DIR, verify_sha256sums
from tiny_genius.tokenizer.corpus import evaluation_documents
from tiny_genius.tokenizer.metrics import evaluate_model
from tiny_genius.tokenizer.thresholds import evaluate_thresholds, load_thresholds

REPO_ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.skipif(
    not (DEFAULT_TOKENIZER_DIR / "FROZEN.json").is_file(),
    reason="frozen tokenizer artifacts not present",
)


def test_frozen_artifacts_exist() -> None:
    for name in (
        "tokenizer.model",
        "tokenizer.json",
        "special_tokens.json",
        "tokenizer_metrics.json",
        "SHA256SUMS",
        "FROZEN.json",
    ):
        assert (DEFAULT_TOKENIZER_DIR / name).is_file()


def test_hashes_and_gate() -> None:
    assert verify_sha256sums(DEFAULT_TOKENIZER_DIR) == []
    tok = Tokenizer.load_frozen()
    assert tok.vocab_size == 32768
    report = evaluate_model(tok._model)
    gate = evaluate_thresholds(report, load_thresholds())
    assert gate.passed, gate.failures
    for _, text in evaluation_documents():
        assert tok.decode(tok.encode(text)) == text
    study = json.loads((DEFAULT_TOKENIZER_DIR / "candidate_study.json").read_text(encoding="utf-8"))
    assert study["winner"]
    assert study["candidates"]
    frozen = json.loads((DEFAULT_TOKENIZER_DIR / "FROZEN.json").read_text(encoding="utf-8"))
    assert frozen["frozen"] is True
    assert frozen["fingerprint"] == tok.fingerprint
    assert frozen["vocab_size"] == 32768
    # Reloading matches.
    again = Tokenizer.load_frozen()
    assert again.encode("import os\n") == tok.encode("import os\n")
    assert again.fingerprint == tok.fingerprint
