"""Unit tests for Stage 4 data-pipeline mechanics."""

from __future__ import annotations

from tiny_genius.data.contamination import match_document, scan_contamination
from tiny_genius.data.contract import validate_source_record
from tiny_genius.data.dedup import estimated_jaccard, exact_dedup, minhash_signature, near_dedup
from tiny_genius.data.license import license_decision, load_allowlist
from tiny_genius.data.normalize import normalize_text, normalized_hash
from tiny_genius.data.packing import pack_documents
from tiny_genius.data.pipeline import load_pipeline_config
from tiny_genius.data.quality import quality_reasons
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT, load_verified_tokenizer
from tiny_genius.tokenizer import Tokenizer
from tiny_genius.tokenizer.specials import EOS_ID, PAD_ID


def test_contract_requires_plan_fields() -> None:
    errors = validate_source_record({"source_id": "x"})
    assert errors
    assert any("license" in e for e in errors)


def test_exact_hash_is_deterministic() -> None:
    assert normalized_hash("abc") == normalized_hash("abc")
    assert normalized_hash("abc") != normalized_hash("abd")


def test_normalize_rejects_nuls() -> None:
    text, reason = normalize_text("ok\x00no")
    assert text is None
    assert reason == "binary"


def test_minhash_near_and_far() -> None:
    a = "def add(a, b):\n    return a + b\n"
    b = "def add(a, b):\n    return a + b\n"
    c = "the mitochondria is the powerhouse of the cell and ATP is made there daily"
    assert estimated_jaccard(minhash_signature(a), minhash_signature(b)) == 1.0
    assert estimated_jaccard(minhash_signature(a), minhash_signature(c)) < 0.5
    kept, removed = exact_dedup(
        [
            {"doc_id": "1", "source_id": "taco", "normalized_hash": "h", "text": a},
            {"doc_id": "2", "source_id": "taco", "normalized_hash": "h", "text": a},
        ]
    )
    assert len(kept) == 1
    assert len(removed) == 1
    long_a = a * 8
    long_b = a * 7 + "def add(a, b):\n    return a + b\n"
    kept2, removed2 = near_dedup(
        [
            {"doc_id": "a", "source_id": "taco", "text": long_a},
            {"doc_id": "b", "source_id": "taco", "text": long_b},
        ],
        cutoff=0.8,
    )
    assert len(kept2) == 1
    assert removed2


def test_quality_reason_codes() -> None:
    thresholds, _, _ = load_pipeline_config()
    short = {"text": "x", "domain": "python", "source_id": "taco"}
    assert "too_short" in quality_reasons(short, thresholds)
    bad = {"text": "def broken(\n", "domain": "python", "source_id": "apps"}
    assert "syntax_invalid" in quality_reasons(bad, thresholds)
    trivial = {
        "text": "What is 1+1?\n\nx",
        "domain": "math",
        "source_id": "openmathinstruct_2",
        "solution": "x",
    }
    assert "openmath_trivial" in quality_reasons(trivial, thresholds)
    wrong = {
        "text": "def solve():\n    return 0\n",
        "domain": "python",
        "source_id": "codecontests_plus",
        "is_correct": False,
    }
    assert "incorrect_solution" in quality_reasons(wrong, thresholds)


def test_contamination_matcher() -> None:
    thresholds, _, _ = load_pipeline_config()
    docs = [
        {
            "doc_id": "c1",
            "source_id": "taco",
            "text": "import os\nimport sys\nfrom typing import Optional, List, Dict, Tuple\n"
            "from pathlib import Path\nimport numpy as np\n",
        },
        {"doc_id": "c2", "source_id": "taco", "text": "def unique_ok():\n    return 123456\n"},
    ]
    kept, hits, report = scan_contamination(docs, thresholds=thresholds)
    assert any(h["doc_id"] == "c1" for h in hits)
    assert all(h["action"] == "removed" for h in hits)
    assert report["n_unresolved"] == 0
    assert any(d["doc_id"] == "c2" for d in kept)
    assert match_document("zzzz unique", {"token": {}, "char": {}}, token_n=12, char_n=50) is None


def test_packing_does_not_join_docs_without_eos() -> None:
    docs = [
        {"token_ids": [10, 11, 12]},
        {"token_ids": [20, 21]},
    ]
    seqs, stats = pack_documents(docs, n_ctx=16, shard_target_tokens=100)
    assert seqs
    flat = seqs[0]
    first_eos = flat.index(EOS_ID)
    assert flat[first_eos] == EOS_ID
    assert 20 in flat
    assert stats["n_ctx"] == 16
    assert PAD_ID in flat


def test_tokenizer_fingerprint_unchanged() -> None:
    tok = load_verified_tokenizer()
    assert tok.fingerprint == EXPECTED_FINGERPRINT
    assert Tokenizer.load_frozen().fingerprint == EXPECTED_FINGERPRINT


def test_unresolved_source_is_blocked() -> None:
    thresholds, _, registry = load_pipeline_config()
    allow = load_allowlist(thresholds)
    tiny = next(s for s in registry["sources"] if s["source_id"] == "tiny_python")
    status, reason = license_decision(tiny, allow)
    assert status == "blocked"
    assert reason
    taco = next(s for s in registry["sources"] if s["source_id"] == "taco")
    status2, _ = license_decision(taco, allow)
    assert status2 == "admitted"
