"""10M milestone gate checks — only when a freeze exists."""

from __future__ import annotations

import json

import pytest

from tiny_genius.config import REPO_ROOT
from tiny_genius.data.contract import validate_source_record
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT
from tiny_genius.tokenizer.artifacts import verify_sha256sums

DEST = REPO_ROOT / "manifests" / "10m"


pytestmark = pytest.mark.skipif(
    not (DEST / "FROZEN_10M.json").is_file(),
    reason="10M milestone not frozen",
)


def test_10m_artifacts_and_thresholds() -> None:
    assert verify_sha256sums(DEST) == []
    frozen = json.loads((DEST / "FROZEN_10M.json").read_text(encoding="utf-8"))
    assert frozen["frozen"] is True
    assert frozen["tokenizer_fingerprint"] == EXPECTED_FINGERPRINT
    metrics = json.loads((DEST / "data_pipeline_metrics.json").read_text(encoding="utf-8"))
    report = json.loads((DEST / "contamination_report.json").read_text(encoding="utf-8"))
    assert report["n_unresolved"] == 0
    for hit in report.get("hits") or []:
        assert hit["action"] == "removed"
        assert hit["unresolved"] is False
    for line in (DEST / "data_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert validate_source_record(rec) == []
    for domain, info in metrics["max_subsource_share"].items():
        if not info["applicable"]:
            assert info["status"] == "not_applicable"
            assert info["reason"]
    assert metrics["tokenizer_fingerprint"] == EXPECTED_FINGERPRINT
    actual = metrics["mixture"]["actual"]["total"]
    # Honest: the 10M gate requires the target scale. Under-target is not a pass.
    assert actual >= 10_000_000
