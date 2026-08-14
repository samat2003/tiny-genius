"""Full pipeline on the synthetic fixture corpus."""

from __future__ import annotations

import json
from pathlib import Path

from tiny_genius.data.ingest import extract_text_from_hf_row, make_document
from tiny_genius.data.metrics import build_metrics
from tiny_genius.data.pipeline import (
    freeze_milestone,
    load_pipeline_config,
    run_stages,
    source_inventory_records,
)
from tiny_genius.reproducibility import fingerprint

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "corpus.jsonl"


def _fixture_docs():
    thresholds, data_cfg, registry = load_pipeline_config()
    sources = {s["source_id"]: s for s in registry["sources"]}
    docs = []
    for index, line in enumerate(FIXTURE.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        source = sources[row["source_id"]]
        text, extra = extract_text_from_hf_row(row["source_id"], row)
        doc = make_document(
            doc_id=f"{row['source_id']}:{index:04d}",
            source=source,
            text=text,
            collection_date=data_cfg["collection_date"],
            quality_score=row.get("score"),
            extra=extra,
        )
        if doc:
            docs.append(doc)
    return thresholds, data_cfg, registry, docs


def test_fixture_pipeline_expected_removals() -> None:
    thresholds, data_cfg, registry, docs = _fixture_docs()
    result = run_stages(docs, thresholds=thresholds, data_cfg=data_cfg)
    removed = result["removed"]
    assert removed["exact_dedup"]
    assert removed["quality"]
    reasons = {code for row in removed["quality"] for code in row["reason_codes"]}
    assert "syntax_invalid" in reasons
    assert "openmath_trivial" in reasons
    assert "low_edu_score" in reasons
    ingest_reasons = {row["reason"] for row in removed["ingest"]}
    assert "incorrect_solution" in ingest_reasons
    assert "not_python" in ingest_reasons
    assert removed["problem_cap"]
    assert result["contamination_report"]["n_hits"] >= 1
    assert result["contamination_report"]["n_unresolved"] == 0
    kept_ids = {d["doc_id"] for d in result["docs"]}
    for hit in result["contamination_report"]["hits"]:
        assert hit["doc_id"] not in kept_ids
        assert hit["unresolved"] is False
    hashes = [d["normalized_hash"] for d in result["docs"]]
    assert len(hashes) == len(set(hashes))


def test_fixture_pipeline_is_deterministic(tmp_path: Path) -> None:
    thresholds, data_cfg, registry, docs = _fixture_docs()
    first = run_stages(docs, thresholds=thresholds, data_cfg=data_cfg)
    second = run_stages(docs, thresholds=thresholds, data_cfg=data_cfg)
    h1 = fingerprint([d["doc_id"] for d in first["docs"]])
    h2 = fingerprint([d["doc_id"] for d in second["docs"]])
    assert h1 == h2
    inventory = source_inventory_records(thresholds, data_cfg, registry)
    metrics = build_metrics(
        inventory=inventory,
        input_docs=docs,
        result=first,
        thresholds=thresholds,
        registry=registry,
    )
    dest = tmp_path / "10m"
    freeze_milestone(
        dest,
        inventory=inventory,
        result=first,
        metrics=metrics,
        thresholds=thresholds,
        data_cfg=data_cfg,
    )
    freeze_milestone(
        tmp_path / "10m-b",
        inventory=inventory,
        result=second,
        metrics=metrics,
        thresholds=thresholds,
        data_cfg=data_cfg,
    )
    a = (dest / "data_manifest.jsonl").read_text(encoding="utf-8")
    b = (tmp_path / "10m-b" / "data_manifest.jsonl").read_text(encoding="utf-8")
    assert fingerprint(a) == fingerprint(b)
    assert (dest / "SHA256SUMS").is_file()
    assert metrics["max_subsource_share"]["python"]["status"] == "not_applicable"
    assert "13B" in metrics["mixture"]["notes"] or "not-applicable" in metrics["mixture"]["notes"]
