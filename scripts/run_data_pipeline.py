#!/usr/bin/env python3
"""Run the Stage 4 data pipeline for the authorized 10M milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_genius.config import REPO_ROOT
from tiny_genius.data.metrics import build_metrics
from tiny_genius.data.pipeline import (
    apply_mixture_cap,
    docs_from_rows,
    load_pipeline_config,
    run_stages,
    source_inventory_records,
    try_stream_hf,
    write_milestone_artifacts,
)
from tiny_genius.reproducibility import set_global_seed


def load_fixture_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", default="10m", choices=["10m"])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "manifests" / "10m")
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=4000)
    parser.add_argument("--no-tokenize", action="store_true")
    args = parser.parse_args()

    thresholds, data_cfg, registry = load_pipeline_config()
    set_global_seed(int(data_cfg["seed"]))
    inventory = source_inventory_records(thresholds, data_cfg, registry)
    sources_by_id = {s["source_id"]: s for s in registry["sources"]}

    docs: list[dict] = []
    if args.fixture:
        fixture_rows = load_fixture_rows(args.fixture)
        by_source: dict[str, list] = {}
        for row in fixture_rows:
            by_source.setdefault(row["source_id"], []).append(row)
        for source_id, rows in by_source.items():
            source = sources_by_id[source_id]
            docs.extend(docs_from_rows(rows, source, data_cfg["collection_date"]))
    else:
        for rec in inventory:
            source = sources_by_id[rec["source_id"]]
            if rec["status"] == "blocked":
                continue
            rows, err = try_stream_hf(source, args.max_rows_per_source)
            if err:
                rec["status"] = "blocked"
                rec["block_reason"] = err
                continue
            rec["fetch_rows"] = len(rows)
            docs.extend(docs_from_rows(rows, source, data_cfg["collection_date"]))

    result = run_stages(
        docs,
        thresholds=thresholds,
        data_cfg=data_cfg,
        refs_dir=REPO_ROOT / data_cfg["contamination_refs_dir"],
        tokenize=not args.no_tokenize,
    )
    targets = {
        "python": int(thresholds["mixture_10m_target"]["python"]),
        "math": int(thresholds["mixture_10m_target"]["math"]),
        "stem": int(thresholds["mixture_10m_target"]["stem"]),
    }
    result["docs"] = apply_mixture_cap(result["docs"], targets)
    metrics = build_metrics(
        inventory=inventory,
        input_docs=docs,
        result=result,
        thresholds=thresholds,
        registry=registry,
    )
    for rec in inventory:
        rec["token_count"] = metrics["post_pipeline_tokens_per_source"].get(rec["source_id"], 0)
        rec["normalized_hash"] = rec.get("normalized_hash")
        rec["raw_hash"] = rec.get("raw_hash")
    actual = int(metrics["mixture"]["actual"]["total"])
    reached = actual >= int(thresholds["mixture_10m_target"]["total"])
    write_milestone_artifacts(
        args.output,
        inventory=inventory,
        result=result,
        metrics=metrics,
        thresholds=thresholds,
        data_cfg=data_cfg,
        freeze=reached,
    )
    print(
        json.dumps(
            {
                "kept_docs": len(result["docs"]),
                "actual_tokens": metrics["mixture"]["actual"],
                "blocked": metrics["blocked_sources"],
                "frozen": reached,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
