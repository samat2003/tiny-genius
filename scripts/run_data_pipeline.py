#!/usr/bin/env python3
"""Run the Stage 4 high-signal pretrain pipeline.

Does not auto-create FROZEN_10M.json. Use --freeze only after an honest build.
13B is a ceiling, not a pad target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_genius.config import REPO_ROOT
from tiny_genius.data.metrics import build_metrics
from tiny_genius.data.pipeline import (
    apply_mixture_cap,
    collect_source_docs,
    docs_from_rows,
    load_pipeline_config,
    run_stages,
    source_inventory_records,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "manifests" / "pretrain",
    )
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--max-docs-per-source", type=int, default=25000)
    parser.add_argument("--no-tokenize", action="store_true")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Write FROZEN_PRETRAIN.json (not FROZEN_10M.json)",
    )
    parser.add_argument(
        "--apply-13b-ceiling",
        action="store_true",
        help="Down-select only if tokens would exceed the 13B plan ceiling",
    )
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
            cap = int(thresholds["thresholds"]["codecontests_max_solutions_per_problem"]["value"])
            fetched, err = collect_source_docs(
                source,
                data_cfg["collection_date"],
                cap=cap,
                max_docs=args.max_docs_per_source,
            )
            if err:
                rec["status"] = "blocked"
                rec["block_reason"] = err
                continue
            rec["fetch_rows"] = len(fetched)
            docs.extend(fetched)

    result = run_stages(
        docs,
        thresholds=thresholds,
        data_cfg=data_cfg,
        refs_dir=REPO_ROOT / data_cfg["contamination_refs_dir"],
        tokenize=not args.no_tokenize,
    )
    if args.apply_13b_ceiling:
        ceiling = {
            "python": int(thresholds["mixture_13b_target"]["python"]),
            "math": int(thresholds["mixture_13b_target"]["math"]),
            "stem": int(thresholds["mixture_13b_target"]["stem"]),
        }
        result["docs"] = apply_mixture_cap(result["docs"], ceiling)
    metrics = build_metrics(
        inventory=inventory,
        input_docs=docs,
        result=result,
        thresholds=thresholds,
        registry=registry,
    )
    for rec in inventory:
        rec["token_count"] = metrics["post_pipeline_tokens_per_source"].get(rec["source_id"], 0)
    write_milestone_artifacts(
        args.output,
        inventory=inventory,
        result=result,
        metrics=metrics,
        thresholds=thresholds,
        data_cfg=data_cfg,
        freeze=args.freeze,
    )
    actual = int(metrics["mixture"]["actual"]["total"])
    print(
        json.dumps(
            {
                "kept_docs": len(result["docs"]),
                "actual_tokens": metrics["mixture"]["actual"],
                "highsignal": metrics.get("highsignal_rejection"),
                "blocked": metrics["blocked_sources"],
                "frozen": args.freeze,
                "output": str(args.output),
                "total": actual,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
