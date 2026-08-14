"""Independently staged data pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT, load_yaml
from tiny_genius.data.contamination import scan_contamination
from tiny_genius.data.contract import validate_source_record
from tiny_genius.data.dedup import apply_problem_cap, exact_dedup, near_dedup
from tiny_genius.data.ingest import (
    expand_source_rows,
    extract_text_from_hf_row,
    make_document,
)
from tiny_genius.data.license import license_decision, load_allowlist
from tiny_genius.data.packing import pack_documents
from tiny_genius.data.quality import apply_quality
from tiny_genius.data.sources import iter_sources, load_source_registry
from tiny_genius.data.tokenize_stage import (
    canonical_tokenize,
    load_verified_tokenizer,
    measurement_token_count,
)
from tiny_genius.reproducibility import collect_environment, fingerprint

THRESHOLDS_PATH = REPO_ROOT / "configs" / "data_thresholds.yaml"
DATA_CFG_PATH = REPO_ROOT / "configs" / "data.yaml"


def load_pipeline_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return load_yaml(THRESHOLDS_PATH), load_yaml(DATA_CFG_PATH), load_source_registry()


def source_inventory_records(
    thresholds: dict[str, Any],
    data_cfg: dict[str, Any],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    allow = load_allowlist(thresholds)
    records: list[dict[str, Any]] = []
    for source in iter_sources(registry):
        status, reason = license_decision(source, allow)
        rec = {
            "source_id": source["source_id"],
            "url": source.get("claimed_url"),
            "license": source.get("claimed_license"),
            "provenance": source.get("claimed_origin"),
            "collection_date": data_cfg["collection_date"],
            "language": "python" if source["domain"] == "python" else "en",
            "domain": source["domain"],
            "quality_score": None,
            "contamination_risk": "unchecked",
            "raw_hash": None,
            "normalized_hash": None,
            "token_count": 0,
            "status": "blocked" if status == "blocked" else "admitted",
            "block_reason": reason,
            "identity_status": source.get("identity_status"),
        }
        errors = validate_source_record(rec)
        rec["contract_errors"] = errors
        records.append(rec)
    return records


def run_stages(
    docs: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    data_cfg: dict[str, Any],
    extra_contam_refs: list[tuple[str, str]] | None = None,
    refs_dir: Path | None = None,
    tokenize: bool = True,
) -> dict[str, Any]:
    tok = load_verified_tokenizer()
    for doc in docs:
        if doc.get("token_count") is None:
            doc["token_count"] = measurement_token_count(tok, doc["text"])

    cap = int(thresholds["thresholds"]["codecontests_max_solutions_per_problem"]["value"])
    contest = [d for d in docs if d.get("source_id") == "codecontests_plus"]
    rest = [d for d in docs if d.get("source_id") != "codecontests_plus"]
    contest_ok: list[dict[str, Any]] = []
    ingest_removed: list[dict[str, Any]] = []
    for doc in contest:
        if doc.get("is_correct") is False:
            ingest_removed.append(
                {
                    "doc_id": doc["doc_id"],
                    "source_id": doc["source_id"],
                    "stage": "ingest",
                    "reason": "incorrect_solution",
                }
            )
            continue
        tag = str(doc.get("language_tag") or "python").lower()
        if tag not in {"python", "python3", "py"}:
            ingest_removed.append(
                {
                    "doc_id": doc["doc_id"],
                    "source_id": doc["source_id"],
                    "stage": "ingest",
                    "reason": "not_python",
                }
            )
            continue
        contest_ok.append(doc)
    contest_kept, cap_removed = apply_problem_cap(contest_ok, cap)
    docs = rest + contest_kept

    exact_kept, exact_removed = exact_dedup(docs)
    near_cfg = thresholds["thresholds"]["near_dedup"]
    near_kept, near_removed = near_dedup(
        exact_kept,
        cutoff=float(near_cfg["jaccard_cutoff"]),
        permutations=int(near_cfg["permutations"]),
    )
    quality_kept, quality_rejected = apply_quality(near_kept, thresholds)
    clean, hits, contam_report = scan_contamination(
        quality_kept,
        thresholds=thresholds,
        extra_refs=extra_contam_refs,
        refs_dir=refs_dir,
    )
    for doc in clean:
        doc["contamination_risk"] = "clear"
    packed_stats: dict[str, Any] = {}
    tokenized: list[dict[str, Any]] = []
    if tokenize:
        tokenized = canonical_tokenize(clean, tok)
        _, packed_stats = pack_documents(
            tokenized,
            n_ctx=int(data_cfg["n_ctx"]),
            shard_target_tokens=int(data_cfg["shard_target_tokens"]),
        )
    else:
        tokenized = clean

    return {
        "docs": tokenized,
        "removed": {
            "ingest": ingest_removed,
            "problem_cap": cap_removed,
            "exact_dedup": exact_removed,
            "near_dedup": near_removed,
            "quality": quality_rejected,
            "contamination": hits,
        },
        "contamination_report": contam_report,
        "packing": packed_stats,
        "tokenizer_fingerprint": tok.fingerprint,
        "environment": collect_environment(seed=int(data_cfg["seed"])),
    }


def mixture_counts(docs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"python": 0, "math": 0, "stem": 0}
    for doc in docs:
        domain = doc.get("domain")
        if domain in counts:
            counts[domain] += int(doc.get("token_count") or 0)
    return counts


def apply_mixture_cap(
    docs: list[dict[str, Any]], targets: dict[str, int]
) -> list[dict[str, Any]]:
    """Down-select only. Never invent tokens or sources."""
    ordered = sorted(docs, key=lambda d: (d["source_id"], d["doc_id"]))
    used = {"python": 0, "math": 0, "stem": 0}
    kept: list[dict[str, Any]] = []
    for doc in ordered:
        domain = doc["domain"]
        n = int(doc.get("token_count") or 0)
        if used.get(domain, 0) + n <= targets.get(domain, 0):
            kept.append(doc)
            used[domain] = used.get(domain, 0) + n
    return kept


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            slim = {k: v for k, v in row.items() if k not in {"text", "token_ids"}}
            handle.write(json.dumps(slim, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_milestone_artifacts(
    dest: Path,
    *,
    inventory: list[dict[str, Any]],
    result: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    data_cfg: dict[str, Any],
    freeze: bool,
) -> None:
    write_jsonl(dest / "data_manifest.jsonl", inventory)
    removed_rows: list[dict[str, Any]] = []
    for rows in result["removed"].values():
        removed_rows.extend(rows)
    write_jsonl(dest / "dedup_manifest.jsonl", removed_rows)
    write_json(dest / "contamination_report.json", result["contamination_report"])
    write_json(dest / "data_pipeline_metrics.json", metrics)
    write_json(dest / "mixture_summary.json", metrics["mixture"])
    from tiny_genius.tokenizer.artifacts import write_sha256sums

    names = (
        "data_manifest.jsonl",
        "dedup_manifest.jsonl",
        "contamination_report.json",
        "data_pipeline_metrics.json",
        "mixture_summary.json",
    )
    write_sha256sums(dest, names)
    if not freeze:
        return
    frozen = dest / "FROZEN_10M.json"
    if frozen.is_file():
        raise RuntimeError(f"milestone already frozen: {frozen}")
    write_json(
        frozen,
        {
            "frozen": True,
            "stage": 4,
            "milestone": "10m",
            "tokenizer_fingerprint": result["tokenizer_fingerprint"],
            "manifest_hash": fingerprint(
                (dest / "data_manifest.jsonl").read_text(encoding="utf-8")
            ),
            "thresholds_version": thresholds["version"],
            "seed": data_cfg["seed"],
            "actual_tokens": metrics["mixture"]["actual"]["total"],
        },
    )


def freeze_milestone(
    dest: Path,
    *,
    inventory: list[dict[str, Any]],
    result: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    data_cfg: dict[str, Any],
) -> None:
    write_milestone_artifacts(
        dest,
        inventory=inventory,
        result=result,
        metrics=metrics,
        thresholds=thresholds,
        data_cfg=data_cfg,
        freeze=True,
    )


def docs_from_rows(
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    collection_date: str,
    *,
    cap: int = 8,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    rows = expand_source_rows(source["source_id"], rows, cap=cap)
    for index, row in enumerate(rows):
        text, extra = extract_text_from_hf_row(source["source_id"], row)
        if not text:
            continue
        doc = make_document(
            doc_id=f"{source['source_id']}:{index:08d}",
            source=source,
            text=text,
            collection_date=collection_date,
            quality_score=row.get("score") if isinstance(row.get("score"), (int, float)) else None,
            extra=extra,
        )
        if doc:
            docs.append(doc)
    return docs


def try_stream_hf(source: dict[str, Any], max_rows: int) -> tuple[list[dict[str, Any]], str | None]:
    if source.get("identity_status") != "verified" or not source.get("hf_id"):
        return [], source.get("block_reason") or "not fetchable"
    try:
        from datasets import load_dataset
    except ImportError:
        return [], "huggingface datasets library not installed"
    try:
        kwargs: dict[str, Any] = {"split": "train", "streaming": True}
        if source.get("hf_config"):
            kwargs["name"] = source["hf_config"]
        ds = load_dataset(source["hf_id"], **kwargs)
    except Exception as exc:  # noqa: BLE001 — must record BLOCKED reason
        return [], f"fetch failed: {exc}"
    rows: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(ds):
            rows.append(dict(row))
            if index + 1 >= max_rows:
                break
    except Exception as exc:  # noqa: BLE001
        if not rows:
            return [], f"stream failed: {exc}"
    return rows, None
