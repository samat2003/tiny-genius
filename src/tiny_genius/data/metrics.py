"""Machine-readable pipeline metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _rate(removed: int, start: int) -> float:
    return 0.0 if start == 0 else removed / start


def build_metrics(
    *,
    inventory: list[dict[str, Any]],
    input_docs: list[dict[str, Any]],
    result: dict[str, Any],
    thresholds: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    removed = result["removed"]
    kept = result["docs"]
    start_n = len(input_docs)
    exact_n = len(removed["exact_dedup"])
    near_n = len(removed["near_dedup"])
    quality_n = len(removed["quality"])
    quality_reasons: Counter[str] = Counter()
    for row in removed["quality"]:
        for code in row.get("reason_codes") or []:
            quality_reasons[code] += 1

    by_source_raw: dict[str, int] = {}
    for doc in input_docs:
        by_source_raw[doc["source_id"]] = by_source_raw.get(doc["source_id"], 0) + int(
            doc.get("token_count") or 0
        )
    by_source_post: dict[str, int] = {}
    by_domain: dict[str, int] = {"python": 0, "math": 0, "stem": 0}
    for doc in kept:
        n = int(doc.get("token_count") or 0)
        by_source_post[doc["source_id"]] = by_source_post.get(doc["source_id"], 0) + n
        by_domain[doc["domain"]] = by_domain.get(doc["domain"], 0) + n

    targets = thresholds["mixture_10m_target"]
    mixture = {
        "target_10m": {
            "python": targets["python"],
            "math": targets["math"],
            "stem": targets["stem"],
            "total": targets["total"],
        },
        "target_13b": thresholds["mixture_13b_target"],
        "actual": {**by_domain, "total": sum(by_domain.values())},
        "deviation_tokens": {
            key: by_domain.get(key, 0) - int(targets[key]) for key in ("python", "math", "stem")
        },
        "math_raw_estimate_vs_plan": {
            "raw_estimate": registry.get("math_raw_estimate_tokens"),
            "plan_target": registry.get("math_plan_target_tokens"),
            "gap": (registry.get("math_raw_estimate_tokens") or 0)
            - (registry.get("math_plan_target_tokens") or 0),
            "resolution": (
                "Down-select only via quality/dedup/contamination and the 10M "
                "proportional cap. No silent mixture change to 2.63B."
            ),
        },
        "python_raw_estimate": registry.get("python_raw_estimate_tokens"),
        "notes": (
            "13B domain targets are not-applicable at the 10M milestone. "
            "Deviation is versus the proportional 10M split."
        ),
    }

    available = [r for r in inventory if r.get("status") != "blocked"]
    by_domain_sources: dict[str, int] = Counter(r["domain"] for r in available)
    share_applicability = {}
    for domain, n_src in by_domain_sources.items():
        applicable = n_src >= 3
        share_applicability[domain] = {
            "available_sources": n_src,
            "applicable": applicable,
            "status": "applicable" if applicable else "not_applicable",
            "reason": None
            if applicable
            else "fewer than 3 available (non-BLOCKED) sources in this domain",
        }

    contest_before = Counter(
        str(d.get("problem_id"))
        for d in input_docs
        if d.get("source_id") == "codecontests_plus" and d.get("problem_id")
    )
    contest_after = Counter(
        str(d.get("problem_id"))
        for d in kept
        if d.get("source_id") == "codecontests_plus" and d.get("problem_id")
    )

    return {
        "raw_token_count_per_source": by_source_raw,
        "exact_duplicate_rate": {
            "removed": exact_n,
            "start": start_n,
            "rate": _rate(exact_n, start_n),
        },
        "near_duplicate_rate": {
            "removed": near_n,
            "start": start_n - exact_n,
            "rate": _rate(near_n, start_n - exact_n),
        },
        "quality_rejection": {
            "removed": quality_n,
            "reason_codes": dict(quality_reasons),
            "rate": _rate(quality_n, start_n),
        },
        "contamination_hits": result["contamination_report"],
        "post_pipeline_tokens_per_source": by_source_post,
        "post_pipeline_tokens_per_domain": by_domain,
        "mixture": mixture,
        "codecontests_solutions_per_problem": {
            "before": dict(contest_before),
            "after": dict(contest_after),
        },
        "packing": result.get("packing") or {},
        "max_subsource_share": share_applicability,
        "blocked_sources": [r["source_id"] for r in inventory if r.get("status") == "blocked"],
        "tokenizer_fingerprint": result.get("tokenizer_fingerprint"),
    }
