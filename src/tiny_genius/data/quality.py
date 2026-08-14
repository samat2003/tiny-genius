"""Domain quality filters with auditable reason codes."""

from __future__ import annotations

import ast
import re
from typing import Any

NON_LATIN = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]"
)
OS_IO = re.compile(
    r"\b(subprocess|os\.system|docker|paramiko|scrapy|BeautifulSoup|"
    r"flask\.|django\.|socket\.socket|powershell)\b",
    re.I,
)


def quality_reasons(doc: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    rules = thresholds["thresholds"]
    text = doc.get("text") or ""
    reasons: list[str] = []
    min_c = int(rules["length_bounds"]["min_chars"])
    max_c = int(rules["length_bounds"]["max_chars"])
    if len(text) < min_c:
        reasons.append("too_short")
    if len(text) > max_c:
        reasons.append("too_long")

    score = doc.get("quality_score")
    if score is not None and float(score) < float(rules["min_quality_score"]["value"]):
        reasons.append("low_edu_score")

    if doc.get("domain") == "python" and rules["code_must_parse"]["value"]:
        parse_target = doc.get("solution") or text
        try:
            ast.parse(parse_target)
        except SyntaxError:
            reasons.append("syntax_invalid")

    if doc.get("source_id") == "openmathinstruct_2":
        solution = doc.get("solution") or ""
        if len(solution) < int(rules["openmath_min_solution_chars"]["value"]):
            reasons.append("openmath_trivial")

    if doc.get("is_correct") is False:
        reasons.append("incorrect_solution")

    if doc.get("language_tag") and str(doc["language_tag"]).lower() not in {
        "python",
        "python3",
        "py",
    }:
        if doc.get("domain") == "python":
            reasons.append("not_python")
    if NON_LATIN.search(text):
        reasons.append("non_english")
    if doc.get("domain") == "python" and OS_IO.search(text):
        reasons.append("os_io_boilerplate")
    return reasons


def apply_quality(
    docs: list[dict[str, Any]], thresholds: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for doc in docs:
        reasons = quality_reasons(doc, thresholds)
        if reasons:
            rejected.append(
                {
                    "doc_id": doc["doc_id"],
                    "source_id": doc["source_id"],
                    "stage": "quality",
                    "reason_codes": reasons,
                }
            )
        else:
            kept.append(doc)
    return kept, rejected
