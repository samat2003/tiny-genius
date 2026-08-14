"""Turn source-specific raw records into contract documents."""

from __future__ import annotations

from typing import Any

from tiny_genius.data.normalize import normalize_text, normalized_hash, raw_hash


def make_document(
    *,
    doc_id: str,
    source: dict[str, Any],
    text: str,
    collection_date: str,
    quality_score: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    normalized, reason = normalize_text(text)
    if normalized is None:
        return None
    doc = {
        "doc_id": doc_id,
        "source_id": source["source_id"],
        "url": source.get("claimed_url"),
        "license": source.get("claimed_license"),
        "provenance": source.get("claimed_origin"),
        "collection_date": collection_date,
        "language": "en" if source["domain"] != "python" else "python",
        "domain": source["domain"],
        "stem_bucket": source.get("stem_bucket"),
        "quality_score": quality_score,
        "contamination_risk": "unchecked",
        "raw_hash": raw_hash(text),
        "normalized_hash": normalized_hash(normalized),
        "token_count": None,
        "text": normalized,
        "status": "admitted",
    }
    if extra:
        doc.update(extra)
    return doc


def extract_text_from_hf_row(source_id: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    extra: dict[str, Any] = {}
    if source_id == "taco":
        if row.get("text"):
            extra["problem_id"] = str(row.get("problem_id") or "")
            extra["is_correct"] = row.get("is_correct", True)
            extra["language_tag"] = row.get("language_tag", "python")
            return str(row["text"]), extra
        sols = row.get("solutions") or []
        if isinstance(sols, str):
            text = sols
        elif isinstance(sols, list):
            text = "\n\n".join(str(s) for s in sols if s)
        else:
            text = str(row.get("question") or "")
        extra["problem_id"] = str(row.get("id") or row.get("question", "")[:80])
        extra["is_correct"] = True
        extra["language_tag"] = "python"
        return text, extra
    if source_id == "apps":
        if row.get("text"):
            extra["problem_id"] = str(row.get("problem_id") or row.get("id") or "")
            extra["is_correct"] = row.get("is_correct", True)
            extra["language_tag"] = "python"
            return str(row["text"]), extra
        sols = row.get("solutions") or "[]"
        if isinstance(sols, str):
            try:
                import json

                parsed = json.loads(sols)
                text = "\n\n".join(parsed) if isinstance(parsed, list) else sols
            except json.JSONDecodeError:
                text = sols
        else:
            text = "\n\n".join(str(s) for s in sols)
        extra["problem_id"] = str(row.get("problem_id") or row.get("id") or "")
        extra["is_correct"] = True
        extra["language_tag"] = "python"
        return text, extra
    if source_id == "openmathinstruct_2":
        problem = str(row.get("problem") or row.get("question") or "")
        solution = str(row.get("generated_solution") or row.get("solution") or "")
        extra["solution"] = solution
        extra["is_correct"] = True
        return f"{problem}\n\n{solution}", extra
    # Generic fallback for fixtures only.
    text = str(row.get("text") or "")
    extra.update({k: row[k] for k in ("problem_id", "is_correct", "language_tag", "solution") if k in row})
    return text, extra
