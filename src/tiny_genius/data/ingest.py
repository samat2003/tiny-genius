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


def looks_python3(code: str, language: str) -> bool:
    lang = (language or "").lower().replace(" ", "")
    if lang in {"py3", "python3", "python"}:
        return True
    if lang in {"py2", "python2", "cpp", "c++", "java", "csharp", "go", "rust"}:
        return False
    return False


def expand_contest_row(row: dict[str, Any], *, cap: int = 8) -> list[dict[str, Any]]:
    """Emit problem+Python3-correct pairs. Skip infrastructure and non-py3."""
    title = str(row.get("title") or row.get("name") or "")
    desc = str(row.get("description") or "")
    pid = str(row.get("id") or row.get("name") or title)
    out: list[dict[str, Any]] = []
    subs = row.get("correct_submissions") or []
    for sub in subs:
        if not isinstance(sub, dict):
            continue
        lang = str(sub.get("language") or "")
        code = str(sub.get("code") or "")
        if not looks_python3(code, lang):
            continue
        text = f"# {title}\n# {pid}\n\n{desc}\n\n# solution\n{code}\n"
        out.append(
            {
                "text": text,
                "problem_id": pid,
                "is_correct": True,
                "language_tag": "python",
                "solution": code,
            }
        )
        if len(out) >= cap:
            break
    return out


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
    keys = ("problem_id", "is_correct", "language_tag", "solution")
    extra.update({k: row[k] for k in keys if k in row})
    return text, extra


def expand_source_rows(
    source_id: str, rows: list[dict[str, Any]], *, cap: int = 8
) -> list[dict[str, Any]]:
    if source_id == "codecontests_plus":
        expanded: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            if payload.get("correct_submissions") is not None:
                expanded.extend(expand_contest_row(payload, cap=cap))
        return expanded
    return rows
