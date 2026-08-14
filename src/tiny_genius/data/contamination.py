"""Contamination scan vs Stage 3 eval corpus and optional local benchmark refs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tiny_genius.tokenizer.corpus import evaluation_documents


def whitespace_ngrams(text: str, n: int) -> set[str]:
    tokens = text.split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def char_windows(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_reference_index(
    refs: list[tuple[str, str]],
    *,
    token_n: int,
    char_n: int,
) -> dict[str, Any]:
    token_map: dict[str, str] = {}
    char_map: dict[str, str] = {}
    for ref_id, text in refs:
        for gram in whitespace_ngrams(text, token_n):
            if gram:
                token_map.setdefault(gram, ref_id)
        for window in char_windows(text, char_n):
            if window:
                char_map.setdefault(window, ref_id)
    return {"token": token_map, "char": char_map}


def stage3_eval_refs() -> list[tuple[str, str]]:
    return [(f"stage3-eval-v1:{name}", text) for name, text in evaluation_documents() if text]


def load_local_benchmark_refs(directory: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Load gitignored HumanEval/MBPP text files if present. Never invent content."""
    found: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in ("humaneval.txt", "mbpp.txt"):
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        text = path.read_text(encoding="utf-8")
        found.append((f"local:{name}", text))
    return found, missing


def match_document(
    text: str, index: dict[str, Any], *, token_n: int, char_n: int
) -> str | None:
    for gram in whitespace_ngrams(text, token_n):
        if gram in index["token"]:
            return index["token"][gram]
    for window in char_windows(text, char_n):
        if window in index["char"]:
            return index["char"][window]
    return None


def scan_contamination(
    docs: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    extra_refs: list[tuple[str, str]] | None = None,
    refs_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg = thresholds["thresholds"]["contamination"]
    token_n = int(cfg["token_ngram"])
    char_n = int(cfg["char_window"])
    refs = stage3_eval_refs()
    missing: list[str] = []
    if refs_dir is not None:
        local, missing = load_local_benchmark_refs(refs_dir)
        refs.extend(local)
    if extra_refs:
        refs.extend(extra_refs)
    index = build_reference_index(refs, token_n=token_n, char_n=char_n)
    kept: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for doc in docs:
        matched = match_document(doc["text"], index, token_n=token_n, char_n=char_n)
        if matched:
            hits.append(
                {
                    "doc_id": doc["doc_id"],
                    "source_id": doc["source_id"],
                    "matched_ref": matched,
                    "action": "removed",
                    "unresolved": False,
                }
            )
        else:
            kept.append(doc)
    report = {
        "n_refs": len(refs),
        "benchmark_refs_missing": missing,
        "n_hits": len(hits),
        "n_unresolved": 0,
        "hits": hits,
    }
    return kept, hits, report
