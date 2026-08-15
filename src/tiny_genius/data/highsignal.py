"""Deterministic high-signal rubric (Phi-1-style bar, pinned for hashes)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT, load_yaml

HIGHSIGNAL_PATH = REPO_ROOT / "configs" / "highsignal_filters.yaml"

OTHER_LANG = re.compile(
    r"(#include\s*<|using namespace|public static void main|"
    r"fn main\s*\(|func main\s*\(|package main\b|defexn\b)",
    re.I,
)
CONTROL = re.compile(r"\b(if|for|while|elif|else|try|except|return)\b")
CP_IO = re.compile(r"\b(input\s*\(|sys\.stdin|print\s*\()")
UTILITY_IO = re.compile(
    r"\b(pathlib|os\.path|os\.remove|shutil|requests\.|urllib|"
    r"BeautifulSoup|selenium|flask|django|tkinter|pygame|argparse|"
    r"subprocess|socket\.socket)\b",
    re.I,
)
OPEN_FILE = re.compile(r"\bopen\s*\(")
TUTORIAL = re.compile(
    r"(how to use|this chapter|pip install|getting started with|in this tutorial)",
    re.I,
)
ALGO_HINT = re.compile(
    r"\b(sort|search|graph|dfs|bfs|dp|dynamic programming|greedy|tree|"
    r"hash|algorithm|complexity|theorem|proof|integral|derivative|"
    r"matrix|numerical)\b",
    re.I,
)
MATH_HINT = re.compile(r"(\\frac|\\sum|\\int|prove that|q\.e\.d|therefore)", re.I)


def load_highsignal_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or HIGHSIGNAL_PATH)


def _code_blob(doc: dict[str, Any]) -> str:
    return str(doc.get("solution") or doc.get("text") or "")


def _nonempty_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def score_document(doc: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Return score 0–5, reason codes, and accept flag. Deterministic."""
    text = doc.get("text") or ""
    domain = doc.get("domain") or ""
    reasons: list[str] = []
    score = 4.0

    if OTHER_LANG.search(text) and domain == "python":
        return {"score": 0.0, "reasons": ["other_language"], "accept": False}

    if TUTORIAL.search(text):
        reasons.append("library_tutorial")
        score = 0.0

    if UTILITY_IO.search(text) or (OPEN_FILE.search(text) and not CP_IO.search(text)):
        if not CP_IO.search(text):
            reasons.append("utility_io")
            score = 0.0

    if domain == "python":
        blob = _code_blob(doc)
        lines = _nonempty_lines(blob)
        trivial_max = int(cfg["trivial"]["max_nonempty_lines"])
        has_cf = bool(CONTROL.search(blob))
        has_problem = "problem" in text.lower() or bool(doc.get("problem_id"))
        if CP_IO.search(blob):
            score = max(score, 4.0)
            reasons.append("cp_stdin_stdout_keep")
        elif len(lines) <= trivial_max and not has_cf and not has_problem:
            reasons.append("trivial")
            score = min(score, 1.0)
        try:
            ast.parse(blob)
        except SyntaxError:
            if "syntax_invalid" not in reasons:
                reasons.append("unparseable_for_signal")
                score = min(score, 1.0)

    if domain == "math":
        sol = str(doc.get("solution") or "")
        if len(sol) < int(cfg["math"]["min_solution_chars"]):
            reasons.append("math_thin")
            score = min(score, 1.0)
        elif MATH_HINT.search(text) or len(sol) >= 40:
            score = max(score, 4.0)

    if domain == "stem":
        if not (ALGO_HINT.search(text) or MATH_HINT.search(text)):
            reasons.append("stem_not_reasoning_dense")
            score = min(score, 1.0)

    threshold = float(cfg["score_min"])
    accept = score + 1e-9 >= threshold and "utility_io" not in reasons
    if "library_tutorial" in reasons or "other_language" in reasons:
        accept = False
    return {"score": float(score), "reasons": reasons, "accept": accept}


def apply_highsignal(
    docs: list[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for doc in docs:
        verdict = score_document(doc, cfg)
        doc["highsignal_score"] = verdict["score"]
        doc["highsignal_reasons"] = verdict["reasons"]
        if verdict["accept"]:
            kept.append(doc)
        else:
            rejected.append(
                {
                    "doc_id": doc["doc_id"],
                    "source_id": doc["source_id"],
                    "stage": "highsignal",
                    "reason_codes": verdict["reasons"] or ["below_threshold"],
                    "score": verdict["score"],
                }
            )
    return kept, rejected
