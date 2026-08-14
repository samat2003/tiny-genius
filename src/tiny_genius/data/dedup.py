"""Exact and near-deduplication (deterministic MinHash)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


def char_shingles(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def minhash_signature(text: str, permutations: int = 128, shingle_n: int = 5) -> tuple[bytes, ...]:
    grams = char_shingles(text, shingle_n)
    if not grams:
        return tuple(b"\x00" * 32 for _ in range(permutations))
    sig: list[bytes] = []
    for index in range(permutations):
        best: bytes | None = None
        prefix = f"{index}:".encode()
        for gram in grams:
            digest = hashlib.sha256(prefix + gram.encode("utf-8")).digest()
            if best is None or digest < best:
                best = digest
        sig.append(best or b"\x00" * 32)
    return tuple(sig)


def estimated_jaccard(left: tuple[bytes, ...], right: tuple[bytes, ...]) -> float:
    if not left:
        return 0.0
    hits = sum(1 for a, b in zip(left, right, strict=True) if a == b)
    return hits / len(left)


def exact_dedup(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for doc in docs:
        key = doc["normalized_hash"]
        if key in seen:
            removed.append(
                {
                    "doc_id": doc["doc_id"],
                    "source_id": doc["source_id"],
                    "stage": "exact_dedup",
                    "reason": "exact_duplicate",
                    "duplicate_of": seen[key],
                }
            )
            continue
        seen[key] = doc["doc_id"]
        kept.append(doc)
    return kept, removed


def near_dedup(
    docs: list[dict[str, Any]],
    *,
    cutoff: float = 0.8,
    permutations: int = 128,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(docs, key=lambda d: (d["source_id"], d["doc_id"]))
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    signatures: list[tuple[str, tuple[bytes, ...]]] = []
    for doc in ordered:
        sig = minhash_signature(doc["text"], permutations=permutations)
        dropped = False
        for other_id, other_sig in signatures:
            if estimated_jaccard(sig, other_sig) >= cutoff:
                removed.append(
                    {
                        "doc_id": doc["doc_id"],
                        "source_id": doc["source_id"],
                        "stage": "near_dedup",
                        "reason": "near_duplicate",
                        "duplicate_of": other_id,
                    }
                )
                dropped = True
                break
        if not dropped:
            signatures.append((doc["doc_id"], sig))
            kept.append(doc)
    return kept, removed


def apply_problem_cap(
    docs: Iterable[dict[str, Any]], cap: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cap CodeContests+ (or contest-like) solutions per problem_id."""
    by_problem: dict[str, list[dict[str, Any]]] = {}
    others: list[dict[str, Any]] = []
    for doc in docs:
        pid = doc.get("problem_id")
        if not pid:
            others.append(doc)
            continue
        by_problem.setdefault(str(pid), []).append(doc)
    kept = list(others)
    removed: list[dict[str, Any]] = []
    for pid, group in by_problem.items():
        group = sorted(group, key=lambda d: d["normalized_hash"])
        kept.extend(group[:cap])
        for extra in group[cap:]:
            removed.append(
                {
                    "doc_id": extra["doc_id"],
                    "source_id": extra["source_id"],
                    "stage": "problem_cap",
                    "reason": "over_problem_cap",
                    "problem_id": pid,
                }
            )
    return kept, removed
