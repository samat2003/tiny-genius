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
    """Deterministic 64-bit MinHash. Same 128-perm / 0.8 cutoff contract."""
    grams = char_shingles(text, shingle_n)
    if not grams:
        return tuple(b"\x00" * 8 for _ in range(permutations))
    bases = [
        int.from_bytes(hashlib.sha256(gram.encode("utf-8")).digest()[:8], "little")
        for gram in grams
    ]
    sig: list[bytes] = []
    mask = (1 << 64) - 1
    for index in range(permutations):
        seed = hashlib.sha256(f"mh:{index}".encode()).digest()
        add = int.from_bytes(seed[:8], "little")
        mul = int.from_bytes(seed[8:16], "little") | 1
        best = min((mul * value + add) & mask for value in bases)
        sig.append(best.to_bytes(8, "little"))
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


def _band_keys(sig: tuple[bytes, ...], band_size: int = 4) -> list[bytes]:
    keys: list[bytes] = []
    for start in range(0, len(sig), band_size):
        keys.append(b"".join(sig[start : start + band_size]))
    return keys


def near_dedup(
    docs: list[dict[str, Any]],
    *,
    cutoff: float = 0.8,
    permutations: int = 128,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """MinHash Jaccard >= cutoff. LSH bands only propose candidates; cutoff is unchanged."""
    ordered = sorted(docs, key=lambda d: (d["source_id"], d["doc_id"]))
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    kept_sigs: dict[str, tuple[bytes, ...]] = {}
    bands: list[dict[bytes, list[str]]] = []
    for doc in ordered:
        sig = minhash_signature(doc["text"], permutations=permutations)
        keys = _band_keys(sig)
        if not bands:
            bands = [{} for _ in keys]
        candidates: list[str] = []
        seen_c: set[str] = set()
        for band_index, key in enumerate(keys):
            for other_id in bands[band_index].get(key, []):
                if other_id not in seen_c:
                    seen_c.add(other_id)
                    candidates.append(other_id)
        dropped = False
        for other_id in candidates:
            if estimated_jaccard(sig, kept_sigs[other_id]) >= cutoff:
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
            kept_sigs[doc["doc_id"]] = sig
            kept.append(doc)
            for band_index, key in enumerate(keys):
                bands[band_index].setdefault(key, []).append(doc["doc_id"])
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
