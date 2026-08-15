"""STAGE4_SMOKE identity. Not the 10M milestone. Not a production corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT
from tiny_genius.data.tokenize_stage import EXPECTED_FINGERPRINT
from tiny_genius.tokenizer.artifacts import sha256_file, verify_sha256sums

SMOKE_ID = "STAGE4_SMOKE"
PYTHON_TOKENS = 7_689_911
MATH_TOKENS = 1_539_907
STEM_TOKENS = 0
TOTAL_TOKENS = 9_229_818
SEED = 42

AUDIT_DIR = REPO_ROOT / "manifests" / "10m"
SMOKE_DIR = REPO_ROOT / "manifests" / "stage4_smoke"
SMOKE_PATH = SMOKE_DIR / "STAGE4_SMOKE.json"
FROZEN_10M = AUDIT_DIR / "FROZEN_10M.json"

AUDIT_FILES = (
    "data_manifest.jsonl",
    "dedup_manifest.jsonl",
    "contamination_report.json",
    "data_pipeline_metrics.json",
    "mixture_summary.json",
)


def corpus_hash(audit_dir: Path = AUDIT_DIR) -> str:
    hasher = hashlib.sha256()
    for name in AUDIT_FILES:
        digest = sha256_file(audit_dir / name)
        hasher.update(f"{digest}  {name}\n".encode("ascii"))
    return hasher.hexdigest()


def load_stage4_smoke(path: Path = SMOKE_PATH) -> dict[str, Any]:
    if FROZEN_10M.is_file():
        raise RuntimeError(
            "FROZEN_10M.json exists; STAGE4_SMOKE must not coexist with a 10M freeze"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_stage4_smoke(payload, audit_dir=AUDIT_DIR)
    if errors:
        raise RuntimeError("STAGE4_SMOKE invalid: " + "; ".join(errors))
    return payload


def validate_stage4_smoke(payload: dict[str, Any], *, audit_dir: Path = AUDIT_DIR) -> list[str]:
    errors: list[str] = []
    if payload.get("identity") != SMOKE_ID:
        errors.append(f"identity {payload.get('identity')!r} != {SMOKE_ID}")
    if payload.get("is_10m_milestone") is not False:
        errors.append("is_10m_milestone must be false")
    if payload.get("gate_g4") != "FAIL":
        errors.append("gate_g4 must be FAIL")
    if payload.get("frozen_10m_json") is not False:
        errors.append("frozen_10m_json must be false")
    if FROZEN_10M.is_file():
        errors.append("FROZEN_10M.json must not exist")
    counts = {
        "python_tokens": PYTHON_TOKENS,
        "math_tokens": MATH_TOKENS,
        "stem_tokens": STEM_TOKENS,
        "total_tokens": TOTAL_TOKENS,
    }
    for key, expected in counts.items():
        if payload.get(key) != expected:
            errors.append(f"{key} {payload.get(key)} != {expected}")
    if payload.get("tokenizer_fingerprint") != EXPECTED_FINGERPRINT:
        errors.append("tokenizer fingerprint mismatch")
    if payload.get("seed") != SEED:
        errors.append("seed must be 42")
    expected_hash = corpus_hash(audit_dir)
    if payload.get("corpus_hash") != expected_hash:
        errors.append("corpus_hash does not match current 10m audit files")
    mixture = json.loads((audit_dir / "mixture_summary.json").read_text(encoding="utf-8"))
    actual = mixture["actual"]
    if actual["python"] != PYTHON_TOKENS or actual["math"] != MATH_TOKENS:
        errors.append("mixture_summary counts drifted")
    if actual["stem"] != STEM_TOKENS or actual["total"] != TOTAL_TOKENS:
        errors.append("mixture_summary total/stem drifted")
    sha_errors = verify_sha256sums(audit_dir)
    errors.extend(f"audit {e}" for e in sha_errors)
    return errors


def load_manifest_shards(audit_dir: Path = AUDIT_DIR) -> list[dict[str, Any]]:
    """Source-level shard descriptors from the preserved Stage 4 audit."""
    shards: list[dict[str, Any]] = []
    path = audit_dir / "data_manifest.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        shards.append(
            {
                "source_id": rec.get("source_id"),
                "domain": rec.get("domain"),
                "status": rec.get("status"),
                "token_count": rec.get("token_count") or 0,
                "identity_status": rec.get("identity_status"),
            }
        )
    return shards
