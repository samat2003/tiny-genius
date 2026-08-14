"""License allowlist filter. Fail closed → BLOCKED, never silent drop."""

from __future__ import annotations

from typing import Any


def load_allowlist(thresholds: dict[str, Any]) -> set[str]:
    return set(thresholds["thresholds"]["license_allowlist"]["values"])


def license_decision(source: dict[str, Any], allowlist: set[str]) -> tuple[str, str | None]:
    """Return (status, reason). status is admitted or blocked."""
    if source.get("decision") == "reject":
        return "blocked", (source.get("block_reason") or "rejected after inspection").strip()
    if source.get("identity_status") != "verified":
        return "blocked", (source.get("block_reason") or "unresolved identity").strip()
    license_id = source.get("claimed_license")
    if not license_id:
        return "blocked", "license unverifiable"
    if license_id not in allowlist:
        return "blocked", f"license {license_id!r} not on allowlist"
    return "admitted", None
