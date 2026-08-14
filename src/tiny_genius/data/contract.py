"""Raw data contract schema (project plan Stage 4)."""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = (
    "source_id",
    "url",
    "license",
    "provenance",
    "collection_date",
    "language",
    "domain",
    "quality_score",
    "contamination_risk",
    "raw_hash",
    "normalized_hash",
    "token_count",
)

SOURCE_LEVEL_REQUIRED = REQUIRED_FIELDS + ("status",)


def validate_source_record(record: dict[str, Any]) -> list[str]:
    missing = [field for field in SOURCE_LEVEL_REQUIRED if field not in record]
    return [f"missing {field}" for field in missing]
