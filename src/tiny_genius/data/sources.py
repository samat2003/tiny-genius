"""Fixed source registry loader. Does not add or substitute sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT, load_yaml

SOURCES_PATH = REPO_ROOT / "configs" / "data_sources.yaml"


def load_source_registry(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or SOURCES_PATH)


def iter_sources(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = registry or load_source_registry()
    return list(data["sources"])
