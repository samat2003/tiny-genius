"""Load and validate Stage 0 YAML configuration and RUN_SPEC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_RUN_SPEC_FIELDS = (
    "project",
    "version",
    "python",
    "reproducibility",
    "device",
    "config",
    "metadata",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_SPEC = REPO_ROOT / "RUN_SPEC.yaml"
DEFAULT_STAGE0_CONFIG = REPO_ROOT / "configs" / "stage0.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {file_path}, got {type(data).__name__}")
    return data


def validate_run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the Stage 0 RUN_SPEC contract. Values are not a frozen training run."""
    missing = [field for field in REQUIRED_RUN_SPEC_FIELDS if field not in spec]
    if missing:
        raise ValueError(f"RUN_SPEC is missing required fields: {missing}")

    project = spec["project"]
    if not isinstance(project, str) or not project.strip():
        raise ValueError("RUN_SPEC.project must be a non-empty string")
    if project != "Tiny Genius":
        raise ValueError(f"RUN_SPEC.project must be 'Tiny Genius', got {project!r}")

    python = spec["python"]
    if not isinstance(python, dict) or "requires" not in python:
        raise ValueError("RUN_SPEC.python.requires is required")

    repro = spec["reproducibility"]
    if not isinstance(repro, dict):
        raise ValueError("RUN_SPEC.reproducibility must be a mapping")
    if "seed" not in repro:
        raise ValueError("RUN_SPEC.reproducibility.seed is required")
    seed = repro["seed"]
    if not isinstance(seed, int):
        raise ValueError("RUN_SPEC.reproducibility.seed must be an integer")

    config = spec["config"]
    if not isinstance(config, dict) or "identity" not in config:
        raise ValueError("RUN_SPEC.config.identity is required")
    if config.get("frozen") is True:
        raise ValueError(
            "Stage 0 RUN_SPEC must not claim a frozen training configuration"
        )

    return spec


def load_run_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate RUN_SPEC.yaml."""
    return validate_run_spec(load_yaml(path or DEFAULT_RUN_SPEC))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load an external YAML config. Defaults to the Stage 0 development config."""
    return load_yaml(path or DEFAULT_STAGE0_CONFIG)
