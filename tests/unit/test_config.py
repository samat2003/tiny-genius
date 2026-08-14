"""Configuration loading and validation."""

from pathlib import Path

import pytest

from tiny_genius.config import load_config, load_run_spec, validate_run_spec

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stage0_config_loads() -> None:
    config = load_config(REPO_ROOT / "configs" / "stage0.yaml")
    assert config["name"] == "stage0-dev"
    assert config["stage"] == 0
    assert config["frozen"] is False
    assert config["project"] == "Tiny Genius"
    assert isinstance(config["reproducibility"]["seed"], int)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_validate_run_spec_rejects_wrong_project() -> None:
    spec = load_run_spec(REPO_ROOT / "RUN_SPEC.yaml")
    spec["project"] = "Rocket AI"
    with pytest.raises(ValueError, match="Tiny Genius"):
        validate_run_spec(spec)


def test_validate_run_spec_rejects_frozen_stage0() -> None:
    spec = load_run_spec(REPO_ROOT / "RUN_SPEC.yaml")
    spec["config"]["frozen"] = True
    with pytest.raises(ValueError, match="frozen"):
        validate_run_spec(spec)
