"""RUN_SPEC.yaml is loadable and matches Stage 0 identity."""

from pathlib import Path

from tiny_genius import load_config, load_run_spec

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_spec_loads_and_points_at_stage0_config() -> None:
    spec = load_run_spec(REPO_ROOT / "RUN_SPEC.yaml")
    config = load_config(REPO_ROOT / spec["config"]["path"])

    assert spec["project"] == "Tiny Genius"
    assert spec["config"]["identity"] == "stage0-dev"
    assert spec["config"]["stage"] == 0
    assert spec["config"]["frozen"] is False
    assert spec["reproducibility"]["seed"] == config["reproducibility"]["seed"]
    assert config["name"] == spec["config"]["identity"]
