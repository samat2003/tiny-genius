"""Repository health checks."""

import sys
from pathlib import Path

import pytest

import tiny_genius
from tiny_genius.reproducibility import collect_environment, git_commit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_python_version_supported() -> None:
    assert sys.version_info >= (3, 10)


@pytest.mark.smoke
def test_required_paths_exist() -> None:
    required = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "RUN_SPEC.yaml",
        REPO_ROOT / "configs" / "stage0.yaml",
        REPO_ROOT / "src" / "tiny_genius" / "__init__.py",
        REPO_ROOT / "configs" / "tiny.yaml",
        REPO_ROOT / "configs" / "tokenizer_thresholds.yaml",
        REPO_ROOT / "configs" / "data_thresholds.yaml",
        REPO_ROOT / "configs" / "data_sources.yaml",
        REPO_ROOT / "300M_Dense_Transformer_Full_Project_Plan.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert missing == []


@pytest.mark.smoke
def test_environment_metadata_has_required_keys() -> None:
    env = collect_environment(seed=42)
    for key in (
        "timestamp_utc",
        "python_version",
        "platform",
        "git_commit",
        "seed",
        "package_version",
    ):
        assert key in env
    assert env["seed"] == 42
    assert env["package_version"] == tiny_genius.__version__


@pytest.mark.smoke
def test_git_commit_is_hex_when_available() -> None:
    commit = git_commit(REPO_ROOT)
    if commit is None:
        pytest.skip("git metadata unavailable")
    assert len(commit) == 40
    int(commit, 16)
