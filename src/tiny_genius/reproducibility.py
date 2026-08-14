"""Minimal Stage 0 reproducibility helpers.

These utilities establish a seed policy and collect environment metadata.
They do not configure distributed training or numerical backends for the 300M run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Seed Python `random` and, when available, PyTorch.

    Exact bit-for-bit reproducibility is expected on CPU with dropout=0.
    CUDA / cuDNN kernels are not guaranteed to match across devices or versions.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["TINY_GENIUS_DETERMINISTIC"] = "1"
    else:
        os.environ.pop("TINY_GENIUS_DETERMINISTIC", None)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    root = repo_root or REPO_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def fingerprint(payload: Any) -> str:
    """Stable SHA-256 fingerprint of a JSON-serializable object."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_environment(*, seed: int | None = None) -> dict[str, Any]:
    """Collect host, Python, and package metadata for a run record."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(),
        "seed": seed,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "deterministic": os.environ.get("TINY_GENIUS_DETERMINISTIC") == "1",
        "package_version": _package_version(),
    }


def _package_version() -> str:
    from tiny_genius import __version__

    return __version__
