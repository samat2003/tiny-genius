"""Tiny Genius — Stage 0 package foundation.

No Transformer implementation lives here yet. This package exposes
configuration loading, seed/determinism helpers, and environment metadata.
"""

from tiny_genius.config import load_config, load_run_spec
from tiny_genius.reproducibility import collect_environment, set_global_seed

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "collect_environment",
    "load_config",
    "load_run_spec",
    "set_global_seed",
]
