"""Tiny Genius — reproducible Transformer engineering.

Stage 2 adds the debug-scale Tiny Transformer. The 300M model is not implemented.
"""

from tiny_genius.config import load_config, load_run_spec
from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.reproducibility import collect_environment, set_global_seed

__version__ = "0.1.0"
__all__ = [
    "TinyModelConfig",
    "TinyTransformer",
    "__version__",
    "collect_environment",
    "load_config",
    "load_run_spec",
    "set_global_seed",
]
