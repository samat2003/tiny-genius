"""Package import and public API."""

from importlib.metadata import version

import tiny_genius


def test_import_and_version() -> None:
    assert tiny_genius.__version__ == "0.1.0"
    assert version("tiny-genius") == tiny_genius.__version__


def test_public_api_exports() -> None:
    for name in (
        "load_config",
        "load_run_spec",
        "set_global_seed",
        "collect_environment",
        "TinyTransformer",
        "TinyModelConfig",
    ):
        assert hasattr(tiny_genius, name)
        assert callable(getattr(tiny_genius, name))
