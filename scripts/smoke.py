#!/usr/bin/env python3
"""Stage 0 smoke check: environment, import, RUN_SPEC, and config load."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")

    import tiny_genius

    print(f"package: tiny_genius {tiny_genius.__version__}")
    print(f"location: {Path(tiny_genius.__file__).resolve()}")

    spec = tiny_genius.load_run_spec(REPO_ROOT / "RUN_SPEC.yaml")
    config = tiny_genius.load_config(REPO_ROOT / "configs" / "stage0.yaml")
    seed = tiny_genius.set_global_seed(spec["reproducibility"]["seed"])
    env = tiny_genius.collect_environment(seed=seed)

    print(f"run_spec.project: {spec['project']}")
    print(f"run_spec.config.identity: {spec['config']['identity']}")
    print(f"config.name: {config['name']}")
    print(f"seed: {seed}")
    print(f"git_commit: {env.get('git_commit')}")

    tiny = tiny_genius.TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml")
    model = tiny_genius.TinyTransformer(tiny)
    print(
        f"tiny_transformer: layers={tiny.n_layers} d_model={tiny.d_model} "
        f"params={model.parameter_count()}"
    )
    tokenizer_dir = REPO_ROOT / "tokenizer" / "FROZEN.json"
    if tokenizer_dir.is_file():
        tok = tiny_genius.Tokenizer.load_frozen()
        print(
            f"tokenizer: vocab={tok.vocab_size} algorithm={tok.algorithm} "
            f"version={tok.version}"
        )
    print("smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
