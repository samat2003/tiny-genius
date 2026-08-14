#!/usr/bin/env python3
"""Run the Stage 3 candidate study and optionally freeze the winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_genius.config import REPO_ROOT
from tiny_genius.tokenizer.study import run_candidate_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-lines", type=int, default=800)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tokenizer")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Retrain even if FROZEN.json exists (writes a new version).",
    )
    args = parser.parse_args()
    dest = args.output
    if args.allow_overwrite:
        frozen = dest / "FROZEN.json"
        if frozen.exists():
            frozen.unlink()
    study = run_candidate_study(extra_lines=args.extra_lines, output_dir=dest, freeze=True)
    print(json.dumps({"winner": study["winner"], "corpus": study["corpus"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
