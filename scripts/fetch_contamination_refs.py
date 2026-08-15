#!/usr/bin/env python3
"""Download HumanEval/MBPP into gitignored data/contamination_refs/."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from tiny_genius.config import REPO_ROOT


def _write_parquet_text(repo: str, filename: str, dest: Path, fields: list[str]) -> None:
    import pyarrow.parquet as pq

    path = Path(hf_hub_download(repo, filename, repo_type="dataset"))
    table = pq.read_table(path)
    chunks: list[str] = []
    for row in table.to_pylist():
        parts = [str(row.get(field) or "") for field in fields]
        chunks.append("\n".join(parts))
    dest.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")


def main() -> int:
    dest_dir = REPO_ROOT / "data" / "contamination_refs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_text(
        "openai/openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet",
        dest_dir / "humaneval.txt",
        ["prompt", "canonical_solution", "test"],
    )
    _write_parquet_text(
        "google-research-datasets/mbpp",
        "full/test-00000-of-00001.parquet",
        dest_dir / "mbpp.txt",
        ["text", "code", "test_list"],
    )
    print(f"wrote {dest_dir / 'humaneval.txt'} and {dest_dir / 'mbpp.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
