"""Fetch Hugging Face datasets from published Parquet/JSONL files.

A failed `datasets.load_dataset()` script is not treated as inaccessibility.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def download_file(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))


def iter_parquet_rows(
    repo_id: str,
    files: list[str],
    *,
    columns: list[str] | None = None,
    max_rows: int | None = None,
    batch_size: int = 16,
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    yielded = 0
    for name in files:
        path = download_file(repo_id, name)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield row
                yielded += 1
                if max_rows is not None and yielded >= max_rows:
                    return


def iter_jsonl_rows(
    repo_id: str,
    filename: str,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    path = download_file(repo_id, filename)
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


def fetch_source_rows(
    source: dict[str, Any], max_rows: int
) -> tuple[list[dict[str, Any]], str | None]:
    """Load rows from published files first, then fall back to datasets streaming."""
    files = source.get("hf_files") or []
    repo = source.get("hf_id")
    if not repo or source.get("identity_status") != "verified":
        return [], source.get("block_reason") or "not fetchable"
    if source.get("decision") == "reject":
        return [], source.get("block_reason") or "rejected"
    try:
        if files and all(name.endswith(".parquet") for name in files):
            rows = list(
                iter_parquet_rows(
                    repo,
                    files,
                    columns=source.get("hf_columns"),
                    max_rows=max_rows,
                )
            )
            return rows, None
        if files and len(files) == 1 and files[0].endswith(".jsonl"):
            return list(iter_jsonl_rows(repo, files[0], max_rows=max_rows)), None
    except Exception as exc:  # noqa: BLE001
        file_err = f"direct file fetch failed: {exc}"
    else:
        file_err = None

    try:
        from datasets import load_dataset

        kwargs: dict[str, Any] = {"split": "train", "streaming": True}
        if source.get("hf_config"):
            kwargs["name"] = source["hf_config"]
        dataset = load_dataset(repo, **kwargs)
        rows = []
        for index, row in enumerate(dataset):
            rows.append(dict(row))
            if index + 1 >= max_rows:
                break
        return rows, None
    except Exception as exc:  # noqa: BLE001
        extra = f"; {file_err}" if file_err else ""
        return [], f"fetch failed: {exc}{extra}"


def iter_source_raw_rows(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield raw published rows without requiring datasets.load_dataset scripts."""
    repo = source.get("hf_id")
    files = source.get("hf_files") or []
    if not repo:
        return
    if files and all(name.endswith(".parquet") for name in files):
        yield from iter_parquet_rows(repo, files, columns=source.get("hf_columns"))
        return
    if files and len(files) == 1 and files[0].endswith(".jsonl"):
        yield from iter_jsonl_rows(repo, files[0])
        return
    rows, err = fetch_source_rows(source, max_rows=10**9)
    if err:
        raise RuntimeError(err)
    yield from rows
