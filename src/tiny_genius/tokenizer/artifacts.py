"""Frozen tokenizer artifact I/O and SHA-256 manifests."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT
from tiny_genius.tokenizer.model import (
    TOKENIZER_FORMAT,
    TOKENIZER_FORMAT_VERSION,
    TokenizerModel,
)
from tiny_genius.tokenizer.specials import SpecialTokens

DEFAULT_TOKENIZER_DIR = REPO_ROOT / "tokenizer"
ARTIFACT_NAMES = (
    "tokenizer.model",
    "tokenizer.json",
    "special_tokens.json",
    "tokenizer_metrics.json",
    "SHA256SUMS",
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_sha256sums(directory: Path, names: tuple[str, ...] | list[str]) -> Path:
    dest = directory / "SHA256SUMS"
    lines = []
    for name in sorted(names):
        if name == "SHA256SUMS":
            continue
        digest = sha256_file(directory / name)
        lines.append(f"{digest}  {name}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def verify_sha256sums(directory: Path) -> list[str]:
    manifest = directory / "SHA256SUMS"
    errors: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            errors.append(f"hash mismatch {name}: {actual} != {digest}")
    return errors


def model_to_jsonable(model: TokenizerModel) -> dict[str, Any]:
    return {
        "format": TOKENIZER_FORMAT,
        "format_version": TOKENIZER_FORMAT_VERSION,
        "version": model.version,
        "algorithm": model.algorithm,
        "vocab_size": model.vocab_size,
        "normalization": model.normalization,
        "specials": model.specials.to_dict(),
        "token_bytes_b64": [base64.b64encode(b).decode("ascii") for b in model.token_bytes],
        "merges": [list(item) for item in model.merges],
        "char_to_id": model.char_to_id,
        "metadata": model.metadata,
        "fingerprint": model.fingerprint,
    }


def model_from_jsonable(payload: dict[str, Any]) -> TokenizerModel:
    if payload.get("format") != TOKENIZER_FORMAT:
        raise ValueError(f"unknown tokenizer format: {payload.get('format')}")
    specials = SpecialTokens(
        pad=payload["specials"]["pad"],
        unk=payload["specials"]["unk"],
        bos=payload["specials"]["bos"],
        eos=payload["specials"]["eos"],
        pad_id=payload["specials"]["pad_id"],
        unk_id=payload["specials"]["unk_id"],
        bos_id=payload["specials"]["bos_id"],
        eos_id=payload["specials"]["eos_id"],
    )
    return TokenizerModel(
        version=payload["version"],
        algorithm=payload["algorithm"],
        vocab_size=payload["vocab_size"],
        normalization=payload["normalization"],
        specials=specials,
        token_bytes=[base64.b64decode(item) for item in payload["token_bytes_b64"]],
        merges=[(int(a), int(b), int(c)) for a, b, c in payload["merges"]],
        char_to_id={str(k): int(v) for k, v in payload.get("char_to_id", {}).items()},
        metadata=payload.get("metadata") or {},
    )


def save_artifacts(
    directory: Path,
    model: TokenizerModel,
    metrics: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = model_to_jsonable(model)
    (directory / "tokenizer.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "tokenizer.model").write_bytes(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    specials = {
        "policy": (
            "Special IDs are reserved and never produced by encode() of raw text. "
            "encode(text) does not insert BOS/EOS unless add_bos/add_eos is true. "
            "decode(..., skip_special_tokens=True) drops specials. "
            "The surface strings <bos> etc. in user text are ordinary bytes."
        ),
        "tokens": model.specials.to_dict(),
        "insertion": {
            "add_bos_default": False,
            "add_eos_default": False,
        },
    }
    (directory / "special_tokens.json").write_text(
        json.dumps(specials, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "tokenizer_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_sha256sums(
        directory,
        ("tokenizer.model", "tokenizer.json", "special_tokens.json", "tokenizer_metrics.json"),
    )


def load_model(directory: Path | None = None) -> TokenizerModel:
    path = (directory or DEFAULT_TOKENIZER_DIR) / "tokenizer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return model_from_jsonable(payload)
