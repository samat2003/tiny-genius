"""Reproducible tokenizer candidate study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT, load_yaml
from tiny_genius.reproducibility import collect_environment
from tiny_genius.tokenizer.artifacts import DEFAULT_TOKENIZER_DIR, save_artifacts, verify_sha256sums
from tiny_genius.tokenizer.corpus import build_corpus
from tiny_genius.tokenizer.metrics import evaluate_model
from tiny_genius.tokenizer.thresholds import evaluate_thresholds, load_thresholds
from tiny_genius.tokenizer.train import train_candidate

CANDIDATES_PATH = REPO_ROOT / "configs" / "tokenizer_candidates.yaml"
FROZEN_PATH = DEFAULT_TOKENIZER_DIR / "FROZEN.json"


def run_candidate_study(
    *,
    extra_lines: int = 800,
    output_dir: Path | None = None,
    freeze: bool = True,
) -> dict[str, Any]:
    dest = output_dir or DEFAULT_TOKENIZER_DIR
    dest.mkdir(parents=True, exist_ok=True)
    if freeze and (dest / "FROZEN.json").is_file():
        raise RuntimeError(
            f"tokenizer already frozen at {dest / 'FROZEN.json'}; "
            "refuse to silently retrain"
        )

    thresholds = load_thresholds()
    candidate_cfg = load_yaml(CANDIDATES_PATH)
    corpus = build_corpus(extra_lines=extra_lines, seed=int(candidate_cfg["seed"]))
    vocab_size = int(candidate_cfg["vocab_size"])

    results: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for spec in candidate_cfg["candidates"]:
        model = train_candidate(
            name=spec["name"],
            algorithm=spec["algorithm"],
            texts=corpus.training,
            vocab_size=vocab_size,
            metadata={
                "description": spec["description"].strip(),
                "training_hash": corpus.training_hash,
                "evaluation_hash": corpus.evaluation_hash,
                "corpus_version": corpus.version,
                "seed": candidate_cfg["seed"],
            },
        )
        report = evaluate_model(model)
        gate = evaluate_thresholds(report, thresholds)
        record = {
            "name": spec["name"],
            "algorithm": spec["algorithm"],
            "description": spec["description"].strip(),
            "vocab_size": model.vocab_size,
            "normalization": model.normalization,
            "fingerprint": model.fingerprint,
            "n_merges": model.metadata.get("n_merges"),
            "n_unused": model.metadata.get("n_unused"),
            "metrics": report.to_dict(),
            "passed": gate.passed,
            "failures": gate.failures,
            "ranking_key": list(gate.ranking_key),
        }
        results.append(record)
        models[spec["name"]] = model
        candidate_dir = dest / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / f"{spec['name']}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    eligible = [item for item in results if item["passed"]]
    eligible.sort(key=lambda item: (item["ranking_key"], item["name"]))
    if not eligible:
        raise RuntimeError(
            "no tokenizer candidate passed predefined thresholds: "
            + json.dumps(results, indent=2)
        )
    winner_name = eligible[0]["name"]
    winner_model = models[winner_name]
    winner_model.metadata["selected"] = True
    rebuild = train_candidate(
        name=winner_name,
        algorithm=winner_model.algorithm,
        texts=corpus.training,
        vocab_size=vocab_size,
        metadata=dict(winner_model.metadata),
    )
    if rebuild.fingerprint != winner_model.fingerprint:
        raise RuntimeError("tokenizer training is not reproducible")
    winner_metrics = evaluate_model(winner_model).to_dict()
    study = {
        "thresholds_version": thresholds["version"],
        "selection_rule": thresholds["selection_rule"].strip(),
        "corpus": {
            "version": corpus.version,
            "evaluation_hash": corpus.evaluation_hash,
            "training_hash": corpus.training_hash,
            "extra_lines": extra_lines,
        },
        "environment": collect_environment(seed=int(candidate_cfg["seed"])),
        "candidates": results,
        "winner": winner_name,
        "rationale": (
            f"{winner_name} is the unique best eligible candidate under the "
            "predefined ranking (tokens/python line, identifier fragmentation, "
            "compression ratio)."
            if len(eligible) == 1
            else (
                f"{winner_name} ranked first among {len(eligible)} eligible "
                "candidates by the predefined ranking key."
            )
        ),
    }
    save_artifacts(
        dest,
        winner_model,
        {
            "winner": winner_name,
            "metrics": winner_metrics,
            "study": study,
        },
    )
    freeze_record = {
        "frozen": True,
        "stage": 3,
        "tokenizer_version": winner_model.version,
        "algorithm": winner_model.algorithm,
        "candidate": winner_name,
        "fingerprint": winner_model.fingerprint,
        "vocab_size": winner_model.vocab_size,
        "corpus": study["corpus"],
        "hashes_ok": verify_sha256sums(dest) == [],
    }
    if freeze:
        (dest / "FROZEN.json").write_text(
            json.dumps(freeze_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Re-hash after adding FROZEN.json? Plan lists SHA256SUMS of the five
        # named artifacts; FROZEN.json is extra metadata and is hashed separately.
        freeze_record["frozen_sha256"] = __import__(
            "tiny_genius.tokenizer.artifacts", fromlist=["sha256_file"]
        ).sha256_file(dest / "FROZEN.json")
        (dest / "FROZEN.json").write_text(
            json.dumps(freeze_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (dest / "candidate_study.json").write_text(
        json.dumps(study, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    study["freeze"] = freeze_record
    return study
