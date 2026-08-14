"""Metric helpers and predefined thresholds load independently of selection."""

from pathlib import Path

from tiny_genius.tokenizer.corpus import evaluation_documents
from tiny_genius.tokenizer.metrics import evaluate_model
from tiny_genius.tokenizer.thresholds import load_thresholds
from tiny_genius.tokenizer.train import train_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_thresholds_file_defines_required_metrics() -> None:
    spec = load_thresholds(REPO_ROOT / "configs" / "tokenizer_thresholds.yaml")
    required = {
        "tokens_per_python_line",
        "tokens_per_function",
        "identifier_fragmentation",
        "operator_fragmentation",
        "import_fragmentation",
        "numeric_literal_handling",
        "unicode_behavior",
        "round_trip_exactness",
        "compression_ratio",
        "vocab_size",
        "byte_fallback_success",
    }
    assert required <= set(spec["metrics"])
    assert spec["metrics"]["vocab_size"]["value"] == 32768
    assert spec["metrics"]["round_trip_exactness"]["value"] == 1.0


def test_evaluate_model_returns_all_plan_metrics() -> None:
    model = train_candidate(
        name="m",
        algorithm="byte_bpe",
        texts=[text for _, text in evaluation_documents()],
        vocab_size=400,
    )
    report = evaluate_model(model)
    assert report.round_trip_exactness == 1.0
    assert report.byte_fallback_success == 1.0
    assert report.unicode_behavior == 1.0
    assert report.tokens_per_python_line > 0
    assert report.compression_ratio > 0
    # Small vocab is not the frozen 32,768 contract.
    assert report.vocab_size == 400
