"""Load predefined Gate G3 thresholds and apply them (no post-hoc tuning)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_genius.config import REPO_ROOT, load_yaml
from tiny_genius.tokenizer.metrics import MetricReport

DEFAULT_THRESHOLDS = REPO_ROOT / "configs" / "tokenizer_thresholds.yaml"


@dataclass(frozen=True)
class ThresholdResult:
    passed: bool
    failures: list[str]
    ranking_key: tuple


def load_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_THRESHOLDS)


def _metric_value(report: MetricReport, name: str) -> float | int:
    return getattr(report, name)


def evaluate_thresholds(report: MetricReport, spec: dict[str, Any]) -> ThresholdResult:
    failures: list[str] = []
    metrics = spec["metrics"]
    for name, rule in metrics.items():
        value = _metric_value(report, name)
        kind = rule["kind"]
        if kind != "hard":
            continue
        direction = rule["direction"]
        if direction == "exact" and value != rule["value"]:
            failures.append(f"{name}: {value!r} != {rule['value']!r}")
        elif direction == "lower" and float(value) > float(rule["max"]):
            failures.append(f"{name}: {value} > max {rule['max']}")
        elif direction == "higher" and float(value) < float(rule["min"]):
            failures.append(f"{name}: {value} < min {rule['min']}")
    ranking = (
        float(report.tokens_per_python_line),
        float(report.identifier_fragmentation),
        -float(report.compression_ratio),
    )
    return ThresholdResult(passed=not failures, failures=failures, ranking_key=ranking)
