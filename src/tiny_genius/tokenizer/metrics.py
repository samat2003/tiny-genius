"""Tokenizer candidate metrics required by the project plan."""

from __future__ import annotations

import io
import tokenize
from ast import AsyncFunctionDef, FunctionDef, walk
from ast import parse as ast_parse
from dataclasses import asdict, dataclass
from typing import Any

from tiny_genius.tokenizer.corpus import evaluation_documents, python_subset, unicode_subset
from tiny_genius.tokenizer.model import TokenizerModel


@dataclass
class MetricReport:
    vocab_size: int
    tokens_per_python_line: float
    tokens_per_function: float
    identifier_fragmentation: float
    operator_fragmentation: float
    import_fragmentation: float
    numeric_literal_handling: float
    unicode_behavior: float
    round_trip_exactness: float
    compression_ratio: float
    byte_fallback_success: float
    extras: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_tokenize(source: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_trip(model: TokenizerModel, texts: list[str]) -> float:
    if not texts:
        return 1.0
    ok = 0
    for text in texts:
        if model.decode(model.encode(text)) == text:
            ok += 1
    return ok / len(texts)


def _python_line_tokens(model: TokenizerModel) -> float:
    counts: list[float] = []
    for source in python_subset():
        for line in source.splitlines():
            if line.strip():
                counts.append(float(len(model.encode(line))))
    return _mean(counts)


def _function_tokens(model: TokenizerModel) -> float:
    counts: list[float] = []
    for source in python_subset():
        try:
            tree = ast_parse(source)
        except SyntaxError:
            continue
        lines = source.splitlines()
        for node in walk(tree):
            if isinstance(node, (FunctionDef, AsyncFunctionDef)) and node.end_lineno:
                snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                counts.append(float(len(model.encode(snippet))))
    return _mean(counts)


def _token_kind_fragmentation(model: TokenizerModel, kinds: set[int]) -> float:
    pieces: list[float] = []
    for source in python_subset():
        for tok in _safe_tokenize(source):
            if tok.type in kinds and tok.string:
                pieces.append(float(len(model.encode(tok.string))))
    return _mean(pieces)


def _import_fragmentation(model: TokenizerModel) -> float:
    counts: list[float] = []
    for source in python_subset():
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                counts.append(float(len(model.encode(line))))
    return _mean(counts)


def _numeric_handling(model: TokenizerModel, max_tokens: int = 6) -> float:
    totals = 0
    good = 0
    for source in python_subset():
        for tok in _safe_tokenize(source):
            if tok.type == tokenize.NUMBER:
                totals += 1
                if len(model.encode(tok.string)) <= max_tokens:
                    good += 1
    return 1.0 if totals == 0 else good / totals


def _compression(model: TokenizerModel, texts: list[str]) -> float:
    n_bytes = 0
    n_tokens = 0
    for text in texts:
        n_bytes += len(text.encode("utf-8"))
        n_tokens += len(model.encode(text))
    return 0.0 if n_tokens == 0 else n_bytes / n_tokens


def _byte_fallback_battery() -> list[str]:
    return [
        "🙂🔥✨",
        "한글テスト",
        "Ω≈åß∂ƒ",
        bytes([0, 1, 2, 254, 255]).decode("latin-1"),
        "é" * 32,
        "a\u0301",  # combining acute
        "\u2603\uFE0F",
        "𐍈",  # four-byte UTF-8
    ]


def evaluate_model(model: TokenizerModel) -> MetricReport:
    docs = evaluation_documents()
    texts = [text for _, text in docs]
    names = [name for name, _ in docs]
    all_round = _round_trip(model, texts)
    unicode_texts = unicode_subset()
    fallback_texts = _byte_fallback_battery()
    return MetricReport(
        vocab_size=model.vocab_size,
        tokens_per_python_line=_python_line_tokens(model),
        tokens_per_function=_function_tokens(model),
        identifier_fragmentation=_token_kind_fragmentation(model, {tokenize.NAME}),
        operator_fragmentation=_token_kind_fragmentation(model, {tokenize.OP}),
        import_fragmentation=_import_fragmentation(model),
        numeric_literal_handling=_numeric_handling(model),
        unicode_behavior=_round_trip(model, unicode_texts),
        round_trip_exactness=all_round,
        compression_ratio=_compression(model, texts),
        byte_fallback_success=_round_trip(model, fallback_texts),
        extras={
            "n_eval_docs": len(texts),
            "eval_names": names,
            "n_fallback_samples": len(fallback_texts),
        },
    )
