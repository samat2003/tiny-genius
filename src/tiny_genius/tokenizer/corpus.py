"""Deterministic tokenizer evaluation and training corpora."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from tiny_genius.reproducibility import set_global_seed

CORPUS_VERSION = "stage3-eval-v1"


def _python_documents() -> list[tuple[str, str]]:
    return [
        (
            "python/imports",
            "import os\nimport sys\nfrom typing import Optional, List, Dict, Tuple\n"
            "from pathlib import Path\nimport numpy as np\n",
        ),
        (
            "python/function",
            "def add(x: int, y: int) -> int:\n    return x + y\n",
        ),
        (
            "python/class",
            "class Point:\n    def __init__(self, x: float, y: float) -> None:\n"
            "        self.x = x\n        self.y = y\n\n"
            "    def dist(self) -> float:\n        return (self.x ** 2 + self.y ** 2) ** 0.5\n",
        ),
        (
            "python/decorator",
            "from functools import wraps\n\ndef logged(fn):\n    @wraps(fn)\n"
            "    def inner(*args, **kwargs):\n        return fn(*args, **kwargs)\n"
            "    return inner\n\n@logged\ndef run() -> None:\n    pass\n",
        ),
        (
            "python/comprehension",
            "squares = [i * i for i in range(10) if i % 2 == 0]\n"
            "mapping = {k: v for k, v in enumerate(squares)}\n"
            "unique = {x for x in squares}\n",
        ),
        (
            "python/generator",
            "def countdown(n: int):\n    while n > 0:\n        yield n\n        n -= 1\n",
        ),
        (
            "python/async",
            "async def fetch(url: str) -> str:\n    await ready()\n    return url\n\n"
            "async def ready() -> None:\n    return None\n",
        ),
        (
            "python/exceptions",
            "try:\n    raise ValueError('bad')\nexcept ValueError as exc:\n"
            "    print(exc)\nfinally:\n    pass\n",
        ),
        (
            "python/context",
            "from contextlib import contextmanager\n\n"
            "@contextmanager\ndef opened(path):\n    f = open(path, 'r', encoding='utf-8')\n"
            "    try:\n        yield f\n    finally:\n        f.close()\n",
        ),
        (
            "python/fstring",
            'name = "Ada"\nvalue = 3.14159\nmsg = f"{name}: {value:.2f}"\n',
        ),
        (
            "python/strings",
            "a = 'single'\nb = \"double\"\nc = '''triple'''\nd = \"\"\"also\"\"\"\n",
        ),
        (
            "python/comments",
            "# leading comment\nx = 1  # inline comment\n",
        ),
        (
            "python/indent",
            "def nest(flag: bool) -> int:\n    if flag:\n        if True:\n"
            "            return 1\n        return 2\n    return 0\n",
        ),
        (
            "python/operators",
            "x = 1 + 2 - 3 * 4 / 5 // 6 % 7 ** 2\n"
            "ok = a == b and c != d or e <= f and g >= h\n"
            "bits = (m << 2) | (n >> 1) & ~mask\n"
            "x += 1\nx := 2\n",
        ),
        (
            "python/walrus_ann",
            "def parse(items: list[int]) -> int | None:\n"
            "    if (n := len(items)) > 0:\n        return n\n    return None\n",
        ),
        (
            "python/unicode_ident",
            "π = 3.14159\nΔt = 1e-3\ndef café(α: float) -> float:\n    return α * π\n",
        ),
        (
            "python/long_ident",
            "def compute_normalized_mean_squared_error_estimate("
            "prediction_vector, target_vector):\n"
            "    return ((prediction_vector - target_vector) ** 2).mean()\n",
        ),
        (
            "python/punctuation",
            "data = {'a': [1, 2, 3], 'b': (4, 5), 'c': {6, 7}}\n"
            "call(foo, bar=baz, *args, **kwargs)\n",
        ),
        (
            "python/nested",
            "result = [[f(x) for x in row if x] for row in matrix if row]\n",
        ),
        (
            "python/tabs_spaces",
            "def mixed():\n\treturn 1\n    return 2\n",
        ),
        (
            "python/windows_lines",
            "line1 = 1\r\nline2 = 2\r\n",
        ),
        ("empty", ""),
        ("short", "x"),
        ("whitespace", " \t\n  \n"),
        ("empty_string_lit", 's = ""\n'),
    ]


def _math_documents() -> list[tuple[str, str]]:
    return [
        ("math/equation", "E = m c^2\nF = G m1 m2 / r^2\n"),
        ("math/fraction", "y = (a + b) / (c - d)\n"),
        ("math/exponent", "σ = 1.2e-3\nN = 6.022e23\n"),
        ("math/ops", "∇·E = ρ/ε0\n∂u/∂t + u·∇u = -∇p + ν∇²u\n"),
        ("math/vars", "let x_i = Σ_j A_{ij} v_j\n"),
        ("math/unicode", "∫_0^∞ e^{-x} dx = 1\nα + β = γ\n"),
    ]


def _stem_documents() -> list[tuple[str, str]]:
    return [
        (
            "stem/prose",
            "The Reynolds number Re = ρ U L / μ is dimensionless.\n",
        ),
        (
            "stem/units",
            "c = 2.99792458e8 m/s\nPlanck's constant h = 6.62607015e-34 J s\n",
        ),
        (
            "stem/formula",
            "ΔG = ΔH - T ΔS\n",
        ),
        (
            "stem/unicode",
            "The wavelength λ = 550 nm. Temperature θ ≈ 300 K. Vector â.\n",
        ),
        (
            "stem/mixed",
            "Compute RMSE for predictions in Python:\n"
            "rmse = ((y_hat - y) ** 2).mean() ** 0.5  # units: Kelvin\n",
        ),
    ]


def _edge_documents() -> list[tuple[str, str]]:
    args = ", ".join(f"arg_{i}: int" for i in range(40))
    long_text = f"def long({args}) -> int:\n    return 0\n"
    return [
        ("edge/ascii", "The quick brown fox jumps over the lazy dog 0123456789.\n"),
        ("edge/unicode", "你好世界 — café — naïve — Москва — عربي — 🙂\n"),
        ("edge/punctuation", "!@#$%^&*()[]{}<>/\\|`~;:,.?\n"),
        ("edge/long", long_text),
        ("edge/quotes", "He said, \"n='ok'\" and left.\n"),
        ("edge/nullish", "zero\x00byte and \x7f\n"),
    ]


def evaluation_documents() -> list[tuple[str, str]]:
    docs = []
    docs.extend(_python_documents())
    docs.extend(_math_documents())
    docs.extend(_stem_documents())
    docs.extend(_edge_documents())
    return docs


def evaluation_text() -> str:
    parts = []
    for _, text in evaluation_documents():
        if text and not text.endswith("\n"):
            parts.append(text + "\n")
        else:
            parts.append(text)
    return "".join(parts)


def python_subset() -> list[str]:
    return [text for name, text in evaluation_documents() if name.startswith("python/")]


def unicode_subset() -> list[str]:
    return [
        text
        for name, text in evaluation_documents()
        if name.startswith("math/") or name.startswith("stem/") or "unicode" in name
    ]


def training_documents(*, extra_lines: int = 4000, seed: int = 42) -> list[str]:
    """Evaluation docs plus deterministic synthetic Python/STEM lines."""
    set_global_seed(seed)
    docs = [text for _, text in evaluation_documents()]
    keywords = [
        "def",
        "class",
        "return",
        "import",
        "from",
        "async",
        "await",
        "yield",
        "except",
        "finally",
    ]
    stems = [
        "compute",
        "normalize",
        "estimate",
        "residual",
        "gradient",
        "tensor",
        "matrix",
        "solver",
        "energy",
        "momentum",
    ]
    for i in range(extra_lines):
        stem = stems[i % len(stems)]
        kw = keywords[i % len(keywords)]
        ident = f"{stem}_{i}_value"
        number = f"{(i % 997) + 1}.{(i * 7) % 1000}e{(i % 9) - 4}"
        line = (
            f"{kw} {ident}(x_{i % 17}: float) -> float:\n"
            f"    y = x_{i % 17} * {number} + {i % 13}\n"
            f"    if y == 0 or y != y:\n"
            f"        return 0.0\n"
            f"    return y ** 2 // 1\n"
        )
        if i % 11 == 0:
            line += f"    # unit: kg m / s^2  λ={i}  Δ={i % 5}\n"
        if i % 19 == 0:
            line += "from typing import Optional, Sequence\nimport math\n"
        docs.append(line)
    return docs


def corpus_hash(texts: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(CORPUS_VERSION.encode("utf-8"))
    for text in texts:
        encoded = text.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    return hasher.hexdigest()


@dataclass(frozen=True)
class CorpusBundle:
    version: str
    evaluation: list[tuple[str, str]]
    training: list[str]
    evaluation_hash: str
    training_hash: str


def build_corpus(*, extra_lines: int = 4000, seed: int = 42) -> CorpusBundle:
    evaluation = evaluation_documents()
    training = training_documents(extra_lines=extra_lines, seed=seed)
    return CorpusBundle(
        version=CORPUS_VERSION,
        evaluation=evaluation,
        training=training,
        evaluation_hash=corpus_hash([text for _, text in evaluation]),
        training_hash=corpus_hash(training),
    )
