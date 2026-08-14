"""Stage 3 tokenizer unit tests (small vocab + frozen contract when present)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_genius.tokenizer.artifacts import DEFAULT_TOKENIZER_DIR, load_model, verify_sha256sums
from tiny_genius.tokenizer.corpus import evaluation_documents
from tiny_genius.tokenizer.specials import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from tiny_genius.tokenizer.train import train_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny() :
    docs = [text for _, text in evaluation_documents()]
    return train_candidate(
        name="unit",
        algorithm="byte_bpe_code",
        texts=docs,
        vocab_size=512,
        version="test",
    )


def test_encode_ids_in_range_and_decode_str() -> None:
    model = _tiny()
    ids = model.encode("def add(x, y):\n    return x + y\n")
    assert ids
    assert all(isinstance(i, int) and 0 <= i < model.vocab_size for i in ids)
    assert isinstance(model.decode(ids), str)


def test_exact_round_trip_suite() -> None:
    model = _tiny()
    samples = [text for _, text in evaluation_documents()]
    samples.extend(["", "x", " \t\n", "🙂", "π = 3.14", "import os\n"])
    for text in samples:
        assert model.decode(model.encode(text)) == text, repr(text)


def test_bos_eos_optional_and_not_colliding() -> None:
    model = _tiny()
    text = "hello"
    raw = model.encode(text)
    wrapped = model.encode(text, add_bos=True, add_eos=True)
    assert wrapped[0] == BOS_ID
    assert wrapped[-1] == EOS_ID
    assert wrapped[1:-1] == raw
    assert model.decode(wrapped, skip_special_tokens=True) == text
    assert "<bos>" in model.decode(wrapped, skip_special_tokens=False)
    # Surface form is ordinary text, not special ID 2.
    encoded_surface = model.encode("<bos>")
    assert BOS_ID not in encoded_surface
    assert EOS_ID not in encoded_surface
    assert PAD_ID not in encoded_surface
    assert UNK_ID not in encoded_surface
    assert model.decode(encoded_surface) == "<bos>"


def test_byte_fallback_unusual_unicode() -> None:
    model = _tiny()
    samples = ["🙂🔥", "한글", "𐍈", "a\u0301", bytes(range(32)).decode("latin-1")]
    for text in samples:
        assert model.decode(model.encode(text)) == text


def test_python_source_preserved() -> None:
    model = _tiny()
    source = "def nest(flag: bool) -> int:\n    if flag:\n        return 1\n    return 0\n"
    assert model.decode(model.encode(source)) == source
    tabs = "def mixed():\n\treturn 1\n"
    assert model.decode(model.encode(tabs)) == tabs
    crlf = "a = 1\r\nb = 2\r\n"
    assert model.decode(model.encode(crlf)) == crlf


def test_code_seed_keeps_operators_compact() -> None:
    model = _tiny()
    assert len(model.encode("==")) <= 2
    assert len(model.encode("import ")) <= 2


@pytest.mark.skipif(
    not (DEFAULT_TOKENIZER_DIR / "FROZEN.json").is_file(),
    reason="frozen tokenizer artifacts not present",
)
def test_frozen_vocab_size_and_hashes() -> None:
    errors = verify_sha256sums(DEFAULT_TOKENIZER_DIR)
    assert errors == []
    model = load_model(DEFAULT_TOKENIZER_DIR)
    assert model.vocab_size == 32768
    assert len(model.token_bytes) == 32768
    from tiny_genius.tokenizer import Tokenizer

    tok = Tokenizer.load_frozen()
    assert tok.vocab_size == 32768
    assert tok.decode(tok.encode("import math\n")) == "import math\n"
    assert tok.config["normalization"] == "identity"
    assert tok.special_tokens.bos_id == BOS_ID
    assert tok.special_tokens.eos_id == EOS_ID
    assert tok.fingerprint == model.fingerprint
    # Fingerprint is recorded in the json payload.
    import json

    payload = json.loads((DEFAULT_TOKENIZER_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    assert payload["fingerprint"] == tok.fingerprint
    assert payload["vocab_size"] == 32768
    # Tiny Transformer debug vocab stays 128 — frozen tokenizer is independent.
    from tiny_genius.model import TinyModelConfig

    tiny = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "tiny.yaml")
    assert tiny.vocab_size == 128
    assert tiny.vocab_size != tok.vocab_size
