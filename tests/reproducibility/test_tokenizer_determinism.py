"""Tokenizer training and encoding are deterministic."""

from tiny_genius.tokenizer.corpus import evaluation_documents
from tiny_genius.tokenizer.train import train_candidate


def test_train_twice_identical_fingerprint_and_ids() -> None:
    texts = [text for _, text in evaluation_documents()]
    first = train_candidate(name="a", algorithm="byte_bpe", texts=texts, vocab_size=400)
    second = train_candidate(name="a", algorithm="byte_bpe", texts=texts, vocab_size=400)
    assert first.fingerprint == second.fingerprint
    assert first.merges == second.merges
    assert first.token_bytes == second.token_bytes
    sample = "def f(x):\n    return x == 1\n"
    assert first.encode(sample) == second.encode(sample)
    assert first.decode(first.encode(sample)) == sample
