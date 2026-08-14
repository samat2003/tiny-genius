"""Train tokenizer candidates from a corpus."""

from __future__ import annotations

from collections import Counter

from tiny_genius.tokenizer.bpe import (
    bytes_to_ids,
    code_seed_pieces,
    train_bpe_merges,
)
from tiny_genius.tokenizer.model import (
    TokenizerModel,
    empty_vocab,
    fill_unused,
    register_merge_bytes,
)
from tiny_genius.tokenizer.specials import FIRST_MERGE_ID, SpecialTokens


def _line_sequences_from_byte_ids(texts: list[str]) -> dict[tuple[int, ...], int]:
    counts: Counter[tuple[int, ...]] = Counter()
    for text in texts:
        for line in text.splitlines(keepends=True):
            encoded = line.encode("utf-8")
            if encoded:
                counts[tuple(bytes_to_ids(encoded))] += 1
    return dict(counts)


def _line_sequences_unicode(
    texts: list[str], char_to_id: dict[str, int]
) -> dict[tuple[int, ...], int]:
    counts: Counter[tuple[int, ...]] = Counter()
    for text in texts:
        for line in text.splitlines(keepends=True):
            if not line:
                continue
            ids = [char_to_id[ch] for ch in line if ch in char_to_id]
            if ids:
                counts[tuple(ids)] += 1
    return dict(counts)


def _assign_char_ids(texts: list[str], start_id: int, limit: int) -> dict[str, int]:
    seen: list[str] = []
    found: set[str] = set()
    for text in texts:
        for char in text:
            if char not in found:
                found.add(char)
                seen.append(char)
    mapping: dict[str, int] = {}
    next_id = start_id
    for char in seen:
        if next_id >= limit:
            break
        mapping[char] = next_id
        next_id += 1
    return mapping


def train_candidate(
    *,
    name: str,
    algorithm: str,
    texts: list[str],
    vocab_size: int,
    version: str = "1.0.0",
    metadata: dict | None = None,
) -> TokenizerModel:
    specials = SpecialTokens()
    token_bytes = empty_vocab(vocab_size, specials)
    char_to_id: dict[str, int] = {}
    start_id = FIRST_MERGE_ID

    if algorithm == "unicode_bpe_bytes":
        char_to_id = _assign_char_ids(texts, start_id, vocab_size)
        for char, token_id in char_to_id.items():
            token_bytes[token_id] = char.encode("utf-8")
        start_id = start_id + len(char_to_id)
        sequences = _line_sequences_unicode(texts, char_to_id)
    elif algorithm in {"byte_bpe", "byte_bpe_code"}:
        sequences = _line_sequences_from_byte_ids(texts)
        if algorithm == "byte_bpe_code":
            # Seed pieces are guaranteed as vocab entries; BPE still sees bytes.
            next_seed = start_id
            for piece in code_seed_pieces():
                if next_seed >= vocab_size:
                    break
                if piece and token_bytes[next_seed] == b"":
                    token_bytes[next_seed] = piece
                    next_seed += 1
            start_id = next_seed
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    n_merges = max(0, vocab_size - start_id)
    merges = train_bpe_merges(sequences, n_merges, start_id=start_id)
    register_merge_bytes(token_bytes, merges)
    used = start_id + len(merges)
    fill_unused(token_bytes, used)

    return TokenizerModel(
        version=version,
        algorithm=algorithm,
        vocab_size=vocab_size,
        normalization="identity",
        specials=specials,
        token_bytes=token_bytes,
        merges=merges,
        char_to_id=char_to_id,
        metadata={
            "candidate": name,
            "n_merges": len(merges),
            "n_unused": vocab_size - used,
            **(metadata or {}),
        },
    )
