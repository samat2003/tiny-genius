"""Deterministic BPE training and encoding (no external tokenizer library)."""

from __future__ import annotations

from collections import Counter

from tiny_genius.tokenizer.specials import FIRST_MERGE_ID, byte_token_id


def bytes_to_ids(data: bytes) -> list[int]:
    return [byte_token_id(b) for b in data]


def apply_seed_pieces(data: bytes, seeds: list[bytes]) -> list[int]:
    """Greedy longest-match of seed byte pieces, otherwise single bytes."""
    if not seeds:
        return bytes_to_ids(data)
    ordered = sorted(set(seeds), key=lambda p: (-len(p), p))
    ids: list[int] = []
    i = 0
    while i < len(data):
        matched: bytes | None = None
        for piece in ordered:
            if piece and data.startswith(piece, i):
                matched = piece
                break
        if matched is None:
            ids.append(byte_token_id(data[i]))
            i += 1
        else:
            # Seed pieces are encoded as their byte sequence for later BPE;
            # atomicity is recovered by treating each seed as an initial token
            # id assigned by the trainer.
            ids.extend(bytes_to_ids(matched))
            i += len(matched)
    return ids


def merge_pair_in_sequence(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    left, right = pair
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def pair_counts(sequences: dict[tuple[int, ...], int]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for seq, freq in sequences.items():
        for a, b in zip(seq, seq[1:]):
            counts[a, b] += freq
    return counts


def best_pair(counts: Counter[tuple[int, int]]) -> tuple[int, int] | None:
    if not counts:
        return None
    # Deterministic: highest frequency, then smaller (left, right).
    return max(counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))[0]


def apply_merge_to_corpus(
    sequences: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int, ...], int]:
    merged: dict[tuple[int, ...], int] = {}
    for seq, freq in sequences.items():
        new_seq = tuple(merge_pair_in_sequence(list(seq), pair, new_id))
        merged[new_seq] = merged.get(new_seq, 0) + freq
    return merged


def train_bpe_merges(
    sequences: dict[tuple[int, ...], int],
    n_merges: int,
    start_id: int = FIRST_MERGE_ID,
) -> list[tuple[int, int, int]]:
    """Learn up to n_merges. Each merge is (left_id, right_id, new_id)."""
    merges: list[tuple[int, int, int]] = []
    current = sequences
    next_id = start_id
    for _ in range(n_merges):
        counts = pair_counts(current)
        pair = best_pair(counts)
        if pair is None or counts[pair] < 2:
            break
        merges.append((pair[0], pair[1], next_id))
        current = apply_merge_to_corpus(current, pair, next_id)
        next_id += 1
    return merges


def apply_bpe(ids: list[int], merges: list[tuple[int, int, int]]) -> list[int]:
    """Apply merges by rank (earlier merge = higher priority)."""
    if not ids or not merges:
        return list(ids)
    rank = {(a, b): i for i, (a, b, _) in enumerate(merges)}
    new_id = {(a, b): nid for a, b, nid in merges}
    symbols = list(ids)
    while len(symbols) >= 2:
        best_i = -1
        best_rank = len(merges) + 1
        for i, pair in enumerate(zip(symbols, symbols[1:])):
            r = rank.get(pair)
            if r is not None and r < best_rank:
                best_rank = r
                best_i = i
        if best_i < 0:
            break
        pair = (symbols[best_i], symbols[best_i + 1])
        symbols = symbols[:best_i] + [new_id[pair]] + symbols[best_i + 2 :]
    return symbols


def code_seed_pieces() -> list[bytes]:
    """Common Python/STEM pieces for the code-aware candidate."""
    keywords = (
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    )
    operators = (
        "==",
        "!=",
        "<=",
        ">=",
        "//",
        "**",
        "->",
        ":=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "//=",
        "**=",
        "<<",
        ">>",
        "...",
    )
    extras = (
        "    ",
        "\t",
        "\n",
        "\n    ",
        "import ",
        "from ",
        "def ",
        "class ",
        "return ",
        "self.",
        "None",
        "True",
        "False",
        "__init__",
        "__name__",
        "print(",
        "range(",
        "len(",
        "typing",
        "Optional",
        "List",
        "Dict",
    )
    pieces = [k.encode("utf-8") for k in keywords]
    pieces.extend(op.encode("utf-8") for op in operators)
    pieces.extend(extra.encode("utf-8") for extra in extras)
    return pieces


def unicode_units(text: str) -> list[bytes]:
    """Each Unicode scalar as UTF-8 bytes (for unicode BPE units)."""
    return [ch.encode("utf-8") for ch in text]
