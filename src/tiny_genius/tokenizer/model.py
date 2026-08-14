"""In-memory tokenizer model and encode/decode."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field

from tiny_genius.tokenizer.normalize import Normalization, normalize
from tiny_genius.tokenizer.specials import (
    BOS_ID,
    BYTE_BASE,
    EOS_ID,
    FIRST_MERGE_ID,
    N_BYTE_TOKENS,
    SpecialTokens,
    byte_token_id,
)
from tiny_genius.tokenizer.trie import ByteTrie

TOKENIZER_FORMAT = "tiny-genius-tokenizer"
TOKENIZER_FORMAT_VERSION = 1


@dataclass
class TokenizerModel:
    version: str
    algorithm: str
    vocab_size: int
    normalization: Normalization
    specials: SpecialTokens
    token_bytes: list[bytes]
    merges: list[tuple[int, int, int]]
    char_to_id: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    _trie: ByteTrie | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.token_bytes) != self.vocab_size:
            raise ValueError(
                f"token_bytes length {len(self.token_bytes)} != vocab_size {self.vocab_size}"
            )

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "normalization": self.normalization,
            "merges": self.merges,
            "token_bytes": [base64.b64encode(b).decode("ascii") for b in self.token_bytes],
            "char_to_id": self.char_to_id,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode raw text. BOS/EOS are optional and not part of the text stream."""
        text = normalize(text, self.normalization)
        data = text.encode("utf-8")
        trie = self.trie()
        ids: list[int] = []
        i = 0
        while i < len(data):
            token_id, length = trie.longest(data, i)
            if token_id is None or length == 0:
                ids.append(byte_token_id(data[i]))
                i += 1
            else:
                ids.append(token_id)
                i += length
        if add_bos:
            ids = [self.specials.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.specials.eos_id]
        return ids

    def trie(self) -> ByteTrie:
        if self._trie is None:
            self._trie = ByteTrie()
            specials = self.specials.ids()
            for token_id, piece in enumerate(self.token_bytes):
                if token_id in specials or not piece or piece.startswith(b"<unused_"):
                    continue
                self._trie.insert(piece, token_id)
        return self._trie

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        specials = self.specials.ids()
        chunks: list[bytes] = []
        for token_id in ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token id out of range: {token_id}")
            if token_id in specials:
                if skip_special_tokens:
                    continue
                chunks.append(self.specials.id_to_token()[token_id].encode("utf-8"))
                continue
            piece = self.token_bytes[token_id]
            if piece:
                chunks.append(piece)
        return b"".join(chunks).decode("utf-8", errors="strict")


def empty_vocab(vocab_size: int, specials: SpecialTokens) -> list[bytes]:
    tokens = [b""] * vocab_size
    tokens[specials.pad_id] = specials.pad.encode("utf-8")
    tokens[specials.unk_id] = specials.unk.encode("utf-8")
    tokens[specials.bos_id] = specials.bos.encode("utf-8")
    tokens[specials.eos_id] = specials.eos.encode("utf-8")
    for byte in range(N_BYTE_TOKENS):
        tokens[byte_token_id(byte)] = bytes([byte])
    return tokens


def fill_unused(token_bytes: list[bytes], start: int) -> None:
    for idx in range(start, len(token_bytes)):
        if not token_bytes[idx]:
            token_bytes[idx] = f"<unused_{idx}>".encode("utf-8")


def register_merge_bytes(
    token_bytes: list[bytes], merges: list[tuple[int, int, int]]
) -> None:
    for left, right, new_id in merges:
        token_bytes[new_id] = token_bytes[left] + token_bytes[right]


# Keep FIRST_MERGE_ID / BYTE_BASE available to trainers.
__all__ = [
    "BYTE_BASE",
    "BOS_ID",
    "EOS_ID",
    "FIRST_MERGE_ID",
    "TOKENIZER_FORMAT",
    "TOKENIZER_FORMAT_VERSION",
    "TokenizerModel",
    "empty_vocab",
    "fill_unused",
    "register_merge_bytes",
]
