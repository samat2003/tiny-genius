"""Explicit special-token policy for the Stage 3 tokenizer.

Reserved IDs are never emitted by encode() of raw text. Surface forms such as
``<bos>`` in user text are encoded as ordinary UTF-8 bytes, not as specials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

BYTE_BASE = 4
N_BYTE_TOKENS = 256
FIRST_MERGE_ID = BYTE_BASE + N_BYTE_TOKENS  # 260

SPECIAL_TOKEN_LIST = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = PAD_TOKEN
    unk: str = UNK_TOKEN
    bos: str = BOS_TOKEN
    eos: str = EOS_TOKEN
    pad_id: int = PAD_ID
    unk_id: int = UNK_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID

    def id_to_token(self) -> dict[int, str]:
        return {
            self.pad_id: self.pad,
            self.unk_id: self.unk,
            self.bos_id: self.bos,
            self.eos_id: self.eos,
        }

    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in self.id_to_token().items()}

    def ids(self) -> frozenset[int]:
        return frozenset(self.id_to_token())

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def byte_token_id(byte: int) -> int:
    if not 0 <= byte <= 255:
        raise ValueError(f"byte out of range: {byte}")
    return BYTE_BASE + byte


def byte_from_token_id(token_id: int) -> int:
    byte = token_id - BYTE_BASE
    if not 0 <= byte <= 255:
        raise ValueError(f"not a byte token id: {token_id}")
    return byte
