"""Public tokenizer API wrapping the frozen artifact."""

from __future__ import annotations

from pathlib import Path

from tiny_genius.tokenizer.artifacts import DEFAULT_TOKENIZER_DIR, load_model, verify_sha256sums
from tiny_genius.tokenizer.model import TokenizerModel
from tiny_genius.tokenizer.specials import SpecialTokens


class Tokenizer:
    """Load-only facade over a frozen Stage 3 tokenizer."""

    def __init__(self, model: TokenizerModel, directory: Path | None = None) -> None:
        self._model = model
        self._directory = directory

    @classmethod
    def load_frozen(cls, directory: str | Path | None = None) -> Tokenizer:
        dest = Path(directory) if directory else DEFAULT_TOKENIZER_DIR
        frozen = dest / "FROZEN.json"
        if not frozen.is_file():
            raise FileNotFoundError(f"tokenizer is not frozen: missing {frozen}")
        errors = verify_sha256sums(dest)
        if errors:
            raise ValueError("tokenizer artifact hash check failed: " + "; ".join(errors))
        return cls(load_model(dest), dest)

    @property
    def vocab_size(self) -> int:
        return self._model.vocab_size

    @property
    def special_tokens(self) -> SpecialTokens:
        return self._model.specials

    @property
    def version(self) -> str:
        return self._model.version

    @property
    def fingerprint(self) -> str:
        return self._model.fingerprint

    @property
    def algorithm(self) -> str:
        return self._model.algorithm

    @property
    def config(self) -> dict:
        return {
            "algorithm": self._model.algorithm,
            "vocab_size": self._model.vocab_size,
            "normalization": self._model.normalization,
            "version": self._model.version,
            "fingerprint": self._model.fingerprint,
            "specials": self._model.specials.to_dict(),
        }

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return self._model.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self._model.decode(ids, skip_special_tokens=skip_special_tokens)
