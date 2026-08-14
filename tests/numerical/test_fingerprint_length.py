"""Light numerical sanity checks that do not require a model."""

from tiny_genius.reproducibility import fingerprint


def test_fingerprint_is_sha256_hex() -> None:
    digest = fingerprint({"ok": True})
    assert len(digest) == 64
    int(digest, 16)
