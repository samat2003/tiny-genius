"""Deterministic seed behavior."""

import random

import pytest

from tiny_genius.reproducibility import fingerprint, set_global_seed


@pytest.mark.reproducibility
def test_same_seed_same_sequence() -> None:
    set_global_seed(123)
    first = [random.random() for _ in range(8)]
    set_global_seed(123)
    second = [random.random() for _ in range(8)]
    assert first == second


@pytest.mark.reproducibility
def test_different_seeds_diverge() -> None:
    set_global_seed(1)
    a = [random.random() for _ in range(8)]
    set_global_seed(2)
    b = [random.random() for _ in range(8)]
    assert a != b


@pytest.mark.reproducibility
def test_fingerprint_is_stable() -> None:
    payload = {"seed": 42, "items": [3, 1, 2]}
    assert fingerprint(payload) == fingerprint({"items": [3, 1, 2], "seed": 42})
    assert fingerprint(payload) != fingerprint({"seed": 43, "items": [3, 1, 2]})


def test_set_global_seed_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        set_global_seed("42")  # type: ignore[arg-type]
