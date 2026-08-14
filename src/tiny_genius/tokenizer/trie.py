"""Byte trie for greedy longest-match encoding."""

from __future__ import annotations


class ByteTrie:
    def __init__(self) -> None:
        self._children: dict[int, ByteTrie] = {}
        self._token_id: int | None = None

    def insert(self, data: bytes, token_id: int) -> None:
        node = self
        for byte in data:
            node = node._children.setdefault(byte, ByteTrie())
        node._token_id = token_id

    def longest(self, data: bytes, start: int) -> tuple[int | None, int]:
        node: ByteTrie = self
        last_id: int | None = None
        last_len = 0
        length = 0
        for i in range(start, len(data)):
            child = node._children.get(data[i])
            if child is None:
                break
            node = child
            length += 1
            if node._token_id is not None:
                last_id = node._token_id
                last_len = length
        return last_id, last_len
