"""EOS-separated packing into 4096-token sequences."""

from __future__ import annotations

from typing import Any

from tiny_genius.tokenizer.specials import EOS_ID, PAD_ID


def pack_documents(
    docs: list[dict[str, Any]],
    *,
    n_ctx: int = 4096,
    shard_target_tokens: int = 1_000_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequences: list[list[int]] = []
    current: list[int] = []
    packed_tokens = 0
    pad_tokens = 0
    truncated_docs = 0

    for doc in docs:
        ids = list(doc["token_ids"])
        if ids and ids[-1] != EOS_ID:
            ids.append(EOS_ID)
        if len(ids) > n_ctx:
            ids = ids[: n_ctx - 1] + [EOS_ID]
            truncated_docs += 1
        if len(current) + len(ids) > n_ctx:
            pad = n_ctx - len(current)
            current.extend([PAD_ID] * pad)
            pad_tokens += pad
            sequences.append(current)
            packed_tokens += n_ctx
            current = []
        current.extend(ids)
    if current:
        pad = n_ctx - len(current)
        current.extend([PAD_ID] * pad)
        pad_tokens += pad
        sequences.append(current)
        packed_tokens += n_ctx

    shards: list[dict[str, Any]] = []
    shard: list[list[int]] = []
    shard_tokens = 0
    shard_idx = 0
    for seq in sequences:
        shard.append(seq)
        shard_tokens += len(seq)
        if shard_tokens >= shard_target_tokens:
            shards.append({"shard_id": f"shard-{shard_idx:05d}", "n_sequences": len(shard)})
            shard_idx += 1
            shard = []
            shard_tokens = 0
    if shard:
        shards.append({"shard_id": f"shard-{shard_idx:05d}", "n_sequences": len(shard)})

    waste = 0.0 if packed_tokens == 0 else pad_tokens / packed_tokens
    stats = {
        "n_sequences": len(sequences),
        "n_shards": len(shards),
        "packed_tokens": packed_tokens,
        "pad_tokens": pad_tokens,
        "waste_ratio": waste,
        "truncated_docs": truncated_docs,
        "n_ctx": n_ctx,
        "shards": shards,
    }
    return sequences, stats
