# Tokenizer (Stage 3)

## Implemented now

A **frozen** production tokenizer artifact at `tokenizer/`, selected by a
reproducible candidate study against **predefined** Gate G3 thresholds
(`configs/tokenizer_thresholds.yaml`).

This is **not** a Stage 4 dataset pipeline. The evaluation/training texts are a
deterministic tokenizer-development corpus only.

## Algorithm

In-repo **UTF-8 byte-level BPE** plus a greedy longest-match trie. No external
tokenizer library is used at runtime.

- Normalization: **identity** (no NFC/NFKC, no tab expansion, no newline rewrite).
- Byte fallback: every UTF-8 byte is a token (IDs 4–259). Unseen Unicode cannot fail.
- `encode(text)` does **not** insert BOS/EOS unless `add_bos` / `add_eos`.
- `decode(ids)` drops specials when `skip_special_tokens=True` (default).
- Exact contract: `decode(encode(text)) == text` for raw `str` input.

## Special tokens

| Token | ID | Role |
|---|---:|---|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Reserved (byte fallback makes this unused for raw text) |
| `<bos>` | 2 | Beginning of sequence |
| `<eos>` | 3 | End of sequence |

IDs 4–259 are the 256 UTF-8 bytes. Merges and seeded code pieces follow.
Remaining IDs are explicit `<unused_N>` slots so the vocabulary is **exactly**
32,768.

The surface string `<bos>` in user text encodes as ordinary bytes, never as ID 2.

## Candidate study

All three candidates used the same corpus (`stage3-eval-v1`), same vocab size,
same identity normalization, and the same hard thresholds (written before
training).

Training corpus: evaluation documents plus 400 deterministic synthetic
Python/STEM functions (`seed=42`).

| Candidate | tok/py line | tok/fn | ident frag | op frag | import frag | numeric ≤6 | unicode RT | round-trip | bytes/token | passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `byte_bpe` | 8.163 | 20.29 | 1.744 | 1.082 | 6.857 | 1.00 | 1.00 | 1.00 | 2.845 | yes |
| **`byte_bpe_code`** | **7.814** | **19.50** | **1.535** | **1.000** | **6.286** | **1.00** | **1.00** | **1.00** | **2.896** | **yes** |
| `unicode_bpe_bytes` | 8.081 | 20.00 | 1.799 | 1.082 | 6.714 | 1.00 | 1.00 | 1.00 | 2.940 | yes |

`unicode_bpe_bytes` had the best raw compression (multi-byte Unicode as atoms)
but worse Python-line and identifier metrics. The predefined ranking is:

1. lower tokens / Python line
2. lower identifier fragmentation
3. higher compression ratio

**Winner: `byte_bpe_code`** — seeded Python keywords, multi-character operators,
and indent/import pieces on top of byte-level BPE.

## Predefined thresholds (hard)

| Metric | Rule |
|---|---|
| vocab_size | exact 32768 |
| round_trip_exactness | exact 1.0 |
| byte_fallback_success | exact 1.0 |
| unicode_behavior | exact 1.0 |
| tokens_per_python_line | ≤ 22 |
| tokens_per_function | ≤ 450 |
| identifier_fragmentation | ≤ 4.0 |
| operator_fragmentation | ≤ 2.0 |
| import_fragmentation | ≤ 18 |
| numeric_literal_handling | ≥ 0.90 |
| compression_ratio | ≥ 1.35 |

## Frozen artifact

| File | Role |
|---|---|
| `tokenizer/tokenizer.model` | Compact JSON model (same payload as `.json`) |
| `tokenizer/tokenizer.json` | Vocab, merges, config, fingerprint |
| `tokenizer/special_tokens.json` | Special-token policy |
| `tokenizer/tokenizer_metrics.json` | Winner metrics + study pointer |
| `tokenizer/SHA256SUMS` | SHA-256 of the four files above |
| `tokenizer/FROZEN.json` | Freeze record (do not silently retrain) |
| `tokenizer/candidate_study.json` | Full candidate comparison |

Fingerprint: `219156db6bbe8c573c0f1654ab9f622c0e8bd51519561ac30d2c13fbf3a01a6e`

Evaluation corpus hash: `6b63dd1277fb53d43567f1a8acfda016cf394d04f210bebb49e2691983520ff7`

Training corpus hash: `99e0d119ce46f45b00c9d7291ffccfe6074ea49bacfcf551f37e4565ef625bc8`

Rebuild: training the winner a second time in the same process produced the
same fingerprint. `scripts/train_tokenizer.py` refuses to overwrite
`FROZEN.json` unless `--allow-overwrite` is passed (new version required).

Verify:

```bash
python scripts/verify_tokenizer.py
```

Load:

```python
from tiny_genius import Tokenizer

tok = Tokenizer.load_frozen()
ids = tok.encode("def add(x, y):\n    return x + y\n")
assert tok.decode(ids) == "def add(x, y):\n    return x + y\n"
```

The Stage 2 Tiny Transformer still uses a **128-id debug vocabulary**. It is
not wired to this 32,768 tokenizer yet (tokenizer/data integration is later).

## Unused slots

About 950 BPE merges were supported by the Stage 3 development corpus; remaining
IDs are named `<unused_N>`. That keeps the contract at exactly 32,768 without
pretending a tiny tokenizer-dev corpus is the 13B-token pretraining mix.
A later tokenizer **version** may retrain on Stage 4 data; that must re-run
metrics and produce new hashes.

## Not implemented

Stage 4 data inventory, dedup, contamination, sharding, or 13B manifests.
