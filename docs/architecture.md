# Architecture

## Implemented now

### Stage 0

- Python package `tiny_genius` with configuration loading and reproducibility helpers.
- `RUN_SPEC.yaml` as the engineering contract (`frozen: false`).

### Stage 2 — Tiny Transformer (debug model)

A decoder-only causal Transformer at **debug scale**. This is **not** the 300M model.

| Parameter | Tiny value |
|---|---:|
| Layers | 4 |
| Hidden (`d_model`) | 256 |
| Heads | 4 |
| Head dim | 64 |
| FFN hidden (SwiGLU) | 640 (plan range 512–768) |
| Context | 256 |
| Vocabulary | 128 integer debug ids |
| Normalization | Pre-norm RMSNorm (`eps=1e-6`) |
| Position encoding | RoPE (`theta=10000`) |
| Embeddings | Tied input/output |
| Linear biases | None |
| Dropout | 0.0 |

Components:

- `TokenEmbedding` plus tied unembedding (`hidden @ W^T`)
- Causal self-attention with an additive future-token mask
- RoPE on Q/K
- Pre-norm `RMSNorm`
- `SwiGLU` FFN
- Residual Transformer blocks and a final RMSNorm

Instantiate from config:

```python
from tiny_genius import TinyModelConfig, TinyTransformer

config = TinyModelConfig.from_yaml("configs/tiny.yaml")
model = TinyTransformer(config)
```

### Stage 3 — Tokenizer (frozen)

Production tokenizer artifacts live in `tokenizer/`. Algorithm: in-repo
byte-level BPE with Python/operator seed pieces (`byte_bpe_code`), identity
normalization, byte fallback, vocab **32768**. See `docs/tokenizer.md`.

The Tiny Transformer is **not** remapped onto this vocabulary yet.

### Stage 4 — Data pipeline (10M milestone)

Deterministic, auditable pipeline under `src/tiny_genius/data/`. See
`docs/data.md` and `docs/data_contract.md`. Raw shards are not in git.

## Planned (later stages)

The frozen **target** from the project plan is a ~296.9M decoder-only Transformer
(24 layers, hidden 960, 15 heads, context 4096, vocab 32768). That recipe is not
implemented and must not be loaded through `TinyModelConfig`.
