# Architecture

## Implemented now (Stage 0)

- Python package `tiny_genius` with configuration loading and reproducibility helpers.
- `RUN_SPEC.yaml` as the Stage 0 contract (`frozen: false`).
- Development config at `configs/stage0.yaml`.
- No model layers, attention, RoPE, RMSNorm, or embeddings.

## Planned (later stages)

The frozen target from the project plan is a decoder-only causal Transformer:

| Component | Target |
|---|---|
| Parameters | ~296.9M |
| Layers | 24 |
| Hidden size | 960 |
| Attention heads | 15 |
| Head dimension | 64 |
| FFN | SwiGLU, hidden 2,560 |
| Normalization | Pre-norm RMSNorm |
| Position encoding | RoPE |
| Context | 4,096 tokens |
| Vocabulary | 32,768 |
| Embeddings | Tied input/output |
| Linear biases | None |

Stage 1 (Tiny Transformer in the README roadmap) implements the same mechanics at
debug scale (4 layers, hidden 256, 4 heads) before any 300M work.

Configuration remains external. Do not hard-code the 300M recipe into package code.
