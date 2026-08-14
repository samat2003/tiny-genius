# Tiny Genius

A from-scratch engineering project for building and training a compact decoder-only Transformer for code, mathematics, and STEM.

The project is developed incrementally, starting with a tiny model and gradually building the infrastructure required for a larger production-scale training run.

## Project Goals

- Build a clean and reproducible Transformer implementation
- Develop a deterministic tokenizer and data pipeline
- Train and validate a small model before scaling up
- Support BF16 training and optional FP8 acceleration
- Support distributed and large-token-batch training
- Build reproducible checkpoints and experiment tracking
- Develop supervised fine-tuning and execution-based reinforcement learning
- Maintain strict data provenance and contamination controls
- Produce reproducible evaluation and release artifacts

## Planned Model

The final model configuration is approximately:

- 300M parameters
- 24 Transformer layers
- 960 hidden dimensions
- 15 attention heads
- 64 dimensions per attention head
- SwiGLU feed-forward network
- RMSNorm
- RoPE positional encoding
- 4,096-token context
- 32,768-token vocabulary
- Tied input/output embeddings
- No linear-layer biases

## Development Roadmap

The project follows a staged engineering process:

1. Repository and development infrastructure
2. Tiny Transformer
3. Tokenizer
4. Data pipeline
5. BF16 training
6. FP8 training
7. Distributed and large-batch training
8. Scaling experiments
9. Frozen training configuration
10. Large-scale pretraining
11. Supervised fine-tuning
12. Execution-based reinforcement learning
13. Evaluation
14. Reproducible release

Each stage has tests, reproducible commands, artifacts, and a clear completion criterion.

## Repository Structure

```text
tiny-genius/
├── README.md
├── 300M_Dense_Transformer_Full_Project_Plan.md
├── pyproject.toml
├── RUN_SPEC.yaml
├── configs/
├── src/
│   └── tiny_genius/
├── scripts/
├── tests/
├── docs/
└── artifacts/
```

## Stage 0 quickstart

```bash
python -m pip install -e ".[dev]"
python -c "import tiny_genius; print(tiny_genius.__version__)"
python scripts/smoke.py
python -m pytest
```

`make test`, `make lint`, and `make smoke` wrap the same checks.

Stage 0 is the engineering foundation (package, config contract, tests). The
Tiny Transformer is not implemented yet.
