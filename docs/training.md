# Training

## Implemented now

### Stage 0

- Seed policy (`set_global_seed`) and environment metadata.

### Stage 2

- A **debug** next-token loop on a fixed integer corpus (`make_tiny_corpus`).
- No tokenizer. Token ids are synthetic modular sequences.
- Optimizer: AdamW, used only to prove learning and resume.
- Command:

```bash
python scripts/train_tiny.py --config configs/tiny.yaml
```

Checkpoints write under `artifacts/` (gitignored except `artifacts/README.md`).

### Checkpoint format (`tiny-genius-checkpoint` v1)

A `torch.save` dict:

| Key | Meaning |
|---|---|
| `format` / `format_version` | Identity (`tiny-genius-checkpoint`, `1`) |
| `step` | Completed training steps |
| `config` | `TinyModelConfig` mapping |
| `model` | `state_dict` |
| `optimizer` | AdamW `state_dict` (or `None`) |
| `rng` | Python and PyTorch RNG payloads |
| `metadata` | Optional extras plus environment snapshot |

Resume restores model weights, optimizer state, step, and RNG so the next
updates match a continuous run on CPU.

### Determinism

`set_global_seed` seeds Python and PyTorch. Tests expect bit-stable trajectories
on **CPU** with `dropout=0`. CUDA / cuDNN are not claimed to be bit-identical
across devices or library versions.

### Stage 3

Tokenizer training is a one-shot freeze (`scripts/train_tokenizer.py`). After
`tokenizer/FROZEN.json` exists, the process will not silently regenerate the
artifact. See `docs/tokenizer.md`.

## Not implemented

BF16/FP8, distributed training, large-token-batch trainers, production
checkpoint sharding, SFT, and RL.

Until the RUN SPEC freeze stage, `RUN_SPEC.yaml` stays `frozen: false`.
