# Training

## Implemented now (Stage 0)

- Seed policy and environment metadata collection.
- A development configuration that is **not** a training recipe.
- No trainer, optimizer, scheduler, checkpointing, BF16, FP8, or distributed code.

## Planned

Training follows the project plan: tiny end-to-end trainability first, then
tokenizer and data manifests, a BF16 baseline, optional FP8, large-token-batch
distributed training, scaling pilots, then a frozen RUN SPEC before the 13B-token
pretraining budget.

### What is frozen when

| When | Frozen ingredient |
|---|---|
| After tokenizer stage | Tokenizer artifact + hash |
| After data pipeline | Dataset manifests and fingerprints |
| After RUN SPEC freeze | Full 300M training contract |
| After SFT/RL | Evaluation decoding settings |

Until the RUN SPEC freeze, `RUN_SPEC.yaml` stays `frozen: false`.

### Seed policy

Every meaningful run records an integer seed. `set_global_seed` currently seeds
Python `random` and `PYTHONHASHSEED`. Additional backends are seeded when added.
