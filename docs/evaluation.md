# Evaluation

## Implemented now (Stage 0–2)

- No public-benchmark runners.
- Stage 2 validates learning only via next-token loss on a synthetic corpus.
- No HumanEval / MBPP integration.
- No decoding or execution harness.

## Planned

Evaluation is a late-stage responsibility. The project plan requires:

- Frozen decoding parameters before SFT/RL evaluation.
- Isolated benchmark prompts and hidden tests (never training targets).
- Recorded benchmark versions, prompt formatting, execution environment,
  raw counts, percentages, and artifact hashes.
- Private holdouts to detect overfitting.

The 99% HumanEval / MBPP figures in the plan are acceptance *targets* to test
honestly, not capabilities of this repository today.
