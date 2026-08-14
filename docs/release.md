# Release

## Implemented now (Stage 0–2)

- Installable package and Stage 2 Tiny Transformer reference implementation.
- Checkpoint format for the debug trainer.
- No production weights, tokenizer artifacts, or evaluation reports.

## Planned release package

A complete release must document exactly what was trained, evaluated, and shipped:

- model configuration
- tokenizer
- data manifest summary
- contamination report
- training metrics
- checkpoint lineage
- SFT manifest summary
- RL task summary
- evaluation report
- release card
- reproducibility instructions

Raw datasets and large checkpoints stay outside Git. The repository stores the
code, configs, hashes, and instructions needed to reconstruct them.
