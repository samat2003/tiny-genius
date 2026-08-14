# Release

## Implemented now (Stage 0)

- Public repository scaffolding and installable package metadata.
- Documentation of what a later release must contain.
- No model weights, tokenizer artifacts, or evaluation reports.

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
