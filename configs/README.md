# Configuration hierarchy

Configurations are external YAML files. Code must not hard-code future 300M
training hyperparameters.

| File | Status | Purpose |
|---|---|---|
| `stage0.yaml` | Implemented now | Development / Stage 0 identity and seed |
| `tiny.yaml` | Implemented (Stage 2) | Debug Tiny Transformer |
| `tokenizer.yaml` / `tokenizer_candidates.yaml` / `tokenizer_thresholds.yaml` | Implemented (Stage 3) | Tokenizer study + Gate G3 |
| `small.yaml` / `medium.yaml` | Planned | Scaling experiments |
| `model_300m.yaml` | Planned | Architecture freeze candidate |
| `pretrain.yaml` / `sft.yaml` / `rl.yaml` | Planned | Training-phase recipes |

`RUN_SPEC.yaml` at the repository root is the contract that names the active
config identity. It stays `frozen: false` until the project plan's RUN SPEC freeze stage.
