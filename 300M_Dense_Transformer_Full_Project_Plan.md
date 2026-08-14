# 300M Dense Transformer — Full Project Execution Plan

## 0. Project Definition

This document converts the existing 300M dense Transformer specification into an execution program that can be handed to a coding agent and implemented from an empty public GitHub repository through the final evaluation/release stage.

The project is deliberately decomposed into small, testable milestones:

> **Public repo → reproducible tiny model → tokenizer/data pipeline → BF16 training → FP8 training → large-batch distributed training → scaling pilots → frozen 300M run → 13B-token pretraining → 600K SFT → execution-based RL → benchmark evaluation → release**

### Important feasibility constraint

The project target is **HumanEval pass@1 ≥99% and MBPP pass@1 ≥99%**, but a 300M-parameter recipe cannot guarantee that outcome. The correct engineering goal is to build a reproducible system that can rigorously test the target while preserving clean contamination controls and immutable run artifacts.

The frozen model target from the source plan is:

| Component | Target |
|---|---:|
| Architecture | Decoder-only causal Transformer |
| Parameters | ~296.9M |
| Layers | 24 |
| Hidden size | 960 |
| Attention heads | 15 |
| Head dimension | 64 |
| FFN | SwiGLU |
| FFN hidden size | 2,560 |
| Normalization | Pre-norm RMSNorm |
| Position encoding | RoPE |
| Context | 4,096 tokens |
| Vocabulary | 32,768 |
| Tokenizer | BPE/SentencePiece-style + byte fallback |
| Embeddings | Tied input/output |
| Linear biases | None |
| Main Transformer dropout | 0.0 |
| Pretraining budget | ~13B tokens |
| SFT dataset | 600K verified examples |
| RL problems | 10K |
| RL samples/problem | 10–50 |
| Main benchmark target | HumanEval ≥99%, MBPP ≥99% |

The architecture and parameter budget are taken from the supplied engineering plan, including the ~296.9M parameter calculation.

---

# 1. Success Definition

The project is successful only when all of the following are true:

1. The public repository can reproduce the documented training/evaluation workflows.
2. The model implementation matches the frozen architecture specification.
3. The tokenizer is frozen, hashed, tested, and reproducible.
4. Training data is represented by immutable manifests with provenance, deduplication, and contamination reports.
5. A BF16 baseline is stable.
6. FP8 training is implemented and validated against the BF16 baseline.
7. Large-token-batch distributed training is stable and measurable.
8. Scaling pilots support the final 300M configuration.
9. The 300M pretraining run completes reproducibly for the frozen budget.
10. The 600K-example SFT corpus is verified and frozen.
11. The SFT model improves useful coding performance without unacceptable regression.
12. The 10K-task execution RL pipeline is verified.
13. Final evaluation is performed with frozen decoding and clean benchmark isolation.
14. The release artifacts document exactly what was trained, evaluated, and released.
15. The 99% benchmark target is achieved **or** the result is honestly reported as a miss with diagnosis.

---

# 2. Engineering Principles

| Principle | Rule |
|---|---|
| Reproducibility | Every meaningful run must have a config, code commit, seed, dataset fingerprint, tokenizer hash, and checkpoint lineage. |
| Freeze before scale | Experiment freely before the RUN SPEC; do not silently change frozen ingredients during the main run. |
| Small-to-large | Prove every system component at tiny scale before spending serious compute. |
| Correctness first | Unit tests and deterministic tests come before performance optimization. |
| BF16 before FP8 | Establish a numerically trustworthy baseline before enabling FP8. |
| Batch size in tokens | Define large-batch behavior using global tokens, not only sequences/examples. |
| Measure throughput | Report tokens/sec/GPU, global tokens/sec, utilization, memory, and checkpoint overhead. |
| Data isolation | Benchmark prompts, reference solutions, and hidden tests never become training targets. |
| Execution verification | Synthetic code data must be executable and test-verified. |
| Immutable artifacts | Run specs, manifests, benchmark definitions, and evaluation settings are versioned/fingerprinted. |
| No silent failure | NaNs, contamination, data corruption, or reproducibility mismatches stop the pipeline and create a recorded incident. |

---

# 3. Repository Target

## 3.1 Public GitHub structure

```text
300m-code-transformer/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── tests.yml
│       ├── lint.yml
│       └── smoke.yml
│
├── configs/
│   ├── tiny.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   ├── model_300m.yaml
│   ├── pretrain.yaml
│   ├── sft.yaml
│   └── rl.yaml
│
├── src/
│   └── model_project/
│       ├── __init__.py
│       ├── config.py
│       ├── model/
│       │   ├── transformer.py
│       │   ├── attention.py
│       │   ├── rope.py
│       │   ├── rmsnorm.py
│       │   ├── swiglu.py
│       │   └── embeddings.py
│       ├── tokenizer/
│       │   ├── train.py
│       │   ├── encode.py
│       │   └── audit.py
│       ├── data/
│       │   ├── inventory.py
│       │   ├── normalize.py
│       │   ├── dedup.py
│       │   ├── contamination.py
│       │   ├── manifest.py
│       │   └── packing.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── optimizer.py
│       │   ├── scheduler.py
│       │   ├── precision.py
│       │   ├── fp8.py
│       │   ├── distributed.py
│       │   ├── checkpointing.py
│       │   └── metrics.py
│       ├── sft/
│       │   ├── verify.py
│       │   ├── generate.py
│       │   └── train.py
│       ├── rl/
│       │   ├── sandbox.py
│       │   ├── rollout.py
│       │   ├── reward.py
│       │   └── train.py
│       └── evaluation/
│           ├── humaneval.py
│           ├── mbpp.py
│           ├── holdouts.py
│           └── report.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark_throughput.py
│   ├── benchmark_memory.py
│   ├── train_tokenizer.py
│   ├── build_manifest.py
│   ├── run_contamination_scan.py
│   ├── run_pilot.py
│   ├── run_sft.py
│   └── run_rl.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── numerical/
│   ├── reproducibility/
│   └── smoke/
│
├── docs/
│   ├── architecture.md
│   ├── data_contract.md
│   ├── training.md
│   ├── fp8.md
│   ├── evaluation.md
│   └── release.md
│
├── artifacts/
│   └── README.md
│
└── RUN_SPEC.yaml
```

### Repository rule

Raw datasets and large checkpoints do **not** belong in the Git repository. The repository stores code, configs, manifests, hashes, reports, and instructions required to retrieve or reconstruct external artifacts.

---

# 4. Stage Map

| Stage | Name | Main result | Must pass before |
|---:|---|---|---|
| 0 | Project contract | Requirements and acceptance tests | Any implementation |
| 1 | Public repo scaffold | Installable/testable repo | Model work |
| 2 | Tiny Transformer | End-to-end trainable model | Tokenizer/data integration |
| 3 | Tokenizer | Frozen tokenizer artifact | Data freeze |
| 4 | Data pipeline | Deterministic manifests/shards | Large training |
| 5 | BF16 trainer | Stable single/multi-GPU baseline | FP8 |
| 6 | FP8 trainer | Stable FP8 path | Large-batch scaling |
| 7 | Large-batch system | Stable global token batch | Scaling pilots |
| 8 | Scaling pilots | Evidence for 300M recipe | RUN SPEC |
| 9 | RUN SPEC freeze | Immutable 300M configuration | Main run |
| 10 | 300M pretraining | Base model + checkpoints | SFT |
| 11 | SFT pipeline | Verified instruction-tuned model | RL |
| 12 | Execution RL | RL checkpoint + reward evidence | Final eval |
| 13 | Evaluation | Public/private benchmark report | Release |
| 14 | Release | Reproducible release package | Project completion |

---

# 5. Stage 0 — Project Contract

## Goal

Convert the research intent into machine-checkable requirements.

## Coding-agent tasks

- Create `README.md` with project purpose and non-goals.
- Create `RUN_SPEC.yaml` schema, even before values are final.
- Define configuration hierarchy.
- Define deterministic seed policy.
- Define versioning policy.
- Define acceptance tests for every major stage.
- Create issue/task IDs corresponding to stages in this document.

## Deliverables

| Artifact | Description |
|---|---|
| `README.md` | Project overview and quickstart |
| `RUN_SPEC.yaml` | Structured run contract |
| `docs/architecture.md` | Model specification |
| `docs/training.md` | Training contract |
| `docs/evaluation.md` | Evaluation contract |
| `docs/release.md` | Release requirements |

## Gate G0

Pass when a new contributor can understand:
- what the model is,
- how it will be trained,
- what is frozen when,
- how success is measured,
- and how to reproduce a result.

---

# 6. Stage 1 — Public GitHub Scaffolding

## Goal

A clean public repository that can install, lint, test, and execute a smoke workflow.

## Tasks

1. Create Python package metadata.
2. Add formatting/lint/type-check tooling.
3. Add CI.
4. Add test discovery.
5. Add config loading.
6. Add structured logging.
7. Add run directory creation.
8. Add run metadata capture:
   - git commit
   - config hash
   - host information
   - Python version
   - library versions
   - random seeds
9. Add `make test`, `make lint`, and `make smoke`.

## Acceptance

```text
git clone <repo>
pip install -e .
make test
make lint
make smoke
```

must succeed on a clean environment.

## Gate G1

**PASS:** repository installation + tests + smoke run all succeed.

---

# 7. Stage 2 — Tiny Transformer

## Goal

Implement the complete model mechanics before optimizing for scale.

## Initial debug model

| Parameter | Tiny value |
|---|---:|
| Layers | 4 |
| Hidden | 256 |
| Heads | 4 |
| Head dim | 64 |
| FFN hidden | 512–768 |
| Context | 256 |
| Vocabulary | Small debug vocabulary |

Use the same architecture concepts as the final model:
- causal attention
- RoPE
- RMSNorm
- SwiGLU
- tied embeddings
- no linear biases

## Tests

### Unit tests

- tensor shapes
- causal mask
- RoPE
- RMSNorm
- SwiGLU
- tied embedding equality
- parameter count
- checkpoint serialization

### Numerical tests

- forward pass finite
- backward pass finite
- gradients nonzero where expected
- loss decreases on a tiny deterministic corpus

### Integration test

Train for a small number of steps and confirm:
- loss decreases,
- checkpoint saves,
- checkpoint reloads,
- resumed run follows expected trajectory.

## Gate G2

**PASS:** tiny model learns and resumes from checkpoint.

---

# 8. Stage 3 — Tokenizer

## Goal

Create and freeze the production tokenizer before large-scale training.

## Candidate study

Evaluate multiple tokenizer candidates on:

| Metric | Why |
|---|---|
| Tokens / Python line | Efficiency |
| Tokens / function | Code compression |
| Identifier fragmentation | Code semantics |
| Operator fragmentation | Syntax efficiency |
| Import fragmentation | Common code patterns |
| Numeric literal handling | Math/code |
| Unicode behavior | Robustness |
| Round-trip exactness | Correctness |
| Compression ratio | Efficiency |

## Required tokenizer properties

- 32,768-token vocabulary
- byte fallback
- BOS/EOS
- explicit special-token policy
- deterministic normalization
- exact encode/decode round trip

## Freeze artifacts

```text
tokenizer/
├── tokenizer.model
├── tokenizer.json
├── special_tokens.json
├── tokenizer_metrics.json
└── SHA256SUMS
```

## Gate G3

**PASS:** tokenizer metrics and round-trip tests meet predefined thresholds and artifact hashes are recorded.

---

# 9. Stage 4 — Data Engineering

## Goal

Create a deterministic, auditable data pipeline.

## Raw data contract

Every source needs:

| Field | Required |
|---|---|
| Source ID | Yes |
| URL/origin | Yes |
| License | Yes |
| Provenance | Yes |
| Collection date | Yes |
| Language/domain | Yes |
| Quality score | Yes |
| Contamination risk | Yes |
| Raw hash | Yes |
| Normalized hash | Yes |
| Token count | Yes |

## Pipeline

```text
raw sources
    ↓
inventory
    ↓
license/provenance filter
    ↓
normalization
    ↓
exact deduplication
    ↓
near-deduplication
    ↓
quality filtering
    ↓
benchmark contamination scan
    ↓
tokenization
    ↓
packing/sharding
    ↓
immutable manifest
```

## Target pretraining mixture

| Domain | Tokens | Share |
|---|---:|---:|
| Python + educational/examples | 10B | 76.9% |
| Mathematics | 2B | 15.4% |
| General STEM | 1B | 7.7% |
| **Total** | **13B** | **100%** |

These values are the target final mixture, not a requirement to begin large-scale data processing on day one.

## Early dataset milestones

Before 13B:

1. 10M-token clean sample.
2. 100M-token clean sample.
3. 1B-token production-style sample.
4. Only then construct the full 13B-token manifest.

## Gate G4

**PASS:** manifest is deterministic, deduplication is complete, provenance is recorded, and contamination scans produce an auditable report.

---

# 10. Stage 5 — BF16 Training Baseline

## Goal

Create the reference implementation against which FP8 and performance optimizations are judged.

## Baseline training contract

| Setting | Starting policy |
|---|---|
| Optimizer | AdamW |
| Learning rate | Pilot 2e-4 to 4e-4 |
| Weight decay | ~0.1 starting point |
| Schedule | Warmup + cosine decay |
| Gradient clipping | Global norm ~1.0 |
| Precision | BF16 compute + FP32 master states where applicable |
| Dropout | 0.0 |
| Checkpointing | Frequent diagnostic milestones |
| Batch | Token-based global batch |
| Randomness | Fully recorded |

The exact values must be selected by pilot, not blindly assumed.

## Metrics

Record every training interval:

- train loss
- validation loss
- gradient norm
- learning rate
- tokens/sec
- tokens/GPU/sec
- throughput efficiency
- memory use
- NaN/Inf counts
- optimizer-step time
- checkpoint time
- data-loader time

## Gate G5

**PASS:** stable training with finite loss/gradients, reproducible checkpoints, and improving validation metrics.

---

# 11. Stage 6 — FP8 Training

## Goal

Introduce FP8 as a controlled optimization feature rather than making it a hidden source of instability.

## Implementation strategy

### Step 1 — BF16 reference

Establish the BF16 result:

```text
model config
    +
data seed
    +
training seed
    ↓
reference loss trajectory
```

### Step 2 — FP8 isolated experiments

Compare:

| Experiment | Purpose |
|---|---|
| BF16 forward/backward | Reference |
| FP8 selected matmuls/linear layers | First FP8 path |
| FP8 + accumulation safeguards | Stability |
| FP8 + distributed training | Real-world path |

The implementation should use the FP8 facilities supported by the target accelerator/software stack; do not build an unsupported custom FP8 arithmetic system solely for appearance.

## FP8 acceptance criteria

- no unexplained NaNs/Infs
- no catastrophic loss divergence
- gradient norms remain stable
- convergence is reasonably comparable to BF16
- checkpoint/restart works
- throughput or memory benefit is measurable

## Gate G6

**PASS:** FP8 becomes a documented, reproducible training mode with quantified tradeoffs against BF16.

---

# 12. Stage 7 — Large-Batch Training

## Goal

Make large-batch training a first-class capability.

The source plan already specifies gradient accumulation to achieve a stable large global token batch within hardware limits. fileciteturn1file0L172-L192

## Define batch size in tokens

For every run record:

```text
micro_batch_sequences
×
sequence_length
×
gradient_accumulation_steps
×
data_parallel_world_size
=
global_tokens_per_optimizer_step
```

## Example progression

Do not hard-code one “magic” batch size before measurement.

| Level | Purpose |
|---|---|
| Small | Debugging |
| Medium | Baseline performance |
| Large | Stable production training |
| Max-safe | Memory/throughput boundary |

## Experiments

Measure:

- tokens/sec
- scaling efficiency
- memory
- communication overhead
- gradient accumulation overhead
- optimizer-step time
- loss behavior
- convergence vs batch size

## Gate G7

**PASS:** large global token batches are stable and the chosen batch is justified by measured throughput and optimization behavior.

---

# 13. Stage 8 — Distributed Training

## Goal

Make one training command scale across multiple accelerators without changing the model semantics.

## Requirements

- distributed initialization
- synchronized gradients
- deterministic sampler state
- checkpointable distributed state
- rank-safe logging
- rank-safe checkpoint writing
- resume after interruption
- correct global-step accounting

## Tests

1. 1-device run.
2. 2-device run.
3. N-device run.
4. interrupt and resume.
5. compare normalized loss trajectories.

## Gate G8

**PASS:** distributed and resumed training behave within documented numerical tolerance.

---

# 14. Stage 9 — Scaling Pilots

## Goal

Use smaller models to validate the training recipe before committing full compute.

## Recommended progression

| Model | Purpose |
|---:|---|
| ~10M | software correctness |
| ~50M | optimizer/data behavior |
| ~100M | scaling behavior |
| ~300M | final architecture |

Keep the following constant whenever possible:

- tokenizer
- data processing
- optimizer implementation
- scheduler implementation
- logging
- checkpoint semantics
- evaluation methodology

## Main questions

1. Does loss scale predictably?
2. Does FP8 remain stable as model size grows?
3. Does large-batch efficiency improve enough to justify the configuration?
4. Does memory scale as expected?
5. Does throughput match compute planning?
6. Are there architecture-specific numerical problems?

## Gate G9

**PASS:** sufficient evidence exists to select and freeze the final 300M run configuration.

---

# 15. Stage 10 — Freeze the RUN SPEC

## Goal

Stop configuration drift before the expensive run.

The source plan explicitly requires freezing architecture, tokenizer, data manifest, sampling weights, optimizer, precision, context length, and training schedule into an exact RUN SPEC. fileciteturn1file0L100-L103

## RUN SPEC must contain

```yaml
project:
model:
tokenizer:
data:
optimization:
precision:
distributed:
batch:
checkpointing:
evaluation:
reproducibility:
artifacts:
```

## Fingerprint

Compute a cryptographic fingerprint over:

- RUN_SPEC
- tokenizer hashes
- data manifest hashes
- code commit
- evaluation config

## Gate G10

**PASS:** exact main-run configuration can be uniquely identified from the artifact set.

---

# 16. Stage 11 — 300M Pretraining

## Goal

Train the frozen 296.9M-parameter architecture for approximately 13B tokens.

The source plan estimates this as roughly 43.8 training tokens per parameter. fileciteturn1file0L138-L141

## Final architecture

| Component | Value |
|---|---:|
| Layers | 24 |
| Hidden | 960 |
| Heads | 15 |
| Head dimension | 64 |
| SwiGLU hidden | 2,560 |
| Context | 4,096 |
| Vocab | 32,768 |
| Parameters | ~296.9M |

## Training behavior to monitor

A “perfect” curve is not the objective. The operational definition is:

- no unexplained persistent loss spikes,
- no NaNs/Infs,
- stable gradient norms,
- decreasing validation loss,
- reasonable diminishing returns,
- no abrupt domain regression,
- reproducible checkpoints.

These criteria match the source plan’s operational definition of stable optimization. fileciteturn1file0L193-L202

## Gate G11

**PASS:** full pretraining completes with auditable checkpoints and no unresolved critical incidents.

---

# 17. Stage 12 — SFT Data Generation and Verification

## Goal

Build a verified 600K-example supervised corpus.

## Target composition

| Segment | Examples |
|---|---:|
| Python interview/coding | 300K |
| MBPP-style mathematics/programming | 200K |
| General STEM + open-source SFT | 100K |
| **Total** | **600K** |

The important constraint is verified examples, not padding the corpus to hit a token number. fileciteturn1file0L203-L224

## Example pipeline

```text
seed tasks
    ↓
task generation/evolution
    ↓
candidate solution
    ↓
test generation
    ↓
sandbox execution
    ↓
deterministic tests
    ↓
adversarial tests
    ↓
deduplication
    ↓
contamination scan
    ↓
quality scoring
    ↓
balanced selection
    ↓
600K frozen manifest
```

## Gate G12

**PASS:** all 600K examples satisfy provenance, correctness, deduplication, and contamination requirements.

---

# 18. Stage 13 — Supervised Fine-Tuning

## Goal

Improve instruction-following and executable coding performance while preserving useful base-model capabilities.

## Evaluation before/after SFT

Track:

- coding holdout
- math holdout
- STEM holdout
- generic language sanity checks
- execution pass rate
- loss/perplexity diagnostics
- regression vs base checkpoint

## Gate G13

**PASS:** SFT improves target behavior without unacceptable regression on protected holdouts.

---

# 19. Stage 14 — RL Problem Construction

## Goal

Construct 10K executable coding problems that are independent from the final benchmark items.

## Each problem must contain

- prompt
- visible/public tests where appropriate
- hidden tests
- adversarial tests
- resource limits
- deterministic environment
- provenance
- contamination status

## Sandbox requirements

- no network
- filesystem isolation
- CPU/memory/time limits
- deterministic dependencies
- controlled process execution

---

# 20. Stage 15 — Execution-Based RL

## Goal

Optimize for executable correctness rather than stylistic preferences.

## Rollout budget

| Samples/problem | Total rollouts for 10K problems |
|---:|---:|
| 10 | 100K |
| 32 | 320K |
| 50 | 500K |

The source plan recommends a staged rollout budget and identifies 32 as a practical default. fileciteturn1file0L281-L301

## Reward hierarchy

1. Parse/import/compile success.
2. Public tests.
3. Hidden tests.
4. Adversarial tests.
5. Resource compliance.
6. Optional code-quality signals.

Correctness must dominate style.

## Algorithm

Implement the project so the RL backend can support:

- GRPO-style group-relative optimization
- rejection sampling / best-of-N followed by supervised updates

## Gate G14

**PASS:** reward improves while private/held-out execution performance does not collapse.

---

# 21. Stage 16 — Final Evaluation

## Evaluation must be frozen before SFT/RL.

| Evaluation | Requirement |
|---|---|
| HumanEval | ≥99% target |
| MBPP | ≥99% target |
| HumanEval+ | Secondary robustness check |
| Private coding holdout | Mandatory |
| Math holdout | Mandatory |
| STEM holdout | Mandatory |

For the stated 99% target, report both percentage and raw correct counts.

The source plan notes that 99% corresponds to at least 163/164 on HumanEval and at least 495/500 on a 500-item MBPP split. fileciteturn1file0L314-L343

## Evaluation invariants

- fixed benchmark version
- fixed execution environment
- fixed decoding parameters
- fixed prompt format
- no benchmark contamination
- raw outputs retained where permitted
- results reproducible from the evaluation artifact

## Gate G15

**PASS:** evaluation report is complete and all integrity checks pass.

---

# 22. Stage 17 — Release

## Release artifacts

The source plan recommends the following artifacts; this execution plan keeps them as release requirements. fileciteturn1file0L466-L478

```text
RUN_SPEC.yaml
tokenizer/
data_manifest.jsonl
dedup_manifest.jsonl
contamination_report.json
pretrain_metrics/
sft_manifest.jsonl
rl_tasks.jsonl
eval_report.json
checkpoint_manifest.json
release_card.md
```

## Release card must include

- architecture
- training compute summary
- token budget
- data summary
- data licensing/provenance policy
- tokenizer
- optimization
- FP8 usage
- batch-size definition
- evaluation results
- contamination statement
- limitations
- known failures
- reproducibility instructions

## Gate G16

**PASS:** another engineer can reconstruct the lineage of the released checkpoint.

---

# 23. Engineering Gates Summary

| Gate | Pass condition | Stop if failed? |
|---|---|---|
| G0 | Project contract complete | Yes |
| G1 | Repo install/test/smoke works | Yes |
| G2 | Tiny model learns | Yes |
| G3 | Tokenizer frozen + tested | Yes |
| G4 | Data manifest auditable | Yes |
| G5 | BF16 stable | Yes |
| G6 | FP8 stable/comparable | Yes |
| G7 | Large-token batch stable | Yes |
| G8 | Distributed + resume works | Yes |
| G9 | Scaling evidence supports 300M | Yes |
| G10 | RUN SPEC frozen | Yes |
| G11 | 13B pretraining complete | Yes |
| G12 | 600K SFT dataset verified | Yes |
| G13 | SFT improves desired behavior | Yes |
| G14 | RL reward improves safely | Yes |
| G15 | Evaluation integrity passes | Yes |
| G16 | Release reproducible | Yes |

---

# 24. What the Coding Agent Should Never Do

The coding agent must not:

- silently change the model architecture during the main run;
- silently replace the tokenizer;
- silently alter dataset mixture weights;
- silently remove contaminated items after training begins;
- delete failed run artifacts;
- overwrite checkpoints without lineage;
- report benchmark percentages without raw counts;
- treat benchmark contamination as a routine preprocessing issue;
- optimize for a perfectly smooth loss curve;
- introduce FP8 before a trustworthy BF16 reference exists;
- report “training completed” when the final checkpoint cannot be restored;
- claim the 99% target was achieved without the specified evaluation protocol.

---

# 25. Coding-Agent Operating Model

The coding agent should work in this loop:

```text
READ REQUIREMENTS
      ↓
INSPECT CURRENT REPO
      ↓
IMPLEMENT ONE STAGE
      ↓
ADD/UPDATE TESTS
      ↓
RUN TESTS
      ↓
RUN SMOKE/INTEGRATION TEST
      ↓
RECORD ARTIFACTS/METRICS
      ↓
VERIFY STAGE GATE
      ↓
COMMIT
      ↓
MOVE TO NEXT STAGE
```

For every stage, the agent must produce:

1. code,
2. tests,
3. documentation,
4. a reproducible command,
5. artifacts/logs,
6. a clear pass/fail statement.

---

# 26. Master Goal Prompt for the Coding Agent

## Copy/paste prompt

You are the primary engineering agent for this repository.

Your objective is to take this repository from an empty/public GitHub scaffold to a fully reproducible implementation of the 300M dense Transformer project described in `PROJECT_PLAN.md`.

You are not being asked to jump directly to the final 300M training run. You must build the project incrementally and prove each layer before moving upward in scale.

### Final objective

Produce a reproducible ~296.9M-parameter decoder-only Transformer for Python, mathematics, and STEM with:

- 24 layers
- hidden size 960
- 15 attention heads
- 64 dimensions/head
- SwiGLU hidden size 2,560
- pre-norm RMSNorm
- RoPE
- 4,096-token context
- 32,768-token tokenizer with byte fallback
- tied input/output embeddings
- no linear biases
- 0.0 Transformer dropout

Train through:

1. ~13B pretraining tokens,
2. 600K verified SFT examples,
3. 10K execution-RL problems with 10–50 rollouts/problem,

then evaluate with the frozen protocol against:

- HumanEval ≥99% pass@1 target,
- MBPP ≥99% pass@1 target,
- HumanEval+,
- private coding holdout,
- math holdout,
- STEM holdout.

Do not claim the target is achieved unless the exact evaluation requirements pass.

### Engineering sequence

Implement the project in this exact order:

1. public repository scaffold;
2. configuration and run metadata;
3. tiny Transformer;
4. unit/integration tests;
5. tokenizer pipeline;
6. data inventory/manifests;
7. deduplication and contamination scanning;
8. BF16 training;
9. checkpoint/restart;
10. distributed training;
11. FP8 training;
12. large global token-batch training;
13. throughput/memory benchmarks;
14. scaling pilots;
15. frozen RUN SPEC;
16. 300M pretraining;
17. SFT data generation/verification;
18. SFT;
19. executable RL sandbox;
20. RL;
21. frozen evaluation;
22. release artifacts.

### Development rules

- Never skip a stage because a later stage is more interesting.
- Never optimize performance before correctness is tested.
- Prefer simple, explicit implementations before abstraction-heavy designs.
- Add tests with every feature.
- Keep public repository code free of hidden machine-specific assumptions.
- Put hardware-specific optimizations behind clean interfaces.
- Keep BF16 as the reference implementation for numerical comparisons.
- Treat FP8 as an optimization mode, not the only training mode.
- Define batch size in global tokens per optimizer step.
- Record all seeds, distributed states, configs, git commits, and artifact hashes.
- Make checkpoint resume a first-class feature.
- Fail loudly on NaN/Inf, corrupted data, invalid manifests, or mismatched checkpoint metadata.
- Do not delete failed experiment artifacts.
- Do not silently alter frozen specifications.

### Stage completion protocol

At the end of every stage:

1. run all relevant tests;
2. run a minimal reproducible example;
3. save relevant logs/metrics;
4. update documentation;
5. state exactly which gate passed or failed;
6. create/update a commit-ready change set.

Do not move to the next stage if the current gate fails.

### Required infrastructure

Build reusable components for:

- configuration loading and validation;
- structured logging;
- deterministic seeding;
- checkpoint metadata;
- checkpoint save/load/resume;
- tokenizer versioning;
- dataset manifests;
- exact/near deduplication;
- contamination scanning;
- token packing;
- distributed execution;
- precision abstraction;
- FP8 backend;
- metric logging;
- experiment directories;
- benchmark execution;
- release reporting.

### Training requirements

The trainer must expose:

- micro-batch size,
- sequence length,
- gradient accumulation,
- world size,
- global tokens per optimizer step,
- optimizer,
- learning rate,
- weight decay,
- scheduler,
- clipping,
- precision mode,
- checkpoint interval,
- evaluation interval,
- seed.

The trainer must report:

- loss,
- validation loss,
- gradient norm,
- learning rate,
- optimizer-step time,
- data-loading time,
- tokens/sec,
- tokens/GPU/sec,
- memory usage,
- checkpoint time,
- NaN/Inf status.

### FP8 requirements

Implement FP8 behind a dedicated precision interface.

Do not assume every environment supports FP8.

The system must:

- detect capability,
- fail clearly when unavailable,
- retain BF16 fallback,
- compare FP8 to BF16 on identical seeds/data/configuration where practical,
- document the selected FP8 implementation,
- record the FP8 mode in run metadata.

### Large-batch requirements

Represent batch size explicitly as:

`micro_batch × sequence_length × accumulation_steps × data_parallel_world_size`

The final run configuration must record the resulting global tokens per optimizer step.

Benchmark increasing global token batch sizes and choose the production batch from measured:

- throughput,
- memory,
- communication overhead,
- optimization stability,
- validation behavior.

### Data integrity requirements

Every source requires:

- provenance,
- license information,
- source identifier,
- hash,
- normalized hash,
- quality score,
- contamination risk,
- token count.

Never put benchmark evaluation prompts/reference solutions into training or SFT generation.

Maintain:

- `data_manifest.jsonl`
- `dedup_manifest.jsonl`
- `contamination_report.json`

### Evaluation requirements

Evaluation configuration must be frozen before SFT/RL.

Record:

- benchmark versions,
- prompt formatting,
- decoding parameters,
- execution environment,
- raw correct counts,
- percentages,
- artifact hashes.

Use private holdouts to detect overfitting.

### Final release requirements

Before declaring the project complete, verify that the release contains:

- model configuration,
- tokenizer,
- data manifest summary,
- contamination report,
- training metrics,
- checkpoint lineage,
- SFT manifest summary,
- RL task summary,
- evaluation report,
- release card,
- reproducibility instructions.

### First action

Start with Stage 0 and Stage 1 only.

Create:

- `README.md`
- `pyproject.toml`
- repository package structure
- configs
- tests
- CI
- `RUN_SPEC.yaml` schema
- initial documentation

Then run the test suite and smoke test.

Do not implement the 300M model yet.

The goal is to establish a clean engineering foundation that can be expanded stage-by-stage without breaking reproducibility.

---

# 27. Suggested Agent Commands

Once Stage 1 exists:

```bash
make test
make lint
make smoke
```

Tiny-model development:

```bash
python scripts/train.py --config configs/tiny.yaml
```

Tokenizer:

```bash
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python -m model_project.tokenizer.audit
```

Data manifest:

```bash
python scripts/build_manifest.py --config configs/data.yaml
python scripts/run_contamination_scan.py --config configs/contamination.yaml
```

Training benchmark:

```bash
python scripts/benchmark_throughput.py --config configs/tiny.yaml
python scripts/benchmark_memory.py --config configs/tiny.yaml
```

Pilot:

```bash
python scripts/run_pilot.py --config configs/pilot.yaml
```

Pretraining:

```bash
python scripts/train.py --config configs/pretrain.yaml
```

SFT:

```bash
python scripts/run_sft.py --config configs/sft.yaml
```

RL:

```bash
python scripts/run_rl.py --config configs/rl.yaml
```

Evaluation:

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

These commands are illustrative interfaces. The coding agent should implement stable CLI contracts and document them before relying on them operationally.

---

# 28. Final Project Roadmap

```text
EMPTY GITHUB REPO
      │
      ▼
[1] Scaffold
      │
      ▼
[2] Tiny Transformer
      │
      ▼
[3] Tokenizer
      │
      ▼
[4] Data Pipeline
      │
      ▼
[5] BF16 Training
      │
      ▼
[6] FP8 Training
      │
      ▼
[7] Large-Batch Distributed Training
      │
      ▼
[8] Scaling Pilots
      │
      ▼
[9] Freeze RUN SPEC
      │
      ▼
[10] 296.9M / 13B-token Pretraining
      │
      ▼
[11] 600K Verified SFT
      │
      ▼
[12] Execution RL / 10K Problems
      │
      ▼
[13] Frozen Evaluation
      │
      ├── HumanEval ≥99% ?
      ├── MBPP ≥99% ?
      ├── Private holdout passes?
      ├── Contamination clean?
      └── Reproducibility passes?
      │
      ▼
[14] RELEASE
```

---

# 29. Completion Statement

The project should be considered **goal achieved** only when:

```text
software reproducible
AND
model specification matched
AND
training reproducible
AND
data provenance/contamination controls passed
AND
SFT verification passed
AND
RL verification passed
AND
evaluation integrity passed
AND
release artifacts complete
AND
benchmark target actually achieved
```

If the benchmark target is missed, the project is still a successful **engineering execution** if all reproducibility and integrity requirements pass, but it must be reported as a target miss rather than a target-achieving model.

This distinction is critical because the original specification explicitly treats the 99% result as a hard acceptance target, not a guaranteed consequence of the 300M recipe.
