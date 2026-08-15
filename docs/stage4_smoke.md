# STAGE4_SMOKE validation report

Engineering/training-path smoke only. **Not** the 10M milestone.
**Not** production pretraining. Stage 5 was not started.

## H. Gate status (read this first)

**G4 = FAIL**

Reason: The exact 10M milestone is not frozen because STEM=0 and the corpus
contains 9,229,818 tokens. STAGE4_SMOKE is an engineering validation corpus
only.

`FROZEN_10M.json` does not exist and was not created.

## A. Corpus identity

| Field | Value |
|---|---|
| Identity | `STAGE4_SMOKE` |
| Manifest | `manifests/stage4_smoke/STAGE4_SMOKE.json` |
| Corpus hash | `e671168c1298ed975549458360d03737a59bd36db86211ae7360af38adef772a` |
| Total tokens | 9,229,818 |
| Python | 7,689,911 |
| Math | 1,539,907 |
| STEM | 0 |
| 10M milestone | no |
| Gate G4 | FAIL |

Corpus hash is SHA-256 of the five preserved `manifests/10m` audit files
(using their existing SHA-256 digests). Those audit files were not rewritten.

## B. Environment

| Field | Value |
|---|---|
| GPU | NVIDIA A10 |
| VRAM | 22,588 MiB advertised |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Python | 3.11.15 |
| Host | ubuntu@129.80.243.31 |

## C. Training-path validation

Existing Tiny Transformer (4 / 256 / 4 / 64 / SwiGLU 640), vocab 32768 from
the frozen tokenizer. Not a 300M trainer.

| Run | n_ctx | batch | accum | dtype | steps | tokens | init loss | final loss | peak VRAM | tok/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| native tiny | 256 | 2 | 1 | bf16 | 8 | 4,096 | 10.3125 | 5.71875 | 207 MiB | 8,792 |
| packing probe | 4096 | 1 | 1 | bf16 | 2 | 8,192 | 10.1875 | 7.21875 | 1,585 MiB | 77,095 |

Dataset/shard loading: `data_manifest.jsonl` (15 source shards; 2 admitted
with tokens, summing to 9,229,818) plus packed smoke sequences through
`pack_documents` and `DataLoader`. Raw 9.23M token-id shards are not in git;
the interface, not a full pretrain epoch, is what this smoke exercises.

## D. Numerical integrity

- All recorded losses finite
- All parameter gradients finite
- Optimizer parameters finite after each step
- No NaN/Inf

## E. Checkpoint

| Artifact | SHA-256 |
|---|---|
| `/tmp/stage4_gpu_smoke.pt` | `151f87228cbc8cc8489b7c726c79324dbc4c8b386d340ba47cfe3aa89fa79dbe` |
| `/tmp/stage4_gpu_smoke_4096.pt` | `c8bc0a4adbfe426daa974a1c3adc75df236b688003752ff3b4f4bd299ccbd17b` |

Save succeeded. `load_checkpoint` + `restore_training_state` matched weights.

## F. Reproducibility

| Item | Value |
|---|---|
| Seed | 42 |
| Tokenizer fingerprint | `219156db6bbe8c573c0f1654ab9f622c0e8bd51519561ac30d2c13fbf3a01a6e` |
| Corpus hash | `e671168c1298ed975549458360d03737a59bd36db86211ae7360af38adef772a` |
| Seed-42 rerun (n_ctx=256, 8 steps) | **identical loss sequence** |

## G. Tests (local repo)

| Command | Result |
|---|---|
| `python -m pytest` | pass (61 tests) |
| `make test` | same as pytest |
| `make lint` / `ruff check src tests scripts` | pass |
| `make smoke` / `scripts/smoke.py` | pass, fingerprint match |

A10 also ran `tests/unit/test_stage4_smoke.py` (pass) and
`scripts/stage4_gpu_smoke.py` (ok).

## What was not done

- No `FROZEN_10M.json`
- No STEM fill
- No tokenizer or `data_thresholds.yaml` change
- No dataset add/remove/substitute
- No Stage 5 trainer
- 9,229,818 tokens were not called 10M
