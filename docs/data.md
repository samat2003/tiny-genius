# Data engineering (Stage 4)

## Implemented now

- Full pipeline: inventory → license → normalize → exact dedup → near dedup
  (MinHash 5-gram, 128 perms, Jaccard ≥ 0.8) → quality → contamination →
  canonical tokenize (`Tokenizer.load_frozen()`, fingerprint check) →
  EOS-separated packing (`n_ctx=4096`) → immutable manifest.
- Predefined thresholds in `configs/data_thresholds.yaml` (not retuned).
- Fixed source registry in `configs/data_sources.yaml` (no substitutions).
- Synthetic fixture tests with planted duplicates, caps, and contamination.

## Not implemented / not authorized this pass

- 100M, 1B, and 13B manifests
- Stage 5 BF16 trainer
- Inventing Hugging Face IDs for unresolved source names

## Mixture accounting

Plan target: Python 10B / Math 2B / STEM 1B.

Math raw intake estimate (~2.63B) exceeds the 2B plan target. The gap is
closed only by documented down-selection (filters + 10M proportional cap),
never by silently adopting 2.63B as the target.

Python raw total is measured from admitted sources only; unresolved names
contribute 0 and stay BLOCKED.

## Re-run

```bash
python scripts/run_data_pipeline.py --milestone 10m
python scripts/verify_data_manifest.py
```

`FROZEN_10M.json` is written only when the 10M token target is actually met
and then refuses in-place overwrite. The 10M sample is **not frozen**. The
existing 9,229,818-token audit is labeled `STAGE4_SMOKE`
(`manifests/stage4_smoke/STAGE4_SMOKE.json`) for engineering-path validation
only. Python KEEP is CodeContests+ (`py3` only);
see `docs/python_dataset_selection.md`. STEM allocation is **Outcome B**
(`docs/fineweb_edu_stem_probe.md`): official FineWeb-Edu has no exclusive
subject labels. Gate G4 remains FAIL. Stage 5 is not started.
