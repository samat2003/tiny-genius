# 300M STAGE4_SMOKE learning experiment

Not Stage 5. Not 10M. G4 remains FAIL. STEM waived for this smoke only.

```bash
python scripts/stage4_300m_learning.py --seq-len 1024 --microbatch 1 --grad-accum 8
```

Reuses `artifacts/stage4_300m_learning/packed_sequences.pt` if present.
Add `--rematerialize` only to rebuild tokens from Hugging Face.

Inspect `artifacts/stage4_300m_learning/{metrics.jsonl,summary.json}`.

## Measured A10 run (kept)

296,925,120 params, BF16, 786 steps, 5,811,471 non-pad tokens.
Train 10.625 → min 1.546 (ppl 4.69). Holdout 2.906. Final step 2.680.
No NaN/Inf/OOM. Final ckpt SHA-256
`797e594f234eb33f0b1731734623134eec1639dfe398af613e39587b58411210`.
