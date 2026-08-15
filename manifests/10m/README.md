# 10M milestone status

**Not frozen.** `FROZEN_10M.json` is absent.

A real file-based run (not an 80-problem probe) produced:

| Domain | Target | Actual |
|---|---:|---:|
| Python | 7,690,000 | 7,689,911 |
| Math | 1,540,000 | 1,539,907 |
| STEM | 770,000 | 0 |
| **Total** | **10,000,000** | **9,229,818** |

Python and math hit the proportional caps. STEM is 0 because FineWeb-Edu parquet has
**no subject/category field**; exclusive bucket allocation would invent labels.

Retained Python tokens after mixture cap are all from CodeContests+ (`py3` correct).
DeepMind / TACO / APPS were ingested from published parquet/JSONL and then removed
by global exact/near-dedup and quality (they overlapped the contest lineage).

Contamination: 368 hits against local HumanEval + MBPP + stage3-eval-v1; all
**removed**; `n_unresolved = 0`. Refs live in gitignored `data/contamination_refs/`.

Do not treat this directory as a frozen 10M training corpus.
