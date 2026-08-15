# FineWeb-Edu STEM allocation probe (Stage 4)

**Outcome B.** FineWeb-Edu cannot provide the required authoritative exclusive
STEM allocation from the approved source.

Inspected 2026-08-14. Thresholds, tokenizer, and the 9,229,818-token
Python/Math audit were not modified. No 770k STEM corpus was constructed.
`FROZEN_10M.json` was not written. Stage 5 was not started.

## Official identity

| Field | Value |
|---|---|
| Dataset | `HuggingFaceFW/fineweb-edu` |
| URL | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| Paper | https://arxiv.org/abs/2406.17557 |
| Hub revision (probe) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| License | ODC-By-1.0 (`license:odc-by` on the card) plus Common Crawl ToU |
| Gated | no |

This is the identity already listed in `configs/data_sources.yaml`.
`HuggingFaceFW/fineweb-edu-score-2` is a **different** dataset and was not used.

## Official configurations and files

114 Hub configs:

- `default` — all crawls (`data/*/*`)
- `sample-10BT`, `sample-100BT`, `sample-350BT` — random token samples
- `CC-MAIN-YYYY-WW` — one config per Common Crawl dump (2013–2025)

No config is named Physical Sciences, Computer Science, Life Sciences,
Engineering/Technology, ML/AI, Mathematics, or STEM.

Published files: 3,038 parquet shards under `data/CC-MAIN-*` and `sample/*`,
plus `.gitattributes` and `README.md`. There is no sidecar metadata file with
subject labels.

## Official schema (card + parquet)

Card `default` features:

`text`, `id`, `dump`, `url`, `date`, `file_path`, `language`,
`language_score`, `token_count`, `score`, `int_score`.

Direct parquet probe (`sample/10BT/000_00000.parquet`, revision above):

`text`, `id`, `dump`, `url`, `file_path`, `language`, `language_score`,
`token_count`, `score`, `int_score`.

The sample shard has no `date` column. Neither schema has
`subject`, `category`, `topic`, `domain`, or STEM-bucket labels.

`score` / `int_score` are the FineWeb-Edu **educational quality** classifier
(0–5; official keep threshold 3). They are not subject labels.

## Small deterministic probe

| Item | Value |
|---|---|
| File | `sample/10BT/000_00000.parquet` |
| File rows | 726,000 (726 row groups) |
| Rows inspected | 512 (row group 0, first 512) |
| Metadata fields | listed above |
| Category values | **none** — no category column |
| `language` | all `en` in the 512 |
| `dump` | all `CC-MAIN-2013-20` |
| `score` range | 2.515625–4.59375 |
| `int_score` | 3 (436), 4 (75), 5 (1) |
| Quality `score >= 2.0` (frozen threshold) | 512 pass / 0 reject |
| Eligible Physical Sciences | 0 (no label) |
| Eligible Computer Science | 0 |
| Eligible Life Sciences | 0 |
| Eligible Engineering/Technology | 0 |
| Eligible ML/AI | 0 |
| Cross-bucket overlap | 0 (nothing assigned) |
| Frozen-tokenizer token counts per bucket | not computed; no assignment |

Quality filtering is compatible with the existing schema (`score` is present).
The frozen threshold was **not** changed. Dedup and contamination were not
re-run on FineWeb because no STEM documents were allocated.

## Why Outcome B

Required buckets (exclusive, one or none):

Physical Sciences, Computer Science, Life Sciences, Engineering/Technology, ML/AI.

Acceptable evidence would be official labels, official category metadata,
documented official partitions, or a deterministic rule on **explicit existing
source metadata**. FineWeb-Edu provides none of these for those five subjects.

Treating educational-web pages as a STEM bucket, URL keyword matching, or
an LLM/heuristic classifier would invent labels. That is forbidden.

Therefore **770,182 additional clean exclusive STEM tokens are not achievable
from the approved source without new authorization.** Exact 10M remains
blocked. Gate G4 remains **FAIL**.

## Authorization required to proceed (not decided here)

Either:

1. Explicit waiver of the exclusive five-bucket STEM requirement for the
   10M engineering milestone, **or**
2. Explicit authorization to add or replace a STEM source that carries
   authoritative exclusive subject labels.

Do not substitute another Hugging Face dataset without that authorization.
