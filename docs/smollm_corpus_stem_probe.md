# SmolLM-Corpus STEM probe (Stage 4)

**STOP.** HuggingFaceTB/smollm-corpus cannot provide the required
authoritative STEM allocation for the missing 770,182 tokens.

Inspected 2026-08-15. `configs/data_thresholds.yaml` and the frozen tokenizer
were not modified. Python/Math STAGE4_SMOKE was not rebuilt. `FROZEN_10M.json`
was not written. Stage 5 was not started.

## A. Official identity

| Field | Value |
|---|---|
| Dataset | `HuggingFaceTB/smollm-corpus` |
| URL | https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus |
| Hub revision | `3ba9d605774198c5868892d7a8deda78031a781f` |
| License on card | ODC-By (`license:odc-by`) |
| Gated | no |
| Citation | Ben Allal et al., SmolLM-Corpus, July 2024 |

This is a **different** dataset from `HuggingFaceFW/fineweb-edu`. It **bundles**
three subsets. The collection license is ODC-By; `python-edu` file bodies are
not in the repo and follow The Stack v2 / Software Heritage terms.

## B. Official configurations and files

| Config | Files | Official role |
|---|---|---|
| `cosmopedia-v2` | 104 parquet `cosmopedia-v2/train-*` | Mixtral-8x7B synthetic textbooks/stories |
| `fineweb-edu-dedup` | 234 parquet `fineweb-edu-dedup/train-*` | Deduplicated FineWeb-Edu educational web |
| `python-edu` | 2 parquet `python-edu/train-*` | Stack-v2 Python IDs scored ≥4 |

Access: `huggingface_hub` parquet download (no `load_dataset` script required).

## C. Schemas (direct parquet)

**fineweb-edu-dedup:** `text`, `id`, `metadata.{dump,url,date,file_path,language,language_score,token_count,score,int_score}`

Same FineWeb-Edu fields. `score` = educational quality 0–5, not subject.

**python-edu:** `blob_id`, `repo_name`, `path`, `length_bytes`, `score`, `int_score`

**No `text` column.** Contents require AWS S3 `softwareheritage` (`content/{blob_id}`).

**cosmopedia-v2:** `prompt`, `text`, `token_length`, `audience`, `format`, `seed_data`

`seed_data` is the **seed source**, not a STEM subject. Official Cosmopedia
docs (v0.1, reused by v2): stanford = scraped Stanford **course catalog**
(all departments); openstax = OpenStax outlines; auto_math_text = AutoMathText
seeds “to improve science knowledge” but “covers more than just math”;
fineweb/web samples ≈ 75%+ general web; stories/wikihow = not STEM.

## D. Small inspection (8,000 cosmopedia rows, first 8 row groups of `train-00000-of-00104.parquet`)

`seed_data`: fineweb 6288, ultrachat 584, openhermes2.5 410, auto_math_text 262,
wikihow 224, stanford 173, wikihow_original 33, openstax 19, khanacademy 7.

`format`: textbook 4987, stories/wikihow/dialogue/etc. `scientific_article` = 6
(style, not subject).

### Representative rows (truncated)

| seed_data | Official meaning | Actual prompt excerpt | STEM? |
|---|---|---|---|
| fineweb | web seed | Ruangguru online school / COVID edtech | no (general) |
| fineweb | web seed | ceramic vs glass induction cooktops | no |
| stanford | Stanford course unit | “Contemporary Asian Filmmakers” / Tsai Ming-liang | **no (humanities)** |
| openstax | OpenStax unit | College Physics AP, Kinematics / Time | physics in this row; OpenStax also has history/sociology |
| auto_math_text | AutoMathText seed | Python `factoradic` permutation tutorial | not a STEM subject label |

**fineweb-edu-dedup row 0:** AACAP “When to Seek Help for Your Child” (parenting/psych).
Educational score 3.375. Not a STEM label.

**python-edu row 0:** `sudajzp/jzp-s-python` `/FBNQ_py/Fib_circle.py`, score 3.84.
Python file ID only.

## E–H. Filtering / tokens / 10M

Not run. No authorized STEM allocation rule exists, so the quality/dedup/
contamination pipeline was not applied to invent 770k STEM tokens.

Quality thresholds **can** apply to FineWeb-Edu-dedup (`score` present) and
to Cosmopedia text length. That does not make either subset STEM.

`python-edu` cannot be quality-scanned until S3 text is fetched; it is Python,
and the Python 10M cap is already filled.

Cross-source overlap: FineWeb-Edu-dedup is FineWeb-derived (same family as the
already-blocked FineWeb-Edu STEM attempt). Cosmopedia `auto_math_text` overlaps
the math domain conceptually. Neither is a reason to admit unlabeled STEM.

## I. Why STEM allocation is not authorized

Required missing bucket is the Stage 4 **STEM** allocation (~770k at 10M).
Earlier exclusive five-bucket labels are **not** present here either.

Acceptable evidence would be official subject labels, official STEM
partitions, or a deterministic rule on explicit source metadata that the
project contract already allows.

Observed:

- No Physical Sciences / CS / Life / Eng / ML configs.
- FineWeb-Edu-dedup has no subject field.
- Cosmopedia `seed_data` is seed **origin**, and Stanford/OpenStax/FineWeb
  seeds are not STEM-only (filmmakers vs physics vs cooktops).
- Using `format == scientific_article` would be style matching, not subject.
- Using an LLM or URL keywords would invent labels.
- Dumping general FineWeb-Edu or Cosmopedia into `domain=stem` would silently
  treat general educational/synthetic web as STEM.

Therefore **770,182 additional clean STEM tokens are not achievable** from
this source under the existing contract.

## J–O. Milestone

| Item | Status |
|---|---|
| Final 10M mixture | Unchanged: 7,689,911 / 1,539,907 / 0 = **9,229,818** |
| `FROZEN_10M.json` | **not created** |
| G4 | **FAIL** |
| Stage 5 | not started |

Authorization still required (not decided here): waive exclusive STEM for the
10M engineering milestone, **or** authorize a source that actually publishes
STEM subject labels.
