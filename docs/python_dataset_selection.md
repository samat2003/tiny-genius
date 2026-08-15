# Python dataset selection (Stage 4, 2026-08-14)

Evidence from Hub cards plus streamed rows. Thresholds were not retuned.

## Ranked decision table

| Dataset | Role | Human/synth | Quality | Algo value | Diversity | Correctness | Dup risk | Contam | License | Useful tokens (order) | Mix | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ByteDance-Seed/Code-Contests-Plus | backbone | human contest | high (py3 only) | high | high | verified-correct field | high within problem | low if tests excluded | CC-BY-4.0 | large | primary | **KEEP** (root parquet parts) |
| deepmind/code_contests | supplemental | human | py3 only | high | overlaps CCP | solutions vs incorrect_solutions | **global dedup** | train split only | CC-BY-4.0 | medium | unique remainder | **KEEP** (authorized alpaca replacement; parquet) |
| BAAI/TACO | supplemental | human+scraped | mixed | high | CF/GFG/etc | solutions field, 22% empty | overlaps CCP/APPS | medium | apache-2.0 | medium | unique remainder | **KEEP** via ALL/*.parquet |
| codeparrot/apps | supplemental | human | uneven tests | medium | interview/CF | not guaranteed | overlaps CF | benchmark-derived | MIT | small | unique remainder | **KEEP** train.jsonl only |
| BertilBraun/TinyPython | — | synthetic Qwen/Phi | template loops | low | very low | untrusted | extreme | low | unverified | 0 | 0 | **REJECT** |
| wuyetao/spp | — | CodeGeeX synth | **systematic errors** | claimed high | low | doctests lie | high | HumanEval-like | unverified | 0 | 0 | **REJECT** |
| jina-ai/textbook | — | ChatGPT 3.5 | synthetic | function-level | designed diverse | unverified | medium | **HumanEval format** | unverified | 0 | 0 | **REJECT** |
| AdapterOcean/pythonbook-standardized_unified | — | textbook | tutorial I/O | low | 2.5k rows | n/a | chapter chunks | low | unverified | ~0 for algo | 0 | **REJECT** |

No additional source was added: `open-r1/codeforces` is ODC-By problem statements **without** human solutions (editorial often null) and overlaps CCP problem text.

## Inspection notes (bad examples not hidden)

**SPP** (`nearest_integer`): docstring says `nearest_integer(5.6) == 5` and `4.5 == 5`; body is `(math.floor(n)+0.5)` — neither nearest-int nor matching tests.

**SPP** (`is_Power_Of_Four(3)` expected True; `16` expected False) — inverted and wrong.

**TinyPython**: seed 0/1 repeated as “find first index below threshold” with near-identical Qwen bodies.

**pythonbook**: Automate the Boring Stuff PDF/I/O chapters (`os.listdir`, PyPDF2).

**Jina**: README states HumanEval-format ChatGPT exercises; `jinaai/code_exercises_40k` is not on the Hub.

**CCP**: `language` is explicit (`py3`/`py2`/`cpp`/`java`). Heuristic `import` matches Java. Filter is **`language == py3` only**. Generators/validators/checkers are C++ testlib and are dropped.

**DeepMind**: 40-problem sample is mostly C++ then Python3 then Java; Python2 still present. Kept as the authorized alpaca replacement; only language==3; globally deduped against CCP.

## Filtering (KEEP source)

- Python3 correct submissions only
- Cap 8 solutions / problem (existing threshold)
- Identity normalize, exact + global near-dedup (existing)
- `ast.parse`, English-only (no non-Latin comments), reject OS/I/O libraries
- Contamination vs stage3-eval-v1 and local HumanEval/MBPP refs

## Mixture

Python = 100% CodeContests+ after filters. Math/STEM sources unchanged (OMI-2 admitted; FineWeb buckets still unresolved).
