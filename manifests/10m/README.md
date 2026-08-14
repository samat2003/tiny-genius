# 10M milestone status

**Not frozen.** `FROZEN_10M.json` is absent because the 10M token target was
not reached without substituting datasets.

- Actual retained tokens after the pipeline: **56,020** (math only).
- Python admitted sources TACO and APPS failed fetch (Hub dataset scripts no
  longer supported). They are BLOCKED with that reason — not replaced.
- CodeContests+, TinyPython, SPP_30k_verified, Jina Textbook, pythonbook,
  Code Contest Python3, and all FineWeb-Edu exclusive buckets are BLOCKED for
  unresolved identity or missing official splits.
- OpenMathInstruct-2 (cc-by-4.0) admitted 80 streamed rows for this inventory
  run. Scaling that source cannot fill the 7.69M Python or 0.77M STEM quotas.

These JSON/JSONL files are an auditable **inventory + probe**, not a frozen
10M training corpus. Reaching 10M requires the BLOCKED sources to become
uniquely identified, licensed, and fetchable — or an explicit authorization
to change the source list (not granted this pass).
