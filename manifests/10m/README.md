# 10M milestone status

**Not frozen.** High-quality data after the authorized Python reselection is
still below 10M tokens, and STEM exclusive FineWeb buckets remain BLOCKED.

Probe run (80 CodeContests+ problems + 80 OpenMathInstruct-2 rows):

| Domain | Tokens after pipeline |
|---|---:|
| Python (CCP py3 correct, cap 8, English) | 457,196 |
| Math (OMI-2) | 56,020 |
| STEM | 0 |
| **Total** | **513,216** |

Scaling CCP further can grow Python, but STEM still has no authorized fetchable
source, and this pass will not invent one. A 10M freeze is therefore withheld.

See `docs/python_dataset_selection.md` for KEEP/REJECT evidence.
