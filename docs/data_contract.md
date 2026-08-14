# Data contract (Stage 4)

Every source, including BLOCKED sources, is recorded with:

| Field | Required |
|---|---|
| Source ID | Yes |
| URL/origin | Yes (null if unresolved) |
| License | Yes (null if unverifiable) |
| Provenance | Yes |
| Collection date | Yes |
| Language/domain | Yes |
| Quality score | Yes (null if the source has none) |
| Contamination risk | Yes |
| Raw hash | Yes (null at source-level inventory) |
| Normalized hash | Yes (null at source-level inventory) |
| Token count | Yes (0 if BLOCKED) |
| Status | `admitted` or `blocked` |

Unknown or unverifiable licenses are **BLOCKED**, not quietly omitted.

Frozen milestone artifacts live in `manifests/10m/` (committed). Raw downloads and
shards live under `data/` (gitignored). HumanEval/MBPP references, if fetched,
live in `data/contamination_refs/` (gitignored) and never enter the manifest.
