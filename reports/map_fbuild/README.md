# reports/map_fbuild — F-build composites. **NOT THE DELIVERABLE.**

> **PLAN_FBuild was HARD-ABORTED on 2026-07-30 (Brian).** Nothing in this directory is shipped.
> The project deliverable is the **A1 / mosaic-path map** in [`../map_region/`](../map_region/).

These rasters are retained as the **evidence base for the abort**, not as a product. Full reasoning:
DECISIONS 2026-07-30 (the Stage-C diagnosis) and 2026-07-30b (the abort), plus PLAN_FBuild §4.4.

## Why it was aborted, in one table

Per-observation `mean(predicted abundance) / mean(labelled fractional_area)`, with the mosaic sampled
at the identical 95,606 labelled tiles (21 obs):

| row | median ratio | max/min across obs | sd(log₁₀) |
|---|---|---|---|
| **mosaic (the deliverable)** | **0.89** | **5.1×** | **0.170** |
| `h1only` (F, no levelling) | 2.22 | 29.4× | 0.328 |
| `resid` | 1.92 | 32.5× | 0.371 |
| `pfree` | 1.35 | 189.6× | 0.532 |
| `full` | 15.54 | 81.3× | 0.412 |

F reduces **within-tile** striping (sd/mean of per-frame means 0.827 → 0.536; windowed partition η²
0.121 → 0.087) but degrades **between-place** level coherence, which is what a regional abundance map
is for. Levelling makes it worse in every variant.

## What the variants are

| variant | offsets applied | note |
|---|---|---|
| `h1only` | none (o=0) | the un-levelled control, PLAN §1 deliverable 5 |
| `full` | `offset_logit` (λ*=0, pre-declared) | **DEAD** — rails 51.8% of co-located tiles, \|o\|max 21.31 logits vs the model's own ±9.21 range |
| `resid` | `offset_residual_only` | solve, then subtract the lon/lat plane |
| `pfree` | `offset_logit_pfree` | plane constrained out *inside* the solve; was briefly the intended headline |

Shared H6 provenance layers (`_n_frames`, `_primary_frame`, `_incidence`, `_offset_source`) apply to
all variants.

## Reproducing

Everything here regenerates in ~10 min from retained inputs — the per-frame logits in
`../f_region_logits/` (906 npz; ~33 GPU-h on Sherlock to recreate, so **do not delete those**) plus
`../figures/fbuild_stagec_offsets.csv`:

```powershell
& $conda run --no-capture-output -n geospatial python -u scripts/f_region_staged.py --allow-partial
& $conda run --no-capture-output -n geospatial python -u scripts/f_region_gates.py
```

Stage C itself re-runs in ~2 min from the cached overlap graph in `../f_stagec/`.

**Plain-named files** (`{tile}_prob.tif`, `{tile}_prob_raw.tif`, `{tile}_abundance.tif`) are the
"headline" copies that `src.striping` and notebook 24 pick up from `--out-dir`. They were written
during the aborted run with `--headline pfree` and **must not be present** — if you see them, delete
them, or a shelved F map will be read as the deliverable.
