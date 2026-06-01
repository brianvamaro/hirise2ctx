# Handoff prompt — next session

**Last updated 2026-06-01 — Stage 7c DONE (9 860 rows / 36 images).
Stage 7b deliberately skipped + documented.**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`
(the `--no-capture-output` + `-u` is non-obvious — see memory note
[[conda_run_no_capture_output]]; without it long probes look hung).

## Read in this order before starting

1. **Memory** [`project_state_2026-06-01.md`](../memory/...) (CURRENT) —
   Stage 7c cohort numbers + the one excluded image + Stage 7d prerequisites.
2. [`PLAN_Compositional.md`](PLAN_Compositional.md) — §3 table now shows
   `7.0 ✅ 7a ✅ ~~7b~~ skipped 7c ✅ 7d 7e`. Top-of-file revisions list captures
   the implicit changes (SP1 in COLOR.JP2, Lambertian cancellation,
   coverage answer, 7b skip rationale).
3. [`DECISIONS.md`](DECISIONS.md) 2026-06-01 entry — cohort numbers, the
   DN→I/F scaling bug discovery + fix, and the slow-outlier note.
4. [`notebooks/14_compositional_feasibility.ipynb`](notebooks/14_compositional_feasibility.ipynb)
   — Stage 7.0 feasibility verdict; sets up the dual-narrative (composition
   + dust-age) framing for Stage 7d.

## Where we are

**Stage 7c = DONE** — `dataset_v2/features_colour.parquet` written:
**9 860 rows across 36 of 37 colour-eligible images**, 145 min wall-clock.
Joinable on `(obs_id, scale_idx, ti, tj)` to `dataset_v2/labels/{ObsId}.parquet`.
Columns: `n_color_pixels`, `IR_iof`, `RED_iof`, `BG_iof`, `IR_over_RED`,
`IR_over_BG`, `dust_index_RED_over_BG`, `cos_incidence`.

Per-image retention is **24 - 31 %** of S=64 tiles (dictated by ~2 - 6 km
colour swath vs 6 km HiRISE footprint). Cohort I/F medians (IR=0.17, RED=0.16,
BG=0.08) are in expected Mars dusty-equatorial regolith range. cos(i) varies
0.30 - 0.76 across the cohort — Lambertian correction is doing meaningful work
for cross-image pooling.

**One image excluded**: `ESP_046803_2325` — has Stage 1 sidecar + COLOR.JP2 but
no `dataset_v2/labels/ESP_046803_2325.parquet` (Stage 4 was never run for this
ObsId, predates this session). Not a Stage 7d blocker.

**Stage 7b = SKIPPED**: architectural decision committed in 286475a. No
per-image colour raster cache; Stage 7c reprojects tile bounds CTX→source-CRS
on read. Full reasoning in [`DECISIONS.md`](DECISIONS.md) 2026-05-31 night
entry + [`PLAN_Compositional.md`](PLAN_Compositional.md) §3 table.

**Path A (banking wins from Stage 6) is still on the docket**, untouched
across this session.

## Goal of this session (Brian to decide at start)

The natural next steps, in order of "tight focus on the compositional thread":

- **B2. Stage 7d — pooled cross-image boulder-rich vs boulder-poor test.**
  Input ready: `dataset_v2/features_colour.parquet`. Per-image standardise each
  colour feature (subtract per-image mean, divide by per-image std), pool
  across all 36 images, run Mann-Whitney U + Cohen's d on boulder-rich
  (`fractional_area >= 1e-2` per the P4 binary partition) vs boulder-poor at
  S=64. Then the partial-correlation dust discriminator (control for
  `dust_index_RED_over_BG`, re-test IR/BG and IR/RED). Headline figure:
  distribution of per-image effect sizes for each feature + the pooled result.
  Output: `dataset_v2/stage7d_pooled.parquet` + a notebook
  (`notebooks/15_stage7d_pooled.ipynb`) with the headline figure + verdict.
  ~1 day. **This is the most natural next thing.**
- **B3. Stage 7e — formal dust analysis.** Atwood-Stone & McEwen 2013-style
  literature-validated dust index (a refinement of the current `RED/BG` proxy),
  plus explicit shadow masking via the Stage 4b `shadow_fraction` machinery.
  Addresses two 7.0 caveats (over-broad dust proxy, polygon-interior pixels
  contaminated by adjacent shadow). Best done after B2 reveals whether the
  current proxy + minimal masking already produces robust pooled results.
  ~1 day if isolated.
- **Path A bank** — still available; ~2-3 hr. P1+P2 full-v2 LOIO sweeps +
  P3+P4 doc reframes. **AskUserQuestion before each expensive sweep.**

## Critical gotchas (carry forward)

- **`conda run` swallows subprocess stdout** unless invoked with
  `--no-capture-output`. Combine with `python -u` + `print=functools.partial(print, flush=True)`
  in every long-running probe. Memory [[conda_run_no_capture_output]] has the recipe.
- **`conda run python -c "..."` doesn't accept multi-line strings** on Windows
  — write the script to a tempfile and run it instead, or use single-line
  semicolons. (Hit during the Stage 7c trio verification.)
- **HiRISE PDS SP1 bug also poisons `COLOR.JP2`** (not just `RED.JP2`). Use
  `src.colour.corrected_source_crs(obs_id, cache_dir)` to load the Stage 1
  sidecar's corrected CRS as a CRS override.
- **COLOR.JP2 stores raw uint16 DN, not I/F** — convert per-image via
  `I/F = DN * scaling_factor + offset` from the LBL. `scaling_factor` varies
  ~5× across the cohort (0.00004 - 0.00021), so without this conversion
  cross-image pooling is meaningless. Bug caught + fixed in
  `scripts/run_stage7c_features.py` 2026-06-01.
- **Lambertian correction cancels in within-image diffs and band ratios** —
  Test A and the partial-correlation discriminator are Lambertian-invariant.
  Cross-image pooling (Stage 7d) DOES need the correction; already applied in
  the Stage 7c output.
- **Inference-time scope (carry-over)**: model features must be derivable from
  CTX alone. Colour features are *analysis-only* (HiRISE-side) per
  `PLAN_Compositional.md §10` — they cannot be fed to the rock-abundance model
  at inference time over CTX-only regions.
- **AskUserQuestion before**: expensive sweeps (Brian-gated), git commits,
  destructive operations on cached artefacts.
- **233 pytest pass baseline (Stage 7c era)** — `test_colour.py` adds 8 tests
  over the Stage 6c baseline. Run `pytest tests/ -q` before any promotion.

## Stage 7 artefacts (reference)

- **Audit / fetch**: `scripts/run_stage7a_audit.py`, `scripts/run_stage7a_fetch.py`,
  `cache_v2/hirise_color/coverage.parquet`, `cache_v2/hirise_color/lbl_metadata.parquet`,
  37 × `{ObsId}_COLOR.{JP2,LBL}` files (~9 GB, gitignored).
- **Feasibility probe**: `scripts/probes/_stage7_feasibility.py`,
  `cache_v2/stage7/test_{a,b}_per_polygon|tile.parquet`,
  `cache_v2/stage7/{test_a,test_b,dust}_summary.parquet`.
- **Feature extraction (Stage 7c)**: `scripts/run_stage7c_features.py`,
  `scripts/probes/_verify_stage7c_trio.py`, `scripts/probes/_summarise_stage7c.py`,
  `dataset_v2/features_colour.parquet` (gitignored, 9 860 rows).
- **Module**: `src/colour.py` — LBL parse, SP1-corrected CRS loader, Lambertian,
  region helpers, `ctx_bounds_to_source_bbox`, `windowed_colour_read`.
- **Tests**: `tests/test_colour.py` — 8 unit tests for the colour primitives.
- **Notebook**: `notebooks/14_compositional_feasibility.ipynb` (executed) +
  `notebooks/_build_14.py`.
- **Figures**: `reports/figures/stage7_test_{a,b}_*.png`.

## Reporting protocol (same as previous handoff)

1. **`PROMOTION_QUEUE.md`**: nothing to move; Stage 7 lives in its own track.
2. **`DECISIONS.md`**: one entry per Stage 7 stage with numbers.
3. **Memory**: `project_state_2026-XX-XX.md` daily; mark previous superseded.
4. **`HANDOFF_NEXT_SESSION.md`**: rewrite priority order based on what landed.
5. **AskUserQuestion before `git commit`** of any of the above.
