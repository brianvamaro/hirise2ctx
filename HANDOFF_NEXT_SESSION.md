# Handoff prompt — next session

**Last updated 2026-05-31 (night) — Stage 7.0 PASS + Stage 7a done.**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`
(the `--no-capture-output` + `-u` is non-obvious — see memory note
[[conda_run_no_capture_output]]; without it long probes look hung).

## Read in this order before starting

1. **Memory** [`project_state_2026-05-31-night.md`](../memory/...) (CURRENT) —
   Stage 7.0 verdict + Stage 7a coverage finding.
2. [`PLAN_Compositional.md`](PLAN_Compositional.md) — top-of-file revisions
   capture every implicit change made during 7.0/7a (PDS layout, swath width,
   trio substitute, SP1 fix, Lambertian-cancels-in-diff, coverage answer).
3. [`DECISIONS.md`](DECISIONS.md) 2026-05-31 entries (verdict table +
   coverage audit) for full numbers.
4. [`notebooks/14_compositional_feasibility.ipynb`](notebooks/14_compositional_feasibility.ipynb)
   — the executed feasibility notebook with figures + tables + verdict.

## Where we are

**Stage 7.0 = PASS (a)** — composition signal detected, dust-controlled. The
strongest evidence is `ESP_055253_2245` (an *anti-signal* image where the
rock-abundance model fails at AUC 0.42) showing a real dust-independent IR/BG
signal (partial r = 0.16, p = 0.037 controlling for `dust_index`). Two of the
three trio images show qualitatively different colour signals (042964 redder,
055253 bluer than surroundings) — supports the "different sources / transport
histories" narrative for the eventual thesis chapter.

**Stage 7a done** — `scripts/run_stage7a_audit.py` HEAD-probed all 39 v2 ObsIds:
**37 / 39 (94.9 %)** have a PDS `COLOR.JP2`, total 9.1 GB. `scripts/run_stage7a_fetch.py`
downloaded all 37 with exponential-backoff retries on transient PDS connection
resets and built `cache_v2/hirise_color/lbl_metadata.parquet` — a unified table
of per-image incidence/emission/scaling/swath fields ready for any Stage 7b-7e
work. The two images without colour are `ESP_055690_2200` and `ESP_055978_2270`
(latter was the original 7.0 trio member, already substituted).

**Stage 7b was eliminated, not done.** On 2026-05-31 night Brian made the
architectural call to *stay in source CRS* — no per-image colour raster reprojection
cache. Stage 7c absorbs what 7b would have done by reprojecting each *tile bounds*
CTX→source-CRS at read time (the Stage 7.0 Test B pattern). The effective ladder
is now `7.0 ✅ → 7a ✅ → ~~7b~~ skipped → 7c → 7d → 7e`. Captured in
[`PLAN_Compositional.md`](PLAN_Compositional.md) §3 table (7b struck through) +
top-of-file revisions item 7, and a full
[`DECISIONS.md`](DECISIONS.md) 2026-05-31 night entry "Stage 7b skipped (folded into 7c)".

**Path A (banking wins from Stage 6) is still on the docket**, untouched this
session.

## Goal of this session (Brian to decide at start)

Three natural next steps after 7.0 + 7a, in order of decreasing "tight focus on
the compositional thread":

- **B1. Stage 7c — per-tile colour features for the full cohort.** Brian
  picked "stay in source CRS" for the 7b approach
  ([PLAN_Compositional.md §3, decided 2026-05-31](PLAN_Compositional.md#3-stage-architecture)) —
  no per-image colour raster cache. Extract per-tile mean IR/RED/BG + ratios +
  `dust_index` for every S=64 tile in colour-covered images by reusing the
  Stage 7.0 Test B pattern (tile-bounds reprojection CTX→source-CRS, windowed
  COLOR.JP2 read). Write `dataset_v2/features_colour.parquet` joinable by
  `(obs_id, scale_idx, ti, tj)`. ~3-5 hr.
- **B2. Stage 7d — pooled cross-image boulder-rich vs boulder-poor test.**
  Build on B1's per-tile feature table: per-image standardise each colour
  feature, pool across all colour-covered images, two-sample (Mann-Whitney U +
  Cohen's d) on the standardised values, partial-correlation dust discriminator.
  Headline figure: distribution of effect sizes across images, and the pooled
  result. ~1 day. Could combine with B1 in a single session.
- **B3. Stage 7e — formal dust analysis.** Atwood-Stone & McEwen 2013-style
  literature-validated dust index (a refinement of the current
  `RED/BG` proxy), and explicit shadow masking via the Stage 4b
  `shadow_fraction` machinery — addresses two of the 7.0 caveats (over-broad
  dust proxy, contamination of polygon-interior pixels by adjacent shadow). Best
  done after B1+B2 reveal whether the rough proxy + minimal masking already
  produces robust pooled results. ~1 day if isolated.

Alternative: **Path A bank** — still available; ~2-3 hr. P1+P2 full-v2 LOIO
sweeps + P3+P4 doc reframes. **AskUserQuestion before each expensive sweep.**

## Critical gotchas (carry forward)

- **`conda run` swallows subprocess stdout** unless invoked with
  `--no-capture-output`. Combine with `python -u` + `print=functools.partial(print, flush=True)`
  in every long-running probe. Memory [[conda_run_no_capture_output]] has the recipe.
  Discovered 2026-05-31 during Stage 7.0 — cost ~30 min before being diagnosed.
- **HiRISE PDS SP1 bug also poisons `COLOR.JP2`** (not just `RED.JP2`). Use
  `src.colour.corrected_source_crs(obs_id, cache_dir)` to load the Stage 1
  sidecar's corrected CRS as a CRS override. Without this, polygon overlap
  with the colour swath is reported as 0 %.
- **Lambertian correction cancels in within-image diffs and band ratios**, so
  Test A and the partial-correlation discriminator are Lambertian-invariant.
  Cross-image pooling (Stage 7d) DOES need the correction — apply via
  `src.colour.lambertian_correct(arr, incidence_deg)`.
- **Inference-time scope (carry-over)**: model features must be derivable from
  CTX alone. Colour features are *analysis-only* (HiRISE-side) per
  `PLAN_Compositional.md §10` — they cannot be fed to the rock-abundance model
  at inference time over CTX-only regions.
- **`models/*`, `dataset_v2*/*`, and `cache_v2/hirise_color/*.JP2` are
  gitignored**; tracked artefacts persist in `scripts/probes/_stage7_*`,
  `cache_v2/stage7/*.parquet`, `notebooks/14_*.ipynb`,
  `DECISIONS.md` 2026-05-31 entries.
- **AskUserQuestion before**: expensive sweeps (Brian-gated), git commits,
  destructive operations on cached artefacts.
- **230 pytest pass baseline (Stage 6c era).** No new tests added in 7.0/7a;
  run `pytest tests/ -q` before any promotion. Adding tests for `src/colour.py`
  is a defensible Stage 7c todo.

## Stage 7 artefacts (reference)

- **Audit / fetch**: `scripts/run_stage7a_audit.py`, `scripts/run_stage7a_fetch.py`,
  `cache_v2/hirise_color/coverage.parquet`, `cache_v2/hirise_color/lbl_metadata.parquet`,
  37 × `{ObsId}_COLOR.{JP2,LBL}` files (~9 GB, gitignored).
- **Feasibility probe**: `scripts/probes/_fetch_color.py` (legacy of the 7.0 trio),
  `scripts/probes/_stage7_feasibility.py`,
  `cache_v2/stage7/test_{a,b}_per_polygon|tile.parquet`,
  `cache_v2/stage7/{test_a,test_b,dust}_summary.parquet`.
- **Module**: `src/colour.py` — LBL parse, SP1-corrected CRS loader, Lambertian,
  region helpers.
- **Notebook**: `notebooks/14_compositional_feasibility.ipynb` (executed) +
  `notebooks/_build_14.py`.
- **Figures**: `reports/figures/stage7_test_{a,b}_*.png`.

## Reporting protocol (same as previous handoff)

1. **`PROMOTION_QUEUE.md`**: nothing to move; Stage 7 lives in its own track.
2. **`DECISIONS.md`**: one entry per Stage 7 stage with numbers.
3. **Memory**: `project_state_2026-XX-XX.md` daily; mark previous superseded.
4. **`HANDOFF_NEXT_SESSION.md`**: rewrite priority order based on what landed.
5. **AskUserQuestion before `git commit`** of any of the above.
