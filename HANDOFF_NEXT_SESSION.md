# Handoff prompt — next session

**Last updated 2026-06-02 — Stage 7d PASS. The §4 + §5 hypothesis tests are now
answered for the v2 cohort. Stage 7e (formal dust analysis) is the natural next
step; Path A bank is still on the docket.**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`
(the `--no-capture-output` + `-u` is non-obvious — see memory note
[[conda_run_no_capture_output]]; without it long probes look hung).

## Read in this order before starting

1. **Memory** [`project_state_2026-06-02.md`](../memory/...) (CURRENT) —
   Stage 7d verdict + numbers + Stage 7e prerequisites.
2. [`PLAN_Compositional.md`](PLAN_Compositional.md) — §3 table now shows
   `7.0 ✅ 7a ✅ ~~7b~~ skipped 7c ✅ 7d ✅ 7e`. Top-of-file revisions list item 8
   captures the Stage 7d verdict.
3. [`DECISIONS.md`](DECISIONS.md) 2026-06-02 entry — method, numbers, three-condition
   verdict, and the band-ratio-vs-single-band interpretation.
4. [`notebooks/15_stage7d_pooled.ipynb`](notebooks/15_stage7d_pooled.ipynb) —
   headline figures + per-image effect-size distributions + verdict cells.

## Where we are

**Stage 7d = DONE (PASS)** — `dataset_v2/stage7d_pooled.parquet` (639 rows).

Headline (P4_area partition, pooled standardised, MW + Cohen's d on rich vs poor):

| feature | d | p |
|---|---|---|
| IR_iof | -0.372 | 1.7e-73 |
| RED_iof | -0.365 | 5.1e-69 |
| IR_over_RED | -0.331 | 1.7e-61 |
| BG_iof | -0.346 | 1.1e-59 |
| IR_over_BG | -0.279 | 9.9e-43 |
| dust_index | -0.252 | 9.3e-33 |

Dust-discriminator (partial-dust residualised on `dust_index_RED_over_BG` per
image, then re-test): 5/5 non-dust features survive at p ≤ 1e-15, |d| 0.07 – 0.16.
**Band ratios IR/BG and IR/RED shrink the LEAST under dust control** (42 %,
54 %) — they carry the strongest composition signal. Single bands shrink 67 –
80 % — mostly dust-loading. Per-image sign consistency 0.77 – 0.83.

**Continuous monotonicity** (Spearman vs `boulder_count`, per-image standardised):
all 6 rhos (−0.12 to −0.17) sign-match the binary effect; partial-dust Spearman
holds on 5/5 non-dust features.

**Take-away**: the dust narrative explains ~50 – 80 % of the raw rich-vs-poor
difference (likely: boulder-rich areas are younger / less dust-loaded), and a
real composition residual remains in the band-ratio features — boulder-rich
material has a different ferric/ferrous signature even after dust correction.
This is the bi-modal narrative the PLAN was set up to detect, and it lands.

**Stage 7c = DONE 2026-06-01**: `dataset_v2/features_colour.parquet` (9 860 rows
× 36 of 37 colour-eligible images). One image excluded:
`ESP_046803_2325` — has Stage 1 sidecar + COLOR.JP2 but no
`dataset_v2/labels/ESP_046803_2325.parquet` (Stage 4 was never run for this
ObsId, predates the Stage 7 work). Not a blocker; flagged as follow-up.

**Path A (banking wins from Stage 6) is still on the docket**, untouched
across the last two sessions.

## Goal of this session (Brian to decide at start)

The natural next steps, in order of "tight focus on the compositional thread":

- **C1. Stage 7e — formal dust analysis.** Two refinements over the Stage 7d
  partial-dust discriminator:
  1. Replace the crude `RED/BG` proxy with the literature-validated
     [Atwood-Stone & McEwen 2013](https://doi.org/10.1029/2013GL058355) dust
     index (uses absolute reflectance + band-shape information, not just a
     two-band ratio).
  2. Explicit shadow masking via the Stage 4b `shadow_fraction` machinery —
     currently the Stage 7c per-tile colour means include shadow pixels, which
     bias single-band I/F downward in boulder-rich tiles (likely a chunk of the
     67 – 80 % single-band dust-shrinkage actually being a shadow artefact).
  Then re-run the §4.2 + §5.2 + §4.3 tests on the refined data and compare to
  the Stage 7d baseline. Output: `dataset_v2/stage7e_pooled.parquet` + a
  notebook `notebooks/16_stage7e_dust_refined.ipynb`. ~1 day. **This is the
  most natural next thing — the Stage 7d result implicitly motivates it.**
- **C2. Per-image attribution table.** Stage 7d has all the per-image effect
  sizes. The PLAN §6 deliverable is a 36-row table assigning each ObsId a
  category in {locally sourced | transported | dust-age difference |
  inconclusive} based on (per-image binary d sign + magnitude, partial-dust d
  sign + magnitude, geographic context). Cheap once the per-image partial-dust
  rows are added (currently only pooled partial-dust is computed). ~3 – 4 hr.
- **C3. Writeup**: `docs/compositional.md` — a paper-Methods style summary of
  Stages 7.0 + 7a + 7c + 7d (and 7e if done). Plot dump + tables + the
  bi-modal narrative. Best done after 7e so the writeup is final.
- **Path A bank** — still available; ~2 – 3 hr. P1+P2 full-v2 LOIO sweeps +
  P3+P4 doc reframes. **AskUserQuestion before each expensive sweep.**

## Critical gotchas (carry forward)

- **`conda run` swallows subprocess stdout** unless invoked with
  `--no-capture-output`. Combine with `python -u` + `print=functools.partial(print, flush=True)`
  in every long-running probe. Memory [[conda_run_no_capture_output]] has the recipe.
- **`conda run python -c "..."` doesn't accept multi-line strings** on Windows
  — write the script to a tempfile and run it instead, or use single-line
  semicolons. Hit during Stage 7c trio verification AND during Stage 7d
  notebook-output inspection (2026-06-02) — fix used in
  `scripts/probes/_inspect_nb15_outputs.py`.
- **HiRISE PDS SP1 bug also poisons `COLOR.JP2`** (not just `RED.JP2`). Use
  `src.colour.corrected_source_crs(obs_id, cache_dir)` to load the Stage 1
  sidecar's corrected CRS as a CRS override.
- **COLOR.JP2 stores raw uint16 DN, not I/F** — convert per-image via
  `I/F = DN * scaling_factor + offset` from the LBL. `scaling_factor` varies
  ~5× across the cohort (0.00004 – 0.00021); without conversion cross-image
  pooling is meaningless. Fixed in `scripts/run_stage7c_features.py` on
  2026-06-01.
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
- **272 pytest pass baseline (Stage 7d era)** — `test_stage7d.py` adds 19
  tests over the Stage 7c baseline. Run `pytest tests/ -q` before any promotion.

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
- **Pooled test (Stage 7d)**: `src/stage7d_pooled.py`,
  `scripts/run_stage7d_pooled.py`, `scripts/probes/_inspect_nb15_outputs.py`,
  `dataset_v2/stage7d_pooled.parquet` (gitignored, 639 rows).
- **Module**: `src/colour.py` — LBL parse, SP1-corrected CRS loader, Lambertian,
  region helpers, `ctx_bounds_to_source_bbox`, `windowed_colour_read`.
- **Tests**: `tests/test_colour.py` (8) + `tests/test_stage7d.py` (19).
- **Notebooks**: `notebooks/14_compositional_feasibility.ipynb`,
  `notebooks/15_stage7d_pooled.ipynb` (both executed).
- **Figures**: `reports/figures/stage7_test_{a,b}_*.png`,
  `stage7d_pooled_effect_sizes.png`, `stage7d_per_image_effects.png`,
  `stage7d_dust_discriminator.png`, `stage7d_spearman_continuous.png`.

## Reporting protocol (same as previous handoff)

1. **`PROMOTION_QUEUE.md`**: nothing to move; Stage 7 lives in its own track.
2. **`DECISIONS.md`**: one entry per Stage 7 stage with numbers.
3. **Memory**: `project_state_2026-XX-XX.md` daily; mark previous superseded.
4. **`HANDOFF_NEXT_SESSION.md`**: rewrite priority order based on what landed.
5. **AskUserQuestion before `git commit`** of any of the above.
