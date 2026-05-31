# Handoff prompt — next session

**Last updated 2026-05-30 (late session) after Stage 6a dev sweep and Brian's
"gains are small" assessment.**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run -n geospatial python …` (never the
env's `python.exe` directly — see memory note [[conda_location]]).

## Read in this order before starting

1. **Memory** `project_state_2026-05-30-late.md` (CURRENT) — Stage 6a outcome.
2. **[`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md)** — Problem catalog, status legend, and
   the "Stage 6a — Dev result (2026-05-30)" section + Problem 4 status updated to
   `◐ DEV-PARTIAL`.
3. **[`docs/modeling_results.md`](docs/modeling_results.md) §12** — full Stage 6a writeup
   (table, mechanism, per-image heterogeneity).
4. **[`scripts/probes/_diag_stage6a_followup_compare.md`](scripts/probes/_diag_stage6a_followup_compare.md)**
   — 6-variant comparison table (the actual numbers).

## Where we are

**Stage 6a (spatial-context neighbour features) is DEV-PARTIAL but the gains are small.**
Brian's 2026-05-30 review: "the difference is pretty small for this improvement".
Concretely:

- **5×5 stencil @ S=32**: only variant that clears the strict criteria (Δ ρ +0.053,
  Δ PR-AUC +0.053) — but at finer scale, so absolute numbers (ρ +0.276, PR-AUC 0.546)
  don't beat the S=64 baseline (ρ +0.283, PR-AUC 0.640).
- **At S=64** (canonical scale): only operational top-K metrics move (default 3×3 →
  precision@top-5 % +0.044, PR-AUC +0.010, Spearman flat).
- **All-grid best**: best Spearman = 5×5 @ S=64 (+0.310, Δ +0.027). Best PR-AUC =
  max-only @ S=64 (0.652, Δ +0.012).

Compared with the P1+P2 wins (P2 alone was +0.114 PR-AUC = +22 % on dev), Stage 6a's
gains are an order of magnitude smaller. **This is consistent with Problem 6's
texture-floor reading**: with the same per-tile CTX texture features, neighbour
aggregation gives diminishing returns at S=64 because the tile already integrates
~320 × 320 m of context.

**Implications for forward planning** (open question — discuss at session start):

- Stage 6b (CTX-source illumination, priority #4 in the order) tests a *different*
  failure-mode mechanism (per-image anti-signal). It might still buy a real lift on
  the 3–10 % of anti-signal images even if the average dev lift is small. But the
  same texture-floor argument suggests the per-tile signal ceiling is real — Stage 6b
  helps where current features *mislead*, not where they're informative.
- Stages 6d (multi-scale), 6e (mosaic-seam), and 6f (ZI-Tweedie loss redesign) all
  carry similar "could be small" risk based on this evidence.
- Item 7 in the recommendation order (THEMIS / HiRISE-decimated as a surrogate) is the
  *off-CTX* unlock — different signal, not subject to the 5 m/px texture floor.

## Goal of this session (Brian to decide at start)

Before doing implementation work, **AskUserQuestion to confirm direction**. Three
plausible paths, with the trade-offs:

**Path A — Continue Stage 6 docket (Stage 6b next).**
- Pros: incremental, the implementation pattern from 6a carries over (parallel
  features dir + repackage + sweep); 6b also tests a sharp H3 hypothesis (anti-signal).
- Cons: 1–2 day cost; if 6a's small-gain pattern repeats, the next 1–2 weeks of
  Stage 6 (6b, 6d, 6e) all return single-digit-percentile lifts.

**Path B — Bank the validated wins (P1+P2 full-v2) + simple Stage 6 items only.**
- Pros: P1+P2 full-v2 promotion is ~1–2 hr and bank-the-wins (+22 % PR-AUC dev gain).
  P3+P4 doc reframe is another ~1 hr. Get the deliverable's headline numbers anchored.
  Defer Stage 6b/6d/6e until evidence justifies the cost.
- Cons: doesn't push the model further. The promoted model is still the
  texture-floor-limited ranker (Problem 2 only partly fixed by P1).

**Path C — Pivot to Stage 7 (compositional, HiRISE 3 bands).**
- Pros: Stage 7.0 feasibility test is 1–2 days on 2–3 images and *de-risks an entirely
  separate research thread* — the instructor's extra goal that the texture-floor
  doesn't bind. Plan is drafted at [`PLAN_Compositional.md`](PLAN_Compositional.md).
- Cons: leaves the modeling result at "dev-validated P1+P2, not promoted to full-v2".
  Committee may want headline modeling numbers before composition.

Recommend Path B as the default — it's cheap, banks real wins, and lets next-session
decide between 6b vs Stage 7 with the deliverable secured.

## Before doing anything

**Working tree status (start-of-session):** clean. Last commits:
- `6b428e3` Stage 6a: spatial-context neighbour features (dev-partial)
- `6e3b9f1` Per-image heterogeneity (notebook 13) + Stage 6 docket + 2026-05-30 review

No staging required.

## Path A detail — Stage 6b CTX-source illumination

Full spec in [`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md) "Stage 6b — CTX-source
illumination angles". Implementation pattern from Stage 6a applies directly:

1. **Download the PDS CTX CUMINDEX** (~200 MB) — Brian-gated; confirm before downloading.
   Lives at `https://pds-imaging.jpl.nasa.gov/data/mro/mars_reconnaissance_orbiter/ctx/CUMINDEX/`.
   Cache to `cache/pds_ctx_cumindex.tab`.
2. **Per-tile spatial join with the Murray Lab SeamMap** (already extracted during
   notebook 13 §3.2; see
   [`scripts/probes/_diag_per_image_breakdown.py`](scripts/probes/_diag_per_image_breakdown.py)).
   Mean 24 CTX sources per HiRISE footprint (range 4–46), so the right granularity is
   per-tile, not per-image.
3. **Build `src/ctx_source_illumination.py`** mirroring `src/spatial_features.py`'s
   pattern — a function that augments a Stage 4b feature parquet with 3 columns
   (`ctx_incidence_angle_mean`, `ctx_emission_angle_mean`, `ctx_phase_angle_mean`,
   area-weighted across CTX sources intersecting each tile).
4. **Scripts**: `scripts/run_stage6b.py` + `scripts/run_stage6b_repackage.py` mirroring
   the Stage 6a counterparts.
5. **Sweep**: mirror `scripts/probes/_sweep_stage6a.py`. Run baseline + Stage 6b at
   S=64 on dataset_v2_dev; report deltas; include the H3 mechanism check (across-image
   per-image AUC ↔ tile-mean CTX_IncidenceAngle correlation, ρ < −0.30 with p < 0.05).

**Acceptance (dev)**: PR-AUC +≥ 0.03 over P1+P2 baseline AND significant negative
AUC↔incidence correlation. If only one clears, document as informative and decide.

## Path B detail — P1+P2 full-v2 promotion + P3+P4 doc reframe

Per the original handoff (now superseded for Stage 6 work but still valid for these
items):

```powershell
$conda = "C:\Users\brian\anaconda3\Scripts\conda.exe"

# P1 (presence-head fix)
& $conda run -n geospatial python scripts/sweep.py `
    --variants lightgbm_two_stage_balanced `
    --dataset-dir dataset_v2 --scheme loio_nfold

# P2 (boulder_count target) — via the probe until --target-col lands on sweep.py
& $conda run -n geospatial python scripts/probes/_sweep_target_reformulation.py `
    --targets boulder_count --scales 3 `
    --dataset-dir dataset_v2
```

**Acceptance**: per [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) P1 / P2 sections —
S=64 ρ ≥ 0.18 AND presence AUC ≥ 0.58 (P1) / PR-AUC delta ≥ +0.05 (P2).

**P3 / P4 are doc-only**: update [`docs/modeling_results.md`](docs/modeling_results.md)
§9 to put PR-AUC + lift@top-K as headline metrics with ROC-AUC demoted; change the
default primary binary target in
[`src/modeling/binary_target.py`](src/modeling/binary_target.py) from `bc_ge_1` to
`fa_gt_1e-2`.

**Cost**: P1+P2 sweeps ~1–2 hr; P3+P4 ~1 hr. AskUserQuestion before each
expensive sweep.

## Path C detail — Stage 7.0 feasibility test

[`PLAN_Compositional.md`](PLAN_Compositional.md) drafted 2026-05-30. The 7.0
**feasibility test** gates the full 5–7 day pipeline:

- 1–2 days
- 2–3 images
- Uses *actual BoulderNet labels* (not model predictions) to isolate the methodology
  from model error
- Tests whether HiRISE 3-band spectra (BLUE-GREEN / RED / NEAR-IR per
  [Delamere et al. 2010](https://doi.org/10.1016/j.icarus.2009.03.012)) of
  boulder-rich areas differ from surroundings
- Central methodological challenge: dust mantle confound. Use `dust_index = RED/BG`
  as discriminator.
- HiRISE colour covers only ~20 % of each image's swath (central CCDs only) —
  bounds the spatial scope.

## Critical gotchas (carry forward from earlier handoff)

- **Inference-time scope**: model features must be derivable from CTX alone. HiRISE
  LBL angles are diagnostic-only. See PROMOTION_QUEUE.md "Inference-time scope".
- **`conda run python -c` rejects multiline strings** — write probes to files under
  `scripts/probes/_*.py`. Hit this 2026-05-30 with the fold-variance probe.
- **`cp1252` stdout encoding fails on Unicode** in some `conda run` paths — write
  probe outputs to `.md` files (pattern in
  [`scripts/probes/_diag_stage6a_followup_compare.py`](scripts/probes/_diag_stage6a_followup_compare.py)).
- **`models/*` and `dataset_v2*/*` are gitignored** — sweep artifacts and augmented
  feature parquets don't persist across machines. The readable tables in
  `scripts/probes/_diag_*.md` and the writeups in `docs/modeling_results.md` are
  the persistent record.
- **220 pytest pass baseline + 15 new tests** for Stage 6a in
  [`tests/test_spatial_features.py`](tests/test_spatial_features.py). Run
  `pytest tests/ -q` before any promotion.
- **AskUserQuestion before**: full-v2 sweeps (Brian-gated; expensive), `git commit`,
  destructive operations on cached artifacts, downloading new external data
  (CUMINDEX, etc).
- **Stage 6a augmented features and packaged dirs already exist** on disk at
  `dataset_v2_dev/features_nbr{,_s5,_max}/` and
  `dataset_v2_dev/packaged/within_image_4fold_nbr{,_s5,_max}/`. No re-augmentation
  needed if a follow-up calls for re-running the Stage 6a dev sweep.

## What we know vs what we suspect (honest status)

Per PROMOTION_QUEUE.md's Problem catalog + Stage 6a result:

- ✓ **Problem 1 (target distribution noise)**: solved by P2 (`boulder_count`), dev +22 %
  PR-AUC. Reproducibly the biggest lever found.
- ◐ **Problem 2 (compression)**: P1 fixes presence-head source only; magnitude-head
  shrinkage remains. Ship as ranker, not calibrated regressor.
- ? **Problem 3 (per-image anti-signal)**: Stage 6b would test CTX-source illumination
  hypothesis. ESP_054000_2255 (notebook 13 §6) is the canonical anti-signal case.
  Stage 6a partly confirms: it helps spatially-coherent images, hurts the
  anti-signal image (ESP_064510_2260 in dev).
- ◐ **Problem 4 (no surrounding spatial context)**: Stage 6a DEV-PARTIAL — confirmed
  the mechanism exists but with small lift; S=64 baseline already near the
  spatial-integration ceiling.
- ✓ **Problem 5 (metric framing)**: P3+P4 doc reframes (queued).
- ✗ **Problem 6 (5 m/px CTX texture floor)**: unresolved. Stage 6a's small gains
  *strengthen* this reading — same texture features, no matter how aggregated,
  approach a per-tile ceiling. Outside-CTX unlocks (THEMIS, HiRISE-decimated) remain
  the only unbinding lever.

## How this session should report progress

Same as previous handoff:
1. **`PROMOTION_QUEUE.md`**: move passed items to "Promoted", failed items to
   "Tried, didn't work".
2. **`DECISIONS.md`**: one entry per promoted item with full-v2 numbers.
3. **`docs/modeling_results.md`**: update §9 headline if P3/P4 land, or add §13 for
   Stage 6b result.
4. **Memory**: `project_state_2026-05-XX.md` with day's outcomes; mark previous
   superseded.
5. **`HANDOFF_NEXT_SESSION.md`**: rewrite priority order based on what landed.
6. **AskUserQuestion before `git commit`** of any of the above.
