# Handoff prompt — next session

**Last updated 2026-05-30 after the large-scale project review.**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run -n geospatial python …` (never the
env's `python.exe` directly — see memory note [[conda_location]]).

## Read in this order before starting

1. **[`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md)** — the live docket. The "Problem catalog
   & priority" section at the top frames everything; the Part A / Part B split tells you
   what kind of work each item is.
2. **[`notebooks/12_compression_diagnostic.ipynb`](notebooks/12_compression_diagnostic.ipynb)**
   — compression diagnosis (§2), `balanced` and `boulder_count` dev wins (§5, §9), and the
   H1–H5 framework (§7).
3. **[`notebooks/13_per_image_heterogeneity.ipynb`](notebooks/13_per_image_heterogeneity.ipynb)**
   — per-image bimodal AUC, three failure modes (§2.1), top-K confusion overlay (§5.2),
   anti-signal deep dive on ESP_054000_2255 (§6).
4. **[`docs/modeling_results.md`](docs/modeling_results.md) §11** — the Phase A2 writeup
   that mirrors notebooks 12+13.
5. **Memory** `project_state_2026-05-30.md` (CURRENT) for the previous-session synthesis.

## Goal of this session

Execute the **priority order** from `PROMOTION_QUEUE.md`'s "Recommendation order" table.
In order:

1. **P1 + P2 to full v2 (✓ dev-validated, bank-the-wins, ~1–2 hr)**
2. **P3 + P4 documentation reframe (✓ documentation-only, ~1 hr)**
3. **Stage 6a — spatial-context neighbour features (? untested, 1–2 days)**
4. **Stage 6b — CTX-source illumination angles (? H3 test, 1–2 days)**
5. **P5 binary classifier calibration fix (✓ predictable ECE drop, ~2 hr)**

✓ = dev-validated / predictable. ? = untested hypothesis; the failure itself is
informative. (See the table for what falsifies each ?.)

## Before doing anything

**Brian-gated**: stage and commit the 2026-05-30 review work first. The working tree has
unstaged changes across:

- `DECISIONS.md`, `PROMOTION_QUEUE.md` (the new framework + Stage 6 items + 2026-05-30 entry)
- `README.md`, `ROADMAP.md` (status updates to reflect modeling shipped + Stage 6)
- `PLAN_ModelImprovement.md` (marked historical at top)
- `CLAUDE.md` §10 (CRISM → HiRISE 3-band update per [Delamere 2010](https://doi.org/10.1016/j.icarus.2009.03.012))
- `docs/modeling_results.md` §5 + §11.5/11.6/11.7/11.8 (Stage 6 naming + boulder_area
  follow-up + per-image findings + 6d/6e/6f docket additions)
- `notebooks/12_compression_diagnostic.ipynb` + `_build_12.py` (CRISM update)
- `notebooks/13_per_image_heterogeneity.ipynb` + `_build_13.py` (already executed; CRISM
  update is text-only)
- `reports/figures/13_*.png` (4 figures from notebook 13)
- `scripts/probes/_diag_per_image_breakdown.py` + `_diag_topk_confusion_map.py` +
  `_diag_nb13_correlations.py` + `_diag_extract_nb13_results.py` + the corresponding `.md`
  result files
- `scripts/probes/_sweep_target_reformulation.py` (extended for `boulder_area` /
  `log_boulder_area`)

Show Brian the file list and a drafted commit message, then ask before `git commit`.

## Priority 1 detail — P1 + P2 full-v2 promotion

The cheap, validated, biggest win. Confirmed dev gains:
- **P1** (`lightgbm_two_stage_balanced`): +0.017 Spearman ρ, +0.018 presence AUC at S=64
  (dev within_image_4fold 20 folds).
- **P2** (`target_col=boulder_count`): **+0.114 PR-AUC (+22 %), +0.131 normalised lift
  (+27 %), +0.111 precision@top-5 % (+20 %)** at S=64 with Spearman / ROC-AUC unchanged.

**Run** (in this order):

```powershell
# P1 — already supported by sweep.py
& $conda run -n geospatial python scripts/sweep.py `
    --variants lightgbm_two_stage_balanced `
    --dataset-dir dataset_v2 --scheme loio_nfold

# P2 — currently only via the probe (needs --target-col flag added to scripts/sweep.py;
# eventually that flag should land but until it does, the probe is the right driver)
& $conda run -n geospatial python scripts/probes/_sweep_target_reformulation.py `
    --targets boulder_count --scales 3 `
    --dataset-dir dataset_v2
```

**Acceptance criteria** (from PROMOTION_QUEUE.md P1 / P2):
- P1: full-v2 Spearman ρ at S=64 ≥ 0.18 (baseline 0.169) AND presence AUC at S=64 ≥ 0.58
  (baseline 0.579). Either condition met → promote.
- P2: full-v2 PR-AUC at S=64 with `boulder_count` > full-v2 PR-AUC with `fractional_area`
  by ≥ +0.05. Spearman ρ should not regress.

**Then**: AskUserQuestion before adding entries to a "Promoted" section in
PROMOTION_QUEUE.md (Brian's call on whether to promote, then move the items out of the
docket).

## Priority 2 detail — P3 + P4 doc reframe

**P3**: update [`docs/modeling_results.md`](docs/modeling_results.md) §9 to put **PR-AUC +
lift@top-K** as the headline metrics on the v2 numbers, with ROC-AUC demoted to a
secondary diagnostic. Should be done AFTER P2 lands so the headline numbers are anchored
to the new target.

**P4**: in [`src/modeling/binary_target.py`](src/modeling/binary_target.py), change the
default primary binary target from `bc_ge_1` to `fa_gt_1e-2` (boulder-rich). Update
`scripts/sweep_binary.py`'s default-target arg if needed. No re-train required — the
existing `models/_sweep_binary/20260529T075754Z/` already has `fa_gt_1e-2` numbers; this
is a doc + default change.

## Priority 3 detail — Stage 6a spatial-context features

Full spec in [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) "Stage 6a — Spatial-context
neighbour features". Implementation summary:

1. Add a neighbour-aggregation pass in [`src/features.py`](src/features.py): for each
   ObsId × scale, lay tiles on the (ti, tj) grid and compute `nbr_mean / nbr_max /
   nbr_std` of each existing numeric feature over the 3 × 3 neighbourhood (8 neighbours
   + self). Use `scipy.ndimage.uniform_filter` / `maximum_filter` / custom std. NaN-pad
   at image edges.
2. Optional multi-scale variant: for tiles at S=64, also include features from the 4
   enclosing S=128 parent tiles (= Stage 6d).
3. Regenerate Stage-4b feature parquet (cheap — per-tile features already computed; just
   add the aggregations).
4. Run a dev sweep with `lightgbm_two_stage_balanced` + `target_col=boulder_count` (=
   P1+P2 baseline) + the new feature columns. Use `dataset_v2_dev` within-image scheme
   for the dev pass.
5. Acceptance (dev): Spearman ρ +≥ 0.05 over P1+P2 baseline, AND PR-AUC +≥ 0.03. If both
   clear, promote to full v2.

**Width sanity**: ~30 base features × 3 stats ≈ 90 new columns; well within LightGBM
range.

## Priority 4 detail — Stage 6b CTX-source illumination angles

Full spec in [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) "Stage 6b — CTX-source illumination
angles". This is the **proper H3 test** that's also inference-time-compatible. Higher
cost than Stage 6a, but the hypothesis is sharp:

1. Download the PDS CTX CUMINDEX (~200 MB) — this is the cumulative index of all CTX
   observations with `INCIDENCE_ANGLE` / `EMISSION_ANGLE` / `PHASE_ANGLE` per source.
2. For each tile, spatial-join with the Murray Lab `SeamMap.shp` to identify dominant CTX
   source(s). [`notebook 13 §3.2`](notebooks/13_per_image_heterogeneity.ipynb) confirms
   each HiRISE footprint averages 24 CTX sources (range 4–46), so this is **per-tile**
   not per-image.
3. Aggregate per-source angles → per-tile angles (area-weighted mean over sources
   intersecting the tile).
4. Add as columns in the Stage-4b feature parquet.
5. Acceptance (dev + full-v2): full-v2 per-image AUC ↔ tile-mean CTX_IncidenceAngle
   correlation is significantly negative (ρ < −0.30, p < 0.05) AND PR-AUC +≥ 0.03 over
   P1+P2.
6. **Fallback if 6b fails**: move to Stage 6c (image-level pre-classifier) or to Stage
   6e (mosaic-seam features) — both are alternative anti-signal mechanisms.

## Critical gotchas

- **Inference-time scope (Brian, 2026-05-29)**: the deliverable runs on CTX-only regions
  where HiRISE is absent. Any model input feature must be derivable from CTX alone. Per
  PROMOTION_QUEUE.md "Inference-time scope" section. HiRISE LBL angles are
  **explicitly NOT model features** (kept for diagnostic analysis only).
- **`conda run python -c` rejects multiline strings** — write probes to files under
  `scripts/probes/_*.py`. See [[conda_location]] memory.
- **`cp1252` stdout encoding fails on unicode** in some `conda run` paths — write probe
  output to `.md` files, don't rely on stdout. See `scripts/probes/_diag_extract_nb13_results.py`
  for the pattern.
- **Never run two `nbconvert --execute` concurrently** — caused ~14-min hangs in earlier
  sessions when overlapping kernels contended for caches.
- **AskUserQuestion before**: full-v2 sweeps (Brian-gated; expensive), `git commit`,
  destructive operations on cached artifacts.
- **220 pytest pass baseline** — run `pytest tests/ -q` before promoting changes;
  parametrized tests in `tests/test_modeling_gbm.py` auto-pick up new variants.
- **Stage 6 distinction**: model-improvement work goes under **Part B** of
  PROMOTION_QUEUE.md (Stage 6a/6b/6c/6d/6e/6f), not under existing-stage P-numbers.
  Variant flag changes / target-col choices / doc edits go under **Part A** (P1–P5).
- **`git`-gated allowlist**: `.claude/settings.local.json` allows only `geospatial` conda
  python/jupyter. Brian reviews + commits.

## What we know vs what we suspect (honest status)

Per PROMOTION_QUEUE.md's "Problem catalog & priority":

- ✓ **Problem 1 (target distribution noise)**: solved by P2 (`boulder_count`); dev win
  reproducibly +22 % PR-AUC. Mechanism understood (count-vs-area; CTX texture features
  respond to count of detection events).
- ◐ **Problem 2 (compression)**: P1 fixes the presence-head over-confidence source ONLY.
  The magnitude-head log-positive-median shrinkage remains. High-bin ratio after P1 is
  0.83 not 1.0. Honest verdict: ship as a ranker, not a calibrated abundance regressor.
  Stage 6f (Zero-Inflated Tweedie) is the long-shot magnitude-head loss redesign — only if
  compression is the binding constraint after P1/P2.
- ? **Problem 3 (per-image anti-signal)**: HiRISE LBL angles do NOT predict performance
  (notebook 13 §4). CTX-source illumination (Stage 6b) is the next test — but we don't
  know it's the cause. Other candidate mechanisms: terrain composition (Stage 6c), mosaic
  seams (Stage 6e), image-specific data issues.
- ? **Problem 4 (no surrounding spatial context)**: only indirect evidence (the S=128
  scale Spearman 0.26 → 0.41 finding). Stage 6a is the direct test; could disappoint if
  the S=128 gain was about coarse label-noise averaging rather than spatial integration
  per se.
- ✓ **Problem 5 (metric framing)**: solved by P3+P4 doc reframes; the metrics already
  exist in code (`src/modeling/evaluate.py`).
- ✗ **Problem 6 (5 m/px CTX texture floor)**: unresolved; the eventual unlock is outside
  CTX (THEMIS rock abundance, HiRISE-decimated as a surrogate).

## Future work (not for this session)

- **THEMIS validation**: see [CLAUDE.md §10](CLAUDE.md). THEMIS rocks > 15 cm vs
  BoulderNet > 1 m means population-scaling calibration is needed regardless of target
  choice ([Nowicki & Christensen 2007](https://doi.org/10.1029/2006JE002798)). Open
  question for `boulder_count`: how to pick `mean_boulder_area_per_boulder` at inference
  on CTX-only regions — see [PROMOTION_QUEUE.md P2 "Open inference-time question"](PROMOTION_QUEUE.md).
- **Stage 7 — Compositional study (HiRISE 3 bands, [Delamere 2010](https://doi.org/10.1016/j.icarus.2009.03.012))**:
  **plan drafted 2026-05-30** in [`PLAN_Compositional.md`](PLAN_Compositional.md).
  **Gate on Stage 7.0 feasibility test** (1–2 days, 2–3 images, actual BoulderNet
  labels not predictions; per §3.1 of the plan) before committing to the full 5-substage
  pipeline (~5–7 days). The 7.0 gate de-risks the methodology end-to-end. Central
  methodological challenge: dust confound (`dust_index = RED/BG` as discriminator). Note:
  HiRISE colour covers only ~20 % of each image's swath (central CCDs only). Originally
  CRISM; switched 2026-05-30. *Future-work, after Stage 6 promotions land.*
- The **brainstormed-not-docketed** alternatives in PROMOTION_QUEUE.md Part B
  (LambdaRank, per-image standardisation, monotonic constraints, post-hoc spatial
  smoothing) — try only if 6a–6f all underperform.

## How this session should report progress

End-of-session, update:
1. **`PROMOTION_QUEUE.md`**: any item that passes full-v2 acceptance moves to the
   "Promoted" section. Items that fail move to "Tried, didn't work" with the reason.
2. **`DECISIONS.md`**: one new entry per promoted item with the full-v2 numbers.
3. **`docs/modeling_results.md`**: if a promoted item changes the headline numbers,
   update §9 (the v2 LOIO modeling A/B table).
4. **Memory**: new `project_state_2026-05-XX.md` with the day's outcomes; mark previous
   memory as superseded.
5. **`HANDOFF_NEXT_SESSION.md`** (this file): rewrite the priority order if Stage 6a/6b
   land or if the priority shifts.
6. **AskUserQuestion before git commit** of any of the above.
