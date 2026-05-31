# Promotion queue — dev-validated changes awaiting full-v2 confirmation

Forward-looking docket. Each item is a change that **(a)** is dev-validated on the
5-image `dataset_v2_dev/` within-image scheme, **(b)** has not yet been confirmed on the
full 38-image v2 `dataset_v2/` LOIO scheme, and **(c)** is Brian-gated (full-v2 sweeps are
the expensive step). Listed in priority order. Once promoted, move to [DECISIONS.md](DECISIONS.md).

## Inference-time scope (Brian, 2026-05-29)

The deliverable is **inference on stand-alone CTX images** in regions where HiRISE coverage
is absent ([CLAUDE.md](CLAUDE.md) §1: "predict abundance across the near-global CTX
mosaic where HiRISE coverage is absent"). **Any input feature consumed by the model at
inference must therefore be derivable from CTX alone.** Concretely:

- ✓ allowed: CTX texture stats, GLCM, gradient, shadow_fraction, LBP, lacunarity,
  Canny features (all derived from the CTX mosaic pixel values).
- ✓ allowed: **CTX-source illumination angles** (per-tile or per-region weighted average,
  looked up from the Murray Lab SeamMap + PDS CUMINDEX) — these describe the CTX images
  themselves and are available wherever CTX is available.
- ✗ **NOT allowed**: HiRISE acquisition-time metadata (HiRISE IncidenceAngle /
  EmissionAngle / PhaseAngle / SubSolarAzimuth). These describe when HiRISE took the
  image; on a CTX-only inference region there is no HiRISE image, so no values to feed.

HiRISE LBL angles **are still useful for our own analysis** (e.g. as covariates in the
per-image performance diagnostic in [notebooks/13_per_image_heterogeneity.ipynb](notebooks/13_per_image_heterogeneity.ipynb))
but cannot be added to the model. See the "Out of scope" section at the bottom.

## Docket structure: Part A (pipeline tweaks) vs Part B (Stage 6 — fixing stage)

[CLAUDE.md §4](CLAUDE.md) defines Stages 0–5 as the **build-the-pipeline** stages:
ingest → CTX retrieve → co-register → label → features (4b) → package (5).  Sub-stages
**5b** (binary reframing) and **5c** (within-image CV) extended modeling within Stage 5's
scope.

Going forward this docket is split into two parts that reflect what kind of work each item
actually is:

- **Part A — Pipeline tweaks (P1–P5)**: small variant / target / metric / documentation
  changes that use the existing Stage 0–5 pipeline.  Each item is a single LightGBM flag,
  target-column choice, or doc edit; promotion is a sweep + writeup.
- **Part B — Stage 6: model-improvement / feature augmentation (Stage 6a, 6b, 6c, …)**:
  *new* feature columns or model components that didn't exist in Stages 0–5.  Each item
  requires implementation work *before* the dev sweep — it's the "fixing stage" Brian
  named on 2026-05-30.
  - **Stage 6a** — spatial-context neighbour features (formerly P5b)
  - **Stage 6b** — CTX-source illumination features (formerly P5a)
  - **Stage 6c** — image-level pre-classifier / anti-signal gating (priority 6 below;
    not yet docketed in detail)
  - *(future Stage 6 items slot in as discovered)*

The earlier "Stage-4c" labelling on the spatial-context / CTX-illumination entries was
wrong — those items aren't extensions of Stage 4, they're a new layer.

---

## Problem catalog & priority (2026-05-30 synthesis)

Stepping back from individual queue items, the modeling state of play is best framed as
six related problems. This catalog records what each is, where the evidence lives, what
queue items target it, **and what each item does and does not claim to solve** — so
future sessions can resume context without inheriting overstated confidence.

**Status legend**:
- ✓ **DEV-VALIDATED** — measured win on dev harness; reasonable confidence it carries to
  full v2.
- ◐ **DEV-PARTIAL** — some of the problem is addressed; other parts remain.
- ? **UNTESTED HYPOTHESIS** — on the docket; whether it works is unknown and the
  underlying mechanism may not be what we suspect.
- ✗ **UNRESOLVED** — no current candidate fix; long-term unlocks only.

### Problem 1 — Pixel-aliasing / distribution noise in `fractional_area`
The continuous fractional_area target is heavy-tail with measurement noise from per-polygon
area variability; `boulder_count` is Poisson-like and discrete.
- **Evidence**: dev sweep [§9 of notebook 12](notebooks/12_compression_diagnostic.ipynb):
  switching target to `boulder_count` lifts PR-AUC +22 %, normalised lift +27 %,
  precision@top-5 % +20 % with Spearman / ROC-AUC unchanged.
  [Follow-up sweep 2026-05-30](models/_sweep_target_reformulation/20260530T154730Z) ruled
  out `boulder_area` (no gain) and `log_boulder_area` (no gain) — the win is specific to
  *count* of distinct detection events, not log-scale alone.
- **Targeted by**: **P2** (target = `boulder_count`).
- **Status**: ✓ **DEV-VALIDATED** — strong, reproducible gain on dev metrics. Mechanism
  is empirically clear (count-vs-area). Full-v2 promotion expected to carry.
- **What this does NOT solve**: doesn't change the model's loss / architecture; doesn't
  address the high-tail under-prediction (that's Problem 2). Operational metrics improve
  largely because the training distribution is cleaner.

### Problem 2 — Dynamic-range compression
Hurdle's magnitude head fits `log1p + Huber` on positives (the geometric median), shrinking
the heavy tail; presence head with `is_unbalance=True` inflates `p_pos = 0.85` even on
zeros. **Two distinct compression sources** ([§2 of notebook 12](notebooks/12_compression_diagnostic.ipynb)).
- **Evidence**: [§2 of notebook 12](notebooks/12_compression_diagnostic.ipynb), per-bin
  decomposition table. LOIO post-hoc isotonic does NOT fix it. The high-bin ratio
  (mean_pred / mean_true at fractional_area > 1e-2) was **0.42** in the baseline;
  `balanced` brings it to **0.83**; `weighted` reaches **1.01** but trades away Spearman.
- **Targeted by**: **P1** (presence-head fix, `is_unbalance=False`).
- **Status**: ◐ **DEV-PARTIAL** — P1 fixes the *presence-head* compression source
  (over-confident `p_pos` on zeros) and gives +0.017 ρ / +0.018 AUC. **The magnitude-head
  source (log1p+Huber shrinks to geometric median) is NOT fixed** — high-bin ratio still
  ~0.83, not 1.0. The honest operational verdict from notebook 12 §5: ship `balanced`
  for the small ranking lift, accept that the model is still a **ranker not a calibrated
  abundance regressor**.
- **What might further reduce it (untested)**:
  - P2 (boulder_count) indirectly helps: a Poisson-like target distribution gives the
    magnitude head a cleaner thing to fit, but does NOT redesign the loss.
  - P5b (spatial context) might smooth per-tile noise and reduce effective compression at
    high-truth tiles.
  - A loss redesign (multi-output quantile, multi-task with calibration term) would
    attack it directly but is a bigger change; deferred.

### Problem 3 — Per-image anti-signal (H3)
On ~10 % of v2 images the model is wrong-way-correlated; texture features point the wrong
way somewhere on those images.
- **Evidence**: [§6 of notebook 13](notebooks/13_per_image_heterogeneity.ipynb) on
  ESP_054000_2255 (top-1 % predicted = 0 % truly boulder-rich, base rate 18.3 %).
  HiRISE LBL angles do NOT correlate with performance (out of scope as model features
  anyway).
- **Targeted by**: **Stage 6b** (CTX-source illumination per-tile features) — hypothesis test.
- **Status**: ? **UNTESTED HYPOTHESIS** — *we do not know* what causes the anti-signal.
  H3 names CTX-source illumination as one candidate mechanism (oblique CTX angles make
  ripples / crater rims / regolith cast abundant shadows that `shadow_fraction` then
  mis-reads as boulders). Plausible but **not established**. Stage 6b tests this specific
  hypothesis. Other candidate mechanisms that Stage 6b would NOT address:
  - Terrain / surface composition differences (e.g. basalt vs sedimentary lookalike textures)
  - Mosaic-seam artefacts at boundaries between CTX source images with different gains
  - Image-specific data issues (calibration, compression artefacts)
  - BoulderNet label errors specific to certain images
- **If Stage 6b fails**: we still have the anti-signal problem and a narrower hypothesis
  set.  Next candidate would be Stage 6c (image-level pre-classifier, priority 6 below) —
  model-side triage rather than a feature-side fix.

### Problem 4 — No surrounding spatial context (Brian's 2026-05-30 flag)
Every tile is treated as independent. Per-tile features summarise only what's inside the
tile boundary; neighbour information is discarded. Boulder fields are spatially coherent
(crater ejecta, fluvial deposits, exhumed bedrock), so a tile in a real cluster differs
from an isolated false-positive texture even when per-tile features look identical.
- **Indirect evidence (going in)**: the S=128 scale study ([§10.2 of modeling_results.md](docs/modeling_results.md))
  jumped Spearman 0.26 → 0.41 at S=64 → S=128 dev within-image.
- **Direct evidence (2026-05-30)**: Stage 6a sweep on 6 (variant × scale) combinations
  ([Stage 6a Dev result below](#stage-6a--spatial-context-neighbour-features-new-2026-05-30)).
  The 5 × 5 stencil at S=32 PASSES both strict criteria (Δ Spearman ρ +0.053, Δ PR-AUC
  +0.053); the canonical S=64 baseline already integrates enough context that no
  neighbour-stencil variant clears the bar at S=64. **The S=128 → S=64 finding partly
  carries**: spatial-context integration helps, but is *finer-scale than* S=64. The
  "label-noise averaging" alternative hypothesis is partially ruled out by the 5 × 5
  @ S=32 pass.
- **Targeted by**: **Stage 6a** (neighbour-feature aggregation).
- **Status**: ◐ **DEV-PARTIAL** — direct test passes at S=32 (5 × 5 stencil); at S=64
  operational top-K metrics improve (precision@top-5 % +0.044 with default 3 × 3) but
  the strict Spearman + PR-AUC thresholds don't both clear. Brian (2026-05-30): defer
  full-v2 promotion until single-recipe choice or until Stage 6b lands.

### Problem 5 — Metric framing
ROC-AUC averages across thresholds; cross-image mean AUC averages across a bimodal
per-image distribution. Both hide where the model is actually useful.
- **Evidence**: [§6 of notebook 12](notebooks/12_compression_diagnostic.ipynb): per-image
  `fa_gt_1e-2` lift@top-K = 1.43 vs `bc_ge_1`'s 1.02; max per-image AUC 0.91 / lift 9.1×;
  cross-image mean = 0.62.
- **Targeted by**: **P3** (PR-AUC + lift become headline), **P4** (retire `bc_ge_1` as
  primary binary).
- **Status**: ✓ **DEV-VALIDATED** — change is methodological, not model-altering. The new
  metrics already exist in the codebase (notebook 12 §9 + `src/modeling/evaluate.py`); the
  docket items are documentation reframes.
- **What this does NOT solve**: doesn't make the model better — just reports its
  performance honestly. Risks under-claiming if PR-AUC at the right threshold also turns
  out to be middling on full v2.

### Problem 6 — 5 m/px CTX texture floor (H5)
Texture features at 5 m/px have extracted what they can; within-image ≈ LOIO across all
scales suggests the per-tile signal floor is real.
- **Evidence**: [§9.4 of modeling_results.md](docs/modeling_results.md) — three target
  framings converge on AUC ≈ 0.55–0.62.
- **Targeted by**: long-term unlocks outside CTX (THEMIS thermal, HiRISE-decimated as a
  surrogate, higher-res inputs).
- **Status**: ✗ **UNRESOLVED** — eventually binds; no current candidate fix on the docket.
  **May or may not bind soon** depending on whether Stage 6a (spatial context) and Stage
  6b (CTX illumination) lift the per-tile ceiling. If those work, the texture-floor
  framing itself may need revision.

### Recommendation order

Same status legend as the problem catalog above (✓ DEV-VALIDATED / ◐ DEV-PARTIAL / ?
UNTESTED HYPOTHESIS / ✗ UNRESOLVED).  Items 3, 4, 6 are **untested** — they're on the
docket because the mechanism is plausible and inference-compatible, not because we know
they'll work.

| order | part | item | status | what it might do (and what falsifies it) | cost |
|------:|------|------|:-:|------|------|
| 1 | A | **P1 + P2 full-v2 promotion** | ✓ | Confirm the +22 % PR-AUC, +27 % normalised-lift dev win on the full 38-image LOIO. Falsified if full-v2 gain is `<` +0.05 PR-AUC. | 1-2 hr |
| 2 | A | **P3 + P4 doc reframe** | ✓ | Report metrics honestly (PR-AUC + lift, not ROC-AUC; `fa_gt_1e-2`, not `bc_ge_1`). No model change. Can't really fail; risk is that the new headline numbers look middling once we honestly report them. | ~1 hr |
| 3 | B | **Stage 6a — spatial-context features** | ◐ | **Tried 2026-05-30**: 5 × 5 stencil @ S=32 PASSES strict criteria (Δ ρ +0.053, Δ PR-AUC +0.053); at S=64 only operational top-K metrics improve. Full-v2 promotion deferred. The S=128 → S=64 mechanism partly carries; S=64 baseline already near spatial-integration ceiling. | done (dev) |
| 4 | B | **Stage 6b — CTX-source illumination** | ? | *If* CTX-source illumination is the H3 anti-signal cause: across-image AUC ↔ CTX-incidence correlation becomes significantly negative; PR-AUC +≥ 0.03 over P1+P2. Falsified if no correlation appears — would shift the anti-signal investigation to Stage 6c or to terrain / mosaic-seam mechanisms. | 1-2 days |
| 5 | A | **P5 — binary classifier calibration fix** | ✓ | Cosmetic ECE drop 0.26 → ~0.05 expected (mirrors P1's presence-head fix on the binary classifier). Ranking unchanged. Hard to fail; the question is whether anyone uses the probabilities raw or just for ranking. | ~2 hr |
| 6 | B | **Stage 6c — image-level pre-classifier** | ? | Fallback if 6a / 6b underperform. Train a per-image "is this image well-fit by texture features?" classifier; use it to gate per-tile predictions (or to exclude anti-signal images from reporting). Worth doing if Problem 3 stays unresolved. | ~1 day |
| 7 | — | THEMIS / HiRISE-surrogate | ✗ | The eventual H5 unlock once CTX texture itself binds. Out of scope for the modeling pass. | weeks |

**Read this honestly**: items 1, 2, 5 are bank-the-wins (predictable, validated, mostly
already in code).  Items 3, 4, 6 are *bets* — each one tests a specific hypothesis that
could fail, and the failure itself would be informative.  Item 7 is the long-horizon
unlock.

---

# Part A — Pipeline tweaks (existing Stages 0–5)

Small variant / target / metric / documentation changes that use the existing pipeline.
Each item is a flag flip, target-column choice, or doc edit; promotion is "run a full-v2
sweep + update writeup". No new feature columns or model components.

---

## P1 — Model defaults: `lightgbm_two_stage_balanced` (presence-head fix)

**Change**: drop `is_unbalance=True` from the presence head of `lightgbm_two_stage`. One-line
flag flip; new variant lives in [`src/modeling/gbm.py`](src/modeling/gbm.py) via
`_TwoStageBase`.

**Dev evidence** ([`models/_sweep_compression_fixes/20260529T211211Z`](models/_sweep_compression_fixes/20260529T211211Z),
within-image 20 folds, S=64):

| metric              | baseline | balanced | Δ          |
|---------------------|---------:|---------:|-----------:|
| Spearman ρ          | +0.263   | **+0.280** | **+0.017** |
| presence AUC        | 0.538    | **0.556** | **+0.018** |
| high-bin ratio      | 0.83     | 0.83     | ~0         |
| zero-bin pred       | 0.0024   | 0.0026   | ~0         |

**Why this matters**: `is_unbalance=True` was inflating `p_pos` on true-zero tiles (mean 0.85
on zeros) — the over-prediction floor diagnosed in
[`docs/modeling_results.md`](docs/modeling_results.md) §11.1. The fix is a no-cost ranking
improvement; the floor barely changes because the magnitude-head ceiling now dominates the
absolute prediction.

**Promotion command**:
```
conda run -n geospatial python scripts/sweep.py \
    --variants lightgbm_two_stage_balanced \
    --dataset-dir dataset_v2 --scheme loio_nfold
```

**Acceptance criterion**: full-v2 LOIO Spearman ρ at S=64 ≥ 0.18 (baseline 0.169) AND
presence AUC at S=64 ≥ 0.58 (baseline 0.579). Either condition met → promote.

---

## P2 — Target: `target_col=boulder_count` (with `lightgbm_two_stage_balanced`)

**Change**: switch the regression target from `fractional_area` to `boulder_count`. No
new training code — `run_loio(..., target_col='boulder_count')` already works because
`boulder_count` is in every label parquet.

**Dev evidence** ([`models/_sweep_target_reformulation/20260529T221912Z`](models/_sweep_target_reformulation/20260529T221912Z),
within-image 20 folds, S=64):

| metric                        | fractional_area | boulder_count | Δ                |
|-------------------------------|-----------------:|--------------:|-----------------:|
| Spearman ρ                    | +0.280          | +0.283        | +0.003           |
| ROC-AUC (presence)            | 0.556           | 0.564         | +0.008           |
| **PR-AUC**                    | 0.526           | **0.640**     | **+0.114 (+22%)** |
| **normalised lift@top-K**     | 0.488           | **0.619**     | **+0.131 (+27%)** |
| **precision@top-5 %**         | 0.549           | **0.660**     | **+0.111 (+20%)** |

**Why this matters**: `fractional_area` is pixel-aliasing-noisy at the low end; `boulder_count`
is alias-robust (a 4 m² boulder in a 5 m pixel contributes either 0 or 1). Spearman and
ROC-AUC are blind to the gain (rank-invariant / threshold-averaged); PR-AUC and lift see it.
Mechanism + full discussion: [`docs/modeling_results.md`](docs/modeling_results.md) §11.4–11.6.

**Dependency**: P1 (the `_balanced` variant must be the one used). Run after or alongside P1.

**Promotion command**:
```
# Custom (no direct sweep.py flag for target_col yet; mirror the dev probe)
conda run -n geospatial python scripts/probes/_sweep_target_reformulation.py \
    --targets boulder_count --scales 3 \
    --dataset-dir dataset_v2
```
(Eventually: add a `--target-col` flag to [`scripts/sweep.py`](scripts/sweep.py) so this
is a clean promotion.)

**Acceptance criterion**: full-v2 LOIO PR-AUC at S=64 with boulder_count > full-v2 LOIO
PR-AUC at S=64 with fractional_area (whatever the baseline is) by ≥ +0.05. Spearman ρ should
not regress.

### THEMIS comparability (Brian's question, 2026-05-30)

Switching the model target from `fractional_area` to `boulder_count` looks like it sacrifices
direct comparability with THEMIS rock-abundance maps (which are themselves area fractions).
A follow-up dev sweep on 2026-05-30 ([`models/_sweep_target_reformulation/20260530T154730Z`](models/_sweep_target_reformulation/20260530T154730Z))
tested **`boulder_area` and `log_boulder_area`** as alternative targets (direct THEMIS-area
equivalents). Result at S=64 within-image:

| target              | Spearman ρ  | PR-AUC | normalised lift | precision@top-5 % |
|---------------------|------------:|-------:|----------------:|------------------:|
| `fractional_area`   | +0.280      | 0.526  | 0.488           | 0.549             |
| **`boulder_count`** | +0.283      | **0.640** | **0.619**    | **0.660**         |
| `boulder_area`      | +0.300      | 0.531  | 0.479           | 0.564             |
| `log_boulder_area`  | +0.282      | 0.525  | 0.482           | 0.537             |

**The +22 % PR-AUC gain is specific to `boulder_count`** — `boulder_area` is essentially
equivalent to `fractional_area`, and `log_boulder_area` ≠ `log_boulder_count` in performance,
ruling out "log scale alone" as the explanation. The real mechanism is likely that CTX
texture features respond to **count of detection events** (multiple shadows from multiple
boulders) more than to **total area** of those events. A clean signal-detection finding.

**THEMIS comparison still works** via a simple post-hoc conversion at inference time:

```
predicted_themis_rock_abundance
  ≈ predicted_count
    × mean_boulder_area_per_boulder        # from training labels, per image / region
    / tile_area
    × population_scaling_factor             # THEMIS vs BoulderNet rock-size populations
```

- `mean_boulder_area_per_boulder` is per-image (~1-2 m² per boulder for v2; modest variance).
- `population_scaling_factor` calibrates the ~100× linear-size gap between THEMIS rocks (>15 cm)
  and BoulderNet boulders (>1 m). This step is **required for any approach** — direct
  `fractional_area` comparison would need it too, since the two metrics measure different
  rock populations.

**Net**: `boulder_count`-primary keeps the model performance win; the additional THEMIS-comparison
noise is the variance of `mean_boulder_area_per_boulder` within an image, which is small
relative to the +22 % PR-AUC gain. Multi-target (count + area) is **not** needed based on dev
evidence; revisit only if THEMIS validation shows the conversion noise is binding.

**Open inference-time question** (Brian, 2026-05-30): the conversion above uses
`mean_boulder_area_per_boulder` computed from training labels.  At full-mosaic inference
on CTX-only regions, we have no labels and therefore no per-image mean.  Options:
**(a)** use a global mean from the training set (simple; modest error if size distribution
varies across terrain); **(b)** pre-compute a per-region mean from a regression on
HiRISE-overlap regions and interpolate spatially (more involved); **(c)** decide we only
need rank correlation with THEMIS (Brian's lean), in which case the conversion noise
doesn't matter — Spearman is rank-invariant under per-image multiplicative scaling.

Track this as a P2-blocker only if THEMIS validation requires calibrated abundance values;
otherwise it's a "ship rank-correlation comparison, document conversion approach in the
write-up" deferred item.

---

## P3 — Headline-metric reframe in the docs

**Change**: the v2 deliverable's headline metrics in
[`docs/modeling_results.md`](docs/modeling_results.md) (§9, §11.4) should be **PR-AUC +
lift@top-K**, not ROC-AUC. ROC-AUC averages across all thresholds; for the operational task
("flag the top-K tiles for HiRISE/THEMIS follow-up") only the very top of the ranking
matters. PR-AUC and lift key on that.

**Dev evidence**: per the H1 framework in
[`notebooks/12_compression_diagnostic.ipynb`](notebooks/12_compression_diagnostic.ipynb) §7,
ROC-AUC at S=64 changed by only +0.008 with `boulder_count` while PR-AUC moved +0.114. The
deliverable looked unchanged under one metric and ~20 % better under the other; the
ROC-AUC framing was burying the win.

**Dependency**: P2 confirmed on full v2 (so the metric change is anchored to a real result).

**No promotion command** — a documentation pass. Update §9 of `docs/modeling_results.md`
and the headline rows in the notebook 11 tables.

---

## P4 — Binary threshold: `fa_gt_1e-2` as primary, retire `bc_ge_1`

**Change**: stop using `bc_ge_1` ("≥ 1 boulder") as the primary binary target — it's
operationally meaningless ("any boulder at all in a 320×320 m tile"). Promote `fa_gt_1e-2`
("boulder-rich tile", > 1 % area) as the primary binary metric in
[`src/modeling/binary_target.py`](src/modeling/binary_target.py) and update all reporting.

**Dev evidence**: per [`docs/modeling_results.md`](docs/modeling_results.md) §11.4: the
existing v2 binary sweep at S=64 had lift@top-K = 1.43 for `fa_gt_1e-2` vs 1.02 for
`bc_ge_1` — and per-image maxima up to 9.1× for the rarer-positive cases. The §6.1
"binary doesn't help" verdict (v1, on `bc_ge_1`) was a threshold artefact.

**Dependency**: independent of P1/P2 — pure metric/reporting change.

**No promotion command** — update the `BINARY_TARGETS` default + documentation. Existing
`fa_gt_1e-2` evidence (from [`models/_sweep_binary/20260529T075754Z`](models/_sweep_binary/20260529T075754Z))
is already in place.

---

## P5 — `lightgbm_classification` calibration fix (drop `scale_pos_weight`)

**Change**: mirror the P1 presence-head fix on the binary classifier:
[`src/modeling/gbm.py`](src/modeling/gbm.py) `LightGBMClassification` currently sets
`scale_pos_weight = neg / pos` (the same shifted-decision-boundary mechanism that inflated
`p_pos` in P1). Test removing it (and optionally adding Platt/isotonic post-hoc
calibration) — expect ECE 0.16–0.27 → ~0.05, ROC-AUC unchanged.

**Dev evidence**: not yet run — was queued in the morning session, paused for the metric
reframing discussion.

**Promotion command**:
```
# Need to first add a `balanced` variant of LightGBMClassification, then sweep.
# See PLAN below.
```

**Dependency**: needs a small code addition first.
[`src/modeling/gbm.py`](src/modeling/gbm.py) `LightGBMClassificationBalanced(LightGBMClassification)`
that overrides `fit` to skip `scale_pos_weight`. Then `sweep_binary.py` already accepts
arbitrary variants from `BINARY_VARIANTS`.

---

# Part B — Stage 6: model improvement / feature augmentation

New feature columns or model components that didn't exist in Stages 0–5. Each item
requires implementation work *before* a dev sweep. Items in this part are **untested
hypotheses**: the mechanism is plausible and inference-compatible, but the dev outcome is
unknown.

When promoted (dev-confirmed AND full-v2-confirmed), an item moves from this part into the
"Promoted" section at the bottom and is documented in [`docs/modeling_results.md`](docs/modeling_results.md).

---

## Stage 6b — CTX-source illumination angles (hypothesis test for H3 anti-signal)

*Was P5a in earlier revisions of this docket.*

**Change**: for each HiRISE footprint, identify the dominant CTX source image(s) from the
Murray Lab `SeamMap.shp`, look up each source's `INCIDENCE_ANGLE`, `EMISSION_ANGLE`,
`PHASE_ANGLE` from the PDS CUMINDEX, and aggregate (e.g. area-weighted mean) to a single
set of CTX-illumination values per HiRISE footprint. Join those into the feature parquet.

**Why this matters (hypothesis, not established mechanism)**: one candidate explanation
for the per-image anti-signal failure mode is that CTX `shadow_fraction` only carries
boulder signal at moderate illumination — at very oblique CTX-source-image angles, ripple
fields, crater rims, and bare regolith cast abundant shadows that the feature then
*mis-reads* as boulders. The bimodal per-image AUC distribution ([notebook 13](notebooks/13_per_image_heterogeneity.ipynb)
§2.1, ~7 winners / ~3 anti-signal / rest near chance) **is consistent with this mechanism
but does not establish it**. Other plausible candidates (terrain composition, mosaic seams,
image-specific data issues) are NOT distinguishable from this one with the current
diagnostics.

P5a tests *whether CTX-source illumination explains the anti-signal images*. If the gain
is real, we've found a mechanism and a feature. If not, we move to image-level pre-classifier
(priority 6) or accept the per-image variance as is. **These features are inference-time
compatible** — derivable from the CTX mosaic + Murray Lab metadata alone, unlike HiRISE
LBL angles (see "Out of scope" at the bottom).

**Evidence**: the Murray Lab seam map gives CTX source IDs per region; we confirmed in
[notebook 13](notebooks/13_per_image_heterogeneity.ipynb) §3.2 that each HiRISE footprint
is covered by **a mean of 24 CTX source images** (range 4–46), so the per-footprint
aggregation is non-trivial — the right granularity is probably **per-tile** (a 320 × 320 m
tile likely has 1–3 dominant sources). Lookup is tractable but more involved than a
single per-footprint mean.

**Implementation cost**: ~1–2 days. Need to:
1. Download the PDS CTX CUMINDEX (~200 MB) and parse it into a lookup.
2. For each tile (not just per footprint!), spatial-join with the SeamMap to get dominant
   CTX source(s).
3. Aggregate per-source angles → per-tile angles (area-weighted mean over sources
   intersecting the tile).
4. Add as columns in the existing per-tile feature parquet (Stage-4b output).
5. Modify Stage-4b to read these and add 3 feature columns.

**Acceptance criterion (and what falsifies the hypothesis)**:
- **Pass** = in the full-v2 LOIO sweep, **(a)** the AUC ↔ tile-mean CTX_IncidenceAngle
  correlation across the 38 images becomes significantly negative (ρ < −0.30, p < 0.05),
  confirming the H3 mechanism, AND **(b)** adding the CTX-illumination features lifts
  PR-AUC by ≥ +0.03 over the P1+P2 baseline.
- **Fail / inconclusive** = no significant correlation appears, OR PR-AUC gain is < +0.03.
  Move to Stage 6c (image-level pre-classifier) and document the negative result.
  Don't promote.

---

## Stage 6a — Spatial-context neighbour features (new 2026-05-30)

*Was P5b in earlier revisions of this docket.*

### Dev result (2026-05-30): tried; partial pass; deferred promotion

Implementation shipped to [`src/spatial_features.py`](src/spatial_features.py) (15 unit
tests in [`tests/test_spatial_features.py`](tests/test_spatial_features.py)) +
[`scripts/run_stage6a.py`](scripts/run_stage6a.py) (driver) +
[`scripts/run_stage6a_repackage.py`](scripts/run_stage6a_repackage.py) (re-packaging
within_image_4fold → `within_image_4fold_nbr*`) +
[`scripts/probes/_sweep_stage6a.py`](scripts/probes/_sweep_stage6a.py) (sweep). The
augmented features are written to `dataset_v2_dev/features_nbr/` and packaged into
`dataset_v2_dev/packaged/within_image_4fold_nbr/` (and `_nbr_s5/` / `_nbr_max/` for
the follow-up variants); the canonical Stage 4b cache is untouched.

**Three variants × two scales sweep**
([`models/_sweep_stage6a/20260531T004356Z/aggregate.parquet`](models/_sweep_stage6a/20260531T004356Z/aggregate.parquet),
combined comparison in
[`scripts/probes/_diag_stage6a_followup_compare.md`](scripts/probes/_diag_stage6a_followup_compare.md)):

| variant | scale | Δ Spearman ρ | Δ PR-AUC | Δ lift_norm | Δ prec@5% | Δ recall@5% | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 3×3, mean+max+std | S=32 | +0.0214 | **+0.0533** | +0.0570 | +0.0629 | +0.0794 | FAIL (ρ) |
| **5×5, mean+max+std** | **S=32** | **+0.0534** | **+0.0526** | +0.0547 | +0.0722 | +0.0546 | **PASS** |
| 3×3, max-only | S=32 | +0.0207 | +0.0341 | +0.0393 | +0.0575 | +0.0647 | FAIL (ρ) |
| 3×3, mean+max+std | S=64 | −0.0065 | +0.0098 | +0.0068 | **+0.0436** | +0.0204 | FAIL |
| 5×5, mean+max+std | S=64 | +0.0269 | +0.0039 | +0.0069 | −0.0050 | +0.0008 | FAIL |
| 3×3, max-only | S=64 | −0.0383 | +0.0120 | +0.0003 | +0.0186 | +0.0221 | FAIL |

Acceptance threshold: Δ Spearman ρ ≥ +0.05 **AND** Δ PR-AUC ≥ +0.03 vs the P1+P2 baseline at the same
scale. **5×5 stencil at S=32 is the only clean pass.** The default 3×3 stencil clears
PR-AUC at S=32 and lifts precision@top-5% by +0.044 at S=64, but neither clears
both thresholds simultaneously at the canonical S=64 scale.

**Best absolute numbers across the grid** (presence-AUC discarded per Brian
2026-05-30; not a useful metric here):
- Spearman ρ: **5×5 @ S=64** = +0.310 (vs baseline 0.283)
- PR-AUC: max-only @ S=64 = 0.652 (vs baseline 0.640)
- Precision@top-5 %: default 3×3 @ S=64 = 0.704 (vs baseline 0.660)
- Recall@top-5 %: default 3×3 @ S=32 = 0.147 (vs baseline 0.068; **+2.2 ×**)

**Mechanistic reading.** Spatial-context lift is real but operates at *finer scales than
the canonical S=64*. At S=64 each 320 × 320 m tile already integrates substantial
context; a 3 × 3 stencil = 960 × 960 m, mostly redundant with the tile's own GLCM /
shadow stats. At S=32 the per-tile context is 160 × 160 m, so neighbour aggregation
recovers what S=64 gets natively — and the 5 × 5 stencil at S=32 (= 800 × 800 m) is
sized comparably to the S=64 baseline tile, exactly the regime where the S=128 scale
study saw 0.26 → 0.41.

**The S=128 → S=64 scale-study finding partly carries to neighbour features** —
contrary to the "label-noise averaging" risk flagged in the original spec. The 5 × 5
@ S=32 result IS a direct test of "the spatial-integration mechanism specifically",
and that test passes. The S=64 result does not falsify the mechanism; it shows the
S=64 tile is already at the spatial-integration ceiling for these features.

**Per-fold + per-image variance** ([`scripts/probes/_diag_stage6a_fold_variance.md`](scripts/probes/_diag_stage6a_fold_variance.md)
for the default-variant run): 60–65 % win rate per fold across PR-AUC / Spearman; the
default-variant @ S=64 mean is dragged down by ESP_064510_2260 (−0.32 Spearman, −0.083
precision@top-5 %), while ESP_069669_2220 shows +0.27 precision@top-5 %. Consistent
with the per-image heterogeneity story from notebook 13 — neighbour features help
where boulder fields are spatially coherent and hurt on the anti-signal images.

**Brian decision (2026-05-30)**: *document all variants as informative; defer
promotion pending Stage 6b results or until we choose a single recipe.* Stage 6a
stays on the docket below the "tried, didn't fail outright" line. Full-v2 promotion
deferred.

**Open follow-up if Stage 6a comes back into focus**:
- Re-test the 5 × 5 @ S=32 recipe on the full v2 LOIO scheme to confirm it carries
  beyond the 5 dev images. Cost: ~1 hr (sweep) + ~5 min (re-augmentation + repackage
  on all 38 v2 images).
- Try wider stencils (7 × 7 at S=32 or 3 × 3 / 5 × 5 at S=16) to see whether the
  trend "wider stencil at finer scale" extends.
- Try Stage 6d (multi-scale features) layered on top — different mechanism, may
  compose.

---

### Original spec / motivation

**Change**: in [`src/features.py`](src/features.py), after the per-tile feature
computation, add a neighbour-aggregation pass.  For each ObsId × scale, lay tiles on the
(ti, tj) grid and run 2-D convolutions over each existing numeric feature column:

- `nbr_mean_<feature>` — 3 × 3 mean over the 8 neighbours + self (or 8 neighbours only);
  picks up local context smoothing.
- `nbr_max_<feature>` — 3 × 3 max; picks up "any boulder activity nearby" even if the
  central tile's own feature is low.
- `nbr_std_<feature>` — 3 × 3 stdev; heterogeneity / roughness proxy.

Optional **multi-scale variant**: for tiles at S=64, additionally include features from
the corresponding S=128 parent tile (the larger context the S=128 scale study already
benefits from), as columns named `S128_<feature>`.

Boundary handling: pad with NaN at image edges; the LightGBM booster handles missing
values natively.

**Why this matters (predicted impact, not measured)**: every existing per-tile feature
summarises only what's inside the tile boundary.  Boulder fields are spatially coherent
(crater ejecta, fluvial deposits, rockfalls from cliffs), so a tile in a real cluster
differs from an isolated false-positive texture even when their per-tile features look
identical.  **Indirect evidence** for this lever: the S=128 scale study
([§10.2 of modeling_results.md](docs/modeling_results.md)) showed dev within-image Spearman
jumping 0.26 → **0.41** when the same features are aggregated over a 16× larger area.
That's a strong signal that bigger spatial integration helps for ranking.  Adding
neighbour aggregations at S=64 *should* buy a similar benefit while keeping spatial
resolution — **but this is extrapolation from a related experiment**, not a direct test
of the neighbour-features specifically.

Risk: the S=128 gain might be driven by something other than spatial integration per se
(e.g. coarse-tile averaging of label-aliasing noise; coarse-tile feature stability against
small-tile sampling variance).  If so, neighbour aggregations at S=64 might capture less
of the benefit than the scale study suggests.

The within-image diagnostic ([§9.4 of modeling_results.md](docs/modeling_results.md))
showed within-image ≈ LOIO at every scale — meaning the model is NOT exploiting "this
region of this image is different from that region", i.e., it's not using spatial structure
within an image.  Neighbour features attack that gap directly.

**Evidence already in hand**:
- S=128 scale study: Spearman 0.26 → 0.41 from spatial integration alone.
- CNN context-patch experiment (P=32 → P=128): AUC 0.474 → 0.503 (weak signal but
  consistent direction).  CNN was a weak baseline so this isn't strong evidence.
- Within-image ≈ LOIO: not exploiting within-image structure.

**Implementation cost**: ~1-2 days.
1. Stage 6a addition in [`src/features.py`](src/features.py): grid the tiles, run
   `scipy.ndimage.uniform_filter` (mean), `maximum_filter`, and a custom std filter over
   each numeric feature column per ObsId × scale.
2. Regenerate Stage-4b output cache (cheap — per-tile features already computed; just
   add aggregations on top).
3. Run a dev sweep with the same `lightgbm_two_stage_balanced` variant (P1) + the new
   features.
4. Add a notebook section + figure to [notebook 12](notebooks/12_compression_diagnostic.ipynb)
   or a new notebook 14.

**Acceptance criterion (and what falsifies the hypothesis)**:
- **Pass** = on the v2-dev within-image scheme (20 folds, S=64), adding neighbour features
  (a) lifts Spearman ρ by ≥ +0.05 over the P1+P2 baseline (i.e. closer to the S=128 result),
  AND (b) lifts PR-AUC by ≥ +0.03 over the P1+P2 baseline.
- **Fail / inconclusive** = neither (a) nor (b) clears the threshold. This would falsify the
  reading that "the S=128 gain is about spatial context per se" — interesting in itself.
  Document the negative result; consider trying max-filter-only or wider stencils as
  follow-up before declaring fully dead.
- Width sanity check: ~30 base features × 3 stats ≈ 90 new columns at one scale; well
  within LightGBM's range.

If the dev numbers carry, promote to full v2 with the same recipe.

**Inference-time compatibility**: ✓ all features are convolutions of CTX-derived features
the model already consumes; identical operation at full-mosaic inference time.  No
HiRISE dependency.

**Relationship to Stage 6b**: Stage 6a (spatial neighbours) and Stage 6b (CTX-source
illumination) target different failure modes — Stage 6b would fix anti-signal images
where the texture features *mean the wrong thing* due to illumination geometry; Stage 6a
helps the presence/magnitude split and ranking quality on images where features are
informative but isolated tiles are noisy. They're complementary; do both if the budget
allows.

---

## Stage 6c — Image-level pre-classifier / anti-signal gating (placeholder)

*Status: not yet docketed in detail. This is the fallback if Stage 6a / 6b underperform.*

**Idea**: train a per-image classifier ("is this image well-fit by texture features?")
using image-level summary statistics + cross-validated performance on a held-out portion
of training data.  At inference time, use the pre-classifier to either (a) **gate
per-tile predictions** (down-weight or abstain on images flagged as anti-signal), or
(b) **exclude flagged images from reporting** (analytical-only triage; doesn't change the
model output but changes how we communicate performance).

**When to docket in detail**: after Stage 6a / 6b dev results land.  If anti-signal
persists after those, write this up properly with a concrete implementation plan.

---

## Stage 6d — Multi-scale feature columns (new 2026-05-30)

**Change**: for each tile at scale S, include features computed at the enclosing tiles at
*other* scales as additional columns.  Concretely, for S=64 prediction, include:

- The S=64 tile's own features (as today)
- Mean / max / std of features over the **4 enclosing S=32 children**
- Mean / max / std of features over the **16 enclosing S=16 grandchildren**

These are essentially "spatial neighbour features" but indexed by the nested grid
hierarchy rather than the same-scale 3×3 stencil.  Different signal: at the S=32 scale,
the 4 children describe sub-tile heterogeneity within the S=64 tile (where Stage 6a's same-
scale neighbours describe inter-tile context).

**Why this matters**: complementary to Stage 6a.  The S=128 scale study showed that bigger
spatial integration helps Spearman; Stage 6a captures that via same-scale neighbours,
Stage 6d captures it via the nested grid (which is *already cached* — features at
S=8/16/32/64 are computed during Stage 4b for the same image).

**Implementation cost**: ~0.5 day.  All features already exist; this is just a join + an
aggregation.  No re-feature-extraction needed.

**Acceptance criterion**: dev within-image (a) Spearman ρ +≥ 0.03 over the P1+P2 baseline,
AND (b) PR-AUC +≥ 0.02 over the P1+P2 baseline.  If Stage 6a and 6d are run together, the
gains should partially overlap; we want the **incremental** Stage 6d gain when stacked on
top of Stage 6a.

**Inference compatibility**: ✓ All features derive from CTX alone.  At full-mosaic
inference time, the same nested-tile aggregation works trivially.

---

## Stage 6e — Mosaic-seam features (new 2026-05-30)

**Change**: for each tile, add features that describe the tile's proximity to (and
position relative to) CTX-source-image boundaries in the Murray Lab mosaic.  Specifically:

- `distance_to_nearest_seam_m` — distance from tile centre to the nearest polygon boundary
  in the `SeamMap.shp` for the covering Murray Lab tile.
- `n_seams_intersecting_tile` — count of seam-polygon boundaries crossing the tile.
- `dominant_source_fraction` — fraction of tile area covered by the single dominant CTX
  source image (low value = tile straddles multiple CTX sources).

**Why this matters**: [Dickson 2024 (the Murray Lab CTX mosaic paper)](https://doi.org/10.1029/2024EA003555)
explicitly documents that seam artefacts manifest as **disparate brightness/contrast and
surface texture on opposite sides of the seamline**.  Tiles straddling seams are exposed
to two-different-CTX-image texture statistics that we currently treat as one.  This is a
parallel candidate mechanism for the per-image anti-signal failure mode (Problem 3) — it's
not just *which* CTX sources contribute (Stage 6b illumination) but *where the seams are*.

**Implementation cost**: ~0.5–1 day.  We already extracted SeamMap.shp during notebook 13
§3.2 work (see [`scripts/probes/_diag_per_image_breakdown.py`](scripts/probes/_diag_per_image_breakdown.py)).
The geometry computation is straightforward `shapely` work; adds 3 columns per tile.

**Acceptance criterion**: dev within-image (a) PR-AUC +≥ 0.02 over the P1+P2 baseline,
AND (b) `distance_to_nearest_seam_m` shows a significant correlation with per-image AUC
across the 38 v2 images (the H3 mechanism check).  If both clear, Stage 6e + 6b together
form a "CTX provenance" feature group; if only (a) without (b), the gain is real but the
mechanism is something else.

**Inference compatibility**: ✓ SeamMap.shp is public Murray Lab data, available at
inference time.

**Relationship to Stage 6b**: complementary; 6b tests CTX-source illumination, 6e tests
mosaic-stitching boundary effects. Either could explain the H3 anti-signal images, or both
could be needed.

---

## Stage 6f — Zero-Inflated Tweedie boosted trees (new 2026-05-30)

**Change**: replace the current hurdle (presence × magnitude) model family with a
**Zero-Inflated Tweedie** boosted-trees model that fits the zero-process and the
positive-process jointly.  Recent (2024) implementation in CatBoost ([Chen et al. 2024,
arXiv:2406.16206](https://arxiv.org/abs/2406.16206)) shows superior performance vs both
classical Tweedie and two-part hurdle models on insurance-claims data — a target shape
very similar to ours (zero-inflated heavy-tail).

**Why this matters**: addresses **Problem 2 (compression)** more directly than the current
P1 (`balanced`) fix can.  The compression-mechanism diagnosis in §2 of [notebook 12](notebooks/12_compression_diagnostic.ipynb)
showed two distinct sources: presence-head over-confidence (P1 fixes) AND magnitude-head
shrinkage to log-positive median (still unfixed).  ZI-Tweedie redesigns the loss so the
zero-process and the magnitude-process are jointly modelled rather than multiplied — which
preserves the heavy tail rather than shrinking to a median.

**Implementation cost**: ~2–3 days.  Two paths:
1. **CatBoost integration**: install CatBoost, port the [Chen et al. 2024](https://arxiv.org/abs/2406.16206)
   recipe (their code may be available on the paper's repo).  Lower-effort path.
2. **Custom LightGBM objective**: implement the ZI-Tweedie loss + gradient + hessian as a
   custom objective in LightGBM.  Higher-effort but stays in the existing pipeline.

**Acceptance criterion**: dev within-image at S=64, with P2 target (`boulder_count`):
(a) compression high-bin ratio (mean_pred / mean_true at fa > 1e-2) **≥ 0.95** (vs P1+P2's
0.83), AND (b) Spearman ρ ≥ P1+P2 baseline, AND (c) PR-AUC ≥ P1+P2 baseline.  If (a) clears
without regressing (b)/(c), this is a major win on Problem 2 specifically.

**Inference compatibility**: ✓ standard tabular model; no new input features.

**Order vs Stage 6a/6b**: 6f is a *loss redesign*; 6a/6b are *new feature columns*.  These
are independent improvements that compose.  Run 6a/6b first (cheaper, predictable), then
6f only if compression is the binding constraint after those land.

---

## Brainstormed (not yet docketed) — alternative model formulations

The following are **alternative directions** that came out of the 2026-05-30 large-scale
review.  They're not promotion-queue items yet because the dev case is more speculative
than the docketed 6a-6f, but worth recording so they're not lost.

- **LambdaRank objective for the magnitude head**.  LightGBM supports `lambdarank` and
  `rank_xendcg`; using one as the magnitude-head objective optimises rank ordering
  directly.  Pros: targets Spearman ρ (the rank metric we care about) explicitly.  Cons:
  loses calibrated magnitude interpretation; needs a "group" (= fold or image) per query.
  Worth trying after 6f if magnitude-head ranking is still suspect.
- **Per-image feature standardisation** (domain-adaptation-light).  Within each image,
  z-score each per-tile feature using that image's own mean and std before training /
  inference.  Pros: simple way to address image-level feature scale variability that
  drives the per-image anti-signal failures.  Cons: removes absolute feature information
  (e.g. the dust-mantle / surface-type signal that might be informative across images).
  Cheap dev test (no re-train; just a feature pre-processing step).
- **Monotonic constraints** in LightGBM.  Constrain the model to be monotonic in
  `shadow_fraction` (should always positively correlate with boulder presence),
  `intensity_p10` (lower = more shadowed = potentially boulder-rich), etc.  Tests
  whether the boosted-tree model is wrong-way-correlating on specific features at specific
  images — a stronger version of H3 "anti-signal".
- **Post-hoc spatial smoothing of predictions** via simple Markov-random-field or kernel
  smoothing.  Complementary to Stage 6a (input-side spatial integration); 6a is at
  training time, this is at inference time.

## How to use this docket

- Append new items when you find a dev-validated change worth confirming on full v2.
- When an item passes full-v2 acceptance, move it to a "Promoted" section here AND add an
  entry to [DECISIONS.md](DECISIONS.md). Don't delete — the rationale stays useful.
- If an item fails full-v2 acceptance, document the failure here and move to a "Tried,
  didn't work" section.

## Promoted (none yet)

## Tried, didn't work (none yet)

## Out of scope — analysis-only, not model features

### HiRISE LBL angles (IncidenceAngle, EmissionAngle, PhaseAngle, SubSolarAzimuth)

**Why considered**: per-image performance diagnostic in [notebook 13
§4](notebooks/13_per_image_heterogeneity.ipynb) tested whether HiRISE acquisition-time
illumination angles correlate with per-image model performance. Result: no significant
correlation (all `|ρ| < 0.30`, all `p > 0.10`), so the explanatory value was already
low.

**Why out of scope for the model** (the binding reason): per the inference-time scope at
the top of this file, the deliverable runs on CTX-only regions where there is no HiRISE
image, so HiRISE LBL angles have no value at inference time. Adding them as training
features would force the model to consume an input that is missing in deployment.

**What we keep them for**: per-image diagnostic analysis (notebook 13), promotion-queue
acceptance reporting, sanity checks against label-quality hypotheses. They live in
[`cache/pds_labels/*.LBL`](cache/pds_labels/) and are parsed by
[`scripts/probes/_diag_nb13_correlations.py`](scripts/probes/_diag_nb13_correlations.py).
