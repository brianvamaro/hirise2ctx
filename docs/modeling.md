# Modeling — Methods

> Paper-Methods style description of the modelling stage on top of the
> packaged train/test dataset produced by the data pipeline. Companion
> to [`modeling_results.md`](modeling_results.md) (results /
> discussion) and to [`methods.md`](methods.md) (data-pipeline
> methods). Written 2026-06-02 at project wrap-up.
>
> Code for everything described below lives under
> [`src/modeling/`](../src/modeling/); the per-stage promotion log is in
> [`PROMOTION_QUEUE.md`](../PROMOTION_QUEUE.md).

---

## 1. Overview

The modelling stage takes the packaged tile-level training data from
the pipeline ([`methods.md` §8](methods.md), `dataset/packaged/` and
`dataset_v2/packaged/`) and learns a function from per-tile CTX
features to per-tile boulder abundance. The deliverable shape is a
LightGBM regressor and a LightGBM classifier, evaluated under
leave-image-out (LOIO) cross-validation and a within-image quadrant CV
sanity check, with three diagnostic refinements layered on (CNN
baseline, spatial-context neighbour features, CTX-source illumination
gate). Two cohort versions are reported: **v1** (9-image priority10,
sparse labels, retired) and **v2** (38-image vClaire, dense labels,
go-forward).

The modelling work is staged so that each model improvement is a
separable, testable variant rather than a monolithic re-train. The
guiding constraint is that **all model features must be derivable from
CTX alone at inference time**: any feature requiring a co-located
HiRISE image is *analysis-only* (used in compositional analysis;
excluded as a model input). This is what makes the trained model
deployable to CTX-only regions where no HiRISE coverage exists.

---

## 2. Targets

Two task framings are run in parallel, on the same packaged data:

- **Regression**: predict `fractional_area` (boulder area / tile area)
  directly. Three loss families to handle the heavy zero-inflation:
  - `log1p_huber`: log1p target transform + Huber loss
  - `tweedie`: native LightGBM Tweedie compound Poisson – Gamma loss
    (`power = 1.5`)
  - `two_stage`: hurdle model — a presence head (binary classifier
    on `fa > 0`) multiplied by a magnitude head (regressor on
    positives only)
- **Classification**: predict a binary boulder-rich label at three
  configurable thresholds:
  - `bc_ge_1`: `boulder_count >= 1` (any boulder visible)
  - `fa_gt_1e-3`: `fractional_area > 1e-3` (some boulder coverage)
  - `fa_gt_1e-2`: `fractional_area > 1e-2` (boulder-rich tile)

All target specifications are frozen as immutable
[`src/modeling/binary_target.py`](../src/modeling/binary_target.py)
`BinaryTarget` dataclasses so the same threshold is used across the
sweep, training, and evaluation code.

A late dev-only target reformulation
([`modeling_results.md §11.6`](modeling_results.md))
tested `log_boulder_count` and confirmed that the operational lift
comes from *count of distinct detection events*, not from
*area*: PR-AUC 0.526 (`fractional_area`) vs 0.640 (`boulder_count`)
on the dev within-image scheme. `boulder_count` was promoted to P2 on
the candidate queue and used as the dev/Stage-6a target column
thereafter.

---

## 3. Features

Per-tile features are computed at four tile scales {S=8, S=16, S=32,
S=64} corresponding to CTX-mosaic-aligned tile sizes of
{40, 80, 160, 320 m}. The feature families, set in
[`config.yaml`](../config.yaml) under `features:`, are:

- `intensity_stats` — per-tile mean / std / min / max / p10 / p50 / p90
  on the CTX raster after coverage masking.
- `glcm` — gray-level co-occurrence matrix descriptors (contrast,
  homogeneity, energy, correlation, ASM, dissimilarity) at three
  distance offsets, computed per-scale.
- `gradient` — Sobel gradient magnitude statistics (mean, std, p90) and
  direction-distribution descriptors (circular variance).
- `shadow_fraction` — fraction of in-tile CTX pixels with DN below a
  per-image shadow-DN threshold; `shadow_fraction_strict` at a tighter
  threshold; `bright_cap_fraction` at the high-DN end.
- `lbp` — local binary pattern 8-bin histograms.
- `lacunarity` — multiscale lacunarity on the binarised shadow mask.
- `subtile_variance` — variance of sub-tile means within each tile (a
  rough texture-roughness proxy).
- `canny_edges` — edge density + orientation entropy from a Canny
  filter.

Full per-feature documentation is in
[`notebooks/08_features_explained.ipynb`](../notebooks/08_features_explained.ipynb).
A separate per-tile `shadow_fraction` join is used in the Stage 7
compositional analysis as a tile-level shadow filter
([`compositional.md §3.5`](compositional.md)).

### 3.1 Stage 6a — spatial-context neighbour features (dev-only)

A neighbourhood-aggregation extension to the feature set
([`src/spatial_features.py`](../src/spatial_features.py)): for each
tile, compute `nbr_<mean|max|std>_<feature>` over a configurable
stencil (3×3 or 5×5) on the per-(ObsId, scale_idx) `(ti, tj)` grid,
NaN-aware to exclude image-edge gaps and Stage-4 eligibility gaps.
Variants are emitted as separate parquet families in
`dataset_v2_dev/features_nbr*/` and repackaged through the standard
split pipeline.

The Stage 6a dev sweep
([`modeling_results.md §12`](modeling_results.md)) found the **5×5
stencil at S=32 clears the strict promotion bar** (Δ Spearman ρ +0.053,
Δ PR-AUC +0.053) but only at that scale; the S=64 variants regress.
Promotion to full v2 was deferred pending Stage 6b results.

### 3.2 Stage 6b — CTX-source illumination per-tile features (dev-only)

For each tile, [`src/ctx_source_illumination.py`](../src/ctx_source_illumination.py)
joins the [Murray Lab CTX mosaic](https://doi.org/10.1029/2024EA003555)
SeamMap shapefile for the covering 4° × 4° tile and emits 7 columns
(`ctx_incidence_mean`, `ctx_incidence_std`, `ctx_emission_mean`,
`ctx_phase_mean`, `ctx_subsolar_az_mean`, `ctx_n_sources`,
`ctx_dominant_source_fraction`). Each HiRISE footprint is typically
covered by 4 – 46 different CTX source images, so the per-tile
illumination geometry is non-trivially heterogeneous.

The Stage 6b full-v2 LOIO sweep
([`modeling_results.md §13`](modeling_results.md)) **falsified the H3
"oblique CTX angle = mis-read shadows" hypothesis** (`ctx_incidence`
shows no significant correlation with per-image performance) but
**empirically validated the alternative H3'** that
*CTX-source heterogeneity* (`mean_n_sources`, `std_ctx_incidence`,
`dominant_source_fraction`) predicts per-image model reliability at p
< 0.05 across all operational metrics. This is the [Dickson 2024
seam-artefact prediction](https://doi.org/10.1029/2024EA003555)
showing up empirically in our model. Stage 6b net-effect on cohort
mean PR-AUC is +0.017 — below the +0.03 promotion bar but pointing at
a real structural effect.

### 3.3 Stage 6c — image-level reliability gate (dev + full-v2)

Stage 6b's per-tile features create bimodal per-image lift (anti-signal
images win, other images regress). Stage 6c moves the same features
up one level: train an *image-level* logistic / ridge / LightGBM
classifier on the 38-image dataset whose per-image baseline LOIO
PR-AUC is known, then apply at inference as a per-image confidence
gate.

The strict promotion criterion (retained-image mean PR-AUC ≥ 0.65
AND retained-tile fraction ≥ 70 % AND retained-set normalised lift ≥
+0.10) **fails across all 20 (gate × cutoff) combinations** tested;
the structural ceiling is that bad images carry disproportionate tile
counts, so dropping enough of them costs more than 30 % of the tiles
every time.

A softer "pooled-global Strategy B" variant *down-weights* each
held-out image's predictions by `(1 − p_bad_image)` from the gate
classifier rather than dropping the image; this is rank-invariant
within an image but changes the global pooled ranking across all
38 folds' tiles. The v1 ridge gate under Strategy B delivers **+0.056
pooled-global PR-AUC**, soft PASS. Stage 6c is recorded as
`◐ DEV-PARTIAL` in [`PROMOTION_QUEUE.md`](../PROMOTION_QUEUE.md).

---

## 4. Cross-validation design

Per-fold splits are produced by [`src/dataset.py`](../src/dataset.py)
into immutable JSON manifests under `dataset/packaged/{scheme}/`. Two
schemes are used:

- **`loio_*fold` (leave-image-out)** — each fold holds out all tiles
  from one image. v1: 9 folds (one empty-truth specificity-only fold);
  v2: 38 folds. This is the headline scheme because the eventual
  scientific use is generalisation to unseen CTX regions.
- **`within_image_4fold`** — each image is split into 4 spatial
  quadrants; each fold holds out one quadrant from one image, trains
  on the other 3 quadrants. v1: 8 × 4 = 32 folds; v2: 38 × 4 = 152
  folds. This is the diagnostic isolator: it removes per-image
  transfer from the problem entirely, so within-image-AUC vs LOIO-AUC
  is a clean test of whether the binding constraint is per-image
  generalisation or per-tile signal.

A multi-scale coherence invariant is enforced in the within-image
split: every S=8 tile must land in the same quadrant as its S=64
parent. Tested in
[`tests/test_within_image_split.py`](../tests/test_within_image_split.py).

---

## 5. Variants and training

Each LightGBM variant is a self-contained class in
[`src/modeling/gbm.py`](../src/modeling/gbm.py):

| Variant | Class | Notes |
|---|---|---|
| `lightgbm_log1p_huber` | `LightGBMLog1pHuber` | log1p target + Huber loss |
| `lightgbm_tweedie` | `LightGBMTweedie` | Tweedie loss, `power=1.5` |
| `lightgbm_two_stage` | `LightGBMTwoStage` | hurdle: presence × magnitude |
| `lightgbm_two_stage_balanced` | `LightGBMTwoStageBalanced` | presence head with `is_unbalance=False` |
| `lightgbm_two_stage_weighted` | `LightGBMTwoStageWeighted` | magnitude head with `sample_weight=y_pos` |
| `lightgbm_two_stage_gamma` | `LightGBMTwoStageGamma` | magnitude head with `objective='gamma'` |
| `lightgbm_two_stage_combined` | `LightGBMTwoStageCombined` | all three balanced/weighted/gamma |
| `lightgbm_classification` | `LightGBMClassification` | auto `scale_pos_weight = neg/pos`, returns probabilities |

Defaults are 400 boosting rounds, `learning_rate=0.05`, `num_leaves=63`,
early stopping after 40 rounds, with all `*_seed` fields fixed and
`deterministic=True` so the sweep is reproducible to LightGBM's
documented determinism guarantees. Per-fold model artefacts (booster,
predictions, metrics) are written under
`models/<variant>/<config_hash>/scale_S{n}/`.

A small CNN baseline (`SmallCNN`,
[`src/modeling/cnn.py`](../src/modeling/cnn.py)) operates on the
Stage 4b context patches (P=32, 64, or 128) at S=32 and S=64. Both a
regression head (`log1p+Huber`) and a classification head
(`BCEWithLogitsLoss(pos_weight=neg/pos)`) are implemented. The
classification head fixes the v1-era below-chance collapse but does
not beat the GBM on dev
([`modeling_results.md §10.1`](modeling_results.md)).

---

## 6. Evaluation

Metrics are computed by
[`src/modeling/evaluate.py`](../src/modeling/evaluate.py) per-fold and
aggregated. Both regression and classification share the same metric
suite:

- **Spearman rho** — rank correlation between prediction and truth.
  Primary regression metric.
- **Presence AUC** — ROC-AUC of `pred > 0` vs `truth > 0`. Sensitive
  to ranking but uniform-prior, often misleading on rare-positive
  targets.
- **PR-AUC** — area under precision-recall curve. Primary
  classification metric on rare-positive targets.
- **Normalised lift @ top-K** — `precision_top_K / base_rate`, with K
  set to the 5th percentile of predicted scores. Operational metric for
  "how much better than random is the top-ranked 5 % of the held-out
  set?"
- **Precision @ 5 %, Recall @ 5 %** — at the same top-K operating
  point.
- **Expected calibration error (ECE)** — per-decile calibration of
  classifier probabilities.
- **Per-image distribution stats** — min / median / max / std of
  per-fold AUC across the cohort, plus per-fold-vs-`BoulderLabel`
  scatter. Important because the cross-image mean buries strong
  per-image bimodality
  ([`modeling_results.md §11.4 / §11.7`](modeling_results.md)).

---

## 7. Reproducibility

Every numerical value in
[`modeling_results.md`](modeling_results.md) derives from one of:

- A LightGBM sweep with a recorded `config_hash` under
  `models/_sweep*/` (regression, binary, within-image, Stage 6a, Stage
  6b).
- A CNN run under `models/cnn_*` or `models/_sweep_smallcnn/`.
- A Stage 6c gate probe under `cache/stage6c/` with the per-fold gate
  predictions persisted.

All sweep scripts (`scripts/sweep.py`, `scripts/sweep_binary.py`,
`scripts/sweep_within_image.py`, `scripts/probes/_sweep_stage6a.py`)
take the packaged dataset directory as the only required input,
write all artefacts under timestamp-keyed directories, and are
re-runnable bit-for-bit (modulo LightGBM determinism settings).
Re-running `python scripts/sweep.py --dataset-dir dataset_v2/packaged/loio_38fold`
against the unchanged packaged data reproduces the v2 LOIO numbers.

The packaged dataset itself is reproducible from
[`config.yaml`](../config.yaml) + the BoulderNet detection shapefiles
via the pipeline described in [`methods.md`](methods.md).
