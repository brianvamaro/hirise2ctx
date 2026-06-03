# Modeling — rock-abundance prediction on CTX

> Paper-style writeup of the modelling stage on top of the packaged
> train/test dataset produced by the data pipeline.
> [`methods.md`](methods.md) covers the data-pipeline methods (Stages
> 0–5); this document covers the modelling stage (Stages 5b – 6c) and
> states the headline result + conclusion. The deep-dive results
> discussion (fold-by-fold numbers, full diagnostic tables, every
> variant × scale cell) is in
> [`modeling_results.md`](modeling_results.md). Written 2026-06-02 at
> project wrap-up; the per-stage promotion log is in
> [`PROMOTION_QUEUE.md`](../PROMOTION_QUEUE.md).

---

## 1. Question and motivation

**Can a model trained on CTX texture features predict per-tile
meter-scale boulder abundance reliably enough to extend HiRISE-derived
rock-abundance maps to CTX-only regions of Mars?**

The motivation is geographic coverage. The upstream BoulderNet
detector produces meter-scale boulder polygons on HiRISE imagery at
~0.25 m/px — superb spatial resolution, but the cumulative HiRISE
coverage of Mars is less than 5 % of the surface. CTX gives near-global
coverage at ~5 m/px ([Malin 2007](https://doi.org/10.1029/2006JE002808),
[Dickson 2024](https://doi.org/10.1029/2024EA003555)). If per-tile CTX
texture features (shadow fraction, intensity statistics, GLCM and other
texture descriptors at 40 – 320 m tile sizes) carry enough signal to
predict HiRISE-derived per-tile boulder abundance with usable accuracy,
the trained model becomes a tool for producing a global rock-abundance
map at CTX resolution — and a downstream input to landing-site safety
analysis, geomorphological mapping, and the kind of compositional
analysis that the separate Stage 7 thread
([`compositional.md`](compositional.md)) operates on top of.

The guiding hard constraint is therefore that **all model features must
be derivable from CTX alone at inference time**. Any feature requiring
a co-located HiRISE image is analysis-only and excluded as a model
input — this includes the HiRISE colour features used in
[`compositional.md`](compositional.md) and the HiRISE-LBL illumination
angles. Without this constraint the model could not be deployed to
CTX-only regions, which is the whole point of the exercise.

The expected outcome at project start was a per-tile abundance
predictor with leave-image-out generalisation good enough to claim
"this CTX tile probably contains a boulder field." The actual outcome
— headline results in §8, full conclusions including the
expected-vs-achieved framing in §10 — is more nuanced: a usable
abundance ranker with a per-tile signal floor at ~5 m/px CTX texture,
falling short of a usable rare-event classifier within the CTX-only
inference-time constraint.

---

## 2. Datasets

| Dataset | Source | Range / extent | Resolution | Use |
|---|---|---|---|---|
| Murray Lab CTX mosaic | [Dickson 2018](https://repository.gatech.edu/server/api/core/bitstreams/d2671fb1-4a1d-4b9b-ad8f-c7c5d3aafda2/content) / [2024](https://doi.org/10.1029/2024EA003555) | global Mars, 4° × 4° tiles | 5 m/px | model input rasters |
| BoulderNet detection polygons | upstream BoulderNet run on HiRISE RED.JP2 | 38 priority HiRISE images (v2 cohort); 9 priority images (v1, retired) | meter-scale polygons, ~10⁴ – 10⁵ per image | label source (truth) |
| HiRISE RED (decimated) | NASA PDS | per-observation | downsampled to ~5 m/px for co-registration; not a model input | co-registration of label polygons onto CTX grid |
| Per-tile CTX feature parquets | computed in this pipeline (Stage 4b) | 0.64 M tiles (v1) / 3.56 M tiles (v2) at scales S=8/16/32/64 | 40 m / 80 m / 160 m / 320 m per side | 52-column feature matrix per tile, train + test |
| Per-tile labels parquets | computed in this pipeline (Stage 4) | same tiles as features | derived `fractional_area`, `boulder_count`, binary thresholds | regression / classification targets |
| Stage 4b context patches | Stage 4b | 28 800 patches at S=32 / 6 600 at S=64 (v1) | raw CTX raster sub-tiles 32 × 32 or 64 × 64 px | CNN baseline input |
| Murray Lab CTX SeamMap | [Dickson 2024](https://doi.org/10.1029/2024EA003555) | global, 4° × 4° tiles | per-source vector geometry | Stage 6b CTX-source illumination features |

Cohort geographic range: ~40 – 46°N (eastern Chryse Planitia / western
Arabia margins) with one outlier at ~16°N, longitude ~0 – 20°E. The
geographic concentration is a deliberate cohort choice (boulder-rich
terrain selection), not a sampling accident; it does mean cohort-level
generalisation claims are scoped to dusty equatorial-band terrain.

Preprocessing/calibration is fully described in
[`methods.md`](methods.md): per-image SP1 CRS correction, sub-pixel
phase-correlation co-registration, CTX-pixel-anchored tile grid,
per-tile shadow-DN threshold from CTX intensity modal statistics, fold
manifest reproducibility (config hash + frozen JSON).

---

## 3. Targets

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

## 4. Features

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

## 5. Cross-validation design

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

## 6. Variants and training

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

## 7. Evaluation

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

## 8. Headline results

Full per-fold numbers and the figure dump are in
[`modeling_results.md`](modeling_results.md); this section pulls the
headline numbers + figures relevant to the rubric.

### 8.1 Per-tile presence-AUC ceiling

Across all variants × scales × cohorts × CV schemes, the per-tile
presence-AUC plateau sits at roughly **0.55 – 0.62**:

| Cohort | Scheme | Variant | Scale | Presence AUC |
|---|---|---|---:|---:|
| v1 (9 images) | LOIO | `lightgbm_two_stage` | S=64 | 0.568 ± 0.102 |
| v1 | Within-image quadrant | `lightgbm_two_stage` | S=64 | 0.578 |
| v2 (38 images) | LOIO | `lightgbm_two_stage` | S=64 | 0.579 |
| v2 | LOIO | `lightgbm_classification`, `bc_ge_1` | S=64 | 0.616 (n=26 single-class-excluded folds) |
| v2 | Within-image quadrant | `lightgbm_two_stage` | S=64 | 0.607 |

The Spearman rank-correlation signal lifts substantially v1 → v2
(driven by label density), but the presence-AUC ceiling rises only
modestly:

| Tile scale | v1 Spearman ρ | v2 Spearman ρ | v1 presence AUC | v2 presence AUC |
|---:|---:|---:|---:|---:|
| S=8  | -0.000 | **+0.096** | 0.508 | 0.559 |
| S=16 | +0.003 | **+0.127** | 0.515 | 0.568 |
| S=32 | +0.018 | **+0.125** | 0.520 | 0.573 |
| S=64 | +0.059 | **+0.169** | 0.568 | 0.579 |

![v2 sweep Spearman bar chart](../reports/figures/10_sweep_spearman_bar.png)

### 8.2 The model is keying on the physically expected feature

GBM split-gain feature importance places **`shadow_fraction` first at
every scale** (12 – 16 % of total split gain), at both v1 and v2. The
remaining top-10 is dominated by intensity percentiles at fine scales
and texture-family features (LBP, edge density, gradient std) at
coarse scales — both consistent with sub-tile shadow patterns +
texture roughness being the causal mechanism. The model is **not**
keying on per-image brightness offsets (which the LOIO protocol
specifically tests against).

![Tweedie S=8 feature importance](../reports/figures/10_feature_importance_tweedie_S8.png)

### 8.3 Class-stratified bimodality (per-image)

Per-image performance is strongly bimodal at the operational threshold
`fa_gt_1e-2` (S=64): median AUC 0.61, max **0.91** (`ESP_042964_2160`,
lift 5.4× over base rate), min 0.40 ("anti-signal" images where the
top-ranked predictions are systematically negative). About 7 of 38 v2
images perform usably (AUC > 0.70), ~4 are anti-signal (AUC < 0.50),
the rest sit near chance. **The cross-image mean buries strong
individual-image performance**; reporting only the cohort-aggregate
AUC understates what the model can do on the subset of images where
it works.

### 8.4 The diagnostic isolator: within-image ≈ LOIO

The decisive diagnostic for "is the binding constraint per-image
generalisation or per-tile signal?" is the within-image quadrant CV.
At every variant × scale × cohort, the within-image AUC sits within
sampling noise of the LOIO AUC (every 95 % bootstrap CI on the mean
delta brackets zero; every Wilcoxon p > 0.05). Training and testing
on the *same image*, with per-image transfer entirely removed from
the problem, does **not** lift the per-tile AUC above the LOIO
ceiling. The binding constraint is therefore **per-tile signal at
5 m/px CTX texture**, not data quantity or per-image transfer.

### 8.5 Stage 6 model improvements

Three model-improvement chunks were tested against the v2 LOIO
baseline:

| Stage | What | Verdict |
|---|---|---|
| 6a | Spatial-context neighbour features (3 × 3 / 5 × 5 stencil aggregation) | 5 × 5 @ S=32 dev-only PASS (Δ Spearman +0.05, Δ PR-AUC +0.05); other variants FAIL. Full-v2 promotion deferred. |
| 6b | CTX-source illumination angles (mean + std + n_sources from Murray Lab SeamMap) | FAIL strict (cohort net flat) but **H3 mechanism falsified** and **alternative H3'** (CTX-source heterogeneity) **empirically validated** at p < 0.05 across per-image metrics. Stage 6e [Dickson 2024](https://doi.org/10.1029/2024EA003555) seam-artefact prediction matches the data. |
| 6c | Image-level reliability gate using Stage 6b features | FAIL strict (no gate × cutoff combination clears the 70 %-tile-retention budget); **SOFT PASS** under Strategy B down-weighting at **+0.056 pooled-global PR-AUC** with no data dropped. |

Stage 6c soft PASS is the operational deliverable: a per-image
confidence weight at inference time that lifts the cohort PR-AUC
without retraining.

---

## 9. Uncertainty, limitations, and validation assessment

### 9.1 Validation checks performed

1. **Leave-image-out cross-validation** as the headline protocol — the
   right protocol because deployment is to unseen CTX regions.
2. **Within-image quadrant CV as a diagnostic isolator** — independent
   measurement of whether the cap is signal or generalisation. Three
   independent target framings (regression, binary classification,
   within-image CV) converge on the same ~0.55 – 0.62 ceiling, the
   strongest available evidence that the binding constraint is signal,
   not data quantity.
3. **Sign tests across 12 (variant × scale) configurations** for the
   v1 sweep — pooled sign-test p = 0.0002 against the
   no-skill null on presence AUC > 0.5; 10/12 cells above zero on
   Spearman (sign-test p = 0.019). Confirms the small signal is not
   sampling noise.
4. **Feature-importance physical-plausibility check** — `shadow_fraction`
   is the top-ranked feature at every scale + cohort, which is the
   physically expected causal cue (boulders cast shadows under oblique
   illumination). Rules out the failure mode where the model latches
   onto a per-image confound that LOIO would not detect.
5. **Class-stratified reporting** — held-out `BoulderLabel` partitions
   reveal the per-image bimodality that the cohort mean buries. Reported
   alongside aggregate metrics so the conclusion is not
   pooling-artefact.
6. **Calibration assessment (ECE)** — classifier probabilities have
   ECE 0.10 – 0.20 across the three binary targets, indicating the
   ranking signal is informative but probability outputs require
   isotonic / Platt recalibration before downstream use.
7. **Determinism guarantees** — all `*_seed` fields fixed,
   `deterministic=True` in [`src/modeling/gbm.py`](../src/modeling/gbm.py).
   Sweeps re-run bit-for-bit; numerical claims in
   [`modeling_results.md`](modeling_results.md) are reproducible to
   LightGBM's documented guarantees.

### 9.2 Limitations

- **Per-tile signal floor at 5 m/px CTX.** Three independent target
  framings put the per-tile presence AUC ceiling at ≈ 0.55 – 0.62 on
  v2. The model is a usable abundance *ranker* (Spearman ρ +0.17 at
  S=64) but not a usable presence *classifier* — top-K lift is 1.07 –
  1.43, modestly above the random baseline. Calibrated quantitative
  abundance prediction is not in reach at this CTX resolution with
  this feature family.
- **Per-image bimodality.** The cohort mean masks strong individual-
  image performance (median per-image AUC 0.61, max 0.91, min 0.40 on
  `fa_gt_1e-2`/S=64). Some images carry usable signal; others
  anti-signal. Stage 6c soft PASS partially compensates via per-image
  down-weighting but does not eliminate the anti-signal cases.
- **Mosaic-seam confound (Stage 6e mechanism).** Stage 6b empirically
  validated that **CTX-source heterogeneity** (a HiRISE footprint
  stitched from many CTX source images with varying acquisition
  geometry) predicts per-image model reliability at p < 0.05 — i.e.
  some of the per-image bimodality is a [Dickson 2024](https://doi.org/10.1029/2024EA003555)
  seam artefact, not a true geological boundary. Reduces the ceiling
  but is partially recoverable via the Stage 6c gate.
- **Cohort geographic concentration.** The 38 v2 images cluster
  geographically (~40 – 46°N, eastern Chryse / western Arabia
  margins). Cohort-level generalisation claims apply to dusty
  equatorial-band terrain; performance on highland, polar, or
  fresh-ejecta terrain is untested.
- **Inference-time scope constraint.** Colour features (used in the
  separate compositional thread,
  [`compositional.md`](compositional.md)) and HiRISE LBL angles cannot
  be model inputs because they require co-located HiRISE coverage,
  which defeats the CTX-only deployment goal. This is a deliberate
  scope choice, not a limitation of the analysis, but it does cap
  what features the model can use.
- **Label completeness on v1 was a confound.** v1's 98 %-zero target
  was partly a *missed-boulder* artefact rather than a true signal
  floor; v2 dense labels lifted the Spearman 3 – 10× at every scale
  (§9.2 of [`modeling_results.md`](modeling_results.md)). The
  remaining presence AUC ceiling at 0.55 – 0.62 is the more honest
  measurement after this confound was controlled for.
- **CNN baseline did not unlock new signal.** v1 CNN performed below
  chance (loss-design problem); v2 `SmallCNNClassifier` with
  `BCEWithLogitsLoss(pos_weight)` fixes the collapse but does not
  beat the GBM at any patch size, including a wider 640 m context
  patch. Suggests the 52 hand-crafted features have already extracted
  what 5 m/px CTX texture can discriminate; the ceiling is not a
  feature-engineering gap that a CNN closes.

---

## 10. Conclusions

### 10.1 Principal findings

1. **A real per-tile signal exists.** Pooled sign tests across 12
   (variant × scale) v1 configurations give p = 0.0002 on presence
   AUC > 0.5, with the model keying on `shadow_fraction` (the
   physically expected feature) at every scale.
2. **The signal is bounded by a per-tile texture floor at 5 m/px.**
   Three independent target framings — regression, binary
   classification, and within-image quadrant CV — converge on a
   presence-AUC ceiling of ≈ 0.55 – 0.62 on v2. Adding data quantity
   tightens error bars but does not move the mean.
3. **The model is a usable abundance ranker, not a usable
   classifier.** Spearman ρ +0.17 at v2 S=64 supports per-tile
   ranking for follow-up prioritisation; top-K lift 1.07 – 1.43 falls
   short of a usable rare-event detector.
4. **Per-image performance is bimodal.** Median per-image AUC 0.61
   with max 0.91 and min 0.40 on `fa_gt_1e-2`/S=64 — strong individual
   results buried by cohort averaging. Stage 6c soft PASS at +0.056
   pooled-global PR-AUC partially recovers this through per-image
   confidence weighting.
5. **Mosaic seam artefacts are a real confound.** Stage 6b
   empirically validated that CTX-source heterogeneity predicts
   per-image model reliability at p < 0.05, matching the [Dickson
   2024](https://doi.org/10.1029/2024EA003555) seam prediction —
   identifying a mechanism for some of the per-image bimodality but
   not a fully recoverable one.

### 10.2 Was the expected outcome achieved?

The expected outcome at project start was a per-tile rock-abundance
predictor good enough to extend HiRISE-derived rock-abundance to
CTX-only regions. **The achieved outcome falls short of that as a
calibrated predictor but lands solidly as an abundance ranker, with a
specific identified ceiling.**

What was achieved:

- A reproducible, sweep-driven LightGBM pipeline producing per-fold
  artefacts under `models/_sweep*/` for every variant × scale × CV
  scheme tested.
- An empirically calibrated estimate of where the per-tile CTX
  texture signal ceiling sits (≈ 0.55 – 0.62 presence AUC), with a
  clean diagnostic (within-image ≈ LOIO) showing the ceiling is a
  signal floor rather than a generalisation problem.
- A characterised + mechanistically-explained anti-signal mode
  (Stage 6b/6c) with a partial operational remedy (Stage 6c soft PASS
  Strategy B).
- A documented + empirically tested set of model-improvement
  candidates (Stage 6a/6b/6c/6d/6e/6f) with verdicts on each.

What was not achieved:

- A model that can produce **calibrated** per-tile boulder abundance
  for downstream geological use without per-image recalibration.
- A model whose top-K predictions can be used directly to flag CTX
  regions for HiRISE follow-up (top-K lift is modestly above 1.0 but
  not strong enough).
- A within-CTX-feature-family resolution of the bimodality.

The closing diagnostic narrows the path forward to **inputs beyond CTX
texture**: thermal/spectral channels (THEMIS rock abundance map),
coarser-than-tile spatial priors, or higher-resolution CTX-equivalent
inputs (HiRISE decimated). The current modelling work documents the
floor and isolates the mechanism — the natural extension is to bring
in non-texture signal that breaks through it. The instructor's
extra-goal compositional analysis (separately delivered in
[`compositional.md`](compositional.md)) is the natural science layer
on top of this engineering deliverable: the modelling thread answered
"how well can CTX predict where boulders are?" and the compositional
thread answered "what are the boulders made of, and how do they differ
from their surroundings?"

---

## 11. Reproducibility

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
