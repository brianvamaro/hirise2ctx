# Modeling (slim) — 5-feature CTX rock-abundance predictor

> A simplified, reportable LightGBM model for predicting per-tile
> meter-scale boulder abundance from CTX texture features. This document
> describes the model as it is used in the project writeup; the deeper
> implementation discussion lives in [`modeling.md`](modeling.md) and
> [`modeling_results.md`](modeling_results.md).

---

## Bottom line

Using 5 physically motivated CTX-texture features (shadow fraction +
surface-roughness summaries), a LightGBM regressor trained under
leave-image-out cross-validation on 36 HiRISE-labelled images produces
a pooled Spearman ρ = **+0.275** between predicted and true per-tile
boulder counts (~33 000 held-out tiles, p ≪ 1e-50). The signal is
small but statistically robust, the model is keying on the physically
expected features, and per-image performance at the "boulder-rich"
threshold is **bimodal**: about 14 % of held-out images reach AUC
≥ 0.70 (usable on those images) while a similar fraction sits below
chance. The overall conclusion: the per-tile CTX texture signal at
5 m/px supports useful relative ranking of boulder abundance but does
not support reliable identification of boulder-rich tiles at the
cohort-aggregate level.

---

## 1. Question

Can a model trained on CTX texture features predict per-tile
meter-scale boulder abundance well enough to extend HiRISE-derived
rock-abundance maps to CTX-only regions of Mars? The deliverable
constraint is that all model inputs must be derivable from CTX alone
at inference time — HiRISE coverage is sparse (under 5 % of the
surface), while CTX is near-global at ~5 m/px. A working predictor
would let HiRISE-quality boulder-abundance maps be extrapolated
across the parts of Mars no HiRISE image has covered.

---

## 2. Data

**Cohort.** 36 of 38 HiRISE images from the v2 cohort. The 2 manifest
images with `unknown` boulder-density label (`ESP_017355_2260` and
`ESP_076499_1160`) were excluded because they are geographic-diversity
picks rather than part of the boulder-rich/poor cohort the model is
trained to predict. The remaining 36 images cluster at ~40 – 46°N on
the eastern Chryse / western Arabia margins.

**Inputs (per CTX tile, 320 m on a side).** Five features, each a
summary statistic computed from the CTX pixel values within the tile:

| Feature | What it measures |
|---|---|
| `shadow_fraction` | fraction of in-tile CTX pixels darker than the per-image shadow threshold — a direct boulder-shadow proxy under oblique sun |
| `shadow_fraction_strict` | the same statistic at a tighter shadow threshold (less noise-sensitive variant) |
| `bright_cap_fraction` | fraction of pixels brighter than the per-image bright threshold — captures saturation and high-albedo surface |
| `grad_mag_std` | Sobel-gradient-magnitude standard deviation — captures sub-tile surface roughness |
| `intensity_std` | per-tile pixel-value standard deviation — captures sub-tile contrast |

Two physically motivated mechanisms are represented:

- **Shadow patterns.** Boulders under oblique illumination cast small
  shadows; the fraction of dark pixels per tile correlates with
  boulder count. The HiRISE colour documentation and standard
  photoclinometry literature both treat shadow fraction as a direct
  proxy for surface-roughness elements.
- **Sub-tile texture roughness.** Boulder fields produce small-scale
  pixel-value variability; gradient-magnitude and intensity standard
  deviation capture this independently of shadow direction.

All five features are derivable from CTX alone — no HiRISE-side
information is used.

**Target.** `boulder_count` — number of HiRISE-detected meter-scale
boulder polygons per tile. Boulder count was preferred over total
boulder area because CTX texture features respond more strongly to
*number* of texture events than to *total area*. For evaluating
per-image classification at the "boulder-rich" threshold, we use
`fractional boulder area ≥ 1 %` as the operationally meaningful cut.

**Tile size.** 320 m × 320 m (`S=64` — 64 CTX pixels on a side,
aligned to the Murray Lab CTX mosaic grid). This is the coarsest
scale at which the per-image label set provides enough positive tiles
to evaluate.

---

## 3. Methods

**Model.** LightGBM with a two-stage hurdle architecture: a binary
presence head (does this tile contain any boulders?) multiplied by a
magnitude head (how many, given some are present?). The presence
head uses class-balanced LightGBM defaults; the magnitude head uses
log1p + Huber loss on positives only. Default hyperparameters: 500
boosting rounds, learning rate 0.05, 63 leaves, early stopping after
50 rounds. No hyperparameter tuning — that is part of "simple."

**Cross-validation.** Leave-image-out (LOIO). For each of the 36
held-out images, the model is trained on the other 35 and predicts on
the held-out image's tiles. This protocol matches the eventual
scientific use of the model (predicting on geographic regions where
no HiRISE coverage exists), so per-fold generalisation measures the
right thing.

**Reported metrics.**

- **Pooled Spearman ρ** between predicted and true `boulder_count`,
  computed across all held-out tiles from all folds (~33 000 tiles).
  This is the headline number.
- **Per-fold Spearman ρ** for each of the 36 held-out images — used
  to characterise per-image variance.
- **Per-image AUC at the boulder-rich threshold** (`fractional area
  ≥ 1 %`), computed on each fold whose held-out image contains both
  rich and poor tiles. This characterises whether the model could be
  used to flag specific boulder-rich tiles for follow-up imaging on a
  per-image basis.

---

## 4. Results

### 4.1 The headline ranking signal

Pooled across 33 102 held-out tiles from 36 LOIO folds, the model
produces a Spearman rank correlation of:

> **ρ = +0.275** between predicted and true `boulder_count`,
> p ≪ 1e-50.

The signal is statistically unambiguous given the sample size but
modest in absolute terms (Cohen's convention calls |ρ| < 0.3 "small").
Tiles ranked higher by the model do contain more boulders on average,
which means the model output is usable as a *relative* abundance
ranking. The model is not a calibrated estimator — predicted counts
do not match true counts at the magnitude level — and the predicted
ranking is not strong enough to reliably identify individual
boulder-rich tiles.

### 4.2 Per-fold variance

Spearman ρ across the 36 LOIO folds:

| | per-fold ρ |
|---|---:|
| mean | +0.151 |
| median | +0.130 |
| min | -0.378 |
| max | +0.684 |
| std | 0.216 |

The mean per-fold ρ is positive at ~3.6 standard errors above zero —
the same conclusion as the pooled result reached on a different
aggregation. The per-fold distribution shows substantial variance:
some held-out images reach ρ > +0.5, while a few are at or below
zero, reflecting the per-image bimodality described next.

### 4.3 Per-image classification at the boulder-rich threshold

For each held-out image that contains both boulder-rich
(`fractional area ≥ 1 %`) and boulder-poor tiles, we report the AUC
of the model's predicted count as a discriminator. This is the
operationally meaningful evaluation: "could the model flag boulder-
rich tiles in this specific image for follow-up?"

![Per-image AUC at boulder-rich threshold](../reports/figures/modeling_slim_per_image_auc.png)

| | per-image AUC at fa ≥ 1 % |
|---|---:|
| median | 0.572 |
| max | 0.880 |
| min | 0.311 |
| fraction with AUC ≥ 0.70 ("usable") | 14 % |
| fraction with AUC < 0.50 ("anti-signal") | 26 % |

The distribution is **bimodal**: on a minority of the cohort the
model identifies boulder-rich tiles reliably (top performers reach
AUC 0.88), while on a comparable minority the model is at or below
chance. The remaining majority sits in a noisy band near 0.5–0.7.
The cohort-aggregate AUC is therefore not a meaningful single
number — the model works on some images and not others, and reporting
only an aggregate would hide the bimodality.

---

## 5. Limitations

- **Boulder-rich classification is not tractable at the cohort
  aggregate.** Per-image AUC is bimodal; reporting a cohort-mean AUC
  would dishonestly imply uniform performance. The model can be used
  on the subset of images where it works (the ones with AUC > 0.70)
  but cannot, from the prediction alone, indicate whether a given
  image is one where it works.
- **Geographic concentration.** The 36-image cohort clusters
  geographically at ~40 – 46°N on the eastern Chryse / western Arabia
  margins. Cohort-level generalisation claims apply to dusty
  equatorial-band terrain; performance on highland, polar, or fresh-
  ejecta terrain is untested.
- **Calibration.** The predicted counts are not calibrated to true
  counts. The output is useful as a relative ranking, not as a
  quantitative boulder-density estimate.
- **CTX resolution.** The 5 m/px input resolution caps the
  information available per tile. Meter-scale boulders occupy a small
  fraction of any pixel; the model is keying on shadow + roughness
  *summaries* rather than on direct boulder detection.

---

## 6. Conclusions

The expected outcome at project start was a per-tile boulder-abundance
predictor good enough to extend HiRISE-derived rock-abundance to
CTX-only regions. The achieved outcome falls short of that as a
stand-alone boulder-rich classifier — at the operationally meaningful
threshold the model is bimodal per-image and the cohort-aggregate
performance is not a usable headline. What works is the *continuous
ranking* (pooled Spearman ρ = +0.275): tiles the model ranks higher do
contain more boulders, useful as an input to downstream science that
can absorb a small per-tile ranking signal — regional abundance maps,
two-stage frameworks, or prioritisation pipelines for HiRISE follow-up
on a per-image basis.

The model keys on the physically expected features (shadow fraction
and surface roughness), so the small signal is consistent with a real
causal mechanism rather than a per-image confound. The honest
interpretation is that the per-tile CTX texture signal at 5 m/px is
the binding constraint — boulders are smaller than a CTX pixel, and
the model is constructing a coarse-resolution proxy rather than
detecting boulders directly. Moving beyond this would need inputs
outside the CTX texture family: thermal channels from THEMIS,
coarser-than-tile spatial context, or higher-resolution CTX-equivalent
imagery.

---

## 7. Artefacts

| Artefact | Path |
|---|---|
| Runner | [`scripts/run_modeling_slim.py`](../scripts/run_modeling_slim.py) |
| Figure builder | [`scripts/probes/_modeling_slim_figures.py`](../scripts/probes/_modeling_slim_figures.py) |
| Per-tile predictions | `dataset_v2/modeling_slim_predictions.parquet` (33 102 rows; gitignored) |
| Per-fold summary | `dataset_v2/modeling_slim_summary.parquet` (37 rows = 36 folds + 1 pooled row; gitignored) |
| Figure | `reports/figures/modeling_slim_per_image_auc.png` |

Re-run via:

```powershell
& "C:/Users/brian/anaconda3/Scripts/conda.exe" run --no-capture-output -n geospatial `
    python -u scripts/run_modeling_slim.py
```

Runtime ~2 min on a CPU-only laptop.
