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

**Source data.** The modelling step consumes two pieces of upstream
data:

- **Boulder polygons** — meter-scale boulder detections on HiRISE
  imagery (~0.25 m/px native resolution). These were produced by a
  YOLO-based updated version of the BoulderNet boulder detector
  ([Amaro et al., 2026](https://doi.org/10.1029/2024JE008769);
  the original Mask R-CNN BoulderNet is
  [Prieur et al., 2023](https://doi.org/10.1029/2023JE008013)),
  one polygon per detected boulder. The polygons provide the
  **truth labels** the model is trained to predict.
- **HiRISE imagery** — the same HiRISE panchromatic scenes
  ([McEwen et al., 2007](https://doi.org/10.1029/2005JE002605))
  the boulder polygons were detected on, decimated to ~5 m/px and
  used only as the *moving image* in the HiRISE → CTX
  co-registration step (next paragraph). Not a model feature: the
  trained model must run on CTX alone at inference time, where no
  HiRISE image exists. Listed as a source because the dataset
  cannot be built without it, even though it does not directly feed
  the model.
- **CTX imagery** — the
  [Murray Lab global CTX mosaic](https://doi.org/10.1029/2024EA003555)
  ([Dickson, 2024](https://doi.org/10.1029/2024EA003555)), a
  ~5 m/px panchromatic image of Mars assembled from many MRO
  Context Camera ([Malin et al., 2007](https://doi.org/10.1029/2006JE002808))
  source images. The CTX mosaic provides the **inputs** from which
  per-tile texture features are computed.

The end-to-end pipeline that turns these two inputs into a tile-level
training dataset is documented at higher detail in
[`methods.md`](methods.md); the next two paragraphs sketch the parts
relevant to the model.

**HiRISE → CTX co-registration.** The boulder polygons live in
HiRISE's native coordinate system (0.25 m/px); to use them as per-tile
labels on the 5 m/px CTX grid they must be aligned. The pipeline
reprojects the polygons onto a common Mars-equirectangular coordinate
system shared with the CTX mosaic, decimates the HiRISE imagery to
~5 m/px to match CTX scale, runs a block-median phase-correlation
routine to estimate a per-image translation shift between the
decimated HiRISE and the corresponding CTX window, and applies the
shift to the polygon coordinates. The result is boulder polygons in
CTX-aligned coordinates. The same per-image shift is applied
uniformly across all tiles in that image; we do not attempt per-tile
registration.

**Per-tile label generation.** Tiles are defined as integer blocks of
CTX pixels (e.g. 64 × 64 CTX pixels = 320 m × 320 m at the S=64
scale used here) anchored to the Murray Lab mosaic's pixel origin —
so the tile grid is reproducible across images by construction. For
each tile the pipeline computes the number of CTX-aligned boulder
polygons that overlap it (`boulder_count`) and the fraction of tile
area covered by polygon footprints (`fractional_area`). Tiles whose
HiRISE coverage mask is incomplete (i.e. the tile partially falls
outside the HiRISE image footprint) are excluded from training and
evaluation, so every label is computed against the same boulder
detector's full coverage of that tile.

![BoulderNet polygon detections overlaid on CTX for two exemplar images](../reports/figures/modeling_slim_boulders_on_ctx.png)

*Figure 1. BoulderNet detections overlaid on the CTX window for two
exemplar held-out images, to make the input data concrete. Each red
dot is the centroid of one detected boulder polygon (over 300 000 on
the left, 138 000 on the right); dense clusters appear saturated red,
sparser regions show through to CTX grey. The 5 m/px CTX surface
context — craters, terrain texture — is visible underneath. Both
images contain many boulders across most of the HiRISE footprint;
the slim model's task is to recover the spatial distribution of those
boulders from the CTX-derived per-tile features alone.*

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
aligned to the Murray Lab CTX mosaic grid). Two constraints set the
scale: tiles must be large enough to suppress per-pixel CTX noise
(at 320 m each tile aggregates over ~4 000 CTX pixels, smoothing the
high-frequency variation that would otherwise dominate the
boulder-related signal), and tiles must be small enough to deliver
a meaningful spatial-resolution improvement over the thermal-IR
rock-abundance maps that are the standard alternative for Mars
surface-rock estimation ([Nowicki & Christensen, 2007](https://doi.org/10.1029/2006JE002798)),
which provide rock-abundance estimates at kilometre scales rather
than the few-hundred-metre scale this pipeline targets. 320 m
satisfies both bounds and is the scale we evaluate.

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
no HiRISE coverage exists), so generalisation to an unseen image
measures the right thing.

**Reported metrics.**

- **Pooled Spearman ρ** between predicted and true `boulder_count`,
  computed across all held-out tiles pooled across the 36 LOIO
  evaluations (~33 000 tiles). This is the headline number.
- **Per-image Spearman ρ** computed within each of the 36 held-out
  images (one number per image, since each LOIO fold holds out exactly
  one image) — used to characterise cross-image variance.
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

### 4.2 Per-image variance

Spearman ρ computed within each held-out image (one number per
LOIO-held-out image, 36 images total):

| | per-image ρ |
|---|---:|
| mean | +0.151 |
| median | +0.130 |
| min | -0.378 |
| max | +0.684 |
| std | 0.216 |

The mean per-image ρ is positive at ~3.6 standard errors above zero —
the same conclusion as the pooled result, reached on a different
aggregation. The distribution shows substantial cross-image variance:
some held-out images reach ρ > +0.5, while a few are at or below
zero, reflecting the per-image bimodality described next.

### 4.3 Per-image classification at the boulder-rich threshold

For each held-out image that contains both boulder-rich
(`fractional area ≥ 1 %`) and boulder-poor tiles, we report the AUC
of the model's predicted count as a discriminator. This is the
operationally meaningful evaluation: "could the model flag boulder-
rich tiles in this specific image for follow-up?"

![Per-image AUC at boulder-rich threshold](../reports/figures/modeling_slim_per_image_auc.png)

*Figure 2. Per-image AUC at the boulder-rich threshold
(`fractional area ≥ 1%`), one bar per held-out image, sorted from
worst on the left to best on the right. Red = anti-signal (AUC < 0.5);
grey = chance-band (0.5 ≤ AUC < 0.7); green = usable on that image
(AUC ≥ 0.7). The distribution is bimodal — a small minority of images
sit clearly above and below the usable / anti-signal thresholds, with
the majority in the noisy chance band.*

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

![Binary boulder-rich tiles highlighted on CTX: truth vs slim model for two exemplar images](../reports/figures/modeling_slim_good_vs_bad.png)

*Figure 3. Boulder-rich tiles highlighted on CTX for two contrasting
held-out images; boulder-poor tiles are kept transparent so the CTX
surface context (craters, terrain) shows through. The left panel of
each row shows truth — green tiles have `fractional area ≥ 1%`. The
right panel shows the slim model's call: green is the top-K tiles by
predicted boulder count, where K is set to match the count of
truth-rich tiles in that image, so both panels show the same number
of "rich" tiles by construction. The question the figure asks is
then whether the model picks the same tiles as the truth threshold.*

***Top row** — `ESP_053989_2260`, a "good" case (per-image AUC 0.88,
Spearman ρ +0.63): the model's green pattern matches the truth's
green pattern closely (476 of 523 rich-tile calls agree).*

***Bottom row** — `ESP_046328_2180`, an "anti-signal" case (AUC 0.34,
ρ −0.38): the model places its green predictions in the *opposite*
region from the truth's green — only 20 of 129 rich-tile calls agree,
well below the ~30% agreement a random ranking would produce at this
base rate. The two big craters anchor the geological context: the
truth puts boulder-rich tiles between and above them, while the
model's predictions cluster around the upper rim and miss the actual
distribution.*

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
| Figures | `reports/figures/modeling_slim_boulders_on_ctx.png`, `reports/figures/modeling_slim_per_image_auc.png`, `reports/figures/modeling_slim_good_vs_bad.png` |
| Figure builders | [`scripts/probes/_modeling_slim_figures.py`](../scripts/probes/_modeling_slim_figures.py), [`scripts/probes/_modeling_slim_panels.py`](../scripts/probes/_modeling_slim_panels.py), [`scripts/probes/_modeling_slim_boulders_overlay.py`](../scripts/probes/_modeling_slim_boulders_overlay.py) |

Re-run via:

```powershell
& "C:/Users/brian/anaconda3/Scripts/conda.exe" run --no-capture-output -n geospatial `
    python -u scripts/run_modeling_slim.py
```

Runtime ~2 min on a CPU-only laptop.
