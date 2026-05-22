# PLAN — Week 3 modeling

**Status:** plan. No code or files modified by this draft.
**Inputs assumed in place:** Stage 4 label parquets at `dataset/labels/{ObsId}.parquet`, Stage 4b per-tile feature parquets (planned separately at `dataset/features/{ObsId}.parquet`), and Stage 4b context patches at `dataset/context_patches/{ObsId}/...` (`features.context_patch.enabled: true` required — the CNN baseline depends on these).
**Out of scope here:** Stage 4b texture-feature implementation (see PLAN_Stage4b.md), Stage 5 packaging beyond what modeling needs (see PLAN_Stage5.md), THEMIS validation, compositional analysis.

## Principle: preserve CTX resolution

Median boulder area is 3.7 m² (DECISIONS.md 2026-05-20), which is ≈0.15 of one
CTX pixel at native 5 m/px. **We are already trying to predict something
smaller than CTX's native resolving power.** Any architecture choice that
throws away CTX information further pushes the prediction problem past
hopeless.

What this rules in / out:

- The 40/80/160/320 m tile ladder (`labeling.tile_sizes_px = [8, 16, 32, 64]`)
  is about **label-aggregation extent** — how much area's worth of boulders
  one label cell summarizes. CTX itself stays at native 5 m/px throughout.
- Stage 4b features and CNN patches both operate on native-resolution CTX.
  No spatial downsampling.
- Section §6 Option C ("train at coarsest scale and disaggregate") is rejected
  outright, not merely deprioritized.
- Intensity quantization (e.g. GLCM `levels`) is a separate-but-related
  question — explicitly preserve enough intensity bins to retain texture
  signal. PLAN_Stage4b.md is the place this gets pinned down.
- The CNN baseline (§4) is non-optional precisely because it consumes
  native-resolution patches without the information loss of hand-crafted
  features.

---

## 1. Problem framing

We are learning a function from CTX-derived per-tile descriptors to a heavily zero-inflated continuous target, with two structural constraints that together rule out most off-the-shelf workflows:

**Target distribution (DECISIONS.md 2026-05-23, 488,554 finest-grid tiles, no filters):**

| Statistic | Value |
|---|---:|
| Fraction of tiles with `fractional_area == 0` | 97.88% |
| Mean | 2.2e-4 |
| Median | 0 |
| P99 | 6.25e-3 |
| Maximum (densest tile in the manifest) | 0.269 |

Concretely: about 478,300 of 488,554 finest tiles are exact zeros, and even the single densest tile in the entire dataset has only 27% of its area covered by boulders. The non-zero tail is sparse (≈10,250 tiles, ≈2.12%) and right-skewed. A naive MSE fit on `fractional_area` will be dominated by the zero mass: a model that always predicts zero achieves an RMSE on the order of the std (≈1.8e-3), already below P99. This is the canonical signature of a process that needs either (a) explicit zero handling, (b) a variance-stabilizing transform, or (c) a distribution-family loss that admits a point mass at zero (Tweedie).

**Group structure (DECISIONS.md 2026-05-23, Stage 4 sweep):** 9 usable ObsIds (ESP_057469_2215 dropped for tile-straddle; ESP_065711_1545 is a deliberate all-zero-ground-truth image), with per-image tile counts at the finest scale ranging from 25,221 (the empty-truth image) to 76,030 (ESP_055714_2270). Tiles within an image are spatially correlated and frequently share CTX texture context, so per-tile random splits would leak. The leave-image-out rule from CLAUDE.md §9 is hard, not advisory.

**Dataset size — two coexisting truths:**

- **In rows**, the dataset is medium-large: 488,554 finest-grid tiles, ~10,250 of which carry non-zero abundance (the positive supervision pool). This is enough to support models with thousands to low-tens-of-thousands of effective parameters — a small CNN (~50k params) is plausible, not reckless.
- **In groups**, the dataset is small: 9 ObsIds today. **The group count, not the row count, is the binding constraint** on what generalizes. A model can fit 478k zeros and 10k positives precisely while still failing on a held-out image because it learned per-image artifacts.

**What 9 groups (growing slowly) imply:**

- **Leave-one-image-out CV (LOIO) is the natural unit.** 9 folds, each trained on 8 images, evaluated on 1. With the empty-truth image (ESP_065711_1545), one fold is degenerate (its truth is all zeros — useful as a specificity check, but not a regression-quality check).
- **Hyperparameter search should be light-touch, not absent.** Nested LOIO would be 9 × 8 = 72 fits — feasible at this dataset size (each LightGBM fit is seconds to a minute). Recommend a small coarse grid (3–5 candidate configurations per model class) evaluated by mean ± std over the 9 LOIO folds, with the chosen configuration documented in the run snapshot. Avoid Bayesian / random search until the dataset grows.
- **Class-of-image stratification matters more than per-tile stratification.** Among the 9 usable images: 4–5 boulder-rich, 2 boulder-poor, 2 unknown, 1 empty-truth. Fold composition is small enough that the empirical distribution of high-fractional-area tiles per fold will swing substantially from one held-out image to the next; **per-fold metric variance is structurally large** and must be reported as such (mean ± std, not a single number).
- **Generalization is the binding constraint, not capacity.** Per-image artifacts (detector confidence drift between images, residual co-registration bias, illumination geometry) are the failure mode to design against. Mitigate via LOIO CV explicitly tracking train-vs-held-out gap, regularization tuned to that gap, and (for the CNN) data augmentation that targets per-image confounds (brightness jitter, contrast jitter, rotation/flip).

This framing is consistent with CLAUDE.md §10's sketch (gradient boosting baseline; consider `log1p` or two-stage; LOIO CV; stratified abundance-bin metrics).

---

## 2. Baseline architecture — gradient-boosted trees

### Why GBM first

Gradient-boosted decision trees are the obvious baseline because:

1. They handle the **tabular** feature space (intensity stats, GLCM, gradient, shadow fraction — Stage 4b output) natively.
2. They tolerate **heterogeneous feature scales** and don't need normalization, so we can iterate on features in Stage 4b without re-tuning the model.
3. They can be evaluated quickly per fold (seconds to minutes per LOIO fit at this dataset size), which is necessary because we will be running 9-fold CV repeatedly.
4. They produce **feature importance and partial-dependence plots** that are diagnostically useful for understanding what CTX texture actually predicts boulder abundance, which is the scientific question behind the engineering one.

### Library recommendation: **LightGBM**

| Library | Pros | Cons |
|---|---|---|
| **LightGBM** | Native Tweedie objective (`objective="tweedie"`, `tweedie_variance_power` ∈ (1,2)); native quantile loss; categorical-feature support without one-hot; fast on CPU; small dependency footprint; mature group-aware CV support via `lgb.cv` with custom folds. | Windows wheel install occasionally finicky; `pip install lightgbm` into the `geospatial` env should work but verify before training begins. |
| XGBoost | Tweedie objective (`reg:tweedie`); widely cited. | Slower than LightGBM at this dataset size; more verbose API; tighter wheel/version coupling. |
| sklearn `HistGradientBoostingRegressor` | Pure-Python install; identical mental model to other sklearn estimators; sklearn-compatible groups via `GroupKFold`. | No Tweedie loss in the regressor (Poisson and squared error only as of 1.5); no quantile loss in the histogram boosting variant. Would force log1p + MSE, sacrificing the cleanest zero-inflation handling option. |

**Recommendation:** **LightGBM** as the default, primarily because the Tweedie objective is the cleanest expression of "continuous target with a point mass at zero" we'll find in standard libraries (Tweedie with `1 < p < 2` is mathematically a compound Poisson-Gamma; it matches our zero-inflated continuous structure almost exactly). XGBoost is the immediate fallback if the LightGBM Windows install fights us. sklearn `HistGradientBoosting` is the third fallback (forces log1p + MSE).

### Inputs and the (X, y, group) join

Model-ready data is a join of two parquet families on `(obs_id, scale_idx, ti, tj)`:

- `dataset/labels/{ObsId}.parquet` — Stage 4 output. Carries the target columns (`fractional_area`, `binary_by_area`, `binary_by_count`, `count_density`, `boulder_area`, `boulder_count`, `tile_area`) plus tile-bound provenance.
- `dataset/features/{ObsId}.parquet` — **Stage 4b output (planned separately, not in this plan's scope)**. Expected to carry per-tile CTX descriptors: intensity stats, GLCM features, gradient stats, shadow-fraction proxy. The exact feature list is a Stage 4b decision; modeling code treats it as an opaque parquet schema discovered at load time.

The join key `(obs_id, scale_idx, ti, tj)` is the natural primary key in both parquets per DATA_DICTIONARY.md §Stage 4. The group key for CV is `obs_id`. Per-image label and feature parquets are concatenated rather than monolithic, so adding images later requires no re-write.

### Single-stage form

One GBM regressor predicts `fractional_area` (or a transform of it — see §3) from all features simultaneously. Loss: Tweedie with `variance_power` tuned on a holdout (the standard range for zero-inflated continuous data is 1.3–1.7). All zero tiles contribute to the fit through the point-mass term of the Tweedie likelihood; the model learns "what does a zero tile look like" implicitly.

### Two-stage (hurdle) form

Two models:

1. **Presence classifier:** binary classification of `fractional_area > 0` (or `binary_by_area`, or a small-positive threshold — open question, see §10). Uses the same features. LightGBM binary classifier with `class_weight` or `is_unbalance=True` because positives are ~2% of the dataset.
2. **Magnitude regressor:** regression of `fractional_area` (or `log(fractional_area)`) trained **only on positive tiles**. About 10,000 positives total — small enough that this model needs heavy regularization, but rich enough to learn at least the broad shape of the tail.

Final prediction: `P(positive) * E[magnitude | positive]`. Note: this **is not** what the literature calls a "true" hurdle model in the maximum-likelihood sense — that would share parameters via a joint likelihood. The "predict-then-multiply" form is the engineering simplification commonly used with gradient boosting; it is consistent with the conditional decomposition `E[Y] = P(Y > 0) * E[Y | Y > 0]` but loses the elegance of a single joint fit.

**Trade-off:** the two-stage form natively respects the zero-inflated structure but doubles the model count, doubles the artifact footprint, and complicates calibration (the product of two reasonably-calibrated models is not necessarily a well-calibrated regression). Single-stage Tweedie is simpler and pricing-actuarial literature (where Tweedie is canonical for zero-inflated insurance claims) suggests it usually wins on RMSE; the two-stage form usually wins on Spearman because it cleanly separates the easy classification problem from the harder magnitude one.

**Recommendation:** **build both**, share a feature pipeline, run them through the same CV harness. The result then becomes one of the §10 open questions to surface to the user at execution time, decided empirically rather than philosophically.

---

## 3. Zero-inflation handling — methods survey and recommendation

For a continuous target with a heavy point mass at zero and a right-skewed positive tail, the standard families are:

| Approach | Mechanism | Library / loss | When it wins |
|---|---|---|---|
| **`log1p` + MSE / Huber** | Variance-stabilizing transform; predicts `log(1 + fractional_area)`; back-transform via `expm1`. | Any regressor. | Simple, well-tested, transparent. Doesn't model the point mass explicitly — zeros stay zeros (log1p(0)=0) and the model just sees a mass at zero in transformed space. Works well in practice as a baseline but is statistically inelegant. |
| **Tweedie regression** | Compound Poisson-Gamma likelihood; admits a point mass at zero and a continuous positive tail; controlled by `variance_power p ∈ (1, 2)`. | LightGBM `objective="tweedie"`; XGBoost `reg:tweedie`; sklearn `TweedieRegressor` (linear). | The textbook fit for this distribution. Canonical in actuarial pricing for the same reason it applies here. |
| **Two-stage / hurdle** | Decompose into `P(Y > 0)` and `E[Y | Y > 0]`. | Any two regressors. | Easy to debug because you can inspect each stage. Natural when the positive-vs-zero decision is qualitatively different from "how much". |
| **Zero-Inflated Poisson / NegBin** | Mixture of a Bernoulli (excess zero) component and a count distribution. | `statsmodels.discrete.count_model`. | Designed for counts, not continuous fractional area. Better fit for `boulder_count` than `fractional_area`. |
| **Quantile regression on the tail** | Predict τ-th quantile of the conditional distribution. | LightGBM `objective="quantile", alpha=τ`. | When the deliverable is "probability of exceeding X% abundance" rather than a point estimate, this is more honest about uncertainty in the rare tail. Pair with the median (τ=0.5) for a robust central tendency. |

**Recommendation:** Make **Tweedie regression with LightGBM** the primary baseline, with **`log1p` + Huber** as a sanity-check shadow model that should track Tweedie roughly. The two-stage form is the second primary model, because it isolates the "where are there boulders at all?" question (the scientifically interesting binary signal — see also the count-vs-area binary disagreement, §11). Quantile regression at `τ ∈ {0.5, 0.9, 0.99}` is a third optional model emitted from the same harness, useful both for uncertainty and for the long-horizon deliverable of "predict abundance percentile across the CTX mosaic."

**Established references for the methodology** (training-knowledge citations, not freshly retrieved — verify versions at implementation time):

- Tweedie regression as a unified treatment of zero-inflated continuous data: Smyth, G. K. (1996), "Regression analysis of quantity data with exact zeros." Used widely in actuarial pricing (e.g. Yang et al. 2018, "Insurance premium prediction via gradient tree-boosted Tweedie compound Poisson models").
- Hurdle and two-part models: Mullahy, J. (1986), "Specification and testing of some modified count data models," *J. Econometrics*. Updated treatments in Cameron and Trivedi (2013), *Regression Analysis of Count Data*, Cambridge, even though our target is continuous fractional area rather than counts.
- LightGBM Tweedie objective: documented in the LightGBM project's parameters page; `tweedie_variance_power` is the controlling hyperparameter.
- Rock-abundance modeling priors on Mars: Christensen (1986) and Nowicki & Christensen (2007) for THEMIS-derived rock-abundance maps (planned for validation, not training input). HiRISE-resolution boulder detection literature (Golombek et al., Wagstaff et al., Hood et al., BoulderNet itself) generally treats boulder counts/area as the response variable. None of those works explicitly use Tweedie / hurdle frameworks; this is a methodological contribution our work can claim.

---

## 4. CNN on context patches — parallel non-optional baseline

The CNN is **not** an "optional advanced" track; it's the natural complement
to the GBM. The two baselines test different hypotheses:

| Question | GBM answers | CNN answers |
|---|---|---|
| What texture descriptors predict boulder abundance? | Among the ones we hand-crafted in Stage 4b, which? (feature importance) | What does CTX texture *look like* in boulder-rich vs boulder-poor tiles, even where we didn't think to measure? (learned-feature visualization) |
| Performance ceiling at CTX-native resolution | Bounded by what Stage 4b summarizes | Bounded by what's recoverable from raw pixels |
| Diagnostic power | "GLCM contrast matters more than shadow fraction in your data" | "There is / isn't additional signal beyond your hand-crafted features" |

The CNN is also the cleanest expression of the §0 *preserve CTX resolution*
principle: it consumes raw native-resolution pixels with no quantization,
no statistic-summary information loss, no choice of distance/angle parameters.
If the hand-crafted GBM beats the CNN, that's strong evidence Stage 4b
captured what's there. If the CNN beats the GBM, that's strong evidence
there's additional CTX signal we haven't measured yet — directly motivating
Stage 4c features or a more capable second model.

### Inputs

Per-tile context patches written by Stage 4b at
`dataset/context_patches/{ObsId}/S{patch_size_px}/{ti}_{tj}.npy` — uint8
arrays of native-resolution CTX, no spatial resampling. Two sizes,
both enabled in Week 3:

| Patch size | Spatial extent | Use |
|---|---|---|
| 32×32 px | 160 m | Matches the S=32-px label tile. CNN sees the exact same pixels the label was computed from, plus a small surrounding context band (patches are *centered* on the tile so 32×32 covers a 32×32 tile exactly; for finer-scale labels at S=8 or S=16, the patch extends beyond the tile and provides context). |
| 64×64 px | 320 m | Matches S=64. Wider context window for the coarser scales. |

The patch size used at inference must match training; this is a Stage 4b /
modeling-config coupling worth recording in provenance.

### Architecture — small CNN, designed against per-image artifacts

```
Input:  1 × 32 × 32  (or 1 × 64 × 64)  uint8 → float32 / 255
Block1: Conv 3x3, 16ch  → BN → ReLU → MaxPool 2x2     ( → 16 × 16 × 16 )
Block2: Conv 3x3, 32ch  → BN → ReLU → MaxPool 2x2     ( → 32 ×  8 ×  8 )
Block3: Conv 3x3, 32ch  → BN → ReLU → GlobalAvgPool   ( → 32 )
Head:   FC 32 → 64 → ReLU → Dropout(0.3) → FC 64 → 1
Output: scalar fractional_area (or two scalars for two-stage)
```

Total parameters: ~30k for 32×32 input, ~35k for 64×64. Small enough to not
overfit 10k positives + 478k informative zeros across 8 training images,
big enough to learn meaningful texture features beyond what 4–6 GLCM
properties summarize. Global average pooling instead of a flatten-then-FC
keeps the parameter count from blowing up.

**BatchNorm before ReLU** stabilizes training and acts as implicit
regularization against per-image brightness/contrast drift — important
because per-image artifacts are the LOIO failure mode.

### Loss + target

Same options as the GBM, plumbed through:
- Single-stage: regression on `log1p(fractional_area)` with Huber loss.
  (No clean Tweedie loss in PyTorch out of the box; the variance-stabilized
  log1p+Huber form is the practical analogue.)
- Two-stage: separate presence-classifier head (BCE loss on `fractional_area > threshold`)
  and magnitude-regression head (Huber loss on `log(fractional_area)` for positives only).
  Either two model objects or one model with two heads + a multi-task loss —
  recommend the simpler two-object form initially.

### Data augmentation — design against per-image artifacts

Per-batch augmentation pipeline:
- Random horizontal + vertical flip (4× sample expansion, free).
- Random 90° rotations (additional 4× — total 16× with flips).
- Random brightness jitter ±15% of the per-tile intensity range.
- Random contrast jitter, factor in [0.85, 1.15].
- Optional: small Gaussian additive noise (σ=2 on the 0-255 scale) to break
  per-image noise signatures.

Crucially: the brightness/contrast/noise augmentations target the specific
confounds we'd expect to leak across CTX images (different illumination
geometry, atmospheric haze, sensor response drift between observations).
Geometric augmentations (flips, rotations) are appropriate because the
texture-to-abundance mapping should be rotation- and reflection-invariant
on bulk Mars surfaces — boulder fields don't have a preferred "up".

### Training protocol

- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4. Standard small-CNN defaults.
- **Batch size:** 256 (positives are 2% of any batch — large batch keeps a few
  positives per gradient step without explicit class-balanced sampling).
- **Alternative class-balanced sampling:** for the two-stage presence head,
  consider a 50/50 positive/negative sampler instead of random. Open
  question for execution time.
- **Early stopping:** primary criterion is validation loss on a single
  held-out image (not held-out tiles within the training images). Patience
  ~10 epochs. **This is the only correct early-stopping protocol given the
  LOIO constraint** — held-out-tiles-within-training-images would leak.
- **Epoch budget:** train for up to 100 epochs at the typical few-thousand-
  batches-per-epoch (488k tiles / 256 batch). ≈few minutes per LOIO fold
  on CPU; ≈seconds per fold on GPU. CPU is acceptable for Week 3 at this
  size.
- **LOIO outer loop:** same 9-fold protocol as the GBM. For each fold,
  train on 8 images, test on 1. Within each fold, the 8 training images
  split internally only for early-stopping validation — and that split is
  ALSO by image (use one of the 8 training images as the early-stopping
  monitor, rotate which one across folds).

### Framework — PyTorch

Use PyTorch. Reasons: standard deep-learning stack; the CNN architecture
above is ~50 lines of `nn.Module`; full training loop is well under 200
lines including LOIO; works without GPU. Avoid Lightning / Keras for a
single small CNN — too much framework overhead for the model size.

Pin `torch` version in the modeling extras of `pyproject.toml`.

### Risks specific to the CNN

- **Brittleness to per-image artifacts.** Larger than for GBM because the
  CNN can latch onto fine-grained image-level texture statistics that hand-
  crafted features wouldn't expose. Mitigation: aggressive augmentation
  (above), heavy BatchNorm, strict LOIO discipline.
- **Variance across LOIO folds.** Expect higher fold-to-fold variance than
  GBM because the CNN is more expressive and small training-set differences
  between folds will produce larger output differences. Mitigation: report
  per-fold metrics individually as well as aggregated.
- **Disk cost of context patches.** Stage 4b §6 estimated ~1 GB for both
  sizes across the dataset. At Week 3 dataset size this is fine; revisit
  if the manifest grows by an order of magnitude (the patch cost scales
  linearly with tile count).
- **GPU not assumed.** Training on CPU is the assumption. If a GPU becomes
  available, batch size can grow and epoch time drops, but no
  architectural change is needed.

### Why this beats "punt to later"

Building the CNN in Week 3 — alongside the GBM, sharing the same `evaluate.py`
harness and same LOIO protocol — costs maybe a day's extra implementation.
The diagnostic value of having both baselines side-by-side on the first
results table is high: it lets us answer "is the bottleneck features or
methodology?" immediately rather than after two phases. The original
draft's recommendation to defer underweighted this.

---

## 5. Cross-validation and evaluation methodology

### Splits

**Primary protocol: leave-one-image-out (LOIO).** 9 folds = 9 ObsIds = 9 train-test partitions. The empty-truth image (ESP_065711_1545) is included; its test fold acts as a specificity stress test ("does the model predict near-zero abundance on a truly empty scene?") rather than a regression-quality measurement.

**Stratification across folds:** with only 9 images, "stratified group K-fold" doesn't quite apply (each "stratum" has 1 group). Instead, when reporting fold-level metrics, **tag each fold with the image's BoulderLabel** (`Boulder rich` / `Boulder poor` / `unknown` / `empty`) and report metric tables grouped by tag. This makes fold-to-fold variance interpretable and exposes the "model works on rich but fails on poor" pattern if it exists.

**Implementation:** sklearn `GroupKFold(n_splits=9)` with `groups = df["obs_id"].factorize()[0]` gives the LOIO split for free; the wrapper in `src/dataset.py` should expose `iter_loio_folds()` returning `(train_idx, test_idx, held_out_obs_id)` tuples, where the held-out ObsId string is carried through to per-fold metric output for stratified reporting. The Stage 5 splitter (PLAN_Stage5.md) writes the same `loio_9fold` partition to `dataset/splits/`; modeling reads from there rather than re-deriving the split, so split definition is single-sourced.

### Per-fold metrics

We need metrics that are robust to the zero-inflated structure. **Report several**, declare one **primary**.

| Metric | Computed on | Role |
|---|---|---|
| **RMSE on `log1p(fractional_area)`** | All tiles in the held-out image. | Variance-stabilized aggregate error. Robust to a few large-tail tiles dominating. Primary candidate. |
| RMSE on raw `fractional_area` | All tiles. | Reference. Will be tiny (the trivial all-zero predictor scores ~1.8e-3); a model has to beat that meaningfully or the result is uninformative. |
| **Spearman ρ** between predicted and true `fractional_area` | All tiles. | Rank-stable to monotonic transforms. Robust to scaling errors. Excellent for "is the model getting the relative ordering right?" |
| **Per-abundance-bin RMSE** | Tiles bucketed by true abundance: `[0, 0]`, `(0, 1e-4]`, `(1e-4, 1e-3]`, `(1e-3, 1e-2]`, `(1e-2, 1)`. Bucket edges TBD; the 5-edge cut above is a starting point that maps roughly to "zero, low, medium, high, very high." | The CLAUDE.md §10 "not a single RMSE dominated by near-zero tiles" requirement. Highlights tail performance. |
| AUC of presence detection (`fractional_area > 0` vs `== 0`) | All tiles. | If we ship the two-stage model, this is the binary stage's metric directly. If single-stage, threshold the predicted abundance. |
| Calibration: mean predicted vs mean actual abundance, per held-out image. | Per-image aggregates. | Whether the model under- or over-predicts in absolute terms — important for the eventual mosaic-scale inference deliverable. |
| Per-image scatter plot of predicted vs true `fractional_area`, log-log | Per fold. | The visual companion to the metric table. Goes into the QA notebook. |

**Recommendation for primary metric:** **Spearman ρ on all-tiles**, reported as `mean ± std` across the 9 folds. Justification:

- Spearman is robust to the choice of target transform (we want it stable whether we train on raw `fractional_area`, `log1p(fractional_area)`, or via Tweedie — they should all produce similar rank orderings).
- Spearman is robust to the zero mass (it doesn't collapse to "predict zero everywhere is great").
- Spearman has a clear scientific interpretation: "does the model correctly rank tiles by abundance?" — which is exactly what the inference deliverable (a CTX-wide abundance map) cares about.

Secondary: per-abundance-bin RMSE table. This is where the §10 "stratified RMSE" requirement lives and is where model failure modes (over/under-prediction at specific abundance levels) become visible.

### Per-scale metric aggregation

When we have 4 scales (40, 80, 160, 320 m), each LOIO fold produces 4 sets of metrics. Report per-scale tables and also a "primary scale" choice (likely 40 m or 80 m — the smaller scales preserve most of the abundance information and are what an inference deployment would actually emit).

---

## 6. Per-scale modeling decisions

We have 4 scales (`tile_sizes_px = [8, 16, 32, 64]` → 40, 80, 160, 320 m). Three plausible per-scale architectures:

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A. One model per scale** | Train 4 independent GBMs, one per scale. | Each model sees a coherent target distribution. No cross-scale leakage. Simplest mental model. | 4× the artifacts. Doesn't exploit cross-scale information (a 40 m tile's coarser context is informative). |
| **B. Single model with scale as a feature** | One GBM, with `scale_idx` (or `tile_size_m`) as a categorical feature. Concatenate all 4 scales' tiles into a single training set. | One artifact. Cross-scale generalization "for free." | Target distribution varies hugely by scale (mean fractional area drops with tile size at the coarse end because dense local patches get averaged with surrounding zeros). The single model must learn a strong scale-conditional output mapping. Risk of underfitting at the rare-event scales. |
| **C. Train at coarsest, predict + downsample** | Fit on the 320 m grid (~6,587 tiles per the Stage 4 sweep table); for inference, predict and disaggregate to finer scales. | Tiny training set → fast iteration. Stable target (the 320 m mean is dominated by smoother bulk-abundance variation, less zero-inflation). | Loses the cross-image fine-scale signal entirely. Disaggregation is a deconvolution problem we can't solve from the abundance map alone. Defeats the purpose of having a fine grid in the first place. |

**Recommendation: Option A first.** Concretely: train 4 GBMs in the same harness, share features and hyperparameters, report all per-scale metrics. After the first round, examine cross-scale correlation in the residuals — if errors are strongly correlated across scales, Option B is worth revisiting (with `scale_idx` as a categorical feature and explicit per-scale weighting in the loss). Option C is parked unless wall-clock cost of the LOIO sweep becomes prohibitive, which is unlikely at this dataset size.

---

## 7. Inference target — CTX-mosaic-scale deployment (sketch)

Eventually the trained model has to run across the Murray Lab CTX mosaic where HiRISE coverage is absent — this is the project's whole point. Architecture sketch (not implementation; explored just enough to make sure the modeling artifacts don't paint us into a corner):

- **Unit of inference:** one Murray Lab 4°×4° tile at a time (~47,420 × 47,420 pixels at ~5 m/px, ~1.7 GB on disk). At 40 m tiles, one Murray Lab tile yields about 35 million tiles to predict. Naive whole-mosaic inference (~100+ tiles globally) would be ~3.5 billion tiles.
- **Pattern:** for each Murray Lab tile, compute the same Stage 4b feature parquet using a thin wrapper around `src/features.py` that operates on a CTX raster without needing labels (since there are no labels off-HiRISE). Run the trained model on the resulting feature parquet. Emit a per-tile parquet of predicted abundance + the original `(ti, tj, tile_size_px)` columns.
- **Memory budget:** per Murray Lab tile, feature extraction over a ~47k × 47k uint8 raster fits in a few GB if done in tiled passes (e.g., 4096×4096 blocks). LightGBM inference on a parquet with tens of millions of rows is straightforward (batched `predict`); GPU not required.
- **Idempotency:** prediction parquets should be content-addressed by `(config_hash, model_hash, murray_tile)` so re-running is a no-op when nothing changed and re-running on a single tile after a model update is one command.
- **Scope today:** `src/inference.py` is a stub for Week 3 — defines the I/O contract (input: CTX raster + Stage 4b feature parquet for an arbitrary CTX region; output: per-tile prediction parquet) but does not run the global sweep. The model artifacts and Stage 4b feature pipeline are the constraints to keep right.

What this rules in / out for modeling-side decisions:

- The training-time feature pipeline must be **deterministic given a CTX raster** (no leakage from labels, no per-image normalization that needs ground truth). If Stage 4b emits features that secretly depend on the label parquet, inference is broken from the start. **Flag this for Stage 4b review** (cross-check: PLAN_Stage4b.md §3.4's image-percentile shadow detector uses the image's own pixel distribution, not labels — consistent with this constraint).
- Model artifacts should be **portable** (LightGBM `Booster.save_model` → text file; pickled models are fine for local but not for archival).

---

## 8. Module + file layout

Proposed additions and small revisions:

```
src/
  dataset.py              # NEW: load + join labels + features parquets per ObsId,
                          # build (X, y, group) arrays, expose iter_loio_folds()
                          # and per-scale subsetting. Reads dataset/splits/ from Stage 5.
                          # Single entry point for all model code.
  models/
    __init__.py           # NEW
    base.py               # NEW: abstract model interface (fit, predict, save, load,
                          # plus a model_hash() for provenance)
    gbm.py                # NEW: single-stage LightGBM with Tweedie or log1p+MSE
    two_stage.py          # NEW: presence classifier + magnitude regressor wrapper.
                          # Internally composes two `gbm.py` models with a shared
                          # fit / predict surface.
    cnn.py                # NEW (stub for Week 3): small CNN on context patches.
                          # File exists, contains interface, not implemented.
  evaluate.py             # NEW: LOIO cross-validation runner; per-fold metric
                          # aggregation; per-fold prediction caching; metric table
                          # construction. Pure function of (model_factory, dataset).
  inference.py            # NEW: stub for off-HiRISE prediction across a CTX region.
                          # Defines I/O contract; full mosaic sweep is a later phase.
scripts/
  train_baseline.py       # NEW: command-line driver for an end-to-end LOIO CV run
                          # with the GBM baseline. Writes:
                          #   models/{model_name}/{config_hash}/fold_{obs_id}/booster.txt
                          #   models/{model_name}/{config_hash}/predictions.parquet
                          #   models/{model_name}/{config_hash}/metrics.json
                          #   models/{model_name}/{config_hash}/snapshot.yaml
  train_two_stage.py      # NEW: same harness, two-stage model variant.
notebooks/
  07_modeling_qa.ipynb    # NEW: per-fold predicted-vs-true scatter, residual plots,
                          # per-abundance-bin RMSE table, learning curves (train vs
                          # validation loss across boosting rounds), feature
                          # importance, per-image error breakdown.
                          # (Number may shift depending on Stage 4b/5 notebook ordering.)
models/                   # NEW: gitignored. Artifact root for trained models.
  {model_name}/{config_hash}/...
```

**Discussion of why this shape:**

- `src/dataset.py` is the **only** place that knows how to assemble a model-ready table from the parquet sources. Everything downstream (`gbm.py`, `two_stage.py`, `evaluate.py`, `inference.py`) consumes its outputs. This keeps the schema knowledge in one file and makes the next-image-onboarding diff small. Note: PLAN_Stage5.md proposes `src/dataset.py` for the splitter; these are the same file — Stage 5 lands the split-building functions, modeling adds the (X, y, group) join + iteration helpers.
- `src/models/base.py` defines an interface (a Protocol or ABC) so that the LOIO runner doesn't know whether it's training a GBM, a two-stage wrapper, or eventually a CNN. The runner calls `model.fit(X_train, y_train, group_train)`, `model.predict(X_test)`, `model.save(path)`.
- `src/evaluate.py` is **stateless** — given a dataset and a model factory, it returns predictions and metrics. This is what makes it easy to swap models later.
- `scripts/train_baseline.py` is the only thing a human runs. It loads the config, builds the dataset, chooses a model factory, runs the LOIO sweep, writes artifacts. All re-running happens at this level.

---

## 9. Provenance + reproducibility

### Config hashing

Every modeling artifact carries the same SHA256 config-hash treatment used elsewhere in the pipeline. The modeling config snapshot includes:

- The pipeline `config.yaml` (so the upstream label / feature parquets are pinned).
- A `modeling:` block in `config.yaml` (new) covering: model class name (`lightgbm_tweedie_single_stage`, `lightgbm_two_stage`, etc.), hyperparameters (learning rate, num leaves, num boost rounds, Tweedie variance power, early-stopping rounds), feature selection (an explicit list of feature-parquet columns to use; default = all), target transform (`identity` / `log1p` / `tweedie`), CV protocol (`leave_one_image_out`), random seed.
- The model artifact's own `model_hash()` (LightGBM booster string SHA256, computed after fit).

`config_hash` plus `model_hash` together uniquely identify a trained model.

### Cached predictions

Per-fold predictions cached as a single parquet at `models/{model_name}/{config_hash}/predictions.parquet` with columns:

| Column | Type |
|---|---|
| `obs_id` | str |
| `scale_idx` | int |
| `ti`, `tj` | int64 |
| `fold_held_out_obs_id` | str |
| `y_true` | float |
| `y_pred` | float |
| `y_pred_presence_prob` | float (two-stage only; null otherwise) |
| `config_hash` | str |
| `model_hash` | str |
| `predicted_at_iso` | str |

This is the eval-replay artifact. Re-computing metrics on these predictions is a few-second pandas operation; we never need to re-train just to re-aggregate.

### Per-fold metrics

`models/{model_name}/{config_hash}/metrics.json` carries the per-fold and aggregated metric table, plus the held-out-image BoulderLabel tag and per-abundance-bin breakdown.

### Reproducibility constraints

- Set explicit seeds in LightGBM (`seed`, `bagging_seed`, `feature_fraction_seed`, `data_random_seed`) and record them in the config snapshot.
- Pin LightGBM version in `pyproject.toml`'s `[project.optional-dependencies]` (a new `modeling` extra) so re-installs reproduce.
- The label and feature parquets each carry their own `config_hash` (Stage 4 + Stage 4b). The modeling config snapshot records the label/feature `config_hash` values it consumed, so a downstream re-train can detect upstream drift.

---

## 10. Open questions to surface via AskUserQuestion at execution time

The next session should not pre-decide these; surface each as a 2–4-option AskUserQuestion before writing the corresponding code.

1. **GBM library.** Options: (a) LightGBM with Tweedie objective [recommended]; (b) XGBoost with `reg:tweedie`; (c) sklearn `HistGradientBoostingRegressor` + `log1p` (pure-Python fallback, no Tweedie loss).
2. **Primary target transform.** Options: (a) Tweedie (`variance_power=1.5`, tuned later) [recommended]; (b) `log1p` + Huber loss; (c) two-stage hurdle (presence + magnitude); (d) ship all three and compare on the LOIO CV.
3. **Primary evaluation metric.** Options: (a) Spearman ρ, mean ± std across 9 folds [recommended]; (b) per-abundance-bin RMSE table with a single "primary bin"; (c) RMSE on `log1p(fractional_area)`.
4. **Per-scale architecture.** Options: (a) one model per scale [recommended]; (b) single model with `scale_idx` feature; (c) train at coarsest only and disaggregate.
5. **CNN positive/negative sampling strategy.** Options: (a) random sampling with batch size 256 [recommended]; (b) explicit 50/50 positive/negative balanced sampler; (c) hard-negative mining after first epoch. (The CNN itself is non-optional — see §4 — only the sampling strategy is open.)
6. **Positive-tile threshold for two-stage / binary classifier.** Options: (a) `fractional_area > 0` (strict zero-vs-nonzero); (b) `binary_by_area` with the current `area_threshold=0.005` (98.84% negative, 1.16% positive); (c) `binary_by_count` (highly imbalanced — only 0.04% positives); (d) a new threshold tuned to balance class size. **This is also the gateway to resolving the count-vs-area disagreement in DECISIONS.md 2026-05-23 (5,504 vs 2 disagreements).**
7. **Whether to apply Stage 3 co-registration shifts at training time.** Already done by Stage 4 by default per DECISIONS.md 2026-05-23, but modeling code should not silently override — surface as a confirmation rather than a decision.

---

## 11. Risks and uncertainties

In rough order of "most likely to bite":

1. **Small-group CV variance.** 9 folds is a small sample; per-fold metric variance will be large. A model that looks "10% better" might be inside the noise envelope. Mitigation: always report mean ± std; do paired-fold comparisons rather than headline-mean comparisons; treat any single-fold improvement as suggestive only.
2. **The 0.27 ceiling.** The maximum `fractional_area` in the entire dataset is 0.269 (DECISIONS.md 2026-05-23). Even on the densest finest-grid tile, 73% of the area is background. A regression target with this small a dynamic range and this much zero-inflation has a structural performance ceiling: even a perfect model will have a modest R² because the signal-to-noise budget is intrinsically small. Be honest about this in reporting. Spearman is the metric that least punishes this.
3. **Count-vs-area binary disagreement** (5,504 area-only positives vs 2 count-only positives at the placeholder thresholds — DECISIONS.md 2026-05-23). This is partly an artifact of the placeholder thresholds (`area_threshold=0.005`, `count_threshold=5`) being miscalibrated against each other, and partly a real signal that small-boulder regions and large-boulder regions are differently distributed. Tune thresholds explicitly during modeling design; don't take either binary column at face value. Flagged as open question §10.6.
4. **Confounding between scale and target.** Coarser scales have lower variance in `fractional_area` (averaging smooths). A single-model-with-scale-feature architecture (§6 Option B) risks the model learning "predict near-zero for large `tile_size_m`" rather than learning texture-to-abundance. Mitigation: §6 Option A (per-scale models) avoids this entirely.
5. **Per-image artifacts as features.** BoulderNet's detection threshold and per-image confidence calibration likely vary subtly between images (different illumination, atmospheric conditions, surface types). The model can pick up on per-image CTX texture signatures (e.g., overall image brightness, characteristic noise patterns) that correlate with the per-image boulder rate, which generalizes poorly to new images. Mitigation: LOIO CV explicitly tests for this; if held-out performance is much worse than within-image performance, this is the cause. Mitigation 2: avoid features that have per-image-constant values (Stage 4b feature design concern).
6. **ESP_065711_1545 (the empty-truth image)** has 25,221 finest tiles with `boulder_area = 0`. Including it in training is fine; including it in CV is fine but its fold metric is degenerate (true is all zero; metrics like Spearman are undefined). Mitigation: in the metric aggregation, mark this fold specially and report it as a separate specificity test ("does the model predict near-zero abundance on a known-empty scene?").
7. **Co-registration residual at the tile-edge scale.** Stage 3 shifts are typically 120–270 m. With 40 m tiles, that's 3–7 tile-widths of slop in the worst case. Stage 4 applies the shifts to polygons before rasterization (DECISIONS.md 2026-05-23), which should fix the systematic component. The remaining noise (non-rigid local distortion, the few-m sub-pixel residual) bleeds boulder edges across tile boundaries at the finest scale. This caps achievable accuracy at the 40 m scale. The 80–160 m scales are less affected and may be the natural primary scale.
8. **Stage 4b features that haven't been designed yet.** This modeling plan assumes Stage 4b emits a useful feature parquet. If the features are uninformative (e.g., GLCM at the chosen offset is dominated by noise rather than texture), the baseline ceiling will be very low regardless of model choice. Mitigation: include a "trivial features only" baseline (just intensity mean + std per tile) as a sanity check — if the full feature set doesn't beat that, the problem is upstream of modeling.
9. **Inference distribution shift.** Training images are clustered around 40–46°N (boulder-rich set) plus a few diversity picks. Off-HiRISE CTX mosaic prediction goes near-globally, including geological terrains the model has never seen. Predictions far from the training distribution are unreliable. The CNN tail of this is worse than the GBM tail. Mitigation, eventually: emit a per-tile distance-to-training-distribution flag with predictions, so downstream consumers can mask out unreliable regions. Out of scope for Week 3 but worth noting in the inference stub.

---

## TL;DR

The plan frames Week 3 as a tabular-regression problem with two hard constraints — heavy zero-inflation (97.88% zeros at the finest scale, 0.269 max) and a tiny number of groups (9 ObsIds) — that together rule out high-capacity models and force leave-one-image-out CV with explicit group-aware splits. It recommends LightGBM with the Tweedie objective as the primary baseline, a two-stage hurdle (presence classifier + magnitude regressor) as a parallel comparison, `log1p` + Huber as a sanity-check shadow, and quantile regression as an uncertainty-aware optional add-on; punts a small CNN on context patches to a follow-up phase while keeping the architectural seam (`src/models/cnn.py` stub, `features.context_patch` toggle, model-agnostic CV harness) so it can plug in later without rework. Primary metric: Spearman ρ across 9 LOIO folds, reported mean ± std, backed by a per-abundance-bin RMSE table and per-image calibration diagnostics. Per-scale architecture: one model per scale first. Module layout adds `src/dataset.py` (shared with Stage 5), `src/models/{base,gbm,two_stage,cnn}.py`, `src/evaluate.py`, `src/inference.py` (stub), and `scripts/train_baseline.py` plus a `notebooks/07_modeling_qa.ipynb`, all carrying the existing `config_hash` provenance treatment with cached per-fold predictions to make eval replayable without retraining. Seven open design questions (GBM library, target transform, primary metric, per-scale architecture, CNN-yes-or-no, positive-tile threshold for the binary stage which also resolves the count-vs-area binary disagreement from DECISIONS.md 2026-05-23, coreg-shift confirmation) are flagged for AskUserQuestion at execution time, and nine concrete risks are enumerated — including the 0.269-max structural ceiling, count-vs-area binary disagreement (5,504 vs 2), small-group CV variance, per-image artifact leakage, and the inference-time distribution shift onto un-HiRISE'd terrains.
