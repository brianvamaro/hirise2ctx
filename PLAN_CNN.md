# PLAN — W2: CNN on context patches (expanded)

**Status:** planned 2026-06-11; **Phase 1 grid EXECUTED + READ same day**
(DECISIONS.md 2026-06-11 Phase-1 entry; sweep `models/_sweep_cnn/20260611T220815Z`).
Outcome in brief: **H-B (photometric augmentation) REFUTED cohort-level** —
every augmented cell ≤ the no-aug floor; geometric augmentation actively
harmful (destroys the cohort-constant 142–186° sun-azimuth shadow prior);
**cell A (no-aug) passes the per-image gate vs Tier 1** (median paired
ΔAUC +0.066, p=0.016; Brian ruled median-of-paired-deltas binding), pooled
PR-AUC fails for all cells (cross-image mis-leveling, diagnosed). Free
follow-ups same session: **rank×scale fusion STRONG** (pooled PR-AUC 0.5932,
prec@5% 0.914 — see §5.0), AdaBN cohort-null but third independent rescue of
ESP_076499_1160. **3-seed update (2026-06-11 late): the single-seed gate
pass does NOT replicate** (median paired ΔAUC +0.066/+0.038/+0.005, p=0.016/
0.059/0.66 — per-image skill stable at median 0.69–0.71, score calibration
seed-unstable), **but the 3-seed ensemble + Tier-1 fusion passes both gates**
(F1(ens): pooled PR-AUC 0.5955 = +0.030, prec@5% 0.887; ensemble per-image
Δ median +0.052, p=0.0065; F3(ens) per-image Δ +0.058, p=0.0001). Recipe was
assembled post-hoc → **S=32 replication (§4.2b) is its held-out
confirmation**. Photometric-only cell still running. Literature
review: [docs/w2_litreview.md](docs/w2_litreview.md).
Expands [PLAN_ModelUsability.md](PLAN_ModelUsability.md) §W2 into an
executable spec. Scope guard: **image features / imagery only** (terrain
covariate on hold per Brian 2026-06-11).

---

## 1. Why revisit the CNN — the ground has moved

The v1 CNN (`src/modeling/cnn.py` SmallCNN, ~35k params, log1p+Huber
regression) was labelled a dead-end on 2026-05-28 and v2 was built with
`features.context_patch.enabled=false`. Four things have invalidated the
basis of that judgment:

1. **Labels were wrong then.** Every v1/v2 result before 2026-06-10 was
   trained and scored on labels displaced ~360 m south (coreg sign bug) —
   and CNNs, which learn *where* texture sits, are plausibly hurt more by
   misaligned supervision than tile-aggregate GBMs.
2. **Dense vClaire labels + the shadow fix** lifted the tabular baseline
   itself (banked: median per-image mAUC 0.657, PR-AUC 0.563).
3. **W1 identified the failure mechanism the CNN can attack.** The
   `distribution_shift` class (real within-image signal, missed at LOIO) was
   causally validated by the bet-1 zscore experiment: per-image
   standardization rescued all three shift images (+0.18..+0.35 AUC) but
   cost raw-feature images — a global transform can't serve both regimes.
   **Photometric augmentation is the way to learn that invariance instead
   of imposing it** (DECISIONS.md 2026-06-11).
4. **GPU is approved and present** (driver reports CUDA 13.1; current torch
   is `2.12.0+cpu`). v1 trained on CPU with `device="cpu"` hardcoded into
   the params default.

What W2 actually tests (two falsifiable hypotheses):

- **H-A (feature ceiling, cause 4):** learned features beat the 2010-era
  handcrafted GLCM/shadow set at 5 m/px.
- **H-B (photometric invariance, cause 1 / shift class):** augmentation
  closes part of the anti-signal / distribution-shift gap that the tabular
  model cannot.

Honest prior: the `texture_decorrelated` images (no within-image signal at
5 m/px) should NOT improve — if they do, suspect leakage, not magic.

## 2. What already exists (reuse, don't rebuild)

| asset | state |
|---|---|
| `src/modeling/cnn.py` | SmallCNN (3 conv blocks → GAP → 2 FC), flip/rot/brightness/contrast/noise augmentation, log1p+Huber regression head, inner-val early stopping |
| `scripts/train_cnn.py` | v1 LOIO driver, per-fold state_dict artifacts |
| `src/modeling/loaders.py` | `gather_patches`, `patch_idx_S*` plumbing, `load_context_patch_stack` |
| `src/features.py` | `_build_context_patches` — Stage 4b emits `context_patches/{obs}_S{32,64}.npy` when `features.context_patch.enabled` |
| `dataset/context_patches/` | v1 patches (18 stacks) — stale georeferencing era, do not reuse |
| `dataset_v2/context_patches/` | **missing** — generation is setup task S2 |

## 3. Setup tasks (½ day)

- **S1 — CUDA torch** into `geospatial`: check `nvidia-smi`, install the
  matching cu12x wheel, verify `torch.cuda.is_available()`. Respect the
  Windows OpenMP gotcha (`import src.modeling` before numpy/pandas in every
  driver/probe). Add `device="cuda"` default-detection to `CNNParams`
  rather than hardcoding.
- **S2 — generate v2 patches**: set `features.context_patch.enabled=true`
  in `config_v2.yaml`, re-run Stage 4b `--all` (~20–30 min, CTX windows
  cached; disk ~a few GB). Then Stage 5 `--all` to refresh `patch_idx_S*`
  in the packaged folds. **Patches are uint8 raw DN — document that DN≤1
  clip pixels pass through; the per-patch input normalization (4.2) must
  be robust to them** (the shadow-threshold lesson).
- **S3 — smoke**: one fold end-to-end on GPU (`_smoke_cnn_one_fold.py`
  exists for v1; port to v2 + classification head), assert runtime/fold and
  no NaN loss.

## 4. Phase 1 — binary CNN, the augmentation contrast (the experiment)

### 4.1 Protocol (fixed in advance)

- **Task:** binary `fa_gt_1e-2` at S=64 (Tier 1 framing), BCE-with-logits +
  `pos_weight`; patches 64 px (= the tile exactly). Secondary (cheap, same
  runs): also log a regression head later only if Phase 1 passes.
- **Splits:** the same 38-fold `loio_nfold`, group-aware inner validation —
  hold out **whole images** (4–5) from each fold's training pool for early
  stopping; never tile-random.
- **Metrics:** per-image meaningful AUC (validity-aware: report n_pos/n_neg,
  exclude validity-failing images from the median), pooled PR-AUC,
  precision@top-5%. presence_auc stays retired. Paired per-fold Wilcoxon vs
  both baselines: banked GBM recipe (mAUC mean 0.637 / median 0.657,
  PR-AUC 0.563, sweep `20260611T054855Z`) and the Tier 1 reference
  classifier (AUC 0.655, `_sweep_binary/20260611T042543Z` — pre-shadow-fix;
  refresh it first as part of S3).
- **Promotion gates (declared now, PLAN_ModelUsability.md §W2):** augmented
  CNN must beat the tabular baseline by **pooled PR-AUC +0.03 OR median
  per-image mAUC +0.05 (validity-passing images)**, paired p < 0.05.
  **Mechanism check:** the H-B claim requires the aug-vs-no-aug contrast to
  improve the `distribution_shift` dossier images specifically.
- **Negative result is a result:** if the augmented CNN does not clear the
  gate, the 5 m/px floor is declared sensor-bound for this cohort and W2
  closes; usability work shifts to W4 scaffold + reliability honesty.

### 4.2 Phase 1 cells (4 runs × 38 folds)

| cell | augmentation | purpose |
|---|---|---|
| A | none | floor; isolates augmentation's total effect |
| B | geometric only (flips, 90° rots) | separates "more data" from invariance |
| C | geometric + photometric (brightness/contrast/gamma jitter, noise) | **the H-B cell** |
| D | C + per-patch input standardization (patch − mean)/std | architecture-level analog of the bet-1 zscore rescue |

Known pitfall to engineer around: **BatchNorm leaks image identity** when
batches are dominated by one image (per-image batch statistics act like a
test-time oracle at train time and shift at LOIO inference). Mitigations:
shuffle batches across images (default DataLoader shuffle over the whole
training pool already does this) and add a GroupNorm/InstanceNorm variant
as a fallback if cell D and BN interact badly.

One seed for the grid; re-run the winning cell with 3 seeds before any
promotion claim (seed variance at n=38 folds is the cheap insurance the
fold-ripple lesson demands).

### 4.2b S=32 replication (Brian, 2026-06-11)

Repeat at the finer scale: 32-px patches on the S=32 grid (160 m tiles,
~150k tiles cohort-wide). To keep the budget sane, not the full grid —
**cell A (no-aug) + the winning S=64 cell only**, after the S=64 grid is
read. Priors to test against: S=32 is where Stage 6a's spatial-context
features passed strictly on dev (finer tiles leave more recoverable
neighborhood signal), but absolute tabular performance at S=32 was far
below S=64 (PR-AUC ~0.29 vs 0.54) — the question is whether raw pixels
move that, i.e. whether the S=64 operating-scale choice survives the CNN
era. Patch stacks for both sizes come from the same Stage 4b re-run (S2),
so this adds no pipeline work, only ~2× fold count at ~4× tiles/fold.
*(Post-grid note 2026-06-11: the S=64 winner is cell A itself — no augmented
cell beat the no-aug floor — so the replication is cell A only at S=32.)*

*(3-seed note 2026-06-11 late: run cell A at S=32 with **3 seeds** and read
the **ensemble + Tier-1 fusion recipe** (`_w2_seed_ensemble.py`), not the
single-seed model — at S=64 the single-seed gate pass did not replicate
across seeds but the ensemble+fusion passed both gates. Since that recipe
was assembled after seeing the S=64 per-seed results, the S=32 run is its
held-out confirmation: pre-declared read = ensemble passes the per-image
gate vs the S=32 Tier-1 baseline AND fusion recovers pooled PR-AUC ≥ that
baseline. Needs the S=32 Tier-1 LightGBM run as the reference first.)*

### 4.3 Budget

GPU fold ≈ 1–3 min (37k patches of 64², ~35k-param net) → 4 cells × 38
folds ≈ 3–8 h wall. Run as background sweeps, one cell at a time, results
into `models/_sweep_cnn/{timestamp}/` with the standard summary/aggregate
parquet layout so `_w1_pistd_verdict.py`-style paired probes work as-is.

## 5. Phase 2 — revised 2026-06-11 after the Phase 1 read + literature review

Phase 1's gate logic anticipated an augmented-cell winner; what actually
emerged is (a) a no-aug CNN with a per-image edge, (b) a diagnosed
cross-image calibration deficit, and (c) a strongly positive fusion fix.
Phase 2 is re-pointed accordingly (full rationale + citations in
[docs/w2_litreview.md](docs/w2_litreview.md)).

### 5.0 Fusion productization (NEW — the positive Phase 1 result)

`cnn_rank × tier1_image_mean` hit pooled PR-AUC 0.5932 / prec@5% 0.914
(single seed). **3-seed update: per-seed fusion tracks the CNN's unstable
pooled calibration (beats Tier 1 on 2/3 seeds), but the 3-seed-ensemble
variant is seed-free and passes both pre-declared gates** — F1(ens) pooled
PR-AUC 0.5955 (+0.030) / prec@5% 0.887 / per-image Δ median +0.052 p=0.0065;
F3(ens) = pooled-rank average is the per-image-strongest (Δ +0.058, win
0.85, p=0.0001, median AUC 0.714). The Tier 1 map recipe candidate is
therefore: **3×SmallCNN seed ensemble for within-image ranking × Tier-1
LightGBM for image-level scale**. Needs: S=32 held-out confirmation
(§4.2b), a clean implementation (not a probe), choice of F1 vs F3 declared
before the S=32 read (F1 if pooled PR-AUC is the binding metric, F3 if
per-image AUC is), and W3-style calibration reporting.

### 5.1 CTX foundation-model embedding probe (REPLACES "SSL from scratch")

[Fang et al. 2026](https://doi.org/10.1029/2025JH000827) released a ViT-B
pretrained (MAE+DINO) on **3.9M crops of the Murray Lab mosaic itself**
([Zenodo 18180801](https://doi.org/10.5281/zenodo.18180801)) — zero domain
gap to our windows. First contact = **frozen GeM-pooled embeddings for the
S=64 tiles at 64-px and 192-px inputs → LightGBM columns in the standard
LOIO harness** (~1 h total). Fine-tune only if the probe shows signal.
Caveats the paper itself flags: MAE subdues sub-pixel roughness (our
signal); embeddings carry illumination geometry (our disease). MOMO
(arXiv:2604.02719, weights public) is the multi-sensor alternative.

### 5.2 Augmentation refinements (post-mortem of cells B–D)

- **FDA / randomized-histogram-matching cell**: cross-image *radiometric*
  augmentation that leaves shadow orientation untouched — better targeted
  than brightness/gamma jitter at the LOIO failure axis.
- **Azimuth-canonical orientation**: rotate patches to a fixed sun direction
  (per-tile `ctx_subsolar_az` from the Stage 6b SeamMap data) instead of
  random rotation; mainly protects the two azimuth outliers
  (ESP_076499_1160 at 228.6°, ESP_068483_2280 at 1.7°/incidence 4.3°).
- **Illumination conditioning**: `ctx_subsolar_az`/`ctx_incidence` scalars
  into the FC head (GBM H3 was null, but the CNN interaction is untested
  and the near-shadowless ESP_068483_2280 is a CNN-specific failure).

### 5.3 Per-image adaptation / reliability (AdaBN aftermath)

Adaptation methods (zscore, photometric aug, AdaBN) all rescue the same
shift-class image(s) and all fail cohort-wide — a class-specific treatment.
Promising residual: **base-vs-AdaBN prediction disagreement as a label-free
inference-time reliability flag** for the shift class (W1 found no other
warning signal for the ~13% silent failures). TENT only if that shows
promise.

### 5.4 Carried Phase 2 items (unchanged)

1. **Tier 2 regression head** (log1p boulder_count) on the winning recipe;
   probabilistic mean+variance head + small ensemble per the canopy-height
   literature (doubles as W4 reliability honesty);
   compression/high-bin-ratio reporting per W3.
2. **Neighborhood context input** (Brian, 2026-06-11) — predict the center
   tile from a 3×3-tile field (192 px = 960 m) or a multi-scale pyramid
   (64 px native + 128/256 px downsampled channels). Rationale: boulder
   fields are coherent structures and adjacent-tile labels correlate at
   ρ=0.72; a raw-pixel context test is distinct from Stage 6a's
   *summary-stat* neighborhoods, which passed at S=32 but bought nothing
   at S=64. Cautions: confounds the augmentation contrast (hence Phase 2),
   grows the `-1` edge-margin tile loss (a full tile ring at 192 px), and
   needs the cheap control: **post-hoc 3×3 smoothing of the output map**,
   which exploits the same autocorrelation with no retraining — learned
   context must beat dumb smoothing to be claimed. Converges with §5.1:
   192 px is also the better input scale for the Fang ViT.
3. **Capacity scaling** — only after the invariance/calibration questions
   are settled, never together with them.
4. **min_confidence=0.5 label filter** (still untested; label-noise lever
   that affects every model or none — run once, applies to all).

## 6. Reporting discipline

- One DECISIONS.md entry per phase with the verdict table; artifacts under
  `models/_sweep_cnn/`; dossier-style per-image deltas for the shift and
  decorrelated classes in every comparison.
- **QA notebook (added 2026-06-11, Brian):** `notebooks/19_w2_cnn.ipynb`
  documenting the exploration once Phase 1 is read — per-cell verdict table
  vs both baselines, per-image AUC deltas (A vs winning cell) highlighting
  the distribution_shift / texture_decorrelated classes, example augmented
  patches per cell (incl. a DN-clip patch under cell D), and training-curve
  samples. Imports from `src/` + the sweep parquets only; figures to
  `reports/figures/`. Build it after the S=64 grid + S=32 replication so it
  covers both.
- AskUserQuestion before: the CUDA install (env mutation), the full Phase 1
  grid launch (multi-hour), and any commit.
- Update HANDOFF_NEXT_SESSION.md + memory at each session end.
