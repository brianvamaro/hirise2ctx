# PLAN — W2: CNN on context patches (expanded)

**Status:** planned 2026-06-11 (Brian + Claude session). Expands
[PLAN_ModelUsability.md](PLAN_ModelUsability.md) §W2 into an executable spec.
Scope guard: **image features / imagery only** (terrain covariate on hold per
Brian 2026-06-11).

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

### 4.3 Budget

GPU fold ≈ 1–3 min (37k patches of 64², ~35k-param net) → 4 cells × 38
folds ≈ 3–8 h wall. Run as background sweeps, one cell at a time, results
into `models/_sweep_cnn/{timestamp}/` with the standard summary/aggregate
parquet layout so `_w1_pistd_verdict.py`-style paired probes work as-is.

## 5. Phase 2 — only if Phase 1 clears the gate

1. **Self-supervised pretraining** on unlabeled CTX patches (sample the
   mosaic freely, no BoulderNet needed; SimCLR/MAE-small), fine-tune on the
   38 images — the label-free way around n=38.
2. **Tier 2 regression head** (log1p boulder_count) on the winning recipe;
   compression/high-bin-ratio reporting per W3.
3. **Multi-scale input** (32+64+128 px pyramids) and capacity scaling —
   only after the invariance question is settled, never together with it.

## 6. Reporting discipline

- One DECISIONS.md entry per phase with the verdict table; artifacts under
  `models/_sweep_cnn/`; dossier-style per-image deltas for the shift and
  decorrelated classes in every comparison.
- AskUserQuestion before: the CUDA install (env mutation), the full Phase 1
  grid launch (multi-hour), and any commit.
- Update HANDOFF_NEXT_SESSION.md + memory at each session end.
