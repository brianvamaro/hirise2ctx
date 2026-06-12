# Handoff prompt — next session

**Last updated 2026-06-12 morning — W2 COMPLETE including the S=32
held-out read.** Nothing is running. The active program is
[PLAN_ModelUsability.md](PLAN_ModelUsability.md) → [PLAN_CNN.md](PLAN_CNN.md);
next session starts Phase 2 (queue below).

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## S=32 held-out verdict (2026-06-12, DECISIONS.md has the full table)

**Recipe formally NOT confirmed — per-image gate missed by 0.0002**
(ensemble Δ median +0.0498 vs +0.05 bar, p=0.0009, win 0.81). The
per-image core claim *replicated in direction and significance*; the
magnitude landed a rounding error under the pre-declared bar — recorded
as declared, no re-reading. **Bigger finding: the fusion half INVERTS at
S=32** — the CNN ensemble is the better pooled model there (0.5454 vs
Tier-1 0.4840, +0.061; handcrafted features degrade at fine tiles, CNN
holds), so fusing with Tier-1 *hurts* pooled. "CNN ranks / X scales" is a
**conditional-leveler recipe**: Tier-1 levels at S=64, the CNN itself at
S=32. S=64 stays the operating scale. Promotion of the S=64 fusion now
needs a fresh pre-declared confirmation (new cohort images, or a
pre-declared fresh-seed S=64 re-run).

## W2 state (what happened 2026-06-11 → early 06-12)

- **Setup S1–S3 done**: CUDA torch installed; v2 context patches generated
  (17 GB, S32+S64 stacks in `dataset_v2/context_patches/`); Tier-1
  classifier refreshed @ S=64 (`models/_sweep_binary/20260611T214042Z`):
  **pooled PR-AUC 0.5651, per-image median AUC 0.6806, prec@5% 0.771**.
- **Phase 1 grid (cells A–D, seed 0)**: every augmented cell ≤ no-aug floor;
  geometric augmentation actively harmful (destroys the cohort-constant
  142–186° sun-azimuth shadow prior). H-B refuted cohort-level.
- **Cell E (photometric_only)**: cohort-equal to A → the harm was the
  geometric half. All 3 distribution_shift images improve under E
  (+0.06/+0.17/+0.06) — H-B mechanism real but weak. ESP_076499_1160
  (228.6° azimuth outlier) instead prefers rotation → azimuth-canonical
  orientation moved up the Phase 2 queue.
- **3-seed replication of cell A**: the seed-0 gate pass does NOT replicate
  (dAUC median +0.066 p=0.016 / +0.038 p=0.059 / +0.005 p=0.66). Per-image
  skill seed-stable (median 0.69–0.71); score calibration is not (pooled
  PR-AUC 0.51/0.56/0.49).
- **THE RESULT — seed-ensemble + Tier-1 fusion passes BOTH gates at S=64**
  (`scripts/probes/_w2_seed_ensemble.py`):
  | variant | pooled PR-AUC | prec@5% | med AUC | dAUC med (v) | p |
  |---|---|---|---|---|---|
  | ens_mean (3 seeds) | 0.5327 | 0.675 | 0.711 | +0.052 | 0.0065 |
  | **F1(ens)** = within-img quantile × T1 image mean | **0.5955** | **0.887** | 0.711 | +0.052 | 0.0065 |
  | F3(ens) = pooled-rank avg | 0.5856 | 0.812 | **0.714** | +0.058 | 0.0001 |
  | Tier 1 (ref) | 0.5651 | 0.771 | 0.681 | — | — |
  **Recipe assembled post-hoc → S=32 is its held-out confirmation** (steps
  1–4 above). F1 if pooled is binding, F3 if per-image is (declared).
- **AdaBN probe**: cohort null (Δ median −0.017 p=0.40) but third
  independent rescue of ESP_076499_1160 (+0.315); base-vs-AdaBN
  *disagreement as a label-free reliability flag* queued (§5.3).
- **S=32 baseline banked**: Tier-1 LightGBM fa_gt_1e-2 @ scale_idx 2:
  AUC 0.660 ± 0.101 (`models/_sweep_binary/20260612T062412Z`; preds under
  `models/lightgbm_classification/2d046f48c722f0a5/`).
- **Notebook 19** (`notebooks/19_w2_cnn.ipynb` + `_build_19.py`) built,
  executed, committed; 5 figures in `reports/figures/19_w2_*.png`; §6
  reports PENDING until re-executed after the chain.
- **Lit review** `docs/w2_litreview.md` (7 sections + ranked queue):
  Bickel (diversity > scale-mixing), canopy-height twin (probabilistic
  ensemble heads), Rodriguez & Wegner read in full (stride-1/no-downsample
  finding → new capacity direction; σ=K/π smoothing recipe), TTA, FDA/RHM,
  Mars FMs (**Fang et al. ViT-B pretrained on OUR Murray mosaic, weights
  Zenodo 18180801** — Phase 2 §5.1 frozen-embedding probe), thermal RA.
- **Cohort expansion**: `cohort_expansion_candidates.csv` — 12 verified
  ObsIds (InSight, VL1, Phoenix-program, southern 64–71°S, N high-lat) + 3
  to-vet rows. NOTE corrections: the two PSP_*_2025 are Viking Lander 1
  (22°N), not Phoenix; ESP_017776_2435 is 2×2-binned (62 cm/px).
- DECISIONS.md has FOUR 2026-06-11 W2 entries (setup, grid read, 3-seed +
  ensemble, cell E). Commits: ab2029d, 57ee93f, b1c2dee, 5eb194e, 7cbbb49.

## Phase 2 queue (PLAN_CNN.md §5, evidence order, post-S=32)

1. **Fang-ViT frozen-embedding probe** (§5.1) — now the lead bet:
   GeM-pooled embeddings at 64 px + 192 px → LightGBM columns, standard
   LOIO (~1 h). Weights: Zenodo 18180801, pretrained on our exact mosaic.
2. **Fusion productization in conditional-leveler form** (§5.0):
   ensemble-CNN ranking × inner-validation-chosen leveler; needs a fresh
   pre-declared confirmation before promotion (cohort expansion images
   are the clean option — `cohort_expansion_candidates.csv` has 23
   verified literature-anchored ObsIds incl. 4 ground-truthed lander
   sites; BoulderNet runs are Brian's side).
3. Augmentation refinements (§5.2): FDA/RHM cell, azimuth-canonical
   orientation, illumination conditioning.
4. AdaBN-disagreement reliability flag (§5.3); TENT only if it works.
5. Carried (§5.4): Tier-2 probabilistic head + ensemble; 3×3 context vs
   smoothing control; stride-1/no-pool capacity variant (Rodriguez &
   Wegner); min_confidence filter.
6. texture_decorrelated dossier reattribution check (CNN 0.59–0.74 vs GBM
   0.41–0.46; S=32 Tier-1 collapse 0.5651→0.4840 while CNN holds is
   further evidence for "feature-set floor" not "sensor floor").

## Critical gotchas (carry forward)

- `conda run` needs `--no-capture-output` + `python -u`; multi-line
  `python -c` fails on Windows — write a probe script.
- `import src.modeling` BEFORE numpy/pandas in torch-adjacent scripts.
- **Each (variant,seed) gets its own config_hash dir** — glob
  `models/cnn_bce_S{P}/*/scale_*` and read seed from snapshot.json
  (`model.params.seed`), never assume one dir.
- **Single-seed claims are unreliable at n=38** — 3-seed protocol for any
  promotion claim (now in PLAN_CNN.md §4.2).
- CNN sweeps don't persist per-epoch history (console only) — add
  history.json if training curves are ever needed.
- Group-aware LOIO always; inference features must be CTX-derivable.
- Per-image AUC ±0.1–0.2 fold-ripple error bars; carry n_pos/n_neg.
- bc≥50 vs fa>1e-2 positive-definition caveat on cross-target deltas.
- AskUserQuestion before: expensive sweeps, env mutation, commits.
- Fast pytest baseline 265 (full 285) + 8 new CNN-cell tests pass.

## Reporting protocol

1. DECISIONS.md — one entry per item with numbers.
2. Memory — supersede `project_state_2026-06-11-w2.md` when the S=32
   verdict lands.
3. This file — rewrite based on what actually lands.
