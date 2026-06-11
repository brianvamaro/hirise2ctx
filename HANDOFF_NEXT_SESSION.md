# Handoff prompt — next session

**Last updated 2026-06-10 (night) — W1 COMPLETE in one session** (rung-1
coreg sign bug found+fixed+re-banked, rungs 2–5 worked, dossier + notebook 18
+ decisions delivered). The active program is
[PLAN_ModelUsability.md](PLAN_ModelUsability.md); next session starts the
**post-W1 next bets** (below) or **W2 setup**.

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## Read in this order

1. [PLAN_ModelUsability.md](PLAN_ModelUsability.md) — W1 status banner.
2. Memory `project_state_2026-06-10d.md` (CURRENT).
3. [DECISIONS.md](DECISIONS.md) — the two 2026-06-10 W1 entries (rung 1 bug +
   rungs 2–5 synthesis).

## Current banked baseline (compare everything against this)

`lightgbm_two_stage_balanced` × `boulder_count` @ S=64, corrected labels +
DN-clip shadow fix — ρ +0.1767, meaningful AUC 0.6372, PR-AUC 0.5633,
prec@top-5% 0.5811; per-image median AUC **0.657**, anti-signal 8 (members
churned; per-image AUC has ±0.1-0.2 fold-ripple error bars — see DECISIONS
2026-06-10 shadow-fix entry). Sweep `models/_sweep_w0/20260611T054855Z`.
Tier 1 classifier: AUC 0.655, lift 1.845, ECE 0.254
(`models/_sweep_binary/20260611T042543Z`, pre-shadow-fix — refresh with the
next binary sweep). presence_auc is RETIRED (unobservable + undefined on
1/4 of images); meaningful_auc is the discrimination metric.
**All pre-fix v2 numbers are stale** (the labels were ~360 m south); slimmer
docs are submitted — erratum decision is Brian's, still open.

## W1 outcome (one line each)

- Geometry: fixed (rung 1); join: clean (rung 2); detections: clean (rung 3).
- Seam-tile masking does nothing; CTX-source effect is regional (ρ≈0.38).
- Anti-signal mechanism: `texture_decorrelated` (3 imgs — small 2–4 m
  boulders in uniform speckle, 5 m/px floor), `distribution_shift` (2 imgs —
  STRONG within-image signal the LOIO model misses; ESP_076499_1160 has
  shadow ρ +0.73 but AUC 0.224), `validity_limited` (3 imgs — too few
  pos/neg tiles to judge).
- Dossier: `dataset_v2/w1_dossier.parquet` (+ `_w1_dossier.md`); notebook 18.
- Decisions: Tier 1 reliability = graded region-level confidence
  (`mean_n_sources`, `dominant_source_fraction` + validity columns), NOT a
  binary flag; **native-CTX pivot NO-GO for now**.

## Next bets (in evidence order)

1. ~~Per-image feature standardization~~ **DONE 2026-06-11, NOT PROMOTED**
   (DECISIONS.md 2026-06-11): all 4 variants fail the declared cohort
   criteria, BUT zscore rescued all 3 distribution_shift images out of
   anti-signal (+0.18..+0.35) at the cost of raw-feature images; the
   raw+std concat dilutes both. Class-specific treatment, not a recipe
   upgrade. Code kept: `loaders.standardize_fold_per_image` /
   `augment_fold_with_per_image`; sweeps `models/_sweep_perimage_std/`.
2. **W2 CNN Phase 1 — NOW SPECCED in [PLAN_CNN.md](PLAN_CNN.md)**
   (2026-06-11): setup S1 CUDA torch (driver CUDA 13.1, AskUserQuestion
   before env mutation) → S2 enable `features.context_patch` in
   config_v2 + Stage 4b/5 re-run (v2 has NO patches yet) → S3 smoke +
   refresh Tier 1 classifier baseline → 4-cell augmentation grid
   (none / geometric / +photometric / +per-patch-std), binary fa_gt_1e-2
   @ S=64, group-aware inner val, gates pre-declared (PR-AUC +0.03 OR
   median mAUC +0.05 on validity-passing images; mechanism check on the
   distribution_shift images). BatchNorm image-leak pitfall documented.
3. **Terrain covariate** — ON HOLD per Brian (2026-06-11): stick to image
   features for now.

## Critical gotchas (carry forward)

- `conda run` needs `--no-capture-output` + `python -u`; multi-line
  `python -c` fails on Windows — write a probe script.
- `import src.modeling` BEFORE numpy/pandas in torch-adjacent scripts.
- Group-aware LOIO always; inference features must be CTX-derivable.
- Per-image AUC is meaningless on near-saturated images — always carry
  n_pos/n_neg (dossier has them).
- Stage 6a nbr artifacts were NOT regenerated post-fix (S=32 only, not in
  the recipe).
- bc ≥ 50 vs fa > 1e-2 positive-definition caveat on cross-target deltas.
- AskUserQuestion before: expensive sweeps, git commits, destructive ops.
- **Fast pytest baseline 265 (full 285).**

## Reporting protocol

1. DECISIONS.md — one entry per item with numbers.
2. Memory — supersede `project_state_2026-06-10d.md` when the next item lands.
3. This file — rewrite based on what actually lands.
