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

1. **Per-image feature standardization** (rank or z-score features within
   each image/window before the GBM): directly targets the
   distribution_shift class; one sweep; promotion criteria vs the banked
   baseline declared in advance. Inference-compatible (window stats only).
2. **W2 CNN Phase 1 with photometric augmentation** (plan §W2; CUDA torch
   install approved; `context_patch_px` Stage 4 re-run needed for patches).
3. **Terrain covariate** (failures concentrate in channels/mesas/crater
   terrain; plains are fine — Serrano mediation; Tanaka map join is
   inference-compatible).

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
