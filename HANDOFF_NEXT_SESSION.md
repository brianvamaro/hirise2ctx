# Handoff prompt — next session

**Last updated 2026-06-12 night — FM probe LANDED + head bake-off DONE;
[PLAN_FM.md](PLAN_FM.md) is now the active plan** (PLAN_CNN.md closed).
Nothing is running. Next session continues the PLAN_FM §2.1 freeze window
(remaining: 1b target re-read, 1d pool×head, 1e micro-sweep+calibration,
1g operating-scale decision, 1h optional 320-px probe), then productization
(§2.2) + pre-declared confirmation (§2.3).

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## Program bests (2026-06-12 night; THREE DECISIONS.md entries with tables)

**mlp_ens3 on t1+gem192: pooled PR-AUC 0.8040 / med per-image AUC 0.8284 /
win 0.96 vs Tier-1** (five days ago: 0.5651 / 0.6806). Key reads:

| recipe (S=64) | pooled PR-AUC | prec@5% | med AUC | dAUC med (v) | win |
|---|---|---|---|---|---|
| **mlp_ens3 × t1+gem192** | **0.8040** | 0.916 | **0.8284** | +0.1465 | **0.96** |
| mlp_ens3 × gem192-only | 0.7852 | **0.936** | 0.8035 | +0.1374 | 0.85 |
| LightGBM t1_gem192 (first probe) | 0.7637 | 0.977 | 0.770 | +0.0746 | 0.89 |
| Tier-1 (ref) | 0.5651 | 0.771 | 0.681 | — | — |

- **Head bake-off (PLAN_FM 1a/1c)**: on the identical gem192-only matrix,
  every non-tree head beat LightGBM (mlp_ens3 0.7852 / kNN 0.7709 /
  logreg 0.7385 / lgbm 0.7146); paired per-image stats make **MLP 3-seed
  ensemble the decisive winner** (vs all others p≤0.003); the other three
  are tied. MLP pooled calibration is seed-wobbly → ensemble is the
  promotable form; exhaustive head tuning deferred until cohort expansion
  (Brian).
- **Feature elimination (1f)**: handcrafted features still add ~+0.02
  pooled/med-AUC under the MLP — NOT free to drop; both variants stay
  candidates to the freeze (simplicity-vs-points = Brian's call; interacts
  with the scale decision: simpler inference matters more at S=32's 4×
  tile count).

- **S=32: the Tier-1 collapse is FIXED** — t1_gem96 pooled 0.7639 vs
  Tier-1 0.4840, both gates pass; scale-robust (0.7639 ≈ 0.7637). The
  3×3-context input is the carrier at both scales; own-tile-only misses
  the per-image bar by a rounding error at both scales.
- **Failure classes rescued**: distribution_shift +0.23 to +0.31,
  texture_decorrelated +0.17 to +0.21, ok_geometry_fixed +0.27 to +0.30.
  emb_only ≈ fused ⇒ queue-item-6 answered: texture_decorrelated was a
  **feature-set floor, not a sensor floor**.
- **ESP_076499_1160 (azimuth outlier) = biggest winner, dAUC +0.458** —
  the image every W1–W2 adaptation only partially rescued. Azimuth read:
  benefit geometry-agnostic (ρ ns vs incidence/azimuth); caveat *present*
  (sin(az) recoverable from embeddings, LOO r=0.588) but harmless.
- **Pool ablation**: GeM(p=3) 0.7637 > mean 0.7071 > cls 0.6961. Declared
  pick logic: t1_gem192 if pooled is binding, t1_gem64_gem192 if per-image.
- **Caveats recorded**: (a) transductive pretraining — the FM saw test
  *pixels* (never labels) during SSL; deployment-matching argument in
  DECISIONS (estimand = Murray-mosaic inference, which is in-corpus
  everywhere, so LOIO is unbiased for it); MOMO disjoint-corpus probe is
  the optional empirical bound. (b) Post-hoc assembly — promotion needs
  the standing pre-declared confirmation on cohort-expansion images.
  Unlike the CNN there is NO seed instability (deterministic end to end).

## What was built (commits c481671 … 66b3493, all pushed to main)

- `scripts/probes/_w2_fang_embed.py` — extraction; **hand-rolled plain-torch
  ViT-B/16** (timm key layout, strict load; NO timm/torchvision installed).
  `--tile-px {64,32}`; own-tile + 3×3-context inputs, bicubic→224,
  (x/255−0.5)/0.5; cls/mean/gem banked; bit-exact center-vs-cached-patch
  geometry assert. 178 s (S=64) / 834 s (S=32) on the 5070.
- `dataset_v2/fang_embeddings/{obs}_P{32,64,96,192}.npz` — 3.5 GB, 100%
  context coverage everywhere (window buffer covers the ring; do NOT
  stitch neighbor patches — only 71% would have full 3×3).
- `scripts/probes/_w2_fang_probe.py` — LOIO probe, `--tile-px`, `--pool`,
  variants {t1_own, t1_ctx, t1_own_ctx, emb_only}; verdict.json per run
  under `models/fang_probe/{label}/{hash}/`.
- `scripts/probes/_w2_fang_patch_visual.py` + 2 alignment figures;
  `scripts/probes/_w2_fang_azimuth.py` + figure (all 19_w2_fang_*).
- Weights: `models/pretrained/mars-mae-dino-vit-base-v1.pth` (341.7 MB,
  untracked; re-download from Zenodo 18180801 if lost).
- `scripts/probes/_w2_fang_heads.py` — head bake-off (`--matrix
  {emb,t1ctx}`, `--heads`, median imputation for t1 columns) +
  `_w2_fang_head_pairs.py` (paired head-vs-head stats →
  `models/fang_probe/head_pairs.json`).
- **PLAN_FM.md** — the active plan (queue + freeze discipline + retired
  list); PLAN_CNN.md closed with a pointer header.
- `notebooks/20_fang_vit_probe.ipynb` (+ `_build_20.py`), executed: variant
  glossary, verdict tables, per-image dAUC by failure class, alignment +
  azimuth figures, top-8 tiles (FM 8/8 vs Tier-1 1/8 on ESP_076499_1160),
  truth-vs-model maps in the classification_slimmer.md style (old
  anti-signal exemplar ESP_046328_2180: slim 0.344 → FM 0.789).

## Next-session queue — PLAN_FM.md §2 is authoritative; summary:

1. **Finish the freeze window (§2.1)**: 1b target re-read (count vs area,
   each vs its OWN Tier-1 baseline); 1d pool×head under the MLP; 1e winner
   micro-sweep + cross-head ensemble + calibration layer (pooled wobble is
   a calibration problem); 1g **operating-scale decision** (re-run winner
   at S=32 — Brian wants the finer map if skill holds); 1h optional 320-px
   probe. Then **freeze ONE recipe** (head, pool, matrix, scale, target —
   feature-elimination call is Brian's).
2. **Productize extraction into src/** (embed arbitrary CTX windows,
   `fang_*` columns as a loader feature source, pytest).
3. **Pre-declare the confirmation protocol** (gates BEFORE any
   expansion-image numbers; `cohort_expansion_candidates.csv`, 23 ObsIds;
   BoulderNet runs are Brian's side).
4. Then per PLAN_FM: Tier-2 on embeddings → **model-evidence report
   (§2.5, Brian: persuasion-grade, BEFORE the map pilot)** → map pilot →
   embedding-space reliability. Optional/gated: MOMO, emb_only@S=32
   overnight, fine-tune go/no-go, per-image-std embeddings (deferred).

## Critical gotchas (carry forward)

- `conda run` needs `--no-capture-output` + `python -u`; multi-line
  `python -c` fails on Windows — write a probe script.
- `import src.modeling` BEFORE numpy/pandas in torch-adjacent scripts;
  set `KMP_DUPLICATE_LIB_OK=TRUE` for bare `python -c` torch one-liners.
- npz naming encodes INPUT px only: P64/P192 = S=64 tiles, P32/P96 = S=32.
- Embedding join is on (obs_id, ti, tj) with validate="one_to_one" — keep
  it keyed, never positional.
- LightGBM handles NaN embedding cols natively (not that any exist: 100%
  coverage).
- Group-aware LOIO always; inference features must be CTX-derivable
  (embeddings are: mosaic-global).
- Per-image AUC ±0.1–0.2 fold-ripple error bars; carry n_pos/n_neg.
- AskUserQuestion before: expensive sweeps, env mutation, commits.
- Fast pytest baseline 265 (full 285) + 8 CNN-cell tests; no tests added
  for probe-tier Fang scripts (add them during productization).

## Reporting protocol

1. DECISIONS.md — one entry per item with numbers (two Fang entries exist).
2. Memory — `project_state_2026-06-12-fang.md` is CURRENT.
3. This file — rewrite based on what actually lands.
