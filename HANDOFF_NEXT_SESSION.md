# Handoff prompt — next session

**Last updated 2026-06-12 afternoon — W2 Phase 2 lead bet LANDED.** The
Fang-ViT frozen-embedding probe passed both gates by the largest margin of
the program, at both scales. Nothing is running. Active program:
[PLAN_ModelUsability.md](PLAN_ModelUsability.md) → [PLAN_CNN.md](PLAN_CNN.md)
§5.1 (now the centerpiece); next session starts productization +
pre-declared confirmation (queue below).

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## The result (2026-06-12, two DECISIONS.md entries with full tables)

**Fang et al. 2026 ViT-B/16 (MAE+DINO, pretrained on 3.9M Murray-mosaic
crops; Zenodo 18180801) frozen GeM embeddings → LightGBM columns:**

| variant (S=64) | pooled PR-AUC | prec@5% | med AUC | dAUC med (v) | gates |
|---|---|---|---|---|---|
| **t1_gem192** | **0.7637** | **0.977** | 0.770 | +0.0746 | both PASS |
| t1_gem64_gem192 | 0.7549 | 0.884 | **0.7777** | **+0.0918** (win 0.93) | both PASS |
| emb_only (no T1 feats) | 0.7424 | 0.876 | 0.752 | +0.0831 | both PASS |
| Tier-1 (ref) | 0.5651 | 0.771 | 0.681 | — | — |
| F1(ens) W2 best (ref) | 0.5955 | 0.887 | 0.711 | +0.052 | — |

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

## What was built (commit c481671 + uncommitted follow-ups)

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

## Next-session queue (evidence order)

1. **Productize embedding extraction out of probe-tier** (src/ module):
   inference must embed arbitrary CTX windows (not just cached tiles);
   wire as optional feature source in the Stage-4b/loaders path so the
   sweep/notebook tooling sees `fang_*` columns natively.
2. **Pre-declare the confirmation protocol** for the embedding recipe on
   the cohort-expansion images (`cohort_expansion_candidates.csv`, 23
   verified ObsIds incl. 4 ground-truthed lander sites; BoulderNet runs
   are Brian's side). Gates should be declared BEFORE any new-image
   numbers are seen; reuse the standard pair (pooled +0.03 / per-image
   +0.05 p<0.05) unless Brian wants stricter.
3. **Notebook 19 (or new 20) Fang section**: verdict tables, per-image
   dAUC bars by failure class, the alignment + azimuth figures, example
   tiles where emb_only wins big (ESP_076499_1160).
4. Optional cheap reads: emb_only @ S=32 overnight (~6 h CPU); MOMO
   disjoint-corpus cross-check (weights public, arXiv:2604.02719);
   ViT fine-tune decision EXPLICITLY deferred until after confirmation.
5. Carried (PLAN_CNN.md §5.4): Tier-2 probabilistic head on the new
   feature set; min_confidence=0.5 label filter; conditional-leveler
   fusion productization is now LIKELY OBSOLETE (embeddings beat it on
   both axes) — retire formally after the confirmation read.

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
