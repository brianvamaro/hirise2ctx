# Handoff prompt — next session

**Last updated 2026-06-13 — freeze (§2.1) + productization (§2.2) + Tier-2
regression (§2.4) all DONE; §2.3/2.5/2.6/2.7 DESIGNED.** [PLAN_FM.md](PLAN_FM.md)
is the active plan. Nothing is running; tree clean.

**The critical-path bottleneck is on Brian's side:** any "confirmed" claim
(§2.3) needs the expansion cohort run through BoulderNet (23 ObsIds in
`cohort_expansion_candidates.csv`) — not yet done. Independent of that, the next
BUILDABLE pieces (no expansion data needed) are:
- **§2.6 deployable head + map pilot** — the recommended next build. Productize
  the frozen `mlp_ens3` head into `src/` (train-on-all-38, save/load/predict —
  it currently exists only inside the LOIO harness), then a map-pilot dry run on
  one Murray tile beyond HiRISE coverage. Spec is in PLAN_FM §2.6 (incl. the
  trivial multi-tile combine + the slice-streaming efficiency note). The visual
  "it works" payoff.
- **§2.7 reliability prototype** — CPU-only on the cached embeddings (Mahalanobis
  / kNN novelty vs the frozen recipe's per-image AUC). Design fleshed in PLAN_FM
  §2.7. Run it when the GPU is free; do NOT launch while a GPU chain is
  overhead-bound (it makes the box CPU-bound).
- **§2.3 declaration** can be WRITTEN any time (pre-data, that's the point) — the
  confirm-then-absorb design + proposed gates are in PLAN_FM §2.3.

Done since the last handoff (commits 9619510, 61184fd, a3a125c): §2.3
confirm-then-absorb design, §2.5 report skeleton (`docs/model_evidence.md`),
§2.6/§2.7 specs, and the §2.4 Tier-2 run (below).

**§2.2 productization (commit 032fa75):** `src/fm_embeddings.py` (ViT + GeM +
3×3-context slicing + `FangEmbedder.embed_window` inference path),
`src/modeling/loaders.py` cached-store join, `tests/test_fm_embeddings.py`.
**Bit-exact parity** vs the cached store (`scripts/probes/_fm_parity_check.py`).

**§2.4 Tier-2 regression (commit a3a125c; DECISIONS.md 2026-06-13):** single-stage
`mlp_reg` (3-seed MLP regressor) wins (Spearman 0.431 fa / 0.386 count, ~2× the
handcrafted baseline); the two-stage hurdle is DROPPED; regression matches the
classifier on rich/poor (meaningful_auc 0.78). Ceiling tested — zero-inflation
is NOT the limiter; compression quantified (~30% tail under-prediction, less than
handcrafted) → a calibration layer is future work. `_fm_tier2_regression.py` +
`_fm_tier2_ceiling.py`. A code review caught + fixed a metrics bug (count was
scored as presence; fix commit 61184fd, [[feedback_no_presence_auc]]).

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## FROZEN RECIPE (Brian sign-off; DECISIONS.md "Freeze window CLOSED")

Banked: `models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/`.

- **Scale S=32** (160 m tiles, 4× finer than S=64).
- **Embedding**: the **96-px (3×3-context)** CTX window → frozen Fang-ViT
  ViT-B/16 (MAE+DINO, Zenodo 18180801) → **GeM(p=3) → one 768-dim vector**.
  **emb-only** — NO own-tile P32, NO handcrafted features. Inference path =
  one embedding vector → MLP (no GLCM/gradient/shadow at map time).
- **Head `mlp_ens3`**: 3-seed MLP 768-256-64-1, dropout 0.2, BCE pos_weight,
  AdamW lr1e-3 wd1e-4, early-stop patience 8 on rotated inner-val; mean of 3
  seed probabilities. Per-fold standardize on train.
- **Target `fa_gt_1e-2`** (fractional_area > 0.01).
- **Numbers**: pooled PR-AUC **0.7832** / prec@5% **0.948** / median
  per-image AUC **0.7865** / dAUC(v) +0.120 / win 0.96; both gates PASS.

## Freeze-window evidence (PLAN_FM §2.1, all DONE; DECISIONS.md has tables)

- **1b target re-read**: FM advantage transfers to EVERY non-degenerate
  target (each vs its OWN Tier-1): fa_gt_1e-2 0.804 / fa_gt_1e-3 0.918 /
  bc_ge_50 0.826 / bc_ge_100 0.731 — all pass both gates. **`bc_ge_1` was
  the wrong count target** (Brian, this session): saturated at S=64 (0.93
  positive = presence), gates fail. Replaced with data-grounded
  bc_ge_50/bc_ge_100 (`scripts/probes/_fm_count_dist.py`; both registered in
  `src/modeling/binary_target.py`). Reverses the W0 "count beats area"
  finding (held only under handcrafted features). Target = Brian's scientific
  call; frozen on fa_gt_1e-2 for continuity.
- **1d pool×head**: GeM 0.8040 > mean 0.8015 > cls 0.7900 under the MLP.
- **1e** (all three add-ons REJECTED): arch sweep mid-pack/none separable →
  default 256×64/d0.2 kept; calibration layer net-harmful (per-image rank →
  pooled 0.5056; the 3-seed mean already fixes the wobble); cross-head
  ensemble dilutes. Plain mlp_ens3 wins.
- **1g operating-scale**: S=32 holds skill (both gates) AND feature
  elimination is FREE at S=32 (emb-only ties t1ctx; the 1f +2-pt gap
  dissolves). S=64 t1ctx is the higher headline (0.8040/0.8284) but Brian
  chose the 4× finer map. 1h (320-px) skipped as moot.

## Key tooling (all committed)

- `scripts/probes/_fm_freeze_window.py` — freeze-window runner (run/eval/pair).
- `scripts/probes/_fm_tier2_regression.py` — Tier-2 regression runner (mlp_reg +
  LGBM single/two-stage); `_fm_tier2_ceiling.py` — ceiling/compression analysis.
- `scripts/probes/_fm_count_dist.py` — per-tile boulder_count distribution.
- `src/fm_embeddings.py` — productized inference path; `src/modeling/loaders.py`
  fang cached-store join; `src/modeling/binary_target.py` has `bc_ge_50/100`.
- (AskUserQuestion before commits / expensive sweeps / env mutation.)

## Next-session queue — PLAN_FM.md §2 is authoritative

DONE: §2.1 freeze, §2.2 productization (032fa75), §2.4 Tier-2 regression
(a3a125c) + the metrics fix (61184fd). DESIGNED: §2.3/2.5/2.6/2.7 (PLAN_FM).
Remaining, in suggested order:

1. **§2.6 deployable head + map pilot — RECOMMENDED next build** (no expansion
   data needed). (a) Productize the frozen `mlp_ens3` head into `src/`:
   train-on-all-38, save/load/predict — it's currently only inside the LOIO
   harness (`_w2_fang_heads.py`). Apply the MLP perf fix (batch 4096 +
   tensors-on-device-once; see `_fm_tier2_regression.py` PERF NOTE). (b) Map-pilot
   dry run on one Murray tile beyond HiRISE coverage. Full spec (incl. the trivial
   multi-tile combine + slice-streaming) in PLAN_FM §2.6.
2. **§2.7 reliability prototype** — CPU-only on cached embeddings
   (Mahalanobis/kNN novelty vs the frozen recipe's per-image AUC). Design in
   PLAN_FM §2.7. Run when GPU is free.
3. **§2.3 pre-declared confirmation** — the *declaration* (gates/baseline/protocol)
   can be WRITTEN any time before data; the *execution* waits on Brian running
   BoulderNet on the 23 expansion ObsIds (`cohort_expansion_candidates.csv`).
   Protocol = confirm-then-absorb (tentative; permanent-holdout re-check flagged).
4. **§2.5 model-evidence report** — skeleton done (`docs/model_evidence.md`); fill
   the prose; headline numbers get the held-out stamp AFTER §2.3 confirmation.
5. Optional/gated: a Tier-2 calibration layer for the tail compression; MOMO
   disjoint-corpus probe; ViT fine-tune (decide after §2.3); per-image-std
   embeddings (deferred).

## Discipline now binding (PLAN_FM §3)

**No more recipe shopping on the 38 images.** The recipe is frozen; the next
number that touches it is the §2.3 pre-declared confirmation on held-out
expansion images. Misses recorded as declared.

## Critical gotchas (carry forward)

- `conda run` needs `--no-capture-output` + `python -u`; multi-line
  `python -c` FAILS on Windows ("arguments contain newlines") — write a probe
  script. Bare `python` is NOT on PATH (only inside the env).
- `import src.modeling` BEFORE numpy/pandas in torch-adjacent scripts.
- npz naming encodes INPUT px: P64/P192 = S=64 tiles, P32/P96 = S=32. The
  frozen recipe uses **P96** (S=32 3×3 context).
- `EmbeddingBank`/join keyed on (obs_id, ti, tj), validate="one_to_one".
- `KNNHead` does NOT standardize/impute → collapses on the mixed-scale t1ctx
  matrix (0.56 vs 0.77 gem-only). Irrelevant to the frozen recipe (emb-only,
  MLP) but don't reuse KNNHead on mixed features.
- Group-aware LOIO always; inference features must be CTX-derivable
  (embeddings are, mosaic-global).
- Per-image AUC ±0.1–0.2 fold-ripple error bars; carry n_pos/n_neg.
- AskUserQuestion before: expensive sweeps, env mutation, commits.
- Run only ONE GPU job at a time (chains here were sequenced via watchers).
- MLP cells at S=32 are SLOW (~15 min ea) but only because the tiny net is
  overhead-bound (GPU ~15% util) over 147k rows × 3 seeds × 38 folds — NOT a
  stall. Speed up ~3–5× next time: batch 4096 + pin full Xt/yt on the device
  once (see `_fm_tier2_regression.py` MLPRegressorEnsemble PERF NOTE).
- Fast pytest baseline 265 (full 285) + 8 CNN tests; no tests for the
  probe-tier Fang/freeze scripts (add during productization).

## Reporting protocol

1. DECISIONS.md — one entry per item with numbers (freeze entry exists).
2. Memory — `project_state_2026-06-12-freeze.md` is CURRENT.
3. This file — rewrite based on what actually lands.
