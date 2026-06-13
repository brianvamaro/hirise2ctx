# Handoff prompt — next session

**Last updated 2026-06-12 (later session) — freeze window CLOSED + §2.2
productization DONE.** [PLAN_FM.md](PLAN_FM.md) is the active plan. Next
session starts at **§2.3 pre-declared confirmation** (write the gates BEFORE
any expansion-image number exists). Nothing is running.

**§2.2 productization landed (commit 032fa75):** `src/fm_embeddings.py` (ViT
encoder + GeM + 3×3-context slicing + `FangEmbedder.embed_window` inference
path), `src/modeling/loaders.py` cached-store join
(`load_fang_store`/`fang_columns_for_keys`/`augment_fold_with_fang`, torch-free),
`tests/test_fm_embeddings.py` (15 tests; full fast suite 292 green),
README + DATA_DICTIONARY entries. **Bit-exact parity** between the productized
`src/` path and the cached store the frozen 0.7832 was measured on
(`scripts/probes/_fm_parity_check.py`: max abs diff 0.0).

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

## Tooling built this session (UNTRACKED — commit before/with productization)

- `scripts/probes/_fm_freeze_window.py` — the freeze-window runner
  (run/eval/pair; the bake-off generalized across scale/pool/target/MLP-arch
  with per-target Tier-1 baselines in the identical LOIO harness).
- `scripts/probes/_fm_count_dist.py` — per-tile boulder_count distribution.
- `scripts/probes/_fm_fw_chain{1,2_count,3_s32}.sh` (+ `.log`) — the chains.
- `src/modeling/binary_target.py` — added `bc_ge_50`, `bc_ge_100`.
- (Brian: no commits were made this session — AskUserQuestion before commits.)

## Next-session queue — PLAN_FM.md §2 is authoritative

1. **§2.2 productize extraction into `src/` — DONE (commit 032fa75)**:
   `src/fm_embeddings.py` (inference path + ViT/GeM/slicing), loader cached-store
   join, 15 pytest, docs; bit-exact parity vs the cached store. The MLP *head*
   is NOT yet productized (still in `_w2_fang_heads.py`) — fold it in with §2.4
   Tier-2 or the map pilot.
2. **§2.3 pre-declared confirmation** (START HERE): write the DECISIONS.md declaration —
   gates, baseline, protocol — BEFORE any expansion-image number exists. New
   images = pure held-out. Inputs: `cohort_expansion_candidates.csv` (23
   ObsIds incl. 4 lander sites); BoulderNet runs are Brian's side.
3. Then per PLAN_FM: Tier-2 regression on embeddings (§2.4, retest
   single-stage vs hurdle) → **model-evidence report (§2.5, Brian:
   persuasion-grade, BEFORE the map pilot)** → map pilot (§2.6) →
   embedding-space reliability (§2.7). Optional/gated: MOMO, ViT fine-tune
   (decide after §3), per-image-std embeddings (deferred).

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
- Fast pytest baseline 265 (full 285) + 8 CNN tests; no tests for the
  probe-tier Fang/freeze scripts (add during productization).

## Reporting protocol

1. DECISIONS.md — one entry per item with numbers (freeze entry exists).
2. Memory — `project_state_2026-06-12-freeze.md` is CURRENT.
3. This file — rewrite based on what actually lands.
