#!/usr/bin/env bash
# PLAN_FM 2.1 freeze-window chain 1: 1b target re-read + 1d pool x head +
# 1e arch micro-sweep cells (S=64). 1g (S=32) launches AFTER the 1d/1e read
# pins the winning form. Cells are cached -- safe to rerun after interruption.
set -e
RUN="/c/Users/brian/anaconda3/Scripts/conda.exe run --no-capture-output -n geospatial python -u scripts/probes/_fm_freeze_window.py run"

# 1e prerequisite first (2 min): kNN on the winner matrix, unblocks the
# cross-head ensemble post-hoc while the rest of the chain runs.
$RUN --matrix t1ctx --head knn50

# --- 1b: per-target Tier-1 baselines, then winner recipe on each target ---
$RUN --matrix t1 --head lgbm --target bc_ge_1
$RUN --matrix t1 --head lgbm --target fa_gt_1e-3
$RUN --matrix t1ctx --head mlp --target bc_ge_1
$RUN --matrix t1ctx --head mlp --target fa_gt_1e-3
$RUN --matrix emb --head mlp --target bc_ge_1
$RUN --matrix emb --head mlp --target fa_gt_1e-3

# --- 1d: pool x head under the MLP (gem incumbent = heads_mlp_*_t1ctx) ---
$RUN --matrix t1ctx --head mlp --pool mean
$RUN --matrix t1ctx --head mlp --pool cls

# --- 1e: MLP arch micro-sweep on the winner matrix (incumbent 256x64/d0.2) ---
$RUN --matrix t1ctx --head mlp --hidden 128 32
$RUN --matrix t1ctx --head mlp --hidden 512 128
$RUN --matrix t1ctx --head mlp --dropout 0.4
$RUN --matrix t1ctx --head mlp --hidden 128 32 --dropout 0.4
$RUN --matrix t1ctx --head mlp --hidden 512 128 --dropout 0.4

echo "FW CHAIN 1 COMPLETE"
