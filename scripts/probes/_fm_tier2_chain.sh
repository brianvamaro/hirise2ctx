#!/usr/bin/env bash
# PLAN_FM 2.4 Tier-2 regression matrix: 3 heads x 2 targets x 2 feature sets.
# emb cells first (the headline single-stage-vs-hurdle read on the frozen
# features), then the Tier-1 handcrafted baselines for the FM lift. Cells cache
# (--force to rerun). The mlp_reg cells are the GPU compute; LightGBM is fast.
set -e
RUN="/c/Users/brian/anaconda3/Scripts/conda.exe run --no-capture-output -n geospatial python -u scripts/probes/_fm_tier2_regression.py"

for FEAT in emb t1; do
  for TGT in fractional_area boulder_count; do
    $RUN --variant lightgbm_tweedie            --target "$TGT" --features "$FEAT"
    $RUN --variant lightgbm_two_stage_balanced --target "$TGT" --features "$FEAT"
    $RUN --variant mlp_reg                     --target "$TGT" --features "$FEAT"
  done
done

echo "FM TIER2 CHAIN COMPLETE"
