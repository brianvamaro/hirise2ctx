#!/usr/bin/env bash
# Re-run the 3 emb boulder_count cells with --force: they ran before the
# meaningful_threshold fix so their banked metrics were presence-based (count>0.01).
# Predictions are deterministic -> identical; only the rich/poor metrics change (now @50).
set -e
RUN="/c/Users/brian/anaconda3/Scripts/conda.exe run --no-capture-output -n geospatial python -u scripts/probes/_fm_tier2_regression.py"
$RUN --variant lightgbm_tweedie            --target boulder_count --features emb --force
$RUN --variant lightgbm_two_stage_balanced --target boulder_count --features emb --force
$RUN --variant mlp_reg                     --target boulder_count --features emb --force
echo "EMBCOUNT FIX COMPLETE"
