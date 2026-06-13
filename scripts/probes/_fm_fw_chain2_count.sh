#!/usr/bin/env bash
# PLAN_FM 1b count-target re-read, REDONE with a real count threshold.
# bc_ge_1 (saturated 0.93 positive at S=64) was presence, not a count split
# (Brian, 2026-06-12). Two grounded thresholds from _fm_count_dist.py:
#   bc_ge_50  -> pos_rate 0.483 (near-median rich/poor split)
#   bc_ge_100 -> pos_rate 0.352, base-rate-matched to fa_gt_1e-2 (0.354)
# Winner recipe (MLP) on each, vs its OWN Tier-1 baseline. Cells cache.
set -e
RUN="/c/Users/brian/anaconda3/Scripts/conda.exe run --no-capture-output -n geospatial python -u scripts/probes/_fm_freeze_window.py run"

for T in bc_ge_50 bc_ge_100; do
  $RUN --matrix t1    --head lgbm --target "$T"
  $RUN --matrix t1ctx --head mlp  --target "$T"
  $RUN --matrix emb   --head mlp  --target "$T"
done

echo "FW CHAIN 2 (count targets) COMPLETE"
