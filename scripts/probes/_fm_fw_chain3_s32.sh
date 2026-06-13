#!/usr/bin/env bash
# PLAN_FM 1g operating-scale decision: re-run the FROZEN winner form at S=32.
# Winner pinned by 1b/1d/1e: mlp_ens3, gem pooling, 256x64/d0.2, target
# fa_gt_1e-2 (incumbent area target). Both feature variants (t1ctx + emb-only)
# so the feature-elimination call carries to the scale decision. S=32 uses the
# P96 (3x3) context input; Tier-1 S=32 baseline resolves via SCALE_CONFIG[32].
# Compare to the S=64 winner (t1ctx 0.8040 / emb 0.7852) at equal target.
set -e
RUN="/c/Users/brian/anaconda3/Scripts/conda.exe run --no-capture-output -n geospatial python -u scripts/probes/_fm_freeze_window.py run --tile-px 32 --target fa_gt_1e-2 --head mlp"

$RUN --matrix t1ctx
$RUN --matrix emb

echo "FW CHAIN 3 (S=32 scale decision) COMPLETE"
