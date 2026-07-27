#!/bin/bash
# run_f_region_stageb.sbatch -- PLAN_FBuild Stage B: per-frame embed + infer on the 907 Stage-A
# cubes -> per-frame P(rich) keyed to a global S=32 tile grid (scripts/f_region_stageb.py). GPU
# array like run_region_array.sbatch; each task takes a round-robin stride of region_frame_list.csv;
# resumable (skips frames whose {PRODUCT_ID}.npz exists). Map venv (torch/GPU), NOT the isis env.
#
# Prereq: Stage A done (907 cubes in $SCRATCH/hirise2ctx/f_region); V2 done WITH subsolar geometry
# (reports/figures/region_frame_incidence.csv). RUN V3 FIRST (this script on ~2 overlapping frames via
# `python scripts/f_region_stageb.py --frames PID1 PID2 ...` + scripts/f_region_v3.py) to confirm
# co-location + parity before this full array. Per-row incidence is physical (no slope to set).
#
# Usage:
#   cd ~/hirise2ctx && git pull && mkdir -p logs
#   sbatch run_f_region_stageb.sbatch
# Sizing: V1 measured ~162M valid tiles, ~33 L40S-h embed. 6 GPUs -> ~5-6 h; --array=0-11 -> ~3 h.
#
#SBATCH --job-name=h2c-region-B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/h2c-region-B-%A_%a.out
#SBATCH --error=logs/h2c-region-B-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs
ml python/3.12.1
source "/home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate"
cd "$HOME/hirise2ctx"

N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
BATCH="${BATCH:-256}"
CUBES="${CUBES:-$SCRATCH/hirise2ctx/f_region}"
OUT="${STAGEB_OUT:-$SCRATCH/hirise2ctx/f_region_logits}"
mkdir -p "$OUT"

echo "task $I/$N  host $(hostname)  GPU ${CUDA_VISIBLE_DEVICES:-none}  batch $BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

TASK_ID=$I N_TASKS=$N python scripts/f_region_stageb.py \
    --cubes-dir "$CUBES" --out-dir "$OUT" --batch "$BATCH"

echo "task $I done -> $OUT"
