#!/bin/bash
# run_region_array.sbatch -- PARALLEL regional map inference (PLAN_RegionalMap §10 decision #5).
# The 19 expansion tiles are independent + map_region is resumable and writes per-tile outputs
# keyed by tile id (no shared manifest), so we fan them across a Slurm job ARRAY: each array task
# is one 1-GPU job that takes a disjoint stride of the tile list. Wall-clock ~= (sequential GPU-h)
# / (concurrent array tasks). 19 tiles ~13-19 GPU-h sequential -> ~2-3 h with 6 GPUs.
#
# Usage (from $HOME/hirise2ctx on Sherlock):
#   mkdir -p logs
#   sbatch run_region_array.sbatch                 # default: the 19 EXPANSION_TILES, 6-way
#   BATCH=96 sbatch run_region_array.sbatch        # match the parity reference exactly
# Re-submitting after a wall-clock/pre-emption resumes (done tiles skipped, partial windows kept).
#
#SBATCH --job-name=h2c-map-arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/h2c-map-arr-%A_%a.out
#SBATCH --error=logs/h2c-map-arr-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs
ml python/3.12.1
source "/home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate"
cd "$HOME/hirise2ctx"

# The 19 net-new expansion tiles (= scripts/map_region.py EXPANSION_TILES; keep in sync).
TILES=(
  E-12_N32 E-12_N36 E-12_N40 E-12_N44
  E-8_N32  E-8_N36  E-8_N40  E-8_N44
  E-4_N32  E-4_N36  E-4_N40  E-4_N44
  E0_N32   E0_N36   E0_N44
  E4_N32   E4_N36   E8_N32   E8_N36
)

# This task's disjoint stride of the tile list.
N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
SEL=()
for (( j=I; j<${#TILES[@]}; j+=N )); do SEL+=( "${TILES[j]}" ); done

# batch 256 better saturates the L40S/A100 (parity-safe: ViT is per-sample). If you run the
# strict parity gate, emit its reference at the SAME --batch (fp16 GEMM kernels vary by batch).
BATCH="${BATCH:-256}"
OUT="${MAP_OUT:-$SCRATCH/hirise2ctx/map_region}"
mkdir -p "$OUT"

echo "task $SLURM_ARRAY_TASK_ID/$N  host $(hostname)  GPU ${CUDA_VISIBLE_DEVICES:-none}  batch $BATCH"
echo "  tiles: ${SEL[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/map_region.py --out-dir "$OUT" --batch "$BATCH" --tiles "${SEL[@]}"
echo "task $SLURM_ARRAY_TASK_ID done -> $OUT"
