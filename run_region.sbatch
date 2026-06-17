#!/bin/bash
# run_region.sbatch -- regional/global map inference on Sherlock's GPU partition
# (PLAN_RegionalMap.md §4a). map_region.py is checkpointed at (tile, read-window)
# granularity, so a wall-clock limit or pre-emption just resumes: re-submitting this
# script picks up where it stopped. Pass extra map_region args after the script name, e.g.
#   sbatch run_region.sbatch --all
#   sbatch run_region.sbatch --tiles E4_N44 E8_N44
#   sbatch run_region.sbatch --tiles E4_N44 --limit-windows 4   # throughput probe
#SBATCH --job-name=h2c-map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/h2c-map-%j.out
#SBATCH --error=logs/h2c-map-%j.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1
source "/home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate"
cd "$HOME/hirise2ctx"

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Write heavy outputs to $SCRATCH (not backed up, but big/fast); symlinked ./reports stays small.
OUT="${MAP_OUT:-$SCRATCH/hirise2ctx/map_region}"
mkdir -p "$OUT"
python scripts/map_region.py --out-dir "$OUT" "$@"
echo "outputs in $OUT  (scp the *_abundance.tif / *_prob.tif back to the laptop)"
