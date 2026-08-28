#!/bin/bash
# run_map_extended_array.sbatch -- map inference over a PLAN-DRIVEN tile list.
#
# The difference from run_region_array.sbatch: that script hardcodes the 19 circum-Chryse
# expansion tiles in a bash array, so growing the map means editing an sbatch file (and
# keeping it in sync with scripts/map_region.py's EXPANSION_TILES by hand -- a comment asks
# you to). This one reads the tile list out of a plan JSON produced by
# scripts/plan_map_extent.py, so extending the map to a new box is:
#
#     python scripts/plan_map_extent.py --lat .. --lon .. \
#         --map-dirs reports/map_region reports/map_extended \
#         --json reports/map_extended/plan.json
#     sbatch run_map_extended_array.sbatch
#
# and nothing in this file changes. Tiles already rendered in ANY --map-dirs directory are
# excluded by the planner, and map_region.py additionally skips any tile whose final GeoTIFF
# already exists in --out-dir, so re-submitting after a wall-clock kill or a pre-emption
# resumes: finished tiles skipped, a partial tile resumes at its last completed window.
#
# Usage (from $HOME/hirise2ctx on Sherlock):
#   mkdir -p logs
#   sbatch run_map_extended_array.sbatch                  # 6-way over plan.json
#   PLAN=reports/map_other/plan.json sbatch run_map_extended_array.sbatch
#   BATCH=96 sbatch run_map_extended_array.sbatch         # match the parity reference exactly
#   # more/fewer concurrent GPUs: edit  #SBATCH --array=0-5
#
#SBATCH --job-name=h2c-map-ext
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/h2c-map-ext-%A_%a.out
#SBATCH --error=logs/h2c-map-ext-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs
ml python/3.12.1
source "/home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate"
cd "$HOME/hirise2ctx"

PLAN="${PLAN:-reports/map_extended/plan.json}"
OUT="${MAP_OUT:-$SCRATCH/hirise2ctx/map_extended}"
BATCH="${BATCH:-256}"       # saturates the L40S/A100; parity-safe (the Fang ViT is per-sample)

# This task's disjoint stride of the plan's to_render list. Striding rather than chunking so a
# task that draws a slow GPU holds up a spread of the map, not one contiguous corner of it.
N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
mapfile -t SEL < <(python - "$PLAN" "$N" "$I" <<'PY'
import json, sys
plan, n, i = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tiles = json.load(open(plan))["to_render"]
print("\n".join(tiles[i::n]))
PY
)

if [ "${#SEL[@]}" -eq 0 ]; then
  echo "task $SLURM_ARRAY_TASK_ID: no tiles in this stride of $PLAN -- nothing to do"
  exit 0
fi

echo "task $SLURM_ARRAY_TASK_ID/$N  host $(hostname)  GPU ${CUDA_VISIBLE_DEVICES:-none}  batch $BATCH"
echo "  plan:  $PLAN"
echo "  out:   $OUT"
echo "  tiles: ${SEL[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# GPU-CONDITIONAL WALL-CLOCK. The shipped baseline ran 17-22 s/window on a 2080 Ti (~44 min/tile,
# 144 windows), but the A1 arm drew Pascal cards at ~202 s/window and timed out at 6 h having
# finished 7 of 26 tiles (project_state_2026-08-24b). --time is 8 h here for headroom; if the
# line above prints a Pascal card (TITAN Xp / GTX 10xx), expect to re-submit and resume rather
# than to finish, and consider requesting a newer card via --constraint.

mkdir -p "$OUT"
python scripts/map_region.py --out-dir "$OUT" --batch "$BATCH" --clean-partials --tiles "${SEL[@]}"
echo "task $SLURM_ARRAY_TASK_ID done -> $OUT"
