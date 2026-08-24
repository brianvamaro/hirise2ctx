#!/bin/bash
# run_rebuild_a1_array.sbatch -- DAG step 11, A1 arm, v2 rebuild generation.
#
# **R06 closes here: A1 has never actually been generated at region scale.** Every A1 number of
# record came from one E8_N44 crop (`striping_a1_infer_crop.py`, which saved only a PNG) or from
# the LOIO skill gate. This is the first A1 raster.
#
# Cost note: A1 is NOT the baseline arm plus a constant. Measured on E-12_N36 (DECISIONS
# 2026-08-19): +0.193 h/tile, of which only 22% is the once-per-tile per-frame native statistic
# (154.9 s) and 78% is re-rasterizing the same ~81 SeamMap polygons onto all 144 read windows
# (2.67 s each). Budget ~26 x 0.193 = ~5 GPU-h ON TOP of the baseline arm's cost.
#
# The 19 parent-tile SeamMaps do not need uploading: `src.striping.load_frames` pulls the
# shapefile out of the tile zip via /vsizip/vsicurl/ range requests and caches it as a GeoPackage,
# and Sherlock fetches the zips anyway.
#
# Usage (from $HOME/hirise2ctx on Sherlock):
#   mkdir -p logs && sbatch run_rebuild_a1_array.sbatch
#
#SBATCH --job-name=h2c-rebuild-a1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/h2c-rebuild-a1-%A_%a.out
#SBATCH --error=logs/h2c-rebuild-a1-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs
ml python/3.12.1
source "/home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate"
cd "$HOME/hirise2ctx"

# Same 26 tiles as the baseline arm -- §5.1's one-common-footprint rule.
TILES=(
  E-12_N32 E-12_N36 E-12_N40 E-12_N44
  E-8_N32  E-8_N36  E-8_N40  E-8_N44
  E-4_N32  E-4_N36  E-4_N40  E-4_N44
  E0_N32   E0_N36   E0_N40   E0_N44
  E4_N32   E4_N36   E4_N40   E4_N44
  E8_N32   E8_N36   E8_N40   E8_N44
  E12_N44  E16_N44
)

N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
SEL=()
for (( j=I; j<${#TILES[@]}; j+=N )); do SEL+=( "${TILES[j]}" ); done

BATCH="${BATCH:-96}"
# The A1 driver takes the head DIRECTLY (no --model-parent), and R07 makes it REFUSE -- strict=True
# -- a head it cannot verify as this arm. The arm id is the VERSIONED string
# `a1_native_perframe_tilesupport_v2` (src.striping.A1_ARM), NOT the bare "a1".
#
# 2026-08-24: training the head with `--norm-arm a1` produced exactly that refusal and killed all
# 6 array tasks in 49 s. `train_deployable_head.py` does `args.norm_arm or infer_norm_arm(store)`,
# and infer_norm_arm("fang_embeddings_a1") already returns the correct versioned id -- so the flag
# should simply be OMITTED and the arm inferred from the store name. Head is now 66ec8b755b9c0b20.
A1_PARENT="${A1_PARENT:-models/deployable_a1_g2}"
A1_HEAD="${A1_HEAD:-$A1_PARENT/66ec8b755b9c0b20}"
OUT="${MAP_OUT:-$SCRATCH/hirise2ctx/map_a1_g2}"
mkdir -p "$OUT"

echo "task $SLURM_ARRAY_TASK_ID/$N  host $(hostname)  GPU ${CUDA_VISIBLE_DEVICES:-none}  batch $BATCH"
echo "  head: $A1_HEAD"
echo "  out:  $OUT"
echo "  tiles: ${SEL[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/striping_a1_map.py \
  --out-dir "$OUT" \
  --batch "$BATCH" \
  --head "$A1_HEAD" \
  --calibration "$A1_PARENT/calibration.npz" \
  --size-floor-basis "$A1_PARENT/size_floor_basis.json" \
  --clean-partials \
  --tiles "${SEL[@]}"

echo "task $SLURM_ARRAY_TASK_ID done -> $OUT"

# ---------------------------------------------------------------------------------------
# NOTE on the A1 calibration: it was banked with --force. Its LOIO Tier-1 ECE is 0.0523
# against a 0.05 gate (DECISIONS 2026-08-23c) and its meta records gates_passed=false. Brian
# ruled the 0.0023 miss immaterial. Tier-2 (abundance) passed cleanly.
#
# NOTE on isotonic: do NOT add --no-isotonic to get an uncalibrated comparison. Every run already
# writes <tile>_prob_raw.tif alongside <tile>_prob.tif, so isotonic-vs-raw is free from one pass.
# ---------------------------------------------------------------------------------------
