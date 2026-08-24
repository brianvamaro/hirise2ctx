#!/bin/bash
# run_rebuild_map_array.sbatch -- DAG step 11, BASELINE arm, v2 rebuild generation.
#
# Distinct from run_region_array.sbatch, which is left untouched as the record of how the
# SHIPPED (pre-rebuild) map was made. Four things differ and every one of them matters:
#
#   1. 26 tiles, not 19. run_region_array covers EXPANSION_TILES -- the net-new tiles of the
#      2026-06 expansion. The rebuild invalidates ALL 26: new labels, new embeddings, new head,
#      new calibration. Rendering only the expansion would silently ship 7 pre-rebuild tiles.
#   2. --model-parent models/deployable_g2. `resolve_model_dir` picks hits[-1] BY NAME, so with
#      the legacy 86c51a5dca220f63 also on disk the default parent is a coin flip. The rebuild
#      head is a5ffca2dcc536855 (norm_arm=none) and must be named explicitly.
#   3. --size-floor-basis. Without it `size_floor_tags` warns and emits NO SIZE_FLOOR_* tags, so
#      52 rasters would ship unable to state which boulders they count (R84).
#   4. BATCH default 96, not 256. 96 is the parity reference's batch, and measurement says the
#      larger batch buys nothing: 32/96/256/512 -> 766/723/730/731 img/s (DECISIONS 2026-08-18c).
#      Bigger batches were a guess; they cost parity comparability for no throughput.
#
# Usage (from $HOME/hirise2ctx on Sherlock):
#   mkdir -p logs && sbatch run_rebuild_map_array.sbatch
# Re-submitting after a wall-clock limit or pre-emption resumes: done tiles are skipped via
# tile_is_reusable, partial windows are kept. See the triage note at the bottom.
#
#SBATCH --job-name=h2c-rebuild-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/h2c-rebuild-base-%A_%a.out
#SBATCH --error=logs/h2c-rebuild-base-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs
ml python/3.12.1
source "/home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate"
cd "$HOME/hirise2ctx"

# All 26 circum-Chryse tiles (= scripts/map_region.py BLOCK_TILES; keep in sync).
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
MODEL_PARENT="${MODEL_PARENT:-models/deployable_g2}"
OUT="${MAP_OUT:-$SCRATCH/hirise2ctx/map_region_g2}"
mkdir -p "$OUT"

echo "task $SLURM_ARRAY_TASK_ID/$N  host $(hostname)  GPU ${CUDA_VISIBLE_DEVICES:-none}  batch $BATCH"
echo "  head parent: $MODEL_PARENT"
echo "  out:         $OUT"
echo "  tiles:       ${SEL[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/map_region.py \
  --out-dir "$OUT" \
  --batch "$BATCH" \
  --model-parent "$MODEL_PARENT" \
  --calibration "$MODEL_PARENT/calibration.npz" \
  --size-floor-basis "$MODEL_PARENT/size_floor_basis.json" \
  --clean-partials \
  --tiles "${SEL[@]}"

echo "task $SLURM_ARRAY_TASK_ID done -> $OUT"

# ---------------------------------------------------------------------------------------
# TRIAGE, if assembly dies with "N cells were written twice with DIFFERENT values":
#   max|delta| <= ~1e-4  -> cross-run fp non-determinism (DECISIONS 2026-08-19b: observed once
#                           in 15 runs, cause unidentified, no ECC on consumer GPUs). The sweep
#                           duplicates 62,559 cells/tile (2.78%), so a resume across two jobs can
#                           collide. Re-run THAT tile with --force.
#   max|delta| ~ 0.1-1   -> the stale-partial case R14 exists for (a real mixture measured at
#                           63.1% of pixels from the wrong run). INVESTIGATE. Do not --force.
# ---------------------------------------------------------------------------------------
