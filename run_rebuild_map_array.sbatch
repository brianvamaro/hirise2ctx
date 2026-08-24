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
# tile_is_reusable, partial windows are kept.
#
# MEASURED on the 2080 Ti this actually landed on, from the 4 surviving tasks of job 40561654
# (region_manifest.json `runs[].elapsed_s` / tiles): 2,469 / 2,519 / 2,725 / 2,605 s per tile
# = **0.717 h/tile**, so 03:00:00 for one tile is ~4x margin. The two tiles that stalled at
# 144/144 windows need no GPU at all on resubmit -- their partials are intact and
# `--clean-partials` only deletes AFTER a successful assemble, so they just re-assemble.
#
#SBATCH --job-name=h2c-rebuild-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-25
#SBATCH --constraint="GPU_SKU:RTX_2080Ti"
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

# ONE TILE PER TASK (was 6 tasks x 4-5 tiles, 2026-08-24d). Two reasons, both measured:
#   * a task that overran or raised took its remaining tiles down with it -- step 11 lost 3
#     never-attempted baseline tiles that way, and the A1 array's 10 h timeout forfeited
#     whatever its tasks had not reached. One tile per task makes the blast radius one tile.
#   * a 26-way array of short jobs also schedules faster on the gpu partition than 6 long ones.
# `ONLY` restricts the run to named tiles, so a resubmit can target just what is left instead
# of spawning 21 tasks that only print "skip":
#   ONLY="E0_N36 E0_N44 E4_N44 E8_N36 E16_N44" sbatch --array=0-4 run_rebuild_map_array.sbatch
if [[ -n "${ONLY:-}" ]]; then
  read -r -a TILES <<< "$ONLY"
fi
N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
if (( I >= ${#TILES[@]} )); then
  echo "task $SLURM_ARRAY_TASK_ID: nothing to do (${#TILES[@]} tile(s) selected)"
  exit 0
fi
SEL=()
for (( j=I; j<${#TILES[@]}; j+=N )); do SEL+=( "${TILES[j]}" ); done

REQUIRE_DEVICE="${REQUIRE_DEVICE:-2080 Ti}"   # see the PIN THE GPU note above
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
  --require-device "$REQUIRE_DEVICE" \
  --tiles "${SEL[@]}"

echo "task $SLURM_ARRAY_TASK_ID done -> $OUT"

# ---------------------------------------------------------------------------------------
# ⚠ PIN THE GPU. This is the single biggest cost lever in step 11, and it is not in the code.
#
# The gpu partition hands out whatever is free, and per-window cost varies **11x** across it.
# MEASURED, both arms, 144 windows per tile:
#
#     NVIDIA GeForce RTX 2080 Ti   17.9 s/window   0.72 h/tile   <- all 6 baseline tasks
#     Tesla P100-PCIE-16GB         91.7 s/window   3.67 h/tile   <- A1 task 0      (5.1x)
#     NVIDIA TITAN Xp             ~202  s/window   8.08 h/tile   <- A1 tasks 1-5  (11.2x)
#
# That, not the code, is why the A1 array timed out at 10 h: one TITAN Xp tile alone exceeds
# the wall. Projected 26-tile A1 arm: ~21 GPU-h on 2080 Ti, ~95 on P100, ~210 on TITAN Xp.
#
# TWO defences, use both:
#   1. `--constraint`, now a directive above: **GPU_SKU:RTX_2080Ti**, VERIFIED 2026-08-24e as the
#      real token on this cluster. `scontrol show node sh03-12n12` ->
#        GPU_GEN:TUR, GPU_BRD:GEFORCE, GPU_SKU:RTX_2080Ti, GPU_MEM:11GB, GPU_CC:7.5
#      16 nodes x 4 = 64 such cards, all sh03-12n*, on AMD EPYC 7502P hosts -- which is also
#      where the baseline arm ran, so this pins the CPU that does A1's per-window rasterize as
#      well as the GPU. The Pascal cards that blew the wall live on the older Intel sh02-* nodes.
#      A command-line `-C` overrides the directive. To find the tokens again:
#        sinfo -p gpu -o "%30N %130f" | tr ',' '\n' | grep -i '^GPU_' | sort -u
#      WARNING: a narrow %f width TRUNCATES before GPU_SKU. That is why the first look at this
#      showed only GPU_BRD and GPU_GEN and made the SKU look unavailable.
#   2. `--require-device`, below, which is checked by the driver itself against the device name
#      inference will actually run on. A Slurm feature name that is wrong or renamed fails
#      SILENTLY; this cannot. Costs ~60 s of startup on a bad allocation instead of the wall.
#
# To deliberately accept a slower card, widen REQUIRE_DEVICE and raise --time together:
#   REQUIRE_DEVICE="P100" sbatch --time=05:00:00 ... run_rebuild_map_array.sbatch
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# THE OVERLAP GUARD IS FIXED -- the triage note that used to live here is obsolete.
#
# What it said to do by hand, `check_overlap` now does: the gate is the FRACTION of duplicated
# cells that disagree, measured on `prob_raw` (pre-isotonic), not a magnitude on `prob`.
#   fp16 GEMM noise, within one run:  242 / 79,059 duplicated cells = 0.31 %   -> passes
#   stale-partial mixture, two runs:  63.1 % of finite pixels                  -> refuses
# Every tile sidecar now carries the full per-layer record under `overlap`, and every run
# prints it, so "did this tile disagree, and how much" no longer needs a laptop session.
#
# Also: a failing tile no longer kills the rest of the task (`run_tile_isolated`). The driver
# prints the traceback, carries on, and exits 1 with a list of failed tiles. **`exit 0` still
# means nothing on its own in this pipeline -- read the `N/M tiles complete` tally.**
# ---------------------------------------------------------------------------------------
