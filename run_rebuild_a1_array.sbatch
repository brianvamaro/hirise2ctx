#!/bin/bash
# run_rebuild_a1_array.sbatch -- DAG step 11, A1 arm, v2 rebuild generation.
#
# **R06 closes here: A1 has never actually been generated at region scale.** Every A1 number of
# record came from one E8_N44 crop (`striping_a1_infer_crop.py`, which saved only a PNG) or from
# the LOIO skill gate. This is the first A1 raster.
#
# COST -- read this before choosing --time again. The first attempt (job 40572495) TIMED OUT
# at 10:00:15 and the budget below does NOT explain that, so one term is still unmeasured.
#
#   baseline arm, measured on the 2080 Ti     0.717 h/tile   (job 40561654 manifest)
#   A1 overhead, measured on the laptop      +0.193 h/tile   (DECISIONS 2026-08-19)
#     of which frame_labels_on               +0.107 h/tile   -> now 0.031 h (bbox prefilter,
#                                                               3.4-4.2x, bit-identical)
#   => expected after the prefilter            0.79 h/tile   -> 5 tiles = 4.0 h, not 10 h
#
# So the prefilter is real and verified (`test_frame_labels_on_prefilter_is_identical`, plus a
# 144-window run on the real E-12_N36 frames: 354.0 s -> 102.8 s, every window bit-identical),
# but it accounts for ~0.08 h of a >1 h/tile shortfall. ⚠ Do NOT repeat the earlier reading that
# `CPU Utilized 09:59:32 / 12.49 % of 8 cores` proves a CPU-bound overrun: the baseline arm is
# GPU-bound and its host process also pegs ~1 core, so that ratio is what a healthy run here
# looks like. The missing term is in `logs/h2c-rebuild-a1-40572495_*.out` -- per-tile
# `DONE in Ns` (now with a read/a1/predict/write breakdown), the `win k/144 ... Ns` lines, and
# the streaming-pass message. --time is 06:00 for ONE tile, which is >7x the expected cost and
# absorbs a 2x surprise; revise it once the log is read.
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
#SBATCH --time=06:00:00
#SBATCH --array=0-25
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
