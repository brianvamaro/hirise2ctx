#!/bin/bash
# run_rebuild_a1_array.sbatch -- DAG step 11, A1 arm, v2 rebuild generation.
#
# **R06 closes here: A1 has never actually been generated at region scale.** Every A1 number of
# record came from one E8_N44 crop (`striping_a1_infer_crop.py`, which saved only a PNG) or from
# the LOIO skill gate. This is the first A1 raster.
#
# COST -- RESOLVED 2026-08-24e from logs/h2c-rebuild-a1-40572495_*.out. The 10 h timeout was
# the GPU ALLOCATION, not this arm: task 0 drew a Tesla P100 and tasks 1-5 TITAN Xps, while the
# baseline arm had RTX 2080 Tis throughout. See the PIN THE GPU note below.
#
#   baseline, 2080 Ti, measured             0.717 h/tile   (job 40561654 manifest)
#   A1 overhead, measured on the laptop    +0.193 h/tile   (DECISIONS 2026-08-19)
#     of which frame_labels_on             +0.107 h/tile   -> 0.031 h after the bbox prefilter
#                                                             (354.0 -> 102.8 s over 144 real
#                                                              windows, every one bit-identical)
#   => A1 on a 2080 Ti                      ~0.79 h/tile   -> 26 tiles = ~21 GPU-h
#
# So A1's own overhead is ~10 % of a tile, and 90 % of the overrun was the card. --time 06:00
# for ONE tile is ~7.5x the 2080 Ti cost; it would NOT fit a TITAN Xp tile, which is deliberate
# -- --require-device refuses that allocation up front rather than half-rendering into a wall.
#
# ⚠ Retracted from 2026-08-24b: `CPU Utilized 09:59:32 / 12.49 % of 8 cores` is NOT evidence of
# a CPU-bound overrun. The baseline arm is GPU-bound and its host process also pegs ~1 core.
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
#   ONLY="E0_N36 E0_N44 E4_N44 E8_N36 E16_N44" sbatch --array=0-4 run_rebuild_a1_array.sbatch
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
  --require-device "$REQUIRE_DEVICE" \
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
#   1. `--constraint` at submit time, in the cluster's feature vocabulary. Find the tokens with
#        sinfo -p gpu -o "%30N %90f" | tr ',' '\n' | grep -i '^GPU_' | sort -u
#      then submit e.g.  sbatch -C "GPU_SKU:RTX_2080Ti" run_rebuild_map_array.sbatch
#      (a command-line -C overrides any --constraint directive in this file).
#   2. `--require-device`, below, which is checked by the driver itself against the device name
#      inference will actually run on. A Slurm feature name that is wrong or renamed fails
#      SILENTLY; this cannot. Costs ~60 s of startup on a bad allocation instead of the wall.
#
# To deliberately accept a slower card, widen REQUIRE_DEVICE and raise --time together:
#   REQUIRE_DEVICE="P100" sbatch --time=05:00:00 ... run_rebuild_a1_array.sbatch
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
