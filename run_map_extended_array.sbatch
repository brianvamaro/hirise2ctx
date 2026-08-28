#!/bin/bash
# run_map_extended_array.sbatch -- map inference over a PLAN-DRIVEN tile list.
#
# Grows the map into a new box. The tile list is DATA -- read from a plan JSON produced by
# scripts/plan_map_extent.py -- so extending the map never edits an sbatch file:
#
#     python scripts/plan_map_extent.py --lat .. --lon .. \
#         --map-dirs reports/map_region reports/map_extended \
#         --json reports/map_extended/plan.json
#     sbatch --array=0-$(( $(python -c "import json;print(len(json.load(open('reports/map_extended/plan.json'))['to_render']))") - 1 )) run_map_extended_array.sbatch
#
# Everything below that isn't the tile list is inherited from run_rebuild_map_array.sbatch,
# which is the script that actually produced the 8 already-rendered tiles this product adopts.
# Those settings are not defaults -- each was paid for:
#
#   * --model-parent models/deployable_g2 + explicit --calibration + --size-floor-basis.
#     `resolve_model_dir` picks hits[-1] BY NAME, so with the legacy 86c51a5dca220f63 also on
#     disk the default parent is a coin flip. Omitting --size-floor-basis emits NO SIZE_FLOOR_*
#     tags, shipping rasters that cannot state which boulders they count (R84). Get either wrong
#     and the 27 new tiles are a DIFFERENT PRODUCT from the 8 adopted ones, silently.
#     The preflight below refuses to start unless the head and calibration digests match the
#     ones recorded in the adopted tiles' sidecars.
#   * BATCH 96, not 256. 96 is the parity reference's batch, and the larger batch buys nothing:
#     32/96/256/512 -> 766/723/730/731 img/s (DECISIONS 2026-08-18c).
#   * ONE TILE PER TASK. A task that overran or raised used to take its remaining tiles with it;
#     step 11 lost 3 never-attempted tiles that way. One tile per task makes the blast radius one
#     tile, and a wide array of short jobs also schedules faster than a few long ones.
#   * --constraint GPU_SKU:RTX_2080Ti AND --require-device. Per-window cost varies 11x across
#     the gpu partition (2080 Ti 17.9 s/win = 0.72 h/tile; P100 91.7; TITAN Xp ~202 = 8.08 h/tile
#     -- which is what blew the A1 array's wall). The constraint is the Slurm-side defence; a
#     feature name that is wrong or renamed fails SILENTLY, so --require-device checks the device
#     inference will actually run on and costs ~60 s on a bad allocation instead of the wall.
#
# Usage (from $HOME/hirise2ctx on Sherlock), for the 27-tile Xanthe/Chryse bridge:
#   mkdir -p logs
#   sbatch --array=0-26 run_map_extended_array.sbatch
# Resubmit after a wall-clock kill or pre-emption: done tiles are skipped (tile_is_reusable),
# partial windows are kept. To target just what is left, name them:
#   ONLY="E-24_N44 E-20_N44" sbatch --array=0-1 run_map_extended_array.sbatch
# To deliberately accept a slower card, widen REQUIRE_DEVICE and raise --time TOGETHER:
#   REQUIRE_DEVICE="P100" sbatch --array=0-26 --time=05:00:00 run_map_extended_array.sbatch
#
#SBATCH --job-name=h2c-map-ext
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-26
#SBATCH --constraint="GPU_SKU:RTX_2080Ti"
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
BATCH="${BATCH:-96}"
MODEL_PARENT="${MODEL_PARENT:-models/deployable_g2}"
REQUIRE_DEVICE="${REQUIRE_DEVICE:-2080 Ti}"
ADOPT_FROM="${ADOPT_FROM:-reports/map_region}"

# ONE TILE PER TASK: this task's single tile out of the plan's to_render (or ONLY).
if [[ -n "${ONLY:-}" ]]; then
  read -r -a TILES <<< "$ONLY"
else
  mapfile -t TILES < <(python - "$PLAN" <<'PY'
import json, sys
print("\n".join(json.load(open(sys.argv[1]))["to_render"]))
PY
)
fi
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
if (( I >= ${#TILES[@]} )); then
  echo "task $SLURM_ARRAY_TASK_ID: nothing to do (${#TILES[@]} tile(s) in $PLAN)"
  exit 0
fi
TILE="${TILES[I]}"

echo "task $SLURM_ARRAY_TASK_ID  host $(hostname)  GPU ${CUDA_VISIBLE_DEVICES:-none}  batch $BATCH"
echo "  plan:        $PLAN  (${#TILES[@]} tiles)"
echo "  head parent: $MODEL_PARENT"
echo "  out:         $OUT"
echo "  tile:        $TILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# PREFLIGHT: the head and calibration this task is about to use must be the SAME CONTENT that
# produced the tiles this product already contains. A path is not enough -- a directory can be
# overwritten in place, and the head dir name is a recipe hash of the training config, not of
# the weights. Costs seconds; catches the one error that would silently split the product in two.
python - "$MODEL_PARENT" "$ADOPT_FROM" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.mapping import artifact_digest          # numpy only; no torch, no GPU

parent, adopted_dir = Path(sys.argv[1]), Path(sys.argv[2])
hits = sorted(p for p in parent.glob("*") if (p / "recipe.json").exists())
if not hits:
    raise SystemExit(f"PREFLIGHT FAIL: no deployable head under {parent}")
head, calib = hits[-1], parent / "calibration.npz"
got = {"head": artifact_digest(head), "calibration": artifact_digest(calib)}
print(f"  resolved head: {head}")
print(f"    head_digest        {got['head']}")
print(f"    calibration_digest {got['calibration']}")

from src.map_manifest import tile_sidecars      # stdlib only; the shared "what is a tile" rule

recs = [json.loads(q.read_text()) for q in tile_sidecars(adopted_dir).values()]
want = {r["head_digest"] for r in recs if "head_digest" in r}
wantc = {r["calibration_digest"] for r in recs if "calibration_digest" in r}
if not want:
    print(f"  (no sidecars under {adopted_dir} to compare against -- skipping the match check)")
    raise SystemExit(0)
if len(want) != 1 or len(wantc) != 1:
    raise SystemExit(f"PREFLIGHT FAIL: {adopted_dir} is already mixed: {len(want)} head "
                     f"digest(s), {len(wantc)} calibration digest(s)")
w, wc = want.pop(), wantc.pop()
if got["head"] != w or got["calibration"] != wc:
    raise SystemExit(
        "PREFLIGHT FAIL: this head/calibration is NOT the one that made the existing tiles.\n"
        f"  existing head_digest        {w}\n  resolved head_digest        {got['head']}\n"
        f"  existing calibration_digest {wc}\n  resolved calibration_digest {got['calibration']}\n"
        f"Set MODEL_PARENT to the right parent, or render into a fresh --out-dir.")
print("  PREFLIGHT OK: head + calibration match the tiles this product already holds")
PY

mkdir -p "$OUT"
python scripts/map_region.py \
  --out-dir "$OUT" \
  --batch "$BATCH" \
  --model-parent "$MODEL_PARENT" \
  --calibration "$MODEL_PARENT/calibration.npz" \
  --size-floor-basis "$MODEL_PARENT/size_floor_basis.json" \
  --clean-partials \
  --require-device "$REQUIRE_DEVICE" \
  --tiles "$TILE"

echo "task $SLURM_ARRAY_TASK_ID done ($TILE) -> $OUT"

# ⚠ `exit 0` means nothing on its own in this pipeline. A failing tile no longer kills the task
# (run_tile_isolated prints the traceback and carries on), so read the `N/M tiles complete`
# tally in the log, and verify the product afterwards:
#   python3 scripts/verify_map_download.py "$OUT" --plan reports/map_extended/plan.json
