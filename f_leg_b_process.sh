#!/bin/bash
# f_leg_b_process.sh -- F pilot leg B: ISIS pipeline for the cohort frames.
# Array-task-aware: each Slurm task processes its stride of cohort_frame_list.csv.
# All projected cubes are kept (KEEP_CUBES always 1 for leg B — the extract step needs them).
#
# Usage (via run_f_leg_b.sbatch, or manually for testing):
#   export TASK_ID=0 N_TASKS=1
#   bash f_leg_b_process.sh [workdir]           # workdir default: $SCRATCH/hirise2ctx/f_leg_b
#
# Requires: ISIS micromamba env (setup_isis_env.sh), ISISDATA set.
# Input:  $REPO/reports/f_leg_b/cohort_frame_list.csv  (built by f_leg_b_frame_list.py)
# Output: $WORK/*.map.cub (one per frame), $WORK/status_${TASK_ID}.csv
set -uo pipefail   # NOT -e: one bad frame must not abort the task

REPO="$(cd "$(dirname "$0")" && pwd)"
LIST="$REPO/reports/f_leg_b/cohort_frame_list.csv"
WORK="${1:-${SCRATCH:-/tmp}/hirise2ctx/f_leg_b}"
MAP="$REPO/f_equirect.map"
TASK_ID="${TASK_ID:-0}"
N_TASKS="${N_TASKS:-1}"
OUT_STATUS="$WORK/status_${TASK_ID}.csv"

[ -f "$LIST" ] || { echo "ERROR: $LIST missing (run scripts/f_leg_b_frame_list.py first)" >&2; exit 1; }
[ -n "${ISISDATA:-}" ] || { echo "ERROR: ISISDATA not set (activate isis env)" >&2; exit 1; }
mkdir -p "$WORK"; cd "$WORK"

# Header-indexed CSV column lookup
col() { head -1 "$LIST" | tr ',' '\n' | grep -nx "$1" | cut -d: -f1; }
PID_COL=$(col PRODUCT_ID); URL_COL=$(col edr_url)

echo "product_id,edr_mb,t_download,t_import,t_spiceinit,t_ctxcal,t_evenodd,t_cam2map,t_total,map_mb,status" > "$OUT_STATUS"

now() { date +%s.%N; }
step() {
    local label="$1"; shift
    local t0; t0=$(now)
    echo "--- $label ---" >> "$WORK/isis_${TASK_ID}.log"
    "$@" >> "$WORK/isis_${TASK_ID}.log" 2>&1
    local rc=$?
    echo "$(now) - $t0" | bc
    return $rc
}

# Select this task's stride from the frame list (round-robin for load balance)
FRAME_NUM=0
tail -n +2 "$LIST" | while IFS= read -r line; do
    if (( FRAME_NUM % N_TASKS == TASK_ID )); then
        pid=$(echo "$line" | cut -d, -f"$PID_COL")
        url=$(echo "$line" | cut -d, -f"$URL_COL")
        echo "=== task $TASK_ID/$N_TASKS: $pid ==="
        # skip if cube already done
        if [ -f "$pid.map.cub" ]; then
            mmb=$(echo "scale=1; $(stat -c%s "$pid.map.cub" 2>/dev/null)/1000000" | bc)
            echo "  $pid: already done ($mmb MB) -- skipped"
            FRAME_NUM=$((FRAME_NUM + 1))
            continue
        fi
        T0=$(now); status=ok
        td=$(step "download($pid)" curl -fsSL --retry 3 --retry-delay 5 -o "$pid.IMG" "$url") \
            || status=download_fail
        mb=$(echo "scale=1; $(stat -c%s "$pid.IMG" 2>/dev/null || echo 0)/1000000" | bc)
        ti=0; ts=0; tc=0; te=0; tm=0
        [ "$status" = ok ] && {
            ti=$(step import mroctx2isis from="$pid.IMG" to="$pid.cub") || status=import_fail; }
        [ "$status" = ok ] && {
            ts=$(step spiceinit spiceinit from="$pid.cub" web="${SPICE_WEB:-no}") || status=spiceinit_fail; }
        [ "$status" = ok ] && {
            tc=$(step ctxcal ctxcal from="$pid.cub" to="$pid.cal.cub") || status=ctxcal_fail; }
        [ "$status" = ok ] && {
            te=$(step ctxevenodd ctxevenodd from="$pid.cal.cub" to="$pid.eo.cub") || status=evenodd_fail; }
        [ "$status" = ok ] && {
            tm=$(step cam2map cam2map from="$pid.eo.cub" to="$pid.map.cub" \
                 map="$MAP" pixres=map) || status=cam2map_fail; }
        tt=$(echo "$(now) - $T0" | bc)
        mmb=$(echo "scale=1; $(stat -c%s "$pid.map.cub" 2>/dev/null || echo 0)/1000000" | bc)
        echo "$pid,$mb,$td,$ti,$ts,$tc,$te,$tm,$tt,$mmb,$status" >> "$OUT_STATUS"
        echo "    $status  total ${tt}s"
        rm -f "$pid.IMG" "$pid.cub" "$pid.cal.cub" "$pid.eo.cub"
        # .map.cub is kept: the extract step needs it
    fi
    FRAME_NUM=$((FRAME_NUM + 1))
done

echo
echo "=== task $TASK_ID summary ($OUT_STATUS) ==="
awk -F, 'NR>1 {n++; t+=$9; if($11=="ok") ok++}
    END { if(n) printf "frames: %d  ok: %d  mean: %.0fs\n", n, ok, t/n }' "$OUT_STATUS"
