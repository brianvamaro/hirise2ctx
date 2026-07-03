#!/bin/bash
# f_timing_test.sh -- F de-risk: time the EDR -> ctxcal -> cam2map pipeline on 10 real frames
# (PLAN_StripingArtifact, decision step 2). Consumes reports/f_timing/frame_list.csv (built on
# the laptop by scripts/f_edr_frame_list.py --verify), writes per-step timings + a summary with
# the x907 (regional) / x86571 (global) extrapolation.
#
# Usage (inside the isis micromamba env -- see setup_isis_env.sh, or via run_f_timing.sbatch):
#   bash f_timing_test.sh [workdir]        # default workdir $SCRATCH/hirise2ctx/f_timing
# Env: KEEP_CUBES=1 keeps the projected .map.cub outputs (default: delete, keep only timings).
set -uo pipefail   # NOT -e: one bad frame must not kill the test; failures are data here.

REPO="$(cd "$(dirname "$0")" && pwd)"
LIST="$REPO/reports/f_timing/frame_list.csv"
WORK="${1:-${SCRATCH:-/tmp}/hirise2ctx/f_timing}"
OUT="$WORK/timing.csv"
MAP="$REPO/f_equirect.map"
[ -f "$LIST" ] || { echo "ERROR: $LIST missing (run scripts/f_edr_frame_list.py first)" >&2; exit 1; }
[ -n "${ISISDATA:-}" ] || { echo "ERROR: ISISDATA not set (activate the isis env; see setup_isis_env.sh)" >&2; exit 1; }
mkdir -p "$WORK"; cd "$WORK"

# header-name-based CSV column lookup (column order may drift)
col() { head -1 "$LIST" | tr ',' '\n' | grep -nx "$1" | cut -d: -f1; }
PID_COL=$(col PRODUCT_ID); URL_COL=$(col edr_url)

echo "product_id,edr_mb,t_download,t_import,t_spiceinit,t_ctxcal,t_evenodd,t_cam2map,t_total,map_mb,status" > "$OUT"
now() { date +%s.%N; }
step() {  # step <label> <cmd...>   -> echoes elapsed seconds, returns cmd status
    local label="$1"; shift          # consume the label; "$@" is now just the command
    local t0; t0=$(now)
    echo "--- $label ---" >> "$WORK/isis_steps.log"
    "$@" >> "$WORK/isis_steps.log" 2>&1
    local rc=$?
    echo "$(now) - $t0" | bc
    return $rc
}

tail -n +2 "$LIST" | while IFS= read -r line; do
    pid=$(echo "$line" | cut -d, -f"$PID_COL")
    url=$(echo "$line" | cut -d, -f"$URL_COL")
    echo "=== $pid ==="
    T0=$(now); status=ok
    td=$(step "download($pid)" curl -fsSL --retry 3 -o "$pid.IMG" "$url") || status=download_fail
    mb=$(echo "scale=1; $(stat -c%s "$pid.IMG" 2>/dev/null || echo 0)/1000000" | bc)
    ti=0; ts=0; tc=0; te=0; tm=0
    [ "$status" = ok ] && { ti=$(step import mroctx2isis from="$pid.IMG" to="$pid.cub") || status=import_fail; }
    [ "$status" = ok ] && { ts=$(step spiceinit spiceinit from="$pid.cub" web=yes) || status=spiceinit_fail; }
    [ "$status" = ok ] && { tc=$(step ctxcal ctxcal from="$pid.cub" to="$pid.cal.cub") || status=ctxcal_fail; }
    [ "$status" = ok ] && { te=$(step ctxevenodd ctxevenodd from="$pid.cal.cub" to="$pid.eo.cub") || status=evenodd_fail; }
    [ "$status" = ok ] && { tm=$(step cam2map cam2map from="$pid.eo.cub" to="$pid.map.cub" map="$MAP" pixres=map) || status=cam2map_fail; }
    tt=$(echo "$(now) - $T0" | bc)
    mmb=$(echo "scale=1; $(stat -c%s "$pid.map.cub" 2>/dev/null || echo 0)/1000000" | bc)
    echo "$pid,$mb,$td,$ti,$ts,$tc,$te,$tm,$tt,$mmb,$status" >> "$OUT"
    echo "    $status  total ${tt}s (dl $td, import $ti, spice $ts, cal $tc, eo $te, map $tm)"
    rm -f "$pid.IMG" "$pid.cub" "$pid.cal.cub" "$pid.eo.cub"
    [ "${KEEP_CUBES:-0}" = "1" ] || rm -f "$pid.map.cub"
done

echo; echo "=== SUMMARY ($OUT) ==="
awk -F, 'NR>1 {n++; t+=$9; if($11=="ok") ok++}
    END { if(n==0) exit
          printf "frames: %d  ok: %d  mean total: %.0f s/frame\n", n, ok, t/n
          printf "extrapolation (serial 1 CPU-task): regional 907 frames = %.0f h;", 907*t/n/3600
          printf "  global 86571 frames = %.0f h (job-array parallel /N)\n", 86571*t/n/3600 }' "$OUT"
column -t -s, "$OUT"
