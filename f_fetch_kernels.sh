#!/bin/bash
# f_fetch_kernels.sh -- fetch the MRO SPICE kernels the 10 timing frames need, so spiceinit
# runs LOCAL (web=no). The ISIS web-SPICE server is version-pinned and answers our ISIS 10
# client with "The SPICE server returned incompatible SPICE data" (DECISIONS 2026-07-02c) --
# but its response still NAMES the right kernels, and the failed run logged them. This pulls
# only: the small kernel dirs (sclk/fk/iak/pck/ik) + the ck/spk selection-db files + the
# specific weekly CK / psp SPK files named in the log + the ctxcal calibration files
# (~1-2 GB total, vs 100s of GB for the full mro area).
#
# Run on a LOGIN node (needs internet), after the timing test has produced isis_steps.log:
#   bash f_fetch_kernels.sh [path/to/isis_steps.log]
# then re-submit:  sbatch run_f_timing.sbatch   (spiceinit defaults to web=no now)
set -euo pipefail

LOG="${1:-${SCRATCH:-/tmp}/hirise2ctx/f_timing/isis_steps.log}"
GROUP_HOME="${GROUP_HOME:-/home/groups/mlapotre}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$GROUP_HOME/$USER/micromamba}"
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"
set +u; micromamba activate isis; set -u
export ISISDATA="${ISISDATA_DIR:-$SCRATCH/isisdata}"

[ -f "$LOG" ] || { echo "ERROR: $LOG not found -- run the timing test once (it logs the" \
    "server-resolved kernel names in its spiceinit blocks) or pass the log path." >&2; exit 1; }

# the specific big time-range kernels (weekly CKs, psp SPKs) the server resolved per frame
mapfile -t FILES < <(grep -o '\$mro/kernels/[^ ,)"]*' "$LOG" | sed 's|^\$mro/||' | sort -u)
echo "kernel files named in the log: ${#FILES[@]}"
printf '  %s\n' "${FILES[@]}"
[ "${#FILES[@]}" -gt 0 ] || { echo "ERROR: no \$mro/kernels/... paths found in $LOG" >&2; exit 1; }

# one rclone include filter (brace list): small dirs whole, dbs, named files, ctxcal files
PATTERNS=( "calibration/**"
           "kernels/sclk/**" "kernels/fk/**" "kernels/iak/**" "kernels/pck/**" "kernels/ik/**"
           "kernels/ck/kernels.*" "kernels/spk/kernels.*" )
PATTERNS+=( "${FILES[@]}" )
BRACES="{$(IFS=,; echo "${PATTERNS[*]}")}"

downloadIsisData mro "$ISISDATA" --include="$BRACES"

echo "OK -- kernels + calibration in $ISISDATA/mro"
echo "next:  sbatch run_f_timing.sbatch    (spiceinit now runs web=no against these)"
