#!/bin/bash
# setup_isis_env.sh -- ISIS environment on Sherlock for the F de-risk timing test
# (PLAN_StripingArtifact: per-source-frame inference). ISIS ships only via conda channels
# (the USGS `usgs-astrogeology` channel, deps from conda-forge) and Sherlock discourages
# system conda, so this uses MICROMAMBA (one static binary, no root, no module). CPU-only.
# Idempotent. Run on a login or sh_dev node.
#
#   bash setup_isis_env.sh
#
# Afterwards each job/session activates with:
#   export MAMBA_ROOT_PREFIX=$GROUP_HOME/$USER/micromamba
#   eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)"
#   micromamba activate isis
#   export ISISROOT=$CONDA_PREFIX ISISDATA=<data dir printed below>
set -euo pipefail

# Login nodes have a per-user process/thread cgroup cap that kills micromamba's parallel
# download AND extract phases with "Resource temporarily unavailable" (hit twice 2026-07-02,
# with thread caps applied). Run this in a compute session:  sh_dev -c 4
case "$(hostname)" in
    *-ln*) echo "WARNING: $(hostname) is a LOGIN node -- the ISIS install has died here twice" >&2
           echo "         with EAGAIN. Strongly recommended:  sh_dev -c 4  then re-run." >&2 ;;
esac

GROUP_HOME="${GROUP_HOME:-/home/groups/mlapotre}"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$GROUP_HOME/$USER/micromamba}"
# ISISDATA on scratch: base area is ~10 GB and it's re-downloadable.
ISISDATA_DIR="${ISISDATA_DIR:-$SCRATCH/isisdata}"

# 1) micromamba binary (static; no install step)
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
mkdir -p "$MAMBA_ROOT/bin"
if [ ! -x "$MAMBA_ROOT/bin/micromamba" ]; then
    echo "fetching micromamba ..."
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$MAMBA_ROOT" bin/micromamba
fi
eval "$("$MAMBA_ROOT/bin/micromamba" shell hook -s bash)"

# 2) ISIS env. The `isis` package lives on the USGS `usgs-astrogeology` channel (NOT
#    conda-forge -- that channel only supplies the dependencies), so both channels are
#    required, USGS first. --download-threads is capped because Sherlock LOGIN nodes have a
#    tight per-user process/thread limit and micromamba's default parallel downloader dies
#    there with "Resource temporarily unavailable" (seen 2026-07-02 on a ~400-pkg
#    transaction). Re-running is safe: the package cache resumes. If it still hits the
#    limit, run this script inside a compute session instead:  sh_dev -c 4
#    Completeness is checked via a sentinel BINARY, not `env list`: a crashed install leaves a
#    registered-but-empty env behind (hit 2026-07-02), and `create -y` on it just re-runs the
#    transaction from the package cache.
if [ ! -x "$MAMBA_ROOT/envs/isis/bin/mroctx2isis" ]; then
    # thread caps via env vars: current micromamba has no --download-threads CLI flag
    MAMBA_DOWNLOAD_THREADS=1 MAMBA_EXTRACT_THREADS=1 \
        micromamba create -y -n isis -c usgs-astrogeology -c conda-forge isis
fi
# activate with nounset OFF: the env's bundled activate.d hooks (e.g. libpdal-core) reference
# unset vars like PDAL_DRIVER_PATH and abort the script under `set -u`
set +u; micromamba activate isis; set -u
export ISISROOT="$CONDA_PREFIX"
echo "ISIS $(head -1 "$ISISROOT/version" 2>/dev/null || echo '?') at $ISISROOT"

# 3) ISIS data. spiceinit runs with web=yes (no local SPICE kernels needed), so we only need
#    the small non-kernel areas: `base` (leap seconds, templates, ~GBs) and the MRO calibration
#    files ctxcal reads (the v0003 flat field lives there). downloadIsisData supports rclone
#    filter args after `--`; if the targeted pull errors, fall back to plain `base` + retry the
#    timing test -- its ctxcal step will name any file still missing.
mkdir -p "$ISISDATA_DIR"
export ISISDATA="$ISISDATA_DIR"
if [ ! -d "$ISISDATA_DIR/base" ]; then
    echo "downloading ISIS base data area -> $ISISDATA_DIR (~10 GB) ..."
    downloadIsisData base "$ISISDATA_DIR"
fi
if [ ! -d "$ISISDATA_DIR/mro/calibration" ]; then
    echo "downloading MRO calibration area (small; NOT the huge kernels) ..."
    downloadIsisData mro "$ISISDATA_DIR" --include="calibration/**" \
        || echo "WARNING: targeted mro pull failed; f_fetch_kernels.sh retries it, or run\
 'downloadIsisData mro $ISISDATA_DIR' manually if ctxcal later reports a missing file."
fi
# NOTE: SPICE kernels are NOT downloaded here. The web-SPICE service is version-pinned
# (rejects our ISIS 10 client with "incompatible SPICE data"), so the timing test runs
# spiceinit web=no against a targeted local fetch -- see f_fetch_kernels.sh.

# 4) smoke test: the four apps the timing pipeline uses resolve + run
for app in mroctx2isis spiceinit ctxcal ctxevenodd cam2map; do
    command -v "$app" >/dev/null || { echo "ERROR: $app not on PATH" >&2; exit 1; }
done
echo "OK -- ISIS ready. ISISDATA=$ISISDATA_DIR"
echo "next: sbatch run_f_timing.sbatch"
