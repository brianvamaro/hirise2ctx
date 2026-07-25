#!/bin/bash
# run_f_region_stagea.sbatch -- PLAN_FBuild Stage A: ISIS-calibrate + project the 907 region
# frames. Reuses the leg-B array worker (f_leg_b_process.sh, now FRAME_LIST-parameterized + summed-
# frame-safe) pointed at reports/figures/region_frame_list.csv. CPU-only. Resumable: each task
# skips frames whose .map.cub already exists, so re-submitting after a timeout or a kernel fetch
# just fills the gaps.
#
# Prereq (once): bash setup_isis_env.sh  (pins isis=10.0.0/csm=3.1.0, patches libcsmapi.so.3).
# Prereq (laptop, done 2026-07-23): scripts/f_build_framelist.py -> region_frame_list.csv (907).
#
# Usage:
#   cd ~/hirise2ctx && git pull && mkdir -p logs
#   sbatch run_f_region_stagea.sbatch
#   # watch:   squeue --me ;  tail -f logs/h2c-region-A-*_*.out
#   # result:  $SCRATCH/hirise2ctx/f_region/{PRODUCT_ID}.map.cub  (kept) + status_${TASK}.csv
#
# Kernel gap (the 2018+ CKs the July mirror lacks): the first pass will spiceinit_fail on those
# frames and LOG the needed kernel names. Then, on a login node:
#   cat $SCRATCH/hirise2ctx/f_region/isis_*.log > /tmp/region_isis.log
#   bash f_fetch_kernels.sh /tmp/region_isis.log     # fetches the missing CK/SPK by name
#   sbatch run_f_region_stagea.sbatch                # resumes; fills the spiceinit_fail holes
# Repeat until the status census is clean (SHERLOCK_RUN Part H).
#
# Sizing: 907 frames, ~10-20 min/frame (probe: long frames ~28 min, population avg shorter).
#   32 tasks -> ~28 frames/task -> ~7-9 h (this default; --time 12h headroom).
#   For faster wall-clock if the queue allows: --array=0-63 and --time=06:00:00 (~14 frames/task).
#   Resumability means an over-run task is cheap to re-submit -- pick whatever the queue grants.
#
#SBATCH --job-name=h2c-region-A
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/h2c-region-A-%A_%a.out
#SBATCH --error=logs/h2c-region-A-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs

GROUP_HOME="${GROUP_HOME:-/home/groups/mlapotre}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$GROUP_HOME/$USER/micromamba}"
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"
set +u; micromamba activate isis; set -u
export ISISROOT="$CONDA_PREFIX"
export ISISDATA="${ISISDATA_DIR:-$SCRATCH/isisdata}"

cd "$HOME/hirise2ctx"

N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
echo "host $(hostname)  task $I/$N  ISISROOT=$ISISROOT  ISISDATA=$ISISDATA"

FRAME_LIST="$HOME/hirise2ctx/reports/figures/region_frame_list.csv" \
    TASK_ID=$I N_TASKS=$N bash f_leg_b_process.sh "$SCRATCH/hirise2ctx/f_region"

echo "task $I done"
