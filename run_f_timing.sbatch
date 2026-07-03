#!/bin/bash
# run_f_timing.sbatch -- F de-risk timing test (PLAN_StripingArtifact): EDR -> ctxcal ->
# cam2map on 10 real frames. CPU-only (ISIS); no GPU. Prereq: bash setup_isis_env.sh (once).
#   sbatch run_f_timing.sbatch
#SBATCH --job-name=h2c-f-timing
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/h2c-f-timing-%j.out
#SBATCH --error=logs/h2c-f-timing-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs

GROUP_HOME="${GROUP_HOME:-/home/groups/mlapotre}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$GROUP_HOME/$USER/micromamba}"
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"
# activate with nounset OFF: the env's activate.d hooks (libpdal-core) reference unset vars
# like PDAL_DRIVER_PATH and would abort under `set -u`
set +u; micromamba activate isis; set -u
export ISISROOT="$CONDA_PREFIX"
export ISISDATA="${ISISDATA_DIR:-$SCRATCH/isisdata}"

cd "$HOME/hirise2ctx"
echo "Host: $(hostname)  ISISROOT=$ISISROOT  ISISDATA=$ISISDATA"
bash f_timing_test.sh
echo "done -- timing.csv in ${SCRATCH}/hirise2ctx/f_timing (copy back to reports/f_timing/)"
