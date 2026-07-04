#!/bin/bash
# run_f_leg_b.sbatch -- F pilot leg B: ISIS pipeline for the training cohort frames.
# Runs f_leg_b_process.sh as a Slurm job array so each task processes its stride of
# cohort_frame_list.csv (round-robin; load-balanced across variable frame sizes).
#
# Prerequisite: bash setup_isis_env.sh (once); reports/f_leg_b/cohort_frame_list.csv
# (built on the laptop by: conda run -n geospatial python scripts/f_leg_b_frame_list.py)
#
# Usage:
#   mkdir -p logs
#   sbatch run_f_leg_b.sbatch
#   # after all tasks complete:
#   ml python/3.12.1 && source /home/groups/mlapotre/$USER/envs/hirise2ctx/bin/activate
#   cd $HOME/hirise2ctx && python scripts/f_leg_b_extract.py
#   # then tar + scp obs_crops back to the laptop
#
# Wall-clock sizing (81 frames, 32 tasks):
#   81 frames / 32 tasks ≈ 2-3 frames/task × 22 min ≈ 55 min → ~1 h
#   Reduce to 0-23 (24 tasks) if the queue is constrained (→ ~75 min).
#
#SBATCH --job-name=h2c-f-legb
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/h2c-f-legb-%A_%a.out
#SBATCH --error=logs/h2c-f-legb-%A_%a.out
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

TASK_ID=$I N_TASKS=$N bash f_leg_b_process.sh "$SCRATCH/hirise2ctx/f_leg_b"

echo "task $I done"
