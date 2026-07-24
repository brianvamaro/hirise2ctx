#!/bin/bash
# run_f_build_probe.sbatch -- PLAN_FBuild V1/V5 sizing probe, Stage A (ISIS on the 5 selected
# frames, KEEP_CUBES=1 so Stage B can embed them). CPU-only; reuses f_timing_test.sh pointed at
# the sizing frame list. Prereq (once): bash setup_isis_env.sh + bash f_fetch_kernels.sh; and on
# the laptop: python scripts/f_build_framelist.py && python scripts/f_build_sizing_frames.py.
#   sbatch run_f_build_probe.sbatch
# Then (GPU, laptop or Sherlock): transfer $SCRATCH/hirise2ctx/f_build_probe/*.map.cub (+timing.csv)
#   and run scripts/f_build_sizing_probe.py --frames-dir <dir> --timing-csv <timing.csv>
#SBATCH --job-name=h2c-fbuild-probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/h2c-fbuild-probe-%j.out
#SBATCH --error=logs/h2c-fbuild-probe-%j.out
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
echo "Host: $(hostname)  ISISROOT=$ISISROOT  ISISDATA=$ISISDATA"

# point the timing worker at the 5-frame sizing list; keep the projected cubes for Stage B.
export FRAME_LIST="$HOME/hirise2ctx/reports/f_build/sizing_frame_list.csv"
export KEEP_CUBES=1
bash f_timing_test.sh "$SCRATCH/hirise2ctx/f_build_probe"
echo "done -- timing.csv + {pid}.map.cub in $SCRATCH/hirise2ctx/f_build_probe"
echo "next: convert cubes to GeoTIFF (isis2std / gdal_translate) and run f_build_sizing_probe.py on a GPU"
