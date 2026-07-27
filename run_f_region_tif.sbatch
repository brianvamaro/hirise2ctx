#!/bin/bash
# run_f_region_tif.sbatch -- convert the 907 Stage-A .map.cub -> .map.tif (GeoTIFF/LZW) IN PLACE
# (same dir), because GDAL's ISIS3 driver SIGSEGVs on the large windowed reads Stage B does
# (DECISIONS 2026-07-27: native 2048^2 reads OK, 4096^2 crash). GTiff reads are solid at any size,
# and gdal_translate reads the cube in its native blocks so the conversion itself doesn't hit the
# bug. CPU array, round-robin stride of region_frame_list.csv, resumable (skips existing .map.tif).
# Stage B then prefers the .map.tif automatically (f_region_stageb.py resolver).
#
# Usage:  cd ~/hirise2ctx && git pull && mkdir -p logs && sbatch run_f_region_tif.sbatch
# ~907 frames * ~45 s / 32 tasks ~= 20-30 min. Adds ~1-1.5 TB of tifs next to the cubes (scratch ample).
#
#SBATCH --job-name=h2c-region-tif
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/h2c-region-tif-%A_%a.out
#SBATCH --error=logs/h2c-region-tif-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -uo pipefail   # NOT -e: one bad frame must not abort the task
mkdir -p logs

GROUP_HOME="${GROUP_HOME:-/home/groups/mlapotre}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$GROUP_HOME/$USER/micromamba}"
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"
set +u; micromamba activate isis; set -u    # the isis env ships gdal_translate
cd "$HOME/hirise2ctx"

DIR="$SCRATCH/hirise2ctx/f_region"
N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
I=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
echo "task $I/$N host $(hostname)  dir $DIR"

k=0
tail -n +2 reports/figures/region_frame_list.csv | tr -d '\r' | cut -d, -f1 | while IFS= read -r pid; do
    if (( k % N == I )); then
        if [ -f "$DIR/$pid.map.cub" ] && [ ! -f "$DIR/$pid.map.tif" ]; then
            if gdal_translate -q -of GTiff -co COMPRESS=LZW -co PREDICTOR=3 -co TILED=YES \
                   "$DIR/$pid.map.cub" "$DIR/$pid.map.tif.tmp"; then
                mv "$DIR/$pid.map.tif.tmp" "$DIR/$pid.map.tif"   # atomic: a killed task leaves no partial
                echo "ok $pid"
            else
                rm -f "$DIR/$pid.map.tif.tmp"; echo "FAIL $pid"
            fi
        fi
    fi
    k=$((k + 1))
done
echo "task $I done"
