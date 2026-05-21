# hirise2ctx

HiRISE boulder detections → CTX rock-abundance paired dataset.
See [CLAUDE.md](CLAUDE.md) for the full build spec; see [DECISIONS.md](DECISIONS.md)
for runtime-verified facts and deviations.

## Status

**Week 1, Stage 0–1 walking skeleton.**

- Manifest load + per-image detection ingest + reprojection to common CTX CRS.
- Sanity check that the CRS chain catches the local-Mars-radius bug.
- One image (`ESP_047976_2020`) verified end-to-end before scaling out.

Stages 2 (CTX retrieval), 3 (co-registration), 4 (labeling), 5 (packaging) are not yet
implemented.

## Setup

```powershell
# uses the existing `geospatial` conda env (GDAL, rasterio, geopandas, pyproj, shapely)
& "C:\Users\brian\anaconda3\Scripts\conda.exe" run -n geospatial pip install -e .
```

`conda` is not on PATH in fresh shells on this machine — invoke `conda.exe` by absolute
path or use the snippet above.

## Run

```powershell
$conda = "C:\Users\brian\anaconda3\Scripts\conda.exe"

# Fast unit tests
& $conda run -n geospatial pytest tests/ -m "not slow" -v

# Slow integration test: end-to-end Stage 0-1 on ESP_047976_2020,
# including the CRS sanity check. Touches the network once (Stage 0.5 CTX header probe)
# and the local detection shapefile.
& $conda run -n geospatial pytest tests/test_sanity_residual_one_image.py -v

# QA notebook (renders polygon footprint + manifest center overlay, writes PNG)
& $conda run -n geospatial jupyter nbconvert --to notebook --execute notebooks/01_detections_qa.ipynb --inplace
```

## Layout

```
src/
  config.py          # load/validate YAML; SHA256 config hash for provenance
  manifest.py        # read hirise_priority10.csv; resolve per-ObsId shapefile
  pds_labels.py      # fetch + cache + parse HiRISE .LBL (authoritative metadata)
  ctx_retrieve.py    # Stage 0.5: Murray Lab URL discovery + header-only CRS probe
  detections.py      # Stage 1: glob shapefile, read per-image .prj (auto-corrects
                     # buggy `D_unnamed` exports via PDS LBL), reproject, cache GPKG
  qa.py              # assert_centroid_consistent sanity check
tests/               # unit tests + one slow integration test
notebooks/           # QA notebooks; figures saved to reports/figures/
cache/
  pds_labels/        # cached .LBL text files (~10-20 KB each)
  reprojected_detections/  # per-ObsId GPKG + provenance JSON
dataset/             # paired output dataset (populated by later stages)
```

## How to grow the dataset

Adding a new image is two steps and zero code changes:

1. Add a row to `hirise_priority10.csv` with at minimum `ObsId`, `ProductId`,
   `BoulderLabel`, `CenterLat`, `CenterLon_180`, `CenterLon_360`, `CTX_TileName`,
   and the URL columns.
2. Drop the BoulderNet detections folder under `detections_root/{ObsId}/` containing
   a `*-mask-nms.shp` (with sidecar `.prj`, `.dbf`, `.shx`).

Then re-run the pipeline.
