# hirise2ctx

HiRISE boulder detections → CTX rock-abundance paired dataset.
See [CLAUDE.md](CLAUDE.md) for the full build spec; see [DECISIONS.md](DECISIONS.md)
for runtime-verified facts and deviations.

## Status

**Stages 0–2 of 5 done; verified end-to-end on ESP_069669_2220.**

| Stage | What | Status |
|---|---|---|
| 0 | Load manifest + config | ✓ |
| 1 | Per-image detection ingest + reproject to common CTX CRS (auto-corrects upstream HiRISE PDS `Standard_Parallel_1=0` bug) | ✓ all 10 manifest rows |
| 2 | Download Murray Lab CTX tile + window around HiRISE footprint + HiRISE coverage mask | ✓ ESP_069669_2220 (other 9 unblocked) |
| 3 | Co-registration (phase-correlation translation) | — not started |
| 4 | Label generation (nested grids, configurable `label_type`) | — not started |
| 5 | Packaging + group-aware splits | — not started |

38 pytest pass. One CTX tile cached (`E000_N40`, ~1.7 GB), two HiRISE JP2s cached.

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

# All tests (fast unit + slow integration; ~45 s total).
# Slow tests auto-skip when their cache prerequisites are missing.
& $conda run -n geospatial pytest tests/ -v

# Fast unit tests only.
& $conda run -n geospatial pytest tests/ -m "not slow" -v

# Stage 2 for one ObsId. First call per Murray Lab tile downloads the tile zip
# (~1.5 GB; cached at cache/ctx_tiles/{murray_tile}.zip). First call per ObsId
# also downloads the HiRISE JP2 (~200-500 MB; cached at cache/hirise_jp2/).
# Subsequent calls for the same ObsId reuse both caches.
& $conda run -n geospatial python scripts/run_stage2.py ESP_069669_2220

# Render QA notebooks (overlay polygons / mask / zooms on the CTX window).
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace notebooks/04_ctx_retrieval_qa.ipynb
```

## Layout

```
src/
  config.py          # load/validate YAML; SHA256 config hash for provenance
  manifest.py        # read hirise_priority10.csv; resolve per-ObsId shapefile
  pds_labels.py      # fetch + cache + parse HiRISE .LBL (authoritative metadata)
  detections.py      # Stage 1: read shapefile (auto-corrects buggy `D_unnamed`
                     # .prj via PDS LBL), reproject, cache GPKG
  ctx_tiles.py       # manifest <-> Murray Lab tile-name translator
  ctx_retrieve.py    # Stage 2: download tile zip, window + write CTX GeoTIFF,
                     # warp HiRISE -> CTX grid to build coverage mask
  hirise_imagery.py  # JP2 cache + decimated read helpers (used by Stage 2 mask
                     # and future Stage 3 co-registration)
  qa.py              # assert_centroid_consistent sanity check
scripts/
  run_stage2.py      # headless per-ObsId Stage 2 driver with progress heartbeat
tests/               # 38 tests; integration ones skip until caches exist
notebooks/
  01_detections_qa.ipynb                  # Stage 1 overlay
  02_investigate_misplaced_detections.ipynb  # the SP1 bug, before the fix
  03_hirise_overlay.ipynb                 # decimated HiRISE imagery overlay
  04_ctx_retrieval_qa.ipynb               # Stage 2: window + mask + zooms
cache/                # (gitignored) regenerable artifacts
  pds_labels/                  # PDS .LBL text files (~10-20 KB each)
  reprojected_detections/      # per-ObsId GPKG + provenance JSON (Stage 1)
  ctx_tiles/                   # Murray Lab zipped tiles + JSON sidecar (Stage 2)
  ctx_windows/                 # per-ObsId CTX GeoTIFF + HiRISE coverage mask
                               # + provenance JSON (Stage 2)
  hirise_jp2/                  # cached HiRISE JP2s (~200-500 MB each)
  hirise_decimated/            # 5 mpp HiRISE GeoTIFFs for co-registration etc.
dataset/
  DATA_DICTIONARY.md           # schema reference for cached artifacts
reports/figures/     # PNGs from QA notebooks
config.yaml          # single source of truth for pipeline parameters
CLAUDE.md            # spec (authoritative)
DECISIONS.md         # runtime-verified facts and deviations
```

## How to grow the dataset

Adding a new image is two steps and zero code changes:

1. Add a row to `hirise_priority10.csv` with at minimum `ObsId`, `ProductId`,
   `BoulderLabel`, `CenterLat`, `CenterLon_180`, `CenterLon_360`, `CTX_TileName`,
   `JP2_URL`, `LabelURL`, and the other URL columns.
2. Drop the BoulderNet detections folder under `detections_root/{ObsId}/` containing
   a `*-mask-nms.shp` (with sidecar `.prj`, `.dbf`, `.shx`).

Then run Stage 1 (`pytest tests/test_sanity_residual_one_image.py` covers the verified-good probe;
new rows go through `detections.stage1_one_image`) and Stage 2 (`scripts/run_stage2.py`).

## Gotchas to read first

These have all bitten us once already and are worth knowing before you touch the code:

- **HiRISE `.prj` SP1 bug** ([DECISIONS.md](DECISIONS.md) 2026-05-20): 4 of 10 BoulderNet
  shapefiles ship with `Standard_Parallel_1 = 0` (datum `D_unnamed`) even though their
  geometry was generated with the PDS-declared projection latitude. `src/detections.py`
  detects this and overrides SP1 with `CENTER_LATITUDE` from the PDS `.LBL`. Don't
  "fix" the override thinking it's wrong — it's auto-correcting an upstream bug.
- **Murray Lab tile URL form** ([DECISIONS.md](DECISIONS.md) 2026-05-21): the URL uses
  the *padded* manifest form (`E000_N40.zip`), not the bare signed-int form
  (`E0_N40.zip`). `ctx_retrieve.ensure_tile_cached` tries the murray form first and
  falls back to the padded form on 404.
- **CTX mosaic CRS surprise**: the actual Murray Lab CRS is
  `Mars_2015_Ocentric_Equirectangular` with inverse flattening 169.894 (oblate), not the
  IAU 2000 sphere we configured for `target_crs`. Sub-pixel discrepancy at 5 m/px so not
  blocking; revisit during Stage 3 if phase correlation shows a systematic
  equator-to-pole bias.
- **HiRISE-coverage constraint** ([DECISIONS.md](DECISIONS.md) 2026-05-21): Stage 4 must
  drop any tile where the cached `{ObsId}_hirise_mask.tif` isn't fully 1. The
  polygon-bbox CTX window contains ~40% HiRISE-unobserved area on average; counting
  those as "no boulders" would inflate zero-tile counts (it's "unobserved", not
  "absent").
- **`/vsicurl/` is 140× slower than bulk download** for HiRISE JP2s on this network.
  Stage 2 uses `download_then_window` mode; don't switch to `/vsicurl/` without
  re-benchmarking. Disk usage of cached tiles + JP2s (~12 GB if all 10 ObsIds get
  fetched) is acceptable.
- **`round_shape` deprecation**: `rasterio.windows.Window.round_shape()` is gone in
  2.x. Use `int(round(window.width))` / `height` manually (see
  `ctx_retrieve.extract_ctx_window`).
- **Source TIFF block size**: Murray Lab tiles have a full-raster internal block
  (47420×47420). Naively copying `src.profile` into a small windowed output triggers
  `_TIFFVSetField: Bad value 47420 for "TileWidth"`. Drop `blockxsize`/`blockysize`
  from the profile before writing.
