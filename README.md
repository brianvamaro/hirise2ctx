# hirise2ctx

HiRISE boulder detections → CTX rock-abundance paired dataset.
See [CLAUDE.md](CLAUDE.md) for the full build spec; see [DECISIONS.md](DECISIONS.md)
for runtime-verified facts and deviations.

## Status

**Stages 0–3 of 5 done.**

| Stage | What | Status |
|---|---|---|
| 0 | Load manifest + config | ✓ |
| 1 | Per-image detection ingest + reproject to common CTX CRS (auto-corrects upstream HiRISE PDS `Standard_Parallel_1=0` bug, polygon side) | ✓ all 10 manifest rows |
| 2 | Download Murray Lab CTX tile + window around HiRISE footprint + HiRISE coverage mask (auto-corrects same SP1 bug on JP2 side) | ✓ full sweep |
| 3 | Co-registration (sub-pixel phase-correlation translation) | ✓ full sweep, thresholds TBD (see `notebooks/05_coregistration_qa.ipynb`) |
| 4 | Label generation (nested grids, configurable `label_type`) | — not started |
| 5 | Packaging + group-aware splits | — not started |

65 pytest pass. ~10 GB of CTX tiles + ~3 GB of HiRISE JP2s cached locally for the full priority10 manifest. Stage 3 solved 9/10 ObsIds with shifts in 118–273 m (CLAUDE.md target ~200 m); ESP_057469_2215 fails gracefully due to a polygon-bbox tile straddle (see DECISIONS.md).

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

# Stage 2 for the entire manifest in one go.
& $conda run -n geospatial python scripts/sweep_stage2.py

# Stage 3 (co-registration) for one ObsId — needs the matching Stage 2 caches.
# Solves a sub-pixel rigid translation (dx, dy) and writes
# cache/coregistration/{ObsId}.json with shift + peak correlation + provenance.
& $conda run -n geospatial python scripts/run_stage3.py ESP_069669_2220

# Stage 3 for every ObsId whose Stage 2 caches exist.
& $conda run -n geospatial python scripts/run_stage3.py --all

# Render QA notebooks (overlay polygons / mask / zooms on the CTX window;
# Stage 3 before/after shifts).
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace notebooks/04_ctx_retrieval_qa.ipynb
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace notebooks/05_coregistration_qa.ipynb
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
  hirise_imagery.py  # JP2 cache + decimated read helpers (auto-applies the SP1
                     # corrected CRS from Stage 1 sidecars; used by Stage 2 mask
                     # and Stage 3 co-registration)
  coregister.py      # Stage 3: warp HiRISE onto CTX grid, pick a power-of-2 FFT
                     # window, sub-pixel phase-correlate, cache (dx, dy) per ObsId
  qa.py              # assert_centroid_consistent sanity check
scripts/
  run_stage2.py      # headless per-ObsId Stage 2 driver
  sweep_stage2.py    # full-manifest Stage 2 sweep (sequential, skips cached)
  run_stage3.py      # headless Stage 3 driver (single ObsId or --all)
tests/               # integration tests skip until caches exist
notebooks/
  01_detections_qa.ipynb                  # Stage 1 overlay
  02_investigate_misplaced_detections.ipynb  # the SP1 bug, before the fix
  03_hirise_overlay.ipynb                 # decimated HiRISE imagery overlay
  04_ctx_retrieval_qa.ipynb               # Stage 2: window + mask + zooms
  05_coregistration_qa.ipynb              # Stage 3: shift distribution + before/after
cache/                # (gitignored) regenerable artifacts
  pds_labels/                  # PDS .LBL text files (~10-20 KB each)
  reprojected_detections/      # per-ObsId GPKG + provenance JSON (Stage 1)
  ctx_tiles/                   # Murray Lab zipped tiles + JSON sidecar (Stage 2)
  ctx_windows/                 # per-ObsId CTX GeoTIFF + HiRISE coverage mask
                               # + provenance JSON (Stage 2)
  hirise_jp2/                  # cached HiRISE JP2s (~200-500 MB each)
  hirise_decimated/            # 5 mpp HiRISE GeoTIFFs for co-registration etc.
  coregistration/              # per-ObsId Stage 3 shift JSON (dx, dy in m + px,
                               # peak correlation, FFT window placement)
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

- **HiRISE `.prj` / JP2 SP1 bug** ([DECISIONS.md](DECISIONS.md) 2026-05-20, 2026-05-22):
  4 of 10 BoulderNet shapefiles ship with `Standard_Parallel_1 = 0` (datum
  `D_unnamed`) even though their geometry was generated with the PDS-declared
  projection latitude. The matching JP2s inherit the same buggy metadata.
  `src/detections.py` corrects the shapefile side; `src/hirise_imagery.py` applies
  the symmetric correction at JP2 read time, replacing the JP2's embedded CRS with
  the Stage 1 sidecar's corrected CRS. Caches built before the JP2-side fix are
  detected via a literal-SP1 comparison and rebuilt automatically (pyproj's
  `.equals()` canonicalizes spherical Equirectangular without SP1, so the literal
  parse is necessary). Don't "fix" either override thinking it's wrong — both are
  auto-correcting the same upstream issue.
- **Murray Lab tile URL form** ([DECISIONS.md](DECISIONS.md) 2026-05-21/22): the URL
  uses a signed-prefix zero-padded form universally. Positive longitudes get
  `E<abs:03d>` (e.g. `E000`, `E012`, `E160`); negative longitudes get `E-<abs:03d>`
  (e.g. `E-040`, NOT `W040`); latitudes follow the same rule at 2 digits (`N20`,
  `N-08`). `ctx_retrieve.ensure_tile_cached` tries the bare `manifest_to_murray`
  output first, then falls back to this canonical form on 404.
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
