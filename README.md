# hirise2ctx

HiRISE boulder detections → CTX rock-abundance paired dataset.
See [CLAUDE.md](CLAUDE.md) for the full build spec; see [DECISIONS.md](DECISIONS.md)
for runtime-verified facts and deviations; see [ROADMAP.md](ROADMAP.md) for the
phase-by-phase index of planning + status.

## Status

**End of CLAUDE.md Week 1-2 data pipeline scope: Stages 0–5 of 5 done.**

| Stage | What | Status |
|---|---|---|
| 0 | Load manifest + config | ✓ |
| 1 | Per-image detection ingest + reproject to common CTX CRS (auto-corrects upstream HiRISE PDS `Standard_Parallel_1=0` bug, polygon side) | ✓ all 10 manifest rows |
| 2 | Download Murray Lab CTX tile + window around HiRISE footprint + HiRISE coverage mask (auto-corrects same SP1 bug on JP2 side) | ✓ full sweep |
| 3 | Co-registration (sub-pixel phase-correlation translation) | ✓ 9/10 solved in 118–273 m (CLAUDE.md target ~200 m) |
| 4 | Label generation on nested ×2 grid (8/16/32/64 CTX px) | ✓ 488,554 finest tiles across 9 ObsIds |
| 4b | Per-tile CTX texture features (9 families) + bundled context patches | ✓ 643,910 feature rows + 3.3 GB context patches (S=32 and S=64) |
| 5 | Leave-image-out splits + dataset packaging | ✓ 2 schemes (`loio_9fold` + `loio_3fold_balanced`) materialised + group-leak assertion in QA |

**125 pytest pass** (fast unit + slow integration). ~10 GB of CTX tiles + ~3 GB of
HiRISE JP2s + ~3.3 GB of context patches + ~1.3 GB of packaged splits cached locally
for the full priority10 manifest. ESP_057469_2215 is excluded from Stage 4 / 4b / 5
sweeps because its polygon bbox straddles a Murray Lab tile boundary (see
[DECISIONS.md](DECISIONS.md) 2026-05-22 entry).

**Next: Week 3 modeling per [PLAN_modeling.md](PLAN_modeling.md).** LightGBM tabular
baseline on `dataset/packaged/loio_9fold/` + optional CNN baseline on the bundled
context patches.

## Setup

```powershell
# uses the existing `geospatial` conda env (GDAL, rasterio, geopandas, pyproj, shapely,
# scikit-image, scikit-learn, pyarrow)
& "C:\Users\brian\anaconda3\Scripts\conda.exe" run -n geospatial pip install -e .
```

`conda` is not on PATH in fresh shells on this machine — invoke `conda.exe` by absolute
path or use the snippet above. Direct invocation of the env's `python.exe` fails with
exit code 127 (env DLLs not on PATH without activation); always go through `conda run`.

## Run

```powershell
$conda = "C:\Users\brian\anaconda3\Scripts\conda.exe"

# All tests (fast unit + slow integration; ~70 s total).
# Slow tests auto-skip when their cache prerequisites are missing.
& $conda run -n geospatial pytest tests/ -v

# Fast unit tests only.
& $conda run -n geospatial pytest tests/ -m "not slow" -v

# ---- Stages 2-5 drivers (each one is independently re-runnable; later stages
#      auto-skip ObsIds whose earlier-stage caches are missing) ----

# Stage 2: CTX windowed retrieval. First call per Murray Lab tile downloads ~1.5 GB;
# first call per ObsId downloads the HiRISE JP2 (~200-500 MB). Subsequent calls reuse.
& $conda run -n geospatial python scripts/run_stage2.py ESP_069669_2220
& $conda run -n geospatial python scripts/sweep_stage2.py            # full manifest

# Stage 3: co-registration. Needs Stage 2 caches.
& $conda run -n geospatial python scripts/run_stage3.py ESP_069669_2220
& $conda run -n geospatial python scripts/run_stage3.py --all

# Stage 4: label generation on the nested x2 grid. Needs Stage 2 + 3. Cheap (~3s/ObsId).
& $conda run -n geospatial python scripts/run_stage4.py ESP_069669_2220
& $conda run -n geospatial python scripts/run_stage4.py --all

# Stage 4b: per-tile CTX texture features + bundled context patches. Needs Stage 4.
# ~20-35s per ObsId (GLCM dominates); ~3 min for the 9-image sweep.
& $conda run -n geospatial python scripts/run_stage4b.py ESP_069669_2220
& $conda run -n geospatial python scripts/run_stage4b.py --all

# Stage 5: build named split schemes + materialise per-fold parquets. Needs Stages 4 + 4b.
# ~23s for both schemes on the 9-image manifest.
& $conda run -n geospatial python scripts/run_stage5.py loio_9fold
& $conda run -n geospatial python scripts/run_stage5.py --all

# ---- QA notebooks (one per major stage; all import from src/) ----
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/01_detections_qa.ipynb              # Stage 1 overlay
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/04_ctx_retrieval_qa.ipynb           # Stage 2: window + mask + zooms
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/05_coregistration_qa.ipynb          # Stage 3: shift distribution + before/after
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/06_labeling_qa.ipynb                # Stage 4: heatmaps, target distribution, nested consistency
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/07_features_qa.ipynb                # Stage 4b: per-image heatmaps, correlation matrix, timing
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/08_features_explained.ipynb         # Stage 4b: per-family math/physics/why + stratified patch viewer
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/09_splits_qa.ipynb                  # Stage 5: fold composition + train-vs-test target dist + group-leak check
```

**Don't run two `nbconvert --execute` against the same notebook concurrently** — caused
~14 min hangs in earlier sessions when overlapping kernels contended for caches.

## Running a parameter sweep

The cheap stages (4 and 4b) are deliberately separated from the expensive ones (2 and
3) so a config sweep doesn't re-run any download or co-registration. Per CLAUDE.md
acceptance #4:

- Changing `labeling.binary_area_threshold` / `binary_count_threshold` / `categorical_bins`:
  Stage 4 re-derives derived label columns from cached base stats in **milliseconds per ObsId**.
  Re-run `scripts/run_stage4.py --all` (~3 s/ObsId).
- Changing `features.glcm.*` / `features.shadow_fraction.*` / `features.enabled`:
  Stage 4b re-extracts features from cached CTX windows. Re-run
  `scripts/run_stage4b.py --all` (~3 min for the full sweep).
- Changing `splits.*`: Stage 5 re-builds split metadata + per-fold parquets. Re-run
  `scripts/run_stage5.py --all` (~25 s for both schemes).

Adding a new manifest row + its BoulderNet detections folder requires no code changes;
re-run Stage 1 → 5 in order (each stage's `--all` skips already-cached ObsIds).

## Layout

```
src/
  config.py          # load/validate YAML; SHA256 config hash for provenance
  manifest.py        # read hirise_priority10.csv; resolve per-ObsId shapefile
  pds_labels.py      # fetch + cache + parse HiRISE .LBL (authoritative metadata)
  detections.py      # Stage 1: read shapefile (auto-corrects buggy `D_unnamed` .prj via
                     # PDS LBL), reproject, cache GPKG
  ctx_tiles.py       # manifest <-> Murray Lab tile-name translator
  ctx_retrieve.py    # Stage 2: download tile zip, window + write CTX GeoTIFF,
                     # warp HiRISE -> CTX grid to build coverage mask
  hirise_imagery.py  # JP2 cache + decimated read helpers (auto-applies the SP1
                     # corrected CRS from Stage 1 sidecars)
  coregister.py      # Stage 3: warp HiRISE onto CTX grid, pick a power-of-2 FFT
                     # window, sub-pixel phase-correlate, cache (dx, dy) per ObsId
  labeling.py        # Stage 4: nested x2 grid anchored to CTX mosaic pixel origin;
                     # 5x sub-pixel polygon rasterisation; sum-up x2 ladder; all
                     # label transforms emitted regardless of `labeling.label_type`
  features.py        # Stage 4b: 9 feature families per-tile (intensity_stats, glcm,
                     # gradient, shadow_fraction, lbp, lacunarity, subtile_variance,
                     # canny_edges) + bundled context patches per (ObsId, patch_size)
  dataset.py         # Stage 5: leave-image-out split construction + in-memory
                     # package_split + streaming iter_train_batches/iter_test_batches
  qa.py              # shared sanity-check helpers
scripts/
  run_stage2.py        # headless per-ObsId Stage 2 driver
  sweep_stage2.py      # full-manifest Stage 2 sweep
  run_stage3.py        # Stage 3 driver (single ObsId or --all)
  run_stage4.py        # Stage 4 driver (single ObsId or --all; excludes ESP_057469_2215)
  run_stage4b.py       # Stage 4b driver (single ObsId or --all)
  run_stage5.py        # Stage 5 driver (named scheme or --all)
  probes/              # throwaway debug scripts (see probes/README.md)
tests/                 # 125 tests; integration tests skip until caches exist
notebooks/
  01_detections_qa.ipynb                  # Stage 1 overlay
  02_investigate_misplaced_detections.ipynb  # the SP1 bug, before the fix
  03_hirise_overlay.ipynb                 # decimated HiRISE imagery overlay
  04_ctx_retrieval_qa.ipynb               # Stage 2: window + mask + zooms
  05_coregistration_qa.ipynb              # Stage 3: shift distribution + before/after
  06_labeling_qa.ipynb                    # Stage 4: heatmaps + nested consistency
  07_features_qa.ipynb                    # Stage 4b: cross-image QA + correlation matrix
  08_features_explained.ipynb             # Stage 4b: per-family math/physics + stratified patch viewer
  09_splits_qa.ipynb                      # Stage 5: fold composition + group-leak check
  _build_*.py                             # source-of-truth Python builders for notebooks 07/08/09
cache/                # (gitignored) regenerable artifacts
  pds_labels/                  # PDS .LBL text files (~10-20 KB each)
  reprojected_detections/      # Stage 1: per-ObsId GPKG + provenance JSON
  ctx_tiles/                   # Stage 2: Murray Lab zipped tiles + JSON sidecar
  ctx_windows/                 # Stage 2: per-ObsId CTX GeoTIFF + HiRISE coverage mask
  hirise_jp2/                  # cached HiRISE JP2s (~200-500 MB each)
  hirise_decimated/            # 5 mpp HiRISE GeoTIFFs for co-registration
  coregistration/              # Stage 3: per-ObsId shift JSON
dataset/              # (gitignored except DATA_DICTIONARY.md)
  DATA_DICTIONARY.md           # schema reference for every artifact below
  labels/                      # Stage 4: per-ObsId parquet + JSON sidecar
  features/                    # Stage 4b: per-ObsId parquet + JSON sidecar
  context_patches/             # Stage 4b: bundled (n, P, P) uint8 stacks per (ObsId, P)
  splits/                      # Stage 5: per-scheme split metadata JSON
  packaged/                    # Stage 5: per-(scheme, fold) X/y parquets + groups.npy + all.parquet
reports/figures/     # PNGs from QA notebooks (committed so visuals persist)
config.yaml          # single source of truth for pipeline parameters
CLAUDE.md            # spec (authoritative)
DECISIONS.md         # runtime-verified facts and deviations
ROADMAP.md           # phase-by-phase index of planning + status
PLAN_Stage4b.md      # Stage 4b architecture plan (shipped)
PLAN_Stage5.md       # Stage 5 architecture plan (shipped)
PLAN_modeling.md     # Week 3 modeling plan
```

## How to grow the dataset

Adding a new image is two steps and zero code changes:

1. Add a row to `hirise_priority10.csv` with at minimum `ObsId`, `ProductId`,
   `BoulderLabel`, `CenterLat`, `CenterLon_180`, `CenterLon_360`, `CTX_TileName`,
   `JP2_URL`, `LabelURL`, and the other URL columns.
2. Drop the BoulderNet detections folder under `detections_root/{ObsId}/` containing
   a `*-mask-nms.shp` (with sidecar `.prj`, `.dbf`, `.shx`).

Then run the pipeline in order:

```powershell
& $conda run -n geospatial python scripts/run_stage2.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage3.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage4.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage4b.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage5.py --all   # re-build all schemes
```

Each `--all` driver skips ObsIds whose caches already exist (Stage 2/3/4/4b) and
re-derives split assignments deterministically (Stage 5).

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
  blocking; revisit if Stage 3 phase correlation shows a systematic equator-to-pole bias
  (it doesn't, as of the 2026-05-22 sweep).
- **HiRISE-coverage constraint** ([DECISIONS.md](DECISIONS.md) 2026-05-21): Stage 4 drops
  any tile where the cached `{ObsId}_hirise_mask.tif` isn't fully 1. The polygon-bbox
  CTX window contains ~40% HiRISE-unobserved area on average; counting those as "no
  boulders" would inflate zero-tile counts (it's "unobserved", not "absent"). The strict
  `coverage == 1.0` eligibility rule was chosen 2026-05-23 over a relaxed `>= 0.95`
  because partial coverage biases `fractional_area` low.
- **`/vsicurl/` is 140× slower than bulk download** for HiRISE JP2s on this network.
  Stage 2 uses `download_then_window` mode; don't switch to `/vsicurl/` without
  re-benchmarking. Disk usage of cached tiles + JP2s (~12 GB if all 10 ObsIds get
  fetched) is acceptable.
- **Source TIFF block size**: Murray Lab tiles have a full-raster internal block
  (47420×47420). Naively copying `src.profile` into a small windowed output triggers
  `_TIFFVSetField: Bad value 47420 for "TileWidth"`. Drop `blockxsize`/`blockysize`
  from the profile before writing.
- **Stage 4b context-patch layout deviates from PLAN_Stage4b.md §6**: patches are
  bundled per (ObsId, patch_size) into a single `.npy` stack, not per-tile
  `{ti}_{tj}.npy` files. The literal per-tile layout would produce ~1.3M small files
  (NTFS-hostile). 18 bundled files mmap-load fine for the CNN DataLoader path;
  `patch_idx_S{32,64}` columns in the feature parquet point into the stack.
- **Stage 5 splits are over images, never tiles** ([CLAUDE.md](CLAUDE.md) §4 acceptance
  #5). Tiles within an image share illumination, surface composition, and BoulderNet
  detector behaviour, so a random per-tile split leaks the per-image background into
  the test fold. Notebook 09 has an explicit group-leak assertion cell that fails the
  notebook execution if any `obs_id` appears in both train and test for any fold.
