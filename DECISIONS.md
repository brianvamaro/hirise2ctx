# DECISIONS

Running log of runtime-verified facts and deviations from `CLAUDE.md`. Each entry is dated.
This is how the runtime unknowns in CLAUDE.md §11 get pinned down permanently.

## 2026-05-20 — initial inspection (ESP_047976_2020)

- **`DETECTIONS_ROOT`** confirmed at `C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_priority10_detections`.
- 10 ObsId subfolders + a `summary.csv` from the BoulderNet run. The shapefile glob
  `{ObsId}/*-mask-nms.shp` resolves to exactly one file per image.
- **ESP_047976_2020 source CRS** (read from `.prj`): `Equirectangular_MARS`, sphere radius
  **3,393,833.2607584 m** (local Mars radius — NOT the IAU2000 3,396,190 m), central
  meridian **180°**, units = metres. This per-image local radius is the canonical example
  of the gotcha described in CLAUDE.md §3.3.
- **Shapefile DBF schema** (1,346 polygons for ESP_047976_2020):
  - `score` (float, 0.10–0.83, mean 0.41) — model confidence
  - `cat_id` (int, all 0) — category id
  - `cat_name` (str, all "boulder")
  - `isin_slice` (bool) — 1,343 True / 3 False
  - `is_at_edge` (bool) — 1,330 False / 16 True (tile-edge detections; flag for later filtering)
  - `id` (int) — polygon id
  - **No explicit size column.** Boulder size must be derived from `geometry.area`
    (or `2*sqrt(area/π)` as an equivalent diameter).
- **Polygon area** (ESP_047976_2020): 0.77–47.76 m², median 3.7 m², mean 5.1 m².
- **Footprint span**: ~6.6 km × 15.9 km; boulder fractional area ≈ 6.6×10⁻⁵ over the bbox.
  Even a "boulder rich" image is essentially all background — consistent with the
  zero-inflation warning in CLAUDE.md §9.

## 2026-05-20 — environment

- Conda **is not on PATH** in fresh PowerShell / Bash sessions on this machine. Use the
  absolute path: `C:\Users\brian\anaconda3\Scripts\conda.exe run -n geospatial python ...`.
- Direct invocation of `C:\Users\brian\anaconda3\envs\geospatial\python.exe` fails with
  exit code 127 (env DLLs not on PATH without activation). Always go through `conda run`.
- The `geospatial` env has Python 3.14.3, geopandas 1.1.3.

## 2026-05-20 — Stage 0–1 sanity-check formulation

CLAUDE.md §3.3 calls for a HiRISE↔CTX residual offset of O(200 m) post-reprojection. The
literal version of that test needs a CTX raster (Stage 2/3, phase correlation). Before
any large download we use a **cheaper equivalent** that catches the same class of bug:

- Reproject detections from per-image `Equirectangular_MARS` → target CTX CRS.
- Compute polygon footprint centroid in the target CRS, inverse-project to lat/lon on the
  target sphere.
- Compare to manifest `CenterLat` / `CenterLon` (great-circle distance on target sphere).
- Threshold: `sanity.centroid_max_km` in `config.yaml` (default 2.0 km). Multi-km offsets
  are diagnostic of the local-radius mistake (e.g. using 3,396,190 m instead of the
  per-image 3,393,833.26 m); failure raises `RuntimeError` with both points + both radii.

This is documented in `src/qa.py::assert_centroid_consistent`. When Stage 2 lands, the real
phase-correlation residual check will join it, not replace it.

## 2026-05-20 — Stage 0.5 deferred to Stage 2; using documented CTX CRS now

We attempted the planned header-only `/vsicurl/` probe to discover the Murray Lab CTX
mosaic CRS at runtime. Three obstacles compounded:

1. **TLS chain on this env is broken.** Both stdlib `urllib` (with `certifi.where()`-backed
   SSL context) AND `requests` fail with `CERTIFICATE_VERIFY_FAILED` against
   `murray-lab.caltech.edu`. The server doesn't ship the full intermediate chain; browsers
   recover via AIA fetching, Python doesn't. `truststore` (which would delegate to the
   Windows trust store) isn't installed.
2. **Murray Lab files are `.zip`-wrapped GeoTIFFs**, not raw `.tif`. A `/vsicurl/` open
   needs to go through `/vsizip/`, which needs the inner filename — not derivable from
   the outer URL.
3. **Tile-name convention mismatch.** Manifest uses `W040_N20`, `E152_S08`. Murray Lab
   uses `E-40_N20`, `E152_N-8` (signed N for south; example URL we verified:
   `MurrayLab_GlobalCTXMosaic_V01_E160_N-20.zip`).

**Decision:** Set `target_crs` directly to the canonical Murray Lab CTX mosaic CRS as
published — IAU 2000 Mars equirectangular, sphere **3,396,190 m**, central meridian
**0°**, units metres. The full Stage 0.5 verification (open a real tile, read its CRS
WKT, confirm the sphere/CM match) moves to Stage 2 because Stage 2 has to solve the
zip-wrapping + tile-name-translation problems anyway. The Stage 0–1 sanity check still
catches what it was designed to catch: a multi-km mismatch flags the wrong-sphere bug
regardless of which Mars equirectangular CRS we use as the target.

**Murray Lab URL pattern (recorded for Stage 2):**
`https://murray-lab.caltech.edu/CTX/V01/tiles/MurrayLab_GlobalCTXMosaic_V01_{tile_name}.zip`
where `{tile_name}` uses Murray Lab's `E{signed_lon}_N{signed_lat}` form, not the
W/S-prefixed manifest form. A translator will be needed in Stage 2.

**Murray Lab paper for reference:** Dickson, Kerber, Fassett, Ehlmann (2018), *A Global,
Blended CTX Mosaic of Mars with Vectorized Seam Mapping: A New Mosaicking Pipeline using
Principles of Non-Destructive Image Editing*.

**Tactical follow-ups (none blocking Stage 0–1):**
- Install `truststore` in `geospatial` to fix HTTPS broadly: `pip install truststore` +
  `truststore.inject_into_ssl()` at module import.
- Or set `GDAL_HTTP_CAINFO` env var pointing at certifi's bundle for `/vsicurl/` GDAL
  calls (independent of Python's stdlib SSL handling).

## 2026-05-20 — Data quality issue: 4 of 10 BoulderNet outputs are mis-located

The Stage 0–1 sanity check (`src/qa.py::assert_centroid_consistent`) caught a data
problem we did not expect. After reprojecting every shapefile to the canonical CTX CRS
(IAU 2000 Mars equirectangular) and inverse-projecting the polygon-footprint centroid
to lat/lon, the residual against `CenterLat` / `CenterLon_180` from the manifest splits
into two cleanly separated regimes:

| Image | Residual | Verdict |
|---|---:|---|
| ESP_055714_2270 | 1879 km | ✗ mis-located by ~47° eastward |
| ESP_054857_2270 | 1941 km | ✗ mis-located by ~48° eastward |
| ESP_047976_2020 |  481 km | ✗ mis-located by ~9° westward |
| ESP_056165_2200 | 1087 km | ✗ mis-located by ~24° westward |
| ESP_069669_2220 |    6 km | ✓ |
| ESP_057469_2215 |   10 km | ✓ (boundary case — sits 0.04° outside its tile but within HiRISE footprint asymmetry) |
| ESP_071093_2210 |    6 km | ✓ |
| ESP_075577_2105 |    7 km | ✓ |
| ESP_039820_1750 |    1 km | ✓ |
| ESP_065711_1545 | (empty shapefile, 0 polygons) |

Latitude matches in every case (Δlat consistently ~0.1°). It's purely a longitude
problem. Each shapefile's `.prj` is correctly formed (same `Central_Meridian=180.0,
Standard_Parallel_1=0.0` for all 9 non-empty images), so this is **not** a CRS
interpretation bug in this pipeline. Tested three alternative CRS interpretations
(cm=0/sp=image_lat, cm=image_lon/sp=0, cm=image_lon/sp=image_lat) — none recover the
mis-located images, and all break the good ones. The polygon coordinates inside those
4 shapefiles really are at the wrong position in the projected CRS.

Visualizations are in `reports/figures/02_good_vs_bad_in_ctx_tile.png` and
`reports/figures/02_all_misplaced.png`. The polygon centroids land **entirely outside
the CTX tile each image is declared to belong to**. Reproducer: `notebooks/02_investigate_misplaced_detections.ipynb`.

**Decisions taken:**
- Switched the Stage 0–1 integration test probe from `ESP_047976_2020` (was the
  originally requested probe — now known bad) to `ESP_069669_2220` (verified good,
  densest tile in the manifest, 1462 polygons).
- Raised `sanity.centroid_max_km` 2.0 → 15.0 to cover the legitimate detection-vs-image
  asymmetry inside a HiRISE footprint. Hundreds-of-km failures still fire loudly.

## 2026-05-20 — Root cause identified; manifest is correct; fix landed

Cross-checked 3 PDS `.LBL` files (ESP_047976_2020, ESP_055714_2270, ESP_069669_2220)
against the manifest. The manifest `CenterLat` / `CenterLon_360` match each image's
`MINIMUM_LATITUDE` / `EASTERNMOST_LONGITUDE` (the SE-corner convention) to 4 decimal
places. **The manifest is the truth.** The shapefile geometry is also correct — only
4 of the 10 `.prj` files are mis-labelled.

Reading all 9 non-empty `.prj` files revealed the fingerprint:

| Variant | Datum line | SP1 in .prj | Residual before fix |
|---|---|---:|---:|
| Good (5 images) | `DATUM["D_MARS", SPHEROID["MARS_localRadius",...]]` | round multiple of 5° near `image_lat` | 1.5–10 km ✓ |
| Bad (4 images) | `DATUM["D_unnamed", SPHEROID["unnamed",...]]` | `0.0` regardless of image latitude | 481–1941 km ✗ |

Geometry was generated with the PDS-correct projection latitude in BOTH cases; only the
`.prj` label is wrong in the bad variant. When pyproj trusts a `Standard_Parallel_1=0`
in equidistant cylindrical, longitudes scale by `1/cos(0)=1` instead of `1/cos(image_lat)`,
which is the source of the multi-hundred-km offset (~6% at lat 20°, ~30% at lat 47°,
amplified by `|lon - cm|`).

**Fix landed:** `src/pds_labels.py` fetches each image's `.LBL` via the manifest's
`LabelURL`, caches under `cache/pds_labels/{ObsId}.LBL`, and parses the PDS keywords.
`src/detections.py::read_detection_shapefile` now detects the buggy fingerprint
(`D_unnamed` in datum + `|SP1 − image_lat| > 15°`) and overrides `Standard_Parallel_1`
in the source CRS with the authoritative `CENTER_LATITUDE` from the `.LBL`. Good
`.prj` files are passed through unmodified.

**Post-fix residual table (all 10 images):**

| ObsId | n_polygons | Status | SP1 was | SP1 now | Residual before | Residual after |
|---|---:|---|---:|---:|---:|---:|
| ESP_055714_2270 | 1974 | sp1_corrected_from_pds_label | 0.0 | 45.0 | 1879 km | **9.4 km** |
| ESP_054857_2270 | 6462 | sp1_corrected_from_pds_label | 0.0 | 45.0 | 1941 km | **5.3 km** |
| ESP_069669_2220 | 1462 | trusted_prj | — | — | 6.3 km | 6.3 km |
| ESP_057469_2215 | 940 | sp1_corrected_from_pds_label | 0.0 | 40.0 | 10.3 km | 10.4 km |
| ESP_071093_2210 | 961 | trusted_prj | — | — | 6.4 km | 6.4 km |
| ESP_047976_2020 | 1346 | sp1_corrected_from_pds_label | 0.0 | 20.0 | 481 km | **7.4 km** |
| ESP_056165_2200 | 26 | sp1_corrected_from_pds_label | 0.0 | 35.0 | 1087 km | **12.5 km** |
| ESP_075577_2105 | 624 | trusted_prj | — | — | 7.5 km | 7.5 km |
| ESP_039820_1750 | 497 | trusted_prj | — | — | 1.5 km | 1.5 km |
| ESP_065711_1545 | 0 | (empty shapefile — diversity pick) | | | | |

The fix triggered on 5 images (4 truly mis-located + ESP_057469_2215, which had
`D_unnamed` and `SP1=0` but a small original residual because its image lon is at
the central meridian — fix is a no-op there in practice). Side note: ESP_056165_2200's
PDS projection latitude is 35°, not the 40° a naive nearest-5° rounding would have
chosen. The LBL-driven approach picks the right value where the heuristic would not.

**TLS fix in the same pass:** `truststore` is now a runtime dep. `src/pds_labels.py`
calls `truststore.inject_into_ssl()` at import time, which delegates SSL verification
to the Windows trust store (and what browsers use). Caltech (where Stage 2 will pull
the CTX mosaic) was the original SSL pain point — `truststore` will fix that too.

**Open work flagged by the user (next):** visual verification that the corrected
polygons land on actual boulders in the HiRISE imagery, not just near the right tile.
CLAUDE.md §7 lists this as a required QA notebook deliverable. Needs one decimated
HiRISE JP2 download (~few hundred MB at full res, but we'll read decimated). Awaiting
user approval on download size.

## 2026-05-20 — Stage 0–1 verification passes (1 image, integration)

Final state of this round:
- All 7 pytest tests pass (6 fast + 1 slow integration on ESP_069669_2220).
- Stage 1 caches `ESP_069669_2220.gpkg` (577 KB, 1,462 polygons in target CRS) +
  provenance JSON sidecar with source/target WKT + config hash.
- `notebooks/01_detections_qa.ipynb` renders to `reports/figures/01_detections_ESP_069669_2220.png`.
- `notebooks/02_investigate_misplaced_detections.ipynb` renders to the two figures
  named above.
- Conda + Windows SSL workaround (certifi context) recorded in
  `~/.claude/.../memory/conda_windows_ssl.md`.

## 2026-05-21 — The SP1 bug is a documented upstream HiRISE PDS issue, not a BoulderNet bug

Web search confirms what we hit is **not a BoulderNet bug** — it's a long-known issue in
**older HiRISE PDS RDR map projection labels**, where `Standard_Parallel_1` was written
as 0 instead of the image center latitude. The HiRISE team has been gradually re-issuing
corrected labels; the 4 affected images in our manifest (ESP_047976_2020, ESP_054857_2270,
ESP_055714_2270, ESP_056165_2200) predate the upstream correction. BoulderNet faithfully
passed the bug through into its `.prj` outputs — not its fault.

USGS Astrogeology publishes a binary patcher, **`fix_jp2_v2`** (Jan 2023 version), that
edits the JPEG2000 GeoTIFF box in place. Our runtime fix in `src/detections.py` +
`src/pds_labels.py` is functionally equivalent for the shapefile side (and also handles
JP2 metadata at read time if/when Stage 3 co-registration needs it), with the advantage
that it never mutates Cayleigh's raw outputs and survives any re-download.

**Implication:** stop calling this a "BoulderNet bug" in code comments and docs — it's
an upstream HiRISE PDS metadata issue. Reporting to Cayleigh is FYI ("you might be
hitting this older-label issue"), not a defect report.

**References:**
- [Planetary GIS — *more HiRISE conversion tips (until labels are fixed)*](http://planetarygis.blogspot.com/2016/07/more-hirise-conversion-tips-until.html) — Trent Hare (USGS Astrogeology). Describes the bug, what versions are affected, and points at the fix tool.
- [`fix_jp2_v2` source + Windows binary (USGS Astrogeology S3)](https://asc-pds-services.s3.us-west-2.amazonaws.com/pigpen/c_FORTRAN_code/) — the official patcher; idempotent, safe on already-corrected labels.
- [HiRISE PDS_JP2 Software](https://www.uahirise.org/tools/pds_jp2.php) — HiRISE PDS JP2 utilities (canonical entry point, doesn't mention the bug directly).
- [HiRISE RDR Software Interface Specification (PDF)](https://hirise.lpl.arizona.edu/pdf/HiRISE_RDR_SIS.pdf) — formal projection convention spec.
- GDAL tickets that wrestled with the same projection-keyword confusion for ~15 years:
  [#2478](https://trac.osgeo.org/gdal/ticket/2478) (`pseudo_standard_parallel_1`),
  [#2706](https://trac.osgeo.org/gdal/ticket/2706) (generalize/correct Equirectangular),
  [#3731](https://trac.osgeo.org/gdal/ticket/3731) (`ProjCenterLatGeoKey` for Mars imagery).

If we ever cache the 4 affected JP2s locally for Stage 3 co-registration, one-time
`fix_jp2_v2 <file>.JP2` on each cleans them permanently and removes the need for the
runtime override on the imagery side.

## 2026-05-21 — Stage 2 (CTX windowed retrieval) landed

Stage 2 builds the cached per-ObsId CTX windows that Stage 4 will read repeatedly.
**Mode = `download_then_window`** (config.yaml `ctx_retrieve.mode`): each unique Murray
Lab tile is downloaded once to `cache/ctx_tiles/{murray_tile}.zip`, the inner GeoTIFF
header is read via `/vsizip/` once, and per-ObsId windows are written to
`cache/ctx_windows/{ObsId}.tif`. `/vsicurl/` streaming was skipped after the HiRISE
experience (memory `feedback-collaboration` #4: ~140× slower vs bulk download).

**Runtime facts pinned down by the first real fetch (`E000_N40` for ESP_069669_2220):**

| Fact | Value |
|---|---|
| Murray Lab URL form | `MurrayLab_GlobalCTXMosaic_V01_{tile_name}.zip` |
| Tile-name convention | Zero-padded manifest form, **not** the bare signed-int form. `E000_N40` works; `E0_N40` returns 404. The retriever in `ctx_retrieve.py` tries the murray-form first and falls back to the padded form automatically. |
| Inner-tif path inside zip | `MurrayLab_GlobalCTXMosaic_V01_{tile_name}/MurrayLab_CTX_V01_{tile_name}_Mosaic.tif` (one nested directory; the `_Mosaic` suffix is the canonical filename) |
| Zip size (E000_N40) | 1,764,328,807 bytes (~1.68 GB) |
| Download throughput | ~29 MB/s sustained from Caltech (single connection, no `/vsicurl/`) |
| Raster shape | 47420 × 47420 px at 5 m/px → ~237 km × 237 km (4° × 4° at lat 40°) |
| Pixel size | 4.99997 m (north-up, e<0) |
| Source band dtype | `uint8` |
| Source CRS (read from tile) | `Mars_2015_Ocentric_Equirectangular_clon_0` — datum `Mars (2015)`, sphere **3,396,190 m** with inverse flattening **169.894447223612** (i.e. *oblate*, not pure sphere). Authority `IAU/49901`. |

**Implication of the CRS finding:** our `config.yaml::target_crs` uses the IAU-2000
sphere (3,396,190 m, **f = 0**) while the actual Murray Lab mosaic uses the IAU/2015
oblate spheroid with f ≈ 1/170. Over a 6 km HiRISE footprint at latitude 40° this is a
sub-pixel discrepancy (well under 5 m), so the Stage 0–1 centroid-residual sanity check
(threshold 15 km) is unaffected and visual overlay alignment looks correct. We are
**not switching `target_crs`** for now — the cost would be re-projecting all 10
Stage-1 caches for a sub-meter correction, and pyproj handles the mixed-sphere
projection of polygons-in-target-sphere onto pixels-in-source-spheroid implicitly via
the affine transform we cache in the tile sidecar. **Open item:** revisit in Stage 3
when phase correlation could surface this as a systematic ~1 px bias.

**Inner block size pitfall (fixed):** Murray Lab's GeoTIFFs are stored with a single
internal block equal to the full raster (47420 × 47420). Naively copying that
`blockxsize`/`blockysize` into a much smaller output produced
`_TIFFVSetField: Bad value 47420 for "TileWidth"`. `src.ctx_retrieve.extract_ctx_window`
now drops `blockxsize`/`blockysize`/`tiled` from the source profile and explicitly sets
`tiled=True, blockxsize=256, blockysize=256` for the output.

**Cache layout in use:**
```
cache/ctx_tiles/{murray_tile}.zip                 # 1-2 GB each; murray-form filename
cache/ctx_tiles/{murray_tile}.json                # source_url, inner_tif, inner_crs_wkt, transform, shape, dtype
cache/ctx_windows/{obs_id}.tif                    # ~5 MB; pixel-aligned to tile origin
cache/ctx_windows/{obs_id}.json                   # bounds, transform, shape, buffer_m, footprint_source, n_polygons_anchor, config_hash
```

**ESP_069669_2220 window provenance (the first real one):**
- Source tile: `E0_N40` (manifest form `E000_N40`, served URL `MurrayLab_GlobalCTXMosaic_V01_E000_N40.zip`)
- Bounds in target CRS: `[38559.80, 2469737.40, 49859.75, 2487157.31]` (m)
- Shape: 3484 × 2260 px → 17.4 km N-S × 11.3 km E-W
- 1462 polygon anchors, buffer 1000 m, `footprint_source = polygon_bbox`
- Visual QA in `reports/figures/04_ctx_window_ESP_069669_2220.png`: top half of the
  window is a boulder-rich textured ejecta surface, bottom half is smooth plains; the
  BoulderNet polygons cluster densely in the textured region — Stages 1 + 2 align as
  expected.

**Tests delta:** 13 → 38 (+17 ctx_tiles, +6 ctx_window_geometry, +2 stage2 integration).
Stage 2 integration tests auto-skip if the tile zip isn't cached, so they're CI-safe.

## 2026-05-21 — Labels-only-on-HiRISE-coverage constraint (Stage 4 design rule landed in Stage 2 output)

User-flagged constraint, confirmed before code change: **Stage 4 must NOT emit labels
outside the HiRISE swath, and must NOT emit labels for HiRISE NaN/0 pixels inside the
swath**. The whole pipeline's premise is HiRISE-derived ground truth; tiles outside the
HiRISE swath have no ground truth, so calling them "boulder absent" would be wrong (it's
"unobserved"). This is the difference between zero-inflation as a statistical property
(real) and zero-inflation as a measurement artifact (avoidable).

**Stage 2 enforcement (landed):** every `stage2_one_image` call now also writes
`cache/ctx_windows/{ObsId}_hirise_mask.tif` — a uint8 raster, same CRS/transform/shape
as `{ObsId}.tif`, with 1 where decimated HiRISE has a valid pixel and 0 elsewhere
(reprojected with `nearest`-neighbor to keep the binary boundary crisp). The provenance
JSON records `hirise_coverage_fraction`.

**Cost paid:** the first stage2 call for any ObsId triggers
`hirise_imagery.ensure_jp2_local`, downloading the JP2 (~200–500 MB). For ESP_069669_2220
this was a no-op (JP2 already cached). For the other 8 ObsIds, ~3 GB total — user signed
off on this disk cost (memory `feedback-collaboration` #5).

**Visual check:** `reports/figures/04_hirise_vs_ctx_ESP_069669_2220.png` shows the
HiRISE swath covers ~60% of the polygon-bbox CTX window — without the mask, ~40% of
Stage 4 tiles would have been spurious "no-boulder" labels from HiRISE-unobserved area.

**Stage 4 implementation rule:** at label-gen time, before computing any per-tile
statistic, intersect with the mask. A tile is *eligible* iff `mask_tile.mean() >= 1.0`
(every pixel covered) — tiles with partial mask coverage are dropped entirely rather
than scaled, because partial-coverage tiles bias `fractional_area` low (the denominator
is full tile area but the numerator is only the covered portion). Open detail:
boundary-tile threshold (`>= 0.95`? `== 1.0`?) — decide in Stage 4.

## 2026-05-22 — SP1 bug also poisons the JP2 metadata (fixed symmetrically)

A pre-sweep probe (`scripts/_probe_jp2_crs.py`) checked the cached
`ESP_047976_2020_RED.JP2` against its Stage 1 sidecar's `source_crs_wkt`:

```
ESP_047976_2020: Stage1='sp1_corrected_from_pds_label'  JP2_SP1=0.0  decimated_SP1=0.0
ESP_069669_2220: Stage1='trusted_prj'                   JP2_SP1=40.0 decimated_SP1=40.0
```

So the JP2 ships with the same `Standard_Parallel_1=0` bug as the matching `.prj` for
the 4 SP1-buggy ObsIds. Stage 2's coverage mask reprojection (and Stage 3 phase
correlation) would silently mis-locate those images on the CTX grid if we trusted the
JP2's embedded CRS.

**Fix landed in `src/hirise_imagery.py`:**
- `_corrected_source_crs(obs_id, cache_dir)` reads the Stage 1 sidecar's
  `source_crs_wkt`; returns None if Stage 1 hasn't run for this ObsId.
- `read_full_footprint_decimated` and `read_native_window` apply that CRS as an
  override at write time (the JP2 *transform* — origin + pixel scale — is correct
  because pixel coordinates were generated under the right projection; only the WKT
  label is wrong, so a label-only swap suffices, no warping required).
- A staleness check via `_crs_equal` compares cached-cache CRS to the corrected CRS
  and triggers a rebuild on mismatch. `_crs_equal` is **not** a simple `pyproj.equals`
  — pyproj's spherical Equirectangular canonical form drops SP1 from its equality
  check, so a buggy SP1=0 cache would otherwise compare equal to the SP1=20 corrected
  CRS. We also literal-parse the SP1 value out of both WKTs and require it to match.
  The literal regex accepts both ESRI WKT1 (`"Standard_Parallel_1"`) and EPSG/WKT2
  (`"Latitude of 1st standard parallel"`) names because pyproj rewrites between them.

**Implication for cached decimated TIFFs built before this fix:** the next time
`read_full_footprint_decimated` opens them, the staleness check fires and the cache
is rebuilt with the corrected CRS. `scripts/_verify_sp1_fix.py` confirms the rebuild
worked for `ESP_047976_2020` (cache SP1 went 0.0 → 20.0).

**Tests:** `tests/test_hirise_imagery_sp1_override.py` covers the override + cache
staleness in 4 fast unit tests.

## 2026-05-22 — Murray Lab URL convention pinned down for all 4 sign quadrants

The May padding fallback (`E0_N40` → `W040_N20`) worked for tiles already cached but
404'd on the new western longitudes ESP_047976_2020 (`W040_N20`) etc. needed. A direct
probe of the catalog (`scripts/_probe_murray_url_variants.py`) settled the convention:

| Manifest | Bare Murray | Live Murray URL form |
|---|---|---|
| `E000_N40` | `E0_N40` | `E000_N40` |
| `E152_S08` | `E152_N-8` | `E152_N-08` |
| `E000_S28` | `E0_N-28` | `E000_N-28` |
| `W040_N20` | `E-40_N20` | **`E-040_N20`** (NOT `W040_N20`) |
| `W052_N36` | `E-52_N36` | **`E-052_N36`** |
| `W024_N28` | `E-24_N28` | **`E-024_N28`** |
| `E160_S20` | `E160_N-20` | `E160_N-20` (already canonical) |

The convention is: `E<signed-abs-3digit>_N<signed-abs-2digit>`. Negatives use an
explicit `-` between the prefix and the zero-padded absolute value, not a W/S prefix.

`src/ctx_retrieve._padded_manifest_form` rewritten accordingly; `manifest_to_murray`
left unchanged so the existing cache filenames (e.g. `E0_N40.zip`) remain stable.
The retriever tries the bare Murray form first, then this canonical form on 404 —
all 7 new manifest tiles resolved on the second try in the 2026-05-22 sweep.

**Tests:** `tests/test_murray_url_padding.py` covers all 7 priority10 manifest tiles
plus 3 already-canonical inputs.

## 2026-05-22 — Polygon bbox straddling tile boundary (ESP_057469_2215)

Found during the Stage 2 sweep: ESP_057469_2215's polygons span x ∈ [-9619, +1019] m
in `target_crs` (~10 km west of and ~1 km east of the prime meridian). The manifest
assigns it to `E000_N40` (Murray tile covering lon 0°-4°E), but only the +1 km east
portion is inside that tile — the bulk lives in the neighbouring `W004_N40` tile,
which we don't fetch. Rasterio clipped the read window to the in-tile slice:

```
requested_bounds_target_crs: [-9619.95, 2439362.55,  1019.99, 2456657.46]
actual_bounds_target_crs:    [-9619.95, 2439362.55, -8599.96, 2456657.46]
```

Result: `actual_shape = [3459, 204]` (17.3 km N-S × 1.0 km E-W, but the strip is
entirely WEST of x=0 so it's outside the E000_N40 tile and reads as zero pixels);
`hirise_coverage_fraction = 0.001`.

**Decision:** not fixing this in Stage 2 now. Multi-tile mosaicking is a separate
engineering effort and would require non-trivial changes to the windowing path.
ESP_057469_2215 is recorded as a known-bad-for-Stage-3 case; `run_stage3.py` catches
the resulting "no power-of-2 ≥ 64 fits" RuntimeError and continues with the rest of
the manifest. DATA_DICTIONARY.md notes the very-low `hirise_coverage_fraction` as the
diagnostic flag for this class of issue.

**Follow-up (not blocking):** at Stage 4 / Stage 5, decide whether to (a) drop
ESP_057469_2215 from the dataset entirely, (b) fetch both neighbouring tiles and
re-window, or (c) reproject the polygons under a different central meridian so they
no longer straddle x=0. Option (c) is cheapest.

## 2026-05-22 — Stage 3 sweep (9 of 10 solved, distribution recorded)

`scripts/run_stage3.py --all` ran in ~5 s wall clock against the 10 cached
`ctx_windows/`. One ObsId failed gracefully (the tile-straddle case below); the other
9 all landed inside CLAUDE.md §3.3's O(200 m) acceptance band:

| ObsId | label | \|shift\| (m) | dx (m) | dy (m) | peak | notes |
|---|---|---:|---:|---:|---:|---|
| ESP_054857_2270 | Boulder rich | 118.0 | -1.2 | -118.0 | 0.63 | SP1-corrected |
| ESP_047976_2020 | Boulder rich | 125.5 | -8.2 | -125.2 | 0.71 | SP1-corrected, end-to-end check ✓ |
| ESP_075577_2105 | Boulder poor | 160.3 | +30.0 | -157.5 | 0.69 | |
| ESP_056165_2200 | Boulder poor | 164.5 | -63.0 | -152.0 | **0.28** | bland plains, weak texture |
| ESP_039820_1750 | unknown | 178.9 | +102.0 | -147.0 | 0.68 | |
| ESP_065711_1545 | unknown | 223.4 | +39.0 | -220.0 | 0.70 | empty-shapefile, nominal footprint |
| ESP_055714_2270 | Boulder rich | 240.8 | +29.2 | -239.0 | 0.60 | SP1-corrected |
| ESP_071093_2210 | Boulder rich | 247.0 | -15.5 | -246.5 | 0.77 | |
| ESP_069669_2220 | Boulder rich | 272.6 | +122.5 | -243.5 | **0.88** | canonical good case |
| ESP_057469_2215 | Boulder rich | — | — | — | — | FAILED — coverage 0.001 (tile straddle, see entry above) |

**Distribution:** |shift| min=118 m, median=179 m, max=273 m. Peak min=0.28 (the
boulder-poor outlier), median=0.69, max=0.88.

**Systematic finding:** every solved `dy` is negative (range -118 to -247 m); `dx`
straddles zero (range -63 to +123 m). The HiRISE imagery sits ~150-250 m NORTH of the
matching CTX features on the polygon-bbox windows. This is the CTX mosaic's ~200 m
N-S registration baseline that CLAUDE.md §3.3 explicitly flags as separate from CRS
handling — so it's expected, not a bug. We don't need to apply a per-image fix; Stage
4 can either ignore the shift (use nominal grid anchor) or apply it (refined anchor).
Leave the policy decision to Stage 4 review.

**Oblate-vs-sphere check:** if the sub-pixel CRS mismatch (DECISIONS.md 2026-05-21)
were systematic, we'd see |shift| correlate with image latitude. The data instead
shows |shift| largely flat across lat 30°N–47°N images and only slightly elevated for
the southern ones, with no clean linear trend — consistent with the discrepancy being
sub-pixel as predicted. **No CRS update needed.**

**Outlier policy still TBD:** the user explicitly chose "no thresholds yet — collect
data first" (AskUserQuestion 2026-05-22). With this data in hand, plausible thresholds
for Stage 4 are `|shift| > 500 m` (failsafe — well above what we've seen) and
`peak < 0.2` (catches even worse texture than ESP_056165_2200's 0.28). Pin down
during Stage 4 design.

Visuals: `reports/figures/05_shift_vs_peak.png` (scatter, labelled by ObsId), plus
nine per-image BEFORE/AFTER overlays at `reports/figures/05_coreg_{ObsId}.png`. The
overlays use a red/blue colour split (CTX = red, HiRISE = blue): the BEFORE panel
shows clear offset, AFTER snaps into co-registration on every image with a non-zero
shift.

## 2026-05-23 — Stage 4 (label generation) landed

Four design choices pinned via AskUserQuestion before any code (all "recommended"
options chosen):

| Question | Decision | Rationale |
|---|---|---|
| Mask-coverage threshold | **`coverage == 1.0` (strict)** | Drop any tile with even one uncovered HiRISE pixel. Zero downward bias in `fractional_area`; the relaxed `>= 0.95` would have biased frac low (numerator scales with covered area, denominator stays full). |
| Stage 3 shifts | **Apply to polygons before rasterization** | Translates each ObsId's polygons by `(dx_m, dy_m)` so HiRISE-derived boulder positions align with CTX texture in the same tile. Grid itself stays anchored to CTX pixel origin (no resampling). Eliminates the systematic ~200 m HiRISE-vs-CTX-feature offset Stage 3 measured. |
| `min_confidence` default | **Leave `null`** | Pass all 14,292 detections through. Tune after histogram review (notebook 06's `binary_comparison.png` is the first look). Matches the no-thresholds-yet pattern from Stage 3. |
| ESP_057469_2215 | **Drop** | Stage 2 window covers 0.1% of HiRISE swath (tile-straddle). 9 of 10 ObsIds remain — `scripts/run_stage4.py --all` excludes by default. |

**Implementation (`src/labeling.py`):**

- **Grid anchoring:** computes the integer mosaic-pixel offset of each CTX window's
  origin, then chooses the largest coarsest-scale-aligned cell range that fits inside
  the window. This guarantees the ×2 ladder is exactly nested — at scale `S`, cell
  `k` corresponds to the same mosaic-pixel block as cells `2k` and `2k+1` at scale
  `S/2`. The `(ti, tj)` indices in the parquet are **absolute** (mosaic-pixel
  coordinates / tile_size_px), so tiles from different images can be co-located in
  CTX space by index alone.
- **Boulder area:** rasterize polygons at 5× sub-pixel resolution (1 m/px) with
  `rasterio.features.rasterize`, then per-tile sums via reshape-and-sum. ~1 m²
  granularity is well under the 3.7 m² median boulder area.
- **Boulder count:** bin polygon centroids into finest-grid cells. Unambiguous at
  borders (each boulder counts once, in the tile owning its centroid). Cheap.
- **Sum-up:** boulder_area and boulder_count are summed up the ×2 ladder via 2×2
  reductions; eligibility is `all()` over the 2×2 — a coarse tile is dropped iff
  any of its 4 sub-tiles is ineligible. Acceptance #2 (nested consistency) was
  verified both in pytest (`test_stage4_nested_consistency_on_real_data`) and
  visually in notebook 06 (max delta = 1.0e-09 over all eligible coarse tiles in
  ESP_069669_2220 — floating-point noise).
- **Label transforms (all emitted regardless of `labeling.label_type`):**
  `fractional_area`, `binary_by_area`, `binary_by_count`, `count_density`. Base
  stats (`boulder_area`, `boulder_count`, `tile_area`) are also stored in every
  row, so changing thresholds re-runs label transforms in milliseconds from the
  cached parquet.
- **Cache:** `dataset/labels/{ObsId}.parquet` + `{ObsId}.json` sidecar. Idempotent.

**Sweep results (`scripts/run_stage4.py --all`, ~2.5 s total for 9 ObsIds):**

| ObsId | n_polys | S=8 eligible | S=16 | S=32 | S=64 |
|---|---:|---:|---:|---:|---:|
| ESP_055714_2270 | 1974 | 76,030 | 18,819 | 4,609 | 1,100 |
| ESP_054857_2270 | 6462 | 37,292 | 9,068 | 2,096 | 419 |
| ESP_069669_2220 | 1462 | 72,821 | 18,043 | 4,428 | 1,062 |
| ESP_071093_2210 | 961 | 55,962 | 13,834 | 3,370 | 792 |
| ESP_047976_2020 | 1346 | 54,054 | 13,363 | 3,259 | 773 |
| ESP_056165_2200 | 26 | 73,335 | 17,411 | 3,901 | 756 |
| ESP_075577_2105 | 624 | 44,560 | 11,018 | 2,691 | 639 |
| ESP_039820_1750 | 497 | 49,279 | 12,162 | 2,953 | 687 |
| ESP_065711_1545 | 0 | 25,221 | 6,226 | 1,518 | 359 |
| **total** | **13,352** | **488,554** | **119,944** | **28,825** | **6,587** |

The 8→16 ratio of 4.07 (vs theoretical 4.00) shows that ~1.8% of would-be
S=16 tiles get dropped because one of their 4 sub-tiles is ineligible, mostly along
the HiRISE swath boundary. Behaves the same way at coarser scales.

**Target distribution (488,554 finest tiles, no filters):**
- 97.88% of finest tiles have `boulder_area == 0` (CLAUDE.md §9 zero-inflation
  warning is correct).
- Mean `fractional_area` = 2.2e-4; median = 0; P99 = 6.25e-3; max = 0.269.
- The maximum (~27% of a tile covered by boulders) is at the densest part of a
  boulder-rich image and matches the visual heatmap.

**Binary-rule contingency at placeholder thresholds** (area>=0.005, count>=5):
- 169 tiles agree positive (0.03%), 482,879 agree negative (98.84%).
- 5,504 tiles fire **binary_by_area only** (1.13%) — many small boulders summing
  to >0.5% area without 5 polygons.
- 2 tiles fire **binary_by_count only** (~0%) — 5+ very small boulders summing to
  <0.5% area.
- The strong asymmetry says the count threshold is too high relative to the area
  threshold for these scales. Either lower it (e.g. 3) or raise the area threshold
  if we want them aligned. This is a Stage 5 / modeling-stage decision — Stage 4
  emits both rules so downstream can pick.

**Test count: 65 → 88** (+20 fast labeling unit tests, +3 slow integration tests
on ESP_069669_2220). `tests/test_labeling.py::test_stage4_nested_consistency_on_real_data`
is the explicit acceptance check.

**QA artifacts:** `notebooks/06_labeling_qa.ipynb` runs top-to-bottom in seconds and
writes 13 figures to `reports/figures/06_*.png`:
- 9 per-image fractional_area heatmaps (S=8 + S=64 side-by-side)
- `06_target_distribution.png` — zero-inflated histogram (linear + log non-zero)
- `06_per_image_distribution.png` — per-image overlay
- `06_binary_comparison.png` — area-vs-count agreement scatter + contingency bars
- `06_nested_consistency_ESP_069669_2220.png` — finest-downsampled vs coarse-emitted

**Dependencies added:** `pyarrow` (parquet engine for pandas). Installed via
`conda install -n geospatial -c conda-forge pyarrow`.

**Texture features (GLCM, intensity stats, gradient, shadow-fraction) are NOT in
this Stage 4 pass.** Per CLAUDE.md acceptance #4, label generation and feature
extraction are separable cheap re-runnable passes; the label parquet stores per-tile
bounds so a future `src/features.py` module can compute features against the cached
CTX windows without re-running anything else. This will be a Stage 4b.

## 2026-05-23 — Stage 4b (per-tile CTX texture features) landed

Five design choices pinned via AskUserQuestion before any code (recommended option chosen
each time):

| Question | Decision | Rationale |
|---|---|---|
| Shadow detector method | **DN-mode + tail offset** | One bincount per image finds the modal DN of HiRISE-mask-covered pixels; thresholds are `mode ± offset`. Stable across tiles within an image. Image-percentile alternative would drift with overall image brightness — boulder-poor scenes (median DN ~165) would get a different absolute threshold than boulder-rich (~95), defeating cross-image comparability. |
| GLCM angle handling | **Rotation-averaged single value per property** | Average `graycoprops` output over the 4 angles `[0, π/4, π/2, 3π/4]`. 6 properties × distances columns, rotation-invariant. Per-angle would 4× the schema for negligible modeling lift on a 10-image manifest (and would couple features to sun-azimuth). |
| Context patch sizes | **Both 32 px and 64 px, enabled by default** | PLAN_modeling.md §4 makes the CNN baseline non-optional, so patches need to exist when Week 3 modeling starts. Disk cost (~3.3 GB total) is well under the QA-artifact budget. |
| LBP variant | **Rotation-invariant uniform (skimage `method='uniform'`, P=8 R=1)** | 10-bin histogram, illumination-robust, robust to image orientation (no coupling to scene rotation). Plain `nri_uniform` would 6× the schema (59 bins for P=8) without a strong modeling case. |
| Deprecated `labeling.{features,context_patch_px}` keys | **Warn-only for one release** | `src/config.py` emits a `DeprecationWarning` when both `labeling.features` and the new top-level `features:` block are present. Stage 4b reads exclusively from the new block; Stage 4 still tolerates the old labeling-only path. Hard-removal is a follow-up commit after one release cycle. |

**Implementation (`src/features.py`, 9 feature families per PLAN_Stage4b.md §3 + §3.5):**

- **Window-once, tile-many**: per-image artifacts (Sobel gradient magnitude/direction, LBP
  map, Canny edge map, per-quantization integer arrays, shadow/strict/bright binary masks)
  are computed once over the full CTX window. Per-tile features are reshape-and-reduce
  operations on rectangular blocks; vectorized via a `(n_tiles, S, S)` stack. The only
  per-tile loops are GLCM (skimage `graycomatrix` can't be vectorised over tiles) and the
  shadow-mask lacunarity gliding-box.
- **GLCM scale-dependent quantization** (PLAN §3.2, citing Clausi 2002): 8 levels at S=8,
  16 at S=16/32, 32 at S=64. Distances [1] at S=8, [1, 2, 3] elsewhere. Schema is stable
  across scales -- finest-scale d=2/d=3 columns are NaN-padded so the concat'd parquet
  has one column set across all scales.
- **Shadow detector**: one `np.bincount` over HiRISE-covered pixels per image; `mode =
  argmax(counts)`. Three derived columns per tile: `shadow_fraction` (DN < mode − 20),
  `shadow_fraction_strict` (DN < mode − 35), `bright_cap_fraction` (DN > mode + 30). The
  asymmetric shadow-vs-bright pair is a stronger boulder signal than either alone (Kirk
  et al. 2008 photoclinometry intuition).
- **Lacunarity** (S ≥ 32 only): integral-image-backed gliding-box on the shadow mask at
  b ∈ {2, 4}. Degenerate below 16 px; emitted as NaN at finer scales.
- **Multi-scale variance** (S ≥ 16): variance of the (S/2)-block means within each tile.
  Essentially free since reshape-and-reduce already runs.
- **Canny edges** (S ≥ 16): density + Shannon entropy of edge-pixel orientations binned
  over `[0, π)` in 8 bins. Tiles with no edges get entropy = 0.
- **Context patches**: bundled per (obs_id, patch_size) into a single `.npy` stack per
  patch size; features parquet stores integer `patch_idx_S32` / `patch_idx_S64` columns
  (-1 means insufficient window margin). **Deviation from PLAN_Stage4b.md §6** which
  prescribed `dataset/context_patches/{ObsId}/S{px}/{ti}_{tj}.npy` -- that layout would
  produce ~1.3M tiny files (NTFS hostility, slow `os.scandir`); the bundled stacks total
  18 files instead and are `np.load(..., mmap_mode='r')`-friendly for the CNN DataLoader.

**Sweep results (`scripts/run_stage4b.py --all`, 176 s wall clock for 9 ObsIds):**

| ObsId | label | n_tiles | dn_mode | shadow%@S=8 | bright%@S=8 | glcm_contrast_d1@S=8 | GLCM time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| ESP_055714_2270 | Boulder rich | 100,558 | 115 | 15.4 | 21.4 | 0.237 | 22.1 |
| ESP_054857_2270 | Boulder rich | 48,875 | 156 | 30.7 | 4.7 | 0.273 | 11.0 |
| ESP_069669_2220 | Boulder rich | 96,354 | 77 | 4.2 | 28.5 | 0.174 | 27.6 |
| ESP_071093_2210 | Boulder rich | 73,958 | 129 | 8.1 | 2.3 | 0.194 | 16.2 |
| ESP_047976_2020 | Boulder rich | 71,449 | 135 | 16.8 | 19.7 | 0.312 | 14.7 |
| ESP_056165_2200 | Boulder poor | 95,403 | 166 | 22.0 | 10.5 | 0.265 | 18.1 |
| ESP_075577_2105 | Boulder poor | 58,908 | 117 | 8.1 | 1.4 | 0.264 |  9.1 |
| ESP_039820_1750 | unknown | 65,081 | 139 | 32.6 | 15.8 | 0.269 | 10.2 |
| ESP_065711_1545 | unknown | 33,324 |  87 |  7.5 | 11.3 | 0.164 |  5.1 |
| **total** | | **643,910** | | | | | **134.1** |

GLCM dominates the per-image budget (~75% of wall clock); everything else combined is
~5–7 s per image. The DN-mode range 77–166 confirms the per-image absolute-threshold
choice was right -- a single image-percentile threshold would be either too dark for
ESP_054857_2270 or too bright for ESP_069669_2220.

**Total disk after sweep:**
- `dataset/features/{ObsId}.parquet` + `{ObsId}.json` ×9: ~210 MB
- `dataset/context_patches/{ObsId}_S32.npy` ×9: ~660 MB (643,869 patches × 32×32 uint8)
- `dataset/context_patches/{ObsId}_S64.npy` ×9: ~2,640 MB (643,508 patches × 64×64 uint8)
- **Total Stage 4b artifacts: ~3.5 GB.** Patch index loss = 41 tiles at S=32 and 402 at
  S=64 fall in the window-edge margin (centred patch can't fit); recorded as `patch_idx
  = -1` in the parquet.

**Feature → target Spearman correlations at finest scale (488,554 tiles, 97.9% zero):**

Top 8 positive: `shadow_fraction_strict` (0.083), `shadow_fraction` (0.079),
`intensity_std` (0.035), `glcm_contrast_d1` (0.033), `glcm_dissimilarity_d1` (0.033),
`intensity_iqr` (0.031), `grad_mag_mean` (0.028), `grad_mag_p90` (0.027).

Top 8 negative: `bright_cap_fraction` (-0.040), `glcm_homogeneity_d1` (-0.033),
`glcm_ASM_d1` (-0.024), `glcm_energy_d1` (-0.024), `glcm_correlation_d1` (-0.023),
`lbp_hist_4` (-0.020), `intensity_skewness` (-0.014), `lbp_hist_3` (-0.014).

All correlations are weak (|r| ≤ 0.08) -- expected given 98% zero-inflation. Shadow
features lead, which is consistent with the shape-from-shading intuition: boulders
generate shadows directly. The negative `bright_cap_fraction` suggests bright/saturated
finest tiles tend to be *less* likely to contain detected boulders (could be specular
flat terrain, or boulders being detection-suppressed against bright backgrounds -- worth
investigating at modeling time). Modeling should evaluate at coarser scales and on
conditional-on-nonzero subsets where the signal is less diluted.

**Test count: 88 → 108** (+20 fast unit + 0 slow integration changes net; the slow Stage
4 integration tests still ran). New tests in `tests/test_features.py` cover:
intensity-stats edge cases (constant/ramp), GLCM quantization + uniform-image zero
contrast + NaN padding, gradient on step function, DN-mode threshold discovery, shadow
fraction on bimodal image, LBP histogram normalization, subtile-variance edge cases,
lacunarity on uniform-vs-clumped masks, tile-stacking correctness, Stage 4b
synthetic-cache end-to-end emit, idempotency, context-patch on/off behaviour, plus two
slow integration tests on ESP_069669_2220 (row-for-row alignment with labels parquet,
sanity ranges on real data).

**QA: `notebooks/07_features_qa.ipynb`** runs top-to-bottom in ~30 s and writes 13 figures
to `reports/figures/07_*.png`: per-image feature heatmaps (×9), GLCM-contrast-vs-target
scatter + gradient-vs-target scatter, Spearman correlation matrix, per-family timing
stacked bar, context-patch samples.

**Open follow-ups, not blocking Stage 5:**
- The strongest single feature today is `shadow_fraction_strict` at r ≈ 0.08. Try
  evaluating Spearman at coarser scales (S=32/64) -- with less zero-inflation the same
  signals should rise.
- `bright_cap_fraction`'s negative correlation merits a look at whether overexposed
  patches are systematically *under*-labeled (BoulderNet may have lower confidence on
  bright tiles).
- Two-stage (presence + magnitude) or log1p-transformed target are the natural next
  modeling moves -- captured in PLAN_modeling.md.

## 2026-05-25 — Stage 5 (leave-image-out splits + dataset packaging) landed

Four design choices pinned via AskUserQuestion before any code (all "recommended" options chosen):

| Question | Decision | Rationale |
|---|---|---|
| Default fold count | **9-fold LOIO primary + 3-fold balanced secondary** | LOIO gives honest per-image variance for headline numbers (one image per test fold = one number per image); the 3-fold variant smooths fold-level metric variance for the modeling sweep when LOIO's single-image test sets are too noisy. Both written; modeler picks at training time. |
| ESP_065711_1545 | **Include in folds with `BoulderLabel='unknown'`** | Its 25,221 finest tiles are real `boulder absent` examples (HiRISE-covered, no detections) -- valuable training signal and a clean false-positive test target. Filtering by `obs_id` at evaluation time recovers a per-image FP-rate report on demand. |
| `all.parquet` | **Emit per scheme** | ~96 MB per scheme; saves the modeler from repeated joins for ad-hoc analysis. Each tile appears exactly once tagged with its test `fold_idx`. |
| Loader pattern | **In-memory `package_split` as the default + streaming `iter_*_batches` as the alternative path** | At 9 images, the joined dataset is ~500 MB and in-memory is trivially fine. Streaming iterator is exposed for the 50-200+ image case per PLAN_Stage5.md §11b -- the API is shipped now so we never have to refactor downstream call sites when the manifest grows. |

**Implementation (`src/dataset.py`):**

- `build_image_inventory(...)` -- pulls Stage 4 sidecars + manifest BoulderLabel into a
  per-image dataframe (one row per ObsId, columns include n_tiles_total, BoulderLabel,
  frac_mean_finest, n_polys_after_filter). Inventory is what stratification operates on.
- `build_split(name, n_folds, stratification, seed, inventory, config_hash)` -- pure-
  Python split construction. Two stratification methods supported: `none` (LOIO with
  `n_folds == n_images` required) and `boulder_label_size_balanced` (greedy: place each
  image into the currently-smallest fold within its label group). Returns metadata dict;
  caller writes JSON via `write_split_metadata`. Idempotent + deterministic given the
  same seed + inventory; stable `split_hash` over the assignment.
- `package_split(metadata, labels_dir, features_dir, output_dir, scale_filter,
  emit_all_parquet, config_hash)` -- in-memory path. Loads each ObsId's labels +
  features once into `per_image[obs]`, then per fold writes X_{train,test}_fold{k}.parquet,
  y_{train,test}_fold{k}.parquet, groups_{train,test}_fold{k}.npy, plus the consolidated
  all.parquet (every tile tagged with its test fold_idx).
- `iter_train_batches(metadata, fold_idx, labels_dir, features_dir, scale_filter)` and
  `iter_test_batches(...)` -- yield one DataFrame per ObsId without materialising the
  full dataset. Same join + scale_filter logic as `package_split` internally.

**Sweep results (`scripts/run_stage5.py --all`, ~23 s total):**

| Scheme | n_folds | Build time | Package time | Total train rows | Disk |
|---|---:|---:|---:|---:|---:|
| `loio_9fold` | 9 | 0.01 s | 15.7 s | 5,151,280 (= 8/9 × 643,910 × 9) | 958 MB |
| `loio_3fold_balanced` | 3 | 0.01 s | 7.0 s | 1,287,820 (= 2/3 × 643,910 × 3) | 385 MB |

X has 55 columns (everything from `dataset/features/*.parquet` minus join keys + config_hash), y has 12 (the label columns + tile bounds context). Disk total is ~1.3 GB across both schemes; everything in `dataset/packaged/` is gitignored.

**`loio_3fold_balanced` per-fold composition** (5 rich + 2 poor + 2 unknown can't produce a perfectly label-balanced 3-fold, but 3-image-balanced is achievable):

| Fold | Test ObsIds | Composition | Test tiles |
|---|---|---|---:|
| 0 | ESP_047976_2020, ESP_055714_2270, ESP_075577_2105 | 2 rich + 1 poor | 230,915 |
| 1 | ESP_054857_2270, ESP_065711_1545, ESP_071093_2210 | 2 rich + 1 unknown | 156,157 |
| 2 | ESP_039820_1750, ESP_056165_2200, ESP_069669_2220 | 1 rich + 1 poor + 1 unknown | 256,838 |

Sum of test tiles = 643,910 (every tile appears in exactly one test fold) ✓.

**Test count: 108 → 125** (+17 in `tests/test_splits.py`):
- Inventory: 2 (round-trip, discover_obs_ids).
- Split construction: 7 (LOIO uniqueness, 3-fold size balance, reproducibility-with-seed,
  different-seed-changes-assignment, group-leak assertion, growth to 12 images,
  `stratification='none'` n_folds validation).
- Packaging: 5 (round-trip, all.parquet emit/skip, groups.npy alignment, scale_filter).
- Streaming iterator: 1 (yields one DataFrame per ObsId with no leak).
- 2 slow integration tests against the real Stage 4/4b outputs.

**QA notebook 09 (`notebooks/09_splits_qa.ipynb`) renders top-to-bottom in ~10 s.**
Saves 4 figures to `reports/figures/09_*.png`: per-image inventory bar chart, per-scheme
fold composition (stacked bars + tile-count line), per-fold target distribution (train
vs test, finest scale, log axis), and a group-leak assertion that runs over both
schemes' actual packaged parquets to confirm no `obs_id` overlap.

**Open follow-ups, not blocking Week 3 modeling:**
- `scale_filter` is currently `null` (every scale included). For a CNN baseline you'd
  restrict to `[64]` (matches the largest context patch); for tabular boosting `[8]`
  (finest, most tiles) is the typical starting point. Switch is a one-line config
  edit + re-run of `scripts/run_stage5.py --all` (~25 s).
- The 3-fold balanced scheme's tile counts per fold vary (156k-257k) because per-image
  tile counts vary widely (33k-100k). Image count is balanced (3/3/3) but tile count
  isn't. Decide at modeling time whether to weight metrics by per-image inverse-size or
  accept the per-fold imbalance.
- The streaming `iter_train_batches` / `iter_test_batches` paths are wired in but
  unused at 9 images. Per PLAN_Stage5.md §11b, the switch trigger is ~50+ images;
  document in modeling docs as the lever to pull when the manifest grows.

## 2026-05-25 — Methods document citation errata + clarifications

On read-through of the freshly committed `docs/methods.md` (commit `c3b8c96`), the
user identified six issues; corrections applied and committed alongside this entry:

1. **BoulderNet attribution was fabricated.** The original draft cited the detector
   as "Cayleigh Sirota et al., unpublished". The actual BoulderNet paper is
   [Prieur, Amaro, Gonzalez, Kerner, Medvedev, Rubanenko, Werner, Xiao, Zastrozhnov &
   Lapôtre (2023), JGR Planets](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JE008013).
   The "Cayleigh" references in earlier DECISIONS.md entries (2026-05-20, 2026-05-21)
   refer to the human who ran the published BoulderNet model on the priority10
   imagery, not its author -- those references stand. The methods document was
   updated to cite Prieur 2023 with no attribution for the run, per the user's
   choice (`Cite Prieur 2023 as the BoulderNet reference and don't attribute the
   priority10 run`).

2. **Minimum-boulder-size claim was wrong.** The original draft asserted boulders
   below ~0.25 m² (one HiRISE pixel) are not reliably detected. On user
   read-through I first replaced it with "empirical floor ~0.77 m²" — also wrong,
   because that was the per-image polygon-area minimum from one image's shapefile
   rather than a model design floor.

   The correct BoulderNet design floor, quoted verbatim from
   [Amaro et al. 2026](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JE008769):
   *"BoulderNet predictions are most accurate for boulders covering areas
   greater than 5 × 5 pixels, regardless of pixel scale. Thus, any detections of
   boulders smaller than this threshold area are filtered out in post-processing."*

   Translating to ground area requires the HiRISE per-image pixel size from the
   PDS `.LBL`'s `MAP_SCALE` keyword, which varies across the manifest because
   HiRISE products are released at multiple binning levels. Audit script:
   `scripts/probes/_boulder_size_audit.py`. Per-image results for the 5 manifest
   images whose `.LBL` is cached locally:

   | ObsId | HiRISE px (m) | 5×5-px (m²) | n polys | n < threshold | % < threshold |
   |---|---:|---:|---:|---:|---:|
   | ESP_055714_2270 | 0.50 | 6.25 | 1,974 | 7 | 0.35 % |
   | ESP_054857_2270 | 0.25 | 1.56 | 6,462 | 0 | 0.00 % |
   | ESP_057469_2215 | 0.50 | 6.25 | 940 | 2 | 0.21 % |
   | ESP_047976_2020 | 0.25 | 1.56 | 1,346 | 22 | 1.63 % |
   | ESP_056165_2200 | 0.50 | 6.25 | 26 | 21 | **80.77 %** |

   **Observations:**
   - The BoulderNet post-processing filter described above was **not applied
     consistently** to the priority10 shapefile copies — sub-threshold polygons
     survived in four of the five audited images (0-2 %), and 80.77 % of
     ESP_056165_2200's 26 polygons are below threshold.
   - **Decision deferred** (user explicit, 2026-05-25): do NOT apply our own
     `min_size_m` filter at Stage 4; report sub-threshold counts but keep all
     polygons in the pipeline. The filter-vs-keep policy is bundled with the
     `min_confidence` threshold decision; both are **Stage 4 configuration
     decisions, not modeling-stage decisions**, because once per-tile
     `boulder_area` / `boulder_count` are aggregated the individual polygon
     contributions are gone from the cached parquet. A one-line config edit
     (set `labeling.detection_filters.min_size_m` to a non-null value) plus
     `scripts/run_stage4.py --all` (~3 s / ObsId) applies any chosen policy.
   - A per-image filter (using each image's `MAP_SCALE` × 5)² as the threshold)
     would require a small extension to `_apply_detection_filters` in
     `src/labeling.py`; today the filter is a single global value. Deferred
     until the filter policy itself is decided.
   - The remaining 4 polygon-bearing images (ESP_069669_2220, ESP_071093_2210,
     ESP_075577_2105, ESP_039820_1750) do not have cached `.LBL` files yet; their
     per-image thresholds will be added when those labels are fetched.

   Methods document `docs/methods.md` §2.2 carries the verbatim Amaro 2026 quote +
   the audit table.

3. **THEMIS / TES rock-abundance citation was fabricated.** The original draft
   cited a non-existent "Christensen et al. 2003" for ~100 m / pixel rock
   abundance from THEMIS. The actual canonical global rock-abundance product
   derives from **TES** (not THEMIS) at ~3 km / pixel:
   [Nowicki & Christensen (2007), JGR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JE002798).
   Replaced.

4. **Dickson Murray Lab CTX mosaic citation was incomplete.** The original draft
   cited "Dickson et al. 2018" with a non-existent DOI. The 2018 work is an
   LPSC abstract (no DOI); the peer-reviewed paper describing the V01 mosaic is
   [Dickson, Kerber, Fassett, Sutton & Ehlmann (2024), *Earth and Space Science*](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA003555).
   Replaced with the 2024 DOI; the 2018 LPSC abstract retained as a secondary
   reference for the original announcement.

5. **"DN" terminology clarified.** The shadow-detection section described
   thresholds in terms of "DN" (Digital Number). User asked whether the Murray
   Lab mosaic actually contains raw DN at this level. It does not -- per Dickson
   2024 the mosaic applies per-image radiometric normalisation, brightness
   balancing across overlapping scenes, and seam blending before quantising to
   uint8. The methods document now flags "DN" as a shorthand for the mosaic
   uint8 brightness value and notes that all thresholds are computed
   per-image-relative-to-mode, so absolute radiometric calibration is not
   load-bearing for the feature.

6. **`5×` sub-pixel rasterization rationale.** User asked for the justification.
   Methods document now records: 5× factor places the rasterization grid at 1 m,
   sitting between the CTX pixel (5 m) and the HiRISE pixel (~0.5 m). Going to
   10× would match HiRISE resolution but quadruples memory (200 MB → 800 MB per
   image) without revealing additional polygon structure, because BoulderNet
   polygon vertices are themselves quantised to the HiRISE pixel grid. Going
   to 20× would over-resolve below the polygon-vertex precision floor.

7. **Parquet definition added.** User asked what Parquet is. Methods §1.3 now
   has a brief "Format note" explainer.

8. **Notebook count corrected.** Methods §9 said "seven QA notebooks"; actual
   count is nine (01 detections, 02 SP1 investigation, 03 HiRISE overlay, 04
   CTX retrieval, 05 co-registration, 06 labeling, 07 features QA, 08 features
   explained, 09 splits QA).

9. **ESP_057469_2215 exclusion reason restated in §8.3.** Was previously
   referenced only via §4.4 cross-reference; now stated inline in the splits
   section too.

**Lesson for future doc-writing sessions.** All citations in finished
documentation should be hyperlinked to canonical DOIs / open-access URLs at
write time (per the existing
[[feedback-hyperlink-citations]] memory rule). Hyperlinking acts as a
verification step: if a citation can't resolve to a real document, the doc
needs to be re-checked rather than the link suppressed. This pass caught
**three citations that turned out to be fabricated or wrong** (Sirota, Christensen
2003, the Dickson 2018 DOI) -- all of which would have been blocked by the
hyperlinking discipline applied at write time rather than retroactively.

## 2026-05-26 — Stage 4 detection filters set; per-image MAP_SCALE coverage completed

Three decisions pinned via AskUserQuestion before any code change:

| Question | Decision | Rationale |
|---|---|---|
| Fetch the 4 missing PDS `.LBL` files | **Yes** | Trivial (~32 KB total, fetched in <1 s with retry on transient `WinError 10054`). Completes the per-image `MAP_SCALE` audit; the boulder-size audit table now covers all 9 polygon-bearing images instead of 5. |
| `labeling.detection_filters.min_size_m` | **1.4105 m** (equivalent-circle diameter for an area threshold of 1.5625 m² = (5 × 0.25 m)²) | Matches the [Amaro et al. 2026](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JE008769) BoulderNet design floor for 0.25 m/px HiRISE binning exactly. Drops only the obviously-undetectable polygons (sub-1.56 m² on 0.25 m/px images); leaves the 0.50 m/px images untouched (their smallest polygons sit at 3.75–4.36 m², all above the 1.5625 m² threshold). The 0.50 m/px images' own 5×5-px floor (6.25 m²) is **not** enforced under this global filter; the 21 sub-6.25-m² polygons in ESP_056165_2200 all survive. The stricter global option (6.25 m²) would over-filter the 0.25 m/px images, where 1.56–6.25 m² polygons are legitimately resolved; a true per-image filter would require a small extension to `_apply_detection_filters` and is deferred. |
| `labeling.detection_filters.min_confidence` | **null** (no `score` cutoff) | All 14,292 detections pass through. The distribution (0.10–0.83, mean 0.41) is broad and the modeler can weight by `score` or filter at training time. |

**Boulder-size audit (all 9 polygon-bearing manifest images, `scripts/probes/_boulder_size_audit.py`):**

| ObsId | HiRISE px (m) | 5×5-px threshold (m²) | n polys | n < threshold | % < threshold |
|---|---:|---:|---:|---:|---:|
| ESP_055714_2270 | 0.50 | 6.25 | 1,974 |  7 |  0.35 % |
| ESP_054857_2270 | 0.25 | 1.56 | 6,462 |  0 |  0.00 % |
| ESP_069669_2220 | 0.25 | 1.56 | 1,462 |  1 |  0.07 % |
| ESP_057469_2215 | 0.50 | 6.25 |   940 |  2 |  0.21 % |
| ESP_071093_2210 | 0.25 | 1.56 |   961 |  1 |  0.10 % |
| ESP_047976_2020 | 0.25 | 1.56 | 1,346 | 22 |  1.63 % |
| ESP_056165_2200 | 0.50 | 6.25 |    26 | 21 | **80.77 %** |
| ESP_075577_2105 | 0.25 | 1.56 |   624 |  9 |  1.44 % |
| ESP_039820_1750 | 0.25 | 1.56 |   497 |  3 |  0.60 % |

**Stage 4 re-run (`scripts/run_stage4.py --all`, ~3 s total).** Polygon-count
deltas after the new filter took effect (9 of 9 ObsIds solved; eligible-tile
counts unchanged because tile eligibility depends on HiRISE mask coverage,
not on polygon presence):

| ObsId | n polys (raw) | n polys (filtered) | Δ |
|---|---:|---:|---:|
| ESP_047976_2020 | 1,346 | 1,324 | −22 |
| ESP_075577_2105 |   624 |   615 |  −9 |
| ESP_039820_1750 |   497 |   494 |  −3 |
| ESP_069669_2220 | 1,462 | 1,461 |  −1 |
| ESP_071093_2210 |   961 |   960 |  −1 |
| (all 0.50 m/px images) | — | — | 0 |
| **Total (9 ObsIds in sweep)** | **13,352** | **13,316** | **−36** (0.27 %) |

**Stage 5 re-run (`scripts/run_stage5.py --all`, ~22 s total).** Both
`loio_9fold` and `loio_3fold_balanced` schemes repackaged. Fold composition
+ per-fold tile counts are identical to the 2026-05-25 state (eligible
tiles unchanged); only the per-tile `boulder_area` / `boulder_count` /
derived label columns shifted, by the small amount the polygon drops imply.
The 643,910-test-tile / 5,151,280-train-row totals for `loio_9fold` and the
643,910-test / 1,287,820-train totals for `loio_3fold_balanced` are unchanged.

**Documentation updates landed alongside this entry:**
- `docs/methods.md` §2.2 — audit table extended to all 9 polygon-bearing
  images; filter-policy paragraph rewritten to describe the chosen
  `min_size_m = 1.4105`, the exact polygons it drops, and the ESP_056165_2200
  caveat (sub-6.25 m² polygons survive under the global filter).
- `scripts/probes/_fetch_missing_labels.py` — small probe added so the
  `ensure_all_labels` retry-on-transient-error workflow is reproducible (the
  bare `ensure_all_labels` call raised `WinError 10054` on the first attempt;
  this probe wraps it in a 3-try backoff loop).

## 2026-05-27 — Week 3 modeling baseline lands (LightGBM x 3 + small CNN)

Four decisions pinned via AskUserQuestion before any code change, plus one
threshold call resolved via probe (see end of entry):

| Question | Decision | Rationale |
|---|---|---|
| Session scope | **GBM + CNN in parallel** | PLAN_modeling.md §4 explicitly calls the CNN non-optional ("the natural complement to the GBM"). Building both this session keeps the side-by-side results table honest from the first run. |
| GBM target/loss variants | **All three** (`lightgbm_tweedie` + `lightgbm_log1p_huber` + `lightgbm_two_stage`) | PLAN §2: Tweedie is the textbook-correct loss for zero-inflated continuous targets; log1p+Huber is the variance-stabilising sanity shadow; two-stage hurdle is the cleanest decomposition of "presence vs. magnitude." All three share the LOIO harness so adding them now is cheap and the comparison is empirical instead of philosophical. |
| Per-scale architecture | **One model per scale, all 4 scales** (`scale_idx ∈ {0, 1, 2, 3}`) | PLAN §6 Option A. ~4× the LightGBM fits but each is seconds at this dataset size. Single-model-with-scale-feature deferred to a follow-up; train-at-coarsest-only rejected outright per the PLAN §0 *preserve CTX resolution* principle. |
| Two-stage positive rule | **`fractional_area > 0`** (strict zero-vs-nonzero) | Probe `scripts/probes/_pick_binary_thresholds.py` confirms strict-presence has 90–99 % Jaccard agreement between the area-based and count-based rules across all four scales, while matched-threshold definitions at intermediate positive rates drop to 0.20–0.55 Jaccard. Strict positivity gives ≥1,800 positives even at the coarsest scale (S=64) — enough for the magnitude regressor — and sidesteps the `binary_count_threshold` calibration mess entirely. |

**Threshold probe (`scripts/probes/_pick_binary_thresholds.py`) — closes the
DECISIONS.md 2026-05-23 `binary_count_threshold` open item.** Joint distribution
of `fractional_area` and `boulder_count` across all 643,910 tiles, 9 ObsIds, 4
scales (post-2026-05-27 Stage 4 re-run):

| scale | tile_size_px | n tiles | n positive (fa > 0) | n binary_by_area @ 0.005 | n binary_by_count @ **placeholder 5** |
|---|---:|---:|---:|---:|---:|
| 0 |  8 | 488,554 | 10,331 (2.12 %) | 5,673 (1.16 %) |   169 (0.035 %) |
| 1 | 16 | 119,944 |  6,855 (5.72 %) | 1,060 (0.88 %) |   397 (0.33 %) |
| 2 | 32 |  28,825 |  3,770 (13.1 %) |   172 (0.60 %) |   651 (2.26 %) |
| 3 | 64 |   6,587 |  1,843 (28.0 %) |    27 (0.41 %) |   409 (6.21 %) |

The placeholder `binary_count_threshold=5` was incoherent across scales — at S=8
it's impossibly high (only 169 tiles ever have ≥5 polygons), at S=64 it's
trivially exceeded (409 tiles). No single (area_threshold, count_threshold) pair
balances against the area rule across all scales (peak matched-threshold Jaccard
0.91 at 2 % target positive rate for S=8, falling to ~0.63 by S=64).

**Decision: `binary_count_threshold: 5 → 1`** in `config.yaml`. This makes
`binary_by_count` mean "any boulder by centroid rule" (≡ `boulder_count > 0`),
which is coherent at every scale. `binary_by_area` left at 0.005 — it remains a
real "appreciable area" diagnostic, distinct from but no longer fighting against
`binary_by_count`. Stage 4 + Stage 5 re-run (~3 s + ~22 s); fold composition
unchanged; only the `binary_by_count` column shifted. 643,910-tile / 5,151,280-row
totals for `loio_9fold` reproduce exactly.

**Modeling-code level: two-stage uses `fractional_area > 0` directly** (see
`src/modeling/gbm.py::POSITIVE_RULE_EPS`) — independent of the config-level
`binary_count_threshold`. The config column is purely diagnostic at this point;
the modeling does not consume it.

### Windows + Python 3.14 + torch 2.12 + MKL OpenMP coexistence

`import torch` on the `geospatial` env fails out-of-the-box with `OSError: [WinError
127] ... shm.dll or one of its dependencies` because (a) torch's `_load_dll_libraries`
does not find `torch/lib/*.dll` when invoked via `conda run`, and (b) numpy/scipy's
MKL OpenMP runtime (`libiomp5md.dll`) clashes with torch's bundled `libomp.dll`.
The fix landed in `src/modeling/__init__.py` and is triggered automatically the
first time any sub-module is imported:

1. `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` — lets the two OMP
   runtimes coexist instead of aborting the interpreter on the second load.
2. `os.add_dll_directory(<torch_root>/lib)` — adds torch's bundled DLL directory to
   Windows' DLL search path.
3. `ctypes.WinDLL("<torch_root>/lib/shm.dll")` — explicit pre-load that populates
   shm.dll's dependency chain into the process so torch's own loader finds them.

**Caller contract:** scripts and notebooks that use the modeling package must
import `src.modeling` BEFORE any `import numpy` / `import pandas`. Numpy loading
first pulls `libiomp5md.dll` into the process ahead of the bootstrap and breaks
the fix order. `scripts/train_cnn.py`, `scripts/probes/_smoke_cnn_one_fold.py`,
and `notebooks/_build_10.py` all follow this contract. Documented as a comment
block at the top of `src/modeling/__init__.py`.

### Modeling artifacts shipped this session

| New | Purpose |
|---|---|
| `src/modeling/__init__.py` | Package init + Windows DLL bootstrap (above). |
| `src/modeling/loaders.py` | Thin loader over `dataset/packaged/{scheme}/`; `Fold` dataclass; `iter_loio_folds()`; `gather_patches()` for CNN. 52 features per X (config_hash_feat + patch_idx_S* excluded from the feature matrix; patch_idx_S* lives on the keys frame for downstream join). |
| `src/modeling/base.py` | `Model` Protocol — `fit / predict / save / load / model_hash`. |
| `src/modeling/evaluate.py` | LOIO runner + Spearman + per-bin RMSE + presence AUC + aggregate (mean ± std, specificity folds split off). Inner-validation rotates training images per fold (PLAN §4) — the test fold is NEVER used as eval_set. |
| `src/modeling/gbm.py` | LightGBM Tweedie + log1p+Huber + Two-stage. All three implement `Model`. |
| `src/modeling/cnn.py` | Small CNN (~30k params at S=32, ~35k at S=64): 3 conv blocks → GAP → 2 FC, BN-before-ReLU, flip/rot/brightness/contrast/noise augmentation. log1p+Huber loss. |
| `src/modeling/inference.py` | Stub for off-HiRISE prediction across an arbitrary CTX region. Defines I/O contract; full mosaic sweep is a follow-up phase. |
| `scripts/train_gbm.py` | Single-variant LOIO driver; writes per-fold booster artifacts. |
| `scripts/train_cnn.py` | CNN LOIO driver; writes per-fold state_dict artifacts. |
| `scripts/sweep.py` | Fan-out over (variant, scale) → summary.parquet + aggregate.parquet. |
| `scripts/probes/_pick_binary_thresholds.py` | Drove the `binary_count_threshold` decision above. |
| `scripts/probes/_diag_torch_import.py` | Drove the Windows DLL bootstrap diagnosis above. |
| `tests/test_modeling_{loaders,evaluate,gbm,group_leak}.py` | 28 unit tests + 6 slow integration tests against the real packaged dataset. Group-leak assertion duplicates the notebook check at the test level. |
| `notebooks/_build_10.py` → `10_modeling_qa.ipynb` | Sweep table, per-fold Spearman by BoulderLabel, predicted-vs-true log-log scatter, per-bin RMSE heatmap, GBM feature importance, CNN-vs-GBM comparison. |
| `pyproject.toml` | New `modeling` optional-dep extra: `lightgbm>=4.0`, `torch>=2.2`, `scikit-learn>=1.4`, `scipy>=1.11`, `pyarrow>=14.0`. |

### Initial sweep result (defaults: 400 boosting rounds, lr=0.05, early_stopping=40)

| variant | scale | tile_size_px | spearman ρ (mean ± std) | presence AUC (mean) |
|---|---:|---:|---:|---:|
| lightgbm_tweedie | 0 | 8 | +0.004 ± 0.020 | 0.513 |
| lightgbm_tweedie | 1 | 16 | −0.006 ± 0.035 | 0.500 |
| lightgbm_tweedie | 2 | 32 | +0.010 ± 0.057 | 0.528 |
| lightgbm_tweedie | 3 | 64 | +0.031 ± 0.086 | 0.550 |
| lightgbm_log1p_huber | 0 | 8 | +0.014 ± 0.026 | 0.528 |
| lightgbm_log1p_huber | 1 | 16 | +0.030 ± 0.076 | 0.534 |
| lightgbm_log1p_huber | 2 | 32 | +0.030 ± 0.099 | 0.534 |
| lightgbm_log1p_huber | 3 | 64 | +0.002 ± 0.076 | 0.516 |
| lightgbm_two_stage | 0 | 8 | +0.000 ± 0.019 | 0.508 |
| lightgbm_two_stage | 1 | 16 | +0.002 ± 0.024 | 0.515 |
| lightgbm_two_stage | 2 | 32 | +0.018 ± 0.067 | 0.520 |
| **lightgbm_two_stage** | **3** | **64** | **+0.059 ± 0.138** | **0.568** |

Headline: every variant's std swamps its mean — PLAN_modeling.md §11.1 ("small-group
CV variance") played out exactly as predicted. The best per-scale model is
`lightgbm_two_stage @ S=64` but the wide std confirms the result is statistically
within the noise envelope, not a strong signal. Paired-fold comparisons (not in
the table) and the per-fold-by-BoulderLabel plot in notebook 10 are the right
diagnostics from here.

**Pytest:** 141 fast + 18 slow = 159 total (was 125 before this session).

### Inputs unchanged

- The 52 features the GBM consumes are the same Stage 4b set used since 2026-05-23.
- The patch stacks `dataset/context_patches/{ObsId}_S{32,64}.npy` are unchanged.
- Fold definitions in `dataset/splits/loio_9fold.json` and
  `dataset/splits/loio_3fold_balanced.json` are unchanged.

## 2026-05-27 — Stage 5c (within-image cross-validation diagnostic) landed

The Stage 5 / Stage 5b sweeps converged on AUC ≈ 0.52–0.55 — two independent target
framings (regression + binary classification) at the same ceiling. The §6.5 reading was
"the 9-image LOIO dataset is at its information ceiling for what 5 m / pixel CTX texture
can discriminate," but that was a one-sided argument: the data could equally be
consistent with "per-image generalisation is the binding constraint, more diverse
HiRISE images would unlock signal." Stage 5c is the diagnostic experiment that decides
between those two hypotheses.

**Design (`PLAN_Stage5c.md`).** Within-image 2×2 spatial-quadrant CV on the same Stage 4b
features and the same LightGBM defaults as Stage 5b. 8 non-empty priority10 images ×
4 quadrants = 32 folds per (variant, scale). `ESP_065711_1545` excluded (empty truth →
all quadrants would be specificity-only). Variants run: `lightgbm_two_stage` (best
regression cell at S=64) and `lightgbm_classification` at `bc_ge_1` (best binary cell
at S=32 / S=64). 8 cells × 32 folds = 256 fits total.

**Implementation.**

- **Split scheme.** `src/dataset.py` gains a new `stratification: within_image` value
  alongside `none` (LOIO) and `boulder_label_size_balanced`. Each (image, quadrant)
  becomes one fold; per-fold `test_obs_ids` / `train_obs_ids` are both the singleton
  list of the *same* image (training data is the other three quadrants of that image).
  `kind: "within-image"` distinguishes the JSON metadata from `"leave-image-out"`.
- **Multi-scale quadrant cut.** The literal PLAN §3 text ("per-scale median") could
  not strictly satisfy the multi-scale coherence invariant in
  `test_within_image_per_scale_quadrant_coherence` (when the finest median isn't a
  multiple of the coarsest factor, S=8 boundary tiles disagree with their S=64
  parent). **Resolved via AskUserQuestion 2026-05-27 to "shared cut from finest scale,
  snapped to a multiple of the coarsest factor."** Floor-snap of `median(ti_S8)` to
  `floor(median / 8) * 8` then `ti_mid_Sk = ti_mid_S8 // (Sk/8)` is exact at every
  scale.
- **Group codes for the inner-validation rotation.** `src.modeling.evaluate.run_loio`
  picks an inner-val image via `unique_train[fold_idx % n_unique]`. With LOIO that's
  unique because the training set spans multiple ObsIds. With within-image, the
  training set is a single ObsId, so using ObsId codes would collide with the
  held-out group. Resolved by storing **per-row quadrant indices (0..3)** in
  `groups_train_fold{k}.npy` / `groups_test_fold{k}.npy` instead of ObsId codes —
  unique_train has 3 distinct quadrant codes (the 4th is the test fold), so the
  rotation works unchanged. Tested in
  `test_within_image_groups_have_3_unique_train_codes_per_fold`.
- **Packaging.** `package_split` now dispatches on `kind`; the within-image branch
  reads each ObsId's labels parquet once and partitions by per-scale quadrant
  predicate. Per-fold parquets (`X_train_fold{k}.parquet`, etc.) have the same
  schema as LOIO so `src.modeling.loaders.load_fold` works unchanged.
- **No model code changed.** Both `lightgbm_two_stage` and `lightgbm_classification`
  run untouched against the new scheme. The only modeling-side adaptation is the
  group code semantic noted above.

**Result.**

`models/_sweep_within_image/20260527T175437Z/`: 8-cell aggregate ran in ~4 minutes.

| variant                   | S  | within-image AUC | LOIO AUC | mean Δ | 95 % CI         | Wilcoxon p |
|---------------------------|----|------------------|----------|--------|------------------|------------|
| `lightgbm_two_stage`      |  8 | 0.524            | 0.508    | +0.016 | [−0.018, +0.051] | 0.64       |
| `lightgbm_two_stage`      | 16 | 0.537            | 0.515    | +0.022 | [−0.018, +0.058] | 0.31       |
| `lightgbm_two_stage`      | 32 | 0.550            | 0.520    | +0.030 | [−0.018, +0.084] | 0.46       |
| `lightgbm_two_stage`      | 64 | 0.578            | 0.568    | +0.010 | [−0.090, +0.097] | 0.74       |
| `lightgbm_classification` |  8 | 0.518            | 0.520    | −0.001 | [−0.052, +0.035] | 0.38       |
| `lightgbm_classification` | 16 | 0.532            | 0.521    | +0.011 | [−0.056, +0.059] | 0.38       |
| `lightgbm_classification` | 32 | 0.542            | 0.546    | −0.005 | [−0.092, +0.059] | 0.64       |
| `lightgbm_classification` | 64 | 0.571            | 0.534    | +0.037 | [−0.022, +0.101] | 0.38       |

**All 8 CIs bracket zero; no Wilcoxon p < 0.05.** Within-image and LOIO give
statistically identical AUC. The diagnostic answer: **per-tile CTX texture signal floor
at 5 m / pixel is the binding constraint**, not per-image generalisation. Three
independent measurements (regression, binary classification, within-image CV) now sit
on the same ceiling.

**Recommendation update (`docs/modeling_results.md` §5 + §7).**

- "Within-image CV" (experiment 1) — ✅ shipped, signal-floor branch confirmed.
- "More HiRISE images" (experiment 2) — ⚠ demoted from "structural unlock" to
  "tightens error bars on the ceiling, does not move it." Worth pursuing for
  statistical power, no longer expected to raise the per-tile AUC.
- "Complementary non-texture signal" (was implicit in `CLAUDE.md` §10 future work) —
  promoted: thermal channels (THEMIS rock abundance) and coarser-than-tile spatial
  context are now the most plausible unlock for the per-tile AUC ceiling. *Spectral channel
  plan updated 2026-05-30: compositional study uses HiRISE 3 bands ([Delamere et al. 2010](https://doi.org/10.1016/j.icarus.2009.03.012)),
  not CRISM.*

**Tests.** 15 new unit tests + 1 slow integration test in
`tests/test_within_image_split.py`, including the strict multi-scale coherence
invariant (`test_quadrant_cuts_are_strictly_coherent_across_scales`) and the
inner-val-rotation invariant
(`test_within_image_groups_have_3_unique_train_codes_per_fold`). Total pytest:
191 → ~207.

**Reproducibility.** `python scripts/run_stage5.py within_image_4fold` rebuilds the
split + packaged scheme deterministically; `python scripts/sweep_within_image.py`
re-runs the 256-fit sweep against the same artifacts in ≈4 minutes on CPU.

## 2026-05-28 — vClaire 40-image detection set: ingest + manifest (Stage 1)

A new, much denser BoulderNet run ("vClaire", inference params `ct-010 ss-256 ov-020
downscaled`) covering 40 HiRISE images arrived in
`C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_40_vClaire`. Goal: build a
parallel v2 dataset to test whether denser/more-complete labels lift the AUC ≈ 0.55
ceiling (PLAN_NewDetections.md §9). Kept fully separate from v1 (`dataset/` untouched).

**A/B versioning (zero-code path).** `config_v2.yaml` mirrors `config.yaml` but points at
`hirise_40_vclaire.csv` / `hirise_40_vClaire` / `cache_v2` / `dataset_v2`, with
`features.context_patch.enabled=false` (CNN is a dead-end; saves the largest disk
chunk). `cache_v2` directory-junctions the detection-INDEPENDENT imagery caches
(`ctx_tiles`, `hirise_jp2`, `hirise_decimated`, `pds_labels`) back to `cache/` so they
are shared, not re-downloaded; detection-derived caches (`reprojected_detections`,
`ctx_windows`, `coregistration`) stay fresh. A `--config` flag was added to
run_stage1/2/3/4/4b/5 + sweep_stage2; **`scripts/run_stage1.py` is new** (there was no
headless Stage 1 driver — Stage 1 had only ever been run from notebook 01).

**Manifest build (`scripts/build_vclaire_manifest.py` → `hirise_40_vclaire.csv`).**
- URLs templated from the PDS RDR convention (`ORB_{orbit//100*100}_{+99}`); all 40
  `LabelURL`s resolved.
- **Center coords come from the PDS footprint midpoint** (`pds_labels.image_footprint`
  MIN/MAX_LAT, E/W_LON), NOT `pds_labels.projection_origin`. **Gotcha:**
  `projection_origin` returns the map-projection central meridian / standard parallel
  (rounded, e.g. lon 180, lat 45) — correct for the SP1 `.prj` fix but wildly wrong as a
  geographic center (first manifest pass put every tile at `E180_*`). Footprint
  midpoints cross-check against the spreadsheet corners to < 1° and reproduce the v1
  tiles for the 3 overlap images.
- `BoulderLabel` from the `Mapping_Images_33_36.xlsx` "Overall…" column → 37 `Boulder
  rich` + 2 `unknown` (vClaire is curated boulder-rich; confirmed with Brian). 3 images
  absent from the spreadsheet (`ESP_017355_2260`, `ESP_076499_1160`, and the dropped
  `ESP_028537_2270`).
- `CTX_TileName` derived by floor-to-4° (validated to reproduce all 10 existing-manifest
  tiles before use). vClaire spans ~15 unique Murray Lab tiles incl. a southern
  `E020_S64` (`ESP_076499_1160`, −63.7°) — a strong geographic-diversity outlier.

**Data-quality findings.**
- **`ESP_028537_2270` truncated** (`.dbf`/`.shp` far smaller than the `.shx` record
  count implies; read fails). Unfixable upstream → **excluded** from the manifest →
  39 rows. `ESP_045878_2235` initially shipped the wrong `-bbox-nms` variant; Brian
  re-exported the `-mask-nms` version (now included).
- **BoulderNet emits many null-geometry records** at this density: rows with a DBF
  entry (score/id) but no polygon. `ESP_017355_2260` is 1.1M rows but only **359,933
  real polygons** (745k null); `ESP_068483_2280` 1.06M → 727k. The priority10 set had
  zero. Confirmed present in the SOURCE shapefiles (not introduced by reproject).
  **Fix:** `src/detections.drop_null_geometries` drops null/empty geoms at Stage 1
  ingest (no-op on v1; detection tests still green) and records `n_polygons_raw` +
  `n_dropped_null_geometry` in the Stage-1 sidecar. True per-image boulder counts span
  9.6k → 727k (≈100–500× v1).

**Filter decision (`detection_filters`).** Reprojected equivalent-circle diameters are
large (pooled median 3.4 m, p5 ≈ 1.9 m) → **~0% below the `min_size_m=1.4105` floor**, so
that filter is a no-op (kept, consistent with v1). Scores: 100% ≥ 0.2, 89% ≥ 0.3,
52% ≥ 0.5 — `min_confidence` kept `null`. The denser set is *more* boulders, not
*smaller*.

**Stage 1 result:** 39/39 reprojected, 0 failures, 32 SP1-corrected. The 1.1M-row
reproject runs in ~16 s — no scale problem. `splits.*` `n_folds` in `config_v2.yaml` are
PLACEHOLDERS to be set to the surviving-image count after Stage 4.

## 2026-05-28 — Stage 3 upgraded to a robust block-median solve

**Why.** The original Stage 3 solved one `(dx, dy)` from a single central FFT sub-window.
On the vClaire set this failed on `ESP_049242_2115`: the central window returned an
anti-correlated junk shift (peak −0.06, with the wrong sign on `dx`) even though the image
is perfectly registerable. The whole-image block validation (added at Brian's request —
"make sure the co-registration is working across the whole image") showed 24 of its 29
128 px blocks correlated strongly (peak ≥ 0.5) and agreed with each other to ±10–20 m —
the single window had simply landed on a bad patch. Brian approved making the solve robust
("vital to get the co-registration right").

**What.** `src/coregister.py`:
- `block_shift_field(hi_warped, ctx_arr, mask, block_px, min_coverage, upsample_factor)` —
  tiles the window, phase-correlates each fully-covered block, returns the per-block local
  shift field. Also the QA primitive for the whole-image validation.
- `_robust_shift_from_field(field, block_peak_min, min_confident_blocks)` — median `(dy, dx)`
  over blocks with local peak ≥ `block_peak_min`; returns None (→ fallback) when fewer than
  `min_confident_blocks` clear the floor.
- `stage3_one_image` now computes BOTH: the single-window solve (kept in provenance as
  `single_window` + as the fallback) and the block-median (primary). Provenance gains
  `method` (`block_median` | `single_window_fallback`), `single_window{...}`, and
  `block_field{block_px, block_peak_min, n_blocks, n_confident_blocks, median_block_peak,
  block_mad_px}`. `shift_m` / `shift_px` / `peak_correlation` are the CHOSEN method's values,
  so Stage 4 consumption is unchanged.
- `warp_hirise_to_ctx_grid` exposed as a public wrapper for QA callers.
- Config: `coregistration.{block_px: 256, block_peak_min: 0.5, min_confident_blocks: 6}` in
  both config.yaml and config_v2.yaml (v1 caches were single-window; these only bite on a
  re-run). `run_stage3.py` threads them through and reports the method tally.

**Result (vClaire 39 images).** |shift| median 195 m (80–327 m); **38/39 block_median, 1
fallback** (`ESP_046803_2325`, genuinely bland). Median confident-block peak 0.71.
`ESP_049242_2115` rescued: peak −0.06 → 0.72, dx +2.5 → −27.1 m. The v1-overlap image
`ESP_069669_2220` is essentially unchanged (273 → 269 m), confirming block-median ≈
single-window where the latter already worked. Every `dy` negative (~200 m north bias =
the CTX-mosaic baseline), as in v1.

**Tests.** +2 (`test_block_shift_field_recovers_uniform_shift`,
`test_block_shift_field_skips_undercovered_blocks`); the slow Stage 3 provenance +
idempotency tests stay green on the block-median path (16/16 in test_coregister.py).

**Docs.** Full method writeup in `docs/methods.md` §5 (rewritten: §5.2 algorithm, §5.3
whole-image validation, §5.4 vClaire results, §5.6 the fallback image). Whole-image
validation rendered in `notebooks/05_coregistration_qa.ipynb` (new section, pointed at v2
via a `CONFIG_NAME` toggle) → `reports/figures/05_wholeimg_*.png`.

**`ESP_046803_2325` dropped (2026-05-28).** The lone single-window fallback. The notebook
05 deep-dive (CTX window, HiRISE warp, block-peak map, single-window overlay) showed
**0 / 210 blocks correlate** — the CTX window is uniformly dust-mantled with no texture
anywhere, not just at the central window. Despite ~367k detected boulders, it is a
high-target / no-input training example (featureless CTX paired with high abundance) that
would add label noise without teaching the CTX→abundance mapping. Brian's call: **drop**.
Added to `EXCLUDED_FROM_SWEEP` in `scripts/run_stage4.py` + `src/features.py` (so
Stage 4/4b/5 skip it → 38 of 39 images proceed); kept in the manifest + Stage 1–3 caches
so the rationale stays inspectable. `config_v2.yaml` `splits.*` `n_folds` must reflect 38
(not 39) once the surviving count is confirmed after Stage 4.

**Open:** a future refinement could *always* use block-median and drop the single-window
entirely, and/or set an automatic accept/flag threshold from the block-field coherence;
deferred until more cohorts confirm the 0.5 / 6-block parameters generalise.

## 2026-05-28 — Stage 4 reprojects detections to the window CRS + boulder-localization verified

**`gdf.to_crs(window_crs)` in `labeling.stage4_one_image`.** The boulder polygons were in
the pipeline `target_crs` (Mars_2000 sphere) while the CTX window — and the (ti, tj) tile
grid anchored to it — is the Murray Lab Mars_2015 oblate CRS. Stage 4 now reprojects the
polygons into `window_crs` right after load (before the co-reg shift + rasterization) so the
labels are placed in the exact frame the tiles live in (correct-by-construction). **Verified
this is a 0.000 m change at our coordinates** (`scripts/probes/_diag_tocrs_displacement.py`):
PROJ's equirectangular uses the shared semi-major radius (3,396,190 m) for both, so the
sphere/oblate definitions are numerically identical here — the prior "sub-pixel approximation"
was actually exact. So the existing labels were already correct; the change makes the
consistency explicit and future-proofs a CTX source whose CRS genuinely differs. Stage 4
`--all` was re-run for clean provenance (bit-identical labels, 38 images).

**Boulder-localization verification (answering "are the boulders correctly located?").**
Three independent checks, all pass:
1. *Full-res HiRISE overlay* (`scripts/probes/_diag_boulder_localization_fullres.py`,
   now also a cell in notebook 01): at 0.25 m/px, the BoulderNet polygons sit exactly on
   individual boulders (bright cap + shadow). Definitive fine-placement proof.
2. *Centroid gate, all 38 images*: mean polygon centroid is 0.2–5.0 km from the manifest
   centre (median 1.1), none near the 15 km gate → no CRS/local-radius gross errors.
3. *Co-registration* (block-median whole-image validation, above): ~6 m residual on good
   images. The sphere/oblate gap (0 m) can't move a boulder out of its 40–320 m tile.

**Notebooks pointed at v2.** 01 (detections QA — + a full-res localization section), 03
(HiRISE overlay), 04 (CTX retrieval QA) now default to `CONFIG_NAME = "config_v2.yaml"`
(toggle back to `config.yaml` for v1). 05 (coregistration) gained the whole-image
block-median validation, the good-vs-fallback deep-dive, and a before/after boulders-on-CTX
overlay. Notebook 02 (SP1-bug investigation) left as v1 — not relevant to vClaire.

## 2026-05-29 — vClaire v2 modeling A/B: Stage 5 + LOIO/binary/within-image sweeps; does denser labelling lift the ceiling?

Completed the v2 pipeline through modeling and ran the A/B against v1 (PLAN_NewDetections.md §9).
Full writeup + tables in `docs/modeling_results.md` §9; this is the decision record.

**What ran.** Stage 5 packaged dataset_v2 (`loio_nfold` = 38 folds; `within_image_4fold` =
152 folds = 38 images × 4 quadrants). Three sweeps on `--dataset-dir dataset_v2`:
regression `models/_sweep/20260529T061553Z/`, binary `_sweep_binary/20260529T075754Z/`,
within-image `_sweep_within_image/20260529T142227Z/`.

**Result (three metrics, two readings — both true).**
- *Target distribution*: v2 zero-tile fraction collapses vs v1 — S=8 0.979→0.500, S=64
  0.720→0.070; "boulder-rich" (>1% area) tiles up ~70–200×. The dominant difference.
- *Regression Spearman*: lifts ~3–10× (v1 ≈0, max +0.059 → v2 +0.10 to +0.17 LOIO), now
  unambiguously non-zero over 38 folds (two_stage S64 +0.169±0.226, ~4.6σ). v2 is a usable
  abundance **ranker**.
- *Presence/binary AUC*: lifts only **modestly** — bc_ge_1 0.52–0.55 (v1) → 0.55–0.62 (v2),
  growing with scale (S64 0.534→0.616). Still a weak **classifier**.
- *Within-image diagnostic*: as in v1, every per-image `within−LOIO` delta CI brackets 0,
  no Wilcoxon p<0.05 → within ≈ LOIO on v2 too.

**Decision / interpretation.** The v1 "≈0.55 signal floor" was **partly a missed-boulder
artifact**: completing the labels lifts the rank signal a lot and the presence ceiling a
little. But the within-image diagnostic (image-count-independent) still shows within ≈ LOIO,
so a **5 m/px texture floor still binds the presence ceiling (~0.6)** — completeness raised
the floor, didn't remove it. Confound called out: v2 changes label density + image count
(9→38) + class balance at once; only the within-image result cleanly attributes to label
completeness. v2 is the **go-forward dataset**; v1 modeling (§§1–8) stands as the frozen
baseline.

**Coarse-scale saturation (new, minor).** At S=64 v2 is ~93% positive, so whole images go
single-class and presence/`bc_ge_1` AUC is undefined for them — handled by dropping those
images (n falls to 25–26 of 38 at S=64). `_diag_within_image_deltas.py` now nan-skips before
the paired stats.

**Infra shipped for the A/B.** `--dataset-dir`/`--scheme` on the 3 sweep drivers (threaded
through `run_loio`→`iter_loio_folds`); `sweep_meta.json` per sweep + `src/modeling/sweep_select.py`
so notebooks/probes pick the right dataset's sweep by `dataset_dir`. `scripts/sweep.py`
default `--variants` still includes `lightgbm_classification`, which in regression mode
truncates the continuous target to all-zero (trivial model) — harmless, filtered out of the
v2 regression view. Notebook 10 **pinned** to the v1 sweep timestamps + a "superseded"
banner; new `notebooks/11_modeling_qa_v2.ipynb` is the v2 QA. README grow-the-dataset recipe
fixed to start at Stage 1; ROADMAP gains a vClaire v2 row.

## Open at this date

- **Stage 3 thresholds (flag/fail)** — collect more data first before pinning down.
  Current distribution suggests `|shift| > 500 m` + `peak < 0.2` as a starting point,
  but Stage 4 will benefit from a few more images and a re-look at ESP_056165_2200
  (the only low-peak case so far).
- **ESP_057469_2215 multi-tile windowing** — see the 2026-05-22 tile-straddle entry.
  Currently dropped from the Stage 4 sweep. Decide whether to fix at Stage 5 / 6.
- **Per-image `min_size_m` extension** — the chosen 2026-05-26 global filter
  (1.4105 m diameter ≡ 1.5625 m² area) enforces the 0.25 m/px floor exactly but
  is lenient for the 0.50 m/px images, where the design floor would be 6.25 m².
  Extending `_apply_detection_filters` to use a per-image threshold computed
  from each `.LBL`'s `MAP_SCALE` is a small code change deferred until/if the
  ESP_056165_2200 surviving-sub-threshold polygons turn into a modeling problem.
- **Modeling hyperparameter search** — first sweep used PLAN §2 defaults (400
  rounds, lr=0.05, num_leaves=63). PLAN §1 calls for a small coarse grid
  (3–5 configurations per variant) evaluated by mean ± std over LOIO. Deferred
  until after a first interpretation of notebook 10.
- **CNN-vs-GBM comparison at matched scales** — once the CNN sweep finishes,
  notebook 10's last section becomes the deciding diagnostic for whether a Stage
  4c feature push is motivated (CNN beats GBM ⇒ recoverable signal we missed).
- **THEMIS validation** — CLAUDE.md §10 future work; compare predicted abundance
  to the THEMIS rock-abundance map at coarse scale.
- ~~**`binary_count_threshold` rebalance**~~ — resolved 2026-05-27 (entry above);
  threshold set to 1 in config.yaml; two-stage modeling uses `fractional_area > 0`
  directly.
- ~~**Cache the 4 missing PDS `.LBL` files**~~ — done 2026-05-26 (all 9 polygon-
  bearing images now have `.LBL` in `cache/pds_labels/`; ~32 KB total).
- ~~**BoulderNet 5×5-px design-floor filter (Stage 4 decision)**~~ — decided
  2026-05-26 (entry above). Global `min_size_m = 1.4105 m` (≡ 1.5625 m² area).
- ~~**`min_confidence` default for the `score` column**~~ — reconfirmed `null`
  2026-05-26 (entry above). Re-evaluate at modeling time if needed.
- ~~**Stage 4b texture features**~~ — landed 2026-05-23 (see entry above). 9 feature
  families, 643,910 rows, 3.5 GB on disk including context patches.

## 2026-05-29 — dynamic-range compression diagnosed; 4 two-stage variants added

Phase A of [`PLAN_ModelImprovement.md`](PLAN_ModelImprovement.md) flagged that the v2
`lightgbm_two_stage` S=64 regressor squashes its predictions into a ~0.007–0.015 band
regardless of truth (over-predicting empty tiles, under-predicting the boulder-rich tail at
~0.42× truth). This session diagnosed the mechanism and tested fixes; full writeup in
[`notebooks/12_compression_diagnostic.ipynb`](notebooks/12_compression_diagnostic.ipynb).

- **Mechanism — two compression sources, not one** (from
  [`scripts/probes/_diag_compression_mechanism.py`](scripts/probes/_diag_compression_mechanism.py)):
  - **Presence head over-confident on zeros**: even on true-zero tiles, mean `p_pos = 0.85`
    (median 0.90). `is_unbalance=True` shifts the boundary to balance classes, but inflates
    `p_pos` on negatives — sets the over-prediction floor.
  - **Magnitude head shrunk to log-positive median**: `mag = pred/p_pos` spans only
    0.009–0.016 while truth spans 5 orders of magnitude. `log1p+Huber-on-positives` fits the
    geometric median; the heavy tail is shrunk away.
- **Post-hoc isotonic recalibration does NOT fix it.** LOIO-correct iso recalibration
  (fit on every-other-fold OOF, applied to held-out fold) leaves the high-bin ratio at 0.48
  (vs raw 0.42), and *drops* mean Spearman 0.169 → 0.157 and AUC 0.579 → 0.572 — the raw
  predictions don't span enough range to be re-stretched, and out-of-range clipping at fold
  boundaries breaks ranking. Compression must be fixed in training.
- **Four new two-stage variants added** to [`src/modeling/gbm.py`](src/modeling/gbm.py),
  each a minimal-diff cousin of `LightGBMTwoStage` via a shared `_TwoStageBase`:
  - `lightgbm_two_stage_balanced` — `is_unbalance=False` (presence-head fix only)
  - `lightgbm_two_stage_weighted` — magnitude head with `sample_weight = y_pos`
  - `lightgbm_two_stage_gamma` — magnitude head with `objective='gamma'`
  - `lightgbm_two_stage_combined` — all three together
- **Dev sweep results (`models/_sweep_compression_fixes/20260529T211211Z`,
  `within_image_4fold` 20 folds, S=32/64):**

  | variant at S=64        | Spearman ρ | presence AUC | high-bin ratio | zero pred |
  |------------------------|-----------:|-------------:|---------------:|----------:|
  | `lightgbm_two_stage` (baseline) | +0.263 | 0.538 | 0.83 | 0.0024 |
  | **`lightgbm_two_stage_balanced`** | **+0.280** | **0.556** | 0.83 | 0.0026 |
  | `lightgbm_two_stage_weighted` | +0.160 | 0.473 | **1.01** | 0.0048 |
  | `lightgbm_two_stage_gamma` | +0.255 | 0.513 | 0.82 | 0.0023 |
  | `lightgbm_two_stage_combined` | +0.160 | 0.440 | 0.99 | 0.0055 |

  - **`balanced` wins on ranking + detection without paying for it** (+0.017 ρ, +0.018 AUC
    at S=64; tail and floor barely move). Free lift.
  - **`weighted` / `combined` recover the tail almost perfectly** (high-bin ratio 0.83 →
    1.01) but trade away Spearman (0.263 → 0.16) and AUC (0.538 → 0.44) and *double* the
    zero-bin over-prediction. They are the right operating point if the deliverable is
    calibrated abundance estimates, the wrong one if it's ranking for follow-up.
  - **`gamma` alone is neutral** — slight compression improvement, slight AUC loss; not
    a clear win.
- **`balanced` is the new default candidate** for full-v2 promotion. The promotion (re-run
  the 38-fold LOIO) is Brian-gated as usual.
- **Tests:** 220 pytest pass (was 212; +8 from the 4 new variants auto-picking up the
  parametrized `test_fit_predict_basic` and `test_save_load_roundtrip`).

## 2026-05-29 evening — Phase A2 reframe (H1 metrics + H2 target reformulation; H3 deferred)

After the §11.3 interventions above produced only marginal gains (+0.017 Spearman, +0.018
AUC), Brian pushed back: "the compression is still there; metric changes are really small."
This session reframed the modeling problem around three new findings:

- **The existing v2 binary sweep already had a stronger story at the operational threshold.**
  Reading
  [`models/_sweep_binary/20260529T075754Z`](models/_sweep_binary/20260529T075754Z) at
  `fa_gt_1e-2` ("boulder-rich"): mean lift@top-K = **1.43 at S=64** (vs 1.02 for `bc_ge_1`);
  per-image AUC is bimodal — median 0.61, max **0.91** (lift 5.4×), and one image at AUC
  0.76 with **lift 9.1×** on a 1.3% base rate. Cross-image mean AUC was washing this out.
- **Five-hypothesis framework** for "compression persists, signal is real" (full discussion
  in [`notebooks/12_compression_diagnostic.ipynb`](notebooks/12_compression_diagnostic.ipynb)
  §7 and [`docs/modeling_results.md`](docs/modeling_results.md) §11.5):
  - H1 metric (mean AUC under-represents) — **implementing**
  - H2 target (`fractional_area` is pixel-aliasing-noisy below ~0.005; `boulder_count`
    is alias-robust) — **implementing**
  - H3 per-image heterogeneity (bimodal AUC; `shadow_fraction` is illumination-dependent) —
    **documented as deferred future work** (needs Stage 4c adding 4 per-image columns from
    cached `.LBL` files)
  - H4 multiplicative hurdle — plausible, test H2 first
  - H5 5 m/px CTX texture floor — eventually binds; unlock is outside CTX
- **Decision (Brian, 2026-05-29 evening): implement H1 + H2, document H3.** Plan:
  - **H1**: enrich [`src/modeling/evaluate.py`](src/modeling/evaluate.py) with PR-AUC,
    normalized lift (= lift × base_rate), precision@k, recall@k at k ∈ {1%, 5%, 10%},
    per-image distribution stats. Computed on both regression (with implicit binary
    derived from the target) and classification runs.
  - **H2**: dev sweep `lightgbm_two_stage_balanced` × {`fractional_area`,
    `log_fractional_area`, `log_boulder_count`} × {S=32, S=64} on the within-image scheme.
    Same composite metric as §11.3 + the H1 additions.
- **H1+H2 dev sweep result (`models/_sweep_target_reformulation/20260529T221912Z`, 6 fits,
  within_image_4fold 20 folds, variant=lightgbm_two_stage_balanced):**

  | target at S=64       | Spearman ρ | ROC-AUC (presence) | ROC-AUC (meaningful) | **PR-AUC** | normalised lift | precision@top-5% |
  |----------------------|-----------:|-------------------:|---------------------:|-----------:|----------------:|------------------:|
  | `fractional_area`    | +0.280     | 0.556              | 0.713                | 0.526      | 0.488           | 0.549             |
  | **`boulder_count`**  | **+0.283** | 0.564              | 0.697                | **0.640**  | **0.619**       | **0.660**         |
  | `log_boulder_count`  | +0.279     | 0.545              | 0.690                | 0.638      | 0.628           | 0.663             |

  - **Switching from `fractional_area` to `boulder_count` lifts PR-AUC by +0.114 (+22%),
    normalised lift by +0.131 (+27%), precision@top-5% by +0.111 (+20%)** while leaving
    Spearman ρ and ROC-AUC essentially unchanged.
  - The H1 framework's prediction confirmed end-to-end: ROC-AUC and Spearman couldn't see
    the gain (rank-invariant / threshold-averaged), but PR-AUC and lift do.
  - `log_boulder_count` ≈ `boulder_count` (the internal log1p+Huber handles the transform).
  - **Mechanism**: `boulder_count` is alias-robust at the low end — a 4 m² boulder in a
    320×320 m tile contributes either 0 or 1, regardless of CTX grid alignment. The
    fractional_area equivalent gets pixel-smeared into a small noisy positive whose
    magnitude depends on grid alignment. Cleaner negatives → sharper hurdle.
- **Recommendation for full-v2 promotion** (Brian-gated): re-run the 38-fold LOIO with
  `target_col=boulder_count` and `lightgbm_two_stage_balanced`. If the +0.114 PR-AUC dev
  signal carries over, this is the new headline product (alongside the §11.3 `balanced`
  presence-head fix).
- **Documentation reframed**: [`docs/modeling_results.md`](docs/modeling_results.md) §11
  fully covers H1-H5 + the §11.3 / §11.4 / §11.6 results; the headline metric for the
  deliverable should be PR-AUC + lift@top-K, not ROC-AUC.
- **220 pytest pass** (unchanged). Notebook 12 ([`notebooks/12_compression_diagnostic.ipynb`](notebooks/12_compression_diagnostic.ipynb))
  is the canonical writeup, ~960 KB rendered.

## 2026-05-29 late — per-image heterogeneity (H3) exploration

Brian's question: which v2 images worked, which didn't, and is there a per-image predictor
that explains the bimodal AUC distribution? Notebook 13
([`notebooks/13_per_image_heterogeneity.ipynb`](notebooks/13_per_image_heterogeneity.ipynb))
joined per-fold metrics from the full-v2 sweeps with manifest + cached PDS `.LBL` data
(IncidenceAngle, EmissionAngle, PhaseAngle, SubSolarAzimuth, all 38/38 images).

- **HiRISE-LBL illumination angles have NO significant correlation with model performance**
  (n = 37, all `p > 0.10`):

  | feature → metric              | bin_rich_auc | bin_rich_lift | bin_rich_ece | reg_spearman |
  |-------------------------------|--------------:|--------------:|--------------:|--------------:|
  | IncidenceAngle                | −0.14 (p=0.42) | 0.00 (p=0.99) | −0.27 (p=0.10) | −0.20 (p=0.23) |
  | EmissionAngle                 | +0.27 (p=0.10) | −0.09 (p=0.60) | +0.16 (p=0.33) | +0.27 (p=0.10) |
  | PhaseAngle                    | +0.08 (p=0.65) | +0.07 (p=0.68) | −0.24 (p=0.15) | −0.06 (p=0.71) |
  | SubSolarAzimuth               | +0.03 (p=0.85) | −0.25 (p=0.14) | +0.06 (p=0.72) | −0.06 (p=0.73) |
  | CenterLat                     | +0.21         | +0.24         | +0.06         | +0.28         |
  | NPolygons                     | +0.27         | −0.06         | +0.04         | +0.24         |
  | **bin_rich_base_rate**        | **+0.36 (p=0.027)** | +0.08 | +0.06 | +0.29 |
  | **reg_mean_true_fa**          | **+0.33 (p=0.044)** | −0.05 | +0.07 | +0.32 |

  Verdict: H3-as-label-quality (HiRISE illumination → label noise) is **NOT a strong
  per-image predictor**. The only significant correlates are `base_rate` and
  `reg_mean_true_fa` (both p ≈ 0.03–0.04), saying "images with more boulder-rich content
  fit better" — partly trivial, partly informative.
- **CTX-source illumination is the untested H3 hypothesis** — Brian's flag, 2026-05-29. The
  model uses **CTX** texture features (`shadow_fraction`), and the Murray Lab mosaic is
  composed of many CTX source images each with its own illumination geometry. Reading the
  SeamMap from the cached Murray Lab tile zips, we found **a mean of 24 CTX sources per
  HiRISE footprint** (range 4–46) — so each footprint is a *blend* of many CTX
  illuminations, not a single one. Getting CTX-source-illumination requires downloading the
  PDS CUMINDEX (~200 MB), joining on the SeamMap source IDs, and aggregating to per-tile (not
  per-footprint!) angles. Docketed as **[PROMOTION_QUEUE.md P5b](PROMOTION_QUEUE.md)**;
  significant work, but the only avenue left for testing H3-feature-quality.
- **Anti-signal failure mode confirmed at ESP_054000_2255**: 812 tiles, 18.3% boulder-rich
  base rate (NOT a rare-positive case), but:
  - Top-1% predicted (8 tiles): **0 boulder-rich** (vs 18.3% expected from random)
  - Top-10% predicted (81 tiles): **4.9% boulder-rich** (~26% of random)
  - Model is genuinely *anti*-correlated; the texture features here point the wrong way.
  - Per-image investigation is the most informative next step on these cases.
- **Failure-mode taxonomy** (3 distinct classes):
  1. **Anti-signal** (AUC < 0.45): wrong-way correlation; texture features mislead.
     Examples: ESP_054000_2255 (AUC 0.40), ESP_055253_2245 (AUC 0.42).
  2. **Rare-positive miss** (base_rate < 0.02, lift = 0): the boulder-rich tiles are so rare
     that even moderate AUC (~0.60) doesn't put a single positive in the top-K. Discrete
     metric artefact at small K; smoothed by `precision@top-5%`.
  3. **Presence/magnitude split**: high presence AUC but low Spearman. Model can detect
     "any boulder" but can't rank magnitudes. Example: ESP_049242_2115 (presence_AUC 0.97,
     Spearman −0.05) — the dataset-level compression failure mode (notebook 12 §2) in
     per-image form.
- **Promotion queue created** at [`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md) — 6 dev-validated
  items awaiting full-v2 confirmation: P1 `balanced`, P2 `boulder_count`, P3 metric reframe,
  P4 retire `bc_ge_1`, P5 `lightgbm_classification` calibration fix, P5a CTX-source
  illumination Stage-4c (the only H3-feature-quality test that's inference-compatible).
- **Inference-time scope rule (Brian, 2026-05-29)**: the deliverable runs on stand-alone
  CTX in regions where HiRISE coverage is absent. **HiRISE-derived per-image features
  (IncidenceAngle/EmissionAngle/PhaseAngle/SubSolarAzimuth) cannot be added to the model**
  — there is no HiRISE image at inference time, so the input would be missing. They remain
  useful for our own per-image diagnostic analysis (notebook 13), but the originally-proposed
  "HiRISE LBL Stage-4c" promotion is **out of scope**. The CTX-source illumination
  Stage-4c addition (P5a) IS in scope, since CTX-source angles can be looked up from the
  Murray Lab SeamMap + PDS CUMINDEX at inference time wherever CTX is available. Documented
  in [`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md) under "Inference-time scope" + "Out of
  scope" sections.

## 2026-05-30 — boulder_area dev sweep + THEMIS conversion path + research-directions synthesis

Brian flagged: switching to `boulder_count` as the regression target sacrifices direct
comparability with THEMIS rock-abundance maps (which are area fractions).  Follow-up dev
sweep + plan-documentation session.

- **Dev sweep on boulder_area** ([`models/_sweep_target_reformulation/20260530T154730Z`](models/_sweep_target_reformulation/20260530T154730Z),
  within_image_4fold 20 folds): at S=64, `boulder_area` and `log_boulder_area` perform
  **essentially identically to `fractional_area`** on PR-AUC (0.531 / 0.525 vs 0.526) and
  normalised lift (0.479 / 0.482 vs 0.488).  The +22 % PR-AUC win from `boulder_count`
  (PR-AUC 0.640) is **specific to count**, not to log-scale and not to area.
  - Likely mechanism: CTX texture features respond to **count** of distinct detection events
    (multiple shadows from multiple boulders) more than to **total area** of those events.
    Boulder size variability within a tile adds noise to area-based targets that count is
    invariant to.
- **THEMIS comparability is preserved** via a simple post-hoc conversion at inference time:
  `predicted_themis_area ≈ predicted_count × mean_boulder_area_per_boulder / tile_area`,
  multiplied by a population-scaling factor to bridge the ~100× rock-size gap (THEMIS
  sees > 15 cm rocks, BoulderNet sees > 1 m boulders).  This population step is
  required for any approach — direct `fractional_area` vs THEMIS comparison would need it
  too.
- **Open inference-time question** (Brian, 2026-05-30): at full-mosaic inference on
  CTX-only regions we have no labels, so no per-image `mean_boulder_area_per_boulder`.
  Options recorded in [PROMOTION_QUEUE.md P2 "Open inference-time question"](PROMOTION_QUEUE.md):
  global mean (simple), per-region pre-computed (involved), or accept that we only need
  rank correlation with THEMIS (Brian's lean — Spearman is rank-invariant under per-image
  scaling).  Track as a P2-blocker only if THEMIS validation requires calibrated abundance
  values.
- **Decision**: **Path A** (`boulder_count` primary target + post-hoc area conversion for
  THEMIS) is the recommended path; multi-target `boulder_count + boulder_area` heads are
  NOT needed based on dev evidence.  Document the conversion approach in the eventual
  THEMIS validation writeup.
- **Research-directions synthesis** (2026-05-30 conversation):
  - Six modeling problems catalogued in [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) under
    "Problem catalog & priority". **Status legend**: ✓ dev-validated, ◐ partial,
    ? untested hypothesis, ✗ unresolved.
    1. ✓ Target distribution noise → **P2** (boulder_count, dev +22 % PR-AUC)
    2. ◐ Compression → **P1** (balanced) **fixes presence-head source only**; the
       magnitude-head log1p+Huber-shrinks-to-median source remains. High-bin ratio
       0.42 → 0.83, not 1.0. Honest verdict: ship as ranker, not as calibrated
       abundance regressor.
    3. ? Per-image anti-signal → **Stage 6b** *tests* the CTX-illumination hypothesis;
       other candidate mechanisms (terrain composition, mosaic seams, image-specific data
       issues, label errors) are not distinguishable from H3 yet. If 6b fails: move to
       Stage 6c (image-level pre-classifier).
    4. ? No surrounding spatial context → **Stage 6a** (Brian, 2026-05-30); indirect
       evidence is the S=128 scale Spearman 0.26 → 0.41 finding. Extrapolation, not
       a direct test. Could disappoint if the S=128 gain was actually about coarse
       label-noise averaging rather than spatial integration per se.
    5. ✓ Metric framing → **P3 + P4** (methodological reframe; metrics already exist
       in code, docket items are documentation reframes)
    6. ✗ 5 m/px CTX texture floor → unresolved; eventual unlock is outside CTX
  - **Docket structure (Brian, 2026-05-30)**: PROMOTION_QUEUE.md split into **Part A —
    Pipeline tweaks (P1–P5)** for existing-pipeline variant / target / metric / doc
    changes, and **Part B — Stage 6: model improvement / feature augmentation** for new
    feature columns and model components.  Part B items are: **Stage 6a** spatial-context
    neighbour features (was P5b), **Stage 6b** CTX-source illumination features (was
    P5a), **Stage 6c** image-level pre-classifier (placeholder, not docketed in detail
    until 6a/6b results land).  Earlier "Stage-4c" labelling was wrong — those items
    aren't extensions of Stage 4.
  - **Priority order (PROMOTION_QUEUE.md)**: ✓ marked items are bank-the-wins; ? marked
    are bets that could fail. (1) ✓ P1+P2 full-v2 promotion, (2) ✓ P3+P4 doc reframe,
    (3) ? Stage 6a spatial context, (4) ? Stage 6b CTX-source illumination, (5) ✓ P5
    binary calibration, (6) ? Stage 6c image-level pre-classifier, (7) ✗ THEMIS /
    HiRISE-surrogate.

## 2026-05-31 — Stage 7.0 compositional feasibility gate: PASS (a)

- **Trio**: `ESP_042964_2160` (high-density positive, model AUC 0.91), `ESP_054000_2255`
  (anti-signal #1, AUC 0.40), `ESP_055253_2245` (anti-signal #2, AUC 0.42, **substituted
  for `ESP_055978_2270` which has no PDS COLOR.JP2** — verified directly).
- **PDS layout** — verified 2026-05-31: PDS publishes a SINGLE
  `{ObsId}_COLOR.JP2` per observation, not separate IRB+RGB JP2s as
  `PLAN_Compositional.md` originally assumed. It is a 3-band band-sequential JP2 in
  band order **[IR (~900 nm), RED (~700 nm), BG (~500 nm)]**, `uint16`, in **I/F**
  units after `physical = DN * SCALING_FACTOR + OFFSET` (both fields in
  `COLOR.LBL`). Resolution typically 0.25 m/px but can be 0.5 m/px (verified for
  `ESP_054000_2255`). URL convention:
  `https://hirise.lpl.arizona.edu/PDS/RDR/ESP/ORB_{orbit_range}/{ObsId}/{ObsId}_COLOR.JP2`.
- **Coverage** — colour swath is empirically ~2.4 km wide on the trio (not 1.0-1.3 km
  as previously estimated). 2 of 3 candidates had COLOR.JP2, suggesting ~60-80 %
  v2-cohort coverage. Full audit deferred to Stage 7a.
- **HiRISE PDS SP1 bug also poisons COLOR.JP2** — same producer artifact as RED.JP2.
  The CRS embedded in COLOR.JP2 reports `Standard_Parallel_1=0` even though pixel
  coords are under the corrected SP1=`pds_center_lat`. The fix in `src/colour.py`:
  override the JP2's CRS with `corrected_source_crs(obs_id)` from the Stage 1 sidecar
  (`cache_v2/reprojected_detections/{ObsId}.json`). Without this override polygon
  swath-overlap is 0 %; with it, 47-65 %.
- **Lambertian correction cancels in within-image diffs and band ratios** —
  `I/F_corrected = I/F_obs / cos(i)` is a per-image multiplicative scalar, so it
  vanishes from (interior − ring) and from `IR/BG`, `IR/RED`, `RED/BG` ratios. The
  Test A and Stage 7e ratio-based discriminators are Lambertian-invariant. Cross-image
  pooling (Stage 7d §4.2) still requires the correction. Added to
  `PLAN_Compositional.md` §5.3.

### Per-image results (probe: `scripts/probes/_stage7_feasibility.py`, n=800/img sampled)

| ObsId | Test A IR/BG d, p | Test A IR/RED d, p | Test A dust_index d, p | Test B dust_index d, p | partial r(IR/BG, rich \| dust) |
|---|---|---|---|---|---|
| ESP_042964_2160 | +0.94, 2.5e-34 | +1.28, 3.4e-38 | +0.77, 1.2e-27 | n_rich=5 (too few) | +0.07, p=0.40 (dust-explained) |
| ESP_054000_2255 | +0.05, 0.82 | −0.19, 9e-4 | +0.09, 0.65 | +0.31, 0.012 | −0.09, p=0.16 |
| **ESP_055253_2245** | **−0.34, 1.8e-12** | **−0.71, 1.7e-30** | **−0.22, 3.4e-7** | n_rich=8 (too few) | **+0.16, p=0.037 (survives dust)** |

### Verdict against `PLAN_Compositional.md` §3.1 pass conditions

- **(a) Pass criterion met** (`p < 0.05` AND `|d| > 0.3` anywhere): YES, all 3 images
  show some signal — ESP_042964_2160 dramatically (d > 0.7 in ratios), ESP_055253_2245
  in band ratios (d −0.34 to −0.71), ESP_054000_2255 in absolute bands only.
- **Dust-confound discriminator**: ESP_042964_2160's IR/BG signal is fully
  dust-attributable (marginal r=0.156 → partial r=0.070, p=0.40). ESP_055253_2245's
  IR/BG signal **survives dust control** (marginal r=0.252 → partial r=0.159,
  p=0.037) — a real composition signal in the *anti-signal* image where the model
  fails. ESP_054000_2255's per-polygon ratio signal is null to begin with.

**Final verdict: PASS (a) — composition signal detected (dust-controlled).**

Scientifically interesting:
- The two boulder populations have *opposite-direction* compositional shifts (042964
  redder than surroundings, 055253 bluer). Suggests different source / transport
  histories.
- The spectral test surfaces a real boulder-vs-surroundings signal in
  ESP_055253_2245, an image where the rock-abundance model fails (AUC 0.42). The
  compositional analysis is complementary, not redundant, to CTX-texture inference.
- ESP_054000_2255 (also anti-signal) shows no compositional signal at the
  per-polygon scale — boulders there are spectrally indistinguishable from
  surroundings except for shadow darkening. Supports H_local for that image.

### Implementation artefacts

- `src/colour.py` — LBL parse, SP1-corrected CRS loader, Lambertian, region-mean
  helper, polygon-ring-mask helper.
- `scripts/probes/_fetch_color.py` — fetches `{ObsId}_COLOR.JP2` + `.LBL` for the trio
  (~842 MB total, single-connection HTTP).
- `scripts/probes/_stage7_feasibility.py` — Test A (paired) + Test B (unpaired) +
  per-image incremental parquet saves.
- `notebooks/14_compositional_feasibility.ipynb` — renders + verdict; partial
  correlation done in-notebook.
- `cache_v2/stage7/{test_a,test_b}_per_polygon|tile{,_<ObsId>}.parquet` and
  `*_summary.parquet`, plus `dust_summary.parquet` for the discriminator.

### Recommendation

Stage 7.0 pass → proceed to Stage 7a (colour-JP2 fetch + audit across the v2 cohort)
when banking-the-wins (Path A) is complete or in parallel. Per the §3.1 conditional
pass framing, the deliverable can be framed as *both* compositional difference
(ESP_055253_2245) and dust-age difference (ESP_042964_2160) — since the two
mechanisms appear in different images, both narratives have evidence.

### Caveats / known-limitations of the feasibility probe

- Each image was sampled to 800 polygons for speed; 259-362 survived the
  `MIN_POLYGON_PIXELS=8 / MIN_RING_PIXELS=16` filter. Full population would
  ~12-30× the sample, giving stronger statistics but no expected qualitative change.
- Test B's per-image partition had too few boulder-rich tiles (n=5, 46, 8) for stable
  Mann-Whitney on two of three images. Pooled-cross-image Test B is a Stage 7d
  follow-up.
- Within-polygon spectra include some boulder shadow. The absolute-band negative
  diffs in all three images are largely shadow effects; band ratios are the
  composition-diagnostic signals.
- Lambertian-only correction. No Hapke / Minnaert; deferred per `PLAN §5.3` until
  per-image effects are shown to dominate.

## 2026-05-31 night — Stage 7a colour-coverage audit + bulk fetch

- **Coverage audit** (`scripts/run_stage7a_audit.py`, HEAD-probed all 39 v2 ObsIds):
  **37 / 39 ObsIds have a PDS `COLOR.JP2`** (94.9 %). The two without are
  `ESP_055690_2200` and `ESP_055978_2270` (the latter was already replaced in the
  Stage 7.0 trio). The 94.9 % rate is substantially higher than
  `PLAN_Compositional.md §8 q1` estimated (~60-80 %).
- **Total fetch volume** for the available 37 products: **9 106 MB ≈ 9.1 GB**.
  Per-image JP2 sizes 86 MB - 705 MB (median ~190 MB). Matching `.LBL`s are
  8 KB each (37 × 8 KB ≈ 300 KB negligible).
- **Coverage cache layout** (`cache_v2/hirise_color/`):
  - `coverage.parquet` -- per-ObsId audit result (URL, HTTP status, byte sizes,
    `has_color` boolean, audit timestamp).
  - `{ObsId}_COLOR.JP2` -- raw JP2 for every available image (the 37).
  - `{ObsId}_COLOR.LBL` -- PDS3 metadata sidecar.
  - `lbl_metadata.parquet` -- unified table of parsed LBL fields per ObsId
    (incidence, emission, phase, solar longitude, scaling/offset for I/F, map
    scale m/px, swath dimensions). Built by `scripts/run_stage7a_fetch.py` after
    all `.LBL`s are present. Use this for any cross-image normalisation /
    photometric correction without re-parsing 37 LBLs.
- **Transient fetch error handling**: first fetch attempt hit a `WinError 10054`
  (PDS LPL connection reset) mid-stream after 197 MB. `scripts/run_stage7a_fetch.py`
  now retries each failed file up to 4 times with 1/5/15/45 s exponential backoff
  before giving up on it. The `.partial` tempfile is auto-cleaned between attempts.
- **PLAN §8 q1 answer**: ~95 % colour coverage. The two without colour are
  `ESP_055690_2200` and `ESP_055978_2270` -- both will be flagged as "colour-test
  not applicable" in any cross-image Stage 7d output rather than dropped from the
  rock-abundance side.

### Cohort-wide colour metadata distribution (after fetch + LBL parse)

From `cache_v2/hirise_color/lbl_metadata.parquet` (37 rows):

- **Incidence angle range**: **40.2° to 72.4°** (cos i 0.30 to 0.76). The
  high-incidence subset (>60°) will have substantial within-polygon shadow
  fractions; per-image shadow masking via the Stage 4b `shadow_fraction` machinery
  becomes more important than the Stage 7.0 trio results suggested. (The trio's
  incidences were 40.7° / 45.6° / 60.3° -- bracketing but not spanning the cohort.)
- **Colour swath width range**: **2018 m to 6351 m**. The PLAN §2.1 originally
  estimated a ~1.0-1.3 km central swath; the trio refined that to ~2.4 km; the
  cohort spans nearly 3× either estimate. Several images cover essentially the
  full HiRISE width (~6 km) in colour, which means *every* boulder polygon in
  those images is colour-eligible -- a much better operational situation than the
  PLAN feared.
- **Colour map scale distribution**: 24 images at **0.5 m/px**, 13 at **0.25 m/px**.
  This is the OPPOSITE of what the trio sample suggested (2 at 0.25, 1 at 0.5).
  Note for Stage 7c: `min_polygon_pixels` and `ring_pixels` thresholds will need
  per-image scaling -- an 8-px polygon at 0.5 m/px is a 2 m² boulder; at 0.25 m/px
  it's 0.5 m². Either rescale thresholds in metres (preferred) or scale
  per-image-aware in pixels.
- **Files cached** (gitignored, total 8.9 GB):
  - `cache_v2/hirise_color/{ObsId}_COLOR.JP2` × 37 (range 86 MB - 705 MB,
    median ~190 MB).
  - `cache_v2/hirise_color/{ObsId}_COLOR.LBL` × 37 (~8 KB each).
  - `cache_v2/hirise_color/coverage.parquet` (audit result, 1 row per v2 ObsId).
  - `cache_v2/hirise_color/lbl_metadata.parquet` (per-image colour metadata for
    fast loading by Stage 7c-7e -- avoids re-parsing 37 LBLs).


## 2026-05-31 night — Stage 7b skipped (folded into 7c)

**Decision**: do not build a per-image colour reprojection cache. Stage 7b in
[`PLAN_Compositional.md`](PLAN_Compositional.md) §3 originally specified "Per-image
radiometric correction + reprojection of colour bands onto the CTX grid". This
stage is now **SKIPPED** and its responsibilities are absorbed into Stage 7c.

**Why**:

1. **The Stage 7.0 Test B probe already proved the "stay in source CRS" pattern works**:
   reproject each *tile bounds* CTX → source-CRS at read time, then do a windowed
   COLOR.JP2 read in the colour's own (SP1-corrected) CRS. No raster-level reprojection
   ever runs on the colour bands.
2. **No double resampling on the colour data**: a 7b pre-reproject + 7c per-tile mean
   would resample the colour bands twice. The source-CRS path resamples zero times —
   tile means are computed on native COLOR.JP2 pixels.
3. **No ~10 GB derived cache** ("colour-on-CTX-grid" raster cache would have been
   comparable to the raw 9.1 GB COLOR.JP2 cache). Disk + reproducibility win.
4. **Lambertian correction moves from a raster pass to per-tile arithmetic**, applied
   inside Stage 7c when extracting I/F. Per §5.3 (and the Stage 7.0 finding), the
   correction cancels in within-image diffs and band ratios, so it's only structurally
   needed for cross-image pooled features in Stage 7d — making "raster-level"
   correction unnecessary in the first place.

**What this changes**:

- No new `cache_v2/colour_reprojected/*` directory.
- `src.colour.windowed_colour_read(obs_id, tile_bounds_ctx, ...)` becomes the Stage
  7c primitive (built atop the Stage 7.0 Test B code in
  `scripts/probes/_stage7_feasibility.py`).
- Stage 7c output (`dataset_v2/features_colour.parquet`) carries Lambertian-corrected
  per-tile band means + ratios + `dust_index`, joinable on
  `(obs_id, scale_idx, ti, tj)` to the existing feature parquet.

**Tradeoff**: every Stage 7c run pays the cost of re-doing windowed COLOR.JP2 reads.
For 13-band tiles × ~37 images × ~thousands of tiles per image, this is many
windowed reads — but each is a small block on a local JP2. Empirically the Stage 7.0
probe ran the same pattern in ~minutes per image. Acceptable.

**Documented in**: `PLAN_Compositional.md` §3 table (7b row struck through),
top-of-file revisions list (item 7), §7 cost table (7b row marked SKIPPED).

## 2026-06-01 — Stage 7c per-tile colour features (cohort run done)

**Stage 7c done.** `dataset_v2/features_colour.parquet` written: **9 860 rows
across 36 of 37 colour-eligible images**, computed by
`scripts/run_stage7c_features.py` over 145 min (~2.4 hr) wall-clock on the
local box. Stage 7d/7e can now run against this parquet without re-touching the
JP2 cache.

### Method (matches the architectural decision recorded above)

For every S=64 tile in every colour-covered image:

1. Reproject the tile's (xmin, ymin, xmax, ymax) from the CTX target CRS into
   the HiRISE source CRS via `src.colour.ctx_bounds_to_source_bbox(...)` (the
   "stay in source CRS" pattern that replaced Stage 7b).
2. Windowed-read the 3-band COLOR.JP2 around that bbox
   (`src.colour.windowed_colour_read`).
3. Compute the per-band mean of valid (non-pad-zero) pixels via
   `src.colour.region_means`, requiring `n_pixels >= 64` (the
   `MIN_TILE_PIXELS` floor; ~16 m² at 0.5 m/px or ~4 m² at 0.25 m/px — really
   just "tile barely overlaps swath").
4. Convert mean DN → I/F per-image via `I/F = DN * scaling_factor + offset`
   from the COLOR.LBL (`scaling_factor` varies ~5× across the cohort, so this
   step is mandatory for Stage 7d cross-image pooling — initial implementation
   missed it and was caught during the trio sanity-run).
5. Apply per-image Lambertian correction: divide the I/F means by
   `cos(incidence_deg)`. cos(i) range across the cohort is 0.30 to 0.76.
6. Emit columns: `obs_id, scale_idx, ti, tj, n_color_pixels, IR_iof, RED_iof,
   BG_iof, IR_over_RED, IR_over_BG, dust_index_RED_over_BG, cos_incidence`.
   `dust_index_RED_over_BG = RED_iof / BG_iof` per PLAN §5.1 — the proxy used
   to discriminate composition signals from dust-loading signals.

### Cohort numbers (from `scripts/probes/_summarise_stage7c.py`)

- **Coverage**: 36 / 37 colour-eligible images produced rows. The one excluded
  is `ESP_046803_2325` — present in the v2 manifest with a Stage 1 sidecar
  + COLOR.JP2 + COLOR.LBL, but no `dataset_v2/labels/ESP_046803_2325.parquet`
  (Stage 4 was never run for this ObsId for some reason; predates this
  session). Flagged as a follow-up; not a blocker for Stage 7d.
- **Per-image tile counts**: min 95, median 251, max 771 (kept).
- **Per-image retention** (kept / total S=64 tiles in the labels parquet):
  24-31 % across all 36 images. Consistent across high-density (`ESP_017355_2260`
  771 / 2927 = 26 %) and low-density (`ESP_048688_2085` 95 / 316 = 30 %) images —
  retention is dictated by the colour swath width (~2-6 km vs the 6 km full
  HiRISE footprint), not by tile content.
- **Cohort I/F medians**: IR=0.169, RED=0.165, BG=0.077. All inside the
  expected 0.05-0.30 range for Mars dusty-equatorial regolith.
- **Cohort dust_index (RED/BG) range**: p5=1.64, p50=1.95, p95=2.35.
- **Per-image dust_index medians**: range 1.53 - 2.45 — real cross-image
  variation, ~50 % spread, consistent with regional dust differences.
- **cos(i) range**: 0.30 (ESP_066634_2210, incidence 72°) to 0.76 (the high-sun
  bracket). The wide range means Lambertian correction is doing meaningful work
  for cross-image pooling (Stage 7d) — without it the high-incidence images
  would systematically look 2.5× darker than the low-incidence ones.

### Runtime characteristics + the slow-outlier note

Wall clock per image varied from **11 s** (ESP_051943_2270, 632 tiles) to
**61 min** (ESP_049242_2115, 572 tiles) — the latter is a 100× slowdown vs
its peers despite similar tile count + map_scale. A second outlier
(ESP_053989_2260, 25 min) showed the same pattern. Both are 0.5 m/px JP2s with
no obvious metadata difference from the fast peers. Likely cause: codec-level
tile-cache misses on the specific JP2 layout of those products. Workaround if
re-runs become a bottleneck: download into a local GeoTIFF first (one-off
~3 min decode), then process from there. Acceptable as-is for the one-shot
Stage 7c run.

### Outputs (gitignored, regenerable)

- `dataset_v2/features_colour.parquet` — 9 860 rows; joinable on
  `(obs_id, scale_idx, ti, tj)` to `dataset_v2/labels/{ObsId}.parquet`.
- `dataset_v2/features_colour_trio.parquet` — superseded sanity-run artefact
  (trio only, 542 rows). Safe to delete; left in place pending Stage 7d use.

### Code changes (tracked)

- `src/colour.py`: added `ctx_bounds_to_source_bbox` + `windowed_colour_read`
  helpers (the Stage 7c primitives that absorbed the would-be 7b reprojection).
- `scripts/run_stage7c_features.py`: new — argparse-driven runner with `--only`,
  `--scale-idx`, `--out`, `--min-tile-pixels` flags.
- `scripts/probes/_verify_stage7c_trio.py`: cross-checks the trio output's
  within-image direction-of-effect against the Stage 7.0 verdict table.
- `scripts/probes/_summarise_stage7c.py`: cohort summary helper (numbers above).
- `tests/test_colour.py`: 8 unit tests covering `region_means`,
  `lambertian_correct`, `ctx_bounds_to_source_bbox`, `windowed_colour_read`,
  and the IR/RED/BG band-index constants. All passing.
- `.gitignore`: `scripts/probes/*.log` added (for the run log).
- **Pytest suite**: 233 fast tests pass; +8 new from `test_colour.py`.

### Stage 7d / 7e prerequisites now satisfied

`dataset_v2/features_colour.parquet` is the single input both stages need.
Stage 7d (pooled cross-image boulder-rich vs boulder-poor test) can now run
without further data engineering. Stage 7e (formal dust analysis) refines the
`dust_index` proxy + adds shadow masking but uses the same parquet as its base.

## 2026-06-02 — Stage 7d pooled cross-image colour test (PASS)

### Verdict

**Stage 7d passes the PLAN_Compositional.md §4 + §5 criteria** at the chosen
thresholds. The boulder-rich vs boulder-poor colour difference is real, broad
across the cohort, AND retains a statistically robust residual after per-image
dust control — supporting both the dust-age narrative (most of the raw effect)
AND a composition narrative (the residual). The band ratios `IR/RED` and
`IR/BG` are the **most compositional** features (smallest dust-shrinkage), as
expected from the [HiRISE colour
documentation](https://www.uahirise.org/pdf/color-products.pdf): ratios are
sensitive to ferric vs ferrous iron and largely independent of dust loading.

### Method (per PLAN §4.2, §4.3, §5.2)

1. Inner-join `dataset_v2/features_colour.parquet` (Stage 7c, 9 860 rows × 36
   images) with `dataset_v2/labels/{ObsId}.parquet` on
   `(obs_id, scale_idx, ti, tj)` at S=64.
2. Add two partition columns:
   - **P4_area** — `fractional_area >= 1e-2` (P4 binary promotion threshold)
   - **P2_count** — `boulder_count > 50`
3. Drop images with < 5 rich OR < 5 poor tiles under the partition. P4_area
   keeps 30 / 36 images (8 355 tiles); P2_count keeps 33 / 36 (8 995 tiles).
4. Per-image standardise each colour feature (subtract mean, divide by std).
5. Pooled Mann-Whitney U + Cohen's d on rich vs poor under three transforms:
   - `mann_whitney_raw` — raw values
   - `mann_whitney_standardised` — z-scored per image (the §4.2 headline)
   - `mann_whitney_partial_dust` — per-image residualised on
     `dust_index_RED_over_BG` (the §5.2 discriminator)
6. Per-image MW + Cohen's d on raw features (heterogeneity check).
7. Spearman rho vs continuous `boulder_count` (per-image standardised + the
   partial-dust variant + per-image) — §4.3 monotonicity check.

### Numbers (`dataset_v2/stage7d_pooled.parquet`, 639 rows)

**Condition 1 — cross-image significance (P4_area, pooled standardised)**:
all 6/6 features pass the |d| ≥ 0.1, p ≤ 1e-3 bar.

| feature | Cohen's d | p-value |
|---|---|---|
| IR_iof | -0.372 | 1.7e-73 |
| RED_iof | -0.365 | 5.1e-69 |
| IR_over_RED | -0.331 | 1.7e-61 |
| BG_iof | -0.346 | 1.1e-59 |
| IR_over_BG | -0.279 | 9.9e-43 |
| dust_index_RED_over_BG | -0.252 | 9.3e-33 |

P2_count gives the same ordering with slightly smaller magnitudes
(|d| 0.21 – 0.28) — the binary partition rule is not driving the conclusion.

**Per-image sign consistency** (P4): 0.77 – 0.83 of eligible images carry the
same effect-size sign as the pooled result, across all 6 features. The signal
is broad, not driven by outliers.

**Condition 2 — dust discrimination (partial-dust, P4_area)**: all 5 non-dust
features survive at |d| ≥ 0.05, p ≤ 0.05.

| feature | partial-d | partial-p | shrinkage vs raw |
|---|---|---|---|
| IR_over_BG | -0.162 | 1.5e-17 | 42 % |
| IR_over_RED | -0.152 | 8.5e-18 | 54 % |
| IR_iof | -0.122 | 6.2e-25 | 67 % |
| RED_iof | -0.082 | 9.1e-18 | 77 % |
| BG_iof | -0.068 | 3.9e-16 | 80 % |

**Key compositional interpretation**: the band-ratio features `IR/BG` and
`IR/RED` shrink the *least* (42 %, 54 %) under dust control — i.e. they carry
the strongest composition signal. The single bands shrink the most (67 – 80 %),
i.e. their effect is mostly dust-loading. This matches [HiRISE colour
documentation](https://www.uahirise.org/pdf/color-products.pdf): ratios index
ferric/ferrous mineralogy and are robust to a multiplicative dust albedo
shift, whereas single-band differences are largely an "amount of dust" signal.

**Condition 3 — continuous-target monotonicity (Spearman rho vs `boulder_count`)**:
all 6 standardised Spearman rhos (−0.123 to −0.172) sign-match the binary
effect direction. Partial-dust Spearman (rhos −0.119 to −0.141) preserves the
sign on 5/5 non-dust features.

### Interpretation

- Boulder-rich tiles are **systematically darker** than boulder-poor tiles of
  the same image (negative IR/RED/BG effects).
- Boulder-rich tiles have **lower dust_index = RED/BG**, i.e. less dust loading
  — consistent with either younger emplacement age OR boulders shedding dust
  off their flanks.
- After per-image residualisation on dust_index, a real residual remains in
  the ratio features (IR/BG, IR/RED) — boulder-rich material has a different
  ferric/ferrous signature even after accounting for dust loading. This is the
  composition signal the PLAN was set up to detect.
- The dust narrative explains roughly 50–80 % of the raw observed effect; the
  composition narrative explains the remaining 20–50 %.

### Decisions encoded

1. **Partition rule**: emit both P4_area and P2_count (per the 2026-06-01
   user decision at session start). Conclusions are rule-robust.
2. **Per-image inclusion threshold**: `min_per_class = 5` rich AND 5 poor
   tiles. Drops 6 images under P4, 3 images under P2 — including
   ESP_054622_2240 (340 rich / 0–2 poor — monoclass) and ESP_059686_2235 /
   ESP_055055_2255 / ESP_048688_2085 (low rich counts). The signal is robust
   to this filter.
3. **Spearman included** for the §4.3 continuous-monotonicity check, not just
   the binary partition tests (per the 2026-06-01 user decision).

### Code changes (tracked)

- `src/stage7d_pooled.py` — new: load_joined, add_partitions, cohen_d,
  mann_whitney_with_effect, spearman_with_p, per_image_standardise,
  residualise_per_image, eligible_images, run_pooled_binary_tests,
  run_per_image_binary_tests, run_spearman_tests, run_all.
- `scripts/run_stage7d_pooled.py` — runner; ~2 s on the cached parquet.
- `scripts/probes/_inspect_nb15_outputs.py` — dev probe to read executed
  notebook outputs.
- `tests/test_stage7d.py` — 19 new unit tests (synthetic data; no parquet
  I/O). All passing.
- `notebooks/_build_15.py` + `notebooks/15_stage7d_pooled.ipynb` — built +
  executed.
- `reports/figures/stage7d_pooled_effect_sizes.png`,
  `stage7d_per_image_effects.png`, `stage7d_dust_discriminator.png`,
  `stage7d_spearman_continuous.png`.
- **Pytest suite**: 272 tests pass (+19 over Stage 7c baseline).

### Outputs (gitignored, regenerable)

- `dataset_v2/stage7d_pooled.parquet` — 639 rows. Schema: `level`, `obs_id`,
  `partition_rule`, `feature`, `test_type`, `controls_for`, `n_images_pooled`,
  `n_rich`, `n_poor`, `n_total`, `mean_rich/poor`, `median_rich/poor`,
  `std_rich/poor`, `statistic`, `p_value`, `effect_size`, `effect_size_type`.

### Next: Stage 7e (formal dust analysis) is the natural follow-up

The partial-dust discriminator above uses the crude `RED/BG` proxy and
no shadow masking. Stage 7e refines both — a literature-validated dust index
([Atwood-Stone & McEwen 2013](https://doi.org/10.1029/2013GL058355))
and explicit shadow exclusion via the Stage 4b `shadow_fraction` machinery —
and should sharpen (or attenuate) the composition residual reported above.

## 2026-06-03 — Stage 7d wrap-up: shadow masking + per-image attribution + final docs

### Project wrap-up scope

Brian asked to wrap up a reportable version of the project. Scope picked
(option B from the 2026-06-03 AskUserQuestion): shadow masking + per-image
attribution before the writeup. Skipped the Atwood-Stone & McEwen 2013 dust
index refinement (Stage 7e full) and the Path A modeling bank — both flagged
as future work in the writeup.

### Shadow masking — tile-level filter on Stage 4b `shadow_fraction`

The Stage 4b features parquet (`dataset_v2/features/{ObsId}.parquet`)
already carries a per-tile `shadow_fraction` column (fraction of in-tile
CTX pixels below the per-image shadow-DN threshold) at S=64. Cohort
distribution: median 0.0605, mean 0.053, max 0.87. Cheap shadow refinement
path: inner-join the column and drop tiles where `shadow_fraction > T` for
T ∈ {0.05, 0.10, 0.20}, compare effect-size pivots. **Tile-level filtering
on CTX-derived shadow_fraction, not pixel-level masking on the COLOR.JP2
itself** — coarser but cheap. Pixel-level masking is deferred to Stage 7e.

Implementation: extended `src/stage7d_pooled.py` with `attach_shadow_fraction`
(inner-join from `dataset_v2/features/`) and `filter_shadow` (no-op if
T=None). Runner accepts `--shadow-threshold` and `--ctx-features-dir` flags.

### Shadow sweep — striking pattern

Pooled standardised + partial-dust effects (P4_area):

| feature | test | baseline | T=0.20 | T=0.10 | T=0.05 |
|---|---|---:|---:|---:|---:|
| IR_iof | raw d | -0.372 | -0.358 | -0.287 | -0.238 |
| IR_iof | partial-dust d | -0.122 | -0.199 | -0.183 | -0.184 |
| BG_iof | raw d | -0.346 | -0.358 | -0.306 | -0.275 |
| BG_iof | partial-dust d | -0.068 | -0.151 | -0.133 | -0.141 |
| IR_over_BG | raw d | -0.279 | -0.216 | -0.127 | -0.067 |
| IR_over_BG | partial-dust d | -0.162 | -0.129 | -0.140 | -0.127 |
| dust_index | raw d | -0.252 | -0.191 | -0.100 (m) | -0.043 (ns) |

(m = marginal p=0.011; ns = not significant p=0.55.)

Three findings:

1. **The raw `dust_index` effect is mostly shadow-driven.** At T=0.05
   the raw rich-vs-poor `dust_index` effect loses significance entirely.
   The crude RED/BG proxy was indexing shadow as much as it was indexing
   actual dust loading.
2. **Partial-dust single-band effects GROW under shadow filtering** —
   IR_iof goes from -0.122 (baseline) to -0.183 (T=0.10). The composition
   residual was being *masked* by shadow contamination at baseline, not
   amplified by it.
3. **Partial-dust band-ratio effects are roughly stable** across thresholds
   (IR/BG: -0.162 → -0.127). The ratios were robust to shadow by
   construction.

Bottom line: composition narrative **strengthens** under shadow control,
particularly for single bands. Headline T=0.10 partial-dust |d| ranges
0.13 – 0.18 across all 5 non-dust features (p ≤ 1e-8).

### Per-image attribution classifier

Implementation: `classify_image` + `build_attribution_table` in
`src/stage7d_pooled.py`. Conservative 3-way classifier on the band-ratio
features (`IR_over_BG`, `IR_over_RED`):

- `composition_residual` — per-image raw effect passes (|d| ≥ 0.20, p ≤ 1e-3)
  AND per-image partial-dust effect also passes (|d| ≥ 0.10, p ≤ 0.05) on
  at least one ratio feature
- `dust_attributable` — raw passes but partial-dust does not
- `no_signal` — no raw effect detected (either truly null OR underpowered)
- `inconclusive` — ambiguous cross-feature

At T=0.10 / P4_area / 26 eligible images:
- 5 `composition_residual`
- 5 `dust_attributable`
- 16 `no_signal`
- 0 `inconclusive`

The 16 `no_signal` images are mostly the small-n-per-class images
(median n_poor ≈ 60, but ESP_055253_2245 has n_poor=165 with n_rich=8;
ESP_069763_2235 has n_rich=99 with n_poor=11). Per-image power is the
binding constraint on the per-image table, not lack of signal — the
pooled cohort evidence is much stronger than the per-image counts
suggest.

**Two striking per-image cases (composition_residual)**: ESP_066634_2210
and ESP_076723_2265 both have *positive* raw rich-vs-poor d (rich looks
redder / more ferric) but *negative* partial-dust d (rich less ferric
after controlling for dust). Clean per-image demonstration that the
composition residual lives independently of, and in opposite direction
from, the dominant dust signal.

### Stage 7d wrap-up verdict

Stage 7 as scoped in PLAN_Compositional.md has landed at a publishable +
properly-bounded conclusion. The boulder-rich vs boulder-poor colour
difference is real (loud), is ~50–80 % dust-attributable and ~20–50 %
composition-attributable in the cohort, and the composition residual
survives both per-image dust control and tile-level shadow filtering.
The composition residual direction is "boulders less ferric-altered than
surrounding regolith," which is consistent with EITHER a locally-sourced
surface-maturity scenario OR a transported provenance scenario (the
[Rodriguez 2016](https://doi.org/10.1038/srep25106) / [Costard 2017](https://doi.org/10.1002/2016JE005230)
megatsunami hypothesis is the natural transported candidate given the
cohort latitudes). Stage 7d alone cannot disambiguate; future work in
the §11 plan covers it.

### Code changes (tracked)

- `src/stage7d_pooled.py` — added `attach_shadow_fraction`,
  `filter_shadow`, `run_per_image_partial_dust`, `classify_image`,
  `build_attribution_table`, `ATTRIBUTION_FEATURES` / thresholds.
- `scripts/run_stage7d_pooled.py` — added `--ctx-features-dir`,
  `--shadow-threshold`, `--attribution-out`, `--no-attribution` flags;
  emits attribution table by default.
- `scripts/probes/_dump_attribution.py` — dev probe to print the per-image
  attribution table for the writeup.
- `tests/test_stage7d.py` — +9 tests (filter_shadow no-op / threshold /
  missing-column error; run_per_image_partial_dust skips dust col;
  run_all with per_image_partial_dust; classify_image no_signal /
  dust_attributable / composition_residual; build_attribution_table).
- `notebooks/_build_16.py` + `notebooks/16_stage7d_shadow_attribution.ipynb`
  — built + executed.
- `reports/figures/stage7d_shadow_sweep.png`,
  `stage7d_partial_dust_shadow_compare.png`,
  `stage7d_attribution_bars.png` — three new figures.
- `docs/compositional.md` — new: Methods + Results + Discussion +
  Limitations + Future-work + References paper-style writeup of Stage 7.
- `docs/modeling.md` — new: Methods companion to modeling_results.md.
- `docs/index.md` — updated to point at the two new docs.
- **Pytest suite**: 281 tests pass (+9 over Stage 7d baseline; +28 over
  Stage 7c baseline).

### Outputs (gitignored)

- `dataset_v2/stage7d_pooled.parquet` (baseline, 639 rows)
- `dataset_v2/stage7d_pooled_shadow_{0.05,0.10,0.20}.parquet` (sweep)
- `dataset_v2/stage7d_per_image_attribution.parquet` (baseline attribution)
- `dataset_v2/stage7d_attribution_shadow_{0.05,0.10,0.20}.parquet` (sweep)

### Unfinished follow-ups (recorded so they don't get lost)

- Stage 7e Atwood-Stone & McEwen 2013 dust-index refinement
- Stage 7e pixel-level HiRISE-side shadow masking
- ESP_046803_2325 Stage 4 backfill (would lift cohort 36→37)
- Provenance disambiguation: manual terrain classification → Robbins &
  Hynek crater catalog cross-ref → upstream source-unit comparison
- Path A model bank: P1+P2 full-v2 LOIO sweep promotion

## 2026-06-03 — Stage 7 Tier 1 + Tier 2 provenance disambiguation

### Scope

Per HANDOFF D1 + PLAN_Compositional.md §11, ran the first two of three
tiers of the provenance-disambiguation programme. Tier 3 (CRISM/HiRISE
upstream source-unit comparison) is left as future work.

### Tier 1 — terrain context cross-reference (Fisher's exact)

User supplied a pre-existing mapping spreadsheet
(`C:/Users/brian/Downloads/Mapping_Images_33_36.xlsx`, Sorted_Lon sheet)
with geological terrain annotations for 37 of 39 v2 ObsIds. Free-text
notes parsed by [`scripts/probes/_terrain_classify.py`](../scripts/probes/_terrain_classify.py)
into structured boolean flags + a `terrain_category`.

Key transport indicators:
- `deposit_flag` = `"Deposit!"` appears in note (6 of 39 cohort images)
- `streamlined_flag` = `"streamlined"` appears in note (1 of 39)
- combined `transport_indicator = deposit_flag OR streamlined_flag` (7 of 39)

Fisher's exact two-sided on `transport_indicator × is_composition_residual`:

| Partition | OR | p | n_trans_comp / n_trans | n_other_comp / n_other |
|---|---:|---:|---:|---:|
| **P2_count** | **23.0** | **0.018** | 3 / 6 | 1 / 24 |
| P4_area | 12.0 | 0.059 | 3 / 7 | 1 / 17 |

**Significant at p < 0.05 under P2_count partition.** Direction holds in
both partitions; magnitude smaller and marginal under P4_area.

### Methodology correction (2026-06-04)

The two ObsIds missing from the spreadsheet (`ESP_017355_2260` in
composition_residual, `ESP_076499_1160` in no_signal) were originally
scored `transport_indicator = False` (treating "missing data" as the
default value). Under that approach the numbers were P2 OR=12.0
p=0.034 and P4 OR=6.38 p=0.10. Brian flagged 2026-06-04 that this is
imputation, not honest exclusion — and since `ESP_017355_2260` is a
composition_residual image whose imputed-False value lands it in the
"other-terrain composition_residual" cell, the imputation
mechanically dilutes the association. The honest fix is to **exclude
those two ObsIds from the test** (they have no terrain row in the
contingency table). The above corrected numbers use the
honest-exclusion approach. The earlier impute-as-False numbers are
preserved here for the historical record but the corrected numbers
supersede in the canonical writeups (`docs/compositional.md` §4.7,
`docs/compositional_slim.md` §4.4).

A second correction: the original P4_area cell-count row above and in
the earlier `compositional.md` table also had a transcription error
(swapped which of the 5 composition_residual images were in the
transport-flagged vs other-terrain cell). The actual contingency
table at P4 under imputation was 3/7 transport-flagged + 2/19
other-terrain, not 2/6 + 3/20. The OR and p reported on the
imputation approach (6.38, 0.10) were the correct test statistic but
the cell counts in the table were wrong. The corrected numbers
above use the honest-exclusion contingency table.

Output: `dataset_v2/terrain_classification_v2.parquet`.

### Tier 2 — crater-distance cross-reference (Kruskal-Wallis)

Fetched the Robbins 2012 Mars crater database from
[craters.sjrdesign.net](https://craters.sjrdesign.net/) (12 MB
TSV.zip → 56 MB TSV, 384 343 craters globally, ≥ 1 km diameter).
Verified projection: both manifest `CenterLat/Lon_180` and Robbins
`LATITUDE/LONGITUDE_CIRCLE_IMAGE` are planetocentric, east-positive
-180..180, IAU 2000 Mars frame. Computed great-circle distance
(haversine on a sphere of R = 3389.5 km, IAU 2009 mean) from each
HiRISE image center to nearest crater of D ≥ {1, 5, 10, 25} km. Both
center distance and rim distance (center distance − D/2, floored at 0)
emitted.

Kruskal-Wallis across 3 attribution categories on rim distance:

| Partition | Diameter threshold | KW H | KW p |
|---|---|---:|---:|
| P2_count | D ≥ 5 km | 0.62 | 0.73 |
| P2_count | D ≥ 10 km | 0.66 | 0.72 |
| P4_area | D ≥ 5 km | 0.39 | 0.83 |

Mann-Whitney composition_residual vs rest also returns no significant
separation at any diameter threshold (all p > 0.30).

**Null finding.** Crater proximity does not separate the attribution
categories. The null is geologically informative: under the
crater-ejecta-locally-sourced interpretation,
`composition_residual` images should have shown significantly closer
crater rim distances. They do not — **weakly disfavours
locally-sourced-from-crater-ejecta**.

Output: `dataset_v2/crater_distance_v2.parquet`. Catalog cached at
`cache_v2/craters/RobbinsCraters_20121016.tsv`.

### Combined verdict

The two tests together give **modest empirical support for the
transported-with-distinct-deposit-character interpretation** over
crater-ejecta-locally-sourced:

1. Crater-ejecta-locally-sourced predicts Tier 2 positive (crater
   proximity). We don't see it.
2. Transported-with-deposit-character predicts Tier 1 positive (deposit
   annotation correlation). We see it at p = 0.018 under P2
   (OR = 23.0; honest-exclusion). The earlier-reported p = 0.034 was
   the impute-as-False number; the methodology correction is recorded
   above.

The **surface-maturity-locally-sourced** alternative (boulders = fresh
parent rock, surroundings = weathered version, from non-crater bedrock)
is NOT directly tested by Tiers 1 or 2 and remains in play. Tier 3
(CRISM/HiRISE upstream source-unit colour comparison) would be needed
to distinguish transported-from-highland-source from
regional-maturity-of-local-bedrock.

### Caveats (recorded in notebook 17 + docs/compositional.md §4.7)

- n = 5 in `composition_residual`; Tier 1 significant but marginal, Tier 2 underpowered.
- Brian's terrain annotations are single-rater.
- Robbins 2012 catalogues only D ≥ 1 km; sub-km secondaries are missed
  (Tier 2 null is robust to this only if the relevant ejecta sources
  are ≥ 1 km).
- Image-center vs tile-level test resolution — a tile-level analysis
  inside each composition_residual footprint would refine but cannot
  easily lift the n = 5 power limitation.

### Code artefacts (tracked)

- `scripts/probes/_dump_browse_terrain.py` — manifest + browse-URL dump
- `scripts/probes/_dump_terrain_excel.py` — Excel inspect
- `scripts/probes/_terrain_join_v2.py` — v2-cohort filter on the spreadsheet
- `scripts/probes/_terrain_classify.py` — terrain parser + emits parquet
- `scripts/probes/_terrain_stats.py` — Fisher's exact (Tier 1)
- `scripts/probes/_crater_distance.py` — R&H fetch + per-image distance (Tier 2)
- `notebooks/_build_17.py` + `notebooks/17_provenance_disambiguation.ipynb` (executed)
- `reports/figures/stage7_tier1_terrain_attribution.png`
- `reports/figures/stage7_tier2_crater_distance.png`
- `docs/compositional.md` — added §4.7 (Tier 1 + Tier 2 results); updated
  §6.2 Q3 verdict (was "not achieved", now "partially achieved");
  updated §8 future-work (Tiers 1 + 2 → done, Tier 3 remains).

### Outputs (gitignored)

- `dataset_v2/terrain_classification_v2.parquet` (39 rows, structured terrain)
- `dataset_v2/crater_distance_v2.parquet` (39 rows, per-image distances)
- `cache_v2/craters/RobbinsCraters_20121016.tsv` (56 MB)

## 2026-06-03 — Slim modeling variant for project writeup

### Scope

User asked for a simplified modelling variant that uses fewer features
and stands on its own as the model for the project report, separate
from the full 52-feature implementation documented in
[`modeling.md`](docs/modeling.md). Built so the reportable modeling
story can be explained at a higher level without losing the headline
conclusion.

### Design choices

- **Feature set**: 5 features, two physically-motivated mechanisms.
  Option B from the planning AskUserQuestion (recommended pick):
  `shadow_fraction`, `shadow_fraction_strict`, `bright_cap_fraction`
  (shadow mechanism); `grad_mag_std`, `intensity_std` (texture
  roughness mechanism). All five are derivable from CTX DN values
  alone.
- **Cohort filter**: drop the 2 manifest `unknown`-BoulderLabel
  images (`ESP_017355_2260`, `ESP_076499_1160`). User flagged a
  preference for "filters that are decisions from before the modeling
  step as opposed to afterwards" → no anti-signal / Stage-6b
  post-modeling filters. A coregistration-peak-correlation cut was
  considered (user originally suggested ≥ 0.9 → 0 images cleared,
  then ≥ 0.5 → all 38 cleared, so effectively no coreg filter).
  Final cohort: 36 of 38 images.
- **Target**: `boulder_count` (P2 promotion winner).
- **Variant**: `lightgbm_two_stage_balanced` (P1 promotion winner)
  with default LightGBM hyperparameters; no hyperparameter tuning.
- **Scale**: S=64 (320 m tiles) only.
- **CV**: LOIO on the 36-image cohort.

### Coregistration-peak-correlation distribution (recorded for context)

The v2 cohort's `coreg_peak_correlation` values span 0.58 – 0.88
(median ~0.71). v2 uses block-median phase correlation rather than
single-peak phase correlation, so the absolute values are not
directly comparable to standard cross-correlation literature. None of
the 38 images is below an empirically reasonable noise floor (~0.3 –
0.5 for cross-instrument HiRISE→CTX matching after decimation), so
the cut at 0.5 effectively removes nothing.

### Numbers

- **Pooled Spearman ρ** = +0.275 across 33 102 held-out tiles from 36
  LOIO folds (p ≪ 1e-50).
- **Per-fold ρ**: mean +0.151, std 0.216, median +0.130, range
  -0.378 to +0.684.
- **Per-image AUC at fa_gt_1e-2**: 35 folds with both classes;
  median 0.572, max 0.880, min 0.311. 14 % above 0.70 ("usable on
  this image"); 26 % below 0.50 ("anti-signal on this image").

### Headline framing for the writeup

The slim model writeup ([`docs/modeling_slim.md`](docs/modeling_slim.md))
is positioned as **the model used in the project report**, not as a
comparison piece against the 52-feature full model. User explicit
direction 2026-06-03: "we don't need to make comparisons to the
original full run -- this is just supposed to be a simpler/slimmer
version that will be easier to explain." First drafts of the doc
included slim-vs-full comparison tables and figures; those were
removed in the final version. The headline conclusion is the same as
the full modeling writeup: real but small ranking signal, per-tile
boulder-rich classification not tractable at CTX resolution, model
keys on the physically expected features.

### Code artefacts (tracked)

- [`scripts/run_modeling_slim.py`](scripts/run_modeling_slim.py) —
  the runner; ~2 min on a CPU-only laptop.
- [`scripts/probes/_modeling_slim_figures.py`](scripts/probes/_modeling_slim_figures.py)
  — the figure builder.
- [`docs/modeling_slim.md`](docs/modeling_slim.md) — the writeup.
- [`docs/index.md`](docs/index.md) — index entry added.
- `reports/figures/modeling_slim_per_image_auc.png` — per-image AUC
  distribution at the boulder-rich threshold (the only figure in
  the writeup).

### Outputs (gitignored, regenerable)

- `dataset_v2/modeling_slim_predictions.parquet` (33 102 rows)
- `dataset_v2/modeling_slim_summary.parquet` (37 rows = 36 folds +
  1 pooled row)

### Recorded for posterity (not surfaced in docs)

An earlier version of the runner trained the 52-feature full model
alongside the slim model on the same filtered training set and
computed pooled and per-fold comparison metrics. The pooled
ρ values were essentially identical (slim +0.275 vs full +0.280),
with the slim model showing slightly higher per-fold mean ρ due to
lower overfitting variance on a few folds. Recorded here in case a
future session wants to revisit the "less is more" framing, but
removed from the current writeup per user direction above.

## 2026-06-10 -- W0 "bank the wins": P2 promoted; P1/P5 null at LOIO; hurdle retained with evidence; Stage 6a partial carry

First session of the model-usability program (PLAN_ModelUsability.md).
All numbers: full-v2 LOIO (38 folds), dataset_v2, n_estimators=400 /
lr=0.05 / early_stopping=40. Sweep artifacts:
`models/_sweep_w0/20260610T221932Z` (S=64 matrix, 3 variants x 2 targets),
`models/_sweep_w0/20260610T223114Z` + `20260610T223410Z` (Stage 6a S=32
pair), `models/_sweep_binary/20260611T002603Z` (P5 classifier).
Probes: `scripts/probes/_sweep_w0.py`, `_w0_paired_deltas.py`.
Harness consistency check: vanilla two_stage x fractional_area reproduced
the documented historical baseline exactly (rho +0.1689 / presence AUC
0.579); the P1+P2 cell reproduced the Stage 6b sweep baseline exactly
(rho +0.1431 / PR-AUC 0.5431).

### Promoted regression recipe (the W0 baseline all later work compares against)

**`lightgbm_two_stage_balanced` x `boulder_count` @ S=64, 51 base Stage-4b
features** -- rho +0.1431, presence AUC 0.6149, PR-AUC 0.5431, normalised
lift 0.5284, precision@top-5% 0.5679, recall@top-5% 0.0624. Per-image
meaningful-AUC (bc >= 50): median 0.594 / max 0.979 / 23.7% of images
> 0.70 / 28.9% < 0.50 (bimodality persists -> W1).

**Tier 1 reference classifier** (unchanged): `lightgbm_classification`
on `fa_gt_1e-2` @ S=64 -- AUC 0.615 +/- 0.114, lift@top-K 1.430,
ECE 0.264 (`models/_sweep_binary/20260529T075754Z`).

### Verdicts (paired per-fold Wilcoxon, n=38)

1. **P2 (target = boulder_count): PROMOTED.** vs fractional_area on the
   same variant: PR-AUC +0.162 (win rate 89%, p < 1e-4), precision@top-5%
   +0.182 (81%, p < 1e-4). Spearman -0.012 (p = 0.38, noise). The +22%
   dev win carried and grew. *Caveat recorded*: the operational metrics
   use the bc >= 50 positive definition vs fa > 1e-2 -- thresholds
   designed equivalent (50 boulders ~ 1% area at S=64) but not identical
   positive sets; cross-target comparisons inherit this convention from
   the dev sweeps.
2. **P1 (balanced presence head): NULL at LOIO.** vs vanilla two_stage on
   boulder_count: all paired deltas n.s. (Spearman -0.010 p=0.10, PR-AUC
   -0.003 p=0.10, meaningful-AUC -0.007 p=0.10). The +0.017 rho dev win
   did not replicate. Kept as default anyway for the calibrated p_pos
   mechanism (no measurable cost).
3. **Single-stage hurdle test: TWO-STAGE RETAINED, now with evidence.**
   `lightgbm_log1p_huber` ties on pooled PR-AUC / lift / precision@top-5%
   (all p > 0.4) but loses per-image meaningful-AUC by -0.022 paired
   (win rate 34%, p = 0.008) and Spearman -0.027 (p = 0.13) on
   boulder_count. Closes the 2026-06-08 open question (memory
   `modeling-single-stage-future`): the hurdle is kept on per-image
   detection evidence, not inertia.
4. **P5 (classifier without scale_pos_weight): NULL.** AUC -0.007
   (p=0.46), ECE -0.003 (p=0.25), Brier -0.0006 (p=0.21), lift -0.15
   (p=0.60). The predicted ECE collapse (0.26 -> ~0.05) did NOT happen:
   held-out-image miscalibration is dominated by between-image base-rate
   / distribution shift, not by the loss weighting. Third dev-win-fails-
   at-LOIO data point tonight (with P1 + the Stage 6c gate history), all
   consistent with per-image heterogeneity as the binding constraint.
   Variant `lightgbm_classification_balanced` stays in the registry,
   documented as tested-null.
5. **Stage 6a 5x5 @ S=32 full-v2: STRICT FAIL, partial carry.** Baseline
   S=32: rho +0.0945 / PR-AUC 0.2750 / prec@5% 0.2974. With nbr_s5
   features (206 cols): rho +0.1665 (delta +0.072 >= +0.05 PASS),
   meaningful-AUC +0.068, PR-AUC +0.0166 (< +0.03 FAIL), prec@5% +0.020.
   The dev PASS carries on Spearman but not PR-AUC. Absolute S=32
   performance (PR-AUC ~0.29) remains far below S=64 (0.543); S=64 stays
   the operating scale and the S=64 recipe does NOT take nbr features
   (dev showed S=64 already at the spatial-integration ceiling).
   Artifacts kept: `dataset_v2/features_nbr_s5/`,
   `dataset_v2/packaged/loio_nfold_nbr_s5/` (S=32 only).

### Code changes

- `src/modeling/gbm.py`: `use_scale_pos_weight` knob on
  `LightGBMClassification` + `LightGBMClassificationBalanced` variant
  (registered in `VARIANT_CONSTRUCTORS` / `CLASSIFICATION_VARIANTS`).
- `scripts/sweep_binary.py`: `--variant` flag.
- `scripts/probes/_sweep_w0.py`: generalized (variant x target x scheme x
  scale) LOIO matrix probe with the flat meaningful-threshold convention
  (fa > 1e-2 / bc >= 50 at every scale, matching `_sweep_stage6a.py`).
- `tests/test_modeling_gbm.py`: +2 tests (30 in file); full suite 283.


## 2026-06-10 -- W1 rung 1: coreg y-shift SIGN ERROR found and fixed; all v2 labels regenerated; baseline re-banked

W1 (error-atlas differential diagnosis, PLAN_ModelUsability.md) opened with
the rung-1 label-geometry audit and immediately found a cause-0 bug, exactly
the class of mundane failure the ladder restructure (Brian directive,
2026-06-10) was designed to catch BEFORE blaming the sensor.

### The bug

`src/coregister.py` converted the phase-correlation row shift to metres as
`dy_m = dy_px * px_y`, **omitting the row->world-y sign flip** (rows grow
southward, world y northward: dy_world = -dy_px * px_y since transform.e < 0).
The array-space solve itself was verified correct (its own post-shift Pearson
check, peaks 0.58-0.88). `labeling._apply_coreg_shift` translated the
polygons by the bad `dy_m`, so every v2 label field was pushed SOUTH when it
needed to go NORTH. All 38 images have HiRISE sitting north of the Murray
mosaic (dy_px < 0; 6-285 m, median ~180 m), so post-"correction" every label
field sat **2x|dy| = 12-570 m (median ~360 m = 1.1 tiles at S=64) south** of
its CTX texture. The x-component was applied correctly throughout.

### How it was caught (the W1 rung-1 probes)

1. **+/-2-tile label-shift rescore** (`scripts/probes/_w1_shift_rescore.py`):
   re-scored the banked W0 predictions against label grids shifted di,dj in
   [-2,+2]. Cohort-mean AUC peaked at (di=+1, dj=0) 0.616 vs 0.598 center,
   monotone along di, symmetric in dj -- a global row-direction misalignment
   of ~1 tile.
2. **Direct displacement measurement**
   (`scripts/probes/_w1_label_ctx_displacement.py`): phase-correlated
   smoothed boulder-density rasters against CTX texture energy. Nominal-
   position labels reproduced the cached HiRISE shift exactly
   (ESP_042964_2160: measured (-36.2, +21.0) px vs cached (-35.9, +21.3));
   as-applied labels showed **2x dy and ~0 dx** (measured (-72.2, -0.1)).
   Same pattern on ESP_066634_2210 and ESP_069763_2235.
3. Coreg solve quality (peak, block MAD, confident fraction) was UNcorrelated
   with per-image AUC (`_w1_coreg_vs_auc.md`) -- the solves were fine, the
   application was wrong.

### Fix + migration (Brian approved fix+regenerate+re-bank in session)

- `src/coregister.py`: new `shift_px_to_world_m()` helper does the conversion
  with the sign flip; `single_window` provenance fixed identically.
- `tests/test_coregister.py`: +2 regression tests (unit sign test + synthetic
  end-to-end world-space recovery). Fast suite 265 pass (was 263).
- `scripts/probes/_w1_migrate_coreg_sign.py`: rewrote the 48 cached
  coregistration JSONs (cache, cache_v2, cache_v2_dev) from their correct
  `shift_px` values; marker field `y_sign_fix_applied: 2026-06-10`.
  Re-solving was unnecessary -- only the metre conversion was bad.
- Stage 4 re-run, all 38 v2 images (apply_coreg_shift=True, now-correct
  shifts); Stage 5 repackaged both schemes (loio_nfold, within_image_4fold).
- **Post-fix validation**: displacement probe residual now sub-pixel
  (<=0.8 px = 4 m, was 72-89 px); cohort rescore surface on the re-banked
  predictions peaks at (0,0) (0.624, all neighbours lower)
  (`_w1_surface_postfix.py`).

### Re-banked W0 baseline (supersedes the 2026-06-10 W0 recipe numbers)

Same 6-cell matrix, corrected labels
(`models/_sweep_w0/20260611T013810Z`; canonical artifact dirs overwritten in
place -- config hash unchanged, e.g.
`models/lightgbm_two_stage_balanced/8c7523615964f5cb/scale_S64_target_boulder_count`).
Every cell improved:

| metric (banked recipe: two_stage_balanced x boulder_count @ S=64) | pre-fix | post-fix |
|---|---|---|
| Spearman rho | +0.1431 | **+0.1878** |
| presence AUC | 0.6149 | 0.6149 (coincidence, unchanged to 4 dp) |
| meaningful AUC | 0.5983 | **0.6243** |
| PR-AUC | 0.5431 | **0.5616** |
| precision@top-5% | 0.5679 | **0.5859** |
| per-image meaningful-AUC median | 0.594 | **0.603** |
| images > 0.70 | 23.7% | **34.2%** |
| images < 0.50 (anti-signal) | 28.9% (11) | **21.1% (8)** |

fractional_area cells gained even more (vanilla two_stage: rho 0.169 ->
0.240, meaningful AUC 0.616 -> 0.675) -- geometric noise was diluting the
continuous target hardest.

**W0 verdicts re-checked on corrected labels (`_w0_paired_deltas.py` on the
new sweep): all hold.** P2 promotion stands (PR-AUC +0.146 p=1e-4,
prec@5% +0.147 p=0.001, Spearman -0.041 n.s.); P1 still null (all n.s.);
hurdle still beats single-stage on boulder_count (log1p_huber Spearman
-0.028, p=0.063, all other deltas n.s.-negative). The promoted recipe is
unchanged: **lightgbm_two_stage_balanced x boulder_count @ S=64**.

### Collateral findings for the W1 dossier

- **Per-image meaningful-AUC is statistically meaningless on near-saturated
  images** (ESP_054622_2240: 4 negatives; ESP_069763_2235: 6; ESP_059686_2235:
  8; ESP_054134_2265: 28; ESP_063429_2240: 20; ESP_068483_2280: 22;
  ESP_045550_2180: 27). The dossier and any reliability flag must carry
  n_pos/n_neg validity columns; rescore surfaces on these images swing
  0.25<->0.85 between adjacent offsets.
- **Surviving anti-signal images (8, post-fix)**: ESP_076499_1160 (0.224,
  rho -0.51 -- WORSE after the fix, strongly inverted, geographic outlier at
  ~64 S, "unknown" cohort label), ESP_055978_2270 (0.245), ESP_054000_2255,
  ESP_046328_2180, ESP_064510_2260, ESP_047976_2020, ESP_049242_2115,
  ESP_059686_2235 (8 neg -- validity-suspect). These are the rung-2..5
  targets; their inversions still need a mechanism.
- The historic "anti-signal favours artifact causes" prior is now empirically
  vindicated: 3 of 11 anti-signal images were pure geometry casualties.

### Tier 1 reference classifier re-banked

`lightgbm_classification` on `fa_gt_1e-2` @ S=64, corrected labels
(`models/_sweep_binary/20260611T042543Z`): AUC 0.655 +/- 0.129 (was 0.615),
lift@top-K 1.845 (was 1.430), ECE 0.254 (was 0.264 -- unchanged, consistent
with the W0 P5 verdict that LOIO miscalibration is between-image base-rate
shift, not loss weighting).


## 2026-06-10 -- W1 rungs 2-5 + synthesis: ladder complete, causes attributed, reliability-flag + native-CTX decisions

Continuation of the same session as the rung-1 entry. All on the re-banked
recipe (two_stage_balanced x boulder_count @ S=64, corrected labels).
Deliverables: notebook 18 (`notebooks/18_w1_error_atlas.ipynb`, runs
top-to-bottom), 38-row dossier (`dataset_v2/w1_dossier.parquet` +
`scripts/probes/_w1_dossier.md`), probes `_w1_rung2_join_audit.py`,
`_w1_rung3_detection_stats.py`, `_w1_rung3_fullres_visual.py`,
`_w1_rung4_seam_error.py`, `_w1_rung5_feature_sign.py`,
`_w1_reliability_proxy.py`, `_w1_build_dossier.py`.

### Ladder verdicts

- **Rung 2 (join/pipeline): CLEAN, excluded.** All 38 images: unique keys,
  zero labels<->features join loss, zero NaN features at S=64, exact
  S=32->S=64 nested-count consistency.
- **Rung 3 (BoulderNet content): no false-positive epidemic.** Anti-signal
  vs cohort indistinguishable on count/density/score/size/edge stats (all
  MWU p>0.35). Full-res sampling (10 images, `w1_rung3_*.png`): detections
  look like real boulders everywhere; the visual difference is that
  anti-signal images carry SMALL (2-4 m, near CTX-invisible) boulders in
  uniformly speckled terrain, controls carry large resolved boulders in
  spatially coherent fields. ESP_055978_2270 is the cohort's sparsest +
  lowest-score detection set (72/km^2, median score 0.42).
- **Rung 4 (CTX content): image-level source correlation REPLICATES post-fix
  (mean_n_sources rho=-0.378 p=0.019; dom_frac +0.376; seam_frac -0.364) but
  seam-tile masking does NOTHING** (single-source-only AUC delta median
  +0.000, improved 29% of images; within-image seam fractions only 0-12%).
  The mechanism is regional/between-image, not within-image seam damage.
- **Rung 5 (mechanism): anti-signal splits into two classes.**
  *texture_decorrelated* (ESP_054000_2255, ESP_064510_2260,
  ESP_049242_2115): within-image texture<->label Spearman ~0 or negative
  while the healthy cohort median is strongly positive (+0.29 to +0.46
  across shadow/gradient/contrast/edge) -- at 5 m/px these images' boulder
  populations genuinely do not modulate CTX texture; a cohort-trained model
  must fail there. *distribution_shift* (ESP_076499_1160, ESP_046328_2180):
  within-image signal is STRONG and cohort-consistent (076499:
  shadow_fraction rho +0.73, would alone give a high within-image AUC) yet
  LOIO AUC is 0.22-0.40 -- the model misses real signal because the
  feature distributions sit outside the training cohort (076499 is the
  ~64degS geographic outlier). This class is fixable in principle
  (per-image normalization / photometric invariance).
- **Validity class:** ESP_055978_2270 (34 pos), ESP_047976_2020 (33 pos),
  ESP_059686_2235 (8 neg) -- their per-image AUCs are too fragile to
  diagnose; reported but not attributed.

### Dossier cause counts (38 images)

ok 20 / ok_validity_limited 8 / validity_limited 3 / texture_decorrelated 3
/ distribution_shift 2 / ok_geometry_fixed 2. Terrain pattern (Serrano
mediation): the attributed failures sit in channels/mesas/crater terrain;
the ok class is overwhelmingly plains.

### Decisions

1. **Tier 1 reliability flag: graded region-level confidence, not a binary
   anti-signal detector.** Seam-tile masking rejected (rung 4). Dispersion
   and feature-shift proxies null (rho -0.15/-0.03); only SeamMap source
   stats predict AUC (rho ~0.38) and they do not separate cleanly (class-C
   failures include single-source images). Ship `mean_n_sources` +
   `dominant_source_fraction` as confidence covariates + per-tile validity
   reporting, and state honestly that ~13% of images (5/38) fail without an
   inference-time warning signal under current features.
2. **Native-CTX pivot: NO-GO for now.** The surviving failure classes are
   not what the pivot fixes (sensor floor for small boulders; per-image
   distribution shift). Next bets in order: (a) per-image feature
   standardization (rank/z within window) -- directly targets class C, one
   sweep; (b) W2 CNN with photometric augmentation (same mechanism + feature
   ceiling); (c) terrain covariate (global geologic map join). Revisit
   native-CTX only if (a)+(b) fail.


## 2026-06-10 -- post-W1 bug-hunt round 2 (Brian-requested): four checks run

Probes: `_w1_check4_presence_and_check1_deadfeat.py`,
`_w1_shadow_threshold_diag.py`, `_w1_geometry_audit_all38.py`,
`_w1_latitude_distortion.py`, `_w1_label_autocorr.py`.

1. **Dead shadow features (CONFIRMED -- candidate bug #2).** All four shadow
   features (`shadow_fraction`, `shadow_fraction_strict`,
   `lacunarity_shadow_b2/b4`) are identically zero across the entirety of
   ESP_046328_2180 and ESP_064510_2260 -- exactly two images, both
   anti-signal. Mechanism: their CTX windows contain large areas
   bottom-clipped to DN=1; the clip spike is the modal DN, so the shadow cut
   `max(0, mode-20) = 0` can never fire (`features.py
   _compute_dn_thresholds`). DN=1 also passes the `arr > 0` validity test, so
   clipped tiles count as valid while carrying no texture. No other image is
   affected (next-lowest whole-image shadow fraction is 2.1%). Fix decision
   pending (Brian): treat DN<=1 as nodata for thresholds + validity, or
   percentile fallback when the mode sits at the clip floor; then regenerate
   features for the two images + re-bank.
2. **All-38 model-free geometry audit (rung 1 closed).** Boulder-raster vs
   CTX-texture phase correlation on every image: 23/39 achieve lock
   (peak>=0.15); locked median residual |dy|=0.65 px, |dx|=1.80 px --
   geometry is clean cohort-wide. Six locked images show 5-66 px residuals
   at modest peaks (045550, 048688, 068402, 071699, 076565 + borderline);
   none are anti-signal images, and single-window proxy bias (boulder
   density vs texture energy on anisotropic terrain) is the likely
   explanation; low-priority follow-up. ESP_064510_2260 shows peak -0.59:
   boulder density ANTI-correlates with texture energy there -- independent
   corroboration of its rung-5 texture_decorrelated class.
3. **Latitude / Plate Carree distortion (quantified, real for one image).**
   No cohort-wide trend (cos(lat) vs AUC rho=+0.002) because the cohort
   clusters at 40-46N, but ESP_076499_1160 (63.7S) is an outlier on every
   axis: GLCM E-W ground scale 2.22 m/px vs the cohort's 3.4-4.6, true
   min-size floor 0.94 m vs 1.16-1.36, bc>=50 true-density threshold 2.26x
   the equatorial value. Concrete mechanism for its distribution_shift
   class: its features live at a different effective ground resolution and
   its labels at a different counting regime. Carry as a known systematic;
   per-image standardization (next-bets #1) is the first treatment.
4. **presence_auc: coincidence verified, metric RETIRED (Brian decision).**
   Per-fold presence AUC changed on 23/38 folds between pre-fix and post-fix
   sweeps (mean |delta| 0.091, max 0.61) yet the means collide at 4 dp
   (0.614934 vs 0.614904) -- a genuine coincidence, and proof the re-bank
   consumed the new labels. Independently, the metric is conceptually void
   (">=1 boulder anywhere in a 320 m tile" is unobservable at 5 m/px) and
   undefined on ~1/4 of images (single-class). Removed from sweep result.md
   headline tables (`_sweep_w0.py`); the parquet column remains for
   back-compat. `meaningful_auc` is the discrimination metric.

**Why the sign-fix only bought ~+0.026 AUC (Brian question, answered with
data):** boulder fields are spatially smooth at 320 m -- shifting labels one
full tile leaves them Spearman rho=0.72 (median) correlated with truth and
89.8% in agreement on the bc>50 binary (`_w1_label_autocorr.py`). The
pre-fix 1.1-tile offset therefore acted as ~10-28% label noise, not
scrambling: the model was still learning a mostly-correct mapping. The
pre-fix rescore surface said the same thing in advance: perfectly undoing
the shift at scoring time was worth only +0.018 (0.598 -> 0.616); the
retrain delivered +0.026 (0.624). The numbers are internally consistent --
nothing suspicious left in the gain size.


## 2026-06-10 -- DN-clip shadow fix LANDED: both target images exit anti-signal; baseline re-banked again

Fix per the round-2 finding: `_compute_dn_thresholds` excludes DN<=1
(bottom-clip) pixels from the modal-DN histogram + percentile fallback when
the cut still lands at the clip floor (`src/features.py`, +2 regression
tests; features tests 22 pass). By construction bit-identical for every
image whose mode was not at the floor, so only the two affected images were
regenerated (modes 1->128 and 1->87; shadow features alive, label
correlations +0.34 / -0.24). Stage 5 repackaged; banked-variant cells
re-swept (`models/_sweep_w0/20260611T054855Z`).

**Per-image effect (banked recipe, meaningful AUC):**
- ESP_046328_2180: 0.396 -> **0.645** (+0.249, exits anti-signal) -- its
  distribution_shift classification was substantially "the model was blind
  on the shadow channel"; reattributed `ok_shadowfeat_fixed`.
- ESP_064510_2260: 0.404 -> **0.531** (+0.127, exits anti-signal) --
  reattributed `ok_shadowfeat_fixed`; its shadow-label correlation is
  negative (boulders in the dark smooth unit), consistent with its
  remaining borderline score.

**Cohort effect:** per-image median AUC 0.603 -> **0.657**; pooled metrics
~flat (rho 0.1878 -> 0.1767, PR-AUC 0.5616 -> 0.5633, meaningful AUC mean
0.6243 -> 0.6372). NOTE: every other image's AUC also moved (up to +-0.18)
because the two fixed images sit in all other folds' training sets --
fold-level retraining ripple. Lesson recorded: per-image AUC carries wide
error bars; borderline anti-signal membership (e.g. ESP_055253_2245
0.571 -> 0.392, ESP_071093_2210 0.737 -> 0.561) churns under feature/
training perturbations, so the dossier's binary <0.5 cut should be read
with that variance in mind. Dossier rebuilt on the new sweep
(`dataset_v2/w1_dossier.parquet`): ok 18 / ok_validity_limited 9 /
distribution_shift 3 / texture_decorrelated 3 / validity_limited 2 /
ok_shadowfeat_fixed 2 / ok_geometry_fixed 1.

**Banked baseline numbers are now those of sweep 20260611T054855Z** (same
recipe identity: two_stage_balanced x boulder_count @ S=64).


## 2026-06-11 -- Bet 1 (per-image feature standardization): NOT PROMOTED; class-specific effect confirmed and documented

W1 next-bet 1 (PLAN_ModelUsability.md; terrain covariate HELD OFF per Brian
2026-06-11 -- image features only). Implementation:
`loaders.standardize_fold_per_image` / `augment_fold_with_per_image`
(+6 unit tests, fast suite 273); sweep probe
`scripts/probes/_sweep_perimage_std.py`; verdict probe `_w1_pistd_verdict.py`.
Cell: two_stage_balanced x boulder_count @ S=64, full-v2 LOIO, vs banked
baseline `models/_sweep_w0/20260611T054855Z`. Promotion criteria declared in
advance: paired Wilcoxon median delta(meaningful_auc) > 0 at p < 0.05 AND
pooled PR-AUC delta >= -0.01.

| method (sweep dir) | rho | mAUC mean | PR-AUC | median dAUC (p) | verdict |
|---|---|---|---|---|---|
| raw baseline | +0.1767 | 0.6372 | 0.5633 | -- | banked |
| rank (164339Z) | +0.2034 | 0.6158 | 0.5650 | -0.013 (0.42) | FAIL |
| zscore (164630Z) | +0.1873 | 0.6079 | 0.5600 | -0.012 (0.48) | FAIL |
| robust (164630Z) | +0.0999 | 0.5725 | 0.5343 | -0.071 (0.04, WORSE) | FAIL |
| aug_zscore raw+std concat (165804Z) | +0.1785 | 0.6304 | 0.5565 | -0.006 (0.55) | FAIL |

**Mechanistic check (the real finding):** zscore rescues ALL THREE
distribution_shift dossier images out of anti-signal --
ESP_076499_1160 0.302 -> 0.629, ESP_055253_2245 0.392 -> 0.741,
ESP_054397_2105 0.489 -> 0.672 -- exactly as the W1 attribution predicted.
The cost lands on images where absolute feature values carry signal, netting
~zero-to-negative cohort-wide. The raw+std concatenation does NOT let the
GBM have both: the rescue dilutes (2 of 3 stay below 0.5) without cohort
gain. Interpretation: with n=38 a single global recipe cannot serve both
feature-regimes; per-image standardization is a *class-specific treatment*,
not a recipe upgrade.

**Disposition:** banked recipe unchanged (raw features). Standardization
artifacts kept under `models/_sweep_perimage_std/`. Carried forward: (a) the
zscore result is direct evidence that the distribution_shift failure class
is real and treatable -> strengthens the case for W2 CNN photometric
augmentation (learn the invariance instead of imposing it globally);
(b) if a reliability-stratified deployment ever ships per-stratum models,
zscore-standardized features are the candidate for the shifted stratum.


## 2026-06-11 -- W2 setup S1-S3 complete (PLAN_CNN.md): CUDA torch, v2 context patches, GPU smoke, baselines refreshed

### S1 -- CUDA torch

- `torch 2.12.0+cu130` installed into `geospatial` from
  `download.pytorch.org/whl/cu130` (replaces `2.12.0+cpu`; same version,
  CUDA 13.0 build). GPU: **NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB**
  (Blackwell; driver 592.00 / CUDA 13.1). `torch.cuda.is_available()` True;
  matmul verified on-device.
- `CNNParams.device` now defaults via `torch.cuda.is_available()`
  (cuda when present, cpu otherwise) instead of hardcoded `"cpu"`.

### S2 -- v2 context patches generated (the v1 "CNN dead-end" config flag retired)

- `config_v2.yaml` `features.context_patch.enabled: false -> true` (the OFF
  comment cited the v1 dead-end verdict, which predates the coreg sign fix +
  DN-clip shadow fix and no longer binds).
- Stage 4b `--all`: 38/38 images in 747 s; `dataset_v2/context_patches/`
  now holds 76 stacks (38 x {S32, S64}), 3,564,767 patches per size,
  **17.0 GB on disk**. Patches are raw uint8 DN; DN<=1 clip pixels pass
  through (0.25% of pixels on ESP_046328_2180, spot-checked).
- Stage 5 `--all` repackaged `loio_nfold` + `within_image_4fold`
  (X_cols 53 -> 55: `patch_idx_S32`/`patch_idx_S64` join columns).
  NOTE: the stage-6 side schemes `loio_nfold_ctx_illum` / `loio_nfold_nbr_s5`
  were NOT refreshed (built by their own repackage scripts; not in the recipe).

### Determinism check -- banked baseline remains valid (Brian-approved)

Re-running Stage 4b regenerates feature parquets in place, which could have
invalidated paired comparisons against the banked sweep. Re-ran the banked
cells (`_sweep_w0.py`, two_stage_balanced x {boulder_count, fractional_area}
@ S=64 -> `models/_sweep_w0/20260611T215447Z`): **per-fold deltas exactly
0.0 on all 38 folds x 4 metrics** vs `20260611T054855Z`. Feature
regeneration is bit-identical; `20260611T054855Z` stays the banked baseline.

### S3 -- smoke + Tier 1 refresh

- GPU smoke (`scripts/probes/_smoke_cnn_v2_one_fold.py`): fold 0, S=64
  classification head, 34,388 train patches, 0 margin rows, **4.6 s/epoch**,
  finite loss/probs. Projects ~1-3 min/fold at full epochs -> the 4-cell x
  38-fold grid fits the planned 3-8 h budget.
- **Tier 1 reference classifier refreshed on post-shadow-fix features**
  (`models/_sweep_binary/20260611T214042Z`): AUC **0.6557 +/- 0.137**,
  lift 1.716, ECE 0.266 (was 0.6548 / 1.845 / 0.254 pre-fix at
  `20260611T042543Z`, now superseded). Mean AUC essentially unchanged, as
  expected (fix touched 2 images).

### Phase 1 code (ready, grid not yet launched)

- `src/modeling/cnn.py`: `AUG_CELLS` = none / geometric / photometric /
  photometric_std (A-D per PLAN_CNN.md §4.2); `_PatchDataset` stages split
  (geometric flips/rots; photometric = brightness/contrast/**gamma
  [0.8, 1.25] log-uniform, new**/noise); cell D per-patch `(x-mean)/std`
  with a **1-DN std floor** (DN<=1 clip-patch guard) applied at train AND
  inference; `aug_cell` on `CNNParams` (default `photometric` = v1 behavior).
- `scripts/sweep_cnn.py`: cells x 38 folds, **group-aware inner val
  (4 whole images, deterministic rotation)**, artifacts
  `models/_sweep_cnn/{timestamp}/` (summary/aggregate parquet, incremental
  write per cell) + per-cell `models/cnn_bce_S{P}/{hash}/..._aug_{cell}/`
  with predictions/metrics/snapshot/per-fold state_dicts. Aggregate carries
  the gate metrics: median per-image AUC, pooled PR-AUC
  (= `average_precision_score` over concatenated held-out tiles, matching
  `pooled_global_pr_auc`), pooled precision@top-5%.
- `scripts/train_cnn.py`: `--aug-cell` pass-through for single runs.
- Tests: +5 (cell validation, cell-A identity, cell-B pixel-multiset
  invariance, cell-D eval-time application + clip-patch finiteness, all-cells
  classifier smoke). **Fast suite 278 pass.**


## 2026-06-11 -- W2 Phase 1 grid READ: H-B (photometric augmentation) REFUTED cohort-level; no-aug CNN beats Tier 1 per-image (single seed); pooled deficit diagnosed as cross-image miscalibration

Grid: `models/_sweep_cnn/20260611T220815Z` (4 cells x 38 LOIO folds, binary
fa_gt_1e-2 @ S=64, 64-px patches, seed 0, 4-image group-aware inner val,
GPU ~25 s/fold for A/B, 1-3 min/fold for C/D -- photometric augmentation is
CPU-bound in the loader; one 31-min throughput stall on C fold 29, training
itself clean). Probes: `_w2_cnn_verdict.py` (gates), `_w2_midgrid_diag.py`
(structure), `_w2_azimuth_spread.py` (illumination). Baselines: Tier 1
refresh (AUC 0.6557, pooled PR-AUC **0.5651**, `_sweep_binary/20260611T214042Z`),
banked GBM (`_sweep_w0/20260611T054855Z`).

### Aggregate (single seed -- the fold-ripple/seed caveat applies everywhere)

| cell | per-image AUC mean | median | pooled PR-AUC | pooled prec@5% |
|---|---|---|---|---|
| A none | **0.6889** | **0.6938** | 0.5095 | **0.5786** |
| B geometric | 0.6423 | 0.6868 | 0.4522 | 0.4365 |
| C geo+photometric | 0.6647 | 0.6661 | 0.5070 | 0.5646 |
| D C+per-patch-std | 0.6061 | 0.6486 | **0.5325** | 0.4885 |

### Verdicts

1. **H-B REFUTED at cohort level.** Every augmented cell is <= cell A on
   per-image AUC (paired vs A: B -0.047 n.s., C -0.024 n.s., D -0.083
   p=0.003) and no augmented cell passes either gate vs Tier 1. Mechanism
   check: the rescue exists ONLY for ESP_076499_1160 (the 64S geographic +
   illumination outlier, subsolar az 228.6 vs cohort 142-186): +0.141/+0.211/
   +0.128 across B/C/D -- consistent, image-specific. ESP_055253_2245 is
   consistently DAMAGED by augmentation (-0.36/-0.29/-0.25); ESP_054397_2105
   mixed. Photometric invariance is not a cohort recipe on n=38.
2. **Geometric augmentation is actively harmful and the mechanism is
   physical**: cohort CTX-source subsolar azimuth is 142-186 deg for 36/38
   images (SeamMap data, `_w2_azimuth_spread.py`) -- the shadow direction is
   a *stable, learnable prior* and flips/rots destroy it. Cell B: pooled
   PR-AUC 0.452, degraded 19/37 images (worst -0.58). Contrast with Bickel
   et al. 2021 (rotation-augmented rockfall *object* detection): rotation
   invariance suits resolved-shape cues, not illumination-locked
   sub-resolution texture. Outliers recorded: ESP_076499_1160 (az 228.6),
   ESP_068483_2280 (az 1.7, incidence 4.3 = near-shadowless; also a top
   cell-A loss vs Tier 1).
3. **H-A partially supported (single-seed, gate-reading caveat).** Cell A vs
   Tier 1 per-image AUC on validity-passing images (n=27): median paired
   delta **+0.066, win 67%, p=0.016**. GATE AMBIGUITY FLAGGED: the
   pre-declared text "median per-image mAUC +0.05" passes under
   median-of-paired-deltas (+0.066) but FAILS under difference-of-medians
   (0.671 -> 0.675 = +0.004; the cohort median barely moves while most
   per-image deltas are positive because wins/losses land on different
   images). **Brian ruling (2026-06-11): median-of-paired-deltas is the
   binding reading** -- cell A passes the gate vs Tier 1, pending only the
   3-seed replication required by PLAN_CNN.md §4.2.
4. **Pooled PR-AUC: all cells fail** (best = cell D 0.5325, -0.033 vs
   Tier 1). Diagnosis (`_w2_midgrid_diag.py`): the CNN's per-image mean
   score tracks the image's true base rate at rank-corr +0.22 (cell A) vs
   Tier 1's +0.41 -- good within-image ranking, mis-leveled across images,
   so the pooled ranking interleaves images badly. Cell D demonstrates the
   trade *within* the CNN family: per-patch standardization gives the best
   pooled PR-AUC and the worst per-image AUC.
5. **texture_decorrelated reattribution candidate**: under cell A the trio
   scores 0.622/0.738/0.594 (ESP_045983_2270 / ESP_049242_2115 /
   ESP_054000_2255) vs GBM 0.449/0.461/0.408. If this survives seeds, the W1
   "no signal at 5 m/px" attribution was really "no signal in the
   handcrafted feature set" for these images. CNN-vs-baseline per-image AUC
   correlation is only rho 0.42-0.49 -- the CNN is a complementary model,
   not a re-ranked GBM.

### Literature review (Brian-requested, docs/w2_litreview.md)

Bickel et al. 2021 multi-domain (PDF read in full: diversity > scale-mixing;
10%-home-labels economics), canopy-height cross-resolution supervision
(Lang et al. -- the structural twin; probabilistic ensemble heads), sub-GSD
density estimation (Rodriguez & Wegner), test-time adaptation (AdaBN /
prediction-time BN / TENT -- our 1k-patch-per-image deployment is the
best case), FDA/RHM cross-image radiometric augmentation, and Mars
foundation models (MOMO HiRISE+CTX+THEMIS weights-public; Fang et al. 2026
CTX-specific ViT on ~4M images; Mars-Bench 20-task benchmark incl. boulder
tasks). Ranked follow-up queue in the doc; Phase 2 SSL re-specced to
"fine-tune public CTX-pretrained backbones first".

### Disposition (pending Brian)

Phase 1 as declared does NOT promote an augmented CNN; W2 does not close as
"sensor-bound" either, because the no-aug CNN's per-image edge + the
diagnosed calibration mechanism point to specific cheap follow-ups:
(a) 3-seed cell A replication (required insurance), (b) AdaBN post-hoc on
the saved cell-A state_dicts (free, the bet-1-zscore analog), (c) "CNN
ranks / GBM scales" score fusion (free arithmetic), (d) photometric-only
cell (de-confounds C, targets the 076499-style outliers), (e) S=32
replication per §4.2b (winner = cell A, so cell A only).

## 2026-06-11 -- W2 3-seed verdict: single-seed gate pass does NOT replicate; 3-seed ensemble + Tier-1 fusion passes both gates (held-out S=32 confirmation pending)

**Seed replication (cell A, S=64, seeds 0/1/2; `sweep_cnn.py` runs
20260611T220815Z / 20260612T014231Z / 20260612T042859Z):**

| seed | per-img AUC median | dAUC median (validity, vs Tier 1) | Wilcoxon p | pooled PR-AUC |
|---|---|---|---|---|
| 0 | 0.6938 | +0.0661 | 0.0162 | 0.5095 |
| 1 | 0.6913 | +0.0383 | 0.0585 | 0.5595 |
| 2 | 0.7088 | +0.0054 | 0.6617 | 0.4913 |

The per-image *skill* is seed-stable (median 0.69-0.71 every seed); the
*score calibration* is not (pooled PR-AUC swings 0.49-0.56; paired deltas
vs Tier 1 shrink to nothing on seed 2). **Under the pre-declared gate +
Brian's median-of-paired-deltas ruling, the single-seed cell-A pass is NOT
confirmed.** Per-seed fusion tracks the same instability (beats Tier 1
pooled on seeds 0/1, misses on 2).

**3-seed ensemble (`scripts/probes/_w2_seed_ensemble.py`):** mean of the 3
seeds' probabilities, then the fusion arithmetic on the ensembled score.
n=37,315 pooled tiles, validity-passing n=27 images, Tier 1 refs 0.5651
pooled / 0.6806 median AUC:

| variant | pooled PR-AUC | prec@5% | med per-img AUC | dAUC med (v) | win | p |
|---|---|---|---|---|---|---|
| ens_mean | 0.5327 | 0.675 | 0.7109 | +0.0521 | 0.70 | 0.0065 |
| F1(ens) = within-img quantile x t1 image mean | **0.5955** | **0.887** | 0.7109 | +0.0521 | 0.70 | 0.0065 |
| F3(ens) = pooled-rank average | 0.5856 | 0.812 | **0.7137** | **+0.0578** | **0.85** | **0.0001** |

F1(ens) passes BOTH pre-declared gates (pooled +0.0304 >= +0.03; per-image
+0.052 >= +0.05 at p=0.0065). F3(ens) passes the per-image gate with the
strongest stats on every per-image measure. **Candidate W2 recipe: 3x
SmallCNN seed ensemble for within-image ranking x Tier-1 LightGBM for
image-level scale.**

**Honesty caveat / ruling:** the ensemble+fusion combination was assembled
after seeing the per-seed S=64 results (seed-ensembling and the fusion were
each independently pre-motivated, but their conjunction as "the recipe" is
post-hoc). Therefore the S=32 replication (PLAN_CNN.md 4.2b) is promoted to
**held-out confirmation**: 3 seeds of cell A at S=32, read ONLY via the
ensemble+fusion recipe against an S=32 Tier-1 baseline (to be trained
first), gates unchanged. F1-vs-F3 choice declared before that read:
**F1 if pooled PR-AUC is binding, F3 if per-image AUC is** (per the W1
graded-reliability framing, per-image is the deployment-relevant one).

**Rodriguez & Wegner 2018 read in full** (PDF from Brian; litreview section 3
updated): their GT pipeline (high-res detector -> Gaussian sigma=K/pi ->
K-pool) is exactly the BoulderNet->tile construction; headline finding =
stride-1 no-downsampling shallow ResNet beats DeepLab v2/v3 for sub-pixel
objects ("any down-sampling inside the network risks losing precious
details") -> new queue item: stride-1/no-pool SmallCNN capacity variant;
their high-density-pocket underestimation is the literature echo of our W0
compression finding; single-band texture-only regime shown workable (cars).

## 2026-06-11 -- W2 cell E (photometric_only) read: geometric aug confirmed as the harmful ingredient; shift class responds to photometric aug 3/3; S=32 confirmation launched

Cell E = photometric jitter only, no flips/rotations (added to AUG_CELLS
post-grid to de-confound cell C). Seed 0, S=64; sweep
`models/_sweep_cnn/20260612T045007Z`; read via
`scripts/probes/_w2_photonly_read.py`.

1. **E vs A (no-aug): cohort-equal** (paired dAUC median +0.024, p=0.64;
   median AUC 0.694 -> 0.698, pooled PR-AUC 0.4919 ~ cell A's seed band).
   Cells B/C/D all lost to A; removing the geometric half removes the harm.
   **The Phase-1 "augmentation hurts" verdict is now attributable to the
   geometric component specifically** -- consistent with the azimuth-prior
   physics (cohort sun azimuth 142-186 deg).
2. **Mechanism (H-B, narrowed): all 3 distribution_shift images improve
   under E vs A**: ESP_054397_2105 +0.055, ESP_055253_2245 +0.172 (0.583 ->
   0.755), ESP_076499_1160 +0.058. Sign-consistent with the pre-declared
   H-B class; too weak to move the cohort gate (p=0.09 vs Tier 1).
   Single-seed caveat applies.
3. **Subtype split within the shift class:** ESP_076499_1160 (the 228.6-deg
   azimuth outlier) does BETTER under full cell C (0.636) than under E
   (0.484) -- rotation is what exposes the net to its anomalous sun
   direction, while the other two shift images are radiometric cases that
   rotation hurts. Matched treatments exist for both subtypes but each
   costs the other; **azimuth-canonical orientation (litreview queue item
   4) is the principled reconciliation** and gains priority.
4. Cell E does not displace cell A as the recipe ingredient (cohort-equal,
   worse pooled prec@5%). The candidate recipe is unchanged; E's value is
   causal attribution + queue re-ranking.

**S=32 held-out confirmation launched** (same session): Tier-1 LightGBM
fa_gt_1e-2 @ scale_idx=2 baseline (CPU) + 3-seed cell-A chain at 32 px
(GPU). Pre-declared read per the 2026-06-11 3-seed entry: ensemble passes
the per-image gate vs S=32 Tier 1 AND fusion recovers pooled PR-AUC >=
baseline; F1 if pooled is binding, F3 if per-image is.

## 2026-06-12 -- W2 S=32 held-out read: recipe formally NOT confirmed (gate missed by 0.0002); per-image effect REPLICATES; fusion direction INVERTS at S=32

3-seed cell-A chain at S=32 (32-px patches, scale_idx=2, ~161k pooled
tiles) vs the S=32 Tier-1 LightGBM baseline (AUC 0.660,
`models/_sweep_binary/20260612T062412Z`). Read by the pre-declared rule
ONLY (`scripts/probes/_w2_s32_confirm.py`, gates committed before the runs
finished):

| variant | pooled_pr | prec@5% | med_auc | dAUC_med(v) | win | p |
|---|---|---|---|---|---|---|
| p0 / p1 / p2 | 0.5318 / 0.5535 / 0.5035 | | 0.70 / 0.67 / 0.67 | +0.030 / +0.047 / +0.045 | | 0.055 / 0.019 / 0.129 |
| ens_mean | 0.5454 | 0.710 | 0.6948 | **+0.0498** | 0.81 | **0.0009** |
| F1(ens) | 0.5160 | 0.685 | 0.6948 | +0.0498 | 0.81 | 0.0009 |
| F3(ens) | 0.5250 | 0.533 | 0.6697 | +0.0533 | 0.89 | 0.0000 |
| Tier-1 (ref) | 0.4840 | 0.607 | 0.6631 | -- | -- | -- |

**Formal verdict:** gate (a) median paired dAUC >= +0.05 -> **FAIL at
+0.0498** (p=0.0009 passes easily; the magnitude missed by 0.0002);
gate (b) fusion pooled >= Tier-1 -> PASS. Compound rule -> **recipe NOT
confirmed at S=32**. Recorded as declared; no post-hoc re-reading.

**What actually replicated and what didn't:**

1. **The per-image core claim replicated almost exactly**: ensemble
   beats Tier-1 per-image at S=32 with d median +0.0498 (S=64: +0.052),
   win 0.81, p=0.0009. Direction + significance identical; magnitude at
   the gate edge. The honest summary is "replicates in direction and
   significance; landed a rounding error under the pre-declared magnitude
   bar."
2. **The fusion HALF of the recipe does NOT transfer -- it inverts.** At
   S=32 the raw CNN ensemble is ALSO the better pooled model (0.5454 vs
   Tier-1 0.4840, +0.061), and fusing drags pooled DOWN (F1 0.5160,
   F3 0.5250 < ens 0.5454): F1 replaces the CNN's leveling with Tier-1's,
   which is the WEAKER leveler at this scale. "CNN ranks / Tier-1 scales"
   is therefore a conditional recipe: use the better leveler per scale,
   not Tier-1 unconditionally. At S=64 that is Tier-1; at S=32 it is the
   CNN itself.
3. **Scale comparison:** handcrafted features degrade badly at S=32
   (Tier-1 pooled 0.5651 -> 0.4840) while the CNN ensemble holds
   (0.5327 -> 0.5454). The CNN's relative advantage GROWS at the finer
   scale, but S=64 absolute numbers remain higher across the board ->
   **S=64 stays the operating scale**; S=32 is where the feature-set
   ceiling shows first (consistent with the texture_decorrelated
   reattribution hypothesis).

**Disposition:** the S=64 fusion recipe keeps its "passed both gates at
S=64, post-hoc assembly, held-out magnitude miss at S=32 by 0.0002"
status -- promotable only with a fresh pre-declared confirmation (options:
new images per cohort_expansion_candidates.csv, or a pre-declared
fresh-seed S=64 re-run). Phase 2 5.0 productization should encode the
conditional-leveler form. Notebook 19 6 re-executed with the verdict;
figures refreshed.

## 2026-06-12 -- W2 Phase 2 lead bet: Fang-ViT frozen-embedding probe PASSES both gates by the largest margin of the program (pooled PR-AUC 0.5651 -> 0.7637)

PLAN_CNN.md 5.1, first contact with the Fang et al. 2026 CTX foundation
model ([doi:10.1029/2025JH000827](https://doi.org/10.1029/2025JH000827)),
ViT-B/16 MAE+DINO pretrained on 3.9M Murray-mosaic crops -- the identical
product our pipeline windows.

**Setup (all probe-tier, no env mutation):**
- Weights: `models/pretrained/mars-mae-dino-vit-base-v1.pth` (341.7 MB,
  [Zenodo 18180801](https://doi.org/10.5281/zenodo.18180801), CC-BY-4.0).
  Checkpoint metadata self-identifies as timm `vit_base_patch16_224`,
  `in_chans=1`, 224 px, standard timm key layout (150 tensors, no head).
- **No timm/torchvision installed**: the encoder forward is hand-rolled in
  plain torch (`scripts/probes/_w2_fang_embed.py`), strict state-dict load.
- Inputs per S=64 tile (37,315 tiles, 38 images): the tile's own cached
  64-px patch, and a 192-px (3x3-tile) box sliced directly from the cached
  Stage 2 CTX window (NOT stitched from neighbor patches -- only 71% of
  tiles have all 8 neighbors emitted; the window buffer gives **100%
  192-px coverage on every image**). Both bicubic-resized to 224,
  normalized (x/255-0.5)/0.5 per the model card.
- Alignment verified two ways: bit-exact assert that the center 64x64 of
  every sampled 192-px slice equals the tile's cached S64 patch (all 38
  images), plus visual figures
  `reports/figures/19_w2_fang_patch_alignment_{ESP_042964_2160,ESP_076499_1160}.png`.
- Pooled embeddings banked per tile: cls / mean / **GeM(p=3)** (probe uses
  GeM per plan) at both input scales -> `dataset_v2/fang_embeddings/`
  (~680 MB). Extraction: 178 s on the 5070.
- Probe (`scripts/probes/_w2_fang_probe.py`): standard LOIO harness
  (loio_nfold, scale_idx 3, fa_gt_1e-2, `lightgbm_classification` with the
  Tier-1 refresh hyperparameters), embeddings appended as feature columns;
  join on (obs_id, ti, tj) validated one-to-one.

**Verdict table (gates: pooled dPR-AUC >= +0.03; per-image dAUC median
>= +0.05 with Wilcoxon p < 0.05 on validity-passing images):**

| variant | pooled_pr | prec@5% | med_auc | dAUC_med(v) | win | p | gates |
|---|---|---|---|---|---|---|---|
| t1_gem64 (52+768) | 0.7531 | 0.941 | 0.7173 | +0.0661 | 0.78 | ~0 | **both PASS** |
| **t1_gem192 (52+768)** | **0.7637** | **0.977** | **0.7700** | +0.0746 | **0.89** | ~0 | **both PASS** |
| emb_only (1536) | 0.7424 | 0.876 | 0.7519 | **+0.0831** | 0.78 | 0.0009 | **both PASS** |
| Tier-1 (ref) | 0.5651 | 0.771 | 0.6806 | -- | -- | -- | -- |
| F1(ens) W2 best, ref | 0.5955 | 0.887 | 0.711 | +0.052 | 0.81 | 0.0065 | (both) |

Artifacts: `models/fang_probe/{t1_gem64,t1_gem192,emb_only}/{hash}/`
(predictions.parquet, metrics.json, verdict.json). Runtime: 5 min per
t1_gem variant, 85 min for emb_only (1536 cols).

**Findings:**

1. **The FM representation carries large signal beyond handcrafted +
   SmallCNN.** +0.199 pooled PR-AUC over Tier-1 (vs +0.030 for the W2
   ensemble-fusion winner); prec@5% 0.977 means the top-5% map tiles are
   essentially all true positives. 192-px input > 64-px input on every
   metric, as the lit review predicted (closer to pretraining scale).
2. **The W1 failure classes are differentially rescued** (t1_gem192 dAUC
   by cause): distribution_shift **+0.228**, ok_geometry_fixed +0.302,
   ok_shadowfeat_fixed +0.196, texture_decorrelated **+0.176**, ok +0.054;
   only validity_limited negative (-0.034). The per-image shift problem
   that killed every W0-W2 dev win at LOIO is exactly where the FM helps.
3. **emb_only nearly matches the fused variants** (0.7424 pooled, best
   per-image dAUC median +0.0831, texture_decorrelated +0.213) -> the
   queue-item-6 reattribution check is effectively answered: the
   texture_decorrelated floor was a **feature-set floor, not a sensor
   floor**. CTX pixels at 5 m/px contain the signal; our 52 handcrafted
   features (and the 30k-param SmallCNN) could not extract it.
4. MAE's "subdued sub-pixel roughness" caveat did not bind at this task
   ceiling; the illumination-geometry caveat remains untested (no
   azimuth-conditioned read yet; ESP_076499_1160 dAUC +0.087 under
   t1_gem192, so the azimuth outlier is not failing).

**Caveats (recorded with the claim):**
- The FM pretrained self-supervised on the full Murray mosaic, which
  includes our test images' terrain. Label-leak-free and standard FM
  practice, but held-out *pixels* are not unseen by the backbone; the
  clean confirmation is the cohort-expansion images (also Murray-covered,
  so the caveat applies cohort-wide and permanently -- note it in any
  writeup rather than trying to remove it).
- Recipe assembled post-hoc like the fusion recipe; same standing rule
  applies -- promotion needs a fresh pre-declared confirmation on new
  cohort images. Unlike the CNN, there is no seed instability: extraction
  is deterministic and LightGBM is config-deterministic.
- Single (pool=GeM, LightGBM-config) cell; cls/mean poolings banked but
  unread.

**Disposition:** Fang-ViT embeddings are the new Tier-1-candidate feature
set at S=64. Queue reshuffle proposed: (a) S=32 embedding read (does the
FM also fix the S=32 Tier-1 collapse?), (b) t1_gem64_gem192 combined cell
+ pool ablation (cheap, cached), (c) fold the embedding columns into the
conditional-leveler productization (5.0) and the cohort-expansion
confirmation as the promotion vehicle. Fine-tuning stays gated on the
probe (now clearly warranted by signal, but inference cost/calibration
first).

## 2026-06-12 -- Fang-ViT follow-ups: S=32 collapse FIXED (scale-robust), GeM pooling confirmed, combined cell = best per-image profile, illumination caveat present-but-harmless

Brian-approved follow-up set (all three picks), run as one chain plus the
azimuth read (~40 min chain + 1-min read). S=32 embeddings extracted first
(`_w2_fang_embed.py --tile-px 32`: 161,005 tiles, P32+P96 inputs, 100%
context coverage again, 834 s GPU; `dataset_v2/fang_embeddings/` now
3.5 GB total).

**1. S=32 read** (vs the S=32 Tier-1 bank: pooled 0.4840 / prec@5% 0.607 /
med AUC 0.6631; 38 real folds -- ESP_047976_2020 has positives at S=32):

| variant | pooled_pr | prec@5% | med_auc | dAUC_med(v) | win | p | gates |
|---|---|---|---|---|---|---|---|
| t1_gem32 (52+768) | 0.6627 | 0.930 | 0.6844 | +0.0481 | 0.70 | 0.0112 | pooled PASS; per-image **FAIL by 0.0019** |
| **t1_gem96 (52+768)** | **0.7639** | **0.966** | **0.729** | **+0.0818** | 0.74 | 0.0025 | **both PASS** |

The S=32 Tier-1 collapse (0.5651 -> 0.4840) is **fixed**: t1_gem96 pooled
0.7639 is statistically identical to the S=64 t1_gem192 number (0.7637).
The FM result is scale-robust, and the 3x3-context input is the carrier at
both scales (96 >> 32 mirrors 192 > 64). Own-tile-only at S=32 again
misses the per-image magnitude bar by a rounding error -- the recurring
"own-scale input is marginal, context input is decisive" pattern.

**2. Pool ablation** (S=64, ctx input, vs GeM 0.7637): mean 0.7071 /
cls 0.6961 pooled (med AUC 0.7331 / 0.7583; both still pass both gates).
**GeM(p=3) confirmed** -- worth +0.06 pooled over either alternative.

**3. Combined cell t1_gem64_gem192** (52+1536): pooled 0.7549 -- slightly
BELOW t1_gem192 alone (0.7637), so the two scales do not add pooled
signal. But it has the **best per-image profile of the program**: med AUC
0.7777, dAUC median +0.0918, win 0.93, p~0; distribution_shift +0.306.
Same pooled-vs-per-image tension as the W2 F1/F3 fusion pair, same
declared resolution: **t1_gem192 if pooled is binding, t1_gem64_gem192 if
per-image is.**

**4. Azimuth-conditioned read** (`_w2_fang_azimuth.py`, Fang et al.'s
CBIR caveat -- shadow-dominated embeddings match by illumination):
- Benefit is geometry-agnostic: per-image dAUC vs incidence rho=-0.058
  (p=0.73), vs circular azimuth-distance rho=+0.16 (p=0.34).
- **ESP_076499_1160 (azimuth outlier, 228.6 deg) is the cohort's biggest
  winner: dAUC +0.458** -- the image that rotation aug, AdaBN, and zscore
  each only partially rescued across W1-W2 is simply solved by the FM.
  ESP_068483_2280 (near-shadowless, incidence 4.3 deg, the CNN-specific
  failure): dAUC +0.026, absolute AUC 0.899.
- The caveat is *present*: LOO ridge on image-mean embeddings recovers
  sin(azimuth) at held-out r=+0.588 (p=1e-4); incidence and cos(az) not
  recoverable. Illumination direction IS in the embeddings; the LightGBM
  head is not harmed by it.
- Figure: `reports/figures/19_w2_fang_azimuth_read.png`; JSON beside the
  t1_gem192 verdict.

Artifacts: `models/fang_probe/{t1_gem32,t1_gem96,t1_cls192,t1_mean192,
t1_gem64_gem192}/{hash}/`. Probe script now parametrized
(`--tile-px {64,32}`, `--pool {gem,mean,cls}`).

**Not run (deliberate):** emb_only at S=32 (161k x 1536 cols, ~6 h CPU --
optional overnight; the S=64 emb_only already answered the feature-floor
question). MOMO disjoint-corpus cross-check remains the optional bound on
the transductive-pretraining caveat.

**Disposition: the probe phase is closed.** The Fang-ViT frozen-embedding
recipe (Tier-1 features + GeM context-input embeddings -> LightGBM) is the
candidate Tier-1 replacement at BOTH scales, pending the standing
pre-declared confirmation on cohort-expansion images. Next-session queue:
productize extraction out of probe-tier into src/ (inference must embed
arbitrary CTX windows), pre-declare the confirmation gates, then the
W3-style calibration/Tier-2 work on top of the new feature set.

## 2026-06-12 -- Head bake-off (PLAN_FM 2.1a/1c/1f): trees are the wrong reader; MLP 3-seed ensemble decisive; handcrafted features still add ~+0.02

Context: PLAN_FM.md created today (Brian-approved) -- the binding
constraint moved from representation to the head/labels/task; queue 2.1
is the freeze-window evidence block. Scripts:
`scripts/probes/_w2_fang_heads.py` (+ `_w2_fang_head_pairs.py`).

**1a. Head classes on the IDENTICAL gem192-only matrix** (768 cols, S=64,
no handcrafted features; identical LOIO harness; fixed hyperparameters --
head-CLASS read, not tuning):

| head | pooled_pr | prec@5% | med_auc | dAUC_med(v) | win | runtime |
|---|---|---|---|---|---|---|
| LightGBM | 0.7146 | 0.856 | 0.7571 | +0.0581 | 0.59 | 878 s |
| logreg (C=1, std.) | 0.7385 | 0.890 | 0.7678 | +0.1006 | 0.78 | 130 s |
| kNN (cosine, k=50) | 0.7709 | 0.913 | 0.7641 | +0.0926 | 0.85 | 80 s |
| mlp seeds 0/1/2 | .787/.745/.765 | .96/.83/.94 | .79/.82/.78 | +0.12..0.14 | 0.89 | ~130 s |
| **mlp_ens3** | **0.7852** | 0.936 | **0.8035** | **+0.1374** | 0.85 | -- |

Every non-tree head beats LightGBM on embeddings (Brian's hypothesis,
confirmed). MLP pooled calibration is seed-wobbly (0.745-0.787; per-image
skill stable) -- the 3-seed ensemble is the promotable form (the SmallCNN
lesson recurs).

**1c. Head-vs-head paired per-image stats** (validity-passing n=27,
`models/fang_probe/head_pairs.json`): mlp_ens3 beats EVERY other head with
clean significance -- vs lgbm +0.0595 (p~0), vs logreg +0.0292 (p=0.0006,
win 0.85), vs knn50 +0.0499 (p=0.0032); lgbm/logreg/knn50 statistically
tied among themselves. Winner is NOT ambiguous: **MLP 3-seed ensemble.**

**1f. Handcrafted-feature elimination check** (winner heads on t1+gem192,
median-imputed T1 columns):

| head | matrix | pooled_pr | prec@5% | med_auc | dAUC_med(v) | win |
|---|---|---|---|---|---|---|
| logreg | t1+gem192 | 0.7639 | 0.897 | 0.7930 | +0.1457 | 0.81 |
| **mlp_ens3** | **t1+gem192** | **0.8040** | 0.916 | **0.8284** | +0.1465 | **0.96** |
| (mlp_ens3 | gem192-only | 0.7852 | 0.936 | 0.8035 | +0.1374 | 0.85) |

The 52 handcrafted columns still add ~+0.019 pooled / +0.025 med AUC under
the MLP (seed spread also narrows: 0.788-0.816; seed-2 win rate 1.00).
**Elimination is NOT free**: dropping them costs ~2 points but removes all
GLCM/gradient/shadow computation from map-time inference (prec@5% is
actually slightly better emb-only, 0.936 vs 0.916). Simplicity-vs-points
is Brian's freeze-time product call; both variants stay candidates.

**New program bests: pooled PR-AUC 0.8040, median per-image AUC 0.8284,
win rate 0.96 vs Tier-1** (was 0.5651 / 0.6806 / -- five days ago).

Still open in the freeze window: 1b target re-read, 1d pool x head,
1e winner micro-sweep + cross-head ensemble + calibration layer,
1g operating-scale decision (S=32 rerun of winner), 1h optional 320-px
probe. Then freeze.

## 2026-06-12 -- Freeze window CLOSED (PLAN_FM 1b/1d/1e/1g): recipe FROZEN with Brian sign-off = mlp_ens3 / GeM / emb-only / S=32 / fa_gt_1e-2

Runner: `scripts/probes/_fm_freeze_window.py` (subcommands run/eval/pair;
generalizes the bake-off across scale/pool/target/MLP-arch with per-target
Tier-1 baselines in the identical LOIO harness). Chains:
`_fm_fw_chain{1,2_count,3_s32}.sh`. All cells banked under
`models/fang_probe/fw_*`. Every cross-target metric reads against its OWN
Tier-1 baseline (positive sets differ -- never compared directly).

**1b. Target-definition re-read.** The FM advantage transfers to EVERY
non-degenerate target (each vs its own Tier-1, S=64, mlp_ens3 t1ctx):

| target | pos_rate | Tier-1 pooled | FM pooled | med_auc | dAUC(v) | p | gates |
|---|---|---|---|---|---|---|---|
| fa_gt_1e-2 (area, incumbent) | 0.35 | 0.5651 | **0.8040** | 0.8284 | +0.147 | ~0 | PASS/PASS |
| fa_gt_1e-3 (area) | 0.70 | 0.7442 | 0.9183 | 0.7576 | +0.112 | 5e-4 | PASS/PASS |
| bc_ge_50 (count) | 0.48 | 0.6729 | 0.8260 | 0.8053 | +0.195 | 1e-4 | PASS/PASS |
| bc_ge_100 (count) | 0.35 | 0.4219 | 0.7312 | 0.8213 | +0.153 | 1e-4 | PASS/PASS |
| bc_ge_1 (presence) | 0.93 | 0.9459 | 0.9432 | 0.6501 | +0.065 | 0.50 | fail/fail |

bc_ge_1 was the WRONG operationalization of a count target (Brian, this
session): at S=64 ">=1 boulder" is saturated (0.93 positive, PR-AUC floor
already 0.93) -- it is presence, not a rich/poor split. Replaced with real
count thresholds grounded in the per-tile boulder_count distribution
(`scripts/probes/_fm_count_dist.py`): bc_ge_50 -> 0.48 (near-median),
bc_ge_100 -> 0.35 (base-rate-matched to fa_gt_1e-2). Both pass both gates
decisively, reversing the bc_ge_1 null. New targets registered in
`src/modeling/binary_target.py`. At matched base rate (bc_ge_100 vs
fa_gt_1e-2): area edges pooled (0.804 vs 0.731) but per-image AUC is
near-identical (0.828 vs 0.821) -> the advantage is target-definition-robust;
the count-vs-area choice stays a scientific decision, not a forced one. This
reverses the W0 "count beats area" finding, which held only under handcrafted
features (area signal dominated by large-polygon/shadow-merge noise there).

**1d. Pool x head** (mlp_ens3, t1ctx, S=64, fa_gt_1e-2): GeM(p=3) **0.8040**
> mean 0.8015 > cls 0.7900 pooled (GeM also best win-rate 0.96). GeM
confirmed under the MLP, matching the lgbm-era ablation.

**1e. Winner micro-sweep + ensembling + calibration** -- all three add-ons
REJECTED; the plain mlp_ens3 is the simplest and best form:
- *Arch sweep* (gem192, S=64, vs incumbent 256x64/d0.2 = 0.8040/0.8284):
  128x32/d0.2 0.7992; 512x128/d0.2 0.8108; 256x64/d0.4 0.8028; 128x32/d0.4
  0.8039; 512x128/d0.4 0.8075. Spread 0.799-0.811, incumbent mid-pack, all 7
  cells pass both gates (dAUC ~+0.14-0.15, win ~0.96), none separable at
  n=38. 512x128 edges +0.007 but selecting it = forking-paths overfit ->
  **default 256x64/d0.2 kept** (deeper tuning deferred to cohort expansion).
- *Calibration layer* (post-hoc on incumbent): per-image rank -> pooled
  0.5056 (COLLAPSE); 50/50 blend -> 0.7352; both leave med_auc 0.8284
  unchanged (rank is monotone within image). Per-image quantile transforms
  destroy the cross-image abundance level pooled PR-AUC rewards. The MLP
  wobble that motivated calibration was already solved by the 3-seed mean
  (incumbent stable 0.8040). **Rejected -- the ensemble mean IS the fix.**
- *Cross-head ensemble* (mlp 3-seed + logreg rank-mean, t1ctx): 0.7995 /
  med 0.8168 -- below mlp_ens3 alone; logreg dilutes. (Aside: kNN cannot
  join on t1ctx -- KNNHead L2-normalizes but does not standardize/impute, so
  the 52 mixed-scale handcrafted cols swamp cosine; t1ctx kNN collapses to
  0.5600 vs 0.7709 gem-only. Documented, not used.) **Rejected.**

**1g. Operating-scale decision** (mlp_ens3, GeM, fa_gt_1e-2; S=32 P96 input
vs S=64 P192; each vs its own-scale Tier-1):

| scale | matrix | tile | pooled | prec@5% | med_auc | dAUC(v) | win | gates |
|---|---|---|---|---|---|---|---|---|
| S=64 | t1+emb | 320 m | 0.8040 | 0.916 | 0.8284 | +0.147 | 0.96 | PASS/PASS |
| S=64 | emb-only | 320 m | 0.7852 | 0.936 | 0.8035 | +0.137 | 0.85 | PASS/PASS |
| S=32 | t1+emb | 160 m | 0.7764 | 0.932 | 0.7884 | +0.120 | 0.96 | PASS/PASS |
| **S=32** | **emb-only** | **160 m** | **0.7832** | **0.948** | **0.7865** | **+0.120** | **0.96** | **PASS/PASS** |

Two findings: (i) **S=32 holds skill** -- both matrices pass both gates
(dAUC +0.12, win 0.96); cost vs S=64 is modest (~-0.03 pooled / -0.04 med)
and prec@5% is actually HIGHER at S=32 (top map tiles more reliable). The
S=32 Tier-1 floor is weaker (0.484) so the absolute FM lift is larger at
S=32 (+0.292 pooled vs +0.239). (ii) **Feature elimination is FREE at S=32**
-- handcrafted features add +2 pts at S=64 (1f) but ~0 at S=32 (emb-only
0.7832 ties/edges t1ctx 0.7764). The 1f simplicity-vs-points tension
dissolves at the finer scale.

**FROZEN RECIPE (Brian sign-off, this session):**
- **Scale S=32** (160 m tiles, 4x finer map than S=64) -- Brian: a finer map
  at held skill materially strengthens the "improves on what's out there"
  motivation.
- **Input / embedding**: the 96-px (3x3-context) CTX window -> frozen
  Fang-ViT ViT-B/16 (MAE+DINO, Zenodo 18180801) -> **GeM(p=3) -> single
  768-dim vector per tile**. emb-only = the `emb` matrix = ctx input ONLY
  (no own-tile P32, no handcrafted features). Inference path is literally
  one embedding vector -> MLP; no GLCM/gradient/shadow at map time.
- **Head**: `mlp_ens3` -- 3-seed MLP (768-256-64-1, dropout 0.2, BCE
  pos_weight, AdamW lr1e-3 wd1e-4, early-stop patience 8 on the rotated
  inner-val image), mean of 3 seed probabilities. Per-fold standardize on
  train (median-impute is a no-op: 100% embedding coverage).
- **Target**: `fa_gt_1e-2` (fractional_area > 0.01; Brian's scientific
  choice -- continuity + highest matched-base-rate pooled; count targets
  remain equally valid per 1b).
- **Numbers**: pooled PR-AUC **0.7832** / prec@5% **0.948** / median
  per-image AUC **0.7865** / dAUC(v) +0.120 / win 0.96; both gates PASS.
- Banked: `models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/`.

Standing caveats carried with every claim: transductive pretraining
(disclosure + deployment-matching argument) and post-hoc assembly (the
freeze precedes the §3 pre-declared confirmation on cohort-expansion
images). The frozen path is deterministic modulo the 3 named seeds.

Skipped: 1h (320-px / 5x5 probe) -- not needed; the S=32 finer-map decision
made the larger-context question moot for the operating recipe (revisit only
if cohort expansion reopens scale/context tuning).

NOTE: stricter freeze discipline now applies (PLAN_FM 3) -- no further recipe
shopping on the 38 images; the next number that touches this recipe is the
§3 pre-declared confirmation on held-out expansion images.

## 2026-06-13 -- Tier-2 regression (PLAN_FM 2.4): MLP wins regression too, FM ~2x lift, single-stage beats the hurdle; zero-inflation ceiling TESTED and set aside, compression quantified

Calibrated-abundance regression on the FROZEN emb-only S=32 features (the
regression sibling of the frozen binary recipe). LOIO over the 38 v2 images,
3 heads x 2 targets x {emb, t1 handcrafted baseline}, runner
`scripts/probes/_fm_tier2_regression.py`. Heads: `mlp_reg` (NEW 3-seed MLP
regressor, single-stage), `lightgbm_tweedie` (single-stage), and
`lightgbm_two_stage_balanced` (the incumbent hurdle). Primary metric Spearman
rho (per-image mean); rich/poor `meaningful_auc` at the operational cut
(fa>1e-2 / count>=50, NOT presence -- see the metrics-bug note below).

**Spearman rho (per-image mean, n=38; threshold-free):**

| head | emb.fa | emb.count | t1.fa | t1.count |
|---|---|---|---|---|
| lightgbm_tweedie (1-stage) | 0.313 | 0.286 | 0.228 | 0.213 |
| lightgbm_two_stage (hurdle) | 0.329 | 0.308 | 0.247 | 0.196 |
| **mlp_reg (1-stage)** | **0.431** | **0.386** | 0.223 | 0.202 |

**Rich/poor meaningful_auc (fa>1e-2 / count>=50):**

| head | emb.fa | emb.count | t1.fa | t1.count |
|---|---|---|---|---|
| lightgbm_tweedie | 0.706 | 0.692 | 0.658 | 0.659 |
| lightgbm_two_stage | 0.722 | 0.707 | 0.664 | 0.649 |
| **mlp_reg** | **0.784** | **0.785** | 0.647 | 0.630 |

Verdicts:
1. **MLP wins regression too**: `mlp_reg` is the best head on the embeddings
   (fa 0.431, count 0.386), the same "MLP is the right reader of dense
   embeddings" result as the classification bake-off.
2. **FM ~2x lift**: emb roughly doubles handcrafted rank-skill (mlp fa 0.431 vs
   t1 0.223; mlp count 0.386 vs 0.202). The FM helps regression at least as much
   as classification.
3. **Single-stage beats the hurdle**: `mlp_reg` (single-stage) 0.431 >> the
   two-stage hurdle 0.329; the hurdle only marginally helps the weaker tree head
   (0.329 vs tweedie 0.313). **The hurdle is not earning its complexity with
   strong features** ([[modeling_single_stage_future]] confirmed directionally).
4. **Regression matches the classifier on rich/poor**: `mlp_reg` emb
   meaningful_auc 0.784 (fa) / 0.785 (count) ~= the frozen Tier-1 classifier's
   per-image AUC 0.7865. Calibrated magnitude comes essentially free -- the
   regression head detects rich/poor at classifier level AND gives a continuous
   value. fa marginally edges count (Spearman 0.431 vs 0.386), consistent with
   the frozen target choice.

**Zero-inflation ceiling -- TESTED and set aside** (Brian asked to quantify it;
`scripts/probes/_fm_tier2_ceiling.py`, post-hoc on banked predictions):

| mlp_reg | emb.fa | t1.fa | emb.count | t1.count |
|---|---|---|---|---|
| mean zero-fraction | 0.163 | 0.163 | 0.166 | 0.166 |
| rho_overall | 0.431 | 0.223 | 0.386 | 0.202 |
| rho_among_positives (y>0) | 0.420 | 0.213 | 0.373 | 0.185 |
| zero "drag" (among_pos - overall) | -0.011 | -0.010 | -0.013 | -0.017 |
| NDCG@5% (ceiling-normalized) | 0.502 | 0.348 | 0.484 | 0.337 |
| NDCG full | 0.851 | 0.801 | 0.855 | 0.809 |

At S=32 only ~16% of tiles are EXACTLY zero, and removing the zeros LOWERS
Spearman by ~0.01 -- i.e. the zeros are the easy part (rank them at the bottom),
not a ceiling. The earlier "0.43 is capped by zero-inflation" framing is
**empirically refuted**. The real limiter is the intrinsic difficulty of ranking
magnitude AMONG boulder-bearing tiles (label noise + meter-scale signal at
5 m/px): rho_among_positives (0.42) ~= rho_overall (0.43). NDCG@5% (0.50 emb vs
0.35 t1) is the ceiling-normalized top-tile-ranking number; read with the
classifier's prec@5%=0.95, the top map tiles are almost all genuinely rich, just
not always the very-most-rich.

**Dynamic-range COMPRESSION quantified** (mlp_reg emb.fa calibration, per true
abundance bin): the model hedges to the mean -- under-predicts the high tail
(top bin 1e-2..max: true 0.0373 -> pred 0.0266, **pred/true=0.71**) and
over-predicts the lows. The FM compresses LESS than handcrafted (top-bin
retention 0.71 vs t1 0.55), so the embeddings recover more of the dynamic range,
but a ~30% tail under-prediction remains -- a candidate for a later calibration
layer (isotonic / quantile mapping). Ranking is fine; absolute high-end values
are squashed.

**Metrics bug found + fixed during a code review (regression test)**:
`run_loio` called `per_fold_metrics` with the hardcoded default
`meaningful_threshold=1e-2`. Correct for the fractional-area target (1% areal
coverage = the rich/poor cut) but for the **boulder_count** target it collapses
to `count > 0.01` == presence (count>=1) -- the degenerate metric we rejected
(the bc_ge_1 trap). Fix: threaded `meaningful_threshold` through `run_loio`
(default 1e-2 preserves all existing callers); the Tier-2 runner sets it
per target (fa 1e-2, count 50 = the bc_ge_50 cut). `per_bin_rmse` still uses
fa bin edges, so for count read Spearman + meaningful_auc(@50), not the per-bin
table. Test: `tests/test_evaluate_meaningful_threshold.py`. The 3 emb-count
cells that ran before the fix were re-run with `--force` (predictions
deterministic -> identical; only the metric changed). Spearman was never
affected (rank-only).

Disposition: the Tier-2 candidate is the **single-stage `mlp_reg` regressor on
emb-only S=32**, target fa (or count -- both transfer). The hurdle is dropped.
Not yet frozen/productized -- Tier-2 freeze + the deployable head come with the
map pilot (PLAN_FM 2.6). A calibration layer for the tail-compression is future
work. Standing caveats unchanged (transductive pretraining; LOIO carries the
selection caveat until the 2.3 confirmation).

## 2026-06-14 -- Deployable head + map pilot (PLAN_FM 2.6 A-E): frozen recipe productized to a single all-data model; first off-HiRISE map rendered

**2.6.A deployable head (NEW `src/modeling/mlp_head.py`).** The frozen `mlp_ens3`
classifier lived only inside the LOIO probe harness (re-trained per fold). A map
needs ONE model trained on ALL images, so the head is now productized:

- `FeatureScaler` (median-impute + z-score, frozen-recipe parity), `build_mlp`,
  `MLPClassifierHead` (one 768-256-64-1 BCE MLP, **Model-protocol compliant** so it
  can also drop into the LOIO harness), and `DeployableHead` (the 3-seed ensemble).
- `DeployableHead.fit(X, y, groups)` rotates one inner-val image PER SEED for early
  stopping (seed s holds out `sorted(unique(groups))[s % n]`), so every image is
  in-training for >= n_seeds-1 seeds and none is permanently excluded; `predict` =
  mean of the seed sigmoid probs. This is each LOIO fold's exact procedure minus the
  test fold -> same recipe, all data. `save`/`load` persist 3 seed state-dicts +
  scalers + a self-describing recipe card (`recipe.json`, carries the frozen cell id
  + LOIO numbers + a config-only `recipe_hash`).
- **Perf fix applied** (handoff / `_fm_tier2_regression.py` PERF NOTE): batch 4096 +
  full train tensor pinned to the device once. Batch size is NOT on the frozen recipe
  card (which names arch/dropout/optimizer/target), so this is an implementation
  choice, not a recipe change; the LOIO 0.7832 stands as the recipe's generalization
  estimate. Trainer `scripts/train_deployable_head.py` assembles the all-38 emb-only
  S=32 matrix by unioning the `loio_nfold` per-fold TEST slices (each image appears
  once -> identical embeddings/labels/groups to the harness, no re-derivation).
- **Trained & banked**: `models/deployable/86c51a5dca220f63/` (38 images,
  pos_rate 0.36, 161,005 tiles, **76 s** for 3 seeds). In-sample sanity (NOT a
  validation number) AUC 0.966, p|pos 0.85 vs p|neg 0.15; save/load round-trip
  max |dp| = 2e-7.

**2.6 B-E map pilot (NEW `src/mapping.py` + `scripts/map_pilot.py`).** First real
exercise of the off-HiRISE inference path. A Murray tile is 4x4 deg (~237 km) vs a
~6 km HiRISE footprint, so almost all of any cohort tile is beyond coverage -- and
the tile zips are already cached, **so no download was needed** (the original plan
budgeted one). The pilot windows a cohort tile adjacent to (not overlapping) one
image's footprint, then runs `read_tile_window -> FangEmbedder.embed_window ->
DeployableHead.predict -> tiles_to_raster -> GeoTIFF/PNG`.

- Result (E4_N44, east of ESP_055253_2245, 3000-px window = 15 km): **8281 tiles,
  all valid, embed+predict in 21 s**; mean P(rich)=0.117, >=0.5 share 0.001, max
  0.78. Overwhelmingly poor -- the honest read for smooth plains beyond a rich
  image's footprint -- but the P(rich) heatmap shows spatially-coherent elevated
  patches that visibly track rougher CTX texture (the model responds to terrain,
  not saturated). `reports/figures/map_pilot_E4_N44_ESP_055253_2245_east.png`,
  GeoTIFF + JSON sidecar in `reports/map_pilot/`.
- **Georef bug found by a post-render check + fixed** (regression-tested): the
  per-tile `(ti, tj)` are anchored to the PARENT TILE pixel origin (CLAUDE.md Stage
  4), but `predict_window` first passed the WINDOW affine (already offset to the
  window corner) into `coarsened_transform`, which then re-added `tj_min*tile_px` ->
  the read offset was double-counted (output xmin landed ~21,700 px / ~108 km too
  far east). Fix: `tile_origin_transform` reconstructs the tile origin from the
  window affine + read offset before coarsening. After the fix the output raster's
  top-left sits 50-52 px (one context margin) inside the window edge, as expected;
  CRS = Mars 2015 Equirectangular, pixel 160 m. Tests in `tests/test_mapping.py`
  (`test_window_placement_not_double_counted`, `test_tile_origin_transform_*`).

**Combine pattern (2.6.C) is unchanged-by-design**: `(ti, tj)` are unique within a
tile; cross-tile scale-out additionally keys on the Murray-tile id (the §2.6 note's
"global mosaic origin" assumption doesn't hold in the current code -- the grid is
tile-anchored -- but for placement within one tile it's exact, and combine is just
raster placement once the Murray-tile id is carried). Not built (pilot is one tile).

Tests: +18 (11 `test_deployable_head.py`, 7 `test_mapping.py`); full fast suite
**312 passed**. Caveats unchanged (transductive pretraining; the deployable model
inherits the LOIO estimate as a conservative bound; §2.3 confirmation still pending
the expansion-cohort BoulderNet runs). The §2.7 reliability overlay is the next
buildable piece (the map currently has no trust layer).

**QA notebooks (CLAUDE.md §7).** Added two, both built via `notebooks/_build_NN.py`
and executed clean (0 error cells): **`21_map_pilot.ipynb`** (deployable recipe card
+ reload check; HONEST held-out truth-vs-model at S=32 from the banked LOIO
predictions — anti-signal ESP_046328_2180 redeemed slim 0.344 → FM 0.748; the
beyond-coverage map) and **`22_freeze_and_tier2.ipynb`** (the recipe-selection arc:
head bake-off "trees are the wrong reader" + paired stats, the freeze + target
transfer, Tier-2 single-stage `mlp_reg`, and the compression curve top-bin pred/true
FM 0.71 vs handcrafted 0.55). A notebook audit vs §7 found target-distribution
already covered (08/09/10/11/12); these two close the FM-program gaps. README +
HANDOFF updated. Figures: `reports/figures/21_deployable_truth_vs_model.png`,
`22_{head_bakeoff,tier2_skill,tier2_compression}.png`.

## 2026-06-14b -- Reliability overlay (PLAN_FM 2.7): novelty VALIDATED as an OOD detector but NOT as a per-image skill predictor at n=38 -> overlay DEFERRED to post-expansion

**Built (CPU-only, no GPU, no expansion data):** `src/reliability.py` -- two
label-free per-tile embedding-novelty scorers over the frozen 768-d GeM cloud:
`MahalanobisNovelty` (distance to the training mean in a PCA-whitened top-256
subspace; truncation = shrinkage) and `KNNNovelty` (mean cosine distance to the
k=50 nearest training tiles, reference subsampled to 20k). Both NaN-safe (margin
tiles score NaN) and deterministic. +10 tests (`tests/test_reliability.py`).

**Validation (`scripts/probes/_fm_reliability_validation.py`), LOIO-honest by
construction:** for each held-out image, fit novelty on the other 37's tiles,
score the held-out tiles (capped 3000/image for a stable median), aggregate to a
per-image median novelty, then **Spearman vs the FROZEN RECIPE's OWN per-image
AUC** (read from the banked `fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet`;
38 images both-class, AUC range 0.564-0.919, median 0.787). Pre-registered bar
(PLAN_FM 2.7): novelty must predict where the FM ITSELF underperforms.

**Verdict -- bar NOT cleared:** Mahalanobis rho=-0.108 (p=0.52), kNN-cos50
rho=-0.141 (p=0.40); bottom-5-AUC flag precision 0.00 for both. Direction is
correct (more novel -> lower AUC) but insignificant at n=38.

**Why (structural, and a good-news FM story):** the FM already absorbed the
covariate-shift failure mode (W1 `distribution_shift` class, +0.18-0.22 dAUC), so
on the 38 images **novelty and skill are decoupled**. Evidence:
- The single most-novel image is a FM *winner*: ESP_076499_1160 (the azimuth
  outlier) is novelty rank **1/38** (Maha 37.1 vs ~14-20 for the rest) yet has
  **AUC 0.868** -- genuinely OOD terrain the FM handles well. Dropping it only
  moves rho to -0.174 (p=0.30) / -0.210 (p=0.21): still insignificant.
- The weakest image, ESP_045983_2270 (AUC 0.564), has one of the LOWEST novelty
  scores (13.9) -- intrinsic (texture/sensor-floor) difficulty, not OOD, which a
  novelty detector cannot see.

**Decision (Brian, 2026-06-14): DEFER the overlay until the cohort expands.**
Novelty IS a valid OOD/extrapolation flag (correctly ranks the known outlier #1)
but must NOT be sold as an accuracy predictor; rather than ship a weakly-justified
trust layer on the map, bank the negative result and re-run this same validation
when the §2.3 expansion images land (n=38 is underpowered -- direction right, CI
wide). The map stays trust-layer-less for now. The module + validation + figure
(`reports/figures/27_reliability_validation.png`,
`reports/reliability/per_image_novelty.csv`) are kept as the ready-to-rerun
record. `predict_window` already returns per-tile `(ti,tj)` keys, so wiring is a
small follow-up if the post-expansion validation passes.

## 2026-06-14b -- Model-evidence report (PLAN_FM 2.5) DRAFTED: prose complete; held-out row + schematic figure pending

Filled `docs/model_evidence.md` from skeleton to full prose (slimmer-doc register,
persuasion-grade; companion to `classification_slimmer.md`). All headline numbers
are group-aware LOIO on the 38 v2 images; the `[held-out: pending]` confirmation
row awaits §2.3. Real figures wired in: §0 `21_deployable_truth_vs_model.png`, §4
`20_fang_perimage_dauc.png` + `20_fang_topk_ESP_076499_1160.png`, §5
`map_pilot_E4_N44_*.png` + `27_reliability_validation.png`, §7
`22_tier2_compression.png`. Two substantive edits vs the skeleton: (a) **§5
reliability rewritten honestly** -- the map ships with NO per-tile accuracy
overlay; novelty is retained only as an OOD/extrapolation warning, with the n=38
decoupling result (Fig 5) stated plainly (per the 2026-06-14b deferral above);
(b) **corrected ESP_046328_2180 to FM AUC 0.748** (skeleton said ~0.79; 0.748 is
the banked LOIO value). prec@5% interpretation fixed to ~95% (skeleton text once
said 98%). Remaining to finish the doc: the §3 ViT->GeM->MLP schematic figure and
the held-out confirmation numbers (both gated on other work). Notebook/figure
inventory confirmed all six referenced PNGs exist on disk.

**Revision (same session, Brian feedback): more visual evidence + a Tier-2
deep-dive.** The doc was restructured around four NEW bespoke figures rendered from
cached data (`scripts/probes/_evidence_{select_exemplars,basis_figure,prediction_gallery,product_figure,tier2_map}.py`):
- **Basis figure** (`model_evidence_basis_hirise_ctx.png`, new §2 "can 5 m/px CTX
  see boulders?"): one image's boulder-RICH vs boulder-POOR 160 m tile, each HiRISE
  (0.5 m/px, BoulderNet polygons outlined) beside co-located CTX (5 m/px) — the
  rich CTX is visibly rougher. **Subtlety handled**: the label grid carries the
  coreg shift, so the figure counts/outlines the RAW reprojected detections
  (centroid-in-tile) and reselects a raw-empty poor tile, so annotations match what
  is drawn (rich 198 boulders/10% vs poor 0/0%).
- **Prediction gallery** (`model_evidence_prediction_gallery.png`, §5): 6 held-out
  images spanning terrain+regime (plains/mesas/crater/channels + rescued
  anti-signal + far-south outlier), per-tile P(rich) heatmap + true-rich white
  contour + per-image AUC/base-rate. Hybrid axis (Brian).
- **Headline gap-fill map** (`model_evidence_gapfill_map.png`, §0): the first
  product-figure attempt (`model_evidence_product.png`, a validated|deployed
  two-panel) **confounded validated-vs-deployed with rich-vs-poor** (left scene 80%
  rich/red, right 0.1%/blue) — Brian flagged it. REPLACED (Brian chose "unify into
  one regional gap-fill map") with ONE continuous 24 km scene over a tile containing
  a training footprint: `_evidence_gapfill_map.py` centres a 5000-px window on
  ESP_045139_2270's footprint in cached E12_N44, the deployable head predicts all
  23,409 tiles (**1 GPU run, 37 s**), the HiRISE-confirmed rich tiles are outlined
  inside the footprint box, the rest is gap-fill. Prediction flows seamlessly across
  the footprint boundary and is NOT saturated (reads crater interiors as poor). No
  rich/poor confound — same terrain throughout. Honest caption: footprint tiles were
  in the all-data model's training; generalisation evidence is the held-out §3–§5.
  Script has a `--force`-gated render-only path (reuses the banked GeoTIFF). The old
  product figure+script were removed.
- **Tier-2 map** (`model_evidence_tier2_map.png`, §8): held-out true vs CTX-only
  predicted area-fraction, same log scale (per-image rho 0.74) — ordering faithful,
  magnitude compressed. From banked `tier2_mlp_reg_emb_fractional_area_S32` preds.

**§8 rewritten into a Tier-2 status/reach/use section** (Brian ask): where it
stands (winning single-stage `mlp_reg`, NOT yet frozen/productised), what's
achievable now (rank-reliable abundance, rho~0.43 med / 2x handcrafted; rich/poor +
NDCG free) vs not (tail compression ~30%), 3 graded use-cases (process science +
hazard classes now; THEMIS-comparable absolutes after calibration), and the to-do
(freeze+productise regressor head, calibration layer, THEMIS validation). Doc now
9 figures (Fig 1-9 consistent), 9 sections; schematic + held-out row still pending.

**Figure-1 iteration (Brian feedback, several rounds) → final 3-panel form**:
`model_evidence_gapfill_map.png` is now **plain CTX | HiRISE ground-truth rock
abundance | model P(boulder-rich)**, all three equal-sized (colourbar slot appended
to every panel via `make_axes_locatable`; panel-0's is hidden so panels 1-2 don't
shrink). Both data panels share the **inferno** colourmap ("brighter = more
boulders") — kept distinct colourbars since truth is log area-fraction and the model
is a 0-1 probability. **Footprint outline fixes**: started as the axis-aligned bbox
(included rotated-strip nodata corners) → minimum-rotated-rectangle (overshot the
acute parallelogram corners) → final = **convex hull of the labelled tiles,
simplified** (follows the true sheared strip). **Truth-panel "missing tiles"
explained + handled**: the strict `coverage==1.0` labeling rule (src/labeling.py:18,
DECISIONS 2026-05-23 — partial coverage low-biases fractional_area) leaves ~5.2% of
footprint cells (343/6565 for ESP_045139_2270) unlabelled along internal HiRISE
CCD-seam / SP1 gaps; filled by nearest-neighbour inside the footprint for a clean
display (`_diag_missing_label_tiles.py` quantified it). White rich-truth contour
dropped from the model panel (redundant with the truth panel).

**Tier-2 compression CORRECTED to two-sided** (Brian spotted Fig-8 map missing the
low/purple end; `_diag_tier2_compression_direction.py`): it is regression-to-the-mean,
NOT just the high tail. Low end OVER-predicted (floors ~0.005, 1.8% of preds
near-zero vs 18% of tiles truly zero — the "missing purple"); high end under-predicted
(top fixed bin pred/true 0.71, top decile 0.53); crossover ≈ rich/poor threshold
0.015. Ranking unaffected (not capped by zeros); fix = ranking-preserving remap that
stretches BOTH ends. §8 prose + Figs 8/9 captions corrected;
[[tier2_compression_calibration_future]] updated. Earlier "high tail ~30%" framing
was incomplete.

## 2026-06-14c -- Calibration / de-compression: PLAN_Calibration.md + Stage 0 (diagnose + post-hoc) done

Brian: the compression is a big, important problem (visible in both the Tier-1
P(rich) and the Tier-2 abundance map); wants a full plan including new training
ideas. Created **PLAN_Calibration.md** (standalone; operationalizes
PLAN_ModelUsability W3 + PLAN_FM item 4) and did Stage 0:

- **`src/calibration.py`** (NEW): `reliability_curve`/`expected_calibration_error`,
  `TemperatureScaler` (1-param, AUC-exact), `IsotonicCalibrator`, `quantile_match`
  (histogram/quantile transfer), `compression_metrics`, `loio_calibrate` (fit on
  other 37 / apply to held-out). +8 tests (`tests/test_calibration.py`); **fast
  suite 330**. **`notebooks/23_calibration_diagnostic.ipynb`** (built via `_build_23.py`,
  executed clean) + figs `23_{tier1_calibration,tier2_compression,tier2_decompression}.png`.

- **Findings (LOIO, banked preds)**: **Tier-1 is already well-calibrated** — ECE
  **0.060**, std 0.36, AUC 0.848; temp scaling (T≈1.70) → ECE 0.049, AUC flat. So
  Tier-1 is NOT the problem; the "mostly rich" maps over rich regions are correct.
  **Tier-2 quantile-matching is the post-hoc win**: top-bin ratio 0.71→**0.87**,
  near-zero pred 1.8%→**18.6%** (= truth), marginal-L1 0.0057→**0.000**, Spearman
  0.651→0.644 (preserved). **Isotonic does NOT help** (fits the compressed mean).

- **Plan framing (Brian's "full consideration")**: compression = the Bayes-optimal
  behaviour of any mean-seeking estimator under aleatoric uncertainty on a skewed
  target → FOUR exhaustive levers: L1 stop predicting the mean (HL-Gauss / quantile
  / distributional / ordinal — `Imani&White 2018`, `Farebrother 2024 "Stop
  Regressing"`, `Koenker 1978`, `Kendall&Gal 2017`, `Cao 2020`), L2 shrink p(y|x)
  (coarser scale, count+Poisson target, min_confidence, multi-scale/fine-tune, more
  data), L3 post-hoc marginal-match (DONE), L4 report the distribution/interval.
  Imbalanced-regression machinery (LDS/FDS `Yang 2021`, density weighting
  `Steininger 2021`) re-weights any objective. Staged execution table + declared
  metrics in the plan. **Discipline: post-hoc calibration does NOT reopen the Tier-1
  freeze; Tier-2 mlp_reg not yet frozen so L1/L2 retraining allowed there.** PLAN_FM
  item 4 + PLAN_ModelUsability W3 cross-referenced; README notebook list updated.
  `fractional_area` currently uses identity+MSE (log1p untried for it -- an L1/L2
  quick win). Diagnostics: `scripts/probes/_diag_{tier2_variant_compression,calibration_preview}.py`.

- **Tier-1 calibrator follow-up (Brian spotted temp scaling trades the ends).**
  `_diag_tier1_isotonic.py`: temperature is ONE global knob, so it fixes the
  over-confident high end at the COST of the low end (split-ECE low 0.043→0.063, high
  0.096→0.021). A FLEXIBLE monotone calibrator does both: **isotonic** → ECE
  0.060→**0.014** (low/high both 0.014) but flat-step ties cost ranking (AUC
  0.848→0.833). Recommendation = a smooth strictly-monotone calibrator (beta /
  monotonic spline) for both-ends fix with minimal ranking loss; gate ECE≤0.05 AND
  AUC±0.005 (isotonic alone breaches the AUC gate). Notebook 23 Tier-1 cell updated to
  show all three. Clarified in the plan: **target transforms (log1p/sqrt/Box-Cox) and
  loss-function changes are both L1** ("what functional of p(y|x) the loss targets").
- **DRAFT calibrated figures (preview only, NOT wired into model_evidence.md; originals
  untouched).** Two:
  - `_evidence_tier2_map_calibrated.py` → `model_evidence_tier2_map_calibrated.png`:
    the §8 Tier-2 abundance map with quantile-matching (LOIO), 3-panel TRUE | raw |
    de-compressed for ESP_053989_2260 — per-image top-bin 0.84→**1.05**, Spearman 0.74
    unchanged; the dark low-abundance feature + crater ring visibly recover.
  - `_evidence_gapfill_map.py --calibrate` → `model_evidence_gapfill_map_calibrated.png`:
    the §0 headline gap-fill map with the **Tier-1 isotonic calibrator** applied to the
    off-HiRISE P(rich) (fit on the 38 labelled images = deployment-honest). Effect
    modest (Tier-1 already ECE 0.06): mean P 0.747→0.666 (over-confident highs corrected
    red→amber), ≥0.5 share 0.79 unchanged (ranking-invariant).
  Confirms the Tier-1 "flexible monotone calibrator" insight transfers to Tier-2 via
  quantile-matching — bounded by ranking (texture floor), not calibration.

## 2026-06-15 -- Calibration prototypes: Tier-1 isotonic (NOT beta), Tier-2 L1 swaps ruled out

Ran the highest-leverage next probes from PLAN_Calibration. Two clear verdicts.

- **Tier-1: ISOTONIC is the calibrator** (Brian caught that isotonic beats beta).
  Added `BetaCalibrator` (Kull 2017, smooth 3-param strictly-monotone; +1 test, suite
  331) and compared LOIO (`_diag_tier1_{isotonic,beta}.py`): isotonic ECE
  0.060→**0.014** (low/high both 0.014) vs beta 0.040 (3 params underfit the
  reliability curve) vs temperature 0.049 (trades the low end for the high). The
  ranking worry was wrong: the LOIO pooled-AUC drop (isotonic 0.848→0.833) is a
  PER-FOLD artifact — a single GLOBAL calibrator is AUC-exact (isotonic +0.0003, beta
  +0.0000), and ties are harmless at n=161k. So isotonic = Tier-1 `CalibrationLayer`;
  beta = smooth fallback. (My earlier "isotonic breaches the AUC gate, use beta" was
  based on the artifact — corrected in the plan + notebook 23.)
- **Tier-2 L1 swaps (log1p, count-Poisson) RULED OUT** (`_diag_tier2_objectives.py`,
  same LOIO protocol, batch 4096): identity vs log1p is a WASH (raw top-bin 0.66→0.67,
  per-image ρ 0.433→0.445 = noise; both →0.87 after quantile-match); count-Poisson is
  WORSE (ρ 0.425, top 0.54, +qmatch only 0.78) because count→area conversion discards
  per-tile boulder-size info. The compression is intrinsic (aleatoric floor), not a
  target-scale artifact -> the cheap L1 swaps don't help. Remaining L1 lever =
  HL-Gauss/quantile; the ranking ceiling needs L2 (coarser scale / less label noise).
  quantile-match (L3) remains the Tier-2 de-compression win (identity+qmatch 0.87).
- **Tier-1 accuracy** (Brian aside, `_diag_tier1_accuracy.py`): 0.800 @0.5 (vs 0.640
  majority-class baseline), balanced-acc 0.775, F1 0.712, prec/rec 0.737/0.689 — LOIO.
  Plan/notebook 23/scorecard updated; metrics gates unchanged.

## 2026-06-15 -- Calibration Stage 2: L1 bake-off ruled out, L2 scale directional, Tier-1≈Tier-2 ranking

Brian chose the expensive Tier-2 path (improve calibration, don't yet productize). Same
emb-only S=32 LOIO protocol as the cheap-swap probe; every readout scored raw AND
+quantile-match, with **paired per-image Wilcoxon** as the must-not-regress guard (the
median-of-medians glance is misleading — both "wins" below shrank under pairing).

- **L1 distributional bake-off** (`_diag_tier2_l1_bakeoff.py`: HL-Gauss histogram loss
  + pinball quantile + neural zero-inflated-lognormal, each a 3-seed MLP head): **all a
  WASH on per-image ranking** vs `mlp_reg`. Best = pinball.median (paired Δ −0.002,
  18/38 wins, **p=0.48**); HL-Gauss.mean Δ −0.017 (p≈0.08), ziln Δ −0.019/−0.025 — i.e.
  tie-to-slightly-worse. Confirms the cheap-swap result for the *heavy* losses too:
  **compression is the intrinsic aleatoric floor, not a loss-shape artefact**, so
  changing the targeted functional moves the *value* (de-compresses) but not the *rank*
  (all qmatch can't fix). **Two keepers:** (1) `pinball.P90` raw top_ratio **0.98** — a
  tail-calibrated point WITHOUT the L3 layer, no ranking cost; (2) ziln.median recovers
  near-zero mass (9.9 % vs truth 18 %). **L4 caveat:** both heads' [P10,P90] cover only
  **~58 %** vs nominal 80 % → they under-estimate their own spread; the honest-interval
  product needs interval recalibration. Artifacts: `models/fang_tier2/l1_bakeoff/`.
- **L2 scale sweep S=32→64** (`_diag_tier2_scale_sweep.py`, mlp_reg identity): raw
  top_ratio 0.66→**0.72**, pooled rho 0.648→**0.695**, per-image rho paired Δmed
  **+0.025** (25/38 images) — directionally the right way but **Wilcoxon p=0.19, NOT
  significant at n=38**, and partly an easier-target artefact (true-zero share 18 %→
  6.9 % at 320 m). So coarsening *probably* helps and the Tier-2 map may run coarser
  than Tier-1, but the in-cohort ranking ceiling is **sticky** — a confident gain needs
  the §2.3 expansion. S=128 deferred (needs a P384 ViT embedding pass + 128-px label
  grid). S=64 per-tile preds banked at `l1_bakeoff/preds_mlp_reg_S64.parquet`.
- **Independent ceiling proof — Tier-1 P(rich) ≈ Tier-2 regressor as a magnitude
  ranker.** P(rich), a classifier that never saw `fractional_area`, ranks it per-image
  at **0.437** — statistically identical to the dedicated regressor's **0.433** (counts
  0.436); within rich tiles only 0.34. Two different model families hit the same ~0.43
  wall ⇒ the magnitude signal in 5 m/px CTX **is** the rich/poor signal. **Direct test
  (qmatch both onto the fa marginal, notebook 23 §7b):** identical marginal, but the
  dedicated regressor keeps a small, borderline ranking edge (pooled 0.642 vs 0.625;
  paired per-image wins 24/38, Wilcoxon p≈0.05). So the **one-model simplification**
  (drop the Tier-2 head, use a quantile-matched P(rich)) is **viable at a ~0.02 ranking
  cost, not free** — a Stage 1/4 option.
- **Net:** L1 fully ruled out as a ranking lever; L2 is the only remaining lever and
  even it is unconfirmed in-cohort. **qmatch (L3) stays the product win** for the
  marginal; the per-tile ceiling is the data. Next greenlit (Brian): `min_confidence`
  label-noise sweep + Stage 2c density/LDS reweighting. PLAN_Calibration §3/§5 + notebook
  23 updated.

## 2026-06-16 -- Calibration Stage 2b/2c CLOSED: reweighting dominated, label-noise harmful -> ceiling is the data

The last two greenlit Tier-2 levers, same emb-only S=32 LOIO + paired per-image
Wilcoxon. Both NEGATIVE -> the expensive de-compression investigation is complete.

- **LDS reweighting** (`_diag_tier2_reweight.py`, Stage 2c): inverse smoothed-density
  sample weights on mlp_reg. De-compresses the RAW marginal more as weighting sharpens
  (top_ratio 0.67 none -> 0.77 lds_sqrt -> 0.88 lds_inv) but at a **significant paired
  ranking cost** (sqrt Δ −0.014 p=0.018; inv Δ −0.032 p=0.015). qmatch recovers the same
  marginal for free with zero ranking cost -> reweighting is **DOMINATED, ruled out**.
- **min_confidence label-noise** (`_diag_tier2_minconf_sweep.py`, Stage 2b): regenerate
  S=32 labels keeping detections with `score >= t` (Stage-4 regen from cached Stage-1/2/3,
  `min_size_m=1.4105` held; swap fa into the folds by (obs_id,ti,tj) since the grid is
  detection-independent). **Validated:** regen(none) reproduces dataset_v2 EXACTLY (0.00
  diff, 0 key-misses). Result **HARMFUL, ruled out**: filtering monotonically degrades
  BOTH ranking and dynamic range -- conf>=0.5 paired Δ −0.021 (p=0.010); conf>=0.7
  collapses rich share 36%->11%, top_ratio 0.66->0.31, paired Δ −0.070 (p<0.001).
  Low-confidence detections are REAL boulders, not removable noise. (Built with
  config_v2.yaml / hirise_40_vclaire.csv -- NOT the v1 config.yaml/priority10 manifest;
  the regen is CPU-only and was run concurrently with the GPU reweight via `--regen-only`.)
- **STAGE 2 COMPLETE -- the whole levers table:** L1 cheap swaps = wash; L1 distributional
  heads = wash; L1+L2 reweighting = dominated; L2 label-noise = harmful; L2 coarser scale
  = directional only (p=0.19). **No in-cohort retraining lever beats mlp_reg+qmatch on
  ranking.** The ~0.43 per-image ceiling is the 5 m/px CTX magnitude floor, confirmed five
  ways. **Path forward is NOT a better model:** ship Stage 1 (productize qmatch + isotonic
  into the map) + §2.3 expansion cohort (the only thing that can raise the ranking ceiling).
  PLAN §3/§5 + notebook 23 (§8 + §9 verdict, +figure) updated.

## 2026-06-17 — Validation-raster sources verified (topo + thermal; PLAN_RegionalMap §3, phase 1)

Net-new retrieval (`src/validation_retrieve.py`) for the regional-validation legs. URLs, sizes,
projections, and longitude domains verified against USGS Astropedia / ASU MarsSpaceFlight today:

- **MOLA MEGDM DEM 463 m/px** — `https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`
  (~2 GB; simple cylindrical; clon 0). Topography underlay + paleoshoreline contours.
- **THEMIS night-IR 100 m/px (60N60S) v14** — `https://planetarymaps.usgs.gov/mosaic/Mars_MO_THEMIS-IR-Night_mosaic_60N60S_100m_v14.tif`
  (**15 GB**; simple cylindrical; **0–360° E**; 8-bit; 213388×71130). Our block (40–48°N) is
  well inside the ±60° coverage. **Too big to download → windowed `/vsicurl/` only.**
- **TES thermal inertia nightside 2005 (Putzig & Mellon) ~3 km/px** —
  `https://mars.asu.edu/data/tes_putzigti/nighttime2005/nmap2003.tif` (7200×3600, 20 ppd;
  simple cylindrical; −180..180; ocentric). Small/global → download or vsicurl both fine.
  **Caveat for leg 2:** `nmap2003.tif` may be a *rendered* TI map rather than physical TI
  values — confirm the value semantics before computing the abundance↔TI rank correlation.
  **No dust-cover-index raster** is served alongside it; the TES DCI confound mask (PLAN §7)
  needs a separate source (Ruff & Christensen 2002 DCI) — deferred until leg 2.

Design notes pinned in code: the source CRS/units are **read from each file, never assumed**
(USGS "simple cylindrical" GeoTIFFs vary between metre-equirectangular and degree-geographic);
our geographic region bounds are projected into the source CRS via pyproj, only the covering
window is read, and the product is warped onto a grid in the **CTX clon_0 CRS** (so it
co-registers with `reports/map_region/` outputs). TLS to these hosts reuses the project-wide
`HIRISE2CTX_INSECURE_TLS=1` opt-in (GDAL `GDAL_HTTP_UNSAFESSL`) for the incomplete-chain case;
otherwise GDAL is pointed at certifi's CA bundle.

## 2026-06-18 — Regional map expanded to 26 tiles; rectangular-artifact + thermal-source findings

- **Map expansion:** 7-tile block → 26-tile circum-Chryse map (box lon[-10,10] lat[32,46] snapped
  to whole tiles + 2 NE tabs). All 26 GeoTIFFs back from Sherlock (job array, ~clean folder).
  Notebook 24 reframed 7→26; regional MOLA re-fetched at bounds [-12,32,20,48].
- **"Rectangular predictions" investigated (Brian's QA):** NOT a pipeline bug. (1) 4096px
  read-window seams don't align with the structure → assembly clean. (2) Per-tile mean abundance
  follows a smooth, geologically-correct N→S gradient (N44≈0.011 → N32≈0.0005), not arbitrary
  per-tile radiometric jumps. The rectangular *impression* = (a) the cosmetic white tile-outline
  gridlines drawn on the mosaic (now faded to alpha 0.12), (b) the SE nodata corner, (c) a real but
  SECONDARY CTX-radiometry effect: abundance has a weak non-monotonic dependence on CTX brightness
  (peaks mid-DN, r≈0.07) so pushbroom orbital-track/frame seams modulate predictions slightly. This
  is the 5 m/px CTX-mosaic floor, not the model; the independent thermal legs are the test of it.
- **TES `nmap2003.tif` is UNUSABLE for leg 2:** 3-band uint8 **RGB rendered map**, no CRS (identity
  transform), values 0–255 — a colorized display image, not physical thermal inertia. Disabled in
  config. Quantitative leg 2 needs a real georeferenced single-band TI raster (candidates: PDS
  Geosciences TES TI maps; THEMIS quantitative TI / Fergason et al. 2006).
- **THEMIS night-IR seam:** our region (lon −12→+20°E) straddles the prime meridian, and the THEMIS
  60N60S v14 mosaic is stored 0–360°E → a windowed read crosses the 360/0 seam, which
  `validation_retrieve` currently refuses. Leg 1 (the visual co-location panel) needs seam-crossing
  handling (two reads either side of 0°, reprojected into the common clon_0 grid). MOLA/TES are
  −180/180 so unaffected.

## 2026-06-18b — THEMIS seam handling + leg-2 TI product chosen

- **THEMIS night-IR seam:** the v14 mosaic is a projected equirectangular with
  **central_meridian=180** (verified from its CRS), so its raster seam is at **lon 0**, which the
  circum-Chryse region (lon −12→+20) crosses. `validation_retrieve` now auto-detects the seam from
  the source CRS (`seam_lon` = cm+180) and splits the read into two halves (`split_bounds_at_seam`),
  reprojecting + merging each — replaces the old `_wrap_lon` modulo. Fetched at `--match-mosaic`
  (160 m, co-registered to the abundance grid) for both legs. (Striped TIFF, no overviews → each
  seam-half windowed read transfers full-width rows over the lat band, ~few GB; one-time.)
- **Leg-2 quantitative TI source = THEMIS-Derived Thermal Inertia (Fergason et al. 2006), NOT TES.**
  32-bit **physical** TI, 100 m/px, same instrument as the night-IR panel. Served as ISIS3 `.cub`
  tiles (30°×60°, ~2.36 GB each) at
  `astropedia.astrogeology.usgs.gov/download/Mars/Odyssey/THEMIS-Global-Thermal-Inertia-Mosaic/Quantitative-32-Bit/THEMIS_TI_Mosaic_Quant_{LAT}{LON}_100mpp.cub`.
  Our region needs **two** tiles: `30N000E` (0–60°E → covers 0–20°E) + `30N300E` (300–360°E → covers
  −12–0°E). GDAL reads `.cub`. Wiring needs multi-tile-source support in `validation_retrieve`
  (current model = one global source → window); deferred until leg-1 ships. TES Putzig physical TI
  (MARSTHERM/PDS, 3 km) is the fallback + carries the dust-cover index for the confound mask, but the
  ASU `nmap2003.tif` is the unusable RGB render (see 2026-06-18).

## 2026-06-18c — Striping-artifact investigation §1+§2 (PLAN_StripingArtifact); mitigation NOT started

> **SUPERSEDED by 2026-06-18d (below).** Brian clarified the artifact is *high-amplitude rectangular
> blocks visible without detrending, tilted (not vertical)* — NOT the faint vertical banding analysed
> here, and NOT tile-assembly seams. The §1a/§1b/§2 findings below stand as facts but addressed the
> wrong feature; the vertical-stripe "identification" and the seam-LINE test were mis-reads. The
> correct cause = CTX source-frame radiometry (18d).

Ran the cheap, no-re-inference characterization on the 26 written map tiles
(`scripts/striping_characterize.py`) + the decisive edge/seam tests on the 9 tiles that have an
abundance raster + cached CTX zip + cached Murray Lab **SeamMap** (`scripts/striping_seam_test.py`).
Figures: `reports/figures/striping_fft_*`, `striping_orientation_summary.png`,
`striping_seam_*.png`, `striping_*_summary.csv`.

- **§1a (geometry):** the structure is **aperiodic** → FFT gives only weak anisotropy (~1.3) and an
  unreliable orientation (FFT = wrong tool). Confirmed: it is **identical in `prob_raw` and
  `abundance`** → in the **raw model output, not introduced by qmatch**. A directional banding metric
  (col-vs-row variance) shows banding is **weak and not strongly vertical** (V≈H≈0.005–0.006) — i.e.
  no strong organised N–S periodic stripe.
- **§1b (edge coincidence) — POSITIVE, robust:** |∇abundance| ~ |∇CTX brightness| Spearman
  ρ ≈ +0.08…+0.27 (**median +0.14, all 9 tiles positive**), vs row-shuffled null ~0.00. The model
  **is** sensitive to CTX brightness/texture. BUT that brightness structure is **dominated by geology**
  (valley networks / ridges / craters; e.g. E12_N44 frames = "closely-spaced valleys"), not linear
  track seams.
- **§2 (gold-standard seam test) — provenance EXISTS but test UNDERPOWERED → inconclusive.** Murray
  Lab **does** ship a per-frame **SeamMap shapefile** (one polygon per source CTX frame +
  PRODUCT_ID/INCIDENCE/EMISSION/IMAGE_TIME), cached for 9 map tiles. Rasterizing footprint boundaries
  and comparing |∇abundance| near vs far from a seam gives ratio **~1.02 (0.86–1.14)** — no real
  elevation. **Confound:** ~758 heavily-overlapping *candidate* frames per tile tile the whole 237 km
  tile densely → every pixel is within ~1 km of a boundary (seam-distance maxes at 1.75 km); and these
  are candidate footprints, not the mosaic's actual per-pixel *selected*-frame seams. So it neither
  confirms nor refutes frame-seams.
- **VERDICT:** the **CTX-brightness-sensitivity mechanism is confirmed**; the **specific CTX
  frame-stitching-seam hypothesis is NOT** (and is somewhat weakened — visible structure looks like
  geology). **Per Brian's standing caution, this is explicitly NOT documented as a "5 m/px CTX-floor
  limitation"** — a failed/underpowered seam test does not prove an irreducible floor. Item stays
  **OPEN**.
- **Notebook 25** (`notebooks/_build_25.py` → `25_striping_artifact.ipynb`, figs
  `reports/figures/25_striping_*`) records all of the above with visuals **+ zoomed edge panels**:
  (i) a Murray-tile-boundary zoom and (ii) an internal-geology-edge zoom.
- **§3 tile-seam zoom — assembly re-verified CLEAN (quantitatively).** A single cross-seam transect
  *looked* like an abundance step, but that was a coincidental geology contrast at that latitude.
  Averaged over the **full** lon=8°E seam, the abundance step is **+8e-5 — at the 16th percentile of
  the interior-column (geology) null**, i.e. *smaller* than ~83% of random interior columns; CTX is
  continuous (−0.4 DN) since Murray "tiles" are just download chunks of one seamless global mosaic.
  So tile seams are NOT the rectangles. The internal-edge zoom confirms §1b at pixel scale (abundance
  edges trace CTX brightness/geology). Net: the "rectangle" impression = cosmetic 4° gridlines +
  geology blocks + the weak CTX-brightness sensitivity, **not** assembly discontinuities.
- **Striping POSITIVELY IDENTIFIED + highlighted** (notebook §1a.2, fig `25_striping_highlight.png`;
  `stripe_enhance` = detrend + average along-stripe). It is **low-amplitude, aperiodic, VERTICAL
  (N–S) banding, ~km-coherent**, visible at Brian's lon 11°E/lat 36°N example and elsewhere. This
  reconciles with the weak full-column banding index (§1a): the bands are ~km-scale, not full-tile
  height, so a full-column-mean variance washes them out. On equipped tile E8_N44 the CTX brightness
  shows vertical bands too, sharing structure (2D stripe ρ ≈ +0.20) but with **column positions only
  weakly aligned** (column-profile ρ ≈ −0.09) → the abundance stripes track CTX **texture/contrast**
  more than mean brightness. ⇒ the planned synthetic test must include a **contrast lever**, not just
  offset/gain.
- **MITIGATION NOT STARTED** (PLAN §4 gate: wait until §1–§3 confirm the cause).
- **NEXT (pre-mitigation):** §3-plan synthetic brightness gain/offset/**contrast**-step susceptibility (clean, no
  seam-density confound) to quantify how much a radiometric step moves predicted abundance; and a
  better-powered §2 using the per-pixel *selected*-frame / per-frame incidence map (SeamMap metadata)
  instead of all candidate footprints. Thermal legs remain the external adjudicator.

## 2026-06-18d — Artifact CAUSE FOUND: CTX source-frame radiometry (rectangular blocks)

Brian corrected the target: the artifact is **high-amplitude rectangular blocks** (visible in the raw
map, no detrend), **tilted not vertical** — i.e. CTX **source frames**, not vertical stripes or
tile-assembly seams. Re-investigated and **positively confirmed** the cause.
Code: `src/striping.py` (now hosts the analysis fns) + `scripts/striping_frame_blocks.py`; visuals in
`notebooks/25_striping_artifact.ipynb` (full rewrite) + `notebooks/24` §2d (raw-CTX). Figures
`reports/figures/{25_artifact_*,26_frameblocks_*}.png`, CSVs `striping_frameblocks_*`.

- **SeamMap is a PARTITION** (sum frame area ≈ union ≈ tile area; ratio 1.0): one source CTX frame per
  pixel. ~800 polygon fragments dissolve (by `PRODUCT_ID`) to ~**46–63 distinct source frames per
  tile**. (Corrects 18c's "758 overlapping candidates" — they don't overlap.) SeamMaps for any tile
  are pulled from the remote Murray zip via `/vsizip/vsicurl/` range requests (no GB download),
  cached as `cache/ctx_tiles/_frames_<tile>.gpkg` (`src.striping.load_frames`).
- **Blocks align with frames** (notebook 25 §1): the bright/dark rectangles are bounded by source-frame
  footprints; e.g. Brian's lon 11°E/lat 36°N block (E8_N36) steps ~0.003→~0.02 across a frame edge.
- **eta² (variance explained by frame) of DETRENDED abundance: median 0.011 vs rotation-null 0.002,
  89% of tiles > null-95p** — frames carry ~5–9× the null abundance structure. The frame-mean
  choropleth reproduces the blocks after geology is removed. THIS is the decisive, robust test.
- **Per-frame effect is mean-brightness-WEAK, texture/contrast-driven:** pooled per-frame
  Spearman(CTX DN, detrended abundance) ≈ **+0.14** (n≈400 frames); geology-controlled near-boundary
  step Spearman(dAbund, dCTX) ≈ **+0.10** (n≈880 adjacent-frame pairs). Both only weakly positive
  because abundance↔brightness is non-monotonic and the embedder passes each frame's full texture.
- **Why a filled block, not a seam line:** the model is per-patch with a **fixed `/255` embedder
  scaling and NO per-frame normalization** (`src/fm_embeddings.py:264`); a frame's radiometry is
  ~uniform across the whole footprint, so every interior patch is biased alike → filled rectangle. An
  edge artifact would be boundary-only. This rules out a stitching-*discontinuity* mechanism.
- **Why invisible in development:** training CTX windows = footprint bbox + `buffer_m=1000` ≈ **8 km**,
  inside a single **~28 km** CTX frame (SeamMap SAMPLES≈5056×~5.5 m), so no within-scene frame seams in
  training; **LOIO scores per-image = per-frame**, blind to a frame-level block offset. It only appears
  when a contiguous deployment scene spans many frames. (Image-level radiometry may be weakly confounded
  with the rich/poor label in the 39-img cohort → the model may even use it as a cue.)
- **VERDICT:** cause = **CTX source-frame radiometry**, established positively (not by elimination), so
  this is NOT a "CTX-floor" cop-out. **Mitigation candidate = per-frame radiometric normalization**
  (re-tint each source frame to a common DN distribution via the SeamMap partition, before embedding)
  = the deferred per-image/per-track standardization bet, applied per frame. **Not implemented**; next
  step = prototype on one tile, re-score, and adjudicate by **LOIO skill preserved (per-image AUC≈0.43)
  + THEMIS/TES thermal ρ ideally up**.

## 2026-06-19/20 — A1 striping mitigation prototype (per-frame CTX normalization)

Brian chose to prototype **A1** (per-frame robust offset+gain CTX normalization before the frozen Fang
embedder) and measure its payoff. Mitigation options + the A-vs-C reasoning are written up in
PLAN_StripingArtifact "MITIGATION" §. Code: `src/striping.py` (`a1_apply`/`a1_normalize_window`/
`a1_normalize_per_frame`, + `eta2`/`load_frames`/`frame_label_map`), `scripts/striping_frame_radiometry.py`,
`scripts/striping_a1_loio.py`, `scripts/striping_a1_infer_crop.py`; 7 new tests (`tests/test_striping.py`).

- **Pipeline framing:** `CTX → [A1] → frozen ViT → embedding → MLP head`. ViT frozen; only the head
  retrains. Embeddings are cached per exact input → A1 = new input → must re-embed (frozen-ViT forward)
  then re-bake the head. A1 acts *before* the frozen ViT (higher ceiling than C, which only adjusts the
  head after it). Training windows are ~single-frame, so A1 at train = per-window normalization.
- **Diagnostic** (`striping_frame_radiometry.py`, 380 frames): between-frame **level spread ≈20 DN**,
  **scale CV ≈0.43** — both large; a robust offset+gain collapses the per-frame DN histograms. → A1 is
  the right first cut. Reference **m0=125 DN, s0=27.7** (global median-of-frame-medians / median-IQR).
- **Implementation:** A1 wired into the embedder (`_w2_fang_embed.py --norm a1 --out-suffix _a1`);
  re-embedded the 38-img cohort → `dataset_v2/fang_embeddings_a1/` (frozen ViT, ~14 min). Loaders got a
  `store_name` param (so head/eval point at either store); full suite **351 passed** (no regression),
  **+7 striping tests → 358**.
- **SKILL GATE** (`striping_a1_loio.py`, frozen `mlp_ens3` LOIO, baseline vs A1):
  baseline median per-image AUC **0.790** / pooled PR **0.777**; A1 **0.766** / **0.771**. **Δ median
  AUC = −0.024** (marginal FAIL of the −0.02 gate), **Δ pooled PR = −0.007** (negligible). Reading: the
  model **was using absolute CTX radiometry as a within-image cue** (the flagged confound); A1 removes
  it at a small within-image cost. Caveat: **LOIO is within-frame and blind to the cross-frame artifact**
  — so this is only A1's *cost*, not its *benefit*. Brian: measure the payoff (eta² + thermal) before
  judging the trade.
- **PAYOFF (eta² on a re-inferred crop)** — `striping_a1_infer_crop.py` on an E8_N44 8-frame / ~75 km
  crop (218,089 of 219,961 160-m prediction tiles), raw P(rich), baseline head+raw CTX vs A1
  head+per-frame-A1 CTX: **eta² 0.196 → 0.141 = 28% reduction** (fig `striping_a1_payoff.png`). So A1
  **partially** removes the artifact — the per-frame block levels flatten but are NOT gone. Residual =
  offset+gain doesn't capture the per-frame *shape/contrast/noise-character* differences the ViT keys
  on (the diagnostic's ~0.4-IQR residual + the frozen-ViT ceiling).
- **NET (A1):** partial 28% artifact reduction for a −0.024 median LOIO cost — real but **not decisive**.
  A1 helps, doesn't solve. **NO DECISION TAKEN** — full option space (A2 / B2 / C / D / E / combos) +
  pros/cons collected in PLAN_StripingArtifact "NEXT SESSION — decision setup". **Key un-run adjudicator
  = THEMIS/TES thermal ρ** (baseline vs A1 vs de-block): it tells whether the −0.024 LOIO cost is real
  (removing signal) or illusory (LOIO is within-frame; the artifact is cross-frame), and it adjudicates
  the de-block option. Measure thermal ρ FIRST next session, then decide.

## 2026-06-20 — striping artifact: literature review complete (5 papers read; NO decision)

Brian provided 5 paywalled PDFs; all read in full. Findings consolidated into
`PLAN_StripingArtifact.md` "LITERATURE & DATA-ROUTE FINDINGS" + memory
`regional_map_rectangular_artifact`. Five independent sources converge on the diagnosis and
sharpen the option menu (esp. F + a new entry A-meta). Compute now framed for **global** inference
(full mosaic ≈ 86,571 source frames), not just the 26-tile box.

- **Mechanism (Dickson 2024, 10.1029/2024EA003555):** each CTX frame is independently contrast-stretched
  to **min/max = mean ± 8σ + a uniform non-linear tone stretch** *before blending*, then feathered →
  the artifact is a per-frame **contrast** rescale (not a brightness offset; matches our mean-DN
  Spearman +0.14), and the nonlinear part is **non-invertible from mosaic pixels** → explains A1's 28%
  ceiling. Authors: mosaic "should not be used for radiometric statistics."
- **Walter 2024 (10.1029/2023EA003491, open):** CTX flat-field **stable to ±2% over the ~20-yr mission**
  → a uniformly `ctxcal`'d EDR set is frame-consistent to ~2% (**sets F's quality ceiling, high**).
  Independently calls Murray's per-image `cubenorm` a "**high-pass filter**" that homogenizes natural
  reflectance — our mechanism, from a calibration expert. Use new **v0003 flat-field**; EDR route
  recovers **12-bit** (mosaic is 8-bit).
- **Fang 2026 (10.1029/2025JH000827):** our embedder assumed the mosaic gives "consistent radiometric
  calibration" (contradicted by Dickson) → frozen ViT never built per-frame invariance → confirms
  **C's frozen-ViT ceiling** (DINO photometric aug already in pretraining; signal survives).
- **Bickel & Valantinas 2025 (10.1038/s41467-025-59395-w):** closest analog (global CNN, same 86k
  corpus) ran on **individual source frames off the ASU stream, NOT the mosaic** → the **F template**;
  deduped ~6% overlap dupes; verified no CTX-imaging radiometric bias. Their local-detection task
  tolerated per-frame radiometry; our cross-frame regression does not → F-for-us = per-frame source +
  per-frame normalization.
- **Zhang 2020 (IEEE GRSL) + Pang 2024 (ISPRS S0924271624003277):** canonical RRN = our **A3**
  (per-frame mean+std LSQ over **image overlaps** + local refine) — but the mosaic discarded overlaps
  (it's a partition) → a proper A3 needs source frames = **F**.

**Data-route facts (verified from cached SeamMap):** 26-tile map spans **907 distinct source frames**
(1,371 footprint-polygons; global ≈ 86,571). Each SeamMap polygon carries **`PDS_IMG`** + **`SESE_LINK`**
(ASU `planetview` browse, resolves but **8-bit display-stretched → NOT clean for regression**) + 50
metadata fields (INCIDENCE/EMISSION/sub-solar-az/local-time/…). **⚠️ DATA-EXISTENCE CHECK (2026-06-20):
the SeamMap `PDS_IMG` URLs are STALE — all 10/10 sampled return 404** (PDS Imaging Node reorganized its
tree post-2024; Dickson hedged links were "valid at time of publication"). Frames are real (CTX 5056-px
labels) and EDRs remain in the **permanent archive** (live index pages on USGS
`pdsimage2.wr.usgs.gov/archive/mro-m-ctx-2-edr-l0-v1.0/mrox_NNNN/` + JPL), but **F must resolve
PRODUCT_ID → current URL via `planetarypy`** (Walter's tool; **not in env**) **or a verified USGS
template — NOT `PDS_IMG`.** Radiometry-preserving F input = resolve-EDR → `ctxcal(v0003)` → project.
One-library resolver dependency, not a blocker; definitive "can we pull an EDR" confirmation = first
action of the 10-frame Sherlock timing test. **Robbins "Fully Controlled" mosaic (10.1029/2022EA002443)
ruled out** as a swap: equatorial **±30°N only** (region is 32–46°N) + "cosmetically" corrected.

**Option-menu changes:** **F** reframed from "ensemble over EDRs" → **per-source-frame inference**
(highest ceiling; needs EDR recalibration + retrain head on source embeddings for train/deploy parity).
**NEW A-meta** = per-frame normalize from illumination metadata (≈0 cost, no pixel pass). **Global-compute
(Brian's constraint):** A-meta/A1/D ≈ free marginal; **F-rebuild heaviest at global** → compute pushes
toward input-side + post-hoc D unless F's quality gain justifies re-deriving the global embedding.
**Decision rule unchanged: thermal-ρ adjudicator FIRST.** Still NO decision taken.

## 2026-06-22 — striping: D ruled out (circular); thermal-ρ retired as mitigation adjudicator

**D (post-hoc de-block) RULED OUT (Brian).** D removes a frame's offset by fitting "regional geology
trend + per-frame constant" and subtracting the constant — but separating the artifact offset from real
geology **requires a model of the abundance field, the very unknown the map exists to discover.** D would
assume regional structure, subtract frame-scale deviations from it, then present the result as a
*discovery* of that structure — circular, and especially poisonous for the circum-Chryse megatsunami
boulder deposits (real, spatially-coherent between-frame variation D could erase or manufacture). The
Poisson/gradient-domain form doesn't escape it (it assumes the seam step is artifact; real geology may
genuinely step at a frame boundary). Out.

**Thermal-ρ RETIRED as a mitigation adjudicator.** Two compounding findings: (a) it's underpowered for
this — baseline abundance↔TI ρ ≈ +0.07 (leg-1) and A1 removes only 28% of a modest artifact (eta² 0.011
vs 0.002 null), so the expected Δρ is inside bootstrap noise → thermal-on-A1 would read "ambiguous" and
add nothing the eye doesn't; (b) D was thermal's one remaining legitimate referee job (the zero-skill-cost
option LOIO is blind to), and D is now out. So thermal has no mitigation-refereeing role left. **It
survives only in its original role: an independent validation leg for the *final* map** (abundance vs
thermal inertia, PLAN_RegionalMap §5) — a different question from "which mitigation."

**Consequence — the decision no longer routes through thermal.** It is now a direct cost-vs-need call:
science needs a clean, trustworthy between-frame map for regional discovery → **F** (per-source-frame,
removes the artifact at the radiometry with NO assumption about the abundance field; ±2% floor); a
partially-mitigated map + honest disclosure is acceptable → cheapest geology-agnostic input fix (A1 /
A1-λ / A2 / A3 / A-meta) + **E**-style disclosure. The circularity that kills D *also* argues against
leaning on the capped A-fixes for the final discovery map, so the live decision is essentially **F vs E**
for the science map. Key clarification this surfaced: **A/F are preferable precisely because they fix the
artifact at the source radiometry (known spurious per Dickson/Walter) without assuming the answer; D was
the lone option that required assuming the abundance distribution.** PLAN_StripingArtifact option table +
decision sequence updated; memory `regional_map_rectangular_artifact` updated. Still NO mitigation chosen.

## 2026-07-02 — F de-risk step 1: EDR resolver SOLVED (no planetarypy); 10-frame timing kit built

**Decision (Brian): de-risk F first** — price the per-source-frame pipeline before choosing F vs E
(the fork framed 2026-06-22). Step 1 = can we reliably pull EDRs; step 2 = the 10-frame Sherlock
ISIS timing test.

**EDR resolver: SOLVED, and cheaper than planned.** The stale SeamMap `PDS_IMG` URLs failed only
because the PDS Imaging Node renamed one path segment: `…/data/mro/mars_reconnaissance_orbiter/ctx/…`
→ `…/img/data/mro/ctx/…` on `planetarydata.jpl.nasa.gov`. The frames never moved, and the SeamMap's
own **`VOLUME_ID` + `PRODUCT_ID`** fields fully determine the live URL:
`https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{volume_id_lower}/data/{PRODUCT_ID}.IMG`.
**Verified 12/12** on mission-spanning volumes (mrox_0009…mrox_3355, 19–250 MB, all ranged-GET 206 +
PDS3 labels), then **10/10** on the actual timing frame list. The **ODE REST API**
(oderest.rsl.wustl.edu) returns the identical Product URL and is the documented fallback if JPL
reorganizes again. **`planetarypy` is NOT needed** — the planned one-library dependency is dropped.
(USGS `pdsimage2.wr.usgs.gov/archive/…` — the other 2026-06-20 candidate — is dead for this: 403
Cloudflare without a UA, 404 with one.) Resolver = `src/ctx_edr.py` (`edr_url`, `frame_table`,
`frames_in_crop`) + 4 tests; re-runnable check = `scripts/probes/_f_edr_url_verify.py`.

**Timing-test kit (step 2) ready for Brian on Sherlock.** Frame list =
`reports/f_timing/frame_list.csv` (`scripts/f_edr_frame_list.py --verify`): the **7 frames of the
E8_N44 A1-payoff crop** (so the site doubles as the before/after comparison once per-frame inference
exists) + 3 era-extreme fills; ~2 GB total. Pipeline per frame: EDR → `mroctx2isis` → `spiceinit
web=yes` (no local SPICE kernels) → `ctxcal` (v0003 flat via ISISDATA mro/calibration) →
`ctxevenodd` → `cam2map` to the mosaic grid (`f_equirect.map`: equirect clon 0, sphere 3396190 m,
5 m/px). Kit = `setup_isis_env.sh` (micromamba — ISIS is conda-forge-only and Sherlock discourages
system conda) + `f_timing_test.sh` (per-step timings, per-frame failure isolation, ×907/×86,571
extrapolation) + `run_f_timing.sbatch` (CPU, `normal` partition) + SHERLOCK_RUN.md **Part E**.
NEXT: Brian runs Part E; the timing.csv adjudicates F's cost line in the F-vs-E decision.

## 2026-07-02b — fix: ISIS conda channel

First Sherlock run of `setup_isis_env.sh` failed at env creation: `isis` does **not** exist on
conda-forge — USGS distributes it via the **`usgs-astrogeology`** anaconda channel (conda-forge
supplies only the dependencies). Fixed: `micromamba create -n isis -c usgs-astrogeology -c
conda-forge isis` (USGS channel first). SHERLOCK_RUN Part E wording corrected likewise.

## 2026-07-02c — F timing test: Sherlock gauntlet + web-SPICE verdict (local kernels)

Getting Part E to first light surfaced five environment facts (all fixed in-repo, SHERLOCK_RUN
Part E "failure modes"):
1. **`isis` is NOT on conda-forge** — it lives on the **`usgs-astrogeology`** channel (conda-forge
   supplies deps only).
2. **Login nodes cannot install the env** — micromamba's parallel download AND extract both die
   with EAGAIN under the per-user thread cgroup, even capped to 1 thread; `sh_dev` works.
   setup_isis_env.sh now warns on `*-ln*` hostnames and gates completeness on the `mroctx2isis`
   binary (a crashed transaction leaves a registered-but-empty env that `env list` calls done).
3. **`micromamba activate` must run under `set +u`** — ISIS's bundled activate.d hooks
   (libpdal-core) reference unset vars (`PDAL_DRIVER_PATH`).
4. **Compute nodes DO have outbound internet** (verified `srun curl` → HTTP 206 on the JPL EDR
   tree) — the timing job's downloads run fine in-batch. (The earlier all-`download_fail` run was
   a script bug: `step()` ate its label as the command; fixed fd2e228.)
5. **The ISIS web-SPICE service is USELESS to us: version-pinned.** `spiceinit web=yes` (ISIS
   10.0.0 client, default `apis/ale/v0.9.1/spiceserver` URL) answers **"The SPICE server returned
   incompatible SPICE data"** on every frame — the server serves a payload for a different ISIS
   version (known failure class; only the server's own release is compatible). **Verdict: F runs
   spiceinit `web=no` on LOCAL kernels.** The server's response still *names* the correct kernels
   in the log, so `f_fetch_kernels.sh` harvests them and pulls a **targeted ~1–2 GB** (small
   kernel dirs + ck/spk db files + the specific weekly CKs / psp SPKs + `calibration/**` for
   ctxcal) instead of the 100s-of-GB full mro area. Also fixed: `downloadIsisData` filter syntax
   is `--include="..."` direct (the setup script's earlier `-- --include` form silently failed →
   calibration was missing too). Production-F implication: global runs use a local ISISDATA
   mirror sized to the frame set — no web-service dependency, batch-friendly.

Partial timing already banked from the failed run: download ≈3–12 s, `mroctx2isis` ≈5–21 s per
frame (129–264 MB EDRs) — those two legs alone ≈ 4 h serial for the 907-frame region.

## 2026-07-02d — F timing: base data area was the missing piece (+ two corrections)

The web=no run failed instantly: `No existing files found matching [kernels.????.db] in
[$ISISDATA/base/kernels/lsk]` — the **base area was never (fully) downloaded**: the first setup
run crashed at env activation *before* `downloadIsisData base`, and the re-run gate tested `-d
base/` (present-but-empty passes). Fixed like the env check: gate on the sentinel FILE spiceinit
needs (`base/kernels/lsk/kernels.*.db`); rclone resumes so re-running setup completes the area.
**Corrections to 2026-07-02c:** (a) `downloadIsisData --include` was silently IGNORED by this
build — the "targeted ~1–2 GB" fetch actually mirrored the FULL mro area (**257 GB** on scratch;
works, heavy; noted in f_fetch_kernels.sh, revisit before any fresh-machine rerun); (b) with the
full mirror, kernel availability is moot — the whole local-SPICE story reduces to "have base +
mro areas, run web=no."
