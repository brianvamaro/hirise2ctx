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
| **P2_count** | **12.0** | **0.034** | 3 / 6 | 2 / 26 |
| P4_area | 6.38 | 0.10 | 2 / 6 | 3 / 20 |

**Significant at p < 0.05 under P2_count partition.** Direction holds in
both partitions; magnitude smaller and only marginal under P4_area.

The two ObsIds missing from the spreadsheet (`ESP_017355_2260` in
composition_residual, `ESP_076499_1160` in no_signal) are scored
`transport_indicator = False`, biasing toward the null — the true effect
may be stronger.

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
   annotation correlation). We see it at p = 0.034 under P2.

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
