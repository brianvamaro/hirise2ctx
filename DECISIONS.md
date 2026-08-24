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
- ~~**BoulderNet emits many null-geometry records** at this density~~: rows with a DBF
  entry (score/id) but no polygon. `ESP_017355_2260` is 1.1M rows but only **359,933
  real polygons** (745k null); `ESP_068483_2280` 1.06M → 727k. The priority10 set had
  zero. Confirmed present in the SOURCE shapefiles (not introduced by reproject).
  **Fix:** `src/detections.drop_null_geometries` drops null/empty geoms at Stage 1
  ingest (no-op on v1; detection tests still green) and records `n_polygons_raw` +
  `n_dropped_null_geometry` in the Stage-1 sidecar. True per-image boulder counts span
  9.6k → 727k (≈100–500× v1).
  **⚠ SUPERSEDED 2026-08-06o — BoulderNet emits nothing of the sort. The three affected
  `.shp` files are BYTE-TRUNCATED (incomplete copies, −354/−132/−173 MB); their `.dbf`
  and `.shx` are complete, so GDAL returns every indexed record and the ones whose
  polygon bytes were never copied read as null geometry. Records are stored
  score-descending, so the survivors are the top-scoring prefix and these images' label
  basis is truncated at a per-image confidence floor. 36 of 39 readable exports drop
  exactly ZERO rows — "at this density" was never the mechanism. See the 2026-08-06o
  entry; `inspect_shapefile_integrity` now detects this at ingest.**

**Filter decision (`detection_filters`).** Reprojected equivalent-circle diameters are
large (pooled median 3.4 m, p5 ≈ 1.9 m) → ~~**~0% below the `min_size_m=1.4105` floor**, so
that filter is a no-op~~ (kept, consistent with v1). Scores: 100% ≥ 0.2, 89% ≥ 0.3,
52% ≥ 0.5 — `min_confidence` kept `null`. The denser set is *more* boulders, not
*smaller*.
**⚠ SUPERSEDED 2026-08-06s (R80): "the size filter is a no-op" is FALSE. The shipped
Stage-4 sidecars record 19,757 polygons dropped by `min_size_m` across 12 images. The
pooled p5 ≈ 1.9 m is dominated by the 26 coarse (0.50 m/px) images, where the filter
genuinely is a no-op; on the 12 fine (0.25 m/px) images it drops 0.006–8.26 % each. That
is R03's mechanism operating exactly where the record says nothing is happening, and it
is why the mixed size floor is a real convention rather than an inert setting.**
**⚠ CAVEAT 2026-08-06o (R58): those score percentages were computed over the PRE-drop
population — i.e. including the rows the pipeline then deletes. On the post-drop
population they read 100 / 97.4 / 77.1 %. The `min_confidence: null` decision is
unaffected, but the statistic quoted here is the exact one that would have exposed R23,
computed on the wrong population.**

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

## 2026-06-16 -- Calibration Stage 2b/2c CLOSED: reweighting dominated, label-noise ~~harmful~~ [NULL, amended 2026-08-06o] -> ceiling is the data

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
  Low-confidence detections are REAL boulders, not removable noise.
  **⚠ SUPERSEDED 2026-08-06 (R56) — this verdict is a two-factor artefact; see the
  2026-08-06 entry "R56 re-scored". The Δ −0.021 varied both the model and the target;
  82 % of it is the target moving. On a common target conf>=0.5 is a NULL and
  "monotonically" is false in the probe's own scorecard.** (Built with
  config_v2.yaml / hirise_40_vclaire.csv -- NOT the v1 config.yaml/priority10 manifest;
  the regen is CPU-only and was run concurrently with the GPU reweight via `--regen-only`.)
- **STAGE 2 COMPLETE -- the whole levers table:** L1 cheap swaps = wash; L1 distributional
  heads = wash; L1+L2 reweighting = dominated; L2 label-noise = ~~harmful~~ **NULL at
  conf>=0.5 (amended 2026-08-06, R56 -- see the 2026-08-06o entry)**; L2 coarser scale
  = directional only (p=0.19). **No in-cohort retraining lever beats mlp_reg+qmatch on
  ranking.** The ~0.43 per-image ceiling is the 5 m/px CTX magnitude floor, ~~confirmed five
  ways~~ **on the frozen Fang-ViT/GeM-96/S=32 embedding -- all five "ways" hold that
  representation fixed (R55), and the label-noise one is now withdrawn as evidence.** **Path forward is NOT a better model:** ship Stage 1 (productize qmatch + isotonic
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

## 2026-07-03 — F timing test COMPLETE: 10/10 frames end-to-end; F is priced

`run_f_timing.sbatch` (job 32611907, `normal` partition, 4 CPU) ran all 10 frames through
EDR → `mroctx2isis` → `spiceinit web=no` (local kernels) → `ctxcal` → `ctxevenodd` → `cam2map`
(5 m/px mosaic grid) with **zero failures** — the first complete end-to-end execution of the F
pipeline. `reports/f_timing/timing.csv` (committed) is the price sheet:

- **Per frame: mean 1323 s ≈ 22.0 min, median ≈ 20.3 min** (range 13.2–33.8 min; scales with
  EDR size, 119–264 MB in-sample). **cam2map is 96.6% of the cost** (762–1984 s); everything
  else (download+import+spiceinit+ctxcal+evenodd) totals ~45–75 s/frame.
- **Regional (907 frames): ≈ 333 CPU-h serial** → embarrassingly parallel per-frame: ~10.4 h on
  a 32-task job array, ~5.2 h on 64. CPU-only (`normal` partition — no GPU queue contention).
- **Global (86,571 frames): ≈ 31,800 CPU-h ≈ 3.6 CPU-years** → needs ~500 concurrent tasks for
  ~2.7 days wall. Heavy but Bickel-precedent scale; a later decision.
- **Storage is the real regional constraint: projected cubes avg ~3.5 GB/frame** (32-bit + NULL
  padding from the rotated footprint) → **~3.2 TB** if all 907 are kept on scratch. Levers:
  stream (project → embed → delete), 16-bit output attribute (~half), or footprint-crop. Global
  (~300 TB) MUST stream.
- Remaining F line-items (already understood): ViT embed of projected frames (GPU, tens of h
  regional incl. ~2–3× overlap redundancy), cohort re-embed from source frames + head re-bake +
  LOIO re-gate (the A1-cycle machinery), overlap composite/dedup à la Bickel.

**The ISIS leg was F's last unknown → the F-vs-E call is now a pure numbers decision (Brian).**

## 2026-07-04 — F pilot leg A (GPU): FAIL — mosaic-trained head is out-of-distribution on calibrated frames

`scripts/f_pilot_crop.py` on 7 E8_N44 crop frames, 4 I/F→uint8 mappings, 2 heads (base + a1).
Full output: `reports/figures/f_pilot_eta2_summary.csv`, `f_pilot_overlap_pairs.csv`,
`f_pilot_{affine,lambert,minnaert,perframe}.png`. Baselines: mosaic raw **0.196** / A1 **0.141**.
Target: ≲ 0.03 (block-free).

**All 4 mappings FAIL — every result is worse than the raw mosaic baseline:**

| mapping | best eta² (median composite, base head) | vs raw mosaic |
|---|---|---|
| perframe | **0.233** | +19% worse |
| affine | 0.282 | +44% worse |
| minnaert | 0.319 | +63% worse |
| lambert | 0.346 | +77% worse |

Choropleth panels confirm visually: strong rectangular blocks present in all mappings;
perframe blocks are narrower in range but still obvious.

**Interpretation — this is train/deploy mismatch, not a fundamental F failure:**
- The mosaic-trained head was never exposed to calibrated-frame embeddings — it maps them to a
  different part of embedding space and makes predictions that vary more between frames, not less.
- Perframe (which most resembles the mosaic's per-frame stretch) is best, consistent with the
  A0 finding that illumination is the dominant cross-frame signal; lambert (overcorrects cos i)
  is worst.
- The overlap I/F agreement is 10.2% median |ratio−1| across all pairs (matches A0); prediction
  overlap disagreement 13–16% (perframe/base) — the head amplifies I/F differences.
- Minnaert k fitted from 7 frames = **0.694** (vs 0.66 in A0 from the same 7 frames with fewer
  valid pixels; consistent range).

**Conclusion: leg A cannot gate F** — you cannot evaluate whether calibrated frames kill the
blocks without a head trained on calibrated-frame embeddings. The only real test is **leg B**
(project ~40–80 cohort frames on Sherlock, re-embed with perframe normalization, re-bake head,
LOIO gate). Decision on whether to proceed to leg B deferred to Brian.

## 2026-07-09 — PHASE 2 H2 (embedding nuisance-subspace removal): FAIL — linear subspace is NOT the artifact axis

Second PLAN_StripingArtifact PHASE 2 item, stacked on the H1 centered store. **Premise:** two
frames imaging the SAME ground should embed to the same point, so a co-located tile's embedding
difference `d = e_i − e_j` is pure frame-nuisance (radiometry/illumination/epoch, zero geology).
Collect many `d`, PCA them, and the top-k directions are a nuisance basis `N`; project it out of
every embedding (`e ← e − (e·N)Nᵀ`), retrain the head, re-gate. Implemented as an optional
`nuisance_basis` pre-transform baked into `DeployableHead` (applied identically in fit/predict,
persisted with the weights → travels to deploy via `load`, so the η² path needs no extra
plumbing — the H1 train/deploy-parity lesson made structural). +4 unit tests (suite **366**).

- **Basis** (`scripts/f_h2_nuisance.py`, `reports/f_leg_b/h2_nuisance_basis.npz`): built from the
  **28 multi-crop TRAINING obs** (not the 7 pilot frames the η² test scores — so `N` is learned
  independently of the artifact test set, no circularity). Each source frame embedded separately
  under `minnaert_center`; **47 within-obs frame pairs, 174 963 co-located difference vectors**.
  The nuisance really is low-rank: **top-4 directions = 84.2%** of between-frame embedding-difference
  variance, top-16 = 89.7%, top-64 = 94.2%.
- **η² sweep** (`scripts/f_h2_eta2.py`, `f_h2_eta2_summary.csv`, `f_h2_eta2_choropleth.png`; pilot
  frames embedded once, heads `models/deployable_f_h2_k{4,16,64}/86c51a5dca220f63`). The k=0 center
  head reproduces H1 exactly (partition 0.128 / median 0.081 / overlap 0.073 → parity confirmed):

  | k | skill Δ med AUC (vs mosaic) | gate | partition η² | median η² | pred overlap |Δp| |
  |---|---|---|---|---|---|
  | 0 (H1) | −0.0139 | PASS | 0.128 | 0.081 | **0.073** |
  | 4 | **−0.0026** | PASS | 0.110 | 0.072 | 0.091 |
  | 16 | −0.0510 | FAIL | 0.149 | 0.080 | 0.124 |
  | 64 | −0.1223 | FAIL | 0.131 | 0.040 | 0.109 |

- **Verdict = FAIL to reopen; refuted as the lever.** Two decisive facts: (1) **even k=64 — which
  removes 94% of the between-frame embedding-difference variance — leaves partition η² at 0.131**,
  essentially H1's 0.128. The frame-block artifact in *predictions* is NOT aligned with the
  directions of largest between-frame embedding *difference*; removing "where embeddings differ most
  across frames" does not remove "where predictions differ by frame." (2) **pred-overlap |Δp| rises
  at every k** (0.073 → 0.09–0.12) — the opposite of the goal; the retrained head, deprived of those
  directions, re-keys on residual frame-correlated structure and co-located predictions disagree
  *more*. Skill collapses monotonically with k (−0.003 → −0.051 → −0.122): large-k projection eats
  geology-informative directions with no η² payoff. Only k=4 survives the skill gate (Δ −0.0026,
  even a hair better than H1, pooled PR +0.039) and gives a marginal partition drop 0.128→0.110, but
  it is nowhere near the η² ≲ 0.05 bar and *worsens* overlap → not adopted. Choropleth: the per-frame
  blocks (incl. the dark **F02** block) persist visually at all k.
- **Interpretation:** the between-frame embedding variance is diffusely entangled with geology across
  many dims (the frozen ViT mixes them nonlinearly), not confined to a low-rank subspace separable by
  a fixed linear projection. This specifically motivates the remaining docket: **H3** (consistency-
  regularized head — optimize prediction agreement on overlaps *directly* in the loss, in-head/
  nonlinear, since a fixed linear subspace is the wrong instrument) and **H4** (overlap-constrained
  leveling — the blocks persist as per-frame level offsets in *prediction* space, esp. F02, which is
  exactly what H4 removes post-hoc). H1 stays the operating baseline. Logs:
  `reports/f_leg_b/h2_{nuisance,eta2,pytest}.log`, `h2_loio` via
  `f_leg_b_loio_summary_minnaert_center_h2_k{4,16,64}.csv`.

## 2026-07-09b — PHASE 2 H3 (consistency-regularized head): FAIL to reopen — artifact removal and skill sit on ONE monotone axis

Third PLAN_StripingArtifact PHASE 2 item, stacked on the H1 centered store — the in-head/nonlinear
instrument H2's failure pointed to. **Premise:** H2 showed the artifact is not a fixed low-rank
*linear* subspace, so instead of projecting embeddings, optimize prediction *agreement* on the
overlaps **directly**. Add a consistency penalty to the head's training loss:
`loss += λ·mean((sigmoid(net(e_i)) − sigmoid(net(e_j)))²)` over co-located overlap tile pairs
(same ground seen through two frames → any predicted-P(rich) difference is artifact by
construction). Implemented as `lambda_consistency` on `MLPClassifierHead`/`DeployableHead`
(random pair minibatch per step; scaled by the fitted train scaler; λ=0 is bit-for-bit the
un-regularized fit). Pairs from `scripts/f_h3_pairs.py` = the SAME 28 multi-crop TRAINING obs as
H2 (47 frame-pairs / 174 963 co-located tile-pairs, subsampled to 40 000; independent of the 7
pilot frames the η² test scores → no circularity). +3 unit tests (suite **369**).

- **η² sweep** (`scripts/f_h2_eta2.py` reused, heads `models/deployable_f_h3_lam{3,10,30,100}`) +
  **skill gate** (`scripts/f_leg_b_loio.py --lambda-consistency`, F-store only, mosaic baseline
  0.786) → Pareto (`scripts/f_h3_pareto.py`, `f_h3_pareto.{csv,png}`):

  | λ | partition η² | median η² | pred overlap |Δp| | skill Δ med AUC | gate | pooled PR-AUC |
  |---|---|---|---|---|---|---|
  | 0 (H1) | 0.128 | 0.081 | 0.074 | −0.0139 | PASS | 0.796 |
  | 3 | 0.126 | 0.068 | 0.057 | −0.0174 | PASS | 0.756 |
  | 10 | 0.102 | 0.052 | 0.048 | −0.0364 | FAIL | 0.702 |
  | 30 | 0.093 | 0.064 | 0.041 | −0.0401 | FAIL | 0.646 |
  | 100 | **0.035** | **0.036** | **0.031** | −0.0771 | FAIL | 0.621 |

- **Verdict = FAIL to reopen; no Pareto point clears both gates.** Unlike H2, the penalty DOES work
  on η²: it drops **monotonically** with λ, and at λ=100 partition η² **0.035 crosses the 0.05
  reopening bar** with pred-overlap 0.031 < input I/F 0.102 (amplification genuinely killed, not
  masked). But artifact-reduction and skill lie on **one monotone axis** — the penalty flattens the
  frame blocks by **compressing the head's global dynamic range** (in-sample p|pos 0.785→0.631,
  p|neg 0.207→0.396; median composite goes uniformly bright in the choropleth), so skill falls in
  lockstep (pooled PR-AUC 0.796→0.621). The Pareto (`f_h3_pareto.png`) is a straight monotone
  trade-off: the skill gate (−0.02) is crossed between λ=3 and λ=10; the η² bar (0.05) only at
  λ=100. **The two acceptable regions never overlap** — the knee is λ≈3, where skill barely passes
  (Δ −0.0174) but partition η² is essentially unchanged (0.128→0.126). To reach η²≤0.05 costs
  Δ −0.077.
- **Interpretation (H2 + H3 together):** the per-frame block variance in *predictions* is not
  separable from geological signal by ANY instrument tried — neither a fixed linear subspace of the
  embedding (H2) nor an in-head nonlinear consistency objective (H3). The frozen ViT entangles
  frame-radiometry with texture so tightly that suppressing cross-frame prediction disagreement
  necessarily suppresses genuine cross-scene ranking. **H1 (per-frame log-median centering, η²
  0.081) stays the operating baseline.** The one untested lever is **H4** — overlap-constrained
  *post-hoc* per-frame additive leveling of predictions (least-squares offsets on the frame-overlap
  graph), which removes the residual F02-class level offset WITHOUT touching within-frame ranking,
  so it cannot collapse dynamic range the way a training-time penalty does. Logs:
  `reports/f_leg_b/h3_loio_lam{3,10,30,100}.log`; η² sweep preserved as
  `f_h2_eta2_summary_h3.csv` / `f_h2_eta2_choropleth_h3.png` (`f_h2_eta2.py` writes the generic
  name, which stays H2's evidence). 169M pairs npz gitignored (recomputable via `f_h3_pairs.py`).
- **Brian ruling (2026-07-09, for H4's design):** **combined levers count toward the reopening
  bar** — η² ≲ 0.05 at skill ≥ −0.02 reached by a *stack* (e.g. H1 centering + H4 post-hoc
  leveling) reopens the 907-frame build; no single lever needs to hit it alone. (Resolves the open
  question staged in `PLAN_H4_Leveling.md` §6.)

## 2026-07-09b(H4) — PHASE 2 H4 (overlap-constrained post-hoc leveling): PILOT PASS mechanically — first lever to reach the bar without collapsing skill; trend-guard caveat

Last untested PHASE-2 lever, and the first to succeed on η² *without* the H3 skill-collapse.
`scripts/f_h4_level.py` reuses the H2/H3 per-frame embedding cache: the H1 head
(`models/deployable_f_center/86c51a5dca220f63`) predicts the 7 aligned E8_N44 crops → 7 P(rich)
rasters on the shared grid (15 overlap edges ≥200 co-located tiles). Solve per-frame **additive
logit offsets** o_f minimizing Σ_edges Σ_colocated [(ℓ_i+o_i)−(ℓ_j+o_j)]² + λ·Σo_f² (exact per-edge
sufficient statistic (δ̄_ij, W_ij) → 7-unknown weighted min-norm LS; gauge median(o)=0). λ picked by
**leave-one-edge-out CV** — the non-circular §3.2 check (post-H4 η² alone would be circular, exactly
what killed option D).

- **Results (λ*=300 by held-out CV; near-flat across λ ∈ {0…1000}):**

  | metric | unleveled (H1) | H4 full offsets | reopening bar |
  |---|---|---|---|
  | partition η² | 0.128 | **0.0505** (λ=1000 → 0.0466) | ≲ 0.05 |
  | median η² | 0.081 | 0.052 | — |
  | **held-out edge-CV \|Δp\|** | 0.074 | **0.035** (halved, FLAT across λ) | drop below 0.073 |
  | in-sample overlap \|Δp\| | 0.074 | 0.034 | — |

- **Verdict = PILOT PASS (mechanically).** The decisive non-circular gate — leave-one-edge-out
  held-out |Δp| — **halves** (0.074→0.035) and is essentially flat across λ, so the offsets
  **generalize to unseen overlaps** rather than memorizing their own edges (the failure mode §3.2
  exists to catch). Partition η² crosses the ≲0.05 bar. **Skill is preserved BY CONSTRUCTION** — a
  per-frame additive logit offset cannot change within-frame ranking, so H3's dynamic-range-collapse
  mechanism is structurally impossible here (per-image AUC provably ~unchanged; pooled leg-B
  confirmation still pending). This is the first PHASE-2 lever to move η² to the bar without killing
  skill: H2 couldn't move η² at all, H3 moved it only by collapsing dynamic range, **H4 removes the
  co-located disagreement on the axis orthogonal to ranking.** Offset signs are physically sane —
  **F02** (known −2.23σ dark, 2014 epoch, over-predicts) gets the most negative offset (−1.54),
  correctly pushing it down.
- **⚠ Caveat — the trend guard fires (§2).** 58% of the offset variance is a smooth lon/lat plane;
  applying only the **residual** offsets (conservatively treating the plane as possible geology)
  leaves partition η² **0.0595** and barely moves |Δp| (0.074→0.070) — i.e. most of the η² reduction
  rides on the smooth component. On only 7 frames a 3-param plane explaining ~58% is near the chance
  level (~50%), and it **demonstrably mis-attributes F02's genuine per-frame 2014 radiometric offset
  to a spatial trend** (trend_fitted −1.04 of F02's −1.54 total), so residual-only *understates* H4.
  A clean artifact-vs-regional-gradient separation is **intrinsically underpowered at pilot scale**
  and resolves only on the dense 907-frame graph (many frames per unit area → a real gradient becomes
  identifiable).
- **Net:** H4 works mechanically; H1+H4 is the first stack to reach the reopening bar (Brian's
  combined-lever ruling). The reopening call now needs (a) the leg-B pooled-skill instruments
  (§3.1 — LOIO rerun with training-obs offsets applied; no cached preds, so it's a real rerun) and
  the THEMIS ρ leg, and (b) a Brian decision on whether the 7-frame trend-guard ambiguity blocks
  reopening or defers to the build. Artifacts: `f_h4_leveling_summary.csv`, `f_h4_offsets.csv`,
  `f_h4_leveling_choropleth.png`, `f_h4_offset_scatter.png`. `PLAN_H4_Leveling.md` §6 verdict folded.
  Notebook 28 §7 (H3) + §8 (H4, with a plain-language trend-guard explainer) document the arc.
- **Brian rulings (2026-07-09b, after seeing notebook 28 §8):** (1) **trust the FULL offsets** — the
  7-frame trend guard is underpowered (~58% ≈ chance for a 3-param plane over 7 points; it mis-blames
  F02's per-frame offset on position), so the pilot verdict is the full-offset η² 0.0505; the clean
  smooth-artifact-vs-regional-gradient split is deferred to the dense 907-frame build. (2) **Next =
  run the leg-B pooled-skill instruments (§3.1)** before any reopening call: solve offsets on the
  47-training-obs frame-overlap graph, apply them to the LOIO per-tile predictions via
  `obs_frame_map.csv`, recompute pooled pr_auc@1e-2 / Spearman / prec@5% (no presence AUC), then the
  THEMIS-ρ leg.

## 2026-07-09b(H4-legB) — PHASE 2 H4 leg-B skill instrument: PASS — leveling preserves skill on real LOIO predictions

The §3.1 empirical follow-up to the H4 pilot (`scripts/f_h4_legb.py`). Per-image AUC is provably
blind to a per-frame additive offset, so the instruments are **pooled** metrics (pooled pr_auc@1e-2,
precision@5%; no presence AUC). Per-frame offsets solved on the **28 multi-crop training-obs overlap
graph** (H1 head on the cached `h2_frame_emb` per-frame embeddings; same solver, λ=300), then applied
at obs level to the `f_leg_b_loio_preds_minnaert_center.csv` composite predictions.

| pipeline | pooled PR-AUC | prec@5% | median img AUC |
|---|---|---|---|
| baseline (mosaic) | 0.7668 | 0.913 | 0.786 |
| H1 (F, unleveled) | 0.7964 | 0.972 | 0.7722 |
| H1+H4 (F, leveled) | 0.7860 | 0.968 | 0.7722 |

- **Verdict = PASS.** Δ pooled PR-AUC (H1+H4 − H1) = **−0.0104** (within the −0.02 gate); H1+H4 stays
  **+0.0192 above the mosaic baseline**; **Δ per-image AUC = exactly 0.0000** — the "skill safe by
  construction" claim confirmed on real predictions (an obs-level shift can't reorder tiles within an
  image). prec@5% 0.972→0.968.
- **Graph structure (⚠ context):** the training-obs frame graph is **fragmented — 58 frames / 47
  overlap edges / 21 connected components** (mostly per-obs cliques; only 11 frames span >1 obs), so
  at leg-B scale the leveling is mostly *within-obs* and the cross-obs skill effect is modest by
  construction. The 907-frame build graph would be far more connected → connectivity is a build-prep
  verify item (PLAN §5).
- **Approximation (honest):** the leg-B store embeds one *composite* window per obs, so the offset is
  applied at obs level (exact for single-frame obs; mean-of-frames for composites). The deploy-
  faithful per-frame-inference LOIO is a build-scale rebuild, deferred.
- **Net:** H4 clears both PHASE-2 gates on the pilot AND now the empirical skill guard. Remaining
  before a reopening call: the **THEMIS-ρ** leg on the leveled pilot map (last §3.1 check), and the
  trend-guard smooth/artifact separation (a build-time item per Brian's ruling). Artifacts:
  `f_h4_legb_summary.csv`, `f_h4_legb_offsets.csv`; notebook 28 §9.

## 2026-07-11 — H4 build-prep verify item 1: the 907-frame overlap graph is FULLY CONNECTED (one gauge)

PLAN_H4_Leveling §5 pre-build check A (`scripts/f_h4_buildprep.py`, log preserved at
`reports/f_leg_b/h4_buildprep_graph.log`): dissolved the SeamMap footprints for all 26 regional
tiles per PRODUCT_ID, unioned across tiles, built the frame-adjacency graph.

- **VERIFIED: 907 unique frames from 1,371 per-tile footprints** — exactly the counts the plan
  carried from DECISIONS 2026-07-02/03, now independently reproduced from the SeamMaps themselves.
- **VERIFIED: the graph is a SINGLE connected component at the strictest criterion** (buffer 0 =
  shared partition boundaries only): **3,584 edges, largest component 907/907 (100%), 0 isolated
  frames, median degree 7**. Buffering only adds edges, so connectivity holds a fortiori at any
  tolerance. **One gauge for the whole region** — no disconnected-component flagging (PLAN §2)
  needed; every frame's H4 offset is identifiable from overlap constraints alone.
- **Ops lesson (why the first run burned 5.7 CPU-h):** the dissolved SeamMap multipolygons are
  pixel-resolution; `buffer()` on the raw geometries is ~quadratic in vertices and stalled the
  250 m/1000 m sensitivity sweeps (which the buffer-0 result had already made moot). Fixed in the
  script: `simplify(50 m)` before buffering + incremental CSV writes + cheap checks run first.
  Job was killed after the decisive row (Brian: waiting for a BoulderNet run to free resources
  before any re-run).
- **Verify item 2 (H1 centering-statistic stability: per-crop vs per-frame median) still PENDING**
  — part B never ran; it's minutes of local file reads once resources free up.

## 2026-07-13 — Planning session: PLAN_FBuild opened (907-frame build now executable-planned); validation-leg relaunch staged; ESP_053989 "moot" ruling reversed; housekeeping

Project review (Brian asked "anything that needs planning?") → three gaps closed, no computation run
(CPU still held by BoulderNet):

1. **[PLAN_FBuild.md](PLAN_FBuild.md) opened (DRAFT)** — PLAN_H4_Leveling §5 ("pre-planning only")
   promoted to an executable build plan: §0 reopening-call checklist (part B / ESP_053989 /
   THEMIS-ρ / Brian), Stage A ISIS 907 frames (≈333 CPU-h, retention ≤1.5 TB peak), Stage B
   per-frame `minnaert_center` inference (~25–40 GPU-h estimate, sizing probe V1; centering
   statistic keyed to part B's answer with both variants pre-declared), Stage C H4 solve on the
   3,584-edge graph **with the trend-guard method pre-declared before any offsets are seen**
   (spatial block-permutation significance + metadata-vs-geology attribution — at 907 frames a
   3-param plane has chance R² ~0.3%, so the test the 7-frame pilot couldn't run becomes powered),
   Stage D mean-of-leveled-logits composite + H6 provenance layers + 5 pre-declared acceptance
   gates (incl. the deploy-faithful per-frame LOIO spot-check the leg-B approximation deferred).
   Open questions for Brian in §7 (Stage-B venue, scratch retention, trend-guard ambiguous branch,
   tile order).
2. **ESP_053989 moot-reversal (build-prep item P4).** DECISIONS 2026-07-05c declared the
   minnaert-inversion fix moot *because F was closed*; H1+H4 reaching the reopening bar voids that
   premise. Whether H1's centering rescued the image (its failure was pooled-stretch floor clipping,
   which centering plausibly fixes) is UNVERIFIED — check its per-image AUC in
   `f_leg_b_loio_preds_minnaert_center.csv` when CPU frees; if still inverted, diagnose before the
   build (candidates in 2026-07-05b caveat 1). Recorded in PLAN_StripingArtifact PHASE-2 state +
   PLAN_FBuild §0.
3. **PLAN_RegionalMap 2026-07-13 refresh** — the parked validation legs are staged to relaunch on
   the F-build map: leg-1 gate stays "not degraded" (improvement reported observationally only —
   the 2026-06-22 thermal-referee retirement stands); leg-2 = THEMIS Fergason TI multi-tile fetch
   (still TODO) with an open decision on a physical TES TI + DCI dust-mask source; legs 3–5
   unchanged (leg 4 in corrected LOIO form).
4. **Housekeeping:** H4 milestone committed (`e94b7df`, 21 files — pilot + leg-B + build-prep A had
   been sitting uncommitted); ROADMAP now indexes PLAN_H4_Leveling + PLAN_FBuild as their own rows;
   README status/next-priorities brought current (was "H3 in flight / H4 staged"); duplicate scratch
   log `reports/figures/_h4_buildprep.log` removed (curated copy lives at
   `reports/f_leg_b/h4_buildprep_graph.log`); branch pushed (was 7 commits ahead); memory index
   trimmed under its size limit.

## 2026-07-14 — Reopening-call checklist cleared: build-prep part B (P2) + ESP_053989 (P4) + THEMIS-ρ (P3) all PASS

BoulderNet freed the machine (CPU ~30%, GPU idle), so the three PENDING §0 items ran (all local,
minutes). **P1–P4 are now green; only P5 (Brian's call) remains.**

- **P2 — H1 centering-statistic stability = STABLE** (`f_h4_buildprep.py` part B,
  `reports/figures/f_h4_buildprep_median_stability.csv`). Within-frame ln-median drift is small vs
  the between-frame spread H1 removes: pilot 3×3 worst **0.0397 (15%)**, cross-crop worst **0.0564
  (22%)** against a between-frame yardstick of 0.256 (H1 log-stretch width 0.285). Worst cross-crop
  pair = G18_025425/ESP_071699. Verdict STABLE (worst < 0.25·between) ⇒ the pre-declared PLAN_FBuild
  §3 branch resolves to the **per-frame median** centering statistic (not the per-lat-band fallback).
- **P4 — ESP_053989 recovered** (`_tmp` pandas check on `f_leg_b_loio_preds_minnaert_center.csv`,
  per-store). Under `minnaert_center` its per-image AUC is **0.884** (mosaic baseline 0.873) — the
  ~0.2 inversion that the 2026-07-05c "moot" ruling assumed is **gone** (H1 centering fixed the
  stretch-floor clipping). The moot-reversal resolves cleanly: **no separate inversion fix needed
  before the build.** (Context, not a blocker: the centered store's worst image is now ESP_055253
  at 0.355 — already the weakest at 0.465 on the mosaic — and centering degrades a few images
  [ESP_064510 −0.235, ESP_069669 −0.185], moving pooled median per-image AUC 0.786→0.772 = the
  known H1 skill Δ −0.0139; these are pre-existing weak scenes, not a new failure.)
- **P3 — THEMIS night-IR ρ not degraded = PASS** (`scripts/f_h4_themis.py`,
  `reports/figures/f_h4_themis_rho.{csv,png}`). Leg-1 harness rerun on the H4-leveled pilot map
  (committed offsets, λ*=300; embeddings cached; THEMIS from `cache_v2/validation/`, reprojected onto
  the pilot 160 m CTX grid). Median composite ρ(P(rich), THEMIS) **0.068 → 0.137 (Δ +0.069)**;
  partition 0.066 → 0.147 (Δ +0.081) over ~218k co-valid tiles. Leveling STRENGTHENS the thermal
  correlation, doesn't degrade it. The unleveled ρ **0.068 matches the regional leg-1 +0.07**,
  validating the harness. (THEMIS mosaic stores scaled brightness-temp DN 1–255, not K; Spearman is
  rank-based so unaffected.)
- **Adversarial review of the whole H1+H4 reopening case** (6 skeptics + adjudicator; workflow
  `fbuild-reopening-adversarial-review`). **Verdict = YELLOW — reopen with guards.** All 5 completed
  lenses independently returned `reopen_with_guard` (none a clean blocker, none a clean green). One
  root cause compounds four of them: the η²=0.0505 PASS holds only under **full** offsets —
  residual-only is 0.0595 (FAILS) — and at n=7 the pilot cannot separate "removed an artifact" from
  "absorbed a real regional gradient." Sharpest new finding (F02/magnitude lens): **corr(offset,
  frame-mean P(rich)) = −0.941**, corr(offset, radiometric-z) = +0.52 (wrong sign) → H4 is flattening
  between-frame means wholesale, a direct risk to a contrast map. Offset energy 85% in 3 of 7 frames
  (J02 +1.71 = 34.5%, F02 −1.54 = 27.9%, P22_009549 −1.39 = 22.8%); only F02 physically vetted, and
  J02 (largest) sits on a radiometrically-NORMAL frame (z −0.19). Two framing objections (0.0505 vs
  0.05; "combined levers count") were checked and **survive** — not real threats.
- **Leave-one-FRAME-out CV run in response** (`scripts/f_h4_lofo.py`, `reports/figures/f_h4_lofo.csv`)
  — the honest generalization instrument the plan never pre-declared (edge-CV is near in-sample on
  the over-determined graph). Predict each held-out frame's offset from its overlaps with the other 6,
  score its held-out seam agreement + η². Result: **held-out |Δp| generalizes cleanly** (median
  0.0365, vs unleveled 0.0738, ≈ the in-sample 0.035), but **η² generalization is MARGINAL** — median
  LOFO η² 0.049, worst-frame 0.0634 (B03 held out), just over a 0.06 guard, driven by the two
  least-pinned large offsets (J02 pred-err 0.49 on only 3 edges; B03 0.33). Cross-check: drop-frame η²
  reproduces the review's probe exactly (B03 0.0804 / F02 0.0802 / P21 0.0857). Read: the build's
  median-degree-7 graph (vs pilot ~4) should pin these offsets better, so LOFO supports reopen-with-
  guards; J02 is the frame to vet.
- **Deploy-faithful per-frame cross-frame skill probe — PASS** (`scripts/f_h4_legb_perframe.py`,
  `reports/figures/f_h4_legb_perframe.csv`; Brian chose "one more cheap probe first"). Converts
  PLAN_FBuild §5 gate #5 to pre-spend: builds the build's true Stage-D composite (**mean of leveled
  logits**) from the cached per-frame logits + per-tile labels (fractional_area>1e-2, scale_idx==2),
  vs leg-B's obs-level mean-shift, over the 28 multi-frame training obs (113,475 tiles). **Δ_deploy
  (leveled − unleveled) = −0.0007** (gate ≥ −0.02) — the per-frame build composite preserves skill.
  **approx_err (leveled − leg-B obs-level) = +0.0000** across all 28 obs (each with divergent frame
  offsets, std>0.05) → leg-B's −0.0104 was a FAITHFUL measure, not the understated lower bound the
  review feared. (Absolute pr_auc 0.920 is in-sample-optimistic — these obs trained the fixed head —
  but Δ_deploy and approx_err cancel the head and are the gate quantities.) **Net: the skill-instrument
  lens concern is retired pre-build; the generalization concern is narrowed to the two under-pinned
  large offsets (J02, B03) that the dense build graph is expected to fix. Reopening posture:
  reopen-with-guards, both pre-spend probes green.**

## 2026-07-23 — Adversarial code+methodology audit of H1+H4: methods SOUND, one genuine build-risk (within-frame incidence ramp) + doc/hygiene items folded into PLAN_FBuild

Brian asked for an independent read on whether H1/H4 are reasonable / have mistakes. Ran a
code-review workflow (`h1-h4-method-audit`, 14 agents, 3 lenses — numerical / statistical / physical
— reading `f_pilot_crop.py` H1 mapping, `f_h4_level.py` solver, `f_h2_eta2.py` scoring, `src/striping.py`
eta2; every non-nit finding adversarially adjudicated against the actual code + re-derived math).
**Verdict: the levers are correct and the pilot conclusion survives scrutiny; 17/18 findings resolve
`expected-by-design`, 1 is a genuine build-scale risk. No pilot number moved; reopening posture
unchanged (reopen-with-guards).**

- **Confirmed correct.** The H4 solver was checked against a brute-force minimizer of the literal
  per-tile objective (match <1e-2 at multiple λ; ANOVA gradient = finite differences). **Sign is
  right** — an over-predicting frame is pushed down; F02's −1.54 is consistent, and its offset is
  predicted from the *other* frames' overlaps to within 0.004 (a genuine per-frame level, not a
  self-fit). The per-edge sufficient statistic (δ̄=mean(ℓ_j−ℓ_i), W=tile count) and gauge
  (median-subtraction) are exact; Tikhonov barely bites (edge weights ≫ λ). H1 is a defensible
  near-nadir photometric normalization; the retrained-head + re-applied-mapping is a **legitimate
  train/deploy match, not a leak** (η² is scored on held-out frame geometry, not labels); circularity
  is genuinely handled (held-out edge-CV uses edges never in the fit).

- **THE ONE GENUINE RISK — within-frame incidence ramp (build-only).** The Minnaert correction uses
  **one incidence scalar per frame** (`cos^k(i)`). The pilot crop is a ~1.3° window (ramp <~0.8%, so
  the pilot PASS is insulated), but full 907 frames span 3–4° latitude → a real ~2% top-to-bottom I/F
  ramp. **Both H1 and H4 are per-frame DC operators and cannot touch a within-frame gradient**, and
  partition η² sees only *between*-frame variance — so the ramp would render as a smooth abundance
  gradient inside each frame block and **never register in the reopening metric**. Same hazard class
  as the original artifact (small radiometric variation × amplifying embedder). **Folded into
  PLAN_FBuild:** §3 per-row `cos^k(i(lat))` divisor (from each frame's N/S incidence endpoints, or
  ISIS `phocube`) + **V5** gate (measure residual ramp on a few full frames; <~0.5% → scalar OK,
  ≥~1% → switch to per-row before the array).

- **Doc / hygiene items (all `expected-by-design`; forward-looking fixes made):**
  1. **k = 0.580 vs pilot-fit 0.694** — harmless: a global-k error is a per-frame constant that
     per-frame median centering removes *exactly* (`d/median(d)` cancels `cos(i_f)^(k*−k)`). Added
     **V6** (re-fit k on 907 + η² sensitivity over k∈[0.55,0.70]).
  2. **"Skill safe by construction"** rescoped to **within-image ranking** (per-image AUC Δ=0); the
     cross-image/pooled effect is the empirical −0.0104 (gate 5). README line scoped; PLAN gate 5
     note added. (DECISIONS 2026-07-09b already carried the "pooled confirmation pending" qualifier.)
  3. **trend_guard `frac>0.5` "SIGNIFICANT" print is NOT a significance test** — R²~Beta(1,2) at n=7
     fires 25% under noise (observed 0.58 → p≈0.17); non-load-bearing (λ/offsets don't depend on it).
     PLAN §4.2 notes it; build uses the pre-declared permutation p-value.
  4. **Partition η² is the headline metric, not median-composite** (the median blends across seams and
     is scored against single-owner labels → deflates; they converge after leveling). PLAN gate 1
     states the convention + adds the **rotation-null geological floor** (`eta2_rotation_null`).
  5. **Calibrated-abundance values move** through the nonlinear CalibrationLayer even though ranking
     is preserved → new **gate 6** (per-bin RMSE / marginal-L1 via `compression_metrics`) where H4
     composes with the CalibrationLayer.
  6. Also shipping the **H1-only (pre-leveling) composite GeoTIFF** + residual-only variant as
     first-class PLAN_FBuild §1 deliverables (free from the saved per-frame logits; makes the
     trend-guard call reversible without a Sherlock re-run).

- **Number correction (already right, confirmed):** worst leave-one-FRAME-out η² = **0.0634** (B03),
  not 0.0857 — 0.0857 is the cruder *drop-frame* probe (P21). DECISIONS 2026-07-14 already states this
  correctly; the audit's own reviewer had briefly conflated them.

Net: audit is a GREEN on correctness with a punch-list, not a blocker. P5 reopening call stands where
it was; the within-frame-ramp check (V5) is the one item that should run at the Stage-B sizing probe
before the 907-frame array. Workflow transcript under the session's `workflows/` dir.

## 2026-07-23b — REOPENING CALL MADE: reopen-with-guards (P5 ✅) + head-to-head comparison vs mosaic + A1 mandated

Brian's P5 decision on the PLAN_FBuild §0 checklist (P1–P4 green; adversarial review YELLOW →
reopen-with-guards; two pre-spend probes green; H1/H4 code audit 2026-07-23 green): **REOPEN the
907-frame F build, with guards.** Added standing requirement: the build must produce a **head-to-head
comparison of the F-build vs the existing mosaic-path map and the A1 fallback, on both quality and
run-cost** — so the ship-vs-fallback call rests on evidence, not sunk cost. Folded into PLAN_FBuild as
deliverable 6 + **§5.1** (`f_map_compare`: quality table [partition η², THEMIS-ρ, pooled
pr_auc@1e-2/prec@5%, held-out edge-CV, visual] recomputed for all maps on ONE common footprint + a
run-cost ledger; a preliminary three-way read wired into the §0.1 early-stop checkpoint at 50–100
frames).

**Baseline scorecard on record** (Explore sweep; to be re-scored apples-to-apples by §5.1 — today's η²
numbers mix a pilot-crop scale and a regional detrended-residual scale, and A1's THEMIS-ρ is missing):
- **Mosaic-path** (26 tiles): pilot-crop η² 0.196 / regional detrended frame-block η² ~0.011; median
  per-image AUC 0.790, pooled PR-AUC 0.777; THEMIS ρ +0.07 (weak); blocks VISIBLE. Cost ~13–19 L40S-h
  / ~2–3 h wall.
- **A1** (per-frame normalization): pilot-crop η² 0.141 (28% ↓); median AUC 0.766 (Δ −0.024) / pooled
  PR 0.771 (Δ −0.007); THEMIS ρ NOT on record (gap §5.1 closes); blocks partially flattened, still
  visible. Cost ≈ 14-min post-hoc re-embed + re-bake (no re-inference).
- **F-build (H1+H4, pilot)**: η² 0.0505; held-out |Δp| 0.035; pooled PR Δ −0.0104 / deploy Δ −0.0007;
  THEMIS ρ 0.068→0.137. Cost ~333 CPU-h ISIS + ~25–40 GPU-h.

PLAN_FBuild status → APPROVED/EXECUTING; ROADMAP refreshed.

**Stage 0 DONE 2026-07-23** (`scripts/f_build_framelist.py` → `reports/figures/region_frame_list.csv`
+ `frame_tile_map.csv`): **907 unique frames / 1,371 frame×tile rows — exact match to plan**; EDR URLs
resolved 907/907 (deterministic template, no network), VOLUME_IDs consistent across tiles, 366 frames
span >1 tile (max 5). SeamMap incidence carried but flagged UNTRUSTED (V2 resolves it from PDS volume
indexes before Stage B). Per-tile counts 31–81 frames.

**Sizing-probe kit SET UP 2026-07-23** (`scripts/f_build_sizing_frames.py` selector +
`scripts/f_build_sizing_probe.py` V1/V5 measurement + `run_f_build_probe.sbatch` + SHERLOCK_RUN
Part G; `f_timing_test.sh` parameterized via `FRAME_LIST` so it can point at the sizing list without
clobbering the timing list). 5 representative frames selected (`reports/f_build/sizing_frame_list.csv`,
FPS over incidence/year/n_tiles): **P16_007374** (5 tiles, inc 42°, 2008 — longest track, the V5
target), **K05_055227** (4, 57°, 2018), **K01_053803** (2, 41°, 2018), **G09_021601** (1, 81°
grazing — V5 photometric stress), **P01_001440** (1, 49°, 2006 — earliest epoch). V5's residual-ramp
measurement is slope-invariant to the per-frame `cos^k(i)` constant, so the untrusted SeamMap
incidence is fine for it.

**PENDING = run Stage-A ISIS on Sherlock** (`sbatch run_f_build_probe.sbatch`, KEEP_CUBES=1) → cubes
→ GeoTIFF → **Stage-B `f_build_sizing_probe.py` on a GPU** → V1 array size (tiles/frame, GPU-h, CPU-h,
scratch) + V5 per-frame-vs-per-row verdict. §7 execution open questions (Stage-B venue, scratch
retention, tile order) to surface at the first Sherlock session.

## 2026-07-24 — F-build Stage-0 probe: Sherlock ISIS env rebuild (csm soname gap) before the sizing run

Submitting the V1/V5 sizing probe (`run_f_build_probe.sbatch`) surfaced that the July ISIS
micromamba env under `$GROUP_HOME/$USER/micromamba` was **gone** (group-home cleanup in the
intervening 3 weeks; `$SCRATCH/isisdata` — 283 GB incl. the full kernel mirror — survived, so no
data re-download). Rebuilding via `setup_isis_env.sh` then failed at `mroctx2isis` load with
**`libcsmapi.so.3: cannot open shared object file`**.

Root cause: the current channel combo **isis 10.0.0 (usgs-astrogeology) + csm 3.1.0 (conda-forge)**
ships `libcsmapi.so` + `libcsmapi.so.3.0.3` (a valid ELF, `file`-confirmed) but **not** the
`libcsmapi.so.3` SONAME symlink isis links against — a csm packaging gap. (Red herring en route: a
first symlink attempt hit `file too short`; a clean `env remove` + `clean -a` + single-thread
recreate produced a byte-identical file, proving it intact — the issue was only the missing link,
not corruption.) Fix: `ln -s libcsmapi.so.3.0.3 $CONDA_PREFIX/lib/libcsmapi.so.3` → ISIS loads (the
`cannot connect to X server` it then prints on no-arg invocation is the healthy GUI-launch path,
irrelevant to the arg-driven pipeline).

Hardened `setup_isis_env.sh` so a future rebuild can't repeat this: (1) **pin `isis=10.0.0
csm=3.1.0`** (unpinned solve drift is what changed the combo); (2) idempotently **create the
`libcsmapi.so.3` symlink** after env build; (3) smoke test now runs a real **`ldd … | grep 'not
found'` LOAD check**, not just `command -v` (the old check passed the broken env because the binary
was on PATH but failed at dynamic-link time). Also learned: `setup_isis_env.sh` is gated on
`mroctx2isis` *existing*, so it silently no-ops on a broken-but-present env — a rebuild must
`micromamba env remove -n isis` first. Env now green; probe resubmitted.

**First probe run (5 frames): 2/5 ok, 3 benign failures diagnosed → 2 Stage-A robustness fixes.**
P16_007374 (5-tile long track, the V5 target) and P01_001440 processed clean to cubes (map 4.35 GB /
0.57 GB; P16 cam2map 1476 s ≈ 25 min, total 28 min — confirms the ~22 min/frame / 333 CPU-h model).
Failures: **K05/K01 `spiceinit_fail`** = missing 2018 CK kernels (`mro_sc_psp_180508_180514.bc`,
`mro_sc_psp_180116_180122.bc`) — the mirror is incomplete for recent dates, so the build needs a
complete kernel fetch; **G09 `evenodd_fail`** = summed image (`SpatialSumming>1`, ctxevenodd
inapplicable) — benign. Fixes: (a) run `f_fetch_kernels.sh` on the probe log for the missing CKs;
(b) `f_timing_test.sh` now skips ctxevenodd when SpatialSumming>1. Both promoted to PLAN_FBuild §2
Stage-A requirements. Re-running the probe after both.

**Probe COMPLETE 2026-07-24 (corrected methodology) — V1 on-plan, V5 → per-row.** Re-run got 4/5
frames (K05/K01 passed spiceinit after the kernel fetch; G09 = transient PDS download fail, NOT quota
— scratch is 0.5/100 TB, so cube retention is a non-issue for the build). Two methodology bugs in the
first `f_build_sizing_probe.py` (both fixed + committed): a `read(1, out_shape=(1,h,w))[0]` that
silently sliced ONE raster row (broke V5, gave a bogus 0.3% valid frac), and a V1 that embedded every
tile in sampled windows incl. the ~50%-nodata canvas (cam2map frames are swaths in big lon/lat bboxes).
Corrected probe:
- **V1 ✅ on plan:** embedder **688 tiles/s** (RTX 5070); **~162M valid S=32 tiles** (counting non-nodata
  tiles, footprint-scaled to undo the probe's long-frame selection bias — frames 43–75% valid) →
  **≈33 L40S-h** (25–40 plan); ISIS ~200–330 CPU-h.
- **V5 → PER-ROW `cos^k(i(lat))` ADOPTED:** geometry-predicted illumination ramp **5.1%** (K05, 57°) /
  3.0% (P16, long) ≫ the 1% bar. (Raw measured ramp 6–46% is real along-track albedo over 300 km
  frames; per-row corrects only the incidence component, leaves geology.) PLAN_FBuild §3 mapping
  updated to per-row; §6 V1/V5 rows closed. Artifact `reports/figures/fbuild_sizing_probe.csv`.

Net: both pre-array unknowns resolved — the build is sized (≈33 L40S-h + ~200–330 CPU-h, on plan) and
Stage B's mapping is finalized (per-row cos^k(i(lat)) + conditional ctxevenodd + complete kernels).

## 2026-07-25 — F build Stage A kit ready (907-frame ISIS array)

Built the Stage-A execution kit (laptop/repo; runs on Sherlock). Stage A = ISIS-calibrate + project
all 907 region frames to `{PRODUCT_ID}.map.cub` (Stage B's input).
- **Worker:** reused the proven leg-B array worker `f_leg_b_process.sh` — parameterized its LIST via
  `FRAME_LIST` and folded in the summed-frame fix (skip `ctxevenodd` when `SpatialSumming > 1`,
  project the calibrated cube) so it is now the single build worker (PLAN §2 mandate satisfied).
- **Array:** `run_f_region_stagea.sbatch` → `FRAME_LIST=reports/figures/region_frame_list.csv`,
  `WORK=$SCRATCH/hirise2ctx/f_region`, 32 tasks / 12 h, resumable (skips existing cubes; 0-63/6 h
  option noted). Probe cost ~200–330 CPU-h → ~7–9 h wall.
- **Kernel-gap flow (self-healing):** the first pass `spiceinit_fail`s on the 2018+ frames the July
  mirror lacks and LOGS the names; `cat $SCRATCH/hirise2ctx/f_region/isis_*.log | f_fetch_kernels.sh`
  fetches them by name (proven on the probe), then re-submit resumes and fills the holes. Runbook =
  SHERLOCK_RUN Part H (submit → census → harvest → resume; final unrecoverable handful → H6
  mosaic-patch per V4, not blocked). Scratch ample (100 TB) → keep all 907 cubes.

Next after Stage A: **V2** (PDS incidence for all 907 — now load-bearing since per-row `cos^k(i(lat))`
needs each frame's N/S endpoints) + **V3** parity gate, then **Stage B**.

## 2026-07-26 — F build V2 (per-frame incidence for the per-row mapping): 907/907 resolved; pooled slope is CONFOUNDED

Stage A complete (907/907 cubes, integrity-clean). V2 = per-frame TRUE incidence + center latitude
for the per-row `cos^k(i(lat))` Stage-B mapping (`scripts/f_region_incidence.py` →
`reports/figures/region_frame_incidence.csv`; cube-based routes were dead ends — caminfo needs the
deleted Level-1 cubes, phocube is too slow — so PDS metadata it is).
- **907/907 resolved** from the PDS volume indexes (812 volumes) — V2 completeness gate PASS, no
  missing frames. incidence 37.1–80.8°, center_lat 30.0–50.5°.
- **SeamMap-vs-PDS incidence: 0 disagreements >1°** — the P20_008839 decimal-shift class does NOT
  appear in the 907 region set (it was a leg-B cohort frame); SeamMap would have been fine here, but
  PDS is used regardless.
- **⚠ The per-row SLOPE (di/dlat) is still open.** The pooled between-frame fit of center-incidence
  vs center-latitude = **+0.019°/°** — CONFOUNDED: pooling 907 frames across many Mars seasons washes
  out the latitude dependence (same-latitude frames at different Ls have very different incidence). It
  is NOT the within-frame di/dlat the per-row correction needs (audit within-family ~0.635; simple
  subsolar geometry for a region well north of the subsolar latitude → closer to ~1). **Resolution:
  make the slope a Stage-B parameter and pin it empirically — apply per-row on a test frame and tune
  so the residual within-frame ramp drops <1% (V5-style check, folds into V3); or add
  SUB_SOLAR_LATITUDE from the index for a physical per-frame slope. Do NOT use the confounded 0.019.**

Next: build Stage B (slope parameterized) → V3 (parity + residual-ramp slope check) → the 907 array.

**Stage-B kit built 2026-07-26** (`scripts/f_region_stageb.py` + `run_f_region_stageb.sbatch`, GPU
array, resumable): per-frame `I/F ÷ cos^k(i(lat)) ÷ per-frame median → fixed log-stretch → uint8 →
FangEmbedder + deployable_f_center → P(rich)`, keyed to a GLOBAL 160 m `(TI,TJ)` grid (round of each
tile's CTX-CRS world center) so overlapping frames co-locate for Stage C. Sizing ~33 L40S-h.
**⚠ Incidence-model correction (supersedes the linear slope):** the residual-ramp slope-pinning
idea does NOT work — like the raw ramp, it is geology-confounded (V5 lesson), and the true slope is
local-time/season dependent. So the linear `--slope` (default 0.635) is a placeholder; the correct
model is **PHYSICAL** — extend V2 to pull `SUB_SOLAR_LATITUDE/LONGITUDE` + `CENTER_LONGITUDE` and
compute `cos(i(φ)) = sinφ·sinφ_s + cosφ·cosφ_s·cos(λ_frame − λ_s)` per row (exact, no fitting;
reproducing the index center incidence is its sanity check).

**Physical incidence LANDED 2026-07-26.** V2 extended with `subsolar_lat/lon` + `center_lon`;
`f_region_stageb.py` computes per-row incidence physically and the `--slope` param is gone. V2 sanity:
physical incidence(center) reproduces the index incidence to **median 0.11° / max 0.23°, 0 frames
>2°** → the per-row gradient is exact and the ramp correction needs no tuning. V3 (`f_region_v3.py`)
now validates **co-location** (overlapping frames land on the same global (TI,TJ) tiles — the Stage-C
prerequisite) + **pre-H4 overlap agreement** on a couple of frames, instead of a slope. Early-stop
guard dropped (overnight Stage B; go/no-go at the Stage-C/D gates). **Ready for V3 → the 907 run.**

## 2026-07-27 — Stage-B first-run: two Sherlock gotchas fixed (head vendoring + ISIS3 out_shape segfault)

Kicking off the V3 Stage-B test on Sherlock hit two environment issues, both fixed:
- **H1 head not on Sherlock.** `models/` is gitignored, so `deployable_f_center` (the H1-retrained
  head) never reached Sherlock — Stage B died at `DeployableHead.load`. Force-added the small (2.6 MB)
  frozen head to git as a deliberate exception (commit 131e6e1) so it travels with `git pull`.
- **ISIS3 driver segfaults on large reads → convert to GeoTIFF.** `rasterio.read(out_shape=...)`
  SIGSEGVs (fixed `frame_median` → native strips, efe9760), but then `process_frame`'s native
  **4096² windowed read ALSO SIGSEGVs** (16.7M px; a 2048²=4.2M read is fine → a size threshold in
  GDAL's ISIS3 driver, faulthandler-confirmed at process_frame:124). Decisive fix: **convert all 907
  `.map.cub` → `.map.tif` (GeoTIFF/LZW)** via `run_f_region_tif.sbatch` (`gdal_translate` reads the
  cube in native blocks so the conversion itself is safe — proven in the probe), and Stage B now
  **prefers `.map.tif`** (resolver order changed). GTiff reads are solid at any window size (the
  laptop ran Stage B on tifs end-to-end). Order: **tif-convert → V3 → 907 Stage B.**

## 2026-07-28 — F build Stage C BUILT (H4 solve at 907-frame scale) — verified on synthetic + real MOLA/THEMIS while Stage B runs

Stage B is still running on Sherlock, so Stage C was built and validated **ahead of its inputs**
(PLAN_FBuild §4). Two new pieces:

- **`src/leveling.py`** — the reusable core. The pilot (`scripts/f_h4_level.py`) solved on a dense
  co-registered raster *stack*; at 907 frames that representation cannot exist, so the same maths
  now runs on Stage B's **sparse per-frame tile lists**: global `(TI,TJ)` → int64 key, exact per-edge
  sufficient statistics `(δ̄_ij, W_ij)` over co-located tiles, dense weighted-LS + λ·Σo² with a
  per-component `median(o)=0` gauge. Sign convention is unchanged and **pinned by a test against a
  verbatim copy of the frozen pilot normal equations** (`test_solver_matches_the_frozen_pilot_normal_equations`).
- **`scripts/f_region_stagec.py`** — the driver: load npzs → census → edges (cached) → λ sweep →
  held-out-edge CV → solve → LOFO → trend guard → CSVs + figures. Emits the offset **table** only;
  Stage D applies it, which is what keeps the H1-only (o=0) / full / residual-only composites all
  reproducible from one Stage-B run.

**VERIFIED AT RUNTIME (scale probe, 907 synthetic frames + real tile counts):**
`candidate_pairs` 0.1 s → 33.7k pairs; `build_edges` **~5 min** at the real ~178k tiles/frame
(8.9 ms/pair intersect); solve 0.05 s; held-out CV ×8 λ ≈ 2 s; **LOFO over all 907 frames 27 s**;
1,000-draw block permutation 0.3 s; peak RAM **1.94 GB** (int64 key + float32 logit for 162M tiles).
Stage C is a **~10-minute laptop step** as planned — it does not need Sherlock.

**Decisions taken while building (all pre-declared, all before any real offset was seen):**
1. **λ grid is relative** — `λ = frac · median(W)`, frac ∈ {0, 1e-3 … 1}. The pilot's absolute λ*=300
   is meaningless at build scale where edges carry 10–1000× more co-located tiles; the fraction grid
   brackets the pilot's operating point (~0.003–0.03·medW) by three decades either side.
2. **Held-out CV keeps the raw logit pairs.** Median |Δp| is a *nonlinear* function of the tile pairs
   and cannot be reconstructed from (δ̄, W), so the edge cache stores a bounded random subsample
   (1,000 pairs/edge, ~100 MB) alongside the exact statistics.
3. **A held-out edge whose removal splits the gauge is SKIPPED and counted, never scored.** On a
   redundant graph (build median degree 7) this is ~0; if *every* fold is undefined the driver
   **hard-exits** rather than picking λ from NaNs — gate 2 rests on this number.
4. **Graph holes get flagged provenance, not a silent zero** (`patch_graph_holes`): `solved` (main
   component) / `component_gauged` (smaller component shifted onto the main gauge by median IDW
   residual — never re-solved, that would mix gauges) / `interpolated` (isolated frame, pure IDW).
   Feeds PLAN_FBuild §1 deliverable 2's offset-provenance layer. P1 says this should never fire.
5. **§4.3 attribution is a rule table** (`lv.trend_verdict`, α=0.05, R² margin 0.05), encoded before
   the data: NO_TREND / FULL / RESIDUAL_ONLY / **AMBIGUOUS → `apply="full_pending_ruling"`**. The
   ambiguous branch deliberately does **not** resolve itself: §4.3 says "default full + H6 diagnostic"
   while §0.1 guard 1 says an ambiguous verdict must not silently become full offsets — both are
   honoured by refusing to auto-apply and escalating to Brian (§7 Q3) with the evidence attached.
6. **Significance is block-permutation throughout** (~4° blocks, whole blocks relocated). Plain
   permutation would destroy short-range autocorrelation too, giving a null that *any* smooth field
   beats — a trend test that always fires. This is also what lets the attribution discriminate at
   all: MOLA/THEMIS and a lat-linked incidence are both spatially smooth.

**Guard-1 binding CHECKED on the real cached rasters** (72 synthetic frames over the actual
circum-Chryse extent, planted per-frame biases, `mola_dem_region.tif` + `themis_night_ir_region.tif`):

| planted offset field | metadata R² (p) | geology R² (p) | plane R² (p) | verdict | correct? |
|---|---|---|---|---|---|
| −0.7·z(MOLA elevation) | 0.449 (0.002) | **0.951** (0.002) | 0.948 (0.002) | **RESIDUAL_ONLY** | ✅ guard 1 binds |
| −0.7·z(acquisition year), spatially random | 0.548 (0.002) | 0.012 (0.780) | 0.017 (0.735) | **NO_TREND** → full | ✅ (no smooth field to guard) |
| −0.7·z(incidence), incidence linear in lat | **0.988** (0.002) | 0.555 (0.002) | 0.982 (0.002) | **FULL** | ✅ metadata can still win |

Offset recovery vs the planted field: **corr +1.000** in both directed cases. The third row is the
one that mattered: because MOLA/THEMIS are themselves smooth, a naive test would collapse to
"always residual-only" — it doesn't.

**Tests:** `tests/test_leveling.py` (28) + `tests/test_region_stagec.py` (4, end-to-end over
synthetic Stage-B npzs incl. the missing-frame census and the edge cache) — 32 passing.

**Outputs Stage C will write** (all under `reports/figures/`): `fbuild_stagec_offsets.csv` (the
deliverable: `offset_logit`, `offset_residual_only`, `offset_source`, LOFO, per-frame covariates),
`fbuild_trend_guard.{csv,png}` (§4.4's mandated name), `fbuild_stagec_{lambda,graph,attribution,
watchlist,missing_frames}.csv`, `fbuild_stagec_offsets.png`; edge cache in `reports/f_stagec/`.

## 2026-07-29 — F build Stage D BUILT (composite + 6 gates + §5.1) — and gate 1's bar had to be re-scoped to survive the change of scale

Commit `afe6fce`. Stage B finished in a parallel session (906/907 npzs, still on Sherlock scratch), so
Stage D was built and its non-F rows RUN on real data. New: `src/fcompose.py`, `src/fgates.py`,
`scripts/f_region_staged.py`, `scripts/f_region_gates.py`, `scripts/f_map_compare.py`,
`scripts/bank_calibration_f.py`, `scripts/striping_a1_map.py`, `--restrict-store/--tag` on
`scripts/striping_a1_loio.py`, 63 tests (fast suite 464).

### VERIFIED AT RUNTIME: the two grids are different lattices, related by an exact integer shift

Stage B keys tiles to an **exact 160.0 m** lattice anchored at the CRS origin; the mosaic-path map is
**159.9991835298017 m** per Murray tile with origins that are not multiples of 160. Measured over all
26 tiles (`fcompose.tile_index_map`, asserted every run):

- within any one tile the relation is a **CONSTANT INTEGER SHIFT** — `TJ = col + Kj`, `TI = Ki − row`
  (e.g. E-12_N32 Kj=−4444/Ki=13335; E0_N44 Kj=1/Ki=17781) — so **no interpolation is needed**;
- but the lattices differ by a fixed **sub-pixel translation of 6.0–80.0 m in x / 7.9–50.3 m in y**,
  and the **E0 lon column sits 1.2 mm from a half-cell rounding tie** (its map pixel centres are
  ~80 m = half a cell from the global node they map to, i.e. on the cell boundary).

Stage D therefore places tiles by the integer shift and **reports** `dx_m`/`dy_m`/`tie_margin_m` per
tile (`fbuild_staged_registration.csv`) instead of pretending the grids coincide. The translation is
well inside the project's own O(200 m) HiRISE↔CTX registration budget (CLAUDE.md), and note Stage B
already accepted a ≤80 m quantisation when it chose an exact-160 key — this is that same choice
surfacing, not a new error. `TI` increases **northward**, so `row = Ki − TI`; a naive
`raster[TI − TI_min]` yields a vertically mirrored map (pinned by a test).

### GATE 1: the pre-declared bar is not interpretable at 907-frame scale (measured, then re-scoped)

η² has no group-count correction, so it grows mechanically with frame count and footprint. Measured on
the **existing mosaic-path map** (read-only probes, then reproduced by the shipped scorer):

| scope | mosaic partition η² | its own rotation-null | reading |
|---|---|---|---|
| merged 26-tile block | **0.3575** | mean 0.2837 / p95 0.3189 | 79% of the "artifact" is reproduced by rolling the field |
| per 4° tile (median) | **0.1850** | p95 0.1268 | ratio 1.46 |
| ~75 km window (median of 234) | **0.1222** | p95 **0.0676** | ratio **1.65** |
| detrended σ=30 px | 0.0123 | p95 0.0023 | the un-mitigated map already passes 0.05 |

So the literal "partition η² ≤ 0.05 on the full block" sits **below the geological floor** (nothing can
pass) while the detrended reading is **already passed by the un-mitigated map** (nothing can fail).
The 0.05 bar was calibrated on a ~75 km / 7-frame crop where the mosaic scores 0.1948 against a null
of 0.083–0.117.

**Brian ruled 2026-07-28:** headline = partition η² on **~75 km windows** (469 coarse px = the pilot
crop's own size), each against **its own** rotation null, with the bar applied to the **median
window**; the full-block number reported **floor-relative** (η² − null_mean, η²/null_p95). Both are
computed for every row on one grid and one quantity (raw P(rich), partition composite).
**Mosaic baseline banked** (`fbuild_gate1_summary.csv`): median-window η² 0.1222, null p95 0.0676,
excess +0.0719, ratio 1.65, 21.4% of windows already under 0.05, `passes_bar` False. Independent
cross-check: the shipped scorer's tile-scale median 0.18499 reproduces the earlier probe's 0.185, and
its windowed null 0.068 lands inside the pilot crop's measured 0.083–0.117 range.

**Consequence to keep in view:** the F rows must be read floor-relative too. H1+H4's pilot 0.0505 is
*below* the mosaic's own windowed floor, which is why each row is scored against **its own** null —
comparing one field's η² to another field's null is meaningless (the null depends on the field's
autocorrelation, and leveling changes it).

### Four more rulings (Brian 2026-07-28), all encoded before any F number existed

1. **§5.1 footprint = the 9 CTX-equipped tiles.** There is **no A1 raster on disk at any extent** —
   `striping_a1_infer_crop.py` only ever saved a PNG — and A1 renormalises raw CTX **DN** before the
   frozen ViT, so there is no post-hoc path from the existing probability rasters. A1 can only cover
   tiles with a cached Murray zip (9 of 26; the rest would need ~30 GB). Every §5.1 row is scored on
   that footprint. `scripts/striping_a1_map.py` generates it (~5–7 GPU-h).
2. **Calibration: re-bank on the F path, ship both.** The banked layer was fitted on the *mosaic-path*
   head's LOIO predictions; Tier-2 is a quantile-match (a marginal-transfer map), and the two heads'
   pooled P(rich) marginals differ by **CDF L1 0.0358** (median 0.2513 → 0.3218) — the same class of
   train/deploy mismatch that killed F pilot leg A. `scripts/bank_calibration_f.py` RAN:
   **Tier-1 LOIO ECE 0.0280 PASS** (bar 0.05); **pooled Tier-2 top_ratio 0.8783 PASS** (band 0.8–1.2;
   the mosaic layer's number of record is 0.8573); the re-bank recovers the true-zero mass far better
   (**marginal L1 5e-6 vs 2.1e-3**, near-zero share 0.194 vs 0.104 against a true 0.188). Gate 6
   reports under **both** layers. Feasibility had to be checked first: the F LOIO preds CSV has no
   `ti`/`tj`, but qmatch needs only the two marginals and the per-obs tile counts match the labels
   **36/36 exactly** (153,663 = 153,663), so the label `fractional_area` marginal is the correct target.
   *Correction made during the run:* I first reported Tier-2 `top_ratio` as a **median over per-image
   ratios** (0.5925) and compared it to the band — but the 0.8573 on record is a **pooled** statistic,
   and a per-image ratio is far harsher (each image's tail is predicted from the other 35). Both are
   now reported; the pooled one is the gate. The per-image spread is itself informative: median 0.5925,
   p10 0.0696 — consistent with the parked LOIO-negative reliability finding.
3. **Gate 5 = the DELTA, not the absolute.** The F head is **in-sample on all 36** images (its
   `train_obs_ids` *are* the 36), so an absolute per-frame pooled pr_auc reads ~0.92 against a LOIO
   number of record of 0.7964 and would pass vacuously by +0.13. Gate 5 scores
   **Δ(H1+H4 − H1) ≥ −0.02**, where the head cancels (what the 2026-07-15 probe did: Δ −0.0007).
   Also measured: **only 21 of the 36 cohort images have CTX source frames inside the 907** (15 of the
   28 multi-frame obs), so the build's own products cannot cover the pre-declared footprint; the gate
   is scoped to the in-region obs and says so on the table.
4. **A1 skill re-run on the 36.** `striping_a1_loio.py --restrict-store fang_embeddings_f_minnaert_center
   --tag _36` (verified: 38 ∩ 36 = 36). Post-hoc row filtering cannot fix it — the 38-image folds
   trained on 37 images rather than 35. The restrict helper is copied **verbatim** from
   `f_leg_b_loio.restrict_fold` (my first hand-rolled version was wrong: `y_train`/`y_test` are label
   DataFrames needing `reset_index`, and `groups_test` must be subset too).

### Other decisions taken while building

- **Composite rule pinned** to `sigmoid(mean_f[logit(prob_f) + o_f])` — mean in LOGIT space, one
  sigmoid at the end (reference: `f_h4_legb_perframe.composites`), with `src.leveling`'s EPS=1e-4 so it
  composes exactly with Stage C's offsets. Calibration is applied **once to the composited P**, and
  abundance from the **raw** composited P (never the isotonic output), mirroring
  `src.mapping.predict_window` — `calibrate_abundance` is nonlinear, so mean-then-calibrate ≠
  calibrate-then-mean (pinned by a test with a convex Tier-2 map).
- **H6 overlap-QA `max |Δp|`**: max over frame PAIRS equals `p_max − p_min` exactly, so the O(k)
  running min/max **is** the O(k²) quantity, not an approximation (pinned against brute force).
- **Provenance layers**: `n_frames`, `primary_frame` (the lowest-incidence contributor — with a mean
  composite no frame "owns" a pixel, so this is provenance-of-record, not the value's source),
  `incidence`, and `offset_source` taking the **worst** contributor's severity
  (solved / component_gauged / interpolated / none).
- **An AMBIGUOUS verdict writes no headline map.** All three variants are written under explicit
  names, but the plain `{tile}_prob.tif` names notebook 24 globs are left absent until Brian rules
  (§0.1 guard 1 vs §4.3's "default to full"); `--headline` overrides explicitly.
- **Gate 1 needs a partition composite**, which only exists during the frame pass, so the driver emits
  `{tile}_{variant}_prob_partition.tif` rather than making the gate script re-read every npz. Note the
  F build's partition raster uses a strictly *smaller* subset of its own predictions than the mean
  composite it ships — that is the price of being label-comparable to the mosaic map, which has one
  value per pixel by construction.
- **Output goes to `reports/map_fbuild/`**, never into `reports/map_region/` (PLAN §1 deliverable 4
  keeps the mosaic map as the comparison object). `src.striping.frame_label_map`/`load_frames`
  hard-wire that directory, so Stage D calls `rasterize` itself via `fcompose.frame_labels_on_grid`.
- **`f_map_compare` masks every row to ONE common finite footprint** before scoring, so a coverage
  difference can never masquerade as a metric difference.
- **Cost ledger provenance is recorded per row**: the mosaic/A1 GPU figures are *planning* estimates
  from the sbatch header (`region_manifest.json` is stale — it records only the last 4-tile array
  task), while the F-build Stage A/B numbers are probe-measured (V1).
- **Two of my own test assertions were vacuous** and are fixed with the reason in the test: a
  symmetric logit pair (0.2/0.8) where mean-of-logits and mean-of-probabilities both give 0.5, and a
  partition-vs-mean comparison under offsets that make the frames agree exactly. Both now use
  discriminating inputs.

**Not yet run:** the F rows of every gate. The A1 map and the A1 LOIO re-run are GPU steps.
SHERLOCK_RUN **Part J** is the transfer + run order.

### 2026-07-29b — adversarial review of Stage D: 36 findings, 17 confirmed, 2 BLOCKERS fixed

7-dimension find→refute-by-default review over the Stage D implementation. The two blockers were both
things my own tests were *structurally unable* to catch, which is the more useful lesson than either
bug:

**Blocker 1 — the gate-5/6 cohort join was off by ~100 km.** `fgates.cohort_tiles_to_global`
reconstructed each labelled tile's world position from `(ti, tj)` plus the observation window's corner
in `cohort_obs_bounds.csv`, on the premise that `ti/tj` are window-relative. They are not:
`src/labeling.py:363-370` emits them anchored at the **parent Murray tile's** `inner_transform`
origin, so the window offset is already baked in (`DATA_DICTIONARY.md:184`; `src/features.py:653`
*subtracts* `mosaic_row_origin` to get back to window coords). Anchoring already-absolute indices at
the window corner added `(col0, row0)·5 m` on top — **measured median displacement 94.7 km in x /
108.8 km in y over all 38 obs** (max 210/219 km). It failed silently: ~79k mis-keyed rows still landed
inside a block tile on finite pixels, so gates 5 and 6 were pairing labels with predictions ~100 km
apart and publishing plausible numbers. **Fixed** by keying off the world bbox the label rows already
carry (`xmin/xmax/ymin/ymax` → centre → Stage B's `round(·/160)`), which also drops the `32*5.0`
pitch approximation (label tiles are 159.9992 m). Verified against the mosaic map on real data:

| | mis-keyed | fixed |
|---|---|---|
| pooled pr_auc@1e-2 | 0.544 | **0.9013** |
| precision@5% | 0.442 | **0.9878** |
| Spearman(fa, p) | **−0.180** | **+0.7954** |

(102,948 labelled tiles on finite pixels, 23 obs, 9 tiles; E16_N44 alone 0.939/+0.791, reproducing the
reviewer's independent probe.) The old unit test could not fail: it pinned `row0=col0=0` and derived
its expectation from the same formula. The replacement uses hand-chosen exact-multiple-of-160 centres
(a half-cell value would sit on a `np.round` half-to-even tie), asserts invariance to the window
offset, and cross-checks against the real parquet geometry.

**Blocker 2 — gate 2 was scored on offsets the map does not use.** `edge_cv_for_offsets` delegated the
headline to `lv.heldout_edge_cv`, which re-solves FULL offsets per fold and never sees the `offsets`
argument — so `heldout_cv_dp` and `passes` were byte-identical for h1only / full / resid, and the
**H1-only row (no offsets at all) was reported as clearing the gate**. Exactly what that function's
docstring claimed to prevent. **Fixed**: the fold loop now lives in `fgates` and rebuilds each
variant's own offsets per fold — `h1only` short-circuits to the baseline with `passes=False`,
`full`/`lcv` solve at their own λ and metric, and `resid` refits its degree-weighted lon/lat plane
**inside** each fold (refitting outside would leak). The old tests asserted only `passes is True`
(which an absurd `full(n, 7.0)` vector also satisfied) and compared `median(edge_dp(zeros))` with
itself.

**Also confirmed and fixed:**
- **Gate 1 scored each row on its own footprint.** The F partition rasters have lower coverage than
  the mosaic map *by construction*, and the reviewer measured that a purely geometric 8–16% coverage
  deficit buys an 11–22% better η²/null ratio — with zero real mitigation. `gate1()` now builds one
  `common_finite` mask per tile (as `f_map_compare.quality_table` already did) and records each row's
  raw coverage alongside.
- **§5.1 did not intersect over TILES.** A row present on only some tiles was compared against rows
  scored on a different tile set; per-tile window-median η² spans 0.033–0.222 across the block, so
  tile composition alone moves a row by ±15% — the same order as the effect being adjudicated. Now
  intersected, with a loud "REDUCED FOOTPRINT" report naming the dropped tiles.
- **`abundance_fidelity`'s thin-data branch** returned a short dict, so gate 6's column selection
  raised `KeyError` *after* writing its CSV. Now NaN-fills the full key set.
- **The frame-index cache was never invalidated**, so a run started during a partial Stage B baked
  that frame set in permanently — and because the print lived in the build branch, a stale cached run
  emitted no frame count at all. Now stamped with `(n_npz, newest mtime)` and auto-rebuilt, with a
  `census:` line and a refusal to write a *headline* (shippable) map from a short Stage B unless
  `--allow-partial`.
- **The resume gate ignored the headline products**, so a second run with `--headline` (or after the
  verdict stopped being AMBIGUOUS) skipped the tile and silently never wrote the plain-named map while
  still printing "headline variant = ...".
- **Gates 5/6 pooled all 38 labelled images** while printing "of 36" — now restricted to the store
  intersection, per ruling 4.
- **`pooled_skill`'s test did not pin average precision**: swapping in `roc_auc_score` left all 23
  tests green, so the column named `pooled_pr_auc` could have been the forbidden presence AUC. Now
  pinned on a fixture that *asserts the two metrics differ* first — my first attempt at this test used
  `y=[1,0,0,1] / p=[.9,.8,.7,.6]`, where average precision and ROC AUC are **both** 0.75, i.e. it was
  itself vacuous.

**Not fixed, recorded instead:** ruling 1's *full-block* floor-relative η² is still computed at
per-4°-tile scope, not on a merged 26-tile raster (the block-scale null costs ~18 min/field at n=40).
The per-tile medians are reported; the merged-block row remains open.

**Measured, not assumed:** `TileAccum.add_frame` costs 21.4 ms per 60k-tile frame, i.e. ~1.7 min of
accumulation for the whole 26-tile × 60-frame × 3-variant run — `np.minimum.at`/`maximum.at` are
2.7–6.0 ms each on current numpy, so the buffered sort-and-`fmax.reduceat` alternative (4.7 ms) is not
worth the complexity. Fast-suite total 478.

## 2026-07-07 — PHASE 2 H1 (per-frame log-median centering): BOTH GATES PASS — η² 0.179→0.081, embedder amplification KILLED

First item of the PLAN_StripingArtifact PHASE 2 docket. **H1 = log-minnaert (k=0.580) + a
per-crop median centering step** (÷cos^k, then ÷ this crop's own median → common center, then
the FIXED centered-pool log stretch). Implemented as a new `minnaert_center` mapping in
`scripts/f_leg_b_embed.py` (and a mirror in `scripts/f_pilot_crop.py` so the η² test applies the
identical mapping the head trains on — the leg-A train/deploy-mismatch guard). Equivalent to H1's
"subtract log-median" because `log(x/median)=log(x)−log(median)`. Constants: k=0.580 (same as leg
B), centered pool stretch I/F **0.8400–1.1170** (pooled p0.5–99.5, 73 crops).

- **Skill gate PASS**: store `dataset_v2/fang_embeddings_f_minnaert_center` (36 obs) → LOIO
  Δ median per-image AUC = **−0.0139** (≥ −0.02); pooled PR-AUC **+0.0296**. (Slightly below plain
  log-minnaert's +0.0067 — centering costs a little within-scene ranking, still passes.)
  `reports/figures/f_leg_b_loio_summary_minnaert_center.csv`.
- **η² PASS** (the decisive test): head retrained on the centered store
  (`models/deployable_f_center/86c51a5dca220f63`, 88 s) → `f_pilot_crop.py --mappings
  minnaert_center --minnaert-k 0.580 --stretch-lohi 0.8400 1.1170` on the 7 E8_N44 frames:
  **η² median composite = 0.081** (was 0.179 for log-minnaert F; A1 = 0.141; mosaic raw = 0.196),
  partition = **0.128** (was 0.277). Both beat A1's 0.141 → H1's own gate (η² < 0.14 at skill
  ≥ −0.02) **PASS**. `f_pilot_minnaert_center.png`; before/after choropleth (log-minnaert vs H1,
  from cached preds) = `f_h1_before_after_choropleth.png` (`scripts/f_h1_compare_fig.py`) —
  the bright frame block in log-minnaert is gone in H1 on the same abundance scale.
- **Root-cause confirmation (the point of the review)**: prediction overlap disagreement fell to
  **0.073** median |Δp| — now *below* the co-located input I/F disagreement of 0.102. The embedder
  is no longer amplifying; centering removed the per-frame **level** term the review (2026-07-05d)
  identified. Residual worst pairs all involve **F02** (the anomalous frame, review-flagged): e.g.
  B03~F02 0.157, F02~P18 0.110; non-F02 pairs are 0.016–0.093.
- **Verdict**: H1 halves the artifact and kills the amplification, but median η² 0.081 does **not**
  yet clear the 907-frame-reopening bar (η² ≲ 0.05). Per the docket, H2 (embedding
  nuisance-subspace removal) stacks on the centered store next. Logs:
  `reports/f_leg_b/h1_{embed,gate,trainhead,eta2}.log`.
- **F02 anomaly confirmed radiometric** (`scripts/probes/_f02_diagnose.py`,
  `reports/f_leg_b/h1_f02_diagnose.log`): the dominant residual block after H1 is frame
  `F02_036739_2274` (the top-center block in the choropleth). Fitting `log(median I/F) = k·log(cos i)
  + b` across the 7 pilot frames, **F02 is −0.114 in log (z = −2.23) below the photometric line** —
  ~11% darker than its incidence predicts; all 6 other frames are within ±0.05 (±0.5σ). Even
  minnaert-corrected it is the darkest (0.116 vs 0.130–0.138). Geometry is unremarkable (incidence
  50.9° mid-pack, emission 9.6° near-nadir); it was acquired **2014** vs **2008** for the cluster it
  overlaps → atmosphere (dust/haze) or calibration-epoch offset, NOT geometry/geology. Consequence:
  F02 gets the **highest** frame-mean P(rich) (0.222 vs ≤0.173) yet overlaps frames predicting
  ≤0.07 over the SAME ground → co-located disagreement is artifact by construction. This is the
  per-frame level offset **H4 (overlap-constrained leveling) is built to absorb**. (Metric note:
  the pilot maps are raw **P(boulder-rich)**, P(fa>1e-2), not CalibrationLayer abundance.)

## 2026-07-05d — REVIEW of the F verdict (Brian request): "physical floor" claim OVERSTATED; correction works on level, the embedder is the amplifier; 6 untested mitigation hypotheses cataloged (lit review)

**Fact-check that revises 2026-07-05c** (`_f_review_overlap_residual.py`,
`reports/f_leg_b/review_overlap_residual.csv`): the "~10% co-located I/F difference is physical
and minnaert removes only incidence" claim measured the RAW overlaps. Post-minnaert the median
pair disagreement is **4.0%** (10.2% raw), and the worst high-Δi pairs drop to **0.7–4%** — the
photometric level correction largely WORKS. Yet predictions disagree 20.4% median even where
corrected I/F agrees to <1% (P21~P22: 1.0%→20.5%; B03~P21: 0.8%→20.4%), and prediction
disagreement anti-correlates with Δincidence (ρ=−0.33). Decomposition of the real floor:
(a) a minority of ANOMALOUS frames — F02_036739's corrected residual stays 10–15% vs every
partner (atmosphere/calibration; physics can't fix, data-driven per-frame offset can);
(b) **embedder/head hypersensitivity — a 1–4% input residual becomes ~20% prediction difference
(5–20× amplification), and our retrained head's loss contained NO cross-frame term**;
(c) genuinely information-level components (shadow rendering, per-frame PSF/sharpness/haze,
multi-year surface change) of unknown share. So "cannot be removed input-side" was too strong:
what's true is that **radiometric MAPPING alone is insufficient**; the invariance/leveling axis
was never tested.

**What leg B tested vs never tested:** tested = 5 I/F→uint8 mappings + retraining with the
UNCHANGED loss. Never tested = (i) data-driven per-frame level correction (per-frame log-median
centering — the A1 move on calibrated frames, WITHOUT perframe's contrast pinning); (ii) any
cross-frame consistency objective in head training (we have overlap tiles = free supervision);
(iii) embedding-space frame-nuisance removal (CORAL-style / pair-difference subspace projection);
(iv) output-side seam-graph leveling of the prediction map (option E, SeamMap polygons known);
(v) fusion beyond the median composite (which already cuts η² 0.277→0.179).

**Literature anchors (verified DOIs, `_f_litreview_queries*.py`):** data-driven relative
radiometric normalization is the EO standard — PIF/IR-MAD ([Canty & Nielsen 2004](https://doi.org/10.1016/j.rse.2003.10.024),
[2007](https://doi.org/10.1016/j.rse.2007.07.013); [Du, Teillet & Cihlar 2002](https://doi.org/10.1016/S0034-4257(02)00029-9));
NASA's HLS harmonization ([Claverie et al. 2018](https://doi.org/10.1016/j.rse.2018.09.002)) with
per-scene BRDF c-factor ([Roy et al. 2016](https://doi.org/10.1016/j.rse.2016.01.023)) is the
canonical "make inputs agree before analysis" product; planetary precedent for per-image mosaic
leveling = THEMIS ([Edwards et al. 2011](https://doi.org/10.1029/2010JE003755)); modern mosaic
gain/offset network optimization ([Li et al. 2022](https://doi.org/10.1109/JSTARS.2022.3229392));
domain adaptation for RS ([Tuia, Persello & Bruzzone 2016](https://doi.org/10.1109/MGRS.2016.2548504)),
[Deep CORAL (Sun & Saenko 2016)](https://doi.org/10.1007/978-3-319-49409-8_35), planetary DA
precedent ([Lagain-adjacent JSTARS 2022](https://doi.org/10.1109/JSTARS.2022.3156371)); CRISM
photometry shows Mars photometric properties vary spatially → one global k is crude
([Fernando et al. 2012](https://doi.org/10.1029/2012JE004194)); Murray mosaic philosophy is
information-preserving + per-pixel provenance so USERS handle radiometric seams
([Dickson et al. 2024](https://doi.org/10.1029/2024EA003555)); accept-and-covariate precedent for
per-image illumination effects in planetary ML ([Bickel et al. 2020](https://doi.org/10.1109/JSTARS.2020.2991588)).

**Hypothesis docket (cheap→expensive; each gated on skill + η²):**
- **H1 per-frame log-median centering** (~1 h GPU, leg-B harness as-is): removes frame DC
  data-driven (fixes F02-class anomalies), keeps contrast. Directly targets the level term of η².
- **H2 embedding nuisance-subspace removal** (hours, closed-form): overlap-pair embedding
  difference directions on pilot+cohort overlaps → project out top-k → retrain head.
- **H3 consistency-regularized head** (1–2 days): loss += λ·(pred diff on co-located overlap
  tiles); sweep λ → skill-vs-η² Pareto; the overlap data is already on disk.
- **H4 overlap-constrained leveling of per-frame predictions** (days) *(corrected 2026-07-06 during
  the PLAN review — originally mislabeled "= E, works without overlap", which would be the
  RULED-OUT circular D)*: per-frame logit offsets solved on the overlap graph from **co-located
  prediction disagreements** (same ground, two frames → artifact by construction, no geology
  assumption) + smooth-trend guard. F-mode only; on the mosaic (a partition, no overlaps) leveling
  degenerates to D and stays ruled out. No-overlap frames get graph-interpolated offsets, flagged
  in H6.
- **H5 stronger physics (Hapke/atmospheric EPF)**: LOW priority — headroom now known small (4%
  residual, mostly anomalous frames that H1 fixes empirically).
- **H6 accept + per-frame provenance/confidence layer** (à la Dickson/Bickel): ship regardless.

**Status: F verdict amended from "closed — cannot work" to "input-mapping-only leg closed;
invariance/leveling legs (H1–H4) untested and cheap."** 907-frame build stays paused pending
H1–H3 evidence of a path to η² ≲ 0.05 at acceptable skill. Decision on running the docket → Brian.

## 2026-07-05c — F leg B η² CONFIRMATION: FAIL — retrained head does NOT remove the artifact (η² 0.18, blocks visible). F does not achieve its purpose.

**The decisive measurement leg A could not give (a head trained on F embeddings), now run.**
`f_pilot_crop.py --mappings minnaert_log --minnaert-k 0.580 --stretch-lohi 0.0965 0.2374
--head-dir models/deployable_f_wl/<hash>` — the 7 E8_N44 overlapping pilot frames, log-minnaert
mapping (the GATE-PASSING recipe, fixed training constants), predicted with the head trained on
`fang_embeddings_f_minnaert_wl`. Head via `train_deployable_head.py --store-name …_wl` (36 imgs,
in-sample AUC 0.956).

**η² on the E8_N44 crop (baselines: mosaic raw 0.196 / A1 0.141 / target ≲ 0.03):**

| composite | η² (retrained F head) |
|---|---|
| partition | 0.277 |
| partition_eroded | 0.278 |
| **median (deploy-relevant)** | **0.179** |

**Blocks are visibly present** in the frame-mean choropleth (`f_pilot_minnaert_log.png`, right
panel — sharp per-frame level jumps). Overlap disagreement: I/F median |ratio−1| = **10.2%**
(unchanged from A0), prediction |diff| = **20.4%** — the embedder still AMPLIFIES the residual
between-frame I/F difference ~2×, exactly as in leg A.

**Verdict — F does NOT solve the striping artifact.** The retrained head barely moves η² (median
0.179 vs mosaic-raw 0.196 = 9% reduction) and is WORSE than the near-free A1 mosaic-side fix
(0.141, 28% reduction). Retraining fixed the *skill* (gate PASS +0.0067) but not the *artifact*,
because they are different failure modes: the gate measures within-scene rich/poor ranking; η²
measures same-ground agreement ACROSS frames. The head was trained on within-image ranking, so it
never learned between-frame invariance — and it CANNOT, because the inputs themselves still differ.

**Root cause (now firmly established):** the ~10% co-located I/F disagreement is **physical** —
illumination (minnaert removes only the incidence part; emission/phase/atmosphere remain), not a
mosaic-construction artifact. ctxcal + minnaert + log-stretch do not remove it. Any texture
embedder amplifies a 10% input difference into ~20% prediction difference → η² ~0.18. The artifact
has an **irreducible input-side floor ~η² 0.14** (what A1's aggressive per-frame offset+gain
reaches); F's photometric model (incidence-only) doesn't even reach that. **The striping artifact
cannot be removed by swapping the mosaic for calibrated per-frame inference.**

**Implication:** the 907-frame regional ISIS build (~333 CPU-h) is NOT justified — its sole
rationale was artifact removal, and F does not deliver it. The skill-gate PASS is real but moot for
this purpose. **F is effectively closed as a striping mitigation.** Decision on the fallback
(ship A1 as the mitigation / pursue E / accept the artifact with a documented caveat) → Brian.
NOTE: this also makes the ESP_053989 minnaert-inversion fix moot (was next in the queue).
Artifacts: `reports/figures/f_pilot_minnaert_log.png`, `f_pilot_eta2_summary.csv` (minnaert_log row).

## 2026-07-05b — F leg B GATE PASSED: minnaert + LOG stretch = Δ median +0.0067 (first PASS); cubic refuted; log domain is the lever

**The gate is cleared.** `f_leg_b_embed.py --mapping minnaert --stretch-pcts 0.5 99.5
--stretch-scale log` (store `fang_embeddings_f_minnaert_wl`) then `f_leg_b_loio.py`:

| variant | Δ median AUC | Δ pooled PR-AUC | mean | win/loss | <0.5 |
|---|---|---|---|---|---|
| perframe | −0.0499 | −0.141 | 0.695 | 11/25 | 8 |
| global | −0.0387 | — | 0.741 | 11/25 | 1 |
| minnaert p2–98 | −0.0341 | — | 0.736 | 14/22 | 2 |
| minnaert wide linear | −0.0236 | −0.024 | 0.753 | 16/20 | 2 |
| minnaert wide **cubic** | −0.0270 | −0.041 | 0.742 | 17/19 | 2 |
| **minnaert wide LOG** | **+0.0067** | **+0.0170** | 0.747 | 18/18 | 3 |

**PASS on the pre-registered metric** (Δ median per-image AUC ≥ −0.02): F now EXCEEDS the mosaic
baseline (0.786 → 0.793) and improves pooled PR-AUC (+0.017). Log-stretch produces the biggest
improvers in the cohort: ESP_068483 0.611→0.846, ESP_069763 0.713→0.832, ESP_059421 0.716→0.840,
ESP_076499 0.849→0.936, ESP_055978 0.796→0.946. Rationale for log: surface texture is
multiplicative contrast, so ln(I/F) gives every scene a level-independent texture-DN budget — the
representation the FM's 8-bit pretraining wants.

**Cubic resampling REFUTED as the lever.** minnaert wide with cubic extract resampling (store
`_cubic`, from `obs_crops_cubic/`) scored −0.0270 — WORSE than the bilinear wide-linear −0.0236.
So the ~40%-HF-texture deficit measured earlier (`blur_check.csv`) is NOT what caps skill; the
I/F→uint8 *domain* (linear vs log) is. The blur hypothesis is closed; cubic will not be pursued.

**Honest caveats (two, both non-fatal):**
1. **Mean AUC still below baseline** (0.747 vs 0.771) because ESP_053989 is badly INVERTED (0.167).
   This is a **minnaert-specific** failure: ESP_053989 = 0.775 under global (no cos^k), 0.15–0.27
   under every minnaert variant. Its two frames (i=42.76°, 46.32°) get cos^0.58 divisors 0.847 vs
   0.826 — a ~2.5% step that global doesn't apply. So the illumination correction itself breaks
   this one image; the median is robust to it but the mean is not. Diagnose before the regional run
   (candidate: per-composite single divisor, or drop cos^k when a composite spans a Δi step).
2. **eta² (block-killing) with the retrained head is STILL unmeasured.** The skill gate proves F
   does not LOSE skill; it does not yet prove F REMOVES the artifact (F's entire purpose). Leg A
   measured eta² only with the mosaic-trained head (invalid). This is the one remaining confirmation
   before committing to the 907-frame regional build.

**Recipe if it holds:** minnaert (k=0.580, incidence from PDS volume indexes — `_f_leg_b_pds_incidence.py`),
pooled p0.5–p99.5 log stretch, bilinear extract. Stores/CSVs: `fang_embeddings_f_minnaert_wl`,
`reports/f_leg_b/variant_summary.csv`, `reports/figures/f_leg_b_loio_{preds,summary}_minnaert_wl.csv`.
**Next-step decision (confirm eta² / fix ESP_053989 / go to head-rebuild+regional) → Brian.**

## 2026-07-30 — Stage C RUN at 906-frame scale: the free H4 solve drifts; `pfree` (plane-free) shipped. The 2026-07-23 within-frame-ramp risk MATERIALISED

`f_region_logits.tgz` brought home (906/907 npz, 203.9M tiles) and Stage C run. **Graph is healthy:**
one connected component at every min-shared-tiles cut (50/100/200/500/1000), 6,014 edges ≥200 shared
tiles, median degree 12, max 45, **zero isolated frames** — P1 confirmed at build scale. One permanent
Stage-B hole (`P21_009378_2200_XI_40N002W`), patched from the mosaic + H6-flagged.

The pre-declared solve returns offsets spanning **32.9 logits** (|o|max 21.31) against a per-tile
logit clip of ±9.21. Everything below is why, and what shipped instead.

### Gate 2's metric is gamed by sigmoid saturation → RE-DECLARED (Brian)

λ*(|Δp|)=0.0 was picked at the grid boundary, CV monotone in λ. The 93% "win" (median |Δp|
0.1622 → 0.0112) is **saturation, not agreement**: `corr(railed fraction, median |Δp|)` = **−0.997**,
and at λ*=0 **51.8%** of co-located tile probabilities sit on a rail. In logit space the gain is only
**7.4%** and is nearly λ-independent.

**Landed:** `lv.edge_dlogit`, `lv.edge_saturated_frac`, `lv.EDGE_METRICS`, `metric=` on
`lv.heldout_edge_cv`. **Brian: gate 2 re-declared** — passes only if |Δlogit| improves **AND** railed
fraction stays ≤2× the unleveled baseline; |Δp| still reported for the audit trail.

### §4.3 verdict could be won by the side with the LOWER R² → FIXED

FULL fired via a `not g_sig` shortcut: metadata R²=0.108 (p=0.019) "beat" geology R²=0.142 (p=0.0579)
only because geology missed α=0.05 by **8 permutation draws in 1000**. Seed sweep: **19 FULL / 1
AMBIGUOUS** over 20 seeds; geology R² > metadata R² in **20/20**. Patched to require the margin
regardless of the loser's significance (NaN R² = *unavailable*, not *lost*). Verdict is now correctly
**AMBIGUOUS → `full_pending_ruling`**.

### NEGATIVE RESULT — selecting λ on |Δlogit| does not fix λ*

λ*(|Δlogit|) = 0.0 too. Held-out |Δlogit| moves only **1.1198 → 1.1431 (2%)** across three decades of
λ while |o|max moves 21.3 → 3.8. **Both CV metrics are edge-LOCAL** and blind to a global drift mode.
The `lcv` variant was therefore a duplicate of `full` and was removed. Recorded so it is not retried.

### Five hypotheses REFUTED before settling on the cause

| hypothesis | test | result |
|---|---|---|
| noise amplified on a soft mode | split-half solve × 24 | slope −0.654 ± 0.054, **0/48 sign flips** |
| objective is indifferent to the ramp | constrained vs free SSR | forbidding it costs 17.9% of residual / **1.28pp of total** |
| leverage points at the block ends | eccentricity of top-gain edges | they sit **closer** to centre (0.86×) |
| a few bad frames/edges | drop top-k and re-solve | top 1% (60 edges) cuts the ramp only 19%; random-1% control flat (−22.66 ± 0.09) |
| weakly determined ("datum defect") | LS error propagation, measured σ | ramp = −23.61 ± 0.99, **24σ from zero** |

Also **order-independent** (edge shuffle/reverse, frame relabel: max Δoffset 1.7e-13 … 9.0e-12), and
a lon-proportional bias field is a **perfect gradient field** (verified to 6.8e-16), so it sums to
zero around every loop — loop closure, the standard check for accumulating survey error, is
*mathematically blind* to it despite median degree 12 and 92.9% loop-consistent edge energy.

### The stochastic model IS wrong — but does not cause the ramp (NOT landed)

Measured components: **σ²_pair = 2.448** (per-tile variance of ℓ_j−ℓ_i), **σ²_sys = 0.401**
(loop-inconsistent residual, dof-corrected). SE(δ̄) if tiles were independent = **0.0092** logits;
actually achievable = **0.633**. So `w = W` **understates each overlap's uncertainty 69×**
(variance 4769×) and the correct inverse-variance weights are nearly **uniform** (max/min 1.03 vs the
current 2550×). Re-solving with corrected weights does **not** remove the ramp (−22.71 → −23.61).
**Deliberately left unlanded** — it changes every quoted error bar and belongs in its own change.

### THE CAUSE — and it was predicted on 2026-07-23

The ramp is **24σ significant AND physically impossible** (23.6 logits vs 0.05 of E-W gradient in the
H1-only control map, 3.24 total observed frame-level spread, 18.42 = the model's whole range; it rails
51.8% of tiles and makes 44% of edges worse). Significance under a wrong model measures consistency,
not truth → **the per-frame additive-offset model is misspecified.**

**This is exactly the 2026-07-23 audit's "ONE GENUINE RISK":** *"Both H1 and H4 are per-frame DC
operators and cannot touch a within-frame gradient"*, with the pilot *"insulated"* by its ~1.3° crop
while build frames span 3–4° latitude. Confirmed and quantified here:

- pair differences are **24.8%** explained by illumination/radiometry jointly — dominated by
  `ln_frame_median` (**R² 0.184, t=+36.8**), i.e. residual radiometry H1 left behind — versus only
  **0.4%** by longitude geometry. Incidence collapses t=−21.3 → +1.0 once radiometry is in, so its
  effect runs *through* radiometry, as the audit's Minnaert argument implies.
- within-frame **along-track (latitude)** overlap position is identifiable and nearly quadruples
  explained variance (R² 0.0069 → 0.0254) — the predicted within-frame ramp, in the predicted axis.
- within-frame **longitude** position is **99.8% collinear** with frame lon separation, so "real
  regional gradient" and "within-frame structure" are **not separable from overlaps even in
  principle** (R² 0.0040 vs 0.0045 — the within-frame version fits marginally better).
- and the per-step term is **patchy, not a gradient**: fitting b per lon tercile gives
  **+0.203 / −0.003 / +0.433** (R² 0.0023 / 0.0000 / 0.0150) — absent in two thirds of the block.
  Fitting one global plane to that and integrating it over 33° is what manufactures 22.7 logits.

This is the standard failure of overlap-only radiometric block adjustment, whose literature models
global **and local** differences rather than one offset per image
([Yang 2020](https://ieeexplore.ieee.org/document/9242244/),
[Pan 2024](https://doi.org/10.1016/j.isprsjprs.2024.09.005),
[Wang 2023 vignetting drift](https://doi.org/10.3390/rs15215129)).

### SHIPPED — `pfree`, the plane-free constrained solve (Brian's call)

`lv.solve_offsets_planefree`: same objective, region-wide plane (constant + lon tilt + lat tilt)
constrained out; 903 of 906 directions still fit exactly. **No tuning constant.** Justified by the
tercile result above — there is no constant gradient to estimate — not by the outcome. `plane_complement`
is **rank-aware** (SVD, not QR): a degenerate frame layout would otherwise have a phantom third basis
column silently delete real signal.

| variant | \|o\|max | ×frame spread (3.24) | frames past ±9.21 | railed | trend verdict |
|---|---|---|---|---|---|
| `full` (λ*=0, pre-declared) | 21.31 | 6.58× | **128** | **51.8%** | AMBIGUOUS |
| `resid` (solve then detrend) | 7.33 | 2.26× | 0 | 2.9% | — |
| **`pfree` (SHIPPED)** | **4.91** | **1.52×** | **0** | 4.9% | **NO_TREND** |

`pfree` strictly dominates `resid` (SSR 4.65e7 vs 5.83e7 — constrain-then-solve beats
solve-then-subtract, a theorem now pinned by test). Explains 91.58% of pairwise disagreement vs the
free solve's 92.86%. **Documented caveat (in the test suite):** where a region-wide gradient genuinely
exists, this solve both discards it *and* biases the local estimates (planted-local recovery 0.98 →
0.63) — acceptable here only because the term is patchy; **not** portable to a region where b is constant.

**Lean guards landed** (Brian: no spectral machinery in the critical path) — `lv.benefit_concentration`
+ `lv.offset_magnitude_report` + `lv.frame_level_spread` → `fbuild_stagec_lean_guards.csv`. These
caught what every edge-local CV passed. **Deferred (Brian):** the "is there a real E-W abundance
gradient" question, and the stochastic-model fix. Stage D run with `--headline pfree --allow-partial`.
Variants are now `h1only` / `full` / `resid` / `pfree`; 99 tests pass.

## 2026-07-30b — F BUILD HARD ABORT (Brian): F fixes local striping and breaks regional level coherence

The build ran end to end (Stage A → B → C → D → gates 1–6, 906/907 frames, 4 variants + mosaic
baseline) and the §0.1 fallback was invoked. **The A1 / mosaic-path map remains the deliverable; no F
map ships.** PLAN_FBuild → CLOSED. Decision evidence, all on the 26-tile circum-Chryse block:

### The gate table (windowed gate 1 = the pre-declared headline)

| gate | bar | h1only | full | resid | pfree | mosaic |
|---|---|---|---|---|---|---|
| 1 windowed η² | ≤ 0.05 | 0.151 ✗ | 0.071 ✗ | 0.089 ✗ | 0.087 ✗ | 0.121 ✗ |
| 1 ratio vs own null | — | 1.471 | 1.223 | **1.135** | 1.172 | 1.528 |
| 2 \|Δlogit\| + railing | improve, railed ≤2× base | ✗ | ✗ (51.8% railed) | **PASS** | ✗ (4.9%) | — |
| 3 THEMIS Δρ | ≥ −0.02 | +0.047 ✓ | +0.024 ✓ | +0.040 ✓ | +0.026 ✓ | — |
| 5 pooled skill Δ vs h1only | ≥ −0.02 | — | −0.089 ✗ | −0.030 ✗ | −0.186 ✗ | — |
| 6 top_ratio | 0.8–1.2 | 1.302 ✗ | 8.744 ✗ | 1.428 ✗ | **0.940 ✓** | — |
| 6 Spearman | — | 0.583 | 0.444 | 0.560 | 0.436 | — |

**No variant clears the absolute η² bar — and neither does the mosaic (0.121)**, so that bar is not
discriminating at this scale (already flagged 2026-07-29 when gate 1 was re-scoped).

### What F genuinely achieved (report this)

Within-tile striping is really reduced. Scale-free (sd/mean of the per-frame means, median over 26
tiles): **mosaic 0.827 → h1only 0.646 → resid 0.575 → pfree 0.536**; windowed partition η² 0.121 →
0.087. Note **h1only is WORSE than the mosaic** on η² (+25.2%), so H4 leveling is what earns the gain.
Gate 3 also improves for every variant. Per-obs discrimination is untouched: median Δap **0.0000**
(worst single obs −0.014) — H4 does not degrade the model's local ranking at all.

### The disqualifying finding — between-place level, against GROUND TRUTH

Gate 5's raw failure is a between-obs LEVEL effect, not a skill loss: obs-centred pooled Δap is
**+0.0068 (resid) / +0.0054 (pfree)**, i.e. both would pass. But obs-centring discards **25.5% of the
label variance** — the between-place accuracy that IS the product. So the level effect was measured
directly, per observation, as `mean(predicted abundance) / mean(labelled fractional_area)`, with the
mosaic sampled at the identical labelled tiles (95,606 tiles, 21 obs):

| row | median ratio | min | max | max/min | **sd(log₁₀)** |
|---|---|---|---|---|---|
| **mosaic** | **0.89** | 0.41 | 2.10 | **5.1×** | **0.170** |
| h1only | 2.22 | 0.67 | 19.63 | 29.4× | 0.328 |
| resid | 1.92 | 0.36 | 11.63 | 32.5× | 0.371 |
| pfree | 1.35 | 0.14 | 27.25 | **189.6×** | **0.532** |
| full | 15.54 | 4.68 | 380.28 | 81.3× | 0.412 |

- the incumbent mosaic map is **well calibrated against truth** (median 0.89, only 5.1× spread);
- every F variant is **1.9–3.1× less stable between places** and over-predicts ~2×;
- **leveling HURTS in every variant** (resid 1.13×, pfree 1.62×, full 1.26× worse than unleveled).

This does not depend on the mosaic being correct: if both maps were right the ratio would be constant
regardless. The spread grows monotonically with the strength of leveling applied while the mosaic is
fixed, so the instability is attributable to F. Per-tile F/mosaic ratios say the same thing
(max/min 6.9× unleveled → 24.6× resid → **72.8×** pfree).

**Verdict:** F trades the artifact it was built to remove for a worse one. For a map whose purpose is
comparing abundance where HiRISE is absent, that is the wrong trade.

### Corrections to earlier readings in this session (recorded so they are not repeated)

1. **Gate 1's tile-scale ratio flatters F.** I cited tile ratio 1.007 (resid) / 1.027 (pfree) vs the
   mosaic's 1.537 as "artifact indistinguishable from its null". But F's tile rotation null is
   **1.29–1.39× higher** than the mosaic's, and pfree's absolute tile η² is only −2.1% vs the mosaic.
   Smoother maps inflate the null and flatter the ratio. **The windowed row is the honest one**
   (η² −26% at +10% null).
2. **"Gate 5 is only a confound" was too generous** — obs-centring removes 25.5% of the target variance.
3. **Retracted:** the claim that the calibrator was banked on the H1-only path and structurally
   favoured h1only. There are no metadata JSONs in `models/deployable{,_f_center}/`; unsubstantiated.
4. **`pfree` was the wrong shipping recommendation.** Chosen on SSR + offset magnitude; it is the
   worst variant on skill (−0.186) and cross-obs Spearman (0.436), and worst on level stability
   (189.6×). Its documented test-suite caveat — the plane constraint biases local estimates — was the
   real signal and I under-weighted it. `resid` was the better-balanced variant.

### Retained deliverables

- **General, stays in the codebase:** `lv.edge_dlogit` / `edge_saturated_frac` / `EDGE_METRICS` /
  `heldout_edge_cv(metric=)`; `lv.benefit_concentration` / `offset_magnitude_report` /
  `frame_level_spread` (the lean guards); the `trend_verdict` margin fix (a genuine bug — a side could
  win holding the LOWER R²); rank-aware `lv.plane_complement`. 99 tests pass.
- **Shelved but kept:** `lv.solve_offsets_planefree`, the `pfree` variant wiring, Stage C/D scripts.
- **Evidence on disk:** `reports/figures/fbuild_*` (gate tables, offsets, lean guards, trend guard),
  `reports/f_stagec/` (edge cache — Stage C reruns in ~2 min from it), `reports/f_region_logits/`
  (906 npz, ~33 GPU-h to regenerate), `reports/map_fbuild/` (4 variants × 26 tiles).
- **NOT landed, still true:** the edge weights `w = W` understate each overlap's uncertainty **69×**
  (σ²_sys 0.401 vs σ²_pair/W); correct inverse-variance weights are near-uniform. Any uncertainty
  quoted from this solve is optimistic. Deferred with F.

## 2026-07-05 — F leg B mapping iteration: global + minnaert both FAIL; gate converges at ≈ −0.034; SeamMap incidence typo found

**Setup:** `f_leg_b_embed.py --mapping {global,minnaert}` (fixed pooled p2–p98 stretch; minnaert
divides each crop by cos^k(i) first, incidence from the SeamMap gpkgs, k fitted from the 63
frames' log-median vs log-cos-i), `f_leg_b_loio.py --f-store …` — same pre-registered gate
(Δ median per-image AUC ≥ −0.02 vs mosaic baseline 0.786 on the common 36 images).

**Results** (`mapping_compare_per_image.csv`):

| mapping | Δ median AUC | below 0.5 | improvers | notes |
|---|---|---|---|---|
| perframe | −0.0499 | 8 | 11 | 2026-07-04b |
| global (0.0713–0.1394) | −0.0387 | 1 | 11 | cures ALL 4 perframe collapses |
| minnaert k=0.531 | −0.0301 | 3 | 15 | ran with a BOGUS incidence (below) |
| minnaert k=0.580 (corrected) | −0.0341 | 2 | 14 | best honest fixed-stretch result |

**SeamMap metadata bug (VERIFIED + fixed):** frame P20_008839_2269_XI_46N046W carries SeamMap
INCIDENCE = 4.2759° — a **decimal-shift of the true 42.76°** (verified against the PDS mrox_0605
volume index via `_f_leg_b_fetch_true_incidence.py`; CTX's ~3PM orbit cannot see i≈4°). It
collapsed BOTH images it covers (ESP_068483 0.238, ESP_053989 0.344) in the first minnaert run.
`OVERRIDES` table added to `_f_leg_b_incidence_check.py`. After correction ESP_068483 recovered
fully (0.746) but ESP_053989 did NOT (0.274): it is the cohort's dimmest scene (I/F median 0.083)
and after ÷cos^k ≈ 0.101 ≈ the stretch floor lo=0.1011, so ~half its pixels clip to black — the
pooled-p2 floor concentrates all its clipping in the single dimmest scene.

**Read:** the mapping family has converged. perframe's catastrophic mode (per-scene contrast
pinning) is cured by any fixed stretch; illumination correction (minnaert) beats plain global on
the dim scenes as predicted; the remaining ≈ −0.034 is mapping-independent within our family and
plateaus well short of the −0.02 bar. Remaining suspects for the floor: (a) the F path's double
bilinear resampling (cam2map + extract reproject) softening 5 m texture vs the mosaic, (b) the
dim-scene clipping (one image, fixable with a wider stretch window, worth ≈ +0.005 median at
most), (c) genuinely different calibrated-frame noise character. **Key unmeasured quantity:** F's
eta² (block-killing) with a RETRAINED head — leg A only measured it with the mosaic-trained head
(invalid). If retrained-F eta² ≈ block-free, −0.03 skill may be a fair price for removing the
artifact entirely (A1's mosaic-side alternative was −0.024 for only a 28% eta² reduction).
**Decision — iterate further / measure retrained-eta² / close F — deferred to Brian.**

## 2026-07-04b — F pilot leg B (LOIO gate): FAIL at −0.0499 — but bimodal, and the mapping (not F itself) is the suspect

**Pipeline (all shipped, SHERLOCK_RUN.md Part F):** `f_leg_b_frame_list.py` resolved **81 unique
CTX frames** covering the 38-image cohort (94 obs×frame pairs; seammap-gpkg build for 10 uncached
tiles). Sherlock 32-task array (`run_f_leg_b.sbatch` → `f_leg_b_process.sh`) ISIS-processed the
frames; `f_leg_b_extract.py` produced **73 I/F crops covering 36/38 obs_ids** (ESP_066634_2210 +
ESP_071093_2210 both depend on the single failed frame K04_054963_2209_XN_40N358W — check
`status_*.csv` on scratch; recoverable). Laptop: `f_leg_b_embed.py` composited crops onto the
mosaic grid, applied **perframe normalization** (composite median→125 / IQR→27.7 DN), embedded
with the frozen Fang recipe → `dataset_v2/fang_embeddings_f/` (36 images, 100% valid tiles).
`f_leg_b_loio.py` ran the pre-registered gate on the **common 36 images, both stores restricted
identically** (train and test) for a fair Δ.

**Gate (Δ median per-image AUC ≥ −0.02): FAIL — Δ = −0.0499** (A1 reference −0.024):

| store | n_img | median AUC | mean AUC | frac ≥ 0.7 | pooled PR-AUC |
|---|---|---|---|---|---|
| baseline (mosaic) | 36 | **0.786** | 0.771 | 0.81 | 0.767 |
| F (calibrated, perframe) | 36 | **0.736** | 0.695 | 0.58 | 0.626 |

**But the per-image Δ is strongly bimodal, not a uniform degradation** (`diag_per_image.csv`,
probe `_f_leg_b_diag.py`):
- 11 images IMPROVE, some sharply: ESP_055978 +0.155 (0.796→0.951), ESP_068483 +0.148,
  ESP_046959/076499/052576/042964 +0.05-0.06.
- 8 images drop below 0.5 (anti-prediction): worst ESP_045550 −0.398 (0.784→0.386),
  ESP_046328 −0.397, ESP_054397 −0.286, ESP_069763/069669/059686 −0.22 to −0.27.

**Diagnostics rule out composite mechanics:** coverage ≈ 100% on every image; overlap fraction
and n_crops uncorrelated with ΔAUC (|ρ| < 0.07). Between-frame illumination mismatch inside a
composite ANTI-correlates in the 6-image gallery sample (`_f_leg_b_crop_stats.py`: improvers
carry the big frame-median ratios 1.43–1.58×, collapsed images 1.02–1.30×) — the last-write-wins
composite + single normalization is not the killer either.

**Over-stretch hypothesis REFUTED on real quantities** (`_f_leg_b_uint8_contrast.py`,
`diag_uint8_contrast.csv`): the perframe mapping pins every F window at uint8 IQR 27–28 by
construction (mosaic windows vary 19–57), and the F/mosaic contrast ratio is null vs ΔAUC
(ρ = +0.09). **The surviving correlate is the composite I/F median: ρ = +0.35 — DIM
(high-incidence) scenes collapse, bright scenes improve** — illumination again, exactly A0's
cos-i axis. Figures: `f_leg_b_diag_scatter.png` (bimodal bars + median scatter),
`f_leg_b_diag_gallery.png` (mosaic-vs-F windows + native-res zooms for the 3 worst collapsed
images / 3 best improvers; collapsed-image F zooms visibly texture-poor vs their mosaic
counterparts).

**Read:** F's calibrated frames carry usable signal (the improvers include some of the best AUCs
in the whole project — 0.951, 0.934, 0.928) but the perframe mapping leaves an
illumination-linked failure mode on the dim half of the cohort. The best-motivated next
iteration is **minnaert** (cos-i correction, k≈0.66–0.694 from A0/A, metadata-only at deploy),
with global-affine as the control; both re-use the crops already on the laptop — **no new
Sherlock work, ~1 h GPU per mapping** (re-embed + re-gate). Decision on iterate-vs-close-F
deferred to Brian.

## 2026-07-03b — F pilot leg A0 (CPU): calibrated frames differ by REAL illumination, not error

`scripts/f_pilot_ifcheck.py` on the 7 aligned E8_N44 crop frames (`f_pilot_ifcheck.png` + CSVs):
- **Per-frame I/F level spread 24.9%** — but **median-vs-cos(incidence) r = +0.83**: the spread is
  dominated by real illumination (B03 i=57.9° darkest 0.093; P18 i=43.1° brightest 0.115), plus
  atmosphere. **Walter's ±2% is flat-field/instrument stability, NOT scene-level constancy** —
  same-incidence pairs DO agree at 1.0–3.3% (P21~P22 pairs), high-Δi pairs disagree 13–22%.
  Median |ratio−1| across all 15 overlap pairs = 10.2%. Overlap correlations 0.56–0.98 (structure
  consistent; offsets multiplicative). IQR CV 0.26 (vs mosaic 0.43): contrast ~1.7× more uniform.
- **The mosaic's per-frame stretch was partly hiding real illumination variability** — F exposes
  it, so an input-side illumination handling layer is a *required* part of F, not an option.
- **Pure Lambert overcorrects** (cos(i) too strong at high i: spread 24.9→21.8% only, B03
  overshoots): Mars is non-Lambertian. The frames' own log(median)–log(cos i) slope gives
  **Minnaert k ≈ 0.66** (classic Mars range) → added a 4th pilot mapping **`minnaert`**
  (metadata-only at deploy once k is fixed — the A-meta idea landing inside F). Expected order
  for block-killing: perframe ≥ minnaert > lambert > affine; skill/physics trade to be judged
  with leg-A eta² + leg-B LOIO.

---

## 2026-08-04 — Review PASS 10 (`labeling-deep`), and an accidental v1 artifact mutation (restored)

**Context.** Code-review verification backlog closed (15/15 high-severity live-path findings; commit
`7bfedb8`), then a four-agent `labeling-deep` second pass opened under Pattern D ("audit the artifact,
not just the computation"). Findings **R74–R86** in `docs/CODE_REVIEW_2026-07-31.md` §4j.

**VERIFY-AT-RUNTIME answers banked:**

- **BoulderNet's inference footprint EQUALS the HiRISE image footprint.** This was an open question in
  the register (an interior detector gap would have made some zero labels false zeros). **REFUTED** on
  four independent tests: no crop (HiRISE-valid ground extends 21–29 m median beyond the extreme
  detection on all four sides, 38/38 images); no margin (unshifted detection density at 40 m from the
  coverage boundary is 1.506 ± 0.112× the image mean — *enriched*; the shifted control does show a
  deficit, so the test has power); no detector-grid periodicity (amplitude at the 512 m CCD pitch and
  the SAHI 256 m / 204.8 m tiling is *below* median); no geometric holes (150 enclosed zero-components,
  rectangularity max 0.859). **Do not re-open.**
- **The coverage mask is wrong in the opposite direction (R74, high, live-shipped).**
  `src/ctx_retrieve.py:507` defines coverage as `hi_arr > 0` on a nearest-neighbour 5 m decimation, but
  HiRISE DN is continuous through 0, so deep-shadow pixels are called "not observed". Since eligibility
  is `all(mask == 1)`, one 5 m pixel deletes a 160 m tile: **3,236 S=32 tiles (1.97 %) dropped, 93.0 %
  of them rich against a 36.0 % base rate, holding 7.70 % of all detected boulder area.** BoulderNet
  detected boulders inside those tiles at 3× density, so the data is there. A shadow-biased deletion of
  exactly the rock-rich tiles.
- **`dataset_v2/labels` is internally sound** — all 3,564,767 rows self-consistent with their sidecars;
  both live packaged schemes match the labels bit-for-bit; both live split JSONs rebuild to an identical
  `split_hash`. **v2 LOIO splits structurally cannot have the `within_image_4fold` vintage drift**
  (fold *i* is `sorted(obs_ids)[i]`, content-independent).
- **The labelling tests pin no wrong science, but pin far less than they appear to.** Mutation testing
  (25 seeded defects against a scratchpad copy of `src/`): **16 of 20 survive `pytest -m "not slow"`**,
  12 of 20 survive the full suite.
- **Open contradiction (R75):** two reviewers measured the `labeling-2` swath-edge strip on the same
  cached masks and got **3.89 %** vs **0.21 %** of S=32 tiles. Unresolved; do not cite either yet.

**DEVIATION — an accidental mutation of v1 data, detected and restored the same day.**

A review agent ran `pytest tests/test_labeling.py`, and at 21:26:35Z this **overwrote four gitignored
v1 artifacts**: `dataset/labels/ESP_069669_2220.{parquet,json}` and
`cache/reprojected_detections/ESP_065711_1545.{gpkg,json}`.

- **Cause (now filed as R77, high).** `test_stage4_runs_on_ESP_069669_2220` passes
  `output_dir=cfg.output_dir` and `test_empty_shapefile.py` passes `cache_dir=cfg.cache_dir` — i.e. the
  **live** `dataset/` and `cache/` trees, not a tmp fixture. The underlying hazard is broader: the
  producers write to config-derived live paths with **no dry-run mode** (`src/labeling.py:543`, `:591`,
  `src/detections.py:151`, `src/coregister.py:436`), so *any* audit that calls one mutates the dataset.
  `load_shift` is a pure read and is NOT the culprit. This had happened before, unnoticed:
  `cache/reprojected_detections/ESP_069669_2220.json` was rewritten 2026-06-10 by
  `test_sanity_residual_one_image.py`.
- **Why it mattered.** The rewrite was **not value-preserving**: the v1 labels predate the 2026-06-10
  y-sign fix, so the test migrated one of nine v1 images across a correctness boundary
  (`max|Δfa|` 0.115 at S=8, `max|Δcount|` 115 at S=64; 3,854 of 96,354 rows differed). `dataset_v2/`
  and `cache_v2/` — the shipped basis — were untouched, verified.
- **Restored.** Original label values survived in `dataset/packaged/loio_9fold/y_test_fold6.parquet`
  (untouched 2026-05-23 vintage). All 96,354 rows restored to an **exact match — 0 differing values
  across all 7 label columns**; tile geometry was never touched (`xmin`/`ymin` bit-identical
  throughout). Sidecar `dy` reverted to `-239.74878038511508` and `config_hash` to `e9962e94…`, so the
  v1 tree is internally consistent again (all 9 images pre-fix, one hash). The sidecar carries a
  **`restored_from`** block naming what was rebuilt and the two fields that could NOT be recovered
  (`written_at_iso`, `coreg_peak_correlation`). The mutated state is backed up outside the repo.
- **Guardrail adopted.** `docs/review_2026-07-31/_prompts.md` §1 now carries an explicit
  **no-producer / no-slow-tests** rule naming the four write sites and the two offending tests, so
  future reviewers inherit it. Brian's call: restore + mark reconstructed (not silently), and keep
  using review subagents with the rule rather than restricting them.

## 2026-08-04b — R74 fix applied, rebuild deliberately deferred (new policy + `docs/PENDING_REBUILD.md`)

**Policy (Brian).** As review findings are fixed, **apply the code fix but defer the re-run**, batching
every rebuild-requiring change into a single pass once the review is complete — so the expensive stages
run once, not once per fix. The accepted cost is deliberate artifact drift in the interim. Tracked in
**`docs/PENDING_REBUILD.md`**, which is the checklist for that eventual rebuild and the record of what
is knowingly out of sync. Items listed there are **accepted divergence, not new findings.**

**R74 fixed in code.** `src/ctx_retrieve.py` gained `_fill_interior_shadow_holes`, called from
`build_hirise_coverage_mask` via a new `max_interior_hole_px=16` kwarg (`0` restores pre-R74 behaviour).

- **Mechanism, for the record.** Three correct-looking things compose into the defect: (i)
  `hirise_imagery.py:192` decimates to 5 m/px with **nearest neighbour**, so one 0.25 m source pixel
  decides a whole 5 m cell; (ii) `ctx_retrieve.py:507` treats `DN == 0` as nodata, but HiRISE RDR DN is
  **continuous through zero** (histogram 0,1,2,3… all populated), so 0 is the bottom of the real
  radiometric range and a shadowed pixel is indistinguishable from unimaged ground; (iii)
  `labeling.py:277-279` requires `mask_min == 1` and `:325` propagates with `.all()`, so one 5 m cell
  deletes the 40/80/160/320 m tiles containing it. Because the dark pixels are **boulder shadows**, the
  deletion is correlated with the target.
- **Fix approach.** Geometry is distinguished from shadow **topologically, not by value**: only regions
  *fully enclosed* by valid data are candidates (so the rotated-rectangle exterior and any missing scan
  reaching the swath edge are untouched, being border-connected), and of those only components
  ≤ `max_interior_hole_px`. Measured shadow holes are 1–2 px (99 % single-pixel), so 16 is ~8× the
  observed scale and far below a plausible dropout. The shared
  `read_full_footprint_decimated` was **not** touched — `src/coregister.py:71` depends on it.
- **Validated read-only** on the 138 cached decimated 5 m/px arrays (never a JP2, never the producer):
  every re-marked pixel has **DN exactly 0**; `ESP_017355_2260` re-marks **1,185 px**, reproducing the
  reviewer's independently measured interior-zero count exactly; asserted that the fix only ever *adds*
  coverage, never alters the swath border, and that `max_interior_hole_px=0` is an exact no-op.
  `pytest -m "not slow"` → **490 passed, 21 deselected**, identical to the review baseline.
- **What the rebuild will move.** ~3,236 S=32 tiles (1.97 %) return, 93 % of them rich, holding 7.70 %
  of all detected boulder area; cohort rich prevalence 0.3598 → **0.3733**. So the frozen recipe's
  headline numbers, every prevalence-dependent statistic, the calibrator's quantile grid and the
  deployed map's upper range all shift. Note the amplification: only **0.0048 %** of valid *pixels* are
  re-marked, but **1.97 %** of *tiles* are recovered — that ratio is the finding.
- **Not invalidated.** Every previously reported number was *correctly computed*; it was computed on a
  population that silently excluded its own rich tail. The direction of the bias on skill is unknown
  (the excluded tiles are both the rockiest and the most shadow-saturated) — measure it at rebuild time
  rather than assuming.

## 2026-08-04c — R75 resolved: both reviewers were right, they counted different populations

Two second-pass reviewers measured the `labeling-2` swath-edge strip on the same cached masks and
appeared to contradict each other (3.89 % vs 0.21 % of S=32 tiles). Settled with a third, independent
measurement — reconstruct the vacated region as `mask AND NOT shift(mask)` using each sidecar's own
`(dx, dy)`, then compute per-tile vacated coverage via an integral image, over all 38 images and all
161,005 eligible S=32 tiles.

| population | measured | share | matches |
|---|---|---|---|
| tiles **overlapping** the vacated strip | 6,202 | **3.85 %** | the 3.89 % reviewer |
| tiles **fully inside** it | 340 | **0.21 %** | the 337 / 0.21 % reviewer |
| ...of which `fa == 0` | 340 | 100 % | the second reviewer's own self-check |
| share of the whole zero class | — | 1.17 % | their 1.16 % |

**Ruling.** Only the **340 tiles (0.21 %)** are labelled zero *by construction*: a partially-vacated
tile can still hold detections in its remaining area, so its zero is not forced. Pass 1's `low`
severity was right for the strict claim. **But the other 5,862 tiles have a *partially depressed* `fa`
— a milder bias over an 18× larger population that neither reviewer filed as its own effect, and which
counting zeros cannot see.** Report both: by-construction 0.21 %, partial depression 3.85 %
(area-weighted). The real fix is to shift the coverage mask with the polygons in Stage 3, which removes
both.

Unaffected and standing: the strip is an **L along the southern *and* western** edges (dy>0 in 38/38,
dx>0 in 30/38), and the shift pushes **82,210 detections (1.39 %)** out of the labelled area.
**One number does not reconcile** — the first reviewer states both "3.89 %" and "2,502 tiles", but
2,502 is not 3.89 % of any denominator in play (161,005 → 6,263; its own 164,273 interior grid →
1.52 %). The percentage reproduces; treat the count as suspect.

## 2026-08-04d — Review PASS 11 (`tests-deep`): mutation testing the large test bodies; 2 of 4 complete

Findings **R87–R90** in `docs/CODE_REVIEW_2026-07-31.md` §4k, plus extensions to R77 and R78.
Complete: `tests-deep-splits`, `tests-deep-features`. **Partial (session limit):**
`tests-deep-within-image.PARTIAL.md`, `tests-deep-region-staged.PARTIAL.md` — renamed with a
`.PARTIAL` suffix so the "an area is done iff its file exists" check keeps reporting them as TODO.

**The structural result: the fast-vs-full mutant survival gap is EXACTLY ZERO in all three suites
measured**, for three different reasons — in `splits` the two `slow` tests only call the metadata
loaders and can never reach `build_split`/`package_split`; in `features` the one `slow` test's
assertion is true by construction; in `region-staged` there are no `slow` tests at all. So
`pytest -m "not slow"` (CLAUDE.md's documented dev loop) is **not weaker** than the full suite; the
full suite is simply not stronger. Survival: labelling 16/20 fast (12/20 full), splits **10/16**,
features **12/22**, within-image **10/15** — all fast == full except labelling.

**No suite pins wrong science.** Across every body attacked, no assertion was found defending a known
defect. The consistent answer is "pins far less than it appears to".

- **R87 (high)** — a `package_split` fallback to a random per-tile split leaves the suite fully green.
  That is the **invariant-6** violation, the one that would invalidate every reported number. Every
  packaging assertion is a row count or a length; nothing checks which `obs_id`s land in which fold
  parquet. The splits are correct today — what is missing is the regression guard.
- **R88 (high)** — the X/y column split is unpinned: dropping `label_cols` from `_split_columns` puts
  `fractional_area` into the feature matrix and the suite stays green; `loaders.py:91-95` has no second
  filter. Silent perfect-score leak.
- **R89 (medium)** — 12 of 22 feature defects survive, including the whole labels→window registration
  arithmetic. **Useful negative:** the register's own fixes for R27 and R28 were run *as mutants* and
  left the file green, so **both can be applied without touching a test.**
- **R78 extended to a third suite, and promoted in importance.** Every `test_features.py` fixture pins
  the mosaic origin to (0,0), which **0 of 52 production sidecars** has (real ranges 894–43,790 /
  183–41,945). With `test_labeling.py` and the `src/fgates.py:211-231` ~100 km gate mis-key this is the
  third instance of one fixture-design defect that has **already caused a real error once**.
  Parameterising the shared fixtures over a real non-zero `(row0, col0)` is one change that fixes three
  suites — the highest-leverage single test fix in the register.
- **R77 extended to a THIRD live-tree producer test** — `test_features.py:485-495::
  test_features_align_with_labels_row_for_row` calls `stage4b_one_image(..., output_dir=repo_root/
  "dataset")` and overwrites the features parquet, its sidecar and both context-patch `.npy` stacks —
  in exchange for an assertion that **cannot fail** (`stage4b_one_image` emits one row per label row by
  iterating the labels groupby, so row-for-row alignment is true by construction).
  **All three are `slow`-marked — verified** — so `-m "not slow"` cannot reach them. Confirmed no
  further live-tree writes occurred: nothing under `dataset*/` or `cache*/` has a mtime after the
  15:21 restore.

**Orchestration lesson, recorded in the brief.** Mutation testing is one full pytest run per mutant, so
these agents are CPU-bound in a way the reading areas are not. Four launched concurrently saturated the
machine and all four were killed by a 600 s no-progress watchdog, losing everything because each planned
to write at the end. Fixed by (i) running at most **two** at a time and (ii) requiring each agent to
write its file **early and update it** — which is why the two session-limit casualties left usable
partial files instead of nothing.

## 2026-08-05 — Review PASS 11 COMPLETE (4 of 4 `tests-deep` areas); the "wrong science" hypothesis is refuted

Findings **R91–R96** added (§4k); PASS 11 closes. All five large test bodies have now been read
line-by-line *and* mutation-tested.

**HEADLINE — a refutation.** §6 of the register nominated "assertions that pin wrong science" as the
highest-yield work left. **Across ~100 seeded defects in five suites, not one assertion was found
defending a known defect.** The hypothesis is wrong. What is actually true is uniform and different:
the suites **pin far less than they appear to** (roughly half of all seeded defects survive), and the
dominant cause is **fixture degeneracy, not missing assertions**.

**Structural result: the fast-vs-full survival gap is exactly zero in every suite**, for four different
reasons (splits' slow tests only call metadata loaders; features' one slow test asserts something true
by construction; region-staged and within-image have no reachable slow coverage). So
`pytest -m "not slow"` is **not weaker** than the full suite — a useful licence, since it is the
documented dev loop and the only one safe to run (the three producer tests are all slow-marked).

Survival: labelling 16/20 · splits 10/16 (**8/14 = 57 %** after discarding equivalent mutants) ·
features 12/22 · within-image 10/15 (**9/14 = 64 %**) · region-staged 15/25 (11/25 with the unit suites).

- **R91 (high)** — the sharpest diagnosis of the pass. Every `within-image` fixture is the same
  symmetric 64×64 square, which makes three "surviving" mutants **literal no-ops** (0 tiles move) while
  on real labels they move **75–78 %**, 27–32 % and 0–4 % of tiles. M13 (cut computed once, reused)
  collapses **8 of 9 real images into one quadrant** — and the suite's own strongest assertion
  (`len(unique_train) == 3`) **would have caught it on a real footprint**. The coverage exists; the
  fixture disarms it. **Fix the fixtures and much of the existing coverage starts working.**
- **R92 (medium)** — the v2 `within_image_4fold` split artifact has drifted from its own labels in
  **29 of 38 images, 125,830 / 3,564,767 tiles (3.53 %)**, independently reproducing R45's 3.5 % by a
  different route — which **localises the fault to the split artifact, not the sweep**. Four candidate
  causes killed, including the obvious one: **the labels' mtimes predate the split**, so the y-sign-fix
  story fails. Cause unexplained; what this review adds is that nothing can detect it.
- **R93 (medium)** — `pfree`, the shipped variant the HARD ABORT verdict was pronounced on, is never
  composited by any test (the fixture omits its column and `:365` asserts the pre-`pfree` variant set,
  so fixing the fixture breaks the suite). **Judged honestly the other way: the wiring was verified
  correct** (`f_region_stagec.py:498` ↔ `VARIANTS["pfree"]`; composite exact to 1e-5), so **the abort
  verdict is NOT impugned.** Coverage gap, not a wrong number.
- **R94 (medium) — seventh instance of Pattern A, and the first inside a test rather than a gate.**
  The `biases=[0.5,-0.5]` fixture makes `h1only`/`full`/`resid` bit-identical (measured max diff
  0.000e+00), so the two tests pinning "the verdict ships the right variant" cannot fail. This extends
  Pattern A's reach: a *test* can be pinned by its fixture exactly as a *gate* is pinned by its
  construction.
- **R95 (medium)** — no assertion checks the georeferencing of any Stage-D raster; a GDAL-order affine,
  an empty CRS and a row/col transpose all survive, though **R01** is a shipped defect of exactly that
  class.
- **R78 promoted, with one positive exception.** The (0,0) mosaic-origin fixture is in three suites and
  already caused the `fgates` ~100 km mis-key. **`test_region_staged.py` is the one suite without it** —
  it uses E-12_N32's real origin and pitch, and `TILE_M` 160→80 kills all 18 tests. **Copy that
  fixture's shape when fixing R78/R91.**
- **Useful negative (R89):** R27's and R28's own fixes were executed *as mutants* and left the suite
  green — **both can be applied without touching a test.**

**Method note for any future test audit:** mutation testing produced all of this; reading alone would
not have. And the honest survival rate requires discarding equivalent mutants — two areas did so
explicitly (57 % and 64 % after discarding, vs 63 % and 67 % raw).

## 2026-08-05b — R77 FIXED: the test suite no longer mutates the live artifact trees

**The review undercounted by half.** It named three producer tests. Fixing those three and re-running
the *full* suite exposed **three more** that no reviewer had flagged:

| test | producer | wrote |
|---|---|---|
| `test_labeling.py::test_stage4_runs_on_ESP_069669_2220` | `stage4_one_image` | `dataset/labels/` |
| `test_empty_shapefile.py` | `stage1_one_image` | `cache/reprojected_detections/` |
| `test_features.py::test_features_align_with_labels_row_for_row` | `stage4b_one_image` | `dataset/features/`, sidecar, both context-patch `.npy` |
| **`test_stage2_one_image.py` (×2)** | `stage2_one_image` | `cache/ctx_windows/` **incl. the HiRISE coverage mask** |
| **`test_coregister.py` (×2)** | `stage3_one_image` | `cache/coregistration/` |
| **`test_sanity_residual_one_image.py`** | `stage1_one_image` | `cache/reprojected_detections/` |

The last one had **already been recorded in DECISIONS as rewriting a cache file on 2026-06-10** — the
data was there and nobody connected it to the pattern. **Lesson: the only reliable detector for this
defect class is to run the suite and diff the tree.** Reading the tests found half of them.

**Fix.** The three `dataset/` writers now take `tmp_path` (the features one stages its labels into the
tmp tree first, because Stage 4b reads labels from `output_dir`). The three `cache/` writers use a new
**`read_only_cache`** factory fixture in `tests/conftest.py`, which **hard-links** the read-side subdirs
into a throwaway cache — the CTX tile zips and HiRISE JP2s are hundreds of MB, so copying per test is
not viable. The hard link is safe **only because each producer's read and write subdirs are disjoint**;
that invariant is written into the fixture docstring, since a producer that truncated a linked path
would write straight through to the original inode.

**Verified.** `pytest -q` (full, 511 passed, incl. all 21 slow) leaves an **identical mtime checksum**
over `dataset/`, `dataset_v2/`, `cache/`, `cache_v2/` — before `7bb77d3f…`, after `7bb77d3f…` — and an
empty "modified in the last 4 minutes" window. **The full suite is now safe to run**, which it has not
been for the life of this project. Adopt the checksum diff as the standing regression check.

**One incidental consequence, recorded in `docs/PENDING_REBUILD.md`:** the pre-fix full-suite run at
09:16 regenerated `cache/ctx_windows/ESP_069669_2220_hirise_mask.tif` *with the R74 fix already
applied*, so for that one v1 image the mask is one generation ahead of its labels. Low stakes (the v1
tree is already stale per R81, nothing live reads it), resolved by the batched rebuild.

## 2026-08-05c — R91 fixed; R92 REFUTED and replaced by R97 (a dev-only change moved a production constant)

**R91 FIXED** (`tests/test_within_image_split.py`). The multi-image fixtures now use four deliberately
ragged extents — different shape *and* origin, `ti_mid != tj_mid` on each — instead of four identical
symmetric 64×64 squares. Demonstrated by mutation against a scratchpad copy, with a control run on the
pristine test file that reproduces the review exactly:

| mutant | old fixture | new fixture |
|---|---|---|
| M13 (cut computed once, reused for every image) | SURVIVED | **KILLED** |
| M04 (ti/tj transposed) | SURVIVED | **KILLED** |
| M01 (median→mean) | SURVIVED | SURVIVED — *equivalent mutant* |
| M05 (pooled-scale median) | SURVIVED | SURVIVED — cut value unpinned |

M13 now fails on exactly the assertion the review predicted (`len(unique_train) == 3` →
`unique_train={3}`, i.e. OBS_001 collapsing wholly into one quadrant under OBS_000's cut). **M01 is
provably equivalent on any rectangular fixture** — for a complete grid the marginal is uniform so
mean == median exactly (0.00 % of tiles move on all four extents); killing it needs a non-rectangular
footprint generator, which is beyond R91. M05 survives because nothing pins the cut's *value*
(the separate `-2` recommendation). Full suite **511 passed**, artifact checksum unchanged.

**R92 REFUTED as filed — the cohorts were inverted.** The finding claimed the **v2** split had drifted
from its own labels in 29 of 38 images (3.53 % of tiles). Two things were wrong.

1. **Quadrant definitions are never persisted.** They are computed inside `build_split`
   (`src/dataset.py:295`) and appear in **neither** the split JSON **nor** the packaged metadata —
   verified by listing the keys of both. So no reviewer read a "stored cut"; every such number was
   reconstructed, and the reconstruction is where the error entered.
2. **Measured against the only durable artifact** (packaged fold membership vs today's code and
   labels, via the production `_compute_quadrant_definitions` + `_quadrant_array_for_image`):

   | cohort | images differing | tiles differing |
   |---|---|---|
   | `dataset_v2` | **0 / 38** | **0 / 3,564,767 (0.00 %)** |
   | `dataset` (v1) | 5 / 8 | 13,969 / 610,586 (2.29 %) |

   **The live v2 split is exactly reproducible.** The drift is confined to the superseded v1 tree.

**The cause — new finding R97.** The cut snaps to `max(SCALE_TO_FACTOR_FROM_FINEST.values())`, and
commit `29b0adb` ("CNN + S128 **HELD as dev-only**") added `128: 16`, **doubling the production snap
step from 8 to 16**. No shipped config emits S=128 tiles. The v1 split predates that commit and sits on
the old 8-tile lattice. The internal tell that the review's probe used the stale factor: its quoted
`ESP_017355_2260 STORED 688 → RECOMPUTED 696` is impossible under a 16-snap, since 696 is not a
multiple of 16. Fix: derive the snap step from the scales actually present in the labels, or gate the
`128` entry behind the dev config — and pin it with a test.

**Process note.** R92 was a *single-agent* finding I folded into the register without independent
verification, and it was wrong in the most misleading way available: right phenomenon, wrong cohort,
with a confident unexplained-cause narrative attached. It survived because two agents made the same
factor-of-8 assumption. **This is the second time a "cause unexplained" flag turned out to mark an
error in the measurement rather than a mystery in the data** — treat that flag as a smell.

## 2026-08-06 — code-review audit correction, product scope, and rebuild hold

**Current handoff:** [docs/CODE_REVIEW_AUDIT_2026-08-06.md](docs/CODE_REVIEW_AUDIT_2026-08-06.md).
It is the correction layer to the 2026-07-31 review and `docs/PENDING_REBUILD.md`: corrected finding
states/actions, safety gates, product semantics, and the complete v2 rebuild DAG live there.

**Safety correction to 2026-08-05b:** the six direct test-output redirects landed, but the conclusion
"the full suite is now safe" is not structurally established. `read_only_cache` hard-links the mutable
derived `hirise_decimated` input, and Stage 2/3 can reopen that path with `"w"` if its cached CRS is
stale. The clean 511-pass checksum run did not take that invalidation branch. Until the audit's
isolation gate is closed, unfiltered/full pytest, slow producer-calling tests, and producers pointed at
repository artifact roots are on hold. Use only the reviewed non-slow loop or inspected synthetic
tests with independent temporary roots. Checksums remain secondary damage detection, not prevention.

**Brian's product decisions:**

1. Build the v2 regional rich-probability/calibrated-abundance mosaic and a matched A1-normalized
   version over the same planned 26-tile, globally anchored footprint. A1 is planned, not yet shipped.
2. Retain and explicitly document the current resolution-dependent minimum-included-label convention
   for the primary product. A separately identified common-floor target may also be produced later,
   but is a different target with separate target-dependent model/calibration/claim lineage.
3. `dataset/` v1 is superseded and will not be rebuilt. Preserve it only as a historical generation;
   current product and rebuild work are v2.

**Rebuild consequence:** R74's current 3,236-tile/prevalence deltas are a conditional counterfactual,
not final output counts. The rebuild must include Stage 3 and Stage 5, fresh forced LOIO predictions,
arm-specific heads/calibrations, and fresh baseline+A1 map generations. Before that begins, close the
test/cache isolation hole; fix the stage-specific gates in the audit; define A1's statistical unit and
enforce train/deploy parity; and make every step accept isolated versioned artifact roots.

**Status terminology correction:** R78 is the partially fixed non-zero mosaic-origin coverage gap.
R91 is fixed by differently shaped and origin-asymmetric rectangular extents; those fixtures are not
ragged footprints. R92 was refuted as filed, and R97 remains the live snapping-constant finding.

This audit/documentation session ran no tests, imports with producer side effects, producers, or
artifact rebuilds.

## 2026-08-06b — R77 residual CLOSED: the suite can no longer write a live artifact, and the audit's mechanism was half wrong

**What was actually wrong.** The 2026-08-05 `read_only_cache` fixture justified hard-linking with "each
producer's read and write subdirs are disjoint". `hirise_decimated/` broke that invariant: it is staged
for reading *and* rewritten by `read_full_footprint_decimated` when the cached CRS is stale. That much
of the audit is right.

**What the audit got wrong, measured.** It predicted the rebuild would truncate the live GeoTIFF through
the link. Controlled probe in a temp dir (rasterio 1.5.0 / GDAL 3.11.4 / NTFS):

| write API | reaches the source through the link? |
|---|---|
| `rasterio.open(p,"w")` | **no** — deletes-then-creates, the link breaks, the source survives |
| `rasterio.open(p,"r+")` | **yes** |
| `open(p,"wb")` | **yes** |
| `Path(p).write_text()` | **yes** |
| `shutil.copy2(other,p)` | **yes** |
| `Path(new).replace(p)` | no — swaps a directory entry |

The production rebuild uses `"w"`. So this was a **latent design error that current library behaviour
masks**, not a demonstrated data-loss path — and "the full suite is safe" and "the live TIFF gets
truncated" were *both* wrong. The fix removes the dependence on rasterio's create path entirely rather
than arguing from it.

**Three layers, because each catches what the others cannot.**

1. `tests/live_artifact_guard.py` — session-wide autouse, refuses `open`/`os.open`/`os.replace`/
   `os.remove`/`os.link`/`shutil.*`/`numpy.save*`/`rasterio.open(mode!="r")`/`to_parquet`/`to_file`
   under `cache*`, `dataset*`, `models`, `reports`, resolving the `cache_v2_dev` junction. Prevention,
   not detection: a checksum tells you what you already destroyed. Deliberately not guarded: `mkdir`
   (`Config.cache_dir` mkdirs on attribute access) and every read.
2. Static AST scan (`tests/test_artifact_isolation.py`) — the guard only fires on code that runs, and
   every producer test `skip`s when its cache is absent. The scan fails anyway.
3. Staging discipline — a hard link lives outside every guarded prefix, so no path-based guard can see
   a write through it. `read_only_cache` now copies everything except `{tile}.zip` / `{obs}_RED.JP2`;
   sidecars beside an archive (`{tile}.json`, GDAL PAM `.aux.xml`) are copied because GDAL rewrites PAM
   in place. Each copy asserts a distinct inode; teardown asserts every linked source is unchanged.

**Two corrections worth carrying forward.** `slow` was never the safety control: re-auditing markers
found **20 non-slow tests that call a producer** (all writing to `tmp_path`). And the 511-pass checksum
run proved one cache state, not isolation — the new end-to-end regression drives
`read_full_footprint_decimated`'s stale-CRS branch on **two temporary roots**, which is the branch that
run never entered.

**Checked and clean, so nobody re-derives it:** SP1=20 Mars equirectangular survives a GeoTIFF round
trip, so `_crs_equal` converges and a corrected cache is *not* rewritten on every call.

`pytest -m "not slow"` → **512 passed, 21 deselected** (490 pre-existing + 22 new), with an
11,218-file path/size/mtime manifest over all six artifact roots bit-identical before and after.
**Not run:** the slow suite. Its four producer tests have not executed since the fixture changed, so the
`only=` staging filter is verified only by a read-only listing of what each producer reads.

## 2026-08-06c — R78 CLOSED: both mosaic-origin mutants are now killed

The 2026-08-05 session re-based the labelling and features fixtures on the real mosaic phase
`(894, 12645)` (from `dataset_v2/labels/ESP_042964_2160.json`; 0 of 52 production sidecars has either
origin at 0) but never ran the mutants, so the fix was unproven. Run now, on **two independent
scratchpad copies of `src/` + `tests/`** — the working-tree `src/` was never modified:

| mutant | before | now |
|---|---|---|
| `src/labeling.py:367-370`, drop `mx_origin_x`/`mx_origin_y` from the tile bounds | suite green | **FAILS** `test_tile_bounds_align_with_mosaic_pixel_grid` (`xmin not aligned at scale 8`) and `test_label_transforms_emit_expected_columns` |
| `src/features.py:653-654`, `ti*S - origin` → `ti*S + origin` | suite green | **FAILS** four Stage-4b tests on the bounds guard (`RuntimeError: some Stage 4 tiles fall outside the cached CTX window`) |

Mutant (a) is the one the review used to make the point: it is the failure
`test_tile_bounds_align_with_mosaic_pixel_grid`'s own docstring names, it displaces real `ymin` by
2,608 km, and the assertion could not see it while the fixture's origin was `(0, 0)`. On the synthetic
fixture the residual is only 2.42 m — the fixture reproduces the *detectability*, not the production
magnitude, which is the whole point of parameterising it on a real phase rather than a large one.

**The last `(0,0)` call site is gone.** `test_ctx_source_illumination.py::test_add_features_end_to_end`
now runs on the real phase. Nothing in it ever asserted anything about a zero-phase window, so it was
the degenerate case, not a complementary one. On the real phase the CTX window is not tile-aligned:
`894 % 4 = 2` and `12645 % 4 = 1`, so tiles start at window row 2 and column 3 — two *different*
offsets, so a row/col swap now fails as well.

**One zero origin is kept on purpose.** `test_alignment_aligned_window` exists to test the aligned
case ("Window origin exactly at mosaic origin"), and `test_alignment_offset_window` supplies the
`(3, 5)` complement. That is the explicit-and-complementary pattern; the R78 defect was the *implicit*
one.

Fast suite green. The producers were not run: every test involved uses `tmp_path` and synthetic data.

## 2026-08-06d — R87/R88 CLOSED: the two catastrophic-regression guards, mutation-verified

Both findings were about *absent guards*, not live wrong numbers, and both checked out clean before
the guards went in — recorded explicitly so nobody later reads "R87 fixed" as "the splits were broken":

- production splitting is group-aware and the v2 LOIO splits cannot drift (labeling-deep-artifact);
- **all 620** packaged `X_*.parquet` files under `dataset/` and `dataset_v2/` carry **zero** label
  columns (read-only scan against `LABEL_COLUMNS ∪ LABEL_CONTEXT_COLUMNS`).

**R87.** Every packaging assertion was a row count or a length, so nothing checked *which* ObsIds
landed in a fold. Now `test_packaged_folds_contain_exactly_the_split_obs_ids` asserts per-fold `obs_id`
set membership in all four parquets *and* in `groups_*.npy`, train/test disjointness, and
held-out-exactly-once across the scheme; `test_within_image_packaged_folds_contain_exactly_the_expected_tiles`
does the same for the (image, quadrant) arm on exact tile-key sets.

**R88.** Two halves, because one was not enough. Stage-5 side: the emitted X and y column sets are now
pinned exactly, per fold and per side. Loader side: `src/modeling/loaders.py::_feature_columns` gained
a `FORBIDDEN_X_COLUMNS` check on both the train and the test parquet. **It raises rather than
filtering** — a target in X means the package is corrupt, and quietly training on the rest would hide
the packaging bug that put it there.

Mutation verification, four mutants on independent scratch copies (working-tree `src/` never touched);
every one of them left the pre-existing suite green:

| mutant | now fails |
|---|---|
| LOIO `package_split` → random per-tile re-split, counts preserved | `test_packaged_folds_contain_exactly_the_split_obs_ids` |
| within-image → random per-tile re-split | `test_within_image_packaged_folds_contain_exactly_the_expected_tiles` |
| within-image → train rows from the **wrong image** | `test_within_image_packaged_folds_contain_exactly_the_expected_tiles` |
| `_split_columns` → drop `label_cols` from the exclusion set | `test_packaged_x_columns_are_exactly_the_expected_feature_set` + the within-image column guard |

Plus a data-side mutant: splicing `fractional_area` / `boulder_count` / `tile_size_m` into a packaged
X parquet now makes `load_fold` raise.

**No artifact impact.** The loader change is a read-side assertion that passes on every existing
package, so nothing enters `docs/PENDING_REBUILD.md`.

## 2026-08-06e — R27 fixed; R28 half-fixed, and both cited numbers checked against the data

Both were unverified in the register. Reproduced read-only against `dataset_v2` before touching
anything — no producer was called.

**R27 — CONFIRMED EXACTLY AS FILED, code fixed, rebuild pending.** 42,015 of 198,320 S ≥ 32 rows have
`lacunarity_shadow_b2` (and `_b4`) exactly `0.0`; **every one** has `shadow_fraction == 0`; the
smallest non-zero value is exactly `1.0`; **nothing** falls in `(0, 1)`. Lacunarity is ≥ 1 by
Cauchy–Schwarz, so `0.0` is unambiguously a sentinel. `_lacunarity_per_tile` now leaves the NaN
prefill in place for `M1 == 0`.

The downstream number needed correcting: the register said "12.6 % of one `features_nbr` file's
rows". Measured over all 38 `features_nbr_s5` files, the pooled share of `nbr_mean_lacunarity_*` in the
impossible interval `(0, 1)` is **2.16 %**, and the worst image is **`ESP_068402_2240` at 16.7 %**
(`ESP_076499_1160` is 13.2 %, which is presumably what the register saw). Mechanism confirmed, blast
radius per-image rather than uniform.

**R28 — mechanism CONFIRMED and stronger than filed; half the fix landed, half is Brian's call.**
skimage 0.26 maps `low_threshold=None` to the constant `0.1` and, with `use_quantiles=False`, applies
it as an **absolute** gradient magnitude. Measured: per-image `edge_density` vs `intensity_std` is
Spearman **ρ = 0.965** over the 38 images (register said 0.894 — different aggregation), a **12.2×**
cohort spread, **33.8 %** of `ESP_068402_2240`'s S=64 tiles with zero edge pixels. The cleanest
demonstration is synthetic: cut the DN spread ~3× and edge density goes 0.345 → 0.0026 (**×0.01**);
under quantile thresholds the same change gives **×1.00**.

Landed: `use_quantiles` is now a config key, `_compute_canny_window` raises if it is enabled without
explicit percentile thresholds, and the false comment ("None -> skimage chooses from gradient
magnitude") is gone from both configs and the data dictionary. **The default is deliberately
unchanged** — picking the quantile pair is a science decision that changes every `edge_*` value and
every number computed from one, so it is Brian's, not mine. Asked; tracked as PENDING_REBUILD row 3.

**A trap worth not rediscovering:** writing the current default into the config as `low_threshold: 0.1`
would *not* preserve behaviour. skimage maps `None` to 0.1 directly but divides an *explicit* threshold
by `dtype_max`, so on a uint8 window it becomes 0.1/255 — which passes nearly every gradient (density
0.345 → 0.384) and is contrast-invariant for entirely the wrong reason. There is a regression test
pinning the two as different.

Six new tests. Fast suite green; nothing regenerated.

## 2026-08-06f — R28 decision: Canny thresholds become quantiles 0.80 / 0.90

Brian chose the quantile pair via AskUserQuestion. `canny_edges` in `DEFAULT_FEATURES_CFG`,
`config.yaml` and `config_v2.yaml` is now `use_quantiles: true`, `low_threshold: 0.80`,
`high_threshold: 0.90` — the top 20 % of each frame's own gradient magnitudes are edge candidates and
the top 10 % are seeds.

**Why that pair.** It is gain-invariant (the ~3× DN-spread test moves synthetic edge density ×1.00,
versus ×0.01 under the old absolute constants) and its synthetic density, 0.130, sits mid-range of the
pre-fix 0.025–0.307 cohort spread, so it preserves roughly the current amount of signal rather than
thinning or flooding the edge map. 0.90/0.95 was the stricter alternative at density 0.062.

A new test asserts both YAMLs agree with `DEFAULT_FEATURES_CFG` on these three keys: `_deep_merge_defaults`
lets a YAML `features:` block override key-by-key, so a stale config would silently reinstate the
absolute thresholds for real runs while the unit tests stayed green on the default.

**Artifact impact** is [PENDING_REBUILD.md](docs/PENDING_REBUILD.md) row 3: `edge_density` and
`edge_orientation_entropy` change for every tile in all 38 images, and with them the six
`nbr_*_edge_*` Stage-6a derivatives and every GBM/W1 number or error-atlas panel computed from one.
Expect the per-image `edge_density` spread to shrink sharply — that is the fix working, not a
regression.

Also decided: **do not run the slow suite yet.** The test-side isolation gate is closed and the fast
loop is green at 526, but the four staged producer tests have not been executed since the fixture
changed; they will run as one batch before the rebuild rather than mid-fixing.

## 2026-08-06g — R97 CLOSED, and it inverts the R92 story: v1's split was right, the splitter drifted

`_compute_quadrant_definitions` snapped the finest-scale median to `max(SCALE_TO_FACTOR_FROM_FINEST
.values())`. Commit `29b0adb` ("CNN + S128 **HELD as dev-only**") added `128: 16` to that table, so a
scale **no shipped config emits** doubled the production snap step from 8 to 16. The step now comes
from the scales present in the image's own labels, intersected with the factor map; a table entry for
an absent scale is inert. The function also raises instead of returning `{}` when no present scale is
known — an empty dict reads downstream as "no tile belongs to any quadrant".

**Measured read-only before changing anything** (recompute per image, compare against the persisted
`quadrant_definitions` in each split JSON — they *are* persisted, per fold, which corrects the
2026-08-05 note saying quadrant definitions live nowhere):

| tree | persisted == step-16 recompute | persisted == step-8 recompute | cuts that move |
|---|---|---|---|
| `dataset_v2` | **38 / 38** | 9 / 38 | **29 of 38 images** |
| `dataset` (v1) | 3 / 8 | **8 / 8** | 6 of 9 |

So the v1 within-image split was built with the correct step and still reproduces exactly; the "543 of
27,307 S=32 tiles (1.99 %) disagree with today's splitter" figure recorded under R92/R97 was
disagreement **with the defect**, not evidence of v1 drift. v2, built after `29b0adb`, is the tree that
now goes stale. `PENDING_REBUILD.md` is corrected in both directions.

**Blast radius is within-image only.** The LOIO splits, the frozen recipe, the deployable head, the
calibration and the regional product do not use quadrant cuts. `dataset_v2/splits/within_image_4fold.json`
and `packaged/within_image_*` need a Stage-5 regeneration; nothing upstream of Stage 5 does.

Three tests: the production ladder (a fixture whose S=8 median deliberately snaps to 8 under the
correct step and 0 under the inflated one, with a guard asserting the two differ), a mixed set that
genuinely contains S=128 and must snap to 16, and rejection when no present scale is known. Reverting
the single line to `max(scale_to_factor.values())` on a scratch copy fails the first.

**The transferable lesson, and it is the third instance:** a lookup table is not a statement about
what a dataset contains. Extending one "for a dev experiment" changed a production constant because a
consumer reduced over the whole table instead of over the data. Two agents then read the constant as 8
and got the wrong answer twice.

## 2026-08-06h — R74 becomes a usable rebuild boundary: hole tests + a provenance chain

The R74 fix shipped 2026-08-04 with **no direct tests** and an implicit boundary: the threshold was a
default kwarg, and nothing in an artifact recorded which algorithm produced it. A config hash over the
YAML is *identical* either side of the change, and a pathname says nothing, so pre- and post-R74 masks
had indistinguishable sidecars. That is precisely the Pattern-D failure `PENDING_REBUILD.md` exists to
control, and it made R74 unusable as a rebuild boundary.

**Tests** — `tests/test_coverage_mask_shadow_fill.py`, ten cases, pure synthetic arrays: small enclosed
hole; hole above the threshold; the threshold boundary (`<=` fills, one pixel more does not); an
edge-connected invalid region; the mixed case where one enclosed puddle and one *small* edge-connected
gap coexist and exactly one may change; `max_interior_hole_px <= 0` as a bit-exact no-op that also does
not mutate its input; add-only-never-remove over random fields at three hole densities, with
`n_filled` cross-checked against the mask delta; all-valid and all-nodata.

**Provenance chain** — the identity now travels with the data rather than with the commit:

| stage | records |
|---|---|
| 2 | `ctx_window_sha256`, and `hirise_mask{method, version, max_interior_hole_px, n_interior_shadow_px_filled, coverage_fraction, sha256}` |
| 3 | `inputs{ctx_window_sha256, hirise_mask_tif, hirise_mask_sha256, coverage_mask}` + a `shift_id` digest over its shift and inputs (timestamp excluded, so a re-solve on unchanged inputs is stable) |
| 4 | `inputs{ctx_window_sha256, hirise_mask_sha256, coverage_mask, coreg_shift_id}` |

`COVERAGE_MASK_VERSION = 2` (1 = pre-R74) must be bumped whenever the output can change for unchanged
inputs. The threshold is now `ctx_retrieve.max_interior_hole_px` in both YAMLs, wired through
`scripts/run_stage2.py` rather than living in a default nobody can see from the artifact.
`build_hirise_coverage_mask` returns a third element; the only caller is `stage2_one_image`.

Stage 3 binds to the mask because it *uses* it — the coverage mask selects the FFT window and gates the
block field, so a pre-R74 and a post-R74 mask can yield different shifts from identical config.

The sharpest test flips a single pixel in the coverage mask and asserts the Stage 4 sidecar's recorded
digest moves while its `config_hash` does not.

**Every sidecar on disk predates these fields**, so the absence of `inputs` / `hirise_mask` is itself
the marker of a pre-2026-08-06 generation — useful during the rebuild.

`pytest -m "not slow"`: 540 passed, 21 deselected; artifact manifest unchanged. No producer was run.

## 2026-08-06i — R04 CLOSED: Stage 5 fails loudly, and a stale package can no longer hide

Two halves, and the review was right that the second is the consequential one.

**The exit code.** `_run_one` swallowed `build_split`'s `ValueError`, `main` discarded its return and
`return 0`d unconditionally, so `raise SystemExit(main())` reported success. `main` now collects the
failed schemes, returns 1, and says explicitly that any existing `packaged/` output for them is stale.

**The staleness detector.** A package recorded its scheme name, `split_hash` and `config_hash`; none
of the three can see that the *contents* of `labels/` changed underneath it — which is exactly the
pre-R74 case, where the config is identical and only the label row set moves. `package_split` and
`_package_within_image_split` now record `source_digests`: a SHA-256 per labels and features parquet,
each label sidecar's R74 `inputs` block, and one rolled-up digest.
`loaders.verify_package_freshness` runs from `load_metadata` and therefore from `load_fold`, and
raises `StalePackageError` on:

1. `split_hash` disagreeing with `splits/{scheme}.json`;
2. the package's cohort disagreeing with the ObsIds in `labels/`, allowing the scheme's declared
   `excluded_obs_ids` — this is the R04 scenario, a cohort expansion whose Stage 5 run failed;
3. recorded source digests disagreeing with the files on disk.

Verification is cached per `(scheme, dataset_dir)` per process, so a 38-fold sweep hashes once, and
`force=True` re-checks.

**Why it warns rather than raises for older packages.** Every package on disk predates
`source_digests`. Verified read-only: all seven (`dataset/{loio_9fold, loio_3fold_balanced,
within_image_4fold}`, `dataset_v2/{loio_nfold, loio_nfold_ctx_illum, loio_nfold_nbr_s5,
within_image_4fold}`) pass checks 1 and 2 and warn on 3. Bricking every existing artifact would have
been worse than the defect; naming them as unverifiable is the honest middle. The v1 within-image
package legitimately covers 8 of 9 ObsIds, which is why the cohort check consults `excluded_obs_ids`.

Ten tests, notably: the driver returning 1 on a build failure and 0 on a healthy run; the
cohort-expansion scenario; and a fixed-cohort label-content change where the test asserts the split
hash and config hash are provably unmoved, so only the content digest can see it.

`pytest -m "not slow"`: 551 passed, 21 deselected; artifact manifest unchanged.

## 2026-08-06j — isolation criterion 4, first tranche: the calibration banker

`scripts/bank_calibration.py` was the audit's named offender and had three defects, all now fixed:

1. **Every path was hard-coded** (`models/fang_probe/.../predictions.parquet`,
   `dataset_v2/labels`, `models/deployable/calibration.npz`), so a scratch rebuild could not fit a
   calibrator without writing the live `models/` tree. All three are flags now, and `--out` may point
   anywhere.
2. **`layer.save(OUT)` ran before the gates were evaluated, and `main` returned 0 regardless.** A run
   that printed `Tier-1 ECE ... FAIL` still overwrote the banked calibrator *and* reported success —
   a fail-open on the artifact the deployed map depends on. Gates are computed first; a failing run
   writes nothing, leaves the existing layer untouched, and exits 1. `--force` banks anyway and
   records `gates_passed: false, forced: true` in the layer's meta so the artifact confesses.
3. **The predictions↔labels join was a bare `how="inner"`.** Any key that failed to join silently left
   the calibration pool — exactly how an R74-recovered or dropped tile disappears unnoticed. It now
   refuses duplicate keys on either side, uses `validate="one_to_one"`, and on an incomplete join
   raises with a sample of the orphan keys and the instruction to re-run LOIO rather than calibrate on
   the intersection.

Eight tests, including: a failing gate leaves a pre-existing layer byte-identical; `--force` records
that it was forced; and both join-integrity refusals.

**Also pinned: the isolation recipe itself.** `Config.resolve` is `(root / raw).resolve()`, and
`Path.__truediv__` discards the left operand when the right is absolute — so *absolute*
`cache_dir`/`output_dir` values in a config genuinely redirect, while relative ones resolve against
`REPO_ROOT` no matter where the YAML lives. Both directions now have assertions in
`tests/test_artifact_isolation.py`; the difference is a scratch rebuild versus overwriting the live
tree, which is too load-bearing to leave as a comment.

Criterion 4 is **not** finished: the Stage 1–5 drivers are config-parameterized (and the absolute-root
recipe above covers them), but the embedding, LOIO-prediction, head and map scripts still hard-code
roots. Criterion 5 (an independent backup of the ignored trees) is untouched.

`pytest -m "not slow"`: 560 passed, 21 deselected; artifact manifest unchanged.

## 2026-08-06k — isolation criterion 4 finished, and an A1 provenance record that lied

**The bug worth naming.** `scripts/striping_a1_map.py` accepts `--head`, and its *region* manifest
recorded `args.head` correctly — but the *per-tile* sidecar recorded the `A1_HEAD` **constant**
(line 190). So a `--head <other>` run produced two provenance records that contradicted each other,
with nothing flagging the disagreement. This is the audit's "the A1 path can report the global default
head even when a different command-line head is used", confirmed by reading and now fixed.

Both A1 records, and both baseline map records, now carry `head`, `head_digest`, `calibration` and
`calibration_digest`. **The baseline tile sidecar previously recorded neither** — only
`calibrated: true/false` — so a `reports/map_region/*.json` could not be traced to the artifacts that
produced it, and two tiles rendered from different heads were indistinguishable. New
`src.mapping.artifact_digest` hashes a file *or* a directory (sorted relative-path + content), because
a `DeployableHead` is a directory and a `CalibrationLayer` is one `.npz`. Digests rather than paths:
a path can be overwritten in place, and a head directory's name is a hash of the *training recipe*,
not of the weights that came out of it.

Also fixed while there: `map_region.py`'s manifest called `model_dir.relative_to(REPO_ROOT)`, which
raises for any head outside the repo — i.e. for exactly the scratch head a rebuild would use.

**Criterion 4 roots, now all flags:**

| script | new flags |
|---|---|
| `map_region.py` | `--ctx-tiles`, `--model-parent` (had `--out-dir`, `--model`, `--calibration`) |
| `map_pilot.py` | `--ctx-windows`, `--ctx-tiles`, `--model-parent`, `--out-map`, `--out-fig`; dead `DATASET_DIR` removed |
| `parity_check.py` | `--ctx-tiles`, `--model-parent` |
| `train_deployable_head.py` | `--dataset-dir` |
| `striping_a1_loio.py` | `--dataset-dir`, `--out-dir` |
| `striping_a1_map.py` | already had `--head`/`--out-dir`; the provenance now matches them |
| `bank_calibration.py` | done 2026-08-06j |

Module constants are kept as the argparse *defaults*, so existing invocations are unchanged. Verified:
all seven import and `--help` cleanly (which exercises the module-level code and the new defaults).

**Left open deliberately.** `resolve_model_dir` still picks `hits[-1]` — the lexicographically last
head — when `--model` is omitted. That is choosing a head by *name*, not by compatibility with the
calibrator or the preprocessing arm, and it is a separate open finding in the audit's "Product
semantics" section. A note now says so at the function rather than in a document nobody reads at the
call site.

`pytest -m "not slow"`: 560 passed, 21 deselected; artifact manifest unchanged.

## 2026-08-06l — the full suite runs non-mutating, by construction

`pytest` (no marker filter) → **581 passed** (560 fast + 21 slow) with an 11,218-file
path/size/mtime manifest over `cache`, `cache_v2`, `dataset`, `dataset_v2`, `models` and `reports`
**bit-identical before and after**. 98.9 s.

This is not the same claim as 2026-08-05's. That run was clean because the cached CRS happened to
match, so the invalidation branch was never taken, and because rasterio happens to delete-then-create;
nothing prevented a write. This one is clean because the writes are refused: mutable derived artifacts
are copied rather than linked, a session-wide guard rejects any write under an artifact root, and a
static scan fails on a producer call handed a live root even when that test skips. The manifest is now
corroboration, not the control.

Two things the run confirmed that only a real execution could:

- the reworked `read_only_cache` `only=` filter stages everything Stage 2 and Stage 3 actually read.
  That was the one thing I could not verify without running it — I had only a read-only listing of
  which filenames match.
- the R04 freshness guard fires in situ: `test_run_loio_classification_end_to_end_on_real_fold`
  emitted `packaged/loio_9fold predates source-digest provenance`, exactly as designed — a warning on
  a legacy package rather than a failure.

**Still not covered, and worth repeating because it is now the only hole:** the guard is *test-only*.
A producer invoked by hand from a notebook or a script against a repository root is unaffected by any
of this. The controls there are the absolute-scratch-root recipe (asserted in
`tests/test_artifact_isolation.py`) and, before the rebuild, criterion 5's backup.

## 2026-08-06m — criterion 5 deferred to a drive; recovery plan written, and a gap it exposed

Brian is getting an external drive, so the backup is deferred and
[docs/ARTIFACT_RECOVERY.md](docs/ARTIFACT_RECOVERY.md) is the interim record: what could be
re-downloaded, what could be regenerated, and what could not.

**The gap the exercise exposed.** The BoulderNet detection shapefiles — the ground truth every label,
model, metric and map in this project derives from — live *outside* the repo at
`../hirise_priority10_detections` (0.01 GB) and `../hirise_40_vClaire` (**4.17 GB**), which means they
were outside every artifact manifest taken during this entire review. All six roots I have been
diffing before and after each test run are the *derived* trees. The irreplaceable input was never
being watched.

**Tiers, all measured:**

- **Tier 0, ≈30 GB, unrecoverable.** The detections (4.18 GB); `dataset/` v1 (5.0 GB, explicitly
  non-reproducible under current code — regenerating it destroys what R81 preserves); the F-programme
  reports (20.2 GB from ~333 Sherlock CPU-h on a CLOSED programme, so the only surviving evidence for
  the HARD ABORT); current figures and `map_region` (0.98 GB).
- **Tier 1, ≈64 GB, re-downloadable.** CTX tile zips (41.4 GB, Murray Lab template in both configs,
  and `ensure_tile_cached` rebuilds the sidecars from the zip header); HiRISE JP2s (19.8 GB, `JP2_URL`
  column of the git-tracked manifests); MOLA/THEMIS (2.4 GB); craters; PDS index/labels. The one soft
  spot is `models/pretrained` — the Fang ViT checkpoint's download URL is not recorded anywhere in the
  repo.
- **Tier 2, ≈84 GB, regenerable — with an asterisk.** Re-running today's code does not reproduce
  today's artifacts: R74, R27, R28 and R97 all changed producer behaviour. Tier 2 recovers *a* dataset,
  not *this* one. For any cited value, treat it as Tier 0.

**Recommended stopgap, not yet executed:** ~11.3 GB of the small Tier-0 set (detections, `dataset/`,
trained models, current figures) copied onto C:. A same-volume copy is not disaster protection, but it
does defend against the failure that actually happened twice here — a producer silently overwriting a
live artifact — and 600 GB is free.

Criterion 5 stays **open**. A recovery plan is not a backup, and the rebuild must not start on one.

## 2026-08-06n — `dataset/` v1 is expendable; backup deferred to a drive

Brian: v1 does not need saving. So the 2026-08-06 decision "superseded, will not be rebuilt, preserve
as a frozen historical artifact" is amended — it is **not preserved either**. Not backed up, and
losing it is accepted.

Worth stating the cost once so nobody rediscovers it as a surprise: v1 is non-reproducible (pre-y-sign
-fix), so every v1 measurement becomes **unre-verifiable** — R81's 236–493 m label offset, R92/R97's
"v1 matches a step-8 recompute 8/8", the 2026-08-04 incident's `max|Δfa|` 0.115 across 3,854 of 96,354
rows. Those *conclusions* live in git-tracked `DECISIONS.md` and the review register; what dies is the
ability to re-derive them from the tree. Nothing current reads v1, so that is a reasonable trade.

The backup itself is deferred until an external drive arrives — the machine has one volume (C:, 600 GB
free) and no removable media. [docs/ARTIFACT_RECOVERY.md](docs/ARTIFACT_RECOVERY.md) now lists the
drive-day set at ≈105 GB, of which **6.3 GB is the small precious core**: the detections (4.18 GB),
trained models minus `pretrained` (1.1 GB), current figures + `map_region` (0.98 GB). v1's 5.0 GB is
dropped from it.

**Isolation criterion 5 remains OPEN.** A recovery plan is not a backup, and the rebuild must not start
on one.

## 2026-08-06 — session close

Sixteen commits, all on `origin/fm-deployable-head-and-map-pilot`. Worktree clean.

**Closed:** R77 (residual), R78, R87, R88, R27, R28, R97, R74 tests+provenance, R04, and isolation
criteria 1–4. Full suite **581 passed** with the artifact manifest bit-identical — non-mutating by
construction rather than by luck.

**Four things this session found that were not in the register:**

1. The audit's R77 mechanism was half wrong. `rasterio.open(p,"w")` deletes-then-creates, so the
   predicted truncation never fires; `open(p,"wb")`, `"r+"`, `write_text` and `copy2` all do. A latent
   design error masked by library behaviour — and *both* "the suite is safe" and "the live TIFF gets
   truncated" were wrong.
2. R97 inverts R92: v1's within-image split matches a step-8 recompute **8/8** and v2 matches the
   inflated step-16 **38/38**. v1 was never drifted; the splitter was.
3. `striping_a1_map.py`'s per-tile sidecar recorded the head *constant* while its region manifest
   recorded `args.head` — two provenance records contradicting each other under `--head`. The baseline
   map's tile sidecar recorded no head or calibration at all.
4. The BoulderNet detections (4.18 GB) live outside the repo and were outside every artifact manifest
   taken during the review. The one irreplaceable input was never being watched.

**Open, in order:** isolation criterion 5 (backup, awaiting a drive — the only remaining *safety*
gate); then **R56 → R23** (see the audit's "Start the next session at step 4"); then
R31/R67/R65/R29/R68 for Stages 2–4; then the A1/map blockers R07/R08/R38/R01/R13/R14. Do not start
the rebuild.

Loose ends: `models/pretrained`'s Fang ViT download URL is recorded nowhere; README/ROADMAP/SHERLOCK
still carry pre-audit claims; the runtime write guard is test-only, so notebooks are uncovered.

## 2026-08-06o — R56 re-scored (verdict withdrawn) and R23's root cause found: three `.shp` files are byte-truncated

Audit step 4. Both results are read-only measurements on banked artifacts; no producer ran, no live
artifact changed.

### R56 — the `min_confidence` verdict was a two-factor comparison, and it does not survive

`_diag_tier2_minconf_sweep.py` trained on labels regenerated at `score >= t` **and scored against
those same regenerated labels**, so its paired per-image Wilcoxon compared `rho(pred_t, y_t)` against
`rho(pred_none, y_none)` — two predictors *and* two targets. Re-scored against **one common target**
(the unfiltered `fractional_area`, verified bit-identical to the shipped `dataset_v2` S=32 labels:
0 key misses, max |diff| = 0.000e+00) on the banked per-tile parquets, 161,005 tiles × 38 images,
keys row-aligned. The reviewer's three-way decomposition reproduces to 4 dp on all six values.

| factor varied (conf ≥ 0.5) | median Δ per-image ρ | wins | p |
|---|---|---|---|
| both (as the probe measured, and what the record quotes) | −0.0210 | 11/38 | 0.0100 |
| **target only** (fixed model, filtered target) | **−0.0172** | 7/38 | **0.0002** |
| **training-label only** (filtered model, COMMON target) | **−0.0034** | 17/38 | **0.4294** |

**82 % of the recorded "harm" is the target moving.** The deliverable — the fixed-target re-score on
the project's standard metrics (`src.modeling.evaluate.per_fold_metrics`, `meaningful_threshold=1e-2`;
no presence AUC anywhere; `n_dropped = 0`, no silent fold loss):

| metric (paired vs `none`, COMMON target, n=38) | conf ≥ 0.5 | conf ≥ 0.7 |
|---|---|---|
| `meaningful_auc` | −0.0028 (15/38, **p=0.314**) | −0.0104 (13/38, p=0.014) |
| `pr_auc@1e-2` | −0.0007 (17/38, **p=0.438**) | −0.0068 (13/38, p=0.015) |
| `precision@5%` | 0.0000 (14/38, **p=0.411**) | −0.0096 (11/38, p=0.018) |
| Spearman ρ | −0.0034 (17/38, **p=0.429**) | −0.0213 (13/38, **p=0.061**) |

Pooled per-bin RMSE and the `rmse_*` rows are reported in the run artifact but must be read as
marginal-calibration diagnostics, not ranking: an arm trained on a 35–77 % smaller-mass target
predicts systematically lower (rich-bin `mean_pred` 0.0247 → 0.0156 → 0.0052 against `mean_true`
0.0373). The scale-invariant metrics above are the clean read.

- **conf ≥ 0.5 is a null**, min p over nine metrics = 0.178 (0.257 under Pratt). It is a *bounded*
  null on the AUC family (95 % CI of the mean Δ `meaningful_auc` [−0.0116, +0.0029]), not merely an
  underpowered one — though the Spearman CI [−0.0186, +0.0099] is wide. Caveat recorded: the **pooled**
  `precision@5%` loss (−0.0143) *is* distinguishable from zero under an image-clustered bootstrap
  (p ≈ 0.03–0.04), so the null holds under equal-per-image weighting, which is the project's
  group-aware reporting basis.
- **conf ≥ 0.7 is directionally harmful** and consistent in sign across all nine metrics, but **no
  metric survives Holm** across the nine (all adj. p ≈ 0.12); across just the four reporting-standard
  metrics the smallest adjusted p is 0.054. Report it as directional, not established.
- **"Monotonically degrades ranking" is false in the probe's own banked scorecard:** per-image ρ
  0.4333 → **0.4563** → 0.3044.
- **"Degrades dynamic range" is a population artefact plus a missing calibrator.** `top_ratio` was read
  at a fixed `fa > 1e-2` while the filter moved the rich share 36 %→27 %→11 %. At a **matched 36 % top
  fraction**: 0.6637 → 0.6231 → 0.5191 (53–59 % of the collapse is the population change). And after
  the **shipped** quantile-match layer, against the common target: **0.8699 → 0.8595 → 0.8294** —
  essentially flat. The record's 0.66 → 0.58 → 0.31 is a raw-marginal number the product never ships.
- Excluding the two R23-truncated images (`ESP_017355_2260` retained mass **1.000000** — literally
  untreated — and `ESP_068483_2280` 0.803) **strengthens** the null (p 0.314→0.550, 0.429→0.681).
  Under LOIO an image's own retained mass governs its *target* treatment, not the training treatment
  of the model scoring it.

**What survives:** filtering *thins* the target rather than cleaning it — the target factor is real and
significant on its own — so nothing shows filtering **helps**, and keeping `min_confidence: null`
remains defensible. **What is withdrawn:** "monotonically degrades", "harmful at conf ≥ 0.5", the
dynamic-range collapse as stated, and the causal claim that this proves low-confidence detections are
real boulders. **The ruling no longer forbids harmonising a confidence floor.**

**But it does not license R23's 0.6173 either.** No arm was ever measured there. 0.6173 sits 45–59 %
of the way from 0.5 to 0.7 (basis-dependent) and interpolates to **56–70 % of the conf070 harm**; three
headline metrics are non-monotone across the arms, so interpolation is not even guaranteed to bracket.

### R23 — the root cause is not BoulderNet. Three `.shp` files were never fully copied.

The register frames R23 as a "score-rank truncation of the detection set" in the upstream export. It is
not. Measured directly on the source shapefiles (read-only, header + `.shx` arithmetic):

| ObsId | `.shp` header declares | on disk | missing | `.dbf` / `.shx` | records whose bytes FIT | pipeline kept | Δ |
|---|---|---|---|---|---|---|---|
| ESP_017355_2260 | 569,266,636 B | 214,884,317 B | **354.4 MB** | complete (1,105,447) | **359,933** | 359,933 | **0** |
| ESP_046803_2325 | 323,962,020 B | 192,091,266 B | **131.9 MB** | complete (658,290) | **367,140** | 367,140 | **0** |
| ESP_068483_2280 | 616,023,244 B | 443,015,777 B | **173.0 MB** | complete (1,057,153) | **727,160** | 727,160 | **0** |

Three control images (`ESP_045139_2270`, `ESP_054622_2240`, `ESP_076499_1160`) are byte-exact complete.
The records are stored **score-descending** (verified over the full `score` column of all three files),
so the records whose bytes survive are exactly the highest-scoring prefix — and the measured cut is
`0.617257 / 0.473420 / 0.406699`, matching the register's floors. The `.shx` index and `.dbf` are
complete, which is why GDAL returns all ~1.1 M rows and the missing tail appears as *null geometry*:
`drop_null_geometries` was faithfully reporting a **truncated file**, not an export artefact.

**Consequences.** (i) The "benign density hygiene" reading in `DECISIONS.md:1194` was wrong for a
different reason than the register says. (ii) The register's rounded floor **0.6173 is 4.25e-05 above**
the true max-of-minima `0.617257475852966`, so "harmonise exactly" at 0.6173 would itself drop 86
polygons from `ESP_017355_2260`. (iii) `ESP_046803_2325`'s exclusion is unrelated — `DECISIONS.md:1258`
records a coregistration failure. (iv) A **fourth remedy** exists that the register does not list:
**re-copy the 659 MB of missing bytes**, which restores the data instead of working around it. Whether
that is possible depends on a complete source copy existing off this disk — nothing in the repo can
establish that, and the audit already flags these 4.18 GB as outside every artifact manifest and not
backed up.

Remedy pricing (read-only; corrected by adversarial verification — several first-pass numbers were
wrong and are not carried forward). At the true floor 0.617257 the 36 unaffected images retain a median
**33.1 %** of kept detection area (**53 %** of cohort label mass discarded, not the 66 % first
reported); ~24,000 of 161,005 tiles change rich/poor class (est., band 23,550–30,657). Excluding the
truncated images costs **2** images (not 3), 18,754 tiles (11.65 %), 21.9 % of rich tiles, 19.9 % of
labelled boulder mass, and drops LOIO 38→36 folds; `ESP_017355_2260` is confirmed the largest
observation at 13,457 S=32 tiles (1.84× the runner-up). Retaining and documenting the mixed floor keeps
~2,000–2,250 mis-classed tiles (1.2–1.4 %) and leaves all 18,754 tiles in those two folds carrying a
level-biased target (low by ≈2.4× and ≈1.35× on own-anchored estimates, not the transported 2.6×/1.46×).

**Also corrected:** confidence filtering is a **Stage 4** operation (`src/labeling.py:96`
`_apply_detection_filters`, called at `:496` inside `stage4_one_image`), not Stage 1 — `DECISIONS.md:813`
already said so. Only the *provenance* half of R23's fix (recording dropped-vs-kept score distributions
and flagging rank truncation in `drop_null_geometries`) is Stage 1. And R76's 41.3 % is a flip rate on
an 8-image prevalence-matched surrogate of **clean** images, never applied to `ESP_017355_2260`; its
~2,200 has denominator all 13,457 tiles.

### R23 REMEDY — DECIDED (Brian, 2026-08-06): **retain and document, temporarily**

> "Not sure if the non-truncated version exists. The end goal is to get new detections (i.e. a V3
> dataset), so for now we can just have a temporary solution of retaining for now and documenting
> this." Fallback if recovery proves impossible: **decide later**.

So the mixed confidence floor is **retained**, exactly as the mixed *size* floor was (2026-08-06
decision 2), and marked **temporary pending the v3 re-detection** — not adopted as a target
definition. Harmonising to 0.617257, excluding the two images, and the byte-range recovery all stay
on the table and are priced above. What shipped for the "document" half:

- **`src/detections.inspect_shapefile_integrity`** — asks the `.shp` header whether the file is
  byte-complete. **`src/detections.describe_null_geometry_drop`** — characterises the dropped
  population on `score` and sets `is_rank_truncation` when every dropped row scores at or below
  every kept row. `stage1_one_image` calls both, **warns loudly** on either finding (it does not
  raise — the cohort is retained by decision), and persists both to the Stage-1 sidecar as
  `source_integrity` / `null_geometry_basis`.
- **`src/labeling._describe_realised_label_basis`** → the Stage-4 sidecar's new
  **`realised_label_basis`**, carrying the per-image `realised_score_floor` plus
  `level_claims_unsafe` + a note for affected images. This is the field that makes the mixed floor
  visible downstream: `detection_filters` is byte-identical across all 38 sidecars and structurally
  cannot express it.
- **`dataset/DATA_DICTIONARY.md`** documents all three blocks, and two long-standing errors there
  are corrected: `n_polygons_after_filter` "equal when both are null, the current default" (both
  configs set `min_size_m: 1.4105`), and the `score` range — **measured 0.100000–0.955996** over the
  39 readable v2 exports (7,645,643 detections), not "0.10–0.83".
- 9 new tests in `tests/test_detections_reprojection.py`. Fast suite **569 passed**, 21 deselected.

**A fourth truncated file, and why the existing gate missed three.** Running the new check over all
40 vClaire exports finds **`ESP_028537_2270` truncated too** — 513.4 MB of 571.9 MB missing (90 %).
That one was already known and excluded (`DECISIONS.md:1190`, `config_v2.yaml:6`), and
`scripts/build_vclaire_manifest.py` was written with an integrity gate specifically for it. **The
gate checked the wrong file.** It validates `.dbf` self-consistency, `.dbf`↔`.shx` record agreement,
and pyogrio's feature count against `.shx` (`:139`, `:158`) — and *all three pass* on a shapefile
whose `.shp` alone is short, because the `.shx` is intact and GDAL reports every record it indexes.
`ESP_028537_2270` was caught only because its `.dbf` was truncated as well. The gate now also calls
`inspect_shapefile_integrity` and fails a folder on `shp_status != "complete"`.
So the same failure mode was diagnosed once, given a name, and given a check — and three further
instances of it still reached the shipped label basis. Corroboration: every readable export's `.dbf`
bottoms out at exactly `0.100000`, including all three truncated ones, so the low-score tail exists
in the tables and is missing only from the geometry.

**Still open on R23:** whether a complete source copy exists off this disk (recovery), and the
product-level statement of the mixture for reader-facing docs. The two affected images should be
excluded from per-image **level** claims (calibration pool, R54's `mean(pred)/mean(true)`,
PLAN_RegionalMap's thermal legs) while remaining valid for rank-only statistics —
`realised_label_basis.level_claims_unsafe` is the machine-readable flag for that, but no consumer
reads it yet. Also: **zero banked sidecars carry any of the new provenance**, so Stage 1 + Stage 4
must re-run before the mixed floor is documented *on disk*. That re-run changes nothing numeric —
verified that `src.dataset.source_digests` is bit-identical with and without the new key, so R04's
content digests and the 7 live packages are untouched.

### Adversarial review of the above — and the bug it caught in the fix itself

Four independent lenses over the diff (correctness / behaviour-change / gap-hunt / docs-honesty).
The measurements survived: every headline number was re-derived from the artifacts, the `.shx`
arithmetic was validated against an independent parse and against geopandas' own non-null count at
21 truncation points, and the R56 re-score reproduced to 4 dp. Two findings were load-bearing and
both are fixed:

1. **HIGH — the fix inverted the decision it was implementing.** Folding `shp_ok` into
   `integrity_check`'s `ok` flag turned the new detector into a *cohort gate*: the next manifest
   rebuild would have written **36 rows instead of 39**, silently deleting the three images Brian
   had just decided to retain. `ok` now means "can this folder be read at all" (a truncated `.shp`
   is readable); truncation is reported, printed loudly, and carried as manifest provenance instead.
   Verified against the real data: 39 kept, set-identical to the committed manifest, with the three
   truncated images retained and `ESP_028537_2270` (unreadable `.dbf`) still excluded.
2. **HIGH — the safety flag could never fire.** `level_claims_unsafe` was derived solely from the
   Stage-1 sidecar's `source_integrity`, which **no banked sidecar has**, so it would have been
   absent on exactly the affected images — and absence is indistinguishable from "checked and
   clean". It is now derived from the *realised floor itself* (measured from the labels in hand)
   against BoulderNet's own 0.100000 detector floor, corroborated by Stage 1 where available and
   re-derived from the sidecar's recorded `source_path` where not. `source_truncated` is now
   tri-state, so unknown never reads as safe. Verified on the live pre-2026-08-06 cache:
   `level_claims_unsafe=True` for both truncated images, `None` for a clean control.

Also fixed from the review: `describe_null_geometry_drop` no longer raises on a non-numeric `score`
column (it ran unguarded inside the producer, against the manifest-driven invariant); a
`_MIN_DROPPED_FOR_RANK_VERDICT = 100` guard stops tied or single-row drops raising a false
"LEVEL is biased low" alarm, and the kept-minus-dropped gap is recorded; `actual > declared` and a
sub-header declared length are now `length_mismatch` / `suspect_header` rather than `complete`; a
short `.shx` marks `n_records_present` a lower bound; and `drop_null_geometries`' docstring no longer
teaches that ~67 % null geometry is normal for dense exports — the mental model that hid R23.
Six wiring/edge tests added on top of the original nine (the review correctly noted the whole
integration could have been deleted with the suite still green). **Fast suite: 575 passed, 21
deselected.**

Confirmed clean by the review and worth not re-deriving: the change is provenance-only —
`src/labeling.py` and the test file have zero deleted lines, the only deleted line in
`src/detections.py` is a docstring terminator, `drop_null_geometries` is byte-identical, the new
Stage-4 key sits outside `inputs` so R04 digests do not move, every sidecar reader uses `.get`/named
access, and there is no `filterwarnings = error` anywhere to trip on the new warnings. The v1
priority10 detection set was scanned and is **clean** — all 10 `.shp`/`.shx`/`.dbf`/`.prj` complete.
A short `.shx`, `.dbf` or `.prj` all raise loudly at read time, so the `.shp` genuinely is the unique
silent failure and targeting it alone is the right call.

## 2026-08-06p — R29/R75 FIXED: the coverage mask now moves with the polygons

Audit step 5, Stage 4. Stage 4 translated every detection polygon by the Stage-3 `(dx, dy)` but gated
eligibility with a coverage mask reprojected from the **unshifted** HiRISE product. The shift is a
whole-product geolocation offset, so the two must move together; leaving the mask still opened an
L-shaped strip along the receding edges that stayed `eligible` while no detection could land in it.

**Fix:** `src/labeling._shift_coverage_mask` translates the mask by the same `(dx, dy)` before
eligibility gating, filling vacated area with 0 (**not** eligible — we have no coverage evidence
there). Applied in `stage4_one_image` between the sanity check and `_build_finest_stats`; opt-out via
`shift_coverage_mask=False`; recorded in the label sidecar as `coreg_mask_shift` (method, version,
shift in px, residual, eligible-pixel count before/after) so pre- and post-fix labels are
distinguishable — without it they are not, which is the Pattern-D failure R74 was also about.

**Correction to the register.** R29's fix bullet says the shifts are "already quantised to CTX
pixels". **They are not** — measured 2026-08-06: **0 of 39** are integer-pixel. They are quantised to
1/20 px by the phase-correlation upsampling (`dx/px` values like 35.1000, 21.2750, 8.2000). A raster
mask cannot move sub-pixel without resampling, so the shift rounds to nearest whole pixel; the
residual is **≤ 0.5 px (2.5 m)** against a median shift of 194.7 m, a ~78× reduction and far below the
160 m tile.

**Validated read-only against R75, via a different route** (shift the mask and recompute eligibility,
vs R75's `mask ∧ ¬shift(mask)` + integral image), over all 38 images at S=32:

| quantity | measured | R75 |
|---|---|---|
| tiles losing eligibility (overlap the vacated strip) | **6,202 (3.85 %)** | **6,202 (3.85 %)** ✓ exact |
| …of which currently `fa == 0` | 1,287 | (340 are *fully* inside the strip — a subset) |
| …of which currently `fa > 0` | 4,915 | the partial-depression population |
| `drow ≤ 0` (northward) | **38/38** | dy>0 in 38/38 ✓ |
| `dcol > 0` (eastward) | 29/38 | dx>0 in 30/38 (off by one; one image rounds to dcol 0) |

**The eligible set is re-registered, not shrunk.** 6,202 tiles drop out on the receding edges and
**6,255 new tiles become eligible on the advancing edges** — where the HiRISE footprint actually is —
for a net **161,005 → 161,058** at S=32. The newly eligible tiles have no labels in the current
parquets, so materialising them requires the Stage-4 re-run already in the rebuild DAG.

7 tests in `tests/test_labeling.py`, including one that pins the **sign** (north = decreasing row):
flip it and the mask moves the wrong way, doubling the misalignment instead of removing it.
Fast suite: **582 passed**, 21 deselected.

## 2026-08-06q — R31 FIXED: a cropped window read can no longer be stamped with the un-cropped transform

Audit step 5, Stage 2. `extract_ctx_window` built a rasterio `Window` from the requested bounds and
then did `src.read(window=window)` — which silently **crops** an overhanging window to the dataset —
while stamping the output with `src.window_transform(window)` on the **un-cropped** window. So a
west/north overhang wrote real in-tile pixels carrying the overhang's coordinates.

**Which directions are actually broken (measured on a synthetic 400×300 px tile, pixel value encoding
its own row/col so the landed source pixel is recoverable):**

| overhang | georeferencing error | also truncated? |
|---|---|---|
| **west** (300 px) | **−1500.0 m** | yes, 400 → 100 cols |
| **north** (200 px) | **+1000.0 m** | yes, 400 → 200 rows |
| east | 0.0 m — correct | yes, silently short |
| south | 0.0 m — correct | yes, silently short |

So west/north are **mis-georeferenced**; east/south are correctly georeferenced but silently *short* —
missing tiles rather than wrong ones. This is also why the audit's "reconstructing bounds from array
shape cannot repair west/north overhang" is right, and now with a mechanism: for west/north the shape
is right and the **origin** is wrong, so shape × wrong-transform reproduces the falsified bounds
exactly — which is precisely what `stage2_one_image` then filed as provenance.

**Fix.** Clip once (`window.crop(src.height, src.width)` — note the argument order; `crop(width,
height)` silently returns a different window) and derive **both** the read and the transform from the
clipped window. Production **raises** on any overhang rather than writing a window that would be
mis-georeferenced or short; a test-only `_allow_partial_tile` hatch exercises the corrected transform
derivation, and a test asserts nothing in `src/` or `scripts/` ever passes it. Plus defence in depth in
`stage2_one_image`: the written bounds must match the requested bounds to 1e-3 m (measured drift on all
49 cached windows is < 1e-6 m).

`crop` not `intersection`: measured, `intersection` raises rasterio's own `WindowError("Intersection is
empty")` when the window is fully disjoint — exactly the "manifest names the wrong CTX_TileName" case —
so the caller would get a generic message instead of the actionable one. `crop` returns a zero-size
window and lets our raise fire.

**Blast radius: none.** Re-derived read-only over every cached window: **cache_v2 39/39 pass, 0 would
raise** — the fix is a genuine no-op for the v2 rebuild and no shipped v2 label is mis-georeferenced by
R31. In v1, **9/10 pass and exactly one raises**: `ESP_057469_2215`, requested `col_off = −1924`
(a 1,924 px = **9,620 m** west overhang) cropped to 204 columns — the register's number, reproduced.
That image is already excluded, and v1 is superseded and not being rebuilt; a v1 Stage-2 re-run would
now abort on it, which is the correct outcome for a window that was producing garbage.

Worth keeping: the a-priori hazard is not small. Murray tiles are 237.1 × 237.1 km and the v2 windows
are median 11.5 × 17.4 km, so for a uniformly placed centre P(overhang) ≈ 11.8 %; observed 1 in 49.
Six of 39 v2 windows sit within 10 km of a tile edge and one within 2 km (`ESP_054134_2265`, 915 m
west). And `hirise_coverage_fraction` is **not** a detector for a partial straddle — the v2 cohort
spans 0.4665–0.6313, so a ~50/50 straddle lands inside the normal band. `ESP_057469_2215` was only
visible at 0.001 because ~90 % of its footprint was outside the tile.

15 tests in `tests/test_ctx_window_geometry.py` (the file previously never called `extract_ctx_window`
at all), on a deliberately non-square synthetic tile so a `crop` argument swap cannot pass by symmetry.
Fast suite: **591 passed**, 21 deselected.

## 2026-08-06r — R68 CLOSED: a guard that could not fire, replaced by one that checks the real property

Audit step 5, Stage 4. Stage 4 carried a "runtime pixel-size guard" comparing the CTX window's pixel
size against its parent mosaic's, cited in `docs/review_2026-07-31/labeling.md` as *the precondition
for the integer-nesting claim*. It is a **tautology**: `ensure_tile_cached` writes
`inner_transform = list(src.transform)[:6]` of the `/vsizip/` tile handle into the tile sidecar, and
`extract_ctx_window` cuts the window from that **same handle** via `src.window_transform(...)`, which
preserves `a`/`e` bit-identically. Measured over all 49 cached windows the two agree to **0.0 exactly**,
so no pipeline-reachable input could make it raise.

Independent proof it had never once fired: its own error message was a **non-f-string** containing a
literal `{murray_tile}` placeholder, and hardcoded `cache/` (wrong for the v2 cohort).

**The property Stage 4 actually depends on** is the *origin phase*: the window's upper-left must sit at
an integer mosaic-pixel offset from the parent tile's origin, because the ×2 tile ladder is anchored on
**absolute mosaic-pixel indices**. It matters because Stage 4's two halves are anchored differently —
`_rasterize_boulders_subpixel` and the eligibility crop are **window**-anchored (`r0_win`/`c0_win`)
while `_count_centroids_per_finest_cell` and the emitted bbox are **mosaic**-anchored. A fractional
phase slides them apart: on a synthetic +0.5 px window a 2×2 m boulder reports `boulder_count = 1`
while `boulder_area` is 0.0 on every tile.

**Fix.** The check now lives inside `_compute_grid_alignment`, at the `int(round(...))` that silently
assumed it, and covers three things: pixel size, origin phase, and a negative origin. Two named
tolerances, and the comment says loudly that they are in **different units** — `GRID_PIXEL_SIZE_TOL_M`
in metres, `GRID_PHASE_TOL_PX` in mosaic pixels. The phase tolerance is 1e-6 px because the measured
worst residual over 49 cached windows is **1.38e-10 px** — *not* bit-zero, so an `== 0` test would break
on live data — while one sub-pixel of the 5× rasteriser is 0.2 px, five orders above.

The negative-origin clause is **R31 defence in depth**: an overhanging window was written with the
requested (un-cropped) transform, so it is misregistered by exactly the overhang; its phase is still
integer, so only a bounds check catches it. v1's `ESP_057469_2215` is −1,924 px (9.6 km) west today.
R31 now refuses to write one, but an already-cached window can still carry it.

**Register corrections.** The fix bullet at `geo-crs-deep.md` uses `assert` — wrong for this codebase
(`python -O` strips assertions, and every other Stage-4 invariant raises). It gives only the **column**
formula while the prose says "on both axes"; the row axis has the **opposite sign** (`e < 0`), and
copying the column form across negates the origin. Its offer to "delete it and remove the
verified-clean claim" as an equal branch is the wrong branch — the property is real, load-bearing and
was unasserted anywhere. And its `1e-6` silently means pixels where the existing guard's `1e-6` meant
metres. Also: `geo-crs-deep.md`'s "that phase relation is what R01 found broken" is **not the same
quantity** — R01 is the `47420 % 32 = 28` offset between *adjacent* Murray tiles' coarse lattices at
mosaic time; R68 is one window's origin phase on its own parent tile.

**Blast radius: none.** No number moves; `_compute_grid_alignment` returns exactly what it did on every
passing input. Re-derived on all 38 v2 images: the guard passes 38/38. Eight tests, and all five
mutants die — tolerance→0 (2 failed), tolerance→1.0 (5), row-sign flip (15), drop the negative-origin
clause (3), drop the pixel-size clause (1). The tolerance→0 mutant is the one that matters: a fixture
exercising only an exactly-zero residual would not have killed it. Fast suite: **603 passed**.

## 2026-08-06s — R80 partial: "the size filter is a no-op" is false

Recorded while diagnosing R80 (the size-floor filter's test coverage). The 2026-05-28 vClaire filter
decision says *"~0% below the `min_size_m=1.4105` floor, so that filter is a no-op"*. **It is not.**
The shipped Stage-4 sidecars record **19,757 polygons dropped** by `min_size_m` across **12 images**.
The pooled p5 ≈ 1.9 m that justified the claim is dominated by the 26 coarse (0.50 m/px) images, where
the filter genuinely is inert; on the 12 fine (0.25 m/px) images it drops **0.006–8.26 % each**.

That is R03's mechanism operating exactly where the record says nothing is happening, and it is why the
mixed size floor Brian chose to retain and document is a **live convention**, not an inert setting. The
2026-05-28 paragraph is annotated in place. R80's remaining work — the end-to-end test with projected
units and a diameter/radius-separating fixture, and a realised **size**-floor provenance analogue to
`realised_label_basis` — is still open; the diagnosis pass's proposed `detector_min_size_px` design was
refuted by measurement and must not ship as drafted.

## 2026-08-06t — R66 CLOSED: a truncated download can no longer reach a permanent cache

Audit step 5, Stage 2. `ensure_jp2_local` streamed a JP2 into a `.partial` sibling and then
`Path.replace`d it onto the cache path **unconditionally**. `HTTPResponse.read(amt)` returns `b""` on a
premature EOF rather than raising, so a dropped connection published a short file that was thereafter
trusted forever.

**The register understated the downstream, and mischaracterised the defect.** The atomicity was never
at fault — `ensure_jp2_local` already wrote to a same-directory `.partial` and renamed atomically. The
*unconditional* rename was the bug. And the consequence is not "best case GDAL raises": measured, it is
deterministically the worst case — **GDAL opens a truncated JPEG2000 happily, reports full dimensions,
and `read()` returns a full-shape array with the missing region silently zero-filled**, which Stage 2
converts straight into "no HiRISE coverage". Same class as R23, and quieter.

**Fix.** New `src/net.py` holding the one check that has to happen between the stream and the rename:
`verify_download` compares bytes received against `Content-Length`, applies any existing size floor,
runs a caller-supplied content validator, and **unlinks the staging file on failure** so a re-run starts
clean. Plus `hirise_imagery.inspect_jp2_integrity`, a pure-bytes JPEG2000 walker on the R23 template.

Two things that make the walker non-obvious, both measured:

- **A JP2 box may carry `Lbox == 0`, meaning "extends to EOF", and all 46 PDS JP2s use exactly that for
  their `jp2c` box.** So the box walk alone can *never* detect truncation. The real test is inside the
  codestream: walk the SOT tile-part chain via each `Psot` and require it to land on the `EOC` marker
  at exactly `size - 2`.
- **No GDAL.** Opening a JP2 through GDAL can write an `.aux.xml` PAM sidecar beside it, and these live
  in artifact roots — three such sidecars already exist in both `hirise_jp2/` dirs, which is also why
  the register counted "48 cached JP2s": 46 `.JP2` + 3 PAM sidecars = 49 entries.

**Strict at commit, lenient at reuse.** Commit-time rejects a truncated download outright. Reuse-time
rejects only a *positive* `"truncated"` verdict and lets `"not_jp2"` through — because
`tests/test_artifact_isolation.py` deliberately stages a GeoTIFF under a `{OBS}_RED.JP2` name, and more
importantly a structurally unusual but legitimate JP2 must not be rejected by a walker that merely
failed to parse it. A truncated *cached* file now raises with the remedy rather than being silently
preferred over `/vsicurl/`; it does not self-repair, because this repo has been bitten twice by writes
to live artifacts and an implicit 500 MB re-download is not something to trigger silently.

**A size floor is not a truncation detector at any value** — a 55 % truncation of the 1.31 GB
`ESP_068483_2280` is 719 MB and clears any floor below the 149 MB smallest genuine file. The floors are
kept because they are free, not because they help.

**Siblings, all verified rather than assumed.** `validation_retrieve._download_raster` had the identical
hole with **no floor at all** (measured pre-fix: 4,883,003 bytes committed against a declared
8,878,189) — fixed. `ctx_retrieve._download_to` read `Content-Length` and only ever forwarded it to the
progress callback — now compared, plus a central-directory check so a truncated zip is never committed
(previously it *was* committed above the 50 MB floor, and because the re-download is gated on
`not zip_path.exists()` while the structural check ran only when the sidecar was absent, a
committed-then-rejected zip **wedged forever**). The PDS `.LBL` path is **safe** and needs no change:
`pds_labels` uses `resp.read()` with no `amt`, which routes to `_safe_read` and raises `IncompleteRead`.
Still unguarded and noted for follow-up: `scripts/probes/_fetch_color.py`,
`scripts/run_stage7a_fetch.py`, `scripts/probes/_fetch_cumindex.py` — three verbatim copies of the same
pattern that write into live roots.

**Nothing on disk is corrupt.** All **46** cached JP2s in each of `cache/` and `cache_v2/` walk to
`complete` (identical sizes across both roots; min 149,019,114 B `ESP_045390_2215`, max 1,306,649,437 B
`ESP_068483_2280`). So no artifact changes, no number moves, and **no rebuild is forced** — this is a
pure guard. It is also not a rebuild blocker for the current cohort: all 39 manifest rows already have a
cached JP2 and both roots hold 24/24 tile zips, so a rebuild on this manifest downloads nothing.

**Deferred deliberately:** the derived-cache staleness key (tagging `hirise_decimated/*.tif` with the
source JP2's size). It addresses staleness, not truncation, and a spurious "stale" verdict would rewrite
files inside two live roots. If it is done later, key on **bytes only** — not mtime, which does not
survive an rsync, a restore or the drive migration that is currently pending.

21 tests in `tests/test_download_integrity.py`, hermetic (a localhost server, synthetic payloads,
`tmp_path`), including an end-to-end dropped-connection simulation. All five mutants die: drop the
length check (4 failures), drop the validator (1), reuse gate no-op (1), walker always "complete" (10),
and reuse gate over-strict on `not_jp2` (1 — the isolation-suite fixture). The live-cache regression
guard is deliberately **not** marked `slow`: it reads only marker headers, 3 ms for all 46 files.
Fast suite: **624 passed**, 21 deselected.

## 2026-08-06u — R80 CLOSED: the size floor is pinned, and the per-image physical floor is now recorded

Audit step 5, Stage 4 gate. Completes 2026-08-06s (which corrected the "no-op" claim).

**The tests (the gate item).** The old fixture used areas 1/100/1000 in EPSG:4326 against a 5.0 m
threshold. It could not tell diameter from radius, and it measured area in **degrees²** — which is
also why every suite run printed `UserWarning: Geometry is in a geographic CRS`. The replacement runs
in a projected metre frame and uses six shapes chosen so that at a 5.0 m floor **only the
equivalent-circle diameter yields {2,3,4}**; radius yields {3}, max-bbox-side {1,2,3,5}, perimeter/π
and area-as-size {1,2,3,4,5}, and equivalent-**square** side {2,3}. Plus inclusive-boundary tests for
both floors (derived with the production expression, so bit-exact rather than approximately at the
boundary), a lat_ts 60-vs-0 characterisation test, and the end-to-end Stage-4 test the gate names —
which did not exist, so deleting the filter call from `stage4_one_image` outright had been surviving
the entire suite. The disappearance of the geographic-CRS warning is the acceptance signal for the
units half, and it is gone (0 occurrences).

**What the skeptic caught that the first mutation pass did not.** Two mutants survived 630 tests:
substituting `geometry.envelope.area` (bounding box) or `geometry.convex_hull.area` for polygon area.
Every fixture shape was an axis-aligned `shapely.box`, for which all three areas are **identical** —
so the polygon-area interpretation was not pinned at all. It is not academic: on `ESP_046328_2180`
(138,373 polygons, none axis-aligned) the median bbox/polygon area ratio is **1.2673**, and measuring
bbox area would silently retain **3,466 of the 6,360** polygons the production floor drops. The
fixture now contains a concave L (a 5×5 square minus a 3×3 corner) whose three areas straddle the
floor — polygon 16 (d 4.51, dropped), hull 20.5 (d 5.11, kept), bbox 25 (d 5.64, kept). Both mutants
now die. A third survivor — computing the surviving-diameter statistic from the combined keep mask
instead of the size mask alone — is killed by making the confidence-dropped polygon the *smallest*
size-survivor.

**Provenance: `realised_size_basis`,** the size-floor analogue of `realised_label_basis`. The first
draft asserted rather than measured, and the skeptic was right to reject it:

- `realised_floor_is_looser_than_configured` was a hardcoded `True`. **It is false** for an image
  whose source frame already equals the CTX frame — reproduced on v1's `ESP_039820_1750` (source
  lat_ts 0, R 3396190, scale exactly 1.000000000000). It is now derived, and `None` when unknown.
- The per-image number the block exists to carry was **not emitted at all**; every mixture-bearing
  field was a hardcoded constant, reproducing the exact defect the block was written to fix. It now
  emits `source_to_target_diameter_scale` = `sqrt((R_t/R_s)² · cos(lat_ts_t)/cos(lat_ts_s))` and
  `realised_physical_min_size_m` = configured / scale, read from the Stage-1 sidecar's own
  `source_crs_wkt`. That single float is what makes a product-level mixture aggregation possible.
- `realised_diameter_floor_m` was the smallest survivor of **both** floors measured in the inflated
  frame, so on images where the size floor bound nothing it reported a "floor" of 2.55 m while the
  note in the same dict said the realised floor was *below* 1.4105 m. Renamed
  `min_surviving_diameter_ctx_frame_m`, computed from the size mask alone, with an explicit
  `size_floor_was_binding`.
- A seeded `area_total_m2: 0.0` was emitted next to `n_in: 3` when no filters were configured — a
  positive false claim, not a missing measurement. Only measured keys are written now.
- `measured_in_crs` recorded a 483-character WKT blob in production (rasterio's CRS has no `.name`,
  so the probe fell through to `to_string()`) while tests saw a 41-character name — the tests were
  exercising a branch production never took. Both now normalise through pyproj.
- CRS parameters are read via `coordinate_operation.params`, not `to_dict()`/`to_proj4()`, which emit
  their own `UserWarning` — a new warning inside a producer would be noise in exactly the suite whose
  acceptance signal is that a warning disappeared.

**Still deliberately not emitted:** a `detector_min_size_px = 5` / `binding_floor` triple. It was
drafted, then refuted by measurement — the detections do not obey a 5-pixel floor, so publishing one
as provenance would assert something false.

**Not changed, on purpose.** Moving the filter before the reprojection, or dividing by the scale
inside it, would make the realised floor match the documented one — and would delete a further
~0.4–3 % of every fine-cohort image's polygons, i.e. redefine the target. Under "retain and document"
the correct action is to record it. `config.yaml`/`config_v2.yaml` and the DATA_DICTIONARY are
corrected accordingly.

Behaviour is identical: `_apply_detection_filters` was verified kept-row-identical against the
pre-change implementation over 5 real cached GPKGs (9,628 to 727,160 polygons) × 8 filter
configurations, n=40, exact match in all 40. Fast suite: **636 passed**, 21 deselected.

## 2026-08-06v — R67 CLOSED: the nominal window's width was spent in the wrong metres, and the nominal itself is undersized

Audit step 5, Stage 2. `nominal_footprint_bounds` — the fallback for an image with zero detections —
spent `nominal_hirise_width_m` in **projected** metres of the equirectangular clon_0 target CRS.
Easting there is `R·lon`, so covering `W` metres of **ground** at latitude φ needs `W/cos(φ)`
projected metres; the old rectangle covered only `W·cos(φ)`, i.e. the window came out too **narrow**,
and by more the further from the equator.

**Severity corrected to LOW, and the register's own numbers were off.** The branch is provably
unreachable for the v2 rebuild: all 39 images take `polygon_bbox` (smallest post-R23 polygon count
9,628). The parent brief's "cohort spans ~11N to 46N, cos(lat) ~0.98 to ~0.69" is wrong — the v2
manifest spans **−63.70 to +52.33**, cos(lat) **0.443 to 0.929**.

**A second, independent defect the register did not name.** `nominal_hirise_length_m` (16,000 m) is
simply too short: **13 of 39** real PDS footprints exceed it, the largest at **43,088 m**. And
measured against the real footprints in projected metres, the 6 km nominal width is too narrow for
**39 of 39** by a median 1,847 m per side — *and still too narrow for 39 of 39* after the cos(lat)
correction, by a median 815 m. So the units fix alone is not sufficient; the nominal is undersized.

**Fix.** `nominal_footprint_bounds` stays a pure geometry function and gains an optional
`footprint` (the image's own PDS extents) plus `buffer_m`. When the footprint is supplied it is
projected corner-by-corner and used directly; otherwise the centre-on-manifest rectangle is built
with the cos(lat) correction. `stage2_one_image` resolves the footprint via
`pds_labels.image_footprint` and records `footprint_source` as `pds_label_footprint` or
`nominal_from_manifest_coslat`, **warning loudly** on the latter because it is known to clip.

All 47 `.LBL`s are cached and all 39 manifest rows have one — but only *incidentally*: `detections`
fetches a label only when the SP1 bug fires, so a new manifest row may not have one. Hence the
fallback survives rather than being replaced by an assertion.

Also guarded: an antimeridian footprint. Independently wrapping west/east longitudes would produce a
~21,000 km bbox in a clon_0 frame. Not reachable in either cohort (max observed span 0.276°), but
silent nonsense if it ever were.

**A test was certifying the bug.** `test_nominal_footprint_bounds_centered_on_manifest_point`
asserted `abs((xmax - xmin) - width_m) <= 2*PX`, i.e. it pinned the projected width to `width_m` and
would have failed on the correct behaviour. Rewritten to assert `width_m/cos(lat)`, plus four new
tests. All five mutants die: flat projected width, multiply-by-cos instead of divide, applying cos
to the north-south axis (northing is `R·lat`, so it must *not* scale), ignoring the footprint
argument, and dropping the antimeridian guard.

**Blast radius: none.** No v2 window takes this branch, so no shipped artifact changes and no rebuild
is forced. `DATA_DICTIONARY`'s `footprint_source` row is updated, including the retired
`nominal_from_manifest` literal that pre-2026-08-06 sidecars carry. Fast suite: **640 passed**.

**Deliberately not done:** a global "window must contain the PDS footprint" assertion. Scoped to the
nominal branch it would be right, but applied globally it would abort a v1 re-run — 3 of the 9 v1
`polygon_bbox` windows already violate it (`ESP_057469_2215` by 9,422 m).

## 2026-08-06w — R65 CLOSED: Stage 3's only quality number is a conditional median, now labelled and accompanied

Audit step 5, Stage 3. `peak_correlation` is the sole per-image quality figure Stage 3 emits, and it
carries two independent defects on the `block_median` path (38 of the 39 v2 images).

**(a) It is bounded below by the threshold it is screened against.**
`_robust_shift_from_field` reports `median(peaks[peaks >= block_peak_min])` — a **conditional**
median over blocks that already cleared the floor — so it lives in `[block_peak_min, 1]` by
construction. Measured over the 38 block-median sidecars: `block_peak_min` is 0.5 and a
`peak_correlation >= 0.5` cohort screen rejects **0 of 38**. It cannot reject anything.

Worth stating precisely, because the register's wording overreaches: min is **0.5779**, i.e. 0.0779
*above* a bound it cannot cross, with **0 of 38 at the floor** and only 3 within 0.10. So the defect
is the **vacuous screen**, not a pile-up at the bound. And the `>= 0.9 -> 0 images cleared` screen at
`DECISIONS.md:2389` is *not* the mirror image of it: 0.9 is above the observed max (0.8751) but is an
**empirical ceiling**, not a structural bound. Only the low end is vacuous by construction.

**(b) It does not score the model that was applied.** On the block path it is a summary of per-block
peaks, not the post-shift correlation of the median shift actually used. On the `single_window`
fallback it *is* the applied shift's own post-shift Pearson. The two are not comparable, and nothing
said so — `peak_correlation_kind` now does.

**Fix — deliberately not a new headline statistic.** The drafted replacement (an unconditional median
over all blocks) was refuted before implementation: it reads ~0.75 on a *perfectly registered* image
that merely has 25 % uncorrelatable blocks, i.e. it conflates registration quality with scene
texture. Instead Stage 3 now emits the **components** and leaves the gate explicit:

- `all_block_peak` {min, p25, median, p75, max} over **every** block — unconditional, so unlike the
  existing figure it *can* fail a screen;
- `confident_fraction` = `n_confident_blocks / n_blocks`;
- `median_block_peak_is_conditional` and `block_mad_px_is_conditional` — the MAD is computed over the
  same confident subset and inherits the identical self-fulfilling shape, which was unlabelled;
- `quality_version` (2), so pre- and post-R65 sidecars are distinguishable.

Together these separate the two cases the old field could not: measured on the live cohort,
`confident_fraction` runs 0.462–1.000 with median **0.962**, so only 1 of 38 images had under half
its blocks correlate. That is also the evidence for the register's severity cap — **no shipped number
is wrong today**; the defect is that the statistic could not have told us if one were.

Corroborating the record: the one image ever excluded on co-registration grounds,
`ESP_046803_2325`, took the single-window fallback with 3 of 44 confident blocks and
`peak_correlation` 0.3229 — it was excluded on the *untruncated* companion evidence, not on
`peak_correlation`. The project had already, informally, stopped trusting the truncated figure.

**Blast radius: provenance only.** No shift, label or downstream number changes. **0 of 39** sidecars
carry the new fields, so Stage 3 must re-run to emit them — fold it into the batched rebuild, where
Stage 3 re-runs anyway; do not run it standalone against `cache_v2`. Four tests, including two images
with an *identical* conditional median that the unconditional distribution tells apart.
Fast suite: **644 passed**.

## 2026-08-06x — R01 part 1: the globally anchored coarse grid exists, and the merge is guarded

Audit step 5, mapping. **This is the primitives + guard half; the driver wiring is part 2.**

**The defect, measured.** A Murray tile is 47,420 native px wide and adjacent tile origins are exactly
47,420 px apart — but `gcd(47420, 32) = 4`, so `47420 % 32 = 28 ≡ −4 (mod 32)` and each tile's coarse
32-px lattice starts at its own sub-cell phase, walking 4 native px (20 m) per 4° step. Over the
26-tile footprint: 8 distinct x-phases, 4 y-phases, every adjacent pair offset by exactly 20 m.

`rasterio.merge` then **floors** each fractional destination offset, converting that sub-cell phase
into a whole-cell placement error. On the shipped mosaic: **25 of 26 tiles displaced**, median
**140 m**, max **198 m**, 21 of 26 beyond half a cell. Downstream and measured rather than asserted —
correcting only the *integer* part lifts the THEMIS validation correlation from |ρ| **0.0741** to
**0.0821** (n=26 tiles, 150k cells sampled each), and that is a lower bound because the ≤0.5-cell
residual remains.

**Two register corrections.** (i) It is **26/26**, not 25/26, that are off the *global* lattice —
25/26 is displacement relative to the mosaic's own arbitrary anchor, and every latitude in the
footprint has a non-zero row phase. Independently reproduced here: the new guard fires 26/26 with 26
distinct sub-cell phases. (ii) The nodata seam is 2 **or** 3 cells depending on phase, not a uniform 2.

**What landed.** `MURRAY_RADIUS_M` / `MURRAY_PPD` / `MURRAY_NATIVE_M` / `COARSE_GRID_ID`, plus
`global_native_origin`, `tile_grid_phase`, `global_cell_transform`, `assert_shared_lattice` and
`assert_murray_sphere`. `predict_window` gains an optional `global_grid`; `mosaic_geotiffs` refuses to
merge off-lattice rasters, and `striping.mosaic_tiles` warns.

Four things worth not re-deriving:

- **A canonical constant, not each tile's `a`.** The cached sidecars carry **four** distinct pixel
  sizes (`…306304` ×14, `…3063035` ×8, `…306295` ×1, `…306302` ×1) and **none equals** the exact
  `π·R/180/11855`. Per-tile `a` would re-import that ULP spread and make the merge offsets
  non-integral; the canonical constant makes `a`, `c`, `f` bit-identical across tiles.
- **The phase convention is pinned deliberately.** `tile_grid_phase` returns `(−origin) % 32` =
  `{16, 20, 24, 28}` for `{N44, N40, N36, N32}`. The complement `origin % 32` gives `{16, 12, 8, 4}`.
  Cross-wiring them is a real mutant, and it is **invisible on an N44 tile** where both are 16 — so
  the test spans four latitudes on purpose.
- **The local→global conversion happens AFTER the window-indexed work.** Everything above it indexes
  the window as `ti*tile_px − row0` with a *local* `row0`; promoting `ti` to ~−16,300 first drives the
  slice origin to ~−521,600, `valid` goes all-False, every prob is NaN and assembly dies on `ti.min()`
  of an empty array. A test covers the ordering.
- **`global_grid` is one tuple, not two arguments.** Making `(ti, tj)` global while still deriving the
  transform from the parent-tile origin lands the raster **~2,600 km** away. The halves are
  inseparable, so the coupling is structural rather than a sentinel on a data argument.

`assert_murray_sphere` parses the radius out of the tile CRS and checks it, because `COARSE_GRID_ID`
asserts `R3396190` and nothing otherwise measured it — the assert-rather-than-measure gap caught twice
already this week. Verified: it reads 3396190.0 from the real shipped CRS.

**The guard fails loudly on the currently shipped products by design** (`require_shared_lattice=False`
reproduces a pre-R01 merge knowingly). `striping.mosaic_tiles` only warns: that is the notebook-24/25
analysis path over already-shipped tiles, whose subject *is* the artifact as it exists.

**Blast radius: none yet.** No driver passes `global_grid`, so every output is byte-identical and the
legacy path is covered by a test. 14 new tests; fast suite **658 passed**.

**Part 2, still to do:** thread `global_grid` through `scripts/map_region.py` and
`scripts/striping_a1_map.py` **in one commit** (or A1 lands on a different lattice than the baseline,
which the 2026-08-06 product decision forbids); fix `window_offsets`, which silently drops 11 cells
per axis once the grid has a phase (`last = extent - win` and `overlap = 3*tile_px`; free — 144
windows either way, and it has never bitten because phase 0 loses 0); correct its docstring contract
"offsets are multiples of tile_px", which the design deliberately abandons and which
`scripts/f_region_stageb.py` also relies on; and record `grid_id` in partials, sidecar and manifest,
treating a **missing** key as a mismatch rather than a KeyError. Part 2 forces the full re-render.

## 2026-08-06y — R01 part 2: both map drivers moved onto the global grid, in one commit

Audit step 5, mapping. Part 1 built the vocabulary; **nothing used it**. This wires it through
`scripts/map_region.py` **and** `scripts/striping_a1_map.py` together, because wiring them separately
would put A1 on a different lattice than the baseline it is compared against — the failure the
2026-08-06 product decision forbids. **R01 is now closed.**

**The sweep loses cells, and each half of the fix alone still loses some.** Re-measured this session
over the real tile size (`extent=47420, win=4096, tile_px=32`), counting cells whose 3×3 context box
fits in the tile but in no single window:

| final offset | overlap | lost cells / axis (phases 4–28) |
|---|---|---|
| tile-aligned | `2*tile_px` | **11** ← shipped |
| tile-aligned | `3*tile_px` | 1 |
| `extent - win` | `2*tile_px` | 10 |
| `extent - win` | `3*tile_px` | **0** |

Phase 0 loses nothing in all four, which is exactly why this never bit. All four use **12 offsets per
axis** → 144 windows, 2,415,919,104 px: the fix is free. The register's plan had the two edits but not
the fact that either alone is insufficient — the `3*tile_px`-only row is new here.

`window_offsets` moved into `src/mapping.py` (logic belongs in importable `src/`; `scripts/map_region`
re-exports it so `striping_a1_map` and `f_region_stageb` import unchanged) and gained
`tile_aligned=True` **as the default**. `f_region_stageb` sweeps ISIS frame cubes on their own phase-0
grid, so leaving the default alone keeps its window set and its per-window partial filenames
byte-identical — the alternative would have silently invalidated `$SCRATCH` partials of an aborted
build for no gain. The docstring now states both contracts instead of the abandoned one.

**The coverage contract is executable, not arithmetic-by-inspection.** New `uncovered_cells()` returns
the cells no window can compute; both drivers call it after building their sweep and refuse to run if
it is non-empty. Choosing a bad `overlap` now fails in microseconds instead of punching a one-cell
stripe of nodata through a product that took 16 GPU-h.

**Provenance you cannot fake.** `TileGlobalGrid` is only obtainable from `tile_global_grid()`, which
parses the sphere radius out of *that tile's* CRS and checks the origin against the native lattice
before returning. `provenance()` is therefore unreachable without those checks having passed — the
structural answer to "provenance that ASSERTS instead of MEASURES", now caught five times. Sidecars
and manifests carry `grid_id`, `grid_cell_m`, `cell_row0/col0`, `grid_phase_px` and **global**
`ti_min/tj_min`, so "are these two products co-registered?" is answerable from two JSONs.

**Resume safety.** Per-window partials carry `grid_id`; a missing key counts as a mismatch (every
pre-R01 partial lacks one). The check runs at **scan time, before any GPU work**, and aborts with the
remedy rather than at assembly after a wasted tile. `--force` discards and recomputes.

**A1's ordering constraint is now a gate.** `frame_stats_160` reads the per-frame normalisation off
`reports/map_region/{tile}_abundance.tif`, so the corrected baseline must be rendered **first**; the
reference raster is checked against the lattice and the run aborts otherwise.

**MEASURED — how far re-anchoring moves the A1 per-frame statistics.** This was the one claim the
part-1 review flagged as reasoned rather than measured, and it gates 5–7 GPU-h. Recomputed on both
lattices for E4_N40 and E8_N44, 74 frames, read-only.

**My first measurement of this was wrong, and the skeptic pass caught it.** I evaluated the induced
shift only *at the frame median*, where the gain term `IQR0·(s_old/s_new − 1)` vanishes identically —
so I reported "≤0.89 DN, 0 of 74 frames move by more than 1 DN" while silently dropping half of the
expression I had quoted in the same sentence. The full difference is
`IQR0·((x−m_new)/s_new − (x−m_old)/s_old)`, and re-measured over each frame's real pixel distribution:

| quantity | median | p95 | max |
|---|---|---|---|
| offset term, at the frame median (the old, partial metric) | 0.045 | 0.479 | 0.891 DN |
| gain slope, per 1 IQR from the median | 0.083 | 0.839 | 2.848 DN |
| **full \|diff\| over the frame's own pixels** | **0.411** | **3.287** | **8.993 DN** |

**11 of 74 frames exceed 1 DN, 6 exceed 2 DN, 2 exceed 5 DN.** The tail is concentrated in *small*
frames — the four worst have 108–2,068 valid 160 m cells — where the robust IQR is poorly determined.

The conclusion survives in weakened form: the **definition** of the statistic is unchanged (robust
median/IQR at 160 m, SeamMap-keyed, off the baseline grid), which is what `models/deployable_a1` was
trained against, so this is not the invalidation `striping_a1_map` warns about (deriving it from
native 5 m DN). But the A1 product is **not** bit-reproducible across the re-anchoring, and the
small-frame tail is the same population **R08** is open on. R08 is a precondition for shipping A1,
not an unrelated finding — that link is new and came out of this measurement.

> ### ⚠ CORRECTED 2026-08-09 by R07 — the premise of the paragraph above is false
> "the 160 m SeamMap-keyed statistic … is what `models/deployable_a1` was trained against" is
> **wrong, and it is the exact inverse of the truth.** Training
> (`scripts/probes/_w2_fang_embed.py:209`) used `a1_stats(arr)` on the **native 5 m** Stage-2
> window — one statistic per *window*, not per frame. I inherited that claim from the
> `striping_a1_map` docstrings and repeated it without checking; the docstrings cite
> `striping_a1_infer_crop.py`, which is another *inference* script, so the whole justification was
> inference matched to inference and called train parity. R07 (**2026-08-09a**) measures it and
> fixes both sides. What still stands from the paragraph above: the lattice-sensitivity numbers
> themselves, and the R08 link.

**The third merge path is closed.** `scripts/striping_frame_blocks.py:85` called `merge()` directly
over all 26 abundance tifs with no lattice check at all — so the one figure whose subject is where
features sit relative to frame boundaries was built with every tile's phase floored into a whole-cell
displacement. It now routes through `striping.mosaic_tiles`, which warns.

**Mutation pass: 17 of 18 mutants killed.** The survivor is stated rather than papered over: the
`(gr + phase_r) % tile_px` check inside `tile_global_grid` is a tautology given the line above it, so
no test can kill it independently. Kept as defence-in-depth against a future change to the phase
convention. Three mutants survived the *first* pass and are worth knowing — a driver test on a
round-numbered synthetic extent cannot see the final-offset bug (`1024 - 256` is tile-aligned; the
fixture is now `1052 ≡ 28 (mod 32)` like the real 47,420), a coverage guard that never fires under
correct constants needs a deliberately holey sweep to test, and asserting `grid_radius_m ==
3396190.0` passes whether the value was parsed or assumed (the fixture now uses `3396190.5`).

**Blast radius: every regional output.** Per-tile shape is unchanged at 1479×1479 at every phase, but
the mosaic origin moves +100.0 m E / −80.0 m S (shape stays 5925×11852) and no resample recovers the
sub-cell part. `docs/PENDING_REBUILD.md` carries the cost. Stage-4 **label** cells stay tile-anchored
deliberately — re-anchoring them would force a relabel + retrain of the frozen recipe for zero
modelling gain — so map cells no longer coincide with label cells and any future map↔label comparison
must **resample rather than index-match**. `reports/map_fbuild/` (aborted F build) and
`reports/map_pilot/` stay on the old lattice; neither is a live comparison row.

**The skeptic pass, and the five things it changed.** Run adversarially over the finished diff
(five lenses, each finding then attacked by three refuters). It was interrupted by a session limit
with 22 of 32 agents dead, so **the geometry and test-quality lenses never reported** — that half of
the diff has only my own verification behind it (the end-to-end geometry reproduction below, and an
18-mutant pass). Of what did return, one finding survived refutation and four more had mechanisms the
refuters reproduced but downgraded. All five are now fixed:

1. **The tile-level resume skip was lattice-blind** — the most dangerous one. `map_one_tile` skipped
   on `{tile}_prob.tif` existing, *before* any grid check. Every pre-R01 tile is on disk, so
   `--all` would have skipped all 26, written a manifest stamped with the **new** `grid_id`, and
   printed "26/26 tiles complete": a rebuild that rendered nothing and then certified itself. Now
   `existing_product_off_lattice` refuses, naming the remedy. A raster with no sidecar is refused too
   — absence must not read as clean.
2. **The acceptance gate never inspected the A1 row** although its docstring claimed to. It globbed
   `*_abundance.tif`, but A1's shipped configuration is uncalibrated (`--calibration` defaults to
   `None`, the row is scored on raw `P(rich)`), so it writes `_prob.tif` + `_prob_raw.tif` and no
   abundance raster. Now driven off the sidecars that claim the grid, checking every raster present.
3. **A corrupt partial could escape the gate.** `np.savez_compressed` writes the zip in place with no
   tmp+rename, so a killed job leaves a truncated `.npz`; `partial_grid_id` let `BadZipFile` escape,
   and it was evaluated *before* the `--force` branch — so `--force` could not clear it either,
   strictly worse than pre-R01. Unreadable now counts as foreign.
4. **THEMIS leg 1 would have misregistered silently.** Notebook 24 correlates the mosaic and the
   THEMIS crop **by array index**; after the rebuild they are still both (5925, 11852) but 0.625 of a
   cell apart. Equal shapes were the trap, not the reassurance. New `assert_coregistered` in
   `src/mapping.py`, wired into `notebooks/_build_24.py` (the notebook regenerates in the rebuild).
5. **The A1 sensitivity claim was understated** — see the corrected table above.

Two documentation claims were also wrong and are corrected: `reject_foreign_partials` said partial
filenames are "unchanged by R01" (the sweep's `step` went 4032 → 4000, so only 1 of 144 collides),
and the `int32` comment said global row indices reach ±2.1e6 (at S=32 the planet spans ±33,342 rows
and ±66,684 columns).

Fast suite **692 passed**, 1 skipped (the acceptance gate skips until a product claims the grid).

## 2026-08-06z — R23 recovery is CLOSED: no complete copy exists, and v3 re-detection is the fix

The one remaining open question on R23 was whether a byte-complete copy of the four truncated
BoulderNet exports still existed somewhere — worth asking because it was the input that got harder to
recover with time, and because finding one would have dissolved R23 rather than leaving it managed.

**Measured 2026-08-09, read-only.** `inspect_shapefile_integrity` over all 40 vClaire exports names
the four and prices the shortfall exactly:

| ObsId | on disk | declared | missing |
|---|---|---|---|
| `ESP_017355_2260` | 214.9 MB | 569.3 MB | 354.4 MB |
| `ESP_028537_2270` | 58.5 MB | 571.9 MB | 513.4 MB |
| `ESP_046803_2325` | 192.1 MB | 324.0 MB | 131.9 MB |
| `ESP_068483_2280` | 443.0 MB | 616.0 MB | 173.0 MB |

Total **1.17 GB** (the three retained images account for 659.3 MB; `ESP_028537_2270` was already
excluded). A filesystem sweep of `C:\Users\brian` plus D:/E:/F: found **6 copies, 0 complete**.

**The decisive detail:** `ESP_017355_2260` exists in three places — the hirise2ctx cohort, the
BoulderNet working tree, and the original `Downloads\Predictions\...` folder — and all three are
**bit-identical** (sha256 `ba030bde8e936e6647d3c438…`, 214,884,317 bytes, same mtime), with intact and
equally identical `.shx` (8,843,676 B) and `.dbf` (175,766,299 B). So the truncation happened
**upstream of every local copy**, at or before the original download. No local re-copy can recover it,
and the missing bytes were never on this machine.

**Brian's ruling (2026-08-09): do not pursue recovery. The fix is a v3 re-detection dataset he will
supply; v2 proceeds as-is in the meantime, and other findings keep being fixed against it.** R23's
retain-and-document remedy (DECISIONS 2026-08-06o) is therefore not a temporary holding position
pending recovery — it is the final disposition for v2. Stop re-opening the hunt.

## 2026-08-10a — R14 closed: a killed write can no longer look finished

R14 must land **before** the R01 re-render, not after: the re-render is 26 tiles × ~1 GPU-h under
a Sherlock wall clock and `map_one_tile` writes its artifacts at the very **end** of each tile —
precisely the window a wall-clock limit hits.

**What was wrong.** `write_geotiff` opened the *destination* in `"w"` mode (measured: the file
existed at 7,799,350 of 7,854,955 bytes before `close()`), and resume was `path.exists()` — no
size, no read, no provenance — keyed on `{tile}_prob.tif`, the **first** of four artifacts. So a
kill between artifacts 1 and 4 left a tile permanently "done" with **no abundance raster**, and
abundance is the deliverable.

**The three kill signatures, and why the obvious check is not enough.** Measured on real rasters:

| signature | `rasterio.open` | full decode | last block | finite count |
|---|---|---|---|---|
| truncated (10–99.99 %) | ✅ passes | ❌ raises | ❌ raises | — |
| valid, 100 % NaN | ✅ | ✅ | ✅ | ❌ catches |
| half the blocks written | ✅ | ✅ | ✅ | ❌ catches |

`rasterio.open` succeeds at every truncation fraction because the first IFD sits at byte 8. The
register's proposed check ("open, check height/width, and that the last block reads") **does**
catch truncation — the earlier note that it "cannot fire" is wrong — but it is blind to the two
nodata-shaped signatures, which are exactly what a wall-clock kill produces. `expect_finite` is
the only test that sees them, so it is load-bearing, not decoration.

**What landed.** `write_geotiff` stages a `.tmp` **sibling** (same volume — `Path.replace` is only
atomic within one), verifies it, then renames; a failure leaves the destination byte-identical,
which matters because `"w"` mode truncates an existing file immediately, so a re-run that died
mid-write used to destroy the good tile it was replacing. `verify_geotiff` decodes **blockwise**
(the mosaic already holds 281 MB in memory) and `expect_finite` is computed on the **float32
cast**, since the cast can turn a finite float64 into `inf` and make the check reject its own
correct output. Tiles commit as a **set** with the sidecar last and a per-raster
`{bytes, sha256, shape, n_finite}` record; resume checks content *and* provenance and prints the
first failing reason; the manifest **merges** instead of clobbering (the shipped one lists 4 tiles
while 26 are on disk, so 22 have no record and `win_px` is unknown for them).

**Corrections to the evidence, carried through.** The stale-mixing figures "1225 partials / 61.5 %
/ 0.4366" came from a state no sequence of rectangular sweeps can produce; the reachable numbers
are **719 / 63.1 % / 0.4933**, and mixing needs a stated precondition (`--force`, or a sweep
interrupted before assembly). The register's `n_predicted_tiles == n_windows·(win/S − 2)²`
invariant is **invalidated by R01**: with a phase the per-window yield is `(win/S − 3)²` for
windows whose shifted origin is not a multiple of `S`, so that assertion now fails on correct
output. Replaced with a derived per-axis cell count.

**Two things found only by building it.**
- **The sweep is a partition, not an overlap.** With `overlap = 3·tile_px` the windows overlap in
  *pixels* (so every cell has full context) but each cell is computed by exactly **one** window —
  measured, 900 cells over 36 windows, 0 duplicated. So the new overlap-disagreement check cannot
  false-positive within a run, and it is precisely a **cross-run** detector: two runs compute the
  same cell set on the same lattice, so a surviving stale partial collides cell-for-cell. That is
  the only check that sees the mixed-run failure, where every file is structurally perfect and the
  raster comes out the right shape.
- **Damage and wrong-lattice needed different answers.** Folding a corrupt `.npz` into R01's
  foreign-partial gate made one truncated file demand `--force`, which also discards every good
  partial beside it — so a wall-clock kill cost the whole tile instead of one window. `partial_status`
  now separates `damaged` (delete and recompute, silently correct) from `foreign` (refuse).

**A bug in my own R01 part 2, found here.** The coverage guard's axes were transposed —
`miss_r` was checked against `phase_c` and vice versa. Inert in the shipped configuration (the
sweep loses no cells at *any* phase, so both orderings return empty), which is exactly why no test
caught it; it would have checked the wrong axis the moment `win_px` or the overlap moved. Fixed,
with a spy test on the pairing. This is the class of thing the skeptic run's geometry lens would
have caught, and that lens is the one that died on the session limit.

**Mutation pass: 13/13 killed.** Seven of them survived the first pass — every driver-level guard,
because `tests/` had zero references to `scripts/map_region`. Fast suite 710 → **731 passed**.

**Rebuild impact: none on its own.** All 188 shipped rasters were swept (opened *and* fully block
decoded): 0 failures, so nothing needs re-rendering to repair damage. But the 26 shipped tiles
carry pre-R14 sidecars with no `rasters` block, so under the new rule they are "unverifiable, not
reusable" — the correct answer, and free, because R01 forces a full re-render anyway.
`--trust-existing` accepts them knowingly.

## 2026-08-09a — R07: A1's train/deploy statistic, measured, unified, and versioned

R07 had never been diagnosed (four sessions hit the limit on it). Diagnosed now, and the register
**understated** it: there are **three** defects, not one, and the register's fix would have closed
only the first.

### 1. Resolution — confirmed, and quantified over all 39 Stage-2 windows

Training (`scripts/probes/_w2_fang_embed.py:209`) took `a1_stats(arr)` on the **native 5 m** CTX
window. Both deploy paths took it from CTX **area-averaged to 160 m** and then applied that gain to
native DN. Measured on the exact arrays training used:

| | median | p95 | max |
|---|---|---|---|
| gain error `IQR_native / IQR_160m` | **1.35×** | 1.83× | 2.15× |
| realised input IQR at deploy | **37.3** | 50.7 | 59.6 |
| % pixels clipped at 0 or 255, deploy | 0.023 % | 2.2 % | 4.35 % |

Training pins the input IQR to **exactly 27.7** on all 39 windows by construction; deploy handed the
frozen ViT 35–115 % wider contrast than it ever saw, and clipped ~10× more pixels. That corroborates
**R38**'s aggravation claim; its "1.50× narrower" sits inside the range but the median is 1.35×.

### 2. Statistical unit — NEW, and the training code's own comment is false

`_w2_fang_embed.py:204` asserted "each training window is ~one CTX source frame". Measured against
the cached dissolved SeamMaps for 38 of 39 windows: only **10 of 38** lie in a single frame; 22 span
two, 3 span three, max four. Dominant-frame share median **80.9 %**, min 48.1 %; only 15/38 reach
≥90 %. So for 28 of 38 windows the training statistic was pooled across 2–4 source frames — training
removed between-**window** level/scale while deployment removed between-**frame** level/scale. These
are different normalizations, so the register's fix ("make both paths native-resolution") would have
left the arms still mismatched.

### 3. The arm was unversioned — NEW

**Eleven** heads — `deployable`, `deployable_a1` and nine F variants — share
`recipe_hash = 86c51a5dca220f63`, and no `recipe.json` mentions the preprocessing arm at all. The
only thing distinguishing them was the parent directory name. Feeding the A1 head raw DN, or the
baseline head A1-normalised DN, yields a plausible raster and no error.

### The docstrings asserted the exact inverse, and I had repeated it

`striping_a1_map` said "the head was trained against the 160 m statistics" and that using native
5 m "invalidates models/deployable_a1" — backwards on both counts. Its evidence was a citation to
`striping_a1_infer_crop.py`, i.e. another *inference* script: inference matched to inference and
called train parity. I inherited that claim on 2026-08-08 and built the 2026-08-06y A1 paragraph on
it; that paragraph now carries a correction notice.

### Fix (Brian, 2026-08-09): full parity, folded into the rebuild — plus arm versioning

One definition, `src.striping.A1_ARM = "a1_native_perframe_tilesupport_v2"`, called by both sides:
unit = one dissolved SeamMap source frame, resolution = native 5 m, support = the frame's extent in
the parent Murray tile, and **no pixel is left at raw DN** — anything in no qualifying frame takes
the tile-wide native statistic (`a1_normalize_native` raises rather than falling back to raw, which
is the R08 defect). Exactness comes free: uint8 percentiles are read from a 256-bin histogram, so
the streamed statistic is the true median/IQR, not a binned estimate.

`norm_arm` is now stamped into `recipe.json` at train time (inferred from `--store-name`, which is
where the arm always lived) and folded into `recipe_hash` **only when declared**, so pre-R07 hashes
and directories are untouched. `require_norm_arm` is deliberately asymmetric: the A1 path **refuses**
an unknown arm (that is the dangerous direction, and the A1 head must be retrained for R07 anyway),
while the baseline path only warns (unversioned + raw DN is the pre-R07 status quo, and blocking the
baseline re-render on a provenance field buys no safety).

### Three things found while implementing, each of which would have blocked or bitten

- **`load_frames` required a rendered abundance raster** — it opened
  `reports/map_region/{tile}_abundance.tif` merely to read a CRS. The 39 training windows span
  **20** Murray tiles while only the 26 map-footprint tiles have that product, so the R07 training
  fix could not have run at all. Frames are a property of the CTX tile; `_tile_crs` now reads the
  tile, falling back to the product.
- **Cost**: the naive implementation rasterized all ~81 frames onto every block and took **~45 min
  per tile**, which would have made per-frame native statistics impractical. A bounding-box
  pre-filter plus `bincount` instead of `np.add.at` brings it to **3.4 min/tile** (~13×): ~69 min for
  all 20 training tiles, ~31 min for the 9 A1 tiles, against ~5–7 GPU-h of A1 inference. All 39
  windows' parent tiles are already cached, so **no downloads are required**.
- **R01's A1 ordering constraint is GONE.** It existed only because the A1 statistic was read off the
  baseline product's grid. It no longer is, so the two rows can be built in either order and A1 is no
  longer sensitive to the re-anchoring. The gate added on 2026-08-08 is removed and replaced by a
  test that pins the independence.

### Status

Code fixed; **artifacts not regenerated**. The A1 embeddings and the A1 head must be re-made under
the new arm, which the batched rebuild already schedules — marginal GPU cost ≈ 0. Until then the
banked A1 numbers stand as measured under the old, mismatched definition: the η² payoff
(0.196 → 0.141) and the −0.024 AUC skill cost came from **different** A1 definitions and are still
not comparable with each other. `docs/PENDING_REBUILD.md` carries the row. R08 remains open and is
narrowed by this work: the fallback population is measurably tiny (0.0081 % of valid pixels on
E-12_N36) but its *contract* is now explicit rather than accidental.

## 2026-08-10b — R13: the context-window nodata gate, and the half of it that was "Record"

**The defect, restated as geometry.** `src/fm_embeddings.py` pins `context_px = 3 * tile_px`, so at
the frozen S=32 the embedder consumes a 96² box while `own_tile_zero_fraction` tested only its
central 32² — **1,024 of 9,216 pixels, so 88.9 % of what the ViT sees was never checked**. After the
96→224 bicubic resize the own tile spans a 74.67-px centre square, so of the 196 patch tokens GeM
pools, only **16 (8.2 %) are purely own-tile** and at most 36 (18.4 %) touch it. A tile whose own 32²
is spotless could therefore be embedded almost entirely black and predicted anyway. Reproduced
through the real `predict_window`: own zero-fraction **0.00**, context zero-fraction **0.8889**, a
**finite** probability 0.0845 emitted, and `n_masked_nodata` reporting nothing wrong.

**Rarity is not the defence; per-cell magnitude is the finding.** Real frozen ViT + the real shipped
head, 384 clean 96-px boxes, against E4_N44's shipped P(rich) IQR of 0.1524: one blackened 32-block
in the ring gives p90 |ΔP| **0.453 ≈ 3.0× IQR**; 92 *scattered* black pixels (ctx 0.00998) give
**0.704 ≈ 4.6× IQR**. Shape matters ~6× at equal pixel count. An island (own clean, ctx 0.889) moves
median P 0.211 → 0.069.

**Threshold = 0.0, on two legs and not three (Brian's call, 2026-08-10).** Compared with `<=`, so it
means "not one nodata pixel in the context box". (i) The frozen head's training set contains **0
nodata pixels in 161,005 context boxes** — 0.0 is the only value reproducing the distribution it was
fitted on, and Stage 4 never tested CTX nodata at all (`coverage_equals_one` is a *HiRISE* coverage
rule), so deploy-time context nodata is strictly OOD with no analogue on either side. (ii) It costs
**290 of 19,685,689** measured cells (1.5e-05; hard ceiling 1,167 map-wide over 56,870,060 cells).
The fix plan's third leg — "scattered zeros are the binding regime, so no fraction threshold is
robust" — is **withdrawn**: the verifier established with the real ViT that DN 0 and the perfectly
legal DN 1 (the Murray bottom-clip floor; min valid DN is 1 in all nine cached tiles) move the
prediction **identically to three decimals** in every regime tested. The damage is caused by
*blackness*, not by the sentinel, so a nodata gate cannot see A1's clip speckle at all and must not
be credited for it. **Honest caveat retained:** at exactly one nodata pixel there is no measurable
sentinel-specific signal, so 0.0 is conservative rather than forced by the data.

**What landed.** `src.mapping.context_zero_fraction` — the same box `slice_context_boxes` slices,
validity rule bit-identical, verified against it exactly on non-tile-aligned origins. It uses a
**lattice-block** form (crop to the cell lattice, reshape-sum to a per-cell count grid, integral
image over *that*), because the obvious full-resolution int64 integral image was benchmarked at
**0.44 s / +419 MB** per production window against **0.016 s / +18 MB** — the fix plan's claim that
folding it in would be *cheaper* than the own-tile loop it subsumes was backwards, so that loop is
left alone. `predict_window` gained `max_context_zero_fraction` (default **0.0**) and **two**
counters: conflating them into `(valid & ~usable)` was the trap that would have left the sidecar
under-reporting exactly as before while looking fixed.

**`max_zero_fraction`'s signature default was 0.5 while every production driver passed 0.3.** It is
now 0.3. The one caller taking the signature default was `scripts/parity_check.py` — the
cross-machine gate, reproducing a configuration nothing ever shipped. That script also **could not
run at all**: `run_window` declared seven parameters and both call sites passed eight positional
arguments, so it raised `TypeError` on every invocation, and `--ctx-tiles` was silently ignored
besides. Fixed, with both thresholds explicit and recorded in the reference npz. Its E4_N44 window
has **0 nodata over the entire 47,420² tile**, so it exercises neither gate; that is now stated in
the docstring together with the command to emit a *second*, gap-bearing reference. Deliberately
**added, not moved**: putting the gap into the only reference makes every future threshold change
break parity ambiguously (numerics drift vs gate change).

**Recording — the half the register asked for and nothing met.** The shipped
`reports/map_region/E4_N44.json` carries no threshold and no mask counts, so a raster rendered with
the gate off is indistinguishable from one rendered with it on. Tile sidecars now carry a
`nodata_gate` block: both thresholds, both counts, and a context-zero-fraction histogram at
`CONTEXT_ZERO_HIST_EDGES` so the threshold can be re-tuned from committed sidecars without a GPU
pass. The two counts are **de-duplicated cell sets**, not summed counters: masked cells never enter a
partial, and at grid phase 0 consecutive read windows share one cell per axis seam, so a sum would
double-count and the re-validation criterion ("the gate drops exactly N cells") would not be
checkable from the product at all. The histogram *is* a per-window sum, and says so in the key
`context_zero_hist_is_window_sum: true` rather than leaving it to be discovered.

**R13 × R14, and it is load-bearing.** `max_zero_fraction` was already a resume-match field. If the
context threshold had not joined it in the sweep manifest, a post-R13 resume would have silently
reused pre-R13 partials computed under no context gate and assembled the mixture without a word.
Partials predating the gate record now yield `null` counts plus a note, never a `0` that would read
as "nothing was masked".

**R13 × R38 — ordering, and the remedy is constrained.** `scripts/striping_a1_map.py` gets the flag
but defaults it to **1.0 (disabled)**, pinned by a test. `src/striping.py` clips A1 output to
`[0, 255]`, so a legal dark pixel is written as the nodata sentinel; measured on the 38 training
windows as a deploy **proxy** (labelled a proxy — the deploy statistic is per SeamMap frame off the
160 m grid, not a whole-window statistic), the share of own-tile-passing cells carrying ≥1 "nodata"
context pixel goes **0.00 % (raw mosaic) → 2.67 % (native A1 statistic) → ~13 % (160 m statistic)**.
Enabling a zero-tolerance gate there first would delete a large slice of the A1 map for a radiometric
reason dressed as a data gap. And when R38 lands, **"clip the floor to 1" is not an acceptable remedy
in this ordering**: it would make blackened pixels invisible to this gate while leaving the embedding
damage intact. Only an explicit nodata mask (or no clip) lets the A1 default be flipped to 0.0
honestly.

**`scripts/f_region_stageb.py` keeps the own-tile-only gate, deliberately.** Its canvas is ~57 %
nodata — the one place a mostly-nodata *context* is common rather than rare — but the F programme was
hard-aborted 2026-07-30, so re-gating it would mean re-running a 907-frame build whose product was
already rejected. Recorded in a comment at the call site, with the drop-in named, for anyone
reopening F.

**Re-validated against real CTX, read-only, 38 s.** Streamed the full 47,420² `E-8_N32` — the one
cached shipped tile with a real mosaic gap — on the shipped (pre-R01, phase-0) lattice and
reproduced every load-bearing number to the digit: 1,282,224 DN-0 px; **2,187,441** interior cells;
**1,280** at own_zf > 0.3, whose mask equals the shipped `E-8_N32_prob.tif` NaN pattern at
**100.000000 %**; and among the 2,186,161 own-passing cells, **exactly 290** with any context nodata
(191 / 131 / 76 / 20 / 0 above 0.1 / 0.2 / 0.3 / 0.4 / 0.5; max **0.4253472**). Forty of those 290
were then re-computed through the real `src.mapping.context_zero_fraction` on independent windowed
reads: **0 mismatches**. Any other count than 290 would have meant the helper's box origin was wrong.

**Blast radius: no rebuild is forced by R13 alone.** ~770 of 56,870,060 shipped cells turn NaN
(exactly 290 of 19,685,689 on the nine tiles measurable by exact block arithmetic). No published
statistic moves at three decimals — `prob_mean`, `rich_share_at_0p5`, the sd(log10 pred/label) level
table and the 26-tile mosaic are all unaffected by 1e-5 of cells. It does change output bytes, so it
folds into the R01/R07 pass at no extra cost; `docs/PENDING_REBUILD.md` row 8. Two corrections to the
finding's own text, both minor: **six** of 26 tiles contain nodata, not five (E-4_N36 814, E-4_N44
45, E-8_N32 1,280, E0_N32 51, E4_N32 1,210, E4_N36 6 = 3,406); and the map-wide "provable" bound
treats out-of-raster neighbours as clean, so the ~153.6 k Murray-tile-edge cells sit outside its
premise (≈2 extra cells at the measured rate — scope the word, not the severity).

**Severity stays low for the shipped mosaic** (≤2.05e-05 of cells, blob-shaped gaps, zero cells
provably above ctx 0.7) and is deliberately **not** inflated to medium on the A1 arm: that risk is
R38's severity manifesting through R13, and double-counting it across two findings is the inflation
this audit warns about. What R13 must not be is closed as "won't fix on rarity grounds" — the
per-cell error is 3–4.6× the map's own IQR and the R38 ordering constraint is real.

**Verification.** 20 new tests in `tests/test_mapping_context_nodata.py`; fast suite 731 → **750
passed**, 1 skipped. Mutation-verified **7/7**, every mutant green against the pre-R13 suite: drop
the context term from `usable`; conflate the two counters; sum instead of de-duplicating the cell
sets; omit the `nodata_gate` block from the sidecar; drop the threshold from the sweep manifest;
un-centre the context box; loosen either default.

**Found while wiring A1, NOT fixed, recorded so it is not re-discovered: the A1 driver has no
resume-match guard.** `scripts/striping_a1_map.py` deletes a `partials/<tile>/_sweep.json` in its
`--clean-partials` path but **never writes one**, and never calls `sweep_manifest`. So R14's
sweep-identity protection covers `map_region.py` only: a resumed A1 run can mix partials from a
different `--win-px`, head, calibrator or masking threshold, with only the `grid_id` lattice check
standing between it and a silent two-run raster. R13's thresholds do land in the A1 *sidecar*, so the
product is at least self-describing, and the context gate is disabled on that arm anyway — but this
is a genuine R14-shaped hole on the A1 path. Left open deliberately: it is R14's scope, not R13's,
and A1 is already blocked behind R08, R38 and R06, so it should be closed in the same change that
unblocks A1 rather than bolted on here.

**R13 was the last Mapping-gate finding.** With R01, R14 and R84 already closed, nothing now blocks
the corrected baseline map except the rebuild's own isolation gate (criterion 5, the ≈110 GB backup).
A1 still needs R08, R38 and R06 — plus the A1 resume guard noted above.

## 2026-08-10c — R38: A1's clip floor was the nodata sentinel, and moving it was not the fix

**The collision.** `a1_apply` clipped to `[0, 255]`, so a valid pixel darker than about
`med - 4.51*iqr` was written as **0** — and 0 is unambiguously "no data" everywhere downstream
(`a1_stats` keeps only `DN > 0`; `src.mapping` inferred its whole nodata mask from `arr == 0`).
Legitimately dark terrain was therefore counted as a mosaic gap. Measured previously on the real
native patch stacks: 0.041 % of valid pixels on the training path, 0.04–0.41 % at deploy, **6.7 %**
of deploy-sim tiles carrying at least one false-black pixel while still passing the mask, and whole
tiles reaching `own_tile_zero_fraction = 1.00` in low-IQR frames. Nothing documented the choice, and
three sibling implementations of the same stretch already floored at 1 and said why.

**The register's own fix — `np.clip(..., 1, 255)` — is necessary and *not sufficient*, and R13 is
how we know.** Measured yesterday with the real frozen ViT: DN 0 and the perfectly legal DN 1 move
the prediction **identically to three decimals** in every regime tested. The damage is *blackness*,
not the sentinel value. So flooring at 1, on its own, would have left the embedding damage entirely
intact while removing the only signal that anything was wrong — the pixels stop reading as 0, so
R13's context gate can no longer see them. That is an A1 map that passes every gate and is still
wrong. Three changes therefore landed together:

1. **Valid pixels floor at `A1_VALID_FLOOR = 1`**, so DN 0 in an A1 array means nodata and nothing
   else. `a1_apply(..., floor=0)` reproduces a pre-R38 artifact deliberately.
2. **`predict_window` takes an explicit `nodata_mask`**, and both A1 drivers pass one derived from
   the **raw** DN before normalization. This is the durable half. Inferring coverage from a pixel
   *value* is only safe while nothing downstream can synthesize that value — A1 could, and the next
   change to the transfer function would quietly make it unsafe again. `nodata_mask=None` keeps the
   inference, which is exact for the raw Murray mosaic (its GeoTIFF declares `nodata=0` and the
   minimum valid DN is 1, because Murray bottom-clips valid data).
3. **The destroyed texture is counted separately** (`a1_clip_counts_from_hist`) and recorded per
   tile as `a1_clip_*`, with `--warn-clip-fraction` (default 1 %). A clipped pixel is a *radiometric*
   loss, not a coverage loss; conflating the two is what produced this finding, so the fix keeps
   them in separate columns.

**Where the count is computed matters.** It comes off the per-frame DN histogram that
`frame_hist_native` already builds — uint8 has 256 values, so "how many pixels clip" is a dot
product, exact and free. The obvious alternative, accumulating per read window, is both approximate
and resume-dependent: windows overlap by 96 px so pixel counts double-count the seams, and a resumed
run only sees the windows it recomputed. Once per tile, from the histogram, has neither problem.

**Measured on real CTX under R07's corrected native statistic — and it is far smaller than the
bracket the finding was filed with.** The 0.04–0.41 % estimate came from the *160 m* statistic, i.e.
R07's mismatch. Streaming whole cached Murray tiles:

| tile | valid px | floored | ceiled | clip fraction | frames affected | worst frame | native IQR min/med/max |
|---|---|---|---|---|---|---|---|
| `E-8_N32` | 2,247,360,528 | 203,806 | 62,440 | **0.011847 %** | 16 / 54 | 0.341 % | 8 / 36 / 123 |
| `E4_N44` | 2,248,656,400 | 3,619 | 29,812 | **0.001487 %** | 12 / 48 | 0.393 % | 11 / 34 / 61 |
| `E8_N44` | 2,248,656,400 | 44,357 | 1,293 | **0.002030 %** | 11 / 31 | 0.072 % | 8 / 30 / 57 |

So **1.5e-05 to 1.2e-04 of valid pixels**, against the 4e-04 – 4.1e-03 the finding was filed with —
between 3× and 27× smaller. R07 is why: the native per-frame IQR runs a median of 30–36 against
`A1_REF_IQR = 27.7`, so the typical gain is a *shrink* (≈0.8×) and almost nothing reaches the bounds.
The frames that do clip are exactly the low-IQR ones (min 8 → gain 3.5×), as the finding predicted;
they are a small minority (11–16 of 31–54 frames touch the bounds at all, and the worst single frame
loses 0.07–0.39 %). Note the split is not one-sided — `E-8_N32` and `E8_N44` mostly *floor* while
`E4_N44` mostly *ceils*, which is why the record keeps the two ends apart.

At ~1e-04 of pixels the case for retuning the transfer function is weak, which is what Brian ruled
on: **record the loss, leave `A1_REF = (125.0, 27.7)` alone.** A test pins those constants so a
silent retune cannot happen. Also confirmed incidentally, and it narrows **R08**: all 133 frames
across the three tiles cleared `A1_MIN_FRAME_PX`, so **0 frames were too small** and the tile-wide
fallback covers only 0.0059–0.0108 % of valid pixels — consistent with the 0.0081 % measured on
E-12_N36 for R07.

**Sibling defect found and fixed — and it was not cosmetic.** `a1_stats` and `a1_stats_from_hist`
substituted `iqr = 1.0` for a degenerate (zero) IQR. The verify doc called that unreachable and
harmless; it is neither harmless nor merely cosmetic, because `a1_stats_native_tile` admits a frame
only when `iqr > 0` — and the fabricated 1.0 **sailed straight through the guard written to catch
exactly this**, handing that frame a gain of `s0/1 = 27.7×` instead of routing it to the fallback
statistic. Both now return NaN, so the guard works as intended.

**Also folded in:** `scripts/striping_a1_infer_crop.py` re-inlined the stretch with its own
`[0, 255]` clip; it now calls `a1_apply` and passes the raw-DN mask, so there is one definition of
the floor rather than three.

**Train and deploy change together, by design.** `scripts/probes/_w2_fang_embed.py` reaches the same
`a1_apply`, so the training input moves in the same commit as the deploy input — a floor changed on
one side only would have re-opened R07's train/deploy mismatch on a second axis. Consequence:
`dataset_v2/fang_embeddings_a1` and `models/deployable_a1` were baked under floor 0 and must be
re-made. That is not new work — it is already row 7 (R07's re-embed + retrain), and R38 rides along
at no extra cost. `docs/PENDING_REBUILD.md` row 9 records it.

**This is what let R13's A1 context gate turn on.** `scripts/striping_a1_map.py
--max-context-zero-fraction` went from its disabled default of 1.0 to **0.0**, matching the baseline
arm. The R13 test that pinned it *disabled until R38* has been rewritten to pin the new invariant
and to assert `A1_VALID_FLOOR > 0`, so the gate cannot be left on if the floor is ever put back.

**Blast radius: nothing on the record moves.** No shipped raster goes through `a1_apply`
(`reports/map_a1/` does not exist — that is R06 — and `reports/map_region/` never imports
`src.striping`). The banked −0.024 LOIO cost and the banked 28 % η² reduction are unaffected, and
the alleged differential-footprint confound in the η² comparison did not occur (218,089 = 467² = the
complete interior grid in both arms). What changes is future A1 output, and the training input it
will be compared against.

**Verification.** 15 new tests in `tests/test_a1_clip_floor.py`; fast suite 751 → **766 passed**, 1
skipped. Mutation-verified 4/4: restore the `[0, 255]` floor; make `predict_window` ignore the
supplied mask and re-infer `arr == 0`; restore `or 1.0` on the degenerate IQR; make the clip counter
report zero. The histogram counter is cross-checked against an independent array-level
implementation on the same data.

## 2026-08-10d — R08 ratified: normalize the fallback population, never drop it

R08's *mechanism* was fixed by R07 (`a1_normalize_native` normalizes every valid pixel and raises
rather than returning raw DN). What stayed open was the **contract**: is the tile-wide fallback the
right answer for a pixel in no qualifying frame, or should those pixels be dropped? And what should
`A1_MIN_FRAME_PX` be? Both are now settled by measurement rather than preference, and **neither
needed a code change** — the ratification is a pair of tests that fail if the contract is later
"tightened".

**R13 is what made the question decidable, and it inverted the intuition.** Dropping a pixel makes
it nodata, and R13's zero-tolerance context gate then sterilises every coarse cell whose 96-px box
touches it. So the cost of dropping is not the pixel count — it is how many *cells* those pixels
sterilise, which depends entirely on their shape. Measured over three whole cached Murray tiles:

| tile | unlabelled valid px | run length med / p90 / max | blocks touched | (a) keep fallback | (b) drop |
|---|---|---|---|---|---|
| `E-8_N32` | 241,928 (0.0108 %) | 1 / 2 / 15 | 29,919 | 1,570 cells (0.0718 %) | 95,855 (**4.38 %**) |
| `E4_N44` | 149,041 (0.0066 %) | 1 / 2 / 6 | 22,116 | 0 (0.0000 %) | 73,811 (**3.37 %**) |
| `E8_N44` | 131,371 (0.0058 %) | 1 / 2 / 6 | 21,013 | 0 (0.0000 %) | 68,077 (**3.11 %**) |

The population is **isolated single pixels** — a median horizontal run of 1 — scattered over 21k–30k
of each tile's 2.19 M blocks. They are rasterization-precision gaps inside the *dissolved* SeamMap
polygons, not real coverage holes; nothing about them looks like missing ground. Dropping them would
trade **3.1–4.4 % of every tile** to avoid a **1e-4** radiometric approximation — wrong by roughly
three orders of magnitude, at a 400–530× amplification in cell-equivalents. This is the same
scattered-vs-blob asymmetry R13 measured on the embedding side (shape matters ~6× at equal pixel
count), biting in the same direction on the masking side.

**Brian's ruling (2026-08-10): ratify the tile-wide fallback as-is.** A nearest-frame refinement was
offered and declined — at this population size and scatter it buys a more faithful scale for a
handful of isolated pixels in exchange for code, and the fallback is already a robust statistic over
the enclosing tile.

**The 50-px floor is a tripwire, not a tuning knob.** Across four real tiles and 214 dissolved
frames, exactly **one** fell below `A1_MIN_FRAME_PX` — E-12_N36 1 of 81 (measured for R07), then
E-8_N32 0 of 54, E4_N44 0 of 48, E8_N44 0 of 31. Its value therefore trades nothing off on real
data; what it must keep doing is *route* a degenerate frame to the fallback rather than admit one,
which is exactly what the R38 fix to `a1_stats`'s `or 1.0` restored (the fabricated IQR had been
walking straight through this guard).

**What R08 does not include any more.** The register's third clause — "the small-frame tail is where
re-anchoring moves the A1 statistic most (>1 DN on 11 of 74 frames)" — was **dissolved by R07**,
which moved the statistic off the baseline product's grid onto the native tile. There is no
re-anchoring sensitivity left to worry about.

**No rebuild consequence.** The code is unchanged, so no artifact drifts and `docs/PENDING_REBUILD.md`
gains no row. The measurement scripts are `r08_price_the_contract.py` and `r38_clip_measure.py` in
the session scratchpad; both are read-only and take ~2–3 min per tile (one tile ran long at 7,903 s
under I/O contention — the work is I/O-bound, not compute-bound).

**Verification.** Two new tests in `tests/test_a1_statistic.py` carrying the measured rationale in
their docstrings: one asserts an unlabelled valid pixel stays valid *and* carries the fallback
statistic specifically, the other pins the floor's role. Fast suite **768 passed**, 1 skipped.

**With this, the audit register has no finding left that is not waiting on hardware.** R08 was the
last one that could be closed by analysis. What remains: **R06** (A1 has never been generated —
needs the rebuild), the **A1 resume guard** (small, R14-shaped, noted 2026-08-10b), and
**R03/R83/R84** (leg 4's pixel-scale size floor). The rebuild itself is still gated on isolation
criterion 5, the ≈110 GB backup.

## 2026-08-11 — the A1 resume guard: R14's protection reached only one of the two drivers

Found 2026-08-10 while landing R13, fixed now. `scripts/striping_a1_map.py` deleted a
`partials/<tile>/_sweep.json` in its `--clean-partials` path but **never wrote one** and never
called `sweep_manifest`. R14's sweep-identity protection therefore covered `scripts/map_region.py`
only, and on the A1 arm the sole thing between a resumed run and a two-run raster was the `grid_id`
check.

**`grid_id` cannot do that job, and R01 is why.** It is a *lattice* identity, and both drivers were
deliberately put on the same lattice in one commit precisely so A1 could never diverge from the
baseline. So it matches by construction between any two A1 runs — including two that used different
heads, window sizes, masking thresholds or A1 statistics. Their partial filenames collide too
(`{row:06d}_{col:06d}.npz`). Every downstream structural check then passes: each `.npz` is perfect,
set-equality is satisfied, and the raster comes out the right shape. R14 measured exactly that state
on the baseline arm — 63.1 % of finite pixels from the stale run, nothing visibly wrong.

**A1 needs *more* identity than the baseline, not the same.** Its input is **derived**: two runs can
agree on window geometry and head and still have normalised the DN differently. `a1_sweep_manifest`
is `map_region.sweep_manifest` plus five fields, each pinning a decision made in this audit:

| field | pins |
|---|---|
| `norm_arm` | R07's statistic definition (native, per frame, tile support) |
| `a1_ref` | the transfer function's target (125.0, 27.7) |
| `a1_clip_floor` | R38 — moving the floor changes pixels |
| `a1_min_frame_px` | R08's ratified fallback boundary |
| `a1_seammap_digest` | the frame partition itself |

**The digest hashes every shapefile sibling, not just the `.shp`** — the `.prj` is load-bearing.
The frames are reprojected into the tile CRS before rasterization, so a changed projection silently
moves which pixels belong to which frame, and therefore every per-frame statistic, without touching
one coordinate in the `.shp`. That is CLAUDE.md's #1 gotcha appearing in a provenance field.

**It digests A1's inputs, not its derived statistics — deliberately.** Hashing the per-frame
`(median, IQR)` dict would be a stronger identity but would force the ~3 min streaming pass *before*
the driver could decide whether a tile is already committed. Input digests are cheap, change exactly
when the statistics would, and let `process_tile` be reordered so the whole identity is built up
front. Consequence, and it is a real improvement: a resumed run over already-done tiles now costs no
streaming reads at all, where before it paid ~3 min per tile to discover it had nothing to do.

Also landed: `tile_is_reusable` is passed the sweep manifest instead of `None` (content alone cannot
see a raster that is structurally perfect and was normalised by a different arm), and the manifest is
recorded as the sidecar's `run` block, matching `map_region`.

**No rebuild consequence** — `reports/map_a1/` has never existed (R06), so there is nothing on disk
to invalidate. `docs/PENDING_REBUILD.md` gains no row.

**Verification.** 16 new tests in `tests/test_a1_resume_guard.py`; fast suite 768 → **784 passed**,
1 skipped. Mutation-verified **5/5**: never write `_sweep.json` (the defect exactly as found); drop
the A1-specific identity from the manifest; drop `a1_seammap_digest`; hash only the `.shp` and ignore
the `.prj`; pass `None` to `tile_is_reusable`. The two wiring tests scan the **AST** of
`process_tile`, not its text, so a docstring describing the guard cannot satisfy them.

**With this, every code task the audit register produced is done.** What remains is R06 (A1 has
never been generated), R03/R83/R84 (leg 4's pixel-scale size floor, a different track), and the
rebuild — gated on isolation criterion 5, the ≈110 GB backup.

## 2026-08-11b — leg 4 / R84: the abundance layer now states which boulders it counts

PLAN_RegionalMap leg 4 (the LOIO truth anchor) and every external comparison were blocked by
**R84**: the deployed abundance raster is quantile-matched onto a pool that **mixes** per-image
detection floors, and `write_geotiff` wrote **no metadata at all**, so a shipped `*_abundance.tif`
could not state what its numbers count. R84 is explicit that R03's remedy is *necessary but not
sufficient* — "fine for legs 1–3 (within-map rank statistics), insufficient for leg 4 … because the
deployed layer's floor is a *mixture* that no per-image sidecar can state."

**R84's 78.4 / 21.6 was flagged unverified. It is now independently verified.** Measured read-only
over all 38 `cache_v2/reprojected_detections/*.gpkg`, `cache_v2/pds_labels/*.LBL` and the S=32 label
pool:

| quantity | R84 claimed | measured 2026-08-11 |
|---|---|---|
| pool size at S=32 | 161,005 | **161,005** ✅ |
| coarse / fine **tile** share | 78.4 % / 21.6 % | **78.3914 % / 21.6086 %** ✅ |
| `calibration.npz` `t2_y` max == pool max `fa` | the proof of which pool | **0.293242 == 0.293242** ✅ |

**The audit's warning was worth heeding: tile share is not image share.** By image the split is
**68.4 / 31.6**. Quoting one for the other is wrong by ten points, and R84's number is the tile one.
Both are now carried, separately and labelled, precisely so they cannot be confused again.

**R83's correction to R03 also confirms, and it inverts R03's picture.** The effective floor is not
the Stage-1 polygon minimum: Stage 4's global `min_size_m = 1.4105 m` (1.5626 m²) is applied
*afterwards*, and it sits above every fine image's natural floor and below every coarse one's. So
the **fine** cohort's floor is uniformly the filter — 1.5626 m² for all 12 — while each **coarse**
image keeps its own, 2.9652–5.5719 m² (1.943–2.664 m diameter) over 26 distinct values. The coarse
cohort is the internally heterogeneous one; R03 had it the other way round because it read the
Stage-1 minima as if they were post-filter. Pool-wide: **27 distinct floors, tile-weighted mean
3.3687 m².**

**What landed.** `src/size_floor.py` — `effective_floor_m2` (the one line that is R83's correction),
`SizeFloorBasis` (measured, versioned, JSON-bankable, refuses a foreign version), and
`product_tags()`. `scripts/measure_size_floor.py` re-derives it in ~215 s and has `--dry-run` plus a
flag for every root. `write_geotiff(tags=...)` is R84's fix proper. Both map drivers stamp the same
basis on **every** raster they write — including `_prob.tif`, not just `_abundance.tif`, because the
rich/poor class is `fa > 1e-2` and inherits the identical floor dependence (R83). The run manifest
records the basis path and digest.

**Design choices worth not relitigating:**
- **Measured, never asserted.** A provenance field that asserts rather than measures has been caught
  four times on this project, so the basis is derived from artifacts, carries its inputs, and can be
  re-derived by one command. `SizeFloorBasis.from_records` is a pure seam so it is testable without
  touching 7 M polygons.
- **The pixel scale comes from the PDS `.LBL`, not the manifest.** `build_vclaire_manifest.py` takes
  `MapPixel_mpp` from the label *spreadsheet*, which is why two cohort rows are blank
  (`LabelSource: none`). Both are 0.5 m/px and the `.LBL`s are cached. Reading the `.LBL` makes a
  blank impossible rather than patching two of them, and the driver **refuses to build a basis** if
  any scale is unknown — a mixture measured with unknown members silently under-counts one cohort.
- **Tile-weighted, not image-weighted.** The layer is calibrated per tile; an image-weighted mean
  floor would over-represent the fine cohort by ~1.5× here.
- **An image with zero pool tiles contributes no floor.** It is in the cohort but not in the
  product, so it must not widen the stated range.
- **The per-image records stay out of the raster header.** 38 records do not belong in every tile's
  metadata; `SIZE_FLOOR_BASIS_VERSION` + the banked JSON is the audit trail.

**A trap the suite caught, worth recording.** `Path("")` is `.`, and `Path(".").exists()` is **True**
— so an unset `--size-floor-basis` sailed past an `exists()` guard and tried to parse the working
directory as JSON. The check is now "non-empty string, and `is_file()`", and a test pins `""`,
`None`, an absent attribute and a *directory* all yielding no tags. Absent must mean untagged, never
fabricated and never a crash.

**Scope, stated honestly.** This makes the size floor *recordable and recorded*, which is what R84
named and what unblocks leg 4's interpretation. Two things it does **not** do:
1. **The per-image half of R03 (item d) is still open** — `map_scale_mpp` and the measured floor are
   not yet persisted into the Stage-1 sidecar or `dataset_v2/labels/{obs}.json`. That is a producer
   change and therefore rebuild-pending; the product-level attribute does not depend on it.
2. **Leg 4 itself has not been re-run.** R83 measured `Spearman(sub-floor area share, per-image AUC)
   = −0.468, p = 0.003`, surviving every control — *including inside the 0.50 m/px cohort alone*
   (−0.467, p 0.016, n 26), so it is mostly small-boulder terrain rather than pixel scale. Re-running
   the LOIO anchor with the floor as a covariate is the next leg-4 step and is now possible.

**The basis is NOT yet banked.** `models/deployable/size_floor_basis.json` does not exist; the
measurement has only been run with `--dry-run`. Writing it is a one-command step into a protected
root and is Brian's call. Until it exists both drivers warn and emit untagged rasters — deliberately
the safe direction. It also **goes stale whenever Stage 4 re-runs**, so the rebuild must re-measure
it; `docs/PENDING_REBUILD.md` carries the note.

**Verification.** 16 new tests in `tests/test_size_floor.py`; fast suite 784 → **800 passed**, 1
skipped. Mutation-verified 5/5: ignore the Stage-4 filter (R03's own error); report image share where
tile share is meant; make `write_geotiff` drop tags again; accept an empty pool; assume the pixel
scale instead of reading it.

## 2026-08-18 — isolation criterion 5: the artifact snapshot exists

The last gate on the batched v2 rebuild. Criterion 5 asked for "an independent backup of the ignored
trees, ≈110 GB", deferred 2026-08-06 pending a drive. Brian supplied one; the snapshot is taken and
verified.

**Target: `D:\HiRISE2CTX Backup`, and it is genuinely independent.** Two distinct physical devices,
confirmed rather than assumed: disk 0 is the internal Micron NVMe (`MTFDKBA2T0QGN`, 1908 GB, C:),
disk 1 is a `WD My Passport 2628` on USB (1863 GB, D:). That matters because the two risks here are
different — a same-volume copy would have controlled the *rebuild overwriting the originals* (the
risk criterion 5 names, and the one that has bitten this project twice) but not device failure. This
covers both. 1012.6 GB free against 125.55 GB required.

**Snapshot: 11,260 files / 125.55 GB, verified 8/8 roots at 0 missing, 0 extra, 0 size mismatch.**

| root | files | GB |
|---|---|---|
| `repo\dataset_v2` | 2,482 | 78.34 |
| `repo\reports` | 2,750 | 21.18 |
| `repo\cache_v2` | 563 | 15.09 |
| `repo\dataset` | 328 | 4.97 |
| `external\hirise_40_vClaire` | 200 | 4.17 |
| `repo\models` | 3,049 | 1.41 |
| `repo\cache` | 1,837 | 0.37 |
| `external\hirise_priority10_detections` | 51 | 0.01 |

Cross-check worth recording: 11,260 files sits right against the **11,218**-file manifest the
2026-08-06 non-mutation verification built over the six repo roots — an independently derived
enumeration landing on the same count, with the difference explained by the two out-of-repo trees
(251 files) and five days of new `reports`/`models` output.

**The procedure is version-controlled: `scripts/backup_artifacts.ps1`.** Read-only on every source;
the only writes are under `-Destination`. `-DryRun` (robocopy `/L`) first, `-SkipCopy` to re-verify an
existing snapshot, `-Hash` for content verification. It refuses to run if the destination sits inside
the source tree, and it checks free space with 5 % headroom before copying.

**Two things the audit's own table got wrong, both now fixed in the procedure.**

1. **The junction trap is real and would have cost 61 GB.** `cache_v2/{ctx_tiles, hirise_decimated,
   hirise_jp2, pds_labels}` are all junctions into `cache/`, and `robocopy /E` **follows junctions**.
   Without `/XJ` this snapshot would have silently duplicated 41.2 GB of Murray CTX zips and 19.8 GB
   of HiRISE JP2s — both re-downloadable. Measured proof the exclusion works: `repo\cache` came out at
   **0.37 GB** instead of 61.4 GB. Both `/XJ` and an explicit `/XD` are used, because this is the
   documented trap and one mechanism is not worth trusting alone.
2. **The 110 GB figure under-counted by ~15 GB.** The table enumerated four derived cache subdirs at
   4.3 GB and missed `cache_v2/hirise_color` (8.89 GB), `cache_v2/validation` (2.24 GB), `craters`,
   `minconf_sweep`, `stage7` and the `pds_*` trees — **the cached PDS `.LBL`s among them**, which are
   load-bearing for the HiRISE SP1 correction *and* for `src.size_floor`'s `MAP_SCALE` (the
   authoritative pixel scale, since the manifest column is blank for two images). The rule is now
   "copy everything except the two big re-downloadable archives", which is simpler, has less to get
   wrong, and captures what an enumeration missed. Real total: **125.55 GB**, not 110.

**Verification level, stated precisely.** What is established is a **path-and-size** match across all
11,260 files — the same standard the 2026-08-06 non-mutation check used. A **SHA-256 content pass**
(`-SkipCopy -Hash`, ~250 GB of reads across both devices) is running separately; until it returns,
this snapshot is verified against truncation, omission and misplacement but **not** against silent
bit corruption. Criterion 5 should be read as closed at that standard and upgraded when the hash pass
lands. The verdict JSON under `D:\HiRISE2CTX Backup\_backup_meta\` records which level was achieved
via its `hashed` flag, so the distinction is auditable rather than remembered.

**What this unblocks.** The rebuild. `docs/PENDING_REBUILD.md`'s nine rows (R74, R27, R28, R29/R75,
R65, R01, R07, R13, R38) plus R03 item (d), R23's provenance sidecars and a post-Stage-4 re-run of
`scripts/measure_size_floor.py` can now be executed as one batched pass, following the DAG and
remaining safety notes in `docs/CODE_REVIEW_AUDIT_2026-08-06.md`. **The snapshot is not a licence to
skip the rest of that document** — the runtime write guard is still test-only, so hand-run producers
and notebooks continue to depend on the absolute-scratch-root discipline. What has changed is that a
mistake is now recoverable instead of terminal.

**Standing caveat.** This snapshot is a *point-in-time* copy of the pre-rebuild state. Once the
rebuild starts writing, it becomes the only record of what the artifacts looked like before —
so it must not be refreshed mid-rebuild, and `--DryRun` remains the right first move on any
re-snapshot.

## 2026-08-18b — rebuild approach: Sherlock, retrain-as-frozen, and the size floor deferred to v3

Design discussion held before drafting the execution plan. Four rulings, plus one measurement that
changes what a "single size floor" can mean.

### Context that reframes the whole rebuild (Brian, 2026-08-18)

**A v3 detection campaign is already in progress** — BoulderNet is being retrained and applied to a
more diverse set of locations. That will force a rerun regardless. So **this rebuild is a WAYPOINT,
not a final result**: its purpose is to characterise honestly what the *current* dataset supports,
understanding that the dataset is about to be superseded. Every decision below follows from that.
(It is also consistent with R23's existing disposition — Brian's ruling there was already "the fix is
a v3 re-detection dataset he will supply".)

### 1. The frozen recipe is RETRAINED AS-IS, not re-selected

The recipe `fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2` was selected by a bake-off on the *pre-fix* labels,
and the rebuild moves the target underneath it: R74 recovers ~3,236 S=32 tiles (93 % rich) and
R29/R75 swaps 6,202 out / 6,255 in, so ~6 % of the pool changes status and rich prevalence moves
**0.3598 → 0.3733**. Strictly, "which recipe is best" reopens.

**Ruled: retrain the frozen recipe unchanged and report the new numbers.** Not a full bake-off. Two
reasons: it preserves the pre-registration property that makes the numbers trustworthy, and
re-selecting against a label basis that v3 will replace spends effort on a waypoint. The banked
headline figures (pooled PR-AUC **0.7832**, median per-image AUC **0.7865**, prec@5 % 0.948) must all
be re-derived because they are prevalence-conditioned — not because they were wrong, but because they
describe a different label basis.

**Scope reduction found while checking this:** the recipe card is embeddings-only
(`embedding: fang_vit_b16_gem_p3`, no tabular features), so **R27 and R28 do not touch the deployed
model at all** — they affect the GBM baseline and the W1 error atlas only.

### 2. GPU work runs on SHERLOCK — and the argument is confidence, not throughput

Brian's reasoning: a long local pass will be interrupted by laptop sleep, and it is easier to have
confidence in a Sherlock run. That is the right criterion — a 16 GPU-h pass that silently suspends at
hour 3 is worse than a slower run you can trust.

**The split falls out of the data, and the Sherlock half is cheap to stage.** Measured:

| needs to go to Sherlock | GB |
|---|---|
| `dataset_v2/context_patches` (own-patch geometry self-check) | 17.00 |
| `models/pretrained` (Fang-ViT ckpt) | 0.32 |
| `cache_v2/ctx_windows` (what the embedder reads) | 0.19 |
| `dataset_v2/labels` + `splits` | 0.01 |
| **total up** | **~17.5** |

Coming back: ~35 rasters + sidecars, order **300 MB**. The 26 Murray CTX zips (41 GB) are fetched
**on** Sherlock from Murray Lab, as the previous regional run did — not uploaded. And **`packaged`
(48.93 GB) does not need to move at all**: it is Stage-5 tabular packaging for the GBM baseline,
irrelevant to an embeddings-only recipe.

Stages 1→5 stay **local**: they need the local archives (41 GB CTX zips, 20 GB JP2s, 4.2 GB
detections) and, because everything is cached, they need **no network**. Sleep mitigation there is
`powercfg /change standby-timeout-ac 0` plus the fact that they are per-image resumable.

**On the queue worry — the answer is a job ARRAY, not one long job.** `run_region_array.sbatch`
already submits 26 independent elements at ~0.6 GPU-h each; `SHERLOCK_RUN.md` records 13–19 GPU-h / N
≈ 2–3 h wall clock on 6 GPUs. Short elements schedule far better than one 16-hour allocation. And
**R14 is what makes pre-emption safe**: before it, a killed job left partials that either crashed
assembly forever or silently mixed two runs at 63 % stale pixels; now atomic staged writes,
CRC-checked partials, content-and-provenance resume and set-equality assembly mean a pre-empted
element can simply be resubmitted. The queue anxiety is largely retired by work already done.

### 3. The size floor: Brian's principle is ACCEPTED, the 1 m target is not attainable, and it moves to v3

**The principle, and it is a better framing than the audit's.** Brian: the mixed floor is wrong
because *CTX resolution does not change* — it is incorrect for the label definition to vary with
whatever HiRISE resolution happened to be available at that location. The audit's disposition was
"retain and document"; this states the mixed floor as a **defect of definition**, not an
inconvenience, and that framing should be carried forward.

**But the arithmetic says a 1 m floor goes the wrong way.** Converted to equivalent-circle diameter
(the unit `min_size_m` uses), measured 2026-08-11:

| | area | diameter |
|---|---|---|
| current global filter | 1.5626 m² | **1.4105 m** |
| fine cohort natural floor (12 images) | 0.830–1.156 m² | **1.028–1.213 m** |
| coarse cohort natural floor (26 images) | 2.965–5.572 m² | **1.943–2.664 m** |

A 1 m diameter floor is **0.785 m² — below the current filter and below even the fine cohort's
detection limit**, so imposing it removes nothing anywhere. It is equivalent to no filter, and it
makes the mixing *worse*, because today's 1.4105 m filter at least trims the smallest fine
detections while removing nothing from the coarse cohort. **A floor cannot be imposed below what the
detector found.** Unification requires *raising* the floor to the worst image's limit —
**2.664 m diameter / 5.572 m²**, set by `ESP_017355_2260`.

Cost of that, already measured: R83 — fine-cohort rich prevalence **halves** (0.326 → 0.164) while
coarse barely moves (0.369 → 0.366), up to **64 %** of one image's tiles flip, and it is a
**re-ranking, not a rescale** (within-image Spearman 0.60–0.98), so calibration provably cannot
absorb it. R03's verifier: ~**67 %** of a fine image's labelled boulder area lies below it.

**Ruled (TENTATIVE, Brian 2026-08-18): option C — make the uniform floor a v3 design requirement.**
Keep mixed-and-documented for this waypoint rebuild; specify uniform (or uniformly-achievable)
HiRISE pixel scale in the v3 campaign, where the locations are being chosen anyway, and *design* a
~1.1–1.2 m uniform floor rather than retrofitting a 2.66 m one. Retrofitting spends 67 % of the fine
cohort's labelled area on a dataset that is about to be replaced. The two rejected alternatives are
recorded because they remain live if v3 slips: **A** common floor at ~2.66 m over all 38 images
(uniform, keeps diversity, coarse and expensive); **B** fine images only at ~1.2 m (uniform *and*
fine, but the pool drops 161,005 → 34,791 tiles = 21.6 % and loses 26 images of spread, cutting
against the v3 diversity goal).

**Methodological caveat for whenever a common floor is set.** What was measured is the **sample
minimum** polygon per image, not a completeness limit — over ~250k polygons the smallest one found is
a noisy estimator. The principled floor is the **size-frequency-distribution rollover per image**.
That is real work and must not be skipped by reusing the sample minimum.

### 4. Efficiency was NEVER reviewed, and it is now a named gap

Checked against the register: the 2026-07-31 review's 35 areas were correctness, provenance and
safety. Every `slow` hit concerns pytest markers; the GPU-h figures concern *wasted* work (R36's
tautological gate authorising ~265 CPU-h + 33 GPU-h), never optimisation. **There is no performance
finding in the register.** With wider-area inference as a goal, that is a real gap.

Evidence of headroom, both found *incidentally* while fixing correctness this tranche — nobody was
looking: R07's per-frame statistic 45 min → **3.4 min/tile (13×)** via a bbox pre-filter +
`bincount`; R13's context gate 0.44 s/+419 MB → **0.016 s/+18 MB (27×)** via a lattice-block form.

Where the cost is: ~0.6 GPU-h/tile, and a tile is 1479² ≈ **2.19 M cells, each taking its own
ViT-B/16 forward at 224×224**. Extrapolated to full Murray coverage (~4,050 tiles) that is ~2,400
GPU-h — the scaling wall.

Hypotheses, ranked by expected payoff, **explicitly not measured**:
1. **The 96→224 bicubic upsample may be pure waste.** Interpolation cannot add information, yet it
   costs 5.4× the pixels and 196 tokens instead of 36 — and attention is quadratic, so that term is
   ~30×. Possibly the largest single win. It IS a recipe change and needs re-validation.
2. **~9× redundant computation** — stride 32 with a 96-px window embeds every pixel ~9 times. Dense
   prediction shares the backbone; the per-box resize is what currently blocks that.
3. Infrastructure only, no recipe change: `torch.compile`, `channels_last`, SDPA, larger batches.
   Perhaps 1.3–2×, low risk.
4. **Distillation** to a small CNN — potentially 10–100× for production, at the cost of the
   frozen-foundation-model provenance.
5. **Cell skipping** via a cheap texture pre-filter — big potential, but it *biases* the map and must
   be validated against full computation.

Items 1 and 2 are recipe changes and therefore belong with **v3**, not this rebuild. **Agreed next
action: profile one tile** (read / resize / ViT / head / write) before proposing anything. Deferred to
a following session by Brian.

### 5. Incidental: seven dead embedding stores must not be regenerated

`dataset_v2/fang_embeddings_f{,_global,_minnaert,_minnaert_w,_minnaert_wl,_minnaert_cubic,_minnaert_center}`
— 0.39 GB each, ~2.7 GB total — are F-programme arms. F was hard-aborted 2026-07-30. They are backed
up so nothing is at risk, but the rebuild must **not** regenerate them, and the execution plan needs
an explicit line saying so.

### 2026-08-18 addendum — the SHA-256 pass FAILED because the laptop slept (NOT a drive fault)

> **RESOLVED, same day.** My first reading of this was a possible hardware fault and it was wrong.
> Brian: the machine went to sleep at that moment because he closed the lid. That explains every
> symptom better — USB devices power down on suspend, so the bus drop, the seven Event 51s mid-I/O,
> the volume re-enumerating on wake and the process dying are all one cause. **The drive is fine; do
> not chase ports, cables or the enclosure.** The fix is simply to keep the machine awake and re-run
> `scripts/backup_artifacts.ps1 -SkipCopy -Hash`.
>
> **This is worth more than a correction, though: it is Brian's own Sherlock argument, demonstrated.**
> Hours earlier he chose Sherlock for the GPU work on exactly this reasoning — "a big pass will end up
> being affected by sleep on the laptop and it will be easier to have confidence it will be run on
> Sherlock." The prediction then materialised within the same session, on the backup verification.
> Two consequences for the rebuild plan: the Sherlock decision is empirically vindicated, and
> **Stages 1–5, which stay local, need explicit sleep mitigation** — `powercfg /change
> standby-timeout-ac 0` for the duration, and do not close the lid. A multi-hour local CPU pass is
> exposed to precisely this.
>
> The paragraphs below are retained as written, except the hardware speculation, which is superseded.

Correction to the entry above, which said a content pass was "running separately". It ran to roughly
95 % (119 of 125.55 GB read) and then **died**, on this:

```
Get-FileHash : The file 'D:\HiRISE2CTX Backup\repo\cache_v2\hirise_color\ESP_045139_2270_COLOR.JP2'
cannot be read: A device which does not exist was specified.
```

"A device which does not exist" is a **USB bus drop**, not a file-level problem. Corroborated in the
System event log at 16:24:00 — **seven `disk` Event ID 51** records (device error during a paging
operation) plus an `Ntfs` Event 98. The volume re-enumerated afterwards (the raw perf counter reset
to zero, which is what first exposed this), so it came back, but it went away mid-read.

**No verdict JSON was written for the hash pass**, so there is **no content verification** of this
snapshot. Do not read the earlier "hashed" language as achieved.

**What IS established, and it is not nothing.** Path-and-size verification passed **twice**: once at
the end of the copy, and again *after* the disconnect (`-SkipCopy`, 8/8 roots, 11,260 files, 0
missing / 0 extra / 0 size mismatch). So every file is present, readable and the right length after
the fault. The 125.8 GB write counter independently corroborates the copy. That is the same standard
the 2026-08-06 non-mutation check used, so **criterion 5 stands closed** — but at path-and-size, not
byte-for-byte.

**~~The device itself is now the open question~~ — SUPERSEDED, see the note at the head of this
addendum.** The cause was suspend, not hardware. The remaining action is simply: keep the machine
awake and re-run `scripts/backup_artifacts.ps1 -SkipCopy -Hash`.

What still stands: until that pass completes, treat the snapshot as **good but unconfirmed at the byte
level**, and keep the source intact on C: rather than relying on D: alone. The residual exposure is
narrow — a bit flip that preserved file length — but it is exactly what the hash pass exists to rule
out, and it has not been ruled out yet. That is a *pending verification*, not a suspected fault.

**This does not block the rebuild.** The snapshot is materially better than no snapshot, every file
verifies present and correctly sized twice over, and the rebuild's own risk (overwriting the
originals) is covered. It does mean the "independent device" reassurance is weaker than it looked at
16:00, and that is worth knowing before the source trees are overwritten.

---

## 2026-08-18c — Map inference PROFILED: the read is 32 % of wall-clock and 86 % of that is `rasterio.open`

Efficiency was the one axis the 2026-07-31 review never covered (35 areas, all correctness /
provenance / safety). Brian asked for one tile profiled before any optimisation is proposed. This is
that measurement. **Read-only throughout** — the profilers live in the session scratchpad, opened
cached artifacts only, and wrote nothing under any live root.

**Setup.** Tile `E-12_N36` (47,420², 1.61 GiB zip), shipped config `--win-px 4096 --batch 96`,
`tile_px=32`, one full mid-tile window (15,876 cells) instrumented stage by stage with
`torch.cuda.synchronize()` around each, extrapolated ×144 windows/tile. GPU = RTX 5070 Laptop
(8 GiB, sm_120), torch 2.12.0+cu130.

| stage | s / window | % | h / tile (×144) |
|---|---|---|---|
| 1 windowed read | 11.117 | **32.1** | 0.445 |
| 2 grid enumeration | 0.002 | 0.0 | 0.000 |
| 3 slice 96² context boxes | 0.047 | 0.1 | 0.002 |
| 4a H2D + normalize | 0.171 | 0.5 | 0.007 |
| 4b **bicubic 96→224** | 0.066 | **0.2** | 0.003 |
| 5 **ViT-B/16 forward** | 22.343 | **64.6** | 0.894 |
| 6 GeM pool + D2H | 0.388 | 1.1 | 0.016 |
| 7 nodata gates | 0.071 | 0.2 | 0.003 |
| 8 head predict | 0.404 | 1.2 | 0.016 |
| 9 rasterize | 0.002 | 0.0 | 0.000 |
| **TOTAL** | **34.611** | 100 | **1.384** |

### Finding 1 (NEW, and it is free): 144 redundant `rasterio.open`s per tile

`scripts/map_region.py:630` calls `read_tile_window(zip_path, …)` **inside** the window loop, and
`src.mapping.read_tile_window` opens `/vsizip/…` on every call. Measured separately:

- `rasterio.open` of the vsizip alone: **7.99 / 7.91 / 7.95 s** (three trials).
- With **one** open held, a 4096² window read costs **1.4 s**, a backward re-read **0.00 s**, and a
  **full sequential pass over the entire 47,420² tile costs 16.6 s** (135.6 Mpx/s).

So of the 0.445 h/tile read term, **144 × 7.95 s = 0.318 h (23 % of total wall-clock) is opening the
same file 144 times**.

**Mechanism, verified rather than assumed.** The inner TIFF is `compression=None`, `tiled=False`,
`blockysize=1` — 47,420 blocks of one scanline each — stored inside a DEFLATE zip member. GDAL must
inflate the member to reach the IFD/strip-offset table, ≈2.25 GB at ≈280 MB/s ≈ 8 s, **per open**.
This is the memory note's "TIFF block-size pitfall", now quantified.

A second consequence of the same geometry: a 4096×4096 square read costs **the same** as the
full-width 4096×47,420 band (9.52 s vs 9.54 s) — **11.58× read amplification**, the square throws
away 91 % of every strip it decompresses. Hoisting the open makes this mostly moot (1.4 s/window),
but band-wise reading would take the read term to ~0.005 h/tile.

**Fix is numerically identical output** — same pixels, same order, same partials, R14 resume
untouched. Read term 0.445 → ~0.058 h/tile; total **1.384 → ~1.00 h/tile, −28 %**.

⚠ **Unmeasured on Sherlock.** The open cost is CPU+I/O and GPU-independent, so on an L40S — where the
GPU term shrinks ~2.3× but the open does not — the read should be a *larger* share, plausibly
~45–55 % of wall-clock with the GPU idle throughout. That inference needs one confirming measurement
on Sherlock before it is quoted as fact.

### Finding 2: the lead hypothesis is real but 6×, not 30×

The 96→224 bicubic upsample was hypothesised as "possibly the biggest single win, ~30× on the
attention term". Measured, same architecture at `img_size=96` (36+1 tokens vs 196+1):

- batch 96: 0.1323 → 0.0217 s/batch = **6.09×**
- batch 256, pure fp16: 725 → 4,091 img/s = **5.64×**

Directionally confirmed, magnitude one fifth of the guess. **Why:** `src.fm_embeddings._build_block`
already uses `F.scaled_dot_product_attention`, so the quadratic term is already a flash kernel and is
*not* where ViT-B/16 spends its time at 197 tokens; the win is the ~5.4× token/pixel reduction in the
patch-embed conv and the MLPs, not the attention. **The resize itself is negligible** — 0.066 s/window,
0.2 %. The cost was never the interpolation, it was the 196 tokens it produces.

This remains a **recipe change** and therefore v3 work, per 2026-08-18b.

### Finding 3: the "1.3–2× free GPU levers" are already spent

Measured on the actual Fang ViT, batch 256, RTX 5070:

| lever | img/s | vs autocast fp16 |
|---|---|---|
| autocast fp16 (shipped) | 725.0 | 1.00× |
| pure fp16 weights | 714.8 | **0.99×** |
| `channels_last` + fp16 | 717.6 | **0.99×** |
| SDPA / flash attention | — | **already in use** |
| `torch.compile` | — | untestable here (no Triton on Windows) |

Batch size is also saturated: 32 / 96 / 256 / 512 → 766 / 723 / 730 / 731 img/s. **Raising `--batch`
buys nothing** and the docstring's "larger batches better saturate an L40S/A100" is unverified on this
GPU and should not be assumed. `torch.compile` is the only untested member of this group and can only
be measured on Sherlock (Linux); at ~13 % of the card's fp16 peak there is theoretical headroom, so it
is worth one Sherlock probe, but it is a *hope*, not a measured lever.

### Corrected cost picture

| scenario | h/tile (RTX 5070) | 26-tile region | full Murray ≈4,050 tiles |
|---|---|---|---|
| as shipped | 1.384 | 36.0 h | 5,600 h |
| **+ hoist the open** (free, no recipe change) | ~1.00 | 26.0 h | 4,050 h |
| + native-96 ViT (**recipe change → v3**) | ~0.25 | 6.5 h | 1,010 h |

Scaled to Sherlock's L40S (the banked ≈0.6 GPU-h/tile, ≈2,400 GPU-h full-Murray figures): hoisting the
open alone should take full Murray to roughly **1,550 GPU-h**, and hoisting + native-96 to roughly
**450 GPU-h**. Those are extrapolations from a laptop GPU and are quoted as such.

**Ruling: the open-hoist is the only optimisation on the table for THIS rebuild** — it is provably
output-identical, touches one call site, and is worth ~28 % of wall-clock. Everything else is either
already done (SDPA, batch), a no-op (fp16, channels_last), or a recipe change that belongs with v3
(native-96, the 9× stride redundancy, distillation). Whether the hoist lands before or after the
rebuild is Brian's call — it is not a correctness dependency.

---

## 2026-08-19 — Rebuild execution plan DRAFTED; three more decisions ruled; A1's cost measured

[PLAN_Rebuild.md](PLAN_Rebuild.md) now exists: the audit's 12-step DAG turned into commands, roots,
per-step verification gates, abort conditions and a local/Sherlock split. **Nothing has been run** —
it is drafted for approval.

### Three decisions ruled (Brian, 2026-08-19)

**1. Build IN PLACE into `dataset_v2`, gated on the SHA-256 backup pass completing first.**
I opened this proposing a separate generation root and Brian pushed back correctly: `D:` *is* a full
copy — `repo\dataset_v2` is inside the 125.55 GB — so rollback exists, and at ~120 MB/s a full
78.34 GB restore is ~11–15 min. That retires the argument I led with.

What survives the challenge, and why the hash pass is the gate: **right now two independent copies
exist, so a length-preserving bit flip is detectable by comparison. After an in-place overwrite `D:`
is the only record, and its byte-integrity is still unproven.** Running the owed hash pass first
closes exactly that. Two lesser arguments are accepted as residual risk: a mid-stage abort leaves a
mixed generation in `dataset_v2` (detectable via the post-2026-08-06 sidecar `inputs` digests, hence
the plan's per-step verify gates), and rollback requires re-attaching `D:`. Brian is handling
stay-awake himself; no `powercfg` change was made.

**2. Land the `rasterio.open` hoist before step 11** (DECISIONS 2026-08-18c). Acceptance is
bit-identical `scripts/parity_check.py` against the existing reference, or it does not land.

**3. FM path only.** R27/R28 change `features/`, but the frozen recipe is embeddings-only, so no GBM
sweep and no W1 error atlas. Stage 4b still runs (`context_patches` feed the embeddings).
PENDING_REBUILD rows 2–3 stay open, annotated *"features regenerated; downstream tabular numbers not
re-derived."*

### The Sherlock split SHRANK — a consequence of 2026-08-18c's measurement

2026-08-18b sized the upload at ~17.5 GB (`context_patches` 17.0 + ckpt 0.32 + `ctx_windows` 0.19 +
labels/splits 0.01), on the assumption that step 6's embeddings run on Sherlock. **They should not.**

- The training pool is **161,005 tiles** = ~161 k ViT forwards. At the measured **730 img/s** that is
  **≈3.7 min of GPU** (≈8 min if both P32 and P96 inputs are banked). So `context_patches` (17.0 GB)
  **never moves**, and the upload drops to the head + calibration artifacts.
- The A1 embedding arm could not move cheaply in any case: `--norm a1` calls
  `src.striping.a1_stats_native_tile`, which **streams the parent Murray tile** and reads the cached
  SeamMaps — `cache/ctx_tiles/` (24 zips, 19 `_seammap_*`, 36 `_frames_*.gpkg`), none of it in the
  17.5 GB list. Locally those inputs are already on disk.
- **Only step 11 is genuinely GPU-heavy**: 26 tiles × 2 arms ≈ 31 GPU-h, ≈23 GPU-h post-hoist.

**Net: everything local except step 11.** This is not a change of mind about Sherlock — the *reason*
for Sherlock (laptop sleep during a long unattended GPU job, demonstrated 2026-08-18) applies to
step 11 and only step 11.

### A1's computational cost, MEASURED (read-only, `E-12_N36`, 81 dissolved source frames)

| component | cost |
|---|---|
| `load_frames` (SeamMap, dissolved by PRODUCT_ID) | 1.45 s |
| `a1_stats_native_tile` — once-per-tile streaming pass | **154.9 s (2.58 min)** |
| `frame_labels_on` — rasterize 81 frames onto one 4096² window | **2.67 s** ×144 |
| `a1_normalize_native` — apply per-frame (median, IQR) | **1.09 s** ×144 |

Provenance from the same run: **80/81** frames got their own statistic, fallback covers **8.1e-5** of
valid pixels, R38 clip fraction **1.9e-4** — all consistent with R08's ratified contract (the fallback
population is isolated single pixels, normalized rather than dropped).

**Per-window A1 overhead 3.756 s; per tile 0.193 h** = 0.043 h stats + **0.150 h** per-window. That is
**+14 %** on the 1.384 h baseline tile — and the distribution is the surprise: **the streaming stats
pass is only 22 % of it; 78 % is re-rasterizing the same 81 unchanging polygons onto all 144 windows.**

Two latent wins, **not** taken in this rebuild (they are a second and third driver change on top of
the hoist, and need Brian's call):
- **Cache/reuse the frame labels across windows.** A whole-tile native int32 label map is ~4.5 GB (the
  code comment says so, which is why it rasterizes per block), but rasterizing once per 4096-row band
  instead of once per window is ~12× on that term: **0.150 h → ~0.012 h, A1's overhead +14 % → +4 %**.
- **`frame_hist_native` pays the same 11.58× strip amplification** as `map_region`: it correctly opens
  once, then reads 144 4096² *squares*, each decompressing full-width strips. Full-width bands would
  take 154.9 s toward ~25 s.

**Rebuild budget from these numbers:** step 6's A1 arm ≈ 24 parent tiles × 2.58 min ≈ **1.0 h** local;
step 11's A1 arm ≈ **5 GPU-h** extra on Sherlock above the A1 renders themselves.

### Two more measurements that CLOSE levers rather than open them

- **Gate-before-embed is worth ≈0 here.** `predict_window` embeds first and masks after, so
  gate-failing cells are computed then discarded. On the profiled window `n_usable == n_valid ==
  15,876` and nonzero = 1.000 — the circum-Chryse mosaic is fully populated. Only relevant to
  full-Murray coverage at high latitude / mosaic gaps.
- **Window overlap duplicates exactly 4.51 % of embeddings** — across the 26 shipped tiles
  `n_predicted_tiles` sums to 59,436,338 against 56,873,466 unique cells (1479²). Deduplicating would
  recover ~4.3 % of GPU time but overlap is precisely what R14's `overlap_disagreement` cross-run
  detector runs on. **Bad trade; not taken.** Relatedly, a larger `--win-px` is *worse*, not better:
  edge clamping grows faster than the overlap shrinks (8192 ≈ 5.3 % duplication vs 4096's 4.51 %), so
  **4096 is already near-optimal** and should not be changed.

### Gotcha recorded

`conda run … python -c` **rejects newlines in the argument** (`NotImplementedError: Support for
scripts where arguments contain newlines not implemented`). Multi-line probes must go in a file.
Companion to the existing `--no-capture-output` note.

### 2026-08-19 addendum — the SHA-256 pass COMPLETED: `VERIFIED`, byte level

Ran `scripts/backup_artifacts.ps1 -SkipCopy -Hash` 10:30:28 → ~11:12 (**≈42 min**), exit 0.
**8/8 roots OK: 0 missing / 0 extra / 0 size mismatch / 0 hash mismatch across 11,260 files /
125.55 GB.** Verdict JSON exists this time —
`D:\HiRISE2CTX Backup\_backup_meta\backup_20260819_103028.json`, `"hashed": true`,
`"verdict": "VERIFIED"`.

**This closes the 2026-08-18 correction.** That entry said "there is **no content verification** of
this snapshot — do not read the earlier 'hashed' language as achieved", and left criterion 5 standing
at path-and-size. It now stands at **byte-for-byte**. The residual exposure it named — "a bit flip
that preserved file length" — is ruled out.

**The lid-closed diagnosis is confirmed by the re-run.** Same drive, same enclosure, same cable, same
~250 GB of reads (both sides), machine kept awake: **no Event 51, no `Ntfs` 98, no bus drop.** The
hardware speculation in the first 2026-08-18 reading was wrong and the suspend explanation was right.

**Gate 0a of [PLAN_Rebuild.md](PLAN_Rebuild.md) is CLOSED.** Remaining pre-rebuild gates: 0b (detach
`D:`), 0c (the open-hoist at bit-identical parity), 0d (fast suite green after 0c), 0e (clean tree).

Measured in passing, and it revises a number quoted earlier the same day: the drive sustains
**53.5 MB/s** read under concurrent C: load (C: 45.4 MB/s), not the ~120 MB/s assumed. A full 78.34 GB
`dataset_v2` restore is therefore **~25 min**, not ~11–15. It does not change decision 3 — 25 minutes
of recovery is still cheap — but the rollback is not as instant as the in-place argument implied.

---

## 2026-08-19b — The open-hoist LANDED and is bit-neutral; and leg 2 caught something else

Gate 0c executed. `src/mapping.py` gains `open_tile()` + a keyword-only `dataset=` on
`read_tile_window`; `scripts/map_region.py` and `scripts/striping_a1_map.py` open **once per tile,
lazily, under `try/finally`**. Seven test fakes gained `**_kw` and eight `monkeypatch.setattr` sites
gained an `open_tile` stub (`_NullSrc`).

### Speed: better than the estimate

Real driver, `E-12_N36`, `--win-px 4096 --limit-windows 3`:

| | win 1 | win 2 | win 3 |
|---|---|---|---|
| before | 33.5 s | 33.6 s | 33.2 s |
| after | 40.6 s (pays the single open) | **22.6 s** | **22.8 s** |

Steady-state **33.4 → 22.7 s/window, −32 %**. Full tile: 40.6 + 143 × 22.7 ≈ **0.91 h** against
1.34 h, i.e. **−32 %**, better than the −28 % projected. Across 52 tile-renders that is ≈**16 GPU-h**.

### The hoist is bit-neutral — proven three ways

1. **Window identity**: `read_tile_window` with and without `dataset=` returns bit-identical `data`,
   `transform` and `crs_wkt` at four offsets (0 differing pixels).
2. **`parity_check.py --rtol 0 --atol 0`**: `max|d| = 0.00e+00` on `prob_raw`, `abundance` *and*
   `prob(cal)`.
3. **Driver partials**: original-code run #2 == hoisted run #1 == hoisted run #2, **bit-identical**
   across all 9 arrays in all 3 partials.

### ⚠ NEW FINDING — `map_region.py` is not bit-reproducible across runs, and isotonic amplifies it

Leg 2 failed on the first comparison. The cause is **not** the hoist: running the **unmodified**
code twice produced *the same disagreement*.

| comparison | code | result |
|---|---|---|
| before vs before2 | **identical (original)** | **DIFFERS** |
| before2 vs after | original vs hoisted | identical |
| after vs after2 | identical (hoisted) | identical |

So run #1 of the session is the outlier; runs #2–#4 all agree. Magnitudes, per partial:

| partial | `prob_raw` | `abundance` | `prob` (calibrated) |
|---|---|---|---|
| `000000_000000` | 1.01e-04 | 6.84e-06 | — |
| `000000_004000` | 1.31e-05 | — | — |
| `000000_008000` | 1.53e-04 | 4.05e-03 | **0.1318** |

The underlying non-determinism is **~1e-4 on `prob_raw`** — ordinary fp16/cuBLAS reduction-order
variation. **The isotonic calibrator turns it into 0.13 on the shipped `prob` raster** by stepping
across a knot, exactly as `parity_check`'s own note warns ("isotonic step-amplifies the prob_raw
diff"). Cause of the run-1 divergence is **not identified** — plausibly cuBLAS algorithm selection
under different free VRAM on the 8 GB card. Observed **once in four runs**; it needs a proper
repeat-run characterisation before step 11, not a single anecdote.

### ⚠ AND the consequence is a live risk to the Sherlock array plan

**Correction to my own 2026-08-19 entry.** I recorded "window overlap duplicates exactly 4.51 % of
embeddings" from the *pre-R01* shipped sidecars. Measured directly on the **current R01 sweep**
(`E-12_N36`, 144 windows, CPU-only enumeration): **2,250,000 cells emitted, 2,187,441 unique,
62,559 duplicated = 2.78 %.** The 4.51 % figure describes the old tile-anchored sweep; **2.78 % is
the right number.**

That also contradicts `overlap_disagreement`'s docstring, which asserts "measured on the sweep this
driver uses, 900 cells over 36 windows with **0 computed twice** … within one run this returns
`(0, 0.0)` **by construction** and can never false-positive." That measurement is from a 36-window
test sweep; **on the real 144-window sweep 62,559 cells ARE computed twice.** The conclusion still
holds within a single run — a cell's value depends only on the cell — but it holds for the *other*
reason the docstring gives, not the partition claim.

**Chain the two findings and the guard becomes a trap:**
- 62,559 cells per tile are computed twice, and
- across runs the same cell can differ by ~1.5e-4, and
- `map_region.py:755` does `if n_dis and max_dis > 1e-6: raise SystemExit(... Refusing to assemble)`.

So a tile whose partials span **two runs — a Slurm pre-emption resume** — can refuse to assemble.
That is precisely the scenario DECISIONS 2026-08-18b decision 6 relies on ("R14 is what makes
pre-emption safe — resubmit and it resumes"). **Not yet observed in the wild** — it needs both a
resume boundary and the non-determinism to bite — but the mechanism is established and it is a
26-tile × 2-arm array job.

**Not fixed here, because the options are Brian's call:** (a) raise the guard threshold above the
observed noise (1e-3 still catches the stale-partial case it was built for — that one showed 63.1 %
of pixels from the wrong run); (b) force deterministic inference (`torch.use_deterministic_algorithms`,
TF32 off, pinned cuBLAS workspace) at some throughput cost; (c) deduplicate cells so windows really
do partition — which would *also* recover the 2.78 % and remove the failure mode, at the price of the
cross-run detector's signal. **This revises my earlier "overlap dedup is a bad trade" call: it is a
better trade than I said**, because the duplication is not only wasted compute, it is the thing that
makes the guard trip.

### 2026-08-19b addendum — characterised: 1 anomaly in 15 runs, hoist fully exonerated, guard stays

Brian's call was "characterise before choosing a threshold". Done: **10 further independent runs** of
the same 3 windows (`E-12_N36`, `--win-px 4096 --limit-windows 3`), plus a decisive control.

**Result: 45 of 45 pairwise comparisons bit-identical.** `prob_raw`, `prob` and `abundance` all
`max|Δ| = 0.000e+00`. **0 pairs would trip the 1e-6 assembly guard.**

**The control that settles it.** I restored `src/mapping.py` *and* `scripts/map_region.py` to their
**pristine `HEAD`** contents from git and re-ran:

| comparison | result |
|---|---|
| **pristine HEAD vs hoisted** | **BIT-IDENTICAL** |
| pristine HEAD vs `hoist_before` (the outlier) | differs, *exactly* the same deltas as every other comparison against it |

So the hoist is **fully exonerated against the committed baseline**, not merely against another run of
itself — this is stronger evidence than the acceptance criterion asked for. **Leg 2 PASSES.**

**The anomaly is real but singular: 1 run in 15.** `hoist_before` — the session's first `map_region`
invocation — disagrees with all fourteen later runs (which agree among themselves across 45+ pairs),
including with pristine HEAD code. Its deltas are identical against every counterpart, so it is one
divergent artifact set, not drifting noise.

**Cause: NOT identified, and I am not going to invent one.** Candidates that fit but are unproven: a
transient cuBLAS algorithm choice under different free VRAM, or an uncorrected bit error (this is a
consumer laptop GPU — **no ECC**). Ruled out: the hoist, the `mapping.py` refactor, code drift, and
run-order determinism.

**Ruling: leave the guard at 1e-6.** Reasoning, and it inverts the framing I opened with:
- The guard is **fail-safe**. It *refuses to assemble*; it does not ship a wrong raster. The cost of a
  trip is re-running one tile with `--force` (~0.9 GPU-h), not a corrupted deliverable.
- At ~1 in 15 runs — and possibly a true one-off — the expected cost across step 11 is small.
- If the mechanism really is an uncorrected memory error, then a tripped guard is **a real error
  signal**, and raising the threshold to 1e-3 would suppress exactly the thing worth knowing about.
- Weakening a detector that was built after a 63.1 %-wrong raster, on the strength of one unexplained
  event, is the wrong trade.

**What to do instead — document the signature.** If step 11 ever dies with
`N cells were written twice with DIFFERENT values (max |Δ| = ...)`: a `max|Δ|` of order **1e-4 or
below** is this phenomenon → re-run that tile with `--force`. A `max|Δ|` of order **0.1–1** is the
stale-partial case R14 was built for → **investigate, do not force.** The 2.78 % duplicated cells are
what makes the collision possible at all.

**Residual risk accepted, and it is bounded:** a resumed tile can refuse assembly; recovery is one
`--force` re-render. **§4a's "do not start step 11" hold is LIFTED.**

---

## 2026-08-20 — REBUILD STARTED. DAG step 1 (Stage 1) COMPLETE and verified

First producer run of the batched v2 rebuild, from code provenance point **`3d74530`**, **in place**
into `cache_v2/reprojected_detections/`. Pre-flight: gates 0a–0e closed, `D:` confirmed detached,
tree clean and synced, target covered by the byte-level `VERIFIED` snapshot.

`conda run -n geospatial python -u scripts/run_stage1.py --config config_v2.yaml --all`
→ **39 / 39 reprojected, 0 failed.** ~9 min wall.

### Verification gate — all checks pass

| check | result |
|---|---|
| images processed | **39/39, 0 failed** (the failure mode here is per-image `FAILED` + `None`, not a nonzero exit, so the count is the real gate) |
| SP1 correction | **32 `sp1_corrected_from_pds_label` + 7 `trusted_prj` = 39**, exactly matching the manifest's `PrjSP1Corrected` tally `{True: 32, False: 7}` |
| **per-image local radius** | **7 distinct radii, 3384416.50 – 3393833.26 m, none equal to the standard 3396190 m** |
| R23 metadata contract | `source_integrity` **39/39**, `null_geometry_basis` **39/39** |
| byte-truncated sources | 3 flagged, all reproducing DECISIONS 2026-08-06o |

**The #1 gotcha is demonstrably handled.** The seven radii come from each image's own PDS `.LBL`
`A_AXIS_RADIUS`, and one of them is **3393833.26 m** — the exact value CLAUDE.md cites as the
canonical example. Nothing hardcoded the standard sphere.

**The R23 truncation numbers reproduce exactly**, which is the strongest available evidence the
re-run is faithful:

| ObsId | raw rows | kept | dropped | realised floor |
|---|---|---|---|---|
| ESP_017355_2260 | 1,105,447 | **359,933** | 745,514 (67.44 %) | 0.617257 |
| ESP_046803_2325 | 658,290 | 367,140 | 291,150 (44.23 %) | 0.473417 |
| ESP_068483_2280 | 1,057,153 | 727,160 | 329,993 (31.22 %) | 0.406698 |

`ESP_017355_2260`'s kept count is **359,933**, matching the audit table's "records that fit 359,933 /
pipeline kept 359,933 / Δ **0**" to the unit. Total kept polygons across 39 images: **6,278,986**.

### Two corrections to PLAN_Rebuild §3's step-1 verify row (now fixed in the plan)

- **`realised_label_basis` is a STAGE 4 field**, not Stage 1 — it is emitted by
  `src/labeling.py:1070` via `_describe_realised_label_basis`. Stage 1's half of the mixed-floor
  contract is `source_integrity` + `null_geometry_basis` only. The plan listed all three under step 1.
- **The "residual HiRISE↔CTX offset O(200 m), not km" check does not belong to step 1.** Stage 1 only
  reprojects; that residual is a **co-registration** quantity and is checked at step 3. Moved.

### Timing: 3× the estimate, and the reason is known

**~9 min against the ~2.6 min** reconstructed from the original build's mtimes. Not a problem, but the
estimate's basis was wrong: R23 added a byte-integrity scan of every source `.shp` plus a score-rank
analysis of the null-geometry population, work the original run never did. **Expect the §3a mtime
estimates to under-read wherever a stage gained provenance work in the fixing tranche** — Stage 2
(R74 mask + digests), Stage 3 (R65 components + `shift_id`) and Stage 4 (R29/R80) are all in that
category. The 1.5–2 day total still holds; the per-stage figures are floors, not forecasts.

### 2026-08-20b — DAG step 2 (Stage 2) COMPLETE and verified

**39 / 39 windows + coverage masks, 0 failed, 11 min**, in place into `cache_v2/ctx_windows/`.
No network: all 39 manifest rows resolved to a cached Murray zip.

| check | result |
|---|---|
| images | **39/39, 0 failed** |
| `ctx_window_sha256` | **39/39 present** |
| coverage-mask `version` | **2 on all 39** — the definitive marker that R74 is in effect (`version: 1` anywhere = the fix did not take) |
| mask method | `decimated_nonzero_with_interior_shadow_fill` ×39 |
| `max_interior_hole_px` | 16 on all 39, config-driven and recorded per artifact |
| R74 shadow-hole fill | **6,514 px re-marked across 39/39 images** — every image had at least one |
| `footprint_source` | **`polygon_bbox` ×39** |
| `hirise_coverage_fraction` | min 0.4666, median 0.5450, max 0.6313 |

**R67 is confirmed on live output.** The audit closed R67 at LOW on the argument that the nominal
fallback branch is "provably unreachable for v2 (39/39 take `polygon_bbox`)". The rebuild reproduces
exactly that: 39/39 `polygon_bbox`, 0 fallbacks. The R67 footprint defects therefore remain latent
and touch no v2 artifact, as the audit predicted.

### ⚠ Correction to the step-1 timing generalisation

After step 1 ran 3.5× **slower** than its mtime estimate I wrote that the §3a figures are "floors, not
forecasts" wherever a stage gained provenance work. **Step 2 ran 5× FASTER — 11 min against 58 min.**
The generalisation was wrong. The correct reading is narrower: **mtime-derived estimates are
unreliable in both directions, and for unrelated reasons.** Step 1 gained R23's byte-integrity scan
and score-rank analysis (slower); step 2's original 58 min *included downloading the Murray tile
zips*, which are now cached (faster). Treat §3a as an order-of-magnitude sanity check, not a schedule.

Net so far: steps 1–2 in **~20 min** against a combined ~61 min estimate.

### The plan's step-2 command was WRONG, and it would have failed immediately

`scripts/run_stage2.py` takes a **single positional ObsId and has no `--all`** — unlike stages 1, 3, 4,
4b and 5. `run_stage2.py --config config_v2.yaml` dies on a missing argument. The DAG step is a
**loop**, now recorded as such in the plan, with a runner that continues past a failure and prints a
tally (Stage 1 established that per-image failures do not produce a nonzero exit, so the count is the
only real gate).

**Also worth recording, because it cost three attempts:** the manifest stores **zero-padded compass**
Murray names (`W008_N32`, `E020_S64`) while the tile cache uses **signed unpadded** ones (`E-8_N32`,
`E20_N-64`). Two successive bad transforms of mine reported "39 missing" and then "18 missing" tiles —
both artifacts of the naming convention, not real gaps. A false reading here implies a ~30 GB download
that is not actually required. The correct transform is `[EW]ddd_[NS]dd` → signed ints →
`E{lon}_N{lat}`, unpadded. (Companion to the `murray_ctx_conventions` memory note.)

### 2026-08-20c — DAG step 3 (Stage 3) COMPLETE and verified; the CRS gate passes

**39 / 39 solved, 0 skipped, ~40 s.** Method tally: **`block_median` 38, `single_window_fallback` 1**
(`ESP_046803_2325`, genuinely bland — 3 of 44 blocks cleared the 0.5 peak floor, below the 6-block
minimum).

### The CRS sanity gate — the abort condition — PASSES

|shift| **min 79.9 m, median 194.7 m, max 327.3 m.** CLAUDE.md's standing rule is that after correct
per-image local-radius reprojection the residual HiRISE↔CTX offset is **O(200 m), not km**, and that a
kilometre-scale result means the CRS handling is wrong and must fail loudly. The median is 194.7 m.
This is the strongest end-to-end confirmation available that step 1's seven distinct per-image radii
were applied correctly — a hardcoded 3396190 m sphere would have shown up here as a km-scale residual.

`dy` dominates `dx` on most images (e.g. +186, +234, +249, +263 m), consistent with the known
systematic north–south offset that the 2026-06-10c sign-error fix addressed.

### R65 verified — components, not a composite

⚠ **My first verification pass reported these "MISSING 38/39". That was my error, not the
pipeline's** — I searched for top-level keys; they are nested under `block_field`, and the one
"missing" image is the fallback, which legitimately has no confident-block population.

`peak_correlation_kind` labels the semantics rather than conflating them:
- **38 × `conditional_median_of_confident_block_peaks`**
- **1 × `post_shift_pearson_of_applied_shift`** (the fallback, peak 0.4396)

That labelling is precisely R65's remedy. The register's own proposed fix — an unconditional median —
was refuted before implementation because it reads ~0.75 on a perfectly registered image with 25 %
junk blocks. `block_field` now carries `quality_version: 2`, `n_blocks`, `n_confident_blocks`,
`confident_fraction`, `median_block_peak`, `median_block_peak_is_conditional`, the full
`all_block_peak` five-number summary, `block_mad_px` and `block_mad_px_is_conditional`.

The `peak_correlation` distribution (min 0.440) dips below the 0.5 block floor only for the fallback
image, whose peak is a *different quantity* — which is exactly the confusion R65 existed to prevent.

**Provenance chain intact: 39/39 sidecars bind the Stage-2 `ctx_window_sha256`**, plus
`hirise_mask_sha256` and the full `coverage_mask` block (`version: 2`). `shift_id` present 39/39. So
the R74 → Stage 3 dependency the audit's DAG asserts is materialised in the artifacts.

**Minor, not a finding:** the fallback image's `block_field` omits `confident_fraction` and
`quality_version` although both are computable there (3/44 = 0.068). The raw counts are present, so
nothing is lost; it is an inconsistency in derived-field emission, worth tidying whenever Stage 3 is
next touched.

### `coregistration.enabled: false` in `config_v2.yaml` is a DEAD KEY

`run_stage3.py` reads only `fft_window_px`, `block_px`, `block_peak_min` and `min_confident_blocks`
from that block. **Nothing anywhere reads `coregistration.enabled`** — the only `"enabled"` consumers
in `src/` and `scripts/` are the `features` block. Stage 3 always computes; Stage 4 applies the shift
by default (`apply_coreg = not args.no_coreg_shift`).

This is a documentation hazard rather than a bug: read literally, the config says v2 has no
co-registration, which would badly misstate the label basis given the y-shift sign-error history.
**Deliberately NOT changed mid-rebuild** — editing a config key now would move `cfg.hash` and
invalidate the Stage-2 sidecars just written against it. Fix after step 12, or delete the key.

Cumulative: steps 1–3 in **~21 min**.

### 2026-08-20d — DAG step 4 (Stage 4) COMPLETE: rich prevalence lands on 0.3733, as predicted

**38 / 38 solved, 0 skipped, ~7 min.** 3,581,340 eligible tiles across all four scales.
`EXCLUDED_FROM_SWEEP` = `{ESP_057469_2215, ESP_046803_2325}` — the first is a v1 row absent from the
v2 manifest, the second is the featureless image, so 39 manifest rows → **38 labelled**.

### THE headline number, and it matched the prediction

| quantity | previous | predicted | **rebuilt** |
|---|---|---|---|
| S=32 pool | 161,005 | ~+3,236 (+1.97 %) | **164,644 (+3,639, +2.26 %)** |
| **rich prevalence (fa > 1e-2)** | 0.3598 | **≈0.3733** | **0.373272** |
| n rich | — | — | 61,457 |
| pool max `fa` | 0.293242 | — | **0.293242** |

**The prevalence matched the R74 counterfactual to four decimal places.** That is a stronger result
than it had any right to be: PENDING_REBUILD explicitly warned "these are not guaranteed final-rebuild
counts because Stage 3 and R23/R29 can change alignment, eligibility, and targets", and all three
*did* change under it. The honest reading is that those changes moved **eligibility** — the pool grew
2.26 % against a predicted 1.97 % — but left the rich **fraction** essentially untouched. The
counterfactual was a better estimator of prevalence than of pool size.

**`pool max fa == 0.293242` reproduces the R84 banked invariant exactly.** Worth noting because R84
pinned `calibration.npz` `t2_y` max == pool max `fa` == 0.293242; the equality survives the rebuild,
so the calibration join will still be able to assert it.

Per-image rich prevalence spans **0.0015** (`ESP_047976_2020`) to **0.9786** (`ESP_054622_2240`),
median 0.2314 — the extreme between-image heterogeneity that motivates LOIO in the first place.

### Provenance gates: all 38/38

`coreg_mask_shift` (R29) **38/38** · `realised_size_basis.realised_physical_min_size_m` (R80)
**38/38** · `realised_label_basis` **38/38** (the field I wrongly attributed to Stage 1 on 2026-08-20).

**R80's realised-floor measurement reproduces exactly.** Seven distinct realised physical floors,
**0.9930 – 1.3664 m**, against the configured `min_size_m: 1.4105`:

| realised floor | images |
|---|---|
| 0.9930 m | 1 |
| 1.1826 m | 12 |
| 1.2315 m | 15 |
| 1.2741 m | 5 |
| 1.3107 m | 2 |
| 1.3414 m | 2 |
| 1.3664 m | 1 |

R80 recorded the range as "0.993–1.367 m, not the configured 1.4105 m" — reproduced to the digit.
**And the seven floors correspond one-to-one with step 1's seven distinct per-image local radii**,
i.e. the seven PDS centre-latitude bands (20–50°). That is the expected mechanism: the floor is
applied in the reprojected CTX frame, so it scales with the local radius. Two independent stages
agreeing on "seven" is a good sign the CRS chain is coherent end to end.

**Confirmed live, the `tile_size_m` trap:** S=32 carries **three** distinct `tile_size_m` floats
(159.99918352980166 / …70 / …74) from per-image FP noise. `tile_size_px == 32` is the only safe pool
selector, exactly as the memory note says.

### Consequence for step 4c

The R83/R84 mixture figures (78.3914/21.6086 % of **161,005** tiles, 27 distinct floors, tile-weighted
mean 3.3687 m²) were measured on the **old** pool. The pool is now **164,644**, so
`measure_size_floor.py` will describe a different population and those banked percentages should be
expected to move. That is the intended behaviour and is exactly why the plan puts 4c after Stage 4.

**Splits unchanged:** 38 images survive, so `loio_nfold: 38` and `within_image_4fold: 152` stand for
step 5. Cumulative: steps 1–4 in **~28 min**.

### 2026-08-20e — Brian's v3 ruling: ONE HiRISE resolution, therefore ONE size threshold

**Brian, 2026-08-20:** the next dataset will use images of **all the same resolution**, so there is a
single minimum size threshold. **The mixed-floor apparatus built for v2 is therefore TEMPORARY and
must be treated as such.**

This upgrades the 2026-08-18b size-floor decision from "deferred to v3 (option C, **tentative**)" to a
**stated v3 design requirement with a concrete mechanism**. It is also the cheapest possible
resolution of the problem: the rejected options all tried to *repair* a mixture after the fact —
option A (common 2.66 m floor over 38 images) and option B (fine-only at ~1.2 m, pool 161,005 →
34,791) both destroy data, and R83 measured that calibration provably cannot absorb the change (it is
a **re-ranking**, within-image Spearman 0.60–0.98, fine rich prevalence halving 0.326 → 0.164).
Choosing uniform-resolution inputs means the mixture is never created, so none of that cost is paid.

**What this means for the code, concretely.** `src/size_floor.py`, `scripts/measure_size_floor.py`,
the `SIZE_FLOOR_*` raster tags and `SizeFloorBasis` are **scaffolding with a known end date**. They
exist to make an unavoidable v2 defect *legible*, not to fix it. Do not invest further in them, and do
not let a future session treat the mixture machinery as permanent architecture. When v3 lands with one
resolution, `n_distinct_floors` collapses toward 1 and most of this can retire — the tags should stay
(a product should always state which boulders it counts) but the mixture bookkeeping should not.

### ⚠ One wrinkle: uniform HiRISE resolution removes only ONE of the two floor-variation sources

Worth stating now, while it is cheap to design around. Step 4 measured **two independent** sources:

1. **The natural detector floor** — the smallest rock BoulderNet found, which varies with HiRISE
   m/px (fine cohort 1.028–1.213 m diameter, coarse 1.943–2.664 m). **Uniform resolution fixes this.**
2. **The global filter's realised physical value** — `min_size_m: 1.4105` is applied **in the
   reprojected CTX frame**, so its realised physical size varies with the equirectangular projection's
   latitude-dependent scale. Step 4 measured **seven distinct realised floors, 0.9930–1.3664 m**, and
   those seven map **one-to-one onto the seven per-image local radii / PDS centre-latitude bands
   (20–50°)** — not onto the two resolution classes. **Uniform resolution does NOT fix this.**

So "one minimum size threshold" needs both: uniform input resolution *and* the size filter applied in
a frame where 1.4105 m means the same thing everywhere — the image's own native/local frame, or with
an explicit cos(lat) correction — rather than in the shared reprojected frame. Otherwise v3 inherits a
smaller but still real spread (~1.4× in diameter across a 20–50° latitude span) from source 2 alone.

Not a v2 action. Recorded as a v3 design input while the measurement that revealed it is fresh.

### A plan defect this step caught

PLAN_Rebuild §3 step 4c had `--detections` pointing at the **raw BoulderNet root**
(`hirise_40_vClaire`). It must be the **Stage-1 reprojected gpkgs**
(`cache_v2/reprojected_detections`, which is the script's own default) — the natural floor has to be
measured in the same projected frame the labels were built in. Corrected in the plan.

### 2026-08-20f — DAG step 4c: size-floor basis BANKED, and a defect found in it

`models/deployable/size_floor_basis.json` now exists for the first time (only `--dry-run` had ever
run). Measured in **33 s**, not the ~6–7 min its docstring predicts — it reads the Stage-1 gpkgs, not
the raw shapefiles.

| quantity | R84 (old 161,005 pool) | **rebuilt (164,644)** |
|---|---|---|
| pool | 161,005 / 38 img | **164,644 / 38 img** |
| effective floor range | — | **1.5626 – 5.5719 m²** |
| tile-weighted mean | 3.3687 m² | **3.3914 m²** |
| tile share 0.5 / 0.25 m/px | 78.3914 / 21.6086 % | **78.73 / 21.27 %** |
| image share | 68.4 / 31.6 % | **68.42 / 31.58 %** (unchanged — image counts didn't move) |
| `n_distinct_floors` | 27 | **27** |

The shares moved slightly, exactly as predicted for a pool that grew by 3,639 tiles. `min_size_m`
1.4105 and `version: v2_mixed_floor_1` recorded. Product sentence emitted:

> *Target = area share of boulders above a per-image detection floor of 1.563–5.572 m² equivalent-circle
> area (1.41–2.66 m diameter); the calibration pool is a mixture of 27 floors over 38 HiRISE images
> (78.7 % at 0.5 m/px, 21.3 % at 0.25 m/px, by pool tile). NOT size-independent rock abundance.*

### ⚠ FINDING: "27 distinct floors" is a FLOATING-POINT ARTIFACT. The real count is 20.

Reconciling `n_distinct_floors` against the `per_image` records did not add up, so I counted at
several precisions:

| rounding | distinct effective floors |
|---|---|
| full float | **27** |
| 12 dp | 27 |
| 9 dp | 25 |
| **6 dp** | **20** |
| 4 dp | 20 |
| 3 dp | 20 |

Five groups of images share a floor to 6 dp but differ at the **1e-9 – 1e-12 m²** level:

| floor (m²) | images | full-precision values |
|---|---|---|
| 3.320527 | 4 | `…103718`, `…103946`, `…109165`, `…111836` |
| 3.600986 | 3 | `…883565`, `…883813`, `…887190` |
| 3.336990 | 2 | `…420708`, `…426182` |
| 4.099416 | 2 | `…180130`, `…183923` |
| 1.562558 | 12 | genuinely identical (all clamped to the global filter constant) |

Differences of ~1e-9 m² are **square nanometres**. They cannot be distinct physical detection floors;
they are ULP noise from each image reprojecting through **its own local-radius CRS**, so identical
native polygon areas land at very slightly different projected areas. The mechanism is the same
per-image-radius machinery step 1 verified — this is its harmless downstream residue, miscounted.

**The correct count is 20**: 12 fine images clamped to one exact global-filter value + **19** distinct
natural floors among the 26 coarse images. **20 is stable from 3 dp to 6 dp**, so it is not an
artifact of a chosen tolerance — unlike 27, which only survives at ≥12 dp.

**Why it matters despite being small.** No *numeric* result is affected — mean, shares, min and max
are computed from values, not counts. But "27" is (a) quoted in R84's DECISIONS entry, and (b)
**stamped onto every raster both map drivers write**, via the product sentence above. A product that
states "a mixture of 27 floors" when there are 20 is asserting a false fact about itself, and that is
precisely the class of number that ends up in a paper.

**This is an inherited error, not a regression** — R84 recorded 27 from the old pool by the same
counting. The rebuild reproduced it faithfully.

**Not patched unilaterally** — it is a `src/size_floor.py` change mid-rebuild. Nothing downstream has
consumed the basis yet (no map has run), and re-measuring costs 33 s, so the fix is cheap whenever
Brian rules. Fix = round to a physical tolerance before `len(set(...))`.

### 2026-08-20g — the floor-count defect is FIXED (Brian: fix now); basis re-banked at 20

`src/size_floor.py`: new `FLOOR_EQUALITY_TOL_M2 = 1e-6`, applied by quantising before
`np.unique` in `from_records`. Version bumped **`v2_mixed_floor_1` → `v2_mixed_floor_2`**, so
`SizeFloorBasis.load` refuses a stale basis rather than silently serving one with the old count.

**Re-measured in 18 s. `n_distinct_floors` 27 → 20. Every other number is bit-unchanged** — pool
164,644, range 1.5626–5.5719 m², tile-weighted mean 3.3914 m², shares 78.73/21.27 by tile and
68.42/31.58 by image. As predicted: the defect was in a *count*, never in a value.

Product sentence now reads "**a mixture of 20 floors over 38 HiRISE images**".

**Why 1e-6 m² and why it is not a tuned knob.** One square millimetre, against detection floors of
1.6–5.6 m² — it cannot merge two genuinely different floors. And the count is **20 at every tolerance
from 1e-3 to 1e-6**, only reaching 27 below ~1e-12. A number stamped on every output raster must not
depend on ULP noise, and this one now provably does not.

**Two tests added** (`tests/test_size_floor.py`), because a tolerance with no test is a tolerance that
drifts: one builds three images whose floors differ by ≤8.1e-9 m² plus a real neighbour and asserts
**2** floors, pinning `FLOOR_EQUALITY_TOL_M2 == 1e-6`; the other asserts a 1e-5 m² difference — ten
times the tolerance — is still counted as two, so the fix cannot be widened into merging real floors.

**Fast suite: 802 passed, 1 skipped, 21 deselected** (was 800; +2 new).

### 2026-08-20h — SEQUENCING ERROR: step 4b was skipped; Stage 5 aborted and re-ordered

**My error, not a decision.** The audit DAG and PLAN_Rebuild §3 both order **4 → 4b → 5**. After
step 4 I wrote a status summary that listed "4 → 4c → 5", omitting 4b, then followed my own summary
instead of the plan on the next two turns. Brian caught it.

**What ran wrongly.** `run_stage5.py --all` started against **2026-06-11 features** — the stale
pre-R27/R28 generation — while the labels underneath had just been rebuilt. Stopped mid-flight.
State when stopped:

| artifact | state |
|---|---|
| `dataset_v2/features/` | 2026-06-11, stale (untouched — 4b never ran) |
| `splits/loio_nfold.json` | rewritten 14:55 today |
| `splits/within_image_4fold.json` | 2026-06-11, not reached |
| `packaged/` | **42 of 1603 files** rewritten from new labels × stale features |

`packaged/` was therefore left genuinely mixed-generation — the exact hazard the in-place decision
accepted (DECISIONS 2026-08-19, ground rule §1.2), realised by a sequencing slip rather than a crash.

**Recovery is clean and needs no deletion.** Neither `run_stage4b.py` nor `run_stage5.py` skips
existing outputs (4b's `exists()` checks are Stage-4 *input* readiness; Stage 5's only "skip" is
`--no-package`). So 4b regenerates every feature + context patch, and a full Stage 5 re-run
overwrites all 42 mixed files. Nothing else in `splits/` is at risk — `loio_nfold_ctx_illum.json`
and `loio_nfold_nbr_s5.json` belong to other stages and Stage 5 does not write them.

### ⚠ The real finding: DAG order is enforced NOWHERE in code

`loaders.verify_package_freshness` is a **read-side** check, invoked from `load_fold`. Stage 5 will
package stale features without complaint; the mismatch only surfaces later, when something tries to
*train* on the package. So:

- nothing stopped Stage 5 from consuming a feature generation three months older than its labels;
- the only thing that caught it was a human reading the plan.

That is a gap worth closing, and not only for 4b: the same class of error is available at every
remaining step — embedding before Stage 5 completes, mapping before calibration is banked, rendering
with a `size_floor_basis.json` older than Stage 4. Each producer already *records* its input digests;
none *checks* them at write time.

Cheapest sufficient fix (deferred, not done mid-rebuild): give each producer a write-time precondition
that its declared inputs' recorded digests match what is on disk — the read-side check already knows
how to compute this, it simply runs too late. Recorded as a post-step-12 item.

**Process correction for the rest of this rebuild:** read the step's row in PLAN_Rebuild §3 before
launching it, not the running summary in conversation. The summary is a convenience; the table is the
runbook.

### 2026-08-20i — DAG step 4b COMPLETE and verified; R27 and R28 both land

**38 / 38 in 957 s (~16 min).** 3,581,340 feature rows — matching Stage 4's eligible-tile total
exactly. Context patches regenerated: **S=32 3.67 GB + S=64 14.67 GB = 18.3 GB** (was 17.0; grew with
the pool). Byte arithmetic checks out — 1024 B × 3,581,340 = 3.67 GB, 4096 B × 3,581,340 = 14.67 GB —
so the equal per-scale counts are correct, not a reporting bug: every feature row carries both a
32-px and a 64-px patch.

**R27 — the out-of-range sentinel is gone.**

| | before | after |
|---|---|---|
| `lacunarity_shadow_b{2,4}` exactly `0.0` (S≥32) | 42,015 (21.2 %) | **0** |
| NaN | — | 42,032 (20.6 %) |
| values in the impossible interval (0, 1) | present via Stage-6a averaging | **0** |
| smallest genuine value | 1.0 | **1.0000** |

All **42,032** shadow-free rows are NaN — 100 % correspondence with `shadow_fraction == 0`, exactly
the population R27 identified.

**R28 — `edge_density` no longer tracks per-image radiometry.**

| | before | after |
|---|---|---|
| per-image Spearman(`edge_density`, `intensity_std`) | **0.965** | **0.2656** (p = 0.107, n.s.) |
| per-image `edge_density` spread | **12.2×** | **1.91×** |
| `ESP_068402_2240` share of S=64 tiles with zero edges | **33.8 %** | **1.9 %** |

The coupling is broken rather than merely reduced: at n=38 the residual correlation is not
significant. Worst-case zero-edge share is now 26.2 % (`ESP_069669_2220`), down from 33.8 %.

These close the *feature* half of PENDING_REBUILD rows 2–3. Per the FM-path-only ruling (2026-08-19)
the downstream tabular numbers — Stage 6a neighbour features, the GBM sweep, the W1 error atlas — are
**not** re-derived, so those rows stay open with that annotation.

### 2026-08-20j — DAG step 5 (Stage 5) COMPLETE and verified, re-run in the correct order

Re-run after 4b landed, so it packaged **fresh labels × fresh features** — what should have happened
the first time (2026-08-20h).

| scheme | folds | files | refreshed |
|---|---|---|---|
| `loio_nfold` | **38** | 230 | **230/230** |
| `within_image_4fold` | **152** | 914 | **914/914** |

`within_image_4fold` `test_rows_sum = 3,581,340` — exactly the feature-row total, so every tile
appears in **exactly one** test fold. `train_rows_sum` 10,744,020 = 3× that, as a 4-fold scheme
requires. The 42 mixed-generation files the aborted run left in `loio_nfold` are overwritten; no
manual deletion was needed.

**R04 freshness gate PASSES on both schemes.** `load_fold("loio_nfold", 0, scale_idx=2)` returns
X_train **(150394, 52)** + X_test **(14250, 52)** = **164,644** — the S=32 pool exactly.
Package metadata binds `split_hash`, `config_hash 9bce49b6214f`, `obs_to_int` (38 images) and
`source_digests` v1 with **per-obs `labels_sha256` + `labels_inputs`** — the R74 chain, so a content
change at fixed cohort is detectable.

### ⚠ Two stale variant packages now pass the gate with a WARNING, not an error

`packaged/loio_nfold_ctx_illum` (2026-05-30) and `packaged/loio_nfold_nbr_s5` (2026-06-10) are Stage-6a
variants that `run_stage5.py --all` does not write, so they still describe the **pre-rebuild** labels.
`verify_package_freshness` **passes** them, emitting:

> *"packaged/… predates source-digest provenance (R04/R74): its label and feature contents cannot be
> verified, so a pre-R74 label generation would be undetectable here. Re-package it before quoting
> numbers from it."*

That is by design and the warning is honest — check 3 cannot run without recorded digests, and checks
1–2 pass because their cohort and split hash are internally consistent. But the practical consequence
is real: **two packages on disk now silently describe the old label basis and will not raise.** Under
the FM-path-only ruling they are out of scope to rebuild, so they should be **deleted or renamed
`.stale`** before step 12, rather than left to be found by a future `load_fold`. Recorded as a step-12
item, not actioned now.

Cumulative compute: steps 1–5 ≈ **50 min**.

### 2026-08-20k — ⚠ NEAR MISS: step 6 was a 4-second silent no-op. Resume made safe.

`scripts/probes/_w2_fang_embed.py --tile-px 32 --norm none` **exited 0 in 4 s having recomputed
nothing** — "cached, skipping" for all 38 images. Its resume was
`if npz.exists() and other.exists(): skip`: existence only, no content comparison, no `--force`.

**Had the exit code been trusted, every downstream artifact would have been built on 2026-06-12
embeddings.** Quantified against the fresh labels:

| | |
|---|---|
| cached embedding tiles | **161,005** (the pre-rebuild pool) |
| fresh label tiles (S=32) | **164,644** |
| images with a mismatched `(ti, tj)` set | **38 of 38** |
| new tiles with **no** cached embedding | **7,390** |

It is not a clean subset either — tiles moved **both** ways. `ESP_017355_2260`: 13,457 cached vs
14,250 labelled, overlap only 13,103, so **354 cached tiles no longer exist**. A size-only check would
have missed that class entirely. Step 7's LOIO join would have silently dropped or misaligned
thousands of tiles per image.

**Two aggravating facts.**

1. **The audit asserts this is already fixed.** Its Modeling/calibration gate row reads
   "`fm-embeddings-3`/`fm-embeddings-4` are fixed so arm/suffix, store provenance, **existence-only
   resume**, persisted model hash, and nuisance-basis consistency are enforced." That is **false for
   the script PLAN_Rebuild step 6 actually invokes.** Whatever was fixed, it was not this path. The
   audit's gate table should not be read as a guarantee about probe-tier drivers.
2. **The npz records no provenance whatsoever** — keys are exactly
   `['ti','tj','valid','cls','mean','gem']`. No config hash, no source digest, no model hash, no arm.
   A stale store and a fresh one are **indistinguishable on disk**, and the arm is carried only by the
   parent directory name — the identical weakness R07 called out for heads.

**Fix (Brian: "add a staleness check that refuses to skip").** New `_cache_is_stale(obs_id, keys)`
compares the cached `(ti, tj)` set against the current labels and skips **only on an exact match**,
reporting how many tiles are new and how many are gone. `--force` added as an escape hatch, but the
check — not the flag — is what makes the resume safe. The key set is the only provenance these stores
carry, so comparing it is the strongest available check short of adding real provenance.

**Five tests** (`tests/test_fang_embed_resume.py`): absent store recomputes; exact match still skips;
a grown pool recomputes; **a same-size but shifted key set recomputes** (the case a count check
misses); a half-written store recomputes.

**Follow-up, not done now:** the npz should carry the label-generation digest and the arm, so
staleness is detectable without reconstructing the key set. Same family as the write-time DAG check
logged in 2026-08-20h. Post-step-12.

**Process note.** This is the second silent no-op this rebuild has produced (after the 4b skip), and
both were caught by reading output rather than exit codes. `--all`-style drivers here fail *quietly*
by design: Stage 1 prints `FAILED` per image and exits 0; Stage 5 used to swallow errors (R04); this
skips. **Exit 0 means nothing in this pipeline. Read the tally.**

**Baseline arm re-run with the check active: `reused 0 cached / recomputed 38 (38 of them because the
cached key set was stale)`, 466 s.** Verified after: **164,644 embeddings vs 164,644 labels, 0 of 38
images mismatched, 0 tiles missing.** Ground rule §1.4 holds — all **252** files across the seven
`fang_embeddings_f*` stores are byte-identical in mtime and size to the pre-run snapshot.

The per-image output vindicates comparing the key **set** rather than a count. `ESP_069669_2220`:
cached 4,428 against 4,430 labels — a net **+2** — but **159 new tiles missing and 157 gone**, so 316
tiles actually changed. A count check would have seen +2 and plausibly passed it.

*Cosmetic, not fixed:* the per-image progress line prints `192-valid=` regardless of `--tile-px`, so
at S=32 it reports "192" where the context box is 96. Harmless but misleading in a log; tidy whenever
the script is next touched.

### 2026-08-20l — DAG step 6 COMPLETE: both arms rebuilt and verified

| arm | runtime | result |
|---|---|---|
| baseline (`--norm none`) | 466 s | 38/38 recomputed, all stale |
| A1 (`--norm a1 --out-suffix _a1`) | 4,281 s (71 min) | 38/38 recomputed, all stale |

**Both stores verified against the fresh labels: 164,644 embeddings vs 164,644 labels, 0 of 38
images mismatched, 0 tiles missing.** `192-valid = 100.0 %` on every image, so no context box spilled
its Stage-2 window.

**Ground rule §1.4 holds after both arms** — all **252** files across the seven `fang_embeddings_f*`
stores byte-identical in mtime and size to the pre-run snapshot. The hard-aborted F programme's
artifacts are untouched, demonstrated rather than asserted.

**A1 runtime reconciles with the direct measurement.** 38 images span **19** distinct parent Murray
tiles; `_A1_TILE_CACHE` streams each once. A1 overhead = 4,281 − 466 = 3,815 s = **201 s per parent
tile**, against the **155 s** measured directly on `E-12_N36` (2026-08-19), the remainder being
per-window `frame_labels_on` + `a1_normalize_native`. R07's per-frame native statistic is doing the
work it is supposed to. Sampled provenance: `E0_N40` 47/47 frames with their own statistic, fallback
0.0083 % of valid px; `E20_N-64` 21/22 frames, fallback 0.0175 % — both consistent with R08's ratified
contract (the fallback population is tiny and normalised, never dropped).

⚠ **My own capture error, recorded so it is not repeated.** I launched both arms piped through
`| tail -30`, so the task log retained only the last 30 lines. That discarded 17 of the 19 A1
streaming messages and led me to briefly read "only 2 parent tiles streamed" as a finding. It was not
— the arithmetic above accounts for all 19. **Do not pipe a long background run through `tail`;** the
pipe truncates the record, and on a rebuild the log *is* the evidence.

Cumulative: steps 1–6 ≈ **2.4 h** of compute.

### 2026-08-21 — DAG step 7 baseline arm COMPLETE; headline numbers re-derived

**First attempt HARD-FAILED, correctly:** `AssertionError: join loss vs T1: 164644 -> 157254` — the
7,390 new tiles exactly. `verdict()` joins fresh predictions against the Tier-1 reference, which for
`fa_gt_1e-2` resolves via `SCALE_CONFIG` to two **2026-06-12 LightGBM sweep artifacts**
(`models/lightgbm_classification/2d046f48c722f0a5/scale_S32_tfa_gt_1e-2/predictions.parquet`,
`models/_sweep_binary/20260612T062412Z/summary.parquet`) keyed to the old pool — i.e. the GBM path
excluded by decision 4.

**This assertion deserves credit.** It refused to compare fresh predictions against a stale reference
instead of quietly joining on the overlap and printing a plausible verdict. That is the opposite of
the two silent no-ops earlier in this rebuild, and it is why the fix keeps the assertion intact rather
than softening it to a warning.

**Fix (Brian): `--no-verdict`** on `_fm_freeze_window.py run`, threaded through both the per-seed and
the ens3 paths. Predictions are banked either way; only the Tier-1 gate is skipped. Justification: the
gate is how the recipe originally *earned* its freeze, decision 1 is retrain-as-is with no bake-off,
and no headline number depends on Tier-1.

Note `write_run_artifacts` runs **before** the verdict, so seed 0's predictions had already been
banked when the first attempt died — the failure cost one seed, not the run.

**Re-run banked all four artifacts** (seeds 0/1/2 + ens3), 164,644 predictions each, ~9 min/seed.

### The frozen recipe on the corrected label basis

| metric | banked (0.3598 prevalence) | **rebuilt (0.373272)** | Δ |
|---|---|---|---|
| pooled `pr_auc@1e-2` | 0.7832 | **0.7826** | −0.0006 |
| median per-image AUC | 0.7865 | **0.7778** | −0.0087 |
| `precision@5%` | 0.948 | **0.9638** | +0.0158 |
| `meaningful_auc` | — | **0.8342** | new |
| Spearman(pred, `fractional_area`) | — | **0.6050** | new |

Predictions↔labels join is exact: **164,644 / 164,644**.

**Read these with the prevalence shift in hand.** Chance PR-AUC *is* the prevalence, so a flat PR-AUC
at a higher prevalence is a small *real* decline: skill above chance is
`(0.7826−0.3733)/(1−0.3733) = 0.6530` against `(0.7832−0.3598)/(1−0.3598) = 0.6614`. `precision@5%`
rising is likewise partly mechanical. The prevalence-insensitive read is **median per-image AUC, down
0.0087**.

**All three agree on a very small decline — and it is not material.** Per-image AUC has sd 0.0886 over
38 images, so SE ≈ 0.0144; a 0.0087 drop sits well inside one standard error. The defensible statement
is that **the frozen recipe transfers to the corrected label basis unchanged**, not that it degraded.
Per-image AUC still ranges widely (fold 30 `ESP_068483_2280` 0.5589 at 82 % positive; fold 37
`ESP_076723_2265` 0.8780), which is the known between-image heterogeneity.

### 2026-08-23 — DAG step 7 A1 arm COMPLETE. The skill gate PASSES, and A1's cost has collapsed.

`striping_a1_loio.py` patched first so it retains tile keys (Brian's ruling, 2026-08-21), then run over
both stores, 38 LOIO folds each.

### SKILL GATE: PASS

| store | median per-image AUC | mean | frac ≥ 0.7 | pooled PR-AUC |
|---|---|---|---|---|
| `fang_embeddings` (baseline) | **0.783499** | 0.784339 | 0.842105 | 0.775446 |
| `fang_embeddings_a1` | **0.781057** | 0.783774 | 0.842105 | 0.783631 |

**Δ median per-image AUC (A1 − baseline) = −0.0024** (gate: ≥ −0.02) → **PASS**
**Δ pooled PR-AUC = +0.0082** — A1 is *better* on the pooled read.

### This is the end-to-end re-check the audit demanded, and the answer changed

The banked A1 figures were **η² 0.196→0.141 and −0.024 AUC**, which the audit explicitly ruled
"came from different A1 definitions and remain non-comparable until [parity is re-checked] end to
end". This run is that re-check. **A1's skill cost is now −0.0024, an order of magnitude smaller than
the −0.024 of record**, and it *gains* +0.0082 pooled PR-AUC.

The mechanism is R07 + R38, exactly as designed. The old number was produced when training used one
whole-window statistic and deployment used a per-frame one — two different normalisations compared as
though they were one. Both sides now call `src.striping.A1_ARM`, the per-frame **native** statistic,
and R38's `A1_VALID_FLOOR = 1` stops the clip manufacturing nodata out of dark terrain. Fixing the
definition removed almost all of the apparent cost.

**Consequence: the "A1 costs skill" caveat should not be carried forward at −0.024.** Any document
quoting it needs updating at step 12. What A1 buys (the η² striping reduction) has *not* been
re-measured here — that is a map-time quantity and comes at step 11/12.

⚠ **η² is NOT re-derived by this run.** Do not pair the new −0.0024 with the old 0.141 η²; they would
again be two different definitions spliced together, which is the precise error this entry records
being fixed.

### Both artifacts are calibration-ready

`reports/figures/loio_{fang_embeddings,fang_embeddings_a1}/predictions.parquet`: **164,644 rows each,
columns exactly `obs_id, ti, tj, y_true, y_pred`, 0 duplicate keys, join to labels 164,644/164,644.**
Step 9 can now bank a calibration for either arm with the same driver. The skill-gate CSVs are
unchanged.

### Cross-harness agreement on the baseline

This script's baseline median per-image AUC is **0.783499**; yesterday's `_fm_freeze_window` banked
**0.7778** on the same embeddings and labels. **Δ = +0.0057.** The harnesses differ deliberately
(different fold plumbing and inner-val rotation), so exact equality was never expected; agreement to
~0.006 on a quantity with sd 0.089 across 38 images is reassuring rather than alarming. **Quote the
`_fm_freeze_window` number as the recipe's figure of record** — it is the one the deployable head is
trained from — and treat this one as the A1 comparison's internal control.

**Code:** `write_arm_predictions()` extracted from `main()` so it is testable, with 4 tests
(`tests/test_a1_loio_keys.py`): both arms in the calibration schema, keys surviving the round-trip,
duplicate keys raising rather than reaching the calibrator, and `--tag` isolating a `--restrict-store`
run from the files of record.

**Gotcha:** `conda run python -` with a heredoc **silently does not write** — it printed success while
leaving the file untouched. Companion to the `python -c` newline rejection. Rule: put multi-line
Python in a file and pass the path.

### 2026-08-23b — DAG step 8 COMPLETE: two heads, distinct arms, and R09's residue closed

| arm | `norm_arm` | recipe_hash | model_hash | in-sample AUC |
|---|---|---|---|---|
| baseline | `none` | **a5ffca2dcc536855** | e13762fd4a71909e | 0.9615 |
| A1 | `a1` | **7bbd8a8e1d377f6e** | 08ee3637dd62f6ff | 0.9796 |

Both on 164,644 × 768, pos_rate 0.3733, 38 images, 0 NaN rows; save/load round-trip max |dp| ≤ 3.1e-07.
~100 s each. Written to **`models/deployable_g2/` and `models/deployable_a1_g2/`** rather than beside
the legacy head, because `map_region.resolve_model_dir` picks `hits[-1]` **by name** — a new hash next
to `86c51a5dca220f63` would be an ambiguity the audit already flagged. Step 11 must pass
`--model-parent` explicitly.

**R07 gate PASSES:** the two arms produce different recipe hashes and neither reuses the legacy one.

### ⚠ R09's residue found live, and closed

The audit lists R09 as fixed. Only its **arm** half was: `norm_arm` is in the hash. Its **metric** half
was not. `FROZEN_RECIPE` hardcoded `loio_pooled_pr_auc: 0.7832` and `loio_med_per_image_auc: 0.7865`,
and that dict is stamped verbatim into every head's `recipe.json` — so **both freshly trained heads
shipped cards asserting 0.7832 / 0.7865 when step 7 measured 0.7826 / 0.7778 on the corrected basis**,
and the **A1 head asserted the baseline's numbers, which it never had under any basis.** Exactly R09's
original failure (the F head claiming 0.7832 against a true 0.7438), reproduced on new artifacts.

**Fix:** the two metric keys are removed from `FROZEN_RECIPE`. A recipe is a *configuration*;
performance is a *measurement of one fit against one label generation*. It cannot be a constant, and
it must not sit inside the recipe hash — hashing it would make an unchanged configuration hash
differently the day its LOIO is re-run. Metrics live beside the head, in step 7's
`predictions.parquet` and the per-arm LOIO summaries.

**Test added** (`tests/test_deployable_head.py`): `FROZEN_RECIPE` may contain no key matching
auc/pr/precision/recall/brier/rmse/spearman/score, and neither 0.7832 nor 0.7865 by value.

**A clean confirmation from the retrain:** recipe hashes changed (`c1783d53…`→`a5ffca2d…`,
`f8795fff…`→`7bbd8a8e…`) while **model hashes did not** (`e13762fd…`, `08ee3637…` both identical to
the first run). Removing metrics from the recipe changed the configuration identity and nothing about
the weights — and it independently re-confirms the head training is deterministic.

### Still open, deferred by ruling: the embedding store is not in the card

The audit requires "the preprocessing arm, embedding-store identity/digest, and target definition in
the head recipe/card/hash". Arm ✓ and target ✓; **store name and store digest are absent** (grep: 0
occurrences). `embedding: "fang_vit_b16_gem_p3"` names the *model*, not the store. So nothing records
that a head consumed `fang_embeddings_a1` rather than `fang_embeddings` except `norm_arm` — a
command-line label, not a measured property of the input. Brian deferred this: nothing is *misstated*
by the absence, and a digest over a 3.5 GB store is a design choice to make deliberately. Post-step-12,
alongside the embedding-npz provenance gap (2026-08-20k) and the write-time DAG check (2026-08-20h) —
all three are the same missing idea: **producers record their inputs but never verify them.**

**A test caught the change and asked to be updated — correctly.**
`test_the_shipped_heads_are_the_unversioned_ones_this_finding_is_about` asserted `armed == 0`,
pinning the pre-R07 snapshot, and carried its own instruction: *"a head now declares norm_arm —
retrained under R07? update this test, it pins the pre-fix state deliberately."* Training the first
armed heads tripped it exactly as designed. Rewritten as
`test_banked_heads_split_into_pre_and_post_R07_generations`, which pins the **invariant** rather than
the snapshot: the legacy collision must still be on disk (it documents what R07 found), and **no
`recipe_hash` may ever span two arms** — if one does, a raster can be rendered with the wrong
preprocessing and no error. Fast suite **812 passed**.

### 2026-08-23c — DAG step 9 COMPLETE: both calibration layers banked; R54's instrument is the real finding

| arm | Tier-1 ECE (LOIO) | gate ≤0.05 | Tier-2 top_ratio | marginal_L1 | banked |
|---|---|---|---|---|---|
| baseline | **0.020405** | PASS (+0.0296) | 0.86 | 0.0003 | `models/deployable_g2/calibration.npz` |
| A1 | **0.052263** | **FAIL (−0.0023)** | 0.87 | 0.0004 | `models/deployable_a1_g2/calibration.npz` (**forced**) |

**The fail-before-write fix worked.** A1's first run exited **1**, wrote nothing, and said so — exactly
the defect the audit recorded ("the layer was saved before the gates were evaluated, so a run that
printed FAIL still overwrote the banked calibrator, and `main` returned 0 regardless").

**Brian's ruling:** 0.0023 over a 0.05 threshold is immaterial; bank A1 with `--force` and render both
arms **with** isotonic. The override is recorded honestly — A1's meta carries
`gates_passed=False, forced=True`, baseline `gates_passed=True, forced=False`. Anyone reading the
layer can see it missed.

**The isotonic-vs-raw comparison Brian asked for is FREE — no second render.** Both map drivers
already emit `<tile>_prob.tif` (isotonic), `<tile>_prob_raw.tif` (**uncalibrated, always kept**) and
`<tile>_abundance.tif` (qmatch) from one pass. A `--no-isotonic` render would spend ~23 GPU-h
producing a `_prob.tif` identical to the `_prob_raw.tif` already on disk.

**R84's invariant holds on both layers:** `t2_y` max = **0.293242** = pool max `fractional_area`,
matching the value step 4 reproduced. The qmatch maps onto the true marginal, tail included.

### ⚠ R54's per-image level instrument is NOT emitted — and what it shows is worse than A1's ECE miss

PLAN_Rebuild §3 step 9 requires "R54's per-image `mean(pred)/mean(true)` distribution + count", and
the audit's gate row states it "is emitted beside pooled results". `grep` finds **no trace** of it in
`bank_calibration.py`. **That is the fourth audit gate row this rebuild has found overstated**, after
existence-only resume, LOIO tile keys, and R09's metrics.

Computed by hand on the abundance product:

| arm | pooled mean(pred)/mean(true) | per-image median | per-image IQR | within 0.8–1.2 | min | max |
|---|---|---|---|---|---|---|
| baseline | **1.0220** | 1.106 | 0.568 – 1.628 | **8/38 (21 %)** | 0.013 | 6.546 |
| A1 | **1.0278** | 1.077 | 0.552 – 1.583 | **7/38 (18 %)** | 0.000 | 5.973 |

**Pooled level agreement is excellent (1.02) while only about one image in five sits within ±20 %**,
and per-image ratios span 0.013× to 6.5×. The pooled figure averages away errors that partly cancel.
This is precisely why R54 exists and why the audit demanded the per-image distribution be reported
*beside* the pooled one with an explicit rule about which governs promotion.

**This matters more than A1's 0.0023 ECE overshoot.** It says the abundance product's **per-place
level** is unreliable even where the pooled marginal is near-perfect — the same axis on which the
F build was hard-aborted 2026-07-30 (between-place level coherence). It does not block step 11; it
governs how the resulting map may be described. **Do not quote the pooled 1.02 as evidence of
per-place level accuracy.**

Emission of the instrument is left as a step-12 item; the numbers above are the record for now.

### 2026-08-23d — Sherlock prep for steps 10–11 complete (nothing submitted)

Everything needed for step 11 is staged, digested and documented in **PLAN_Rebuild §4b**. No job has
been submitted and nothing has been uploaded — this is preparation only.

**Upload is 347 MB, nine items, each sha256'd** (table in §4b). Heads + calibrations + the size-floor
basis + the 341 MB Fang checkpoint + the two new sbatch files. The two basis copies share digest
`4e22a85aa1f02135` on purpose: the basis is a property of the **label pool**, not of CTX
preprocessing, so both arms take the same file, one beside each head so each arm is self-contained.

Not uploaded, fetched on Sherlock: the 26 Murray zips (~41 GB) and the SeamMap shapefiles
(`load_frames` pulls them from the zips via `/vsizip/vsicurl/`). Not moved at all:
`context_patches` (18.3 GB) and `packaged` (50 GB).

**Two NEW sbatch scripts; `run_region_array.sbatch` is deliberately left untouched** as the record of
how the shipped map was made. The rebuild scripts differ in four ways, each load-bearing:

1. **26 tiles, not 19** — the old script covers `EXPANSION_TILES`; rendering only those would
   silently ship 7 pre-rebuild tiles inside a "rebuilt" map.
2. **`--model-parent models/deployable_g2`** — `resolve_model_dir` picks `hits[-1]` **by name**, so
   with the legacy `86c51a5dca220f63` present the default parent is a coin flip.
3. **`--size-floor-basis`** — absent it, `size_floor_tags` warns and emits **no** `SIZE_FLOOR_*`
   tags; 52 rasters would ship unable to state which boulders they count.
4. **`BATCH=96`, not 256** — 96 is the parity reference's batch, and the 256 default was a guess:
   measured throughput is 32/96/256/512 → 766/723/730/731 img/s, i.e. flat. The larger batch cost
   parity comparability for nothing.

The A1 script passes `--head` directly (that driver has no `--model-parent`), and R07 makes it
**refuse** a head it cannot verify as the `a1` arm — so it must be `7bbd8a8e1d377f6e`, never the
legacy `models/deployable_a1/86c51a5dca220f63`.

⚠ **`git pull` on Sherlock is not optional.** The open-hoist (`bdc1d19`) is what makes step 11
≈23 GPU-h instead of ≈31; without it every tile pays 144 × 7.95 s of redundant `rasterio.open`. The
same pull carries `--no-verdict`, the embedding staleness check and the A1 tile-key fix.

Both sbatch files carry the assembly-failure triage inline (`max|Δ| ≲ 1e-4` → re-run that tile with
`--force`; `~0.1–1` → the stale-partial case R14 exists for, investigate), so whoever is at the
terminal at 2 a.m. does not have to find DECISIONS to interpret a crash.

### 2026-08-23e — Pre-flight review caught a Sherlock-only crash in the A1 arm

Reviewing the step-11 scripts before submission rather than after. One real defect, one non-issue
confirmed, and the rest validated.

**⚠ THE DEFECT: `src/striping.py` used two roots for one concept.**
`CTX_ZIP_DIR = cache_v2/ctx_tiles` but `SEAM_DIR = cache/ctx_tiles`. On this laptop those are the
**same directory** — `cache_v2/ctx_tiles` is an NTFS junction into `cache/` — so the split has been
invisible for months. On Sherlock there is no junction: SHERLOCK_RUN.md §2 does
`ln -sfn $SCRATCH/hirise2ctx/cache $HOME/hirise2ctx/cache_v2`, so **`cache_v2` exists and
`$HOME/hirise2ctx/cache` does not.**

Consequence: `load_frames` would find no cached GeoPackage, no local SeamMap, fetch the shapefile
over `/vsizip/vsicurl/` (fine), and then **die on `g.to_file(cache_gpkg)` writing into a directory
that does not exist** — with no `mkdir` anywhere in the path. That is the entire A1 arm, failing on
its first tile, after the array had been queued and GPUs allocated.

**Fix:** `SEAM_DIR = CTX_ZIP_DIR` (one root, resolves through the Sherlock symlink) plus
`cache_gpkg.parent.mkdir(parents=True, exist_ok=True)` on write. On Windows this is a no-op — the
junction already made the two paths identical — so no local artifact moves.

**3 tests** (`tests/test_seammap_cache_root.py`): the two roots must be equal on every platform; a
`load_frames` call against a non-existent cache root must create it and write; the second call must
read back what the first wrote. The first test is the one that matters — it forbids the two-root
split from reappearing, which is the shape of the bug, not its symptom.

*Left alone:* `scripts/f_leg_b_frame_list.py:37` has its own `SEAM_DIR = cache/ctx_tiles` copy. It is
an F-programme script and F was hard-aborted 2026-07-30, so it is not on any live path — noted, not
touched.

**Confirmed NOT an issue: A1 has no ordering dependency on the baseline raster.** `_tile_crs` reads
the CRS from the **tile zip** and only falls back to `reports/map_region/{tile}_abundance.tif` if the
zip is absent. On Sherlock the zips are present, so either arm may run first — the plan's reading of
the stale DAG text is correct, now verified in code rather than inferred from a comment.

**Validated mechanically, both scripts:** every flag used exists in the driver's argparse (7 each);
the tile list equals `BLOCK_TILES` exactly (26, order-insensitive); and the `--array=0-5` stride
covers each tile exactly once. Environment lines (`ml python/3.12.1`, the group venv path) match
SHERLOCK_RUN.md verbatim.

**Upload shrinks to ~5.3 MB.** `setup_sherlock_env.sh` downloads the 341 MB Fang checkpoint itself,
so it is already on Sherlock if the venv exists — only the two `*_g2` model directories need to move,
which is small enough for the OnDemand browser.

Fast suite **815 passed**.

### 2026-08-24 — Step 11 A1 array failed on the R07 guard. My error, caught before any GPU work.

All **6 tasks of array 40561659 failed in 49 s**, exit 1, before a single tile was rendered:

```
ValueError: the head (models/deployable_a1_g2/7bbd8a8e1d377f6e) declares norm_arm='a1'
but this path supplies 'a1_native_perframe_tilesupport_v2'. These are different input
distributions; the output would look plausible and be wrong.
```

**Cause: I passed `--norm-arm a1` at step 8.** The arm identifier is *versioned* —
`src.striping.A1_ARM == "a1_native_perframe_tilesupport_v2"` — and the bare string `a1` is not it.

**It was entirely avoidable.** `train_deployable_head.py` does
`norm_arm = args.norm_arm or infer_norm_arm(args.store_name)`, and
`infer_norm_arm("fang_embeddings_a1")` **already returns the correct versioned id**. Omitting the flag
would have been right. I overrode correct inference with a literal I invented when drafting the plan's
step-8 row, then executed my own draft. The baseline arm survived only because `--norm-arm none`
happens to equal `NO_NORM_ARM` exactly.

**The guard behaved perfectly** — strict refusal, before any compute, with an error that names both
values and the consequence ("the output would look plausible and be wrong"). This is the third R07/R14
guard to catch something real this rebuild, and the second time a *loud* failure has saved a silent
one: an unversioned match would have rendered 26 A1 tiles with a head trained on a different input
distribution.

**Fix:** A1 head retrained with the flag omitted → `norm_arm=a1_native_perframe_tilesupport_v2`,
**recipe_hash `66ec8b755b9c0b20`**. `model_hash` is **`08ee3637dd62f6ff`, unchanged** — identical
weights, only the arm label and the configuration identity moved, which also re-confirms training
determinism for the third time. The wrong-armed `7bbd8a8e1d377f6e` is deleted so nobody can point at
it. `calibration.npz` and `size_floor_basis.json` are unaffected.

**Two tests added** (`tests/test_deployable_head.py`), because the broken invariant spanned two
modules and was untested: `infer_norm_arm("fang_embeddings_a1")` must equal `striping.A1_ARM`; the
bare `"a1"` must be *rejected* by `require_norm_arm(..., strict=True)`; and a head built from each
store's inferred arm must satisfy its own driver.

`run_rebuild_a1_array.sbatch` now points at `66ec8b755b9c0b20` and carries the reason inline.
PLAN_Rebuild step 8 amended to say **omit `--norm-arm`**.

**Unrelated but worth recording: Sherlock allocated RTX 2080 Ti (11 GB), not L40S.** The ≈23 GPU-h
estimate and the `--time` limits (8 h baseline / 10 h A1) were sized against an L40S. A 2080 Ti is
roughly 2–3× slower on fp16 ViT, so tasks may hit the wall. That is **not** a failure — R14 makes it
resumable; re-`sbatch` the same command and finished tiles are skipped, partial windows kept.
