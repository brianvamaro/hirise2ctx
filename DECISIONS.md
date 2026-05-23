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

## Open at this date

- **Stage 3 thresholds (flag/fail)** — collect more data first before pinning down.
  Current distribution suggests `|shift| > 500 m` + `peak < 0.2` as a starting point,
  but Stage 4 will benefit from a few more images and a re-look at ESP_056165_2200
  (the only low-peak case so far).
- **ESP_057469_2215 multi-tile windowing** — see the 2026-05-22 tile-straddle entry.
  Currently dropped from the Stage 4 sweep. Decide whether to fix at Stage 5 / 6.
- **`min_confidence` default for the `score` column** — leave `null` until the
  per-tile distribution after labels is reviewed; the binary contingency table in
  notebook 06 is the first data-grounded input.
- **`binary_count_threshold` rebalance** — current placeholder 5 is too high vs
  area threshold 0.005 (only 2 count-only tiles vs 5,504 area-only). Decide at
  modeling time which side to commit to.
- **BoulderNet 5×5-px design-floor filter (Stage 4 decision, not yet made)** —
  per the 2026-05-25 Methods errata entry, sub-threshold polygons survive in
  4 of 5 audited shapefiles (0-2 % in the textured images, 80.77 % in
  ESP_056165_2200). The filter hook is at Stage 4
  (`labeling.detection_filters.min_size_m`, currently null); applying a value
  + re-running `scripts/run_stage4.py --all` is the implementation. Both this
  and the `min_confidence` decision are Stage 4 decisions because once
  per-tile aggregates are computed the polygon-level contributions can't be
  undone downstream.
- **Cache the 4 missing PDS `.LBL` files** to complete the per-image pixel-size
  audit (ESP_069669_2220, ESP_071093_2210, ESP_075577_2105, ESP_039820_1750).
  Trivial — one-time fetch, ~10-20 KB each — but currently uncached because
  these were `trusted_prj` images that didn't need SP1 correction at Stage 1.
  Without these, the boulder-size audit table in `docs/methods.md` §2.2 is
  incomplete (5 of 9 polygon-bearing images covered).
- ~~**Stage 4b texture features**~~ — landed 2026-05-23 (see entry above). 9 feature
  families, 643,910 rows, 3.5 GB on disk including context patches. Next is Stage 5
  splitter, then Week 3 modeling.
