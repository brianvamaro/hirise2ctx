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

## Open at this date

- **Per-image CTX windows for the remaining 9 ObsIds** — Stage 2 helpers are ready; this
  is a 7-more-tile-downloads operation (~10 more GB on disk) plus the cheap windowing.
  Deferred until needed by Stage 3 (co-registration) or a Stage-4 sweep.
- **Stage 3 co-registration** — phase correlation between decimated HiRISE JP2s and
  cached CTX windows; will need the 2 already-cached HiRISE JP2s in `cache/hirise_jp2/`
  plus stage-2 windows for those two ObsIds (`ESP_069669_2220` ✓, `ESP_047976_2020`
  needs its `W040_N20` → `E-40_N20` (with padding fallback to `E040_N20`) tile fetch).
- **Sphere vs oblate-spheroid CRS mismatch (sub-pixel)** — revisit during Stage 3 if
  phase correlation shows a systematic ~1 px bias along the equator-to-pole axis.
- **`min_confidence` default** — leave `null` until distribution is reviewed.
