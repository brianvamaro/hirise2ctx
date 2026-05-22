# DATA DICTIONARY

Schema reference for cached intermediate artifacts and (later) the paired training
dataset. Grows stage-by-stage. Refer to [CLAUDE.md](../CLAUDE.md) for the pipeline spec
and [DECISIONS.md](../DECISIONS.md) for runtime-verified facts.

## Stage 1 — `cache/reprojected_detections/`

Per-image BoulderNet detection polygons, reprojected into the target CTX CRS.

### `{ObsId}.gpkg`
GeoPackage with one layer `detections`. One row per BoulderNet polygon. Columns
mirror the source BoulderNet shapefile DBF schema (see [DECISIONS.md](../DECISIONS.md)
2026-05-20 entry):

| Column | Type | Meaning |
|---|---|---|
| `geometry` | Polygon | Boulder outline in **target CRS** (Mars 2000 Equidistant Cylindrical, metres) |
| `score` | float | BoulderNet confidence, 0.10–0.83 in our manifest |
| `cat_id` | int | Always 0 |
| `cat_name` | str | Always `"boulder"` |
| `isin_slice` | bool | True if polygon lies within a BoulderNet slice (not at the slice border) |
| `is_at_edge` | bool | True if polygon touches a tile edge (candidate for filtering) |
| `id` | int | BoulderNet polygon id |

### `{ObsId}.json` (sidecar)
| Field | Type | Meaning |
|---|---|---|
| `obs_id` | str | HiRISE Observation ID |
| `n_polygons` | int | Polygon count in the cached GeoPackage |
| `source_path` | str | Absolute path of the BoulderNet shapefile that was read |
| `source_mtime_iso` | str | mtime of the source shapefile when cached (UTC ISO) |
| `source_crs_wkt` | str | The shapefile's `.prj` WKT (after SP1 correction if applied) |
| `target_crs_wkt` | str | The target CRS WKT this reprojection produced |
| `config_hash` | str | SHA256 of the config snapshot that produced this cache |
| `correction.status` | str | `trusted_prj` or `sp1_corrected_from_pds_label` |
| `correction.original_sp1_deg` | float | (sp1_corrected) the buggy SP1 value found in the `.prj` |
| `correction.corrected_sp1_deg` | float | (sp1_corrected) the PDS-LBL `CENTER_LATITUDE` used as replacement |
| `correction.pds_center_lat_deg` | float | (sp1_corrected) raw PDS `CENTER_LATITUDE` |
| `correction.pds_center_lon_deg` | float | (sp1_corrected) raw PDS `CENTER_LONGITUDE` |
| `correction.pds_a_axis_km` | float | (sp1_corrected) PDS `A_AXIS_RADIUS` (image-local Mars radius) |
| `written_at_iso` | str | When the cache was written (UTC ISO) |

## Stage 2 — `cache/ctx_tiles/`

Murray Lab CTX mosaic v01 tiles, downloaded once per unique tile and reused across all
ObsIds that share it (e.g. `E000_N40` covers 3 manifest images). Filename uses the
Murray Lab form (output of `src.ctx_tiles.manifest_to_murray`), even when the URL Murray
Lab actually served uses the padded manifest form — the sidecar's `resolved_tile_name`
records which form worked.

### `{murray_tile}.zip`
Raw upstream archive. Contains one nested directory and one GeoTIFF; do not unzip — the
pipeline reads via `/vsizip/`.

### `{murray_tile}.json` (sidecar)
| Field | Type | Meaning |
|---|---|---|
| `murray_tile` | str | Murray-form tile name we constructed for the URL (e.g. `E0_N40`) |
| `resolved_tile_name` | str | Tile name in the URL that actually returned 200 — `murray_tile` or its zero-padded fallback |
| `source_url` | str | The successful upstream URL |
| `downloaded_at_iso` | str | When the zip was fetched |
| `zip_bytes` | int | On-disk zip size in bytes |
| `inner_tif` | str | Path inside the zip to the single GeoTIFF (`MurrayLab_GlobalCTXMosaic_V01_{T}/MurrayLab_CTX_V01_{T}_Mosaic.tif`) |
| `inner_crs_wkt` | str | The mosaic's WKT — `Mars_2015_Ocentric_Equirectangular_clon_0` (oblate spheroid, f ≈ 1/170) — distinct from our pipeline `target_crs` (pure sphere, f = 0); the discrepancy is sub-pixel at 5 m/px |
| `inner_transform` | list[6] | Affine `[a, b, c, d, e, f]` of the source raster |
| `inner_shape` | list[int, int] | `[height, width]` of the source raster, normally `[47420, 47420]` for a 4°×4° tile |
| `inner_dtype` | str | Source pixel dtype, normally `uint8` |

## Stage 2 — `cache/ctx_windows/`

Small per-ObsId GeoTIFFs windowed out of the parent CTX tile around the HiRISE
polygon-footprint bbox (with `ctx_retrieve.buffer_m` of buffer), snapped to the source
tile's integer pixel grid. Read repeatedly by Stage 4 (labeling) and Stage 3
(co-registration).

### `{ObsId}.tif`
Single-band uint8 GeoTIFF. Pixel size matches the source mosaic (~5 m/px). Tile-internal
blocks are 256×256 (compression `deflate`). CRS is whatever the source tile reported —
currently `Mars_2015_Ocentric_Equirectangular`, **not** our `target_crs`; this is
intentional (preserves pixel-exact alignment with the upstream mosaic for Stage 4's grid
anchor).

### `{ObsId}_hirise_mask.tif`
Single-band uint8 GeoTIFF, **same CRS / transform / shape** as `{ObsId}.tif`. 1 where the
decimated HiRISE (5 m/px) has a valid (non-zero) pixel after reprojection onto the CTX
grid; 0 elsewhere. Stage 4 **must** consume this raster to gate label generation —
emitting a `boulder_area = 0` label outside the HiRISE swath would incorrectly inflate
the zero-tile count (the absence of polygons there is "no HiRISE coverage", not
"no boulders observed"). The mask also handles NaN/0 pixels *inside* the swath's
rotated-rectangle outline (those areas are HiRISE-unobserved too). Generated using
`nearest`-neighbor resampling so the binary boundary stays crisp.

### `{ObsId}.json` (sidecar)
| Field | Type | Meaning |
|---|---|---|
| `obs_id` | str | HiRISE Observation ID |
| `source_murray_tile` | str | Murray-form tile name (matches `cache/ctx_tiles/{name}.zip`) |
| `source_zip` | str | Absolute path of the cached tile zip |
| `source_inner_tif` | str | Path inside the zip to the source GeoTIFF |
| `requested_bounds_target_crs` | list[4] | `[xmin, ymin, xmax, ymax]` we asked rasterio for (in target CRS metres) |
| `actual_bounds_target_crs` | list[4] | `[xmin, ymin, xmax, ymax]` actually written, from the output GeoTIFF (may differ from requested by ≤ 1 px due to rasterio window rounding) |
| `actual_transform` | list[6] | Affine of the output GeoTIFF |
| `actual_shape` | list[int, int] | `[height, width]` of the output |
| `buffer_m` | float | Buffer added around the polygon-footprint bbox (or used by the nominal-footprint fallback) |
| `footprint_source` | str | `polygon_bbox` (normal path) or `nominal_from_manifest` (empty-shapefile fallback using `manifest.CenterLat`/`CenterLon_180`) |
| `n_polygons_anchor` | int | Polygon count in the Stage-1 cache used to compute bounds (0 for the fallback path) |
| `hirise_mask_path` | str | Absolute path of the companion `{ObsId}_hirise_mask.tif` |
| `hirise_coverage_fraction` | float | Share of CTX-window pixels with HiRISE coverage (`mean(mask == 1)`). Typically ~0.4–0.7 because the HiRISE diagonal swath occupies only part of the polygon-bbox window. A very small value (<0.05) usually means the polygon-bbox straddles a Murray Lab tile boundary — the cached window only covers the in-tile portion and the bulk of the HiRISE swath lives in the neighbouring tile. ESP_057469_2215 is the priority10 example (0.001). |
| `config_hash` | str | SHA256 of the config snapshot that produced this cache |
| `extracted_at_iso` | str | When the window was written (UTC ISO) |

## Stage 3 — `cache/coregistration/`

Per-ObsId sub-pixel rigid-translation correction from HiRISE to CTX, solved via phase
correlation on a power-of-2 sub-window of the warped imagery. Empty `(dx, dy)`-style
provenance is the entire deliverable; Stage 4 reads this when it wants to refine the
nominal grid anchor (and may decide to ignore it for flagged outliers — see
`notebooks/05_coregistration_qa.ipynb`).

### `{ObsId}.json`
| Field | Type | Meaning |
|---|---|---|
| `obs_id` | str | HiRISE Observation ID |
| `ctx_window_tif` | str | Absolute path of the Stage 2 CTX window this shift is relative to |
| `ctx_transform` | list[6] | Affine `[a, b, c, d, e, f]` of the CTX window — included so future code can convert (dx, dy) between metres and pixels without re-opening the GeoTIFF |
| `ctx_crs_wkt` | str | WKT of the CTX window's CRS (`Mars_2015_Ocentric_Equirectangular`) |
| `fft_window.size_px` | int | Side length of the FFT sub-window, in CTX pixels — always a power of 2, ≤ `coregistration.fft_window_px` from config |
| `fft_window.row_off` | int | Row offset of the sub-window's top-left corner inside the CTX window |
| `fft_window.col_off` | int | Column offset, ditto |
| `fft_window.config_max_px` | int | Upper bound used at solve time (`coregistration.fft_window_px`) |
| `shift_px.dy` | float | Sub-pixel translation in pixel rows: shift to apply to the HiRISE-on-CTX-grid array so it aligns with CTX |
| `shift_px.dx` | float | Sub-pixel translation in pixel columns, same convention |
| `shift_m.dy` | float | `dy_px * abs(ctx_transform.e)` — translation in metres |
| `shift_m.dx` | float | `dx_px * abs(ctx_transform.a)` |
| `shift_m.magnitude` | float | Euclidean magnitude — CLAUDE.md §3.3 expects O(200 m) |
| `peak_correlation` | float | Pearson correlation between CTX sub-window and the shift-corrected HiRISE sub-window, computed over the still-valid interior (margin-cropped). Used as a confidence proxy — higher is better; bland-plains scenes produce low values regardless of the true shift. |
| `upsample_factor` | int | `skimage.registration.phase_cross_correlation` `upsample_factor` (default 20 → 0.05 px ≈ 0.25 m granularity at 5 m/px) |
| `config_hash` | str | SHA256 of the config snapshot that produced this shift |
| `solved_at_iso` | str | When the shift was solved (UTC ISO) |
