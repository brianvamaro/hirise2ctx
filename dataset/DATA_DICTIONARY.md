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
| `score` | float | BoulderNet confidence. **Measured range 0.100000–0.955996** over the 39 readable v2 vClaire exports (7,645,643 detections, 2026-08-06); the previously documented "0.10–0.83" was wrong at the top end. Every readable export's `.dbf` bottoms out at exactly 0.100000 — the detector's own floor. ⚠ Three images' **cached polygons** nevertheless start at 0.617257 / 0.473420 / 0.406699 because their source `.shp` is byte-truncated (R23); see `realised_label_basis` in the Stage-4 sidecar and **DECISIONS 2026-08-06o**. |
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
| `n_polygons_raw` | int | Polygon count read from the source, before null-geometry rows were dropped |
| `n_dropped_null_geometry` | int | Rows with a `.dbf` record but no polygon, dropped at ingest |
| `source_integrity` | obj | **Is the source `.shp` byte-complete?** (added 2026-08-06). A shapefile's 100-byte header declares its own total length, so a partially-copied `.shp` is self-diagnosing. `status` ∈ `complete` / `truncated` / `unreadable`; when truncated also `declared_bytes`, `actual_bytes`, `missing_bytes`, `n_records_index`, `n_records_present`. **4 of 40 v2 exports are truncated** — `ESP_017355_2260` (−354 MB), `ESP_046803_2325` (−132 MB), `ESP_068483_2280` (−173 MB) and the long-excluded `ESP_028537_2270` (−513 MB). See **DECISIONS 2026-08-06o**. |
| `null_geometry_basis` | obj or null | **What population was dropped** (added 2026-08-06). Null when nothing was dropped. Carries `n_rows` / `n_dropped` / `n_kept` / `dropped_fraction` and the kept-vs-dropped `score` distributions, plus `is_rank_truncation` — True when every dropped row scores at or below every kept row, the fingerprint of a score-ordered truncation rather than sparse export noise. When True, `realised_score_floor` is the confidence floor this image's labels actually sit at. This is the record whose absence let R23 be filed as "benign density hygiene". |
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
| `footprint_source` | str | How the window bounds were chosen. `polygon_bbox` — the normal path, the detections' bbox plus `buffer_m` (all 39 v2 windows). `pds_label_footprint` — empty shapefile, sized from the image's own PDS extents (preferred fallback since 2026-08-06). `nominal_from_manifest_coslat` — empty shapefile *and* no usable `.LBL`, so a `nominal_hirise_width_m/cos(lat)` × `nominal_hirise_length_m` rectangle centred on `manifest.CenterLat`/`CenterLon_180`; this emits a `RuntimeWarning` because the nominal is measurably undersized (too narrow for 39/39 real footprints, too short for 13/39). Pre-2026-08-06 sidecars use the retired literal `nominal_from_manifest`, which spent its width in *projected* metres (R67). |
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

## Stage 4 — `dataset/labels/`

Per-ObsId paired tile dataset, one row per (scale, tile) cell. Stage 4 emits all
derived label transforms regardless of `labeling.label_type` so downstream code can
pick its target without re-running label generation.

### `{ObsId}.parquet`
Tidy table; one row per emitted tile. Ineligible tiles (mask coverage < 1.0, or
any sub-tile ineligible at coarser scales) are dropped, not written as NaN.

| Column | Type | Meaning |
|---|---|---|
| `obs_id` | str | HiRISE Observation ID |
| `scale_idx` | int | Index into `labeling.tile_sizes_px` (0 = finest) |
| `tile_size_px` | int | Tile side in CTX pixels (8, 16, 32, or 64 today) |
| `tile_size_m` | float | Tile side in metres (`tile_size_px * px_x`, ~40, 80, 160, 320 m) |
| `ti` | int64 | **Absolute** tile row index in mosaic-pixel coords (`mosaic_pixel_row / tile_size_px`). Cross-image-comparable; ti at scale 2S is `ti(S) // 2`. |
| `tj` | int64 | Absolute tile column index, same convention |
| `xmin`, `ymin`, `xmax`, `ymax` | float | Tile bounds in the source CTX mosaic CRS (`Mars_2015_Ocentric_Equirectangular`, metres) |
| `boulder_area` | float | Base stat: total polygon area inside the tile, in m². Computed once on the finest grid via 5×-sub-pixel rasterization, summed up the ×2 ladder. |
| `boulder_count` | int64 | Base stat: number of polygons whose centroid lies inside the tile. Unambiguous at borders (each boulder counted once). |
| `tile_area` | float | Constant per scale; equal to `tile_size_px^2 * px_x * px_y` (~1600, 6400, 25600, 102400 m² for 8/16/32/64-px tiles at ~5 m/px) |
| `fractional_area` | float | Derived: `boulder_area / tile_area`. Primary regression target. Heavily zero-inflated; see DECISIONS.md 2026-05-23 distribution. ⚠ **This is not size-independent rock abundance.** It is the area share of boulders *large enough to have been detected in this particular HiRISE image*, and that floor is **per-image and cohort-dependent** (R03/R83/R84, measured 2026-08-11). Stage 4's global `min_size_m = 1.4105 m` equivalent-circle diameter (1.5626 m²) is applied *after* Stage 1, and it sits **above** every 0.25 m/px image's natural floor and **below** every 0.50 m/px image's — so the fine cohort's effective floor is uniformly the filter (1.5626 m², 12 images) while each coarse image keeps its own, **2.9652–5.5719 m² (1.943–2.664 m diameter), 26 distinct values**. The coarse cohort is the internally heterogeneous one. Consequence: `fractional_area` values are **not** commensurable between cohorts, and any comparison against an external rock-abundance product must match the floor. The deployed raster's mixture is recorded in its `SIZE_FLOOR_*` GeoTIFF tags (`src.size_floor`, `scripts/measure_size_floor.py`). |
| `binary_by_area` | bool | Derived: `fractional_area >= labeling.binary_area_threshold` |
| `binary_by_count` | bool | Derived: `boulder_count >= labeling.binary_count_threshold` |
| `count_density` | float | Derived: `boulder_count / tile_area` (per-m² density) |
| `categorical` | Int64 | Derived: bin index from `pd.cut(fractional_area, labeling.categorical_bins)`. Only emitted when `categorical_bins` is non-empty. |
| `config_hash` | str | SHA256 of the config snapshot that produced this row |

### `{ObsId}.json` (sidecar)
| Field | Type | Meaning |
|---|---|---|
| `obs_id` | str | HiRISE Observation ID |
| `n_polygons_stage1` | int | Polygon count in the Stage 1 cache (pre-filter) |
| `n_polygons_after_filter` | int | Polygon count after applying `detection_filters.min_confidence` / `min_size_m`. Equal to `n_polygons_stage1` only when **both** are null — which is **not** the current default: both live configs set `min_size_m: 1.4105` (`min_confidence` is null). |
| `detection_filters` | obj | Snapshot of the **configured** `labeling.detection_filters` (`min_confidence`, `min_size_m`). ⚠ This is byte-identical across all 38 v2 sidecars, so it **cannot** tell you what basis a given image was actually labelled at — use `realised_label_basis` for that. |
| `realised_label_basis` | obj | **The confidence floor these labels were actually built at, per image** (added 2026-08-06). `detection_filters` records the *configured* floor; this records the *realised* one — `realised_score_floor` is the minimum BoulderNet `score` surviving into the labels. For **36 of 38** v2 cohort images that is ~0.10; for `ESP_017355_2260` it is **0.617257** and for `ESP_068483_2280` **0.406699**, because those source `.shp` files are byte-truncated (R23). (The third truncated export, `ESP_046803_2325`, has no labels, hence 38 − 2 = 36.) Always present: `convention` (always `mixed_per_image_confidence_floor`), `temporary_pending`, `decision`, `realised_score_floor`, `score_max`, `score_p1`, `score_median`, `source_truncated` (True / False / **null when unknown — never read null as safe**), `stage1_provenance`. Present when the image is affected: `level_claims_unsafe`, `level_claims_note`, `source_missing_bytes`, `realised_floor_exceeds_expected_by`, `stage1_rank_truncation`, `stage1_dropped_fraction`. `level_claims_unsafe` is derived from the realised floor itself, so it does **not** require a Stage-1 re-run. See **DECISIONS 2026-08-06o**. |
| `coreg_shift_applied` | bool | Whether the Stage 3 (dx, dy) was applied to polygons before rasterization |
| `coreg_shift_m` | obj or null | `{dx, dy, magnitude}` in metres if `coreg_shift_applied` and Stage 3 cache exists |
| `coreg_peak_correlation` | float or null | Stage 3 peak correlation when shift was applied |
| `tile_sizes_px` | list[int] | The ×2 ladder used (`labeling.tile_sizes_px`) |
| `tile_sizes_m` | list[float] | Same in metres |
| `grid_anchor` | str | Always `"ctx_pixel_origin"` today |
| `mosaic_row_origin`, `mosaic_col_origin` | int | Window (0, 0) at this integer mosaic-pixel offset. Combined with `tile_sizes_px` and `tile_size_px * (ti, tj)`, you can compute exact mosaic-pixel coords for any tile. |
| `finest_grid_cells` | list[int] | `[n_ti, n_tj]` candidate cells on the finest grid (before mask-eligibility filtering) |
| `eligibility_rule` | str | Always `"coverage_equals_one"` today (DECISIONS.md 2026-05-23) |
| `eligible_tiles_per_scale` | obj | `{tile_size_px: count}` of tiles actually emitted to the parquet |
| `total_candidate_tiles_per_scale` | obj | `{tile_size_px: count}` before mask filtering — divide eligible/total for the per-scale yield |
| `subpixel_factor` | int | Polygon rasterization oversample (default 5; 1 m sub-pixel at 5 m/px CTX) |
| `subpixel_area_m2` | float | `(px_x * px_y) / subpixel_factor^2` — the area precision of `boulder_area` |
| `binary_area_threshold` | float | Snapshot used to compute `binary_by_area` |
| `binary_count_threshold` | int | Snapshot used to compute `binary_by_count` |
| `categorical_bins` | list | Snapshot of `labeling.categorical_bins` |
| `label_type_primary` | str | The user-declared primary target (`fractional_area`); informational, all label columns are always emitted |
| `ctx_window_tif`, `hirise_mask_tif` | str | Absolute paths to the Stage 2 inputs this label run consumed |
| `parquet_path` | str | Absolute path of the per-tile parquet (companion to this sidecar) |
| `config_hash` | str | Provenance |
| `written_at_iso` | str | When the parquet was written (UTC ISO) |

> **`realised_size_basis`** (obj, Stage-4 sidecar, added 2026-08-06) — the size-floor analogue of
> `realised_label_basis`, carrying R03/R83/R84's mixed **physical size** floor. `detection_filters`
> records the *configured* `min_size_m` and is byte-identical across all 38 v2 sidecars, so it cannot
> express the mixture. Keys: `convention` (`mixed_per_image_size_floor`), `size_metric` (always
> `equivalent_circle_diameter_2sqrt_area_over_pi` — R80 showed this was unpinned anywhere else),
> `configured_min_size_m` / `configured_min_area_m2`, `measured_in_frame` / `measured_in_crs` /
> `measured_in_crs_is_projected`, `realised_diameter_floor_m` (the smallest surviving diameter —
> measured, exactly as `realised_score_floor` is), `diameter_p1_m`, `diameter_median_m`,
> `n_dropped_by_size`, `n_dropped_by_confidence` (attributed **separately**, so a polygon failing
> both is not double-counted), `size_floor_applied` / `confidence_floor_applied` (a configured floor
> with no `score` column is silently skipped, so "configured" and "applied" must be distinguishable),
> `area_total_m2`, `area_dropped_by_size_m2`, `dropped_by_size_fraction`,
> `dropped_by_size_area_fraction`, and `realised_floor_is_looser_than_configured` +
> `realised_floor_note`. See **DECISIONS 2026-08-06u**.

> ⚠ **`source_integrity` and `null_geometry_basis` (Stage 1) and `realised_label_basis` /
> `realised_size_basis` (Stage 4)
> are emitted from 2026-08-06 onward and appear in ZERO currently banked sidecars** — the live
> `cache_v2` and `dataset_v2` trees predate them. Until Stage 1 / Stage 4 re-run, **absence of these
> keys does not mean the source was checked and found clean.** Treat a missing key as *unknown*.
> See **DECISIONS 2026-08-06o** and [PENDING_REBUILD.md](../docs/PENDING_REBUILD.md).

## Stage 4b — `dataset/features/`

Per-tile CTX-derived feature vectors. One row per (scale, ti, tj), joinable 1:1 with the
matching row in `dataset/labels/{ObsId}.parquet` on `(obs_id, scale_idx, tile_size_px,
ti, tj)`. Iterates the eligible-tile set from Stage 4 — features are NOT recomputed for
tiles that Stage 4 dropped.

### `{ObsId}.parquet`
Tidy table; one row per emitted tile. Schema is stable across scales — columns that only
apply at certain scales (lacunarity at S ≥ 32, subtile_variance + canny at S ≥ 16, GLCM
distances > 1 at S ≥ 16) are NaN at the smaller scales rather than absent.

**Operational columns**

| Column | Type | Meaning |
|---|---|---|
| `obs_id`, `scale_idx`, `tile_size_px`, `ti`, `tj` | various | Join key with the labels parquet (identical types + values per row) |
| `valid_pixel_fraction` | float | Share of tile pixels inside the HiRISE coverage mask. 1.0 by construction today (Stage 4 eligibility = strict coverage); recorded as an explicit column so a future relaxed-eligibility config can filter downstream |
| `config_hash` | str | Provenance |

**Intensity stats** — 10 columns, available at every scale

| Column | Type | Meaning |
|---|---|---|
| `intensity_mean`, `intensity_std` | float | Mean and population stddev of CTX DN inside the tile (uint8, so values in [0, 255]) |
| `intensity_min`, `intensity_max` | float | Min and max DN |
| `intensity_p10`, `intensity_p50`, `intensity_p90` | float | 10th / 50th / 90th percentile DN |
| `intensity_iqr` | float | p75 − p25 |
| `intensity_skewness`, `intensity_kurtosis` | float | Centered-moment skewness and excess kurtosis. 0 for uniform-intensity tiles by construction (avoids 0/0 propagation) |

**GLCM (gray-level co-occurrence matrix)** — 18 columns (6 properties × 3 distance bins),
NaN-padded at finest scale where only d=1 is computed

Quantization is scale-dependent (PLAN_Stage4b.md §3.2): 8 levels at S=8, 16 at S=16/32,
32 at S=64. Angles `[0, π/4, π/2, 3π/4]` rotation-averaged into a single value per
(property, distance). Provenance sidecar records the exact `levels_per_scale` and
`distances_per_scale` actually used.

| Column pattern | Type | Meaning |
|---|---|---|
| `glcm_contrast_d{1,2,3}` | float | GLCM contrast `Σ_{i,j} (i-j)² P(i,j)` — angle-averaged; NaN where distance not computed (e.g. d2/d3 at S=8) |
| `glcm_dissimilarity_d{1,2,3}` | float | `Σ_{i,j} |i-j| P(i,j)` |
| `glcm_homogeneity_d{1,2,3}` | float | `Σ_{i,j} P(i,j) / (1 + (i-j)²)` |
| `glcm_energy_d{1,2,3}` | float | `sqrt(Σ_{i,j} P(i,j)²)` |
| `glcm_correlation_d{1,2,3}` | float | Normalized GLCM correlation; can be 0 on constant-intensity tiles (skimage emits NaN; we fill with 0 per `_GLCM_NAN_FILL`) |
| `glcm_ASM_d{1,2,3}` | float | Angular Second Moment = `Σ_{i,j} P(i,j)²` (energy²) |

**Gradient (Sobel)** — 5 columns

| Column | Type | Meaning |
|---|---|---|
| `grad_mag_mean`, `grad_mag_std` | float | Mean and stddev of Sobel gradient magnitude over the tile (after `sigma=1.0` Gaussian smoothing) |
| `grad_mag_p90`, `grad_mag_p99` | float | 90th and 99th percentile gradient magnitude. P99 added 2026-05-23 because boulder edges are rare bright outliers that saturate P90 in busy tiles |
| `grad_dir_circvar` | float | Magnitude-weighted circular variance of gradient direction (angle doubled to handle 180° edge-direction ambiguity). 0 = perfectly aligned edges; 1 = isotropic |

**Shadow / bright-cap** — 3 columns, all per-image DN-mode-derived

Per-image absolute DN cuts are stored in the provenance sidecar (`dn_thresholds.mode`,
`.shadow`, `.shadow_strict`, `.bright`); the per-tile columns just count pixels.

| Column | Type | Meaning |
|---|---|---|
| `shadow_fraction` | float | Fraction of tile pixels with DN < (image_mode − 20) |
| `shadow_fraction_strict` | float | Fraction with DN < (image_mode − 35); separates true shadows from dark terrain |
| `bright_cap_fraction` | float | Fraction with DN > (image_mode + 30); sunlit boulder tops |

**LBP (Local Binary Patterns)** — 10 columns

Rotation-invariant uniform LBP (`skimage.feature.local_binary_pattern` with `P=8, R=1,
method='uniform'`), producing 10 distinct labels (0..9 = P+2). Per-tile histogram is
normalized to sum to 1.

| Column | Type | Meaning |
|---|---|---|
| `lbp_hist_0` .. `lbp_hist_9` | float | Normalized count of LBP-label-k pixels in the tile. Sum = 1.0 (modulo float roundoff) |

**Lacunarity** — 2 columns, S ≥ 32 only (NaN at S=8, S=16)

Gliding-box lacunarity on the shadow mask: `L(b) = E[M²] / E[M]²` where M is the sum of
shadow pixels inside a b×b sliding box. L=1 means uniform shadow distribution; L>1 means
clustered/gappy.

| Column | Type | Meaning |
|---|---|---|
| `lacunarity_shadow_b2`, `lacunarity_shadow_b4` | float | Lacunarity at gliding-box sizes 2 and 4 CTX pixels. **NaN when the tile has no shadow pixels** — there is no gliding-box statistic to compute, and lacunarity is ≥ 1 by Cauchy–Schwarz, so no in-range value can encode "not computable" |

> **R28 / 2026-08-06.** Tiles with `shadow_fraction == 0` used to emit **`0.0`**, an
> out-of-range sentinel: 42,015 of 198,320 S ≥ 32 rows in `dataset_v2/features/`, every one
> with `shadow_fraction == 0`, smallest non-zero value exactly 1.0, and nothing in `(0, 1)`.
> Stage 6a's neighbour aggregation is NaN-aware but not sentinel-aware, so it averaged the
> sentinel in with real measurements: **2.16 %** of `nbr_mean_lacunarity_*` rows pooled
> (worst image `ESP_068402_2240` at **16.7 %**) sit in the impossible interval `(0, 1)`.
> The producer now emits NaN. **`dataset_v2/features/**` and every `features_nbr_*` derived
> from it still carry the old sentinel** until the batched rebuild — see
> [docs/PENDING_REBUILD.md](../docs/PENDING_REBUILD.md).

**Subtile variance** — 1 column, S ≥ 16 only (NaN at S=8)

Variance of the 4 sub-block means within each tile (sub-block side = S/2). Captures
internal heterogeneity that single-tile std misses. Free given the nested ×2 ladder.

| Column | Type | Meaning |
|---|---|---|
| `intensity_subtile_var` | float | Variance of `[(top-left mean), (top-right mean), (bottom-left mean), (bottom-right mean)]` |

**Canny edges** — 2 columns, S ≥ 16 only (NaN at S=8)

Canny edges computed once over the full CTX window (`sigma=1.0`). Per-tile reductions:

> **R28 / 2026-08-06 — read this before using `edge_*` for science.** "skimage-default
> thresholds" means the **absolute constants 0.1 / 0.2** on the `img_as_float` image, *not*
> thresholds derived from this image's gradient distribution (`config.yaml` used to claim
> the opposite). So `edge_density` partly measured how much radiometric contrast the CTX
> frame happens to have: across the 38-image cohort, per-image `edge_density` tracks
> per-image `intensity_std` at Spearman **ρ = 0.965** with a **12.2×** spread, and
> **33.8 %** of `ESP_068402_2240`'s S = 64 tiles have zero Canny edge pixels. On a synthetic
> scene, cutting the DN spread ~3× collapses edge density 100-fold.
>
> The shipped configs now use `canny_edges.use_quantiles: true` with `0.80 / 0.90` —
> **percentiles of each frame's own gradient magnitude**, which is gain-invariant (×1.00 on
> the same test). **Every `edge_*` value currently in `dataset*/features/**` is still the
> old absolute-threshold version** and changes at the batched rebuild; see
> [docs/PENDING_REBUILD.md](../docs/PENDING_REBUILD.md).

| Column | Type | Meaning |
|---|---|---|
| `edge_density` | float | Canny-edge pixels / tile pixels (in [0, 1]) |
| `edge_orientation_entropy` | float | Shannon entropy of edge-pixel gradient orientations, binned over `[0, π)` in 8 bins. 0 for tiles with no edges; up to log(8) ≈ 2.08 for perfectly isotropic edges |

**Context patch references** — 2 columns when `features.context_patch.enabled` is true,
absent otherwise

| Column | Type | Meaning |
|---|---|---|
| `patch_idx_S32` | int32 | Row index into `dataset/context_patches/{ObsId}_S32.npy`. -1 if the tile is too close to the window edge for a centered 32-px patch |
| `patch_idx_S64` | int32 | Row index into `dataset/context_patches/{ObsId}_S64.npy`. -1 ditto |

### `{ObsId}.json` (sidecar)
| Field | Type | Meaning |
|---|---|---|
| `obs_id` | str | HiRISE Observation ID |
| `n_tiles_total` | int | Total rows in the parquet (matches Stage 4's `sum(eligible_tiles_per_scale)`) |
| `per_scale_tile_counts` | obj | `{tile_size_px: row_count}` per scale |
| `enabled_features` | list[str] | Feature families actually computed (subset of `intensity_stats`, `glcm`, `gradient`, `shadow_fraction`, `lbp`, `lacunarity`, `subtile_variance`, `canny_edges`) |
| `ctx_window_tif`, `hirise_mask_tif` | str | Absolute paths of the Stage 2 inputs this feature run consumed |
| `labels_parquet` | str | Absolute path of the Stage 4 labels parquet whose rows this feature run mirrors |
| `mosaic_row_origin`, `mosaic_col_origin` | int | Carried from the labels sidecar so feature consumers can reconstruct (ti, tj) → mosaic-pixel without joining |
| `dn_thresholds` | obj | `{mode, shadow, shadow_strict, bright, method}` — per-image absolute DN cuts derived from `np.bincount` on HiRISE-covered pixels. `method` is `dn_mode_offset` normally or `image_percentile_fallback` for windows with < 1000 covered pixels (ESP_057469_2215 class) |
| `glcm` | obj or null | `levels_per_scale`, `distances_per_scale`, `angle_average`, `properties`, `max_distances_in_schema`, `nan_fill` |
| `lbp` | obj or null | `method`, `P`, `R`, `n_bins` |
| `lacunarity` | obj or null | `box_sizes_px`, `min_tile_size_px` |
| `context_patch` | obj | `{enabled, sizes_px, patch_files (absolute paths), patch_counts, patch_bytes_estimate}` when patches were emitted; `{enabled: false}` otherwise |
| `timings_per_image_seconds` | obj | Wall-clock for each per-image artifact (`dn_thresholds`, `gradient_window`, `lbp_window`, `canny_window`, `glcm_quantize`) |
| `timings_per_scale_seconds` | obj | `{tile_size_px (str): {feature_family: seconds}}` — per-scale GLCM is the bottleneck (~5–28 s per image total) |
| `parquet_path` | str | Absolute path of the feature parquet |
| `config_hash` | str | Provenance |
| `written_at_iso` | str | When the parquet was written (UTC ISO) |

## Stage 4b — `dataset/context_patches/`

Raw CTX uint8 chips centered on each emitted tile's center, bundled per (ObsId, patch
size) into a single `.npy` stack instead of per-tile files (DECISIONS.md 2026-05-23
deviation from PLAN_Stage4b.md §6: 1.3M individual files would be NTFS-hostile and slow
to scan; 18 bundled files use `np.load(..., mmap_mode='r')` for the CNN DataLoader path).

### `{ObsId}_S{patch_size}.npy`
Uint8 array of shape `(n_valid_patches, patch_size, patch_size)`. `n_valid_patches` may
be less than the row count of the feature parquet — tiles within `patch_size // 2` of
the CTX-window edge can't fit a centred patch and get `patch_idx_S{patch_size} = -1` in
the feature parquet rather than a row in the .npy. The shortfall is small (41 of 643,910
at S=32; 402 at S=64 across the priority10 sweep).

To load a specific patch by features-parquet row `i`:

```python
import numpy as np, pandas as pd
df = pd.read_parquet("dataset/features/ESP_069669_2220.parquet")
patches = np.load("dataset/context_patches/ESP_069669_2220_S64.npy", mmap_mode="r")
idx = int(df.iloc[i]["patch_idx_S64"])
patch = patches[idx]  # (64, 64) uint8 view; copy if you'll mutate
```

## PLAN_FM — `dataset_v2/fang_embeddings/`

Frozen Fang-ViT embeddings, one bundled `.npz` per (ObsId, input size). Written by
`scripts/probes/_w2_fang_embed.py`; the productized extraction/inference path that
*reproduces* them bit-for-bit is `src/fm_embeddings.py` (parity asserted by
`scripts/probes/_fm_parity_check.py`). These are the feature source of the frozen
recipe (DECISIONS.md 2026-06-12): `mlp_ens3` on the S=32 96-px 3×3-context
GeM(p=3) embedding, emb-only, `fa_gt_1e-2`.

### `{ObsId}_P{px}.npz`
`px` encodes the **input** size, not the tile size: `P96` = the S=32 3×3-context input
(the frozen one), `P32` = the S=32 own-tile input, `P192`/`P64` = the S=64 3×3-context/
own-tile inputs. Each tile's box is bicubic-resized to 224 and normalized `(x/255−0.5)/0.5`
before the ViT. Arrays, all row-parallel to that image's tile keys at the matching scale:

| array | dtype | shape | meaning |
|---|---|---|---|
| `ti`, `tj` | int32 | `(n,)` | mosaic-anchored tile indices (join key with `obs_id`) |
| `valid` | bool | `(n,)` | False where the context box spilled past the CTX-window edge |
| `cls` | float32 | `(n, 768)` | ViT [CLS]-token embedding |
| `mean` | float32 | `(n, 768)` | mean-pooled patch tokens |
| `gem` | float32 | `(n, 768)` | **GeM(p=3)** patch tokens — the frozen pooling |

Invalid rows carry the raw ViT output in the npz but are set to **NaN** on load
(`src.modeling.loaders.load_fang_store`) so the head imputes them; at S=32/P96 coverage
is 100% (the 96-px ring fits inside the window buffer for every tile). Join onto packaged
folds with `augment_fold_with_fang(fold, px=96, replace=True)` (emb-only) — the lookup is
keyed one-to-one on `(obs_id, ti, tj)`, never positional. Loader column names are
`fang_{pool}{px}_{000..767}` (e.g. `fang_gem96_000`).

## Stage 5 — `dataset/splits/`

Group-aware leave-image-out split metadata. One JSON file per named scheme; multiple
schemes coexist (the modeler picks one at training time). Splits are over **images**,
never tiles -- random per-tile splits leak per-image background into the test fold
(CLAUDE.md acceptance #5).

### `{name}.json`

| Field | Type | Meaning |
|---|---|---|
| `name` | str | Scheme name (e.g. `loio_9fold`, `loio_3fold_balanced`) |
| `kind` | str | Always `"leave-image-out"` today |
| `n_folds` | int | Number of folds |
| `stratification` | str | `"none"` (LOIO, requires n_folds == n_images) or `"boulder_label_size_balanced"` (greedy size-balanced k-fold within label groups) |
| `seed` | int | RNG seed for the deterministic shuffle inside stratified assignment. Recorded even when unused |
| `manifest_obs_ids` | list[str] | The ObsIds the scheme operates on -- sorted, deduped, derived from `dataset/labels/*.parquet` on disk at build time |
| `folds[].fold_idx` | int | Zero-based fold index |
| `folds[].test_obs_ids` | list[str] | ObsIds in this fold's test set |
| `folds[].train_obs_ids` | list[str] | ObsIds in this fold's train set (complement of test within `manifest_obs_ids`) |
| `folds[].test_summary` | obj | `{n_images, n_tiles_total, n_tiles_finest, boulder_labels: {label: count}, frac_mean_finest_avg}` for the test side |
| `folds[].train_summary` | obj | Same shape, for the train side |
| `config_hash` | str | Provenance: SHA256 of the config snapshot that produced this split |
| `split_hash` | str | SHA256 over `{name, kind, n_folds, stratification, manifest_obs_ids, folds}` -- a stable id for the split assignment, independent of timestamps |
| `written_at_iso` | str | UTC ISO timestamp |

## Stage 5 — `dataset/packaged/{name}/`

Per-fold train/test parquets ready for training, materialised from the split + label +
feature inputs. Produced by `package_split` (in-memory concat path). For the 50-200+
image case, the streaming `iter_train_batches` / `iter_test_batches` API yields per-
ObsId DataFrames without materialising the full dataset (see `src/dataset.py`).

### `X_{train,test}_fold{k}.parquet`
Feature side. Schema: tile-key columns + every column in `dataset/features/{ObsId}.parquet`
except `obs_id`/`scale_idx`/`tile_size_px`/`ti`/`tj`/`config_hash` (those are kept as
join keys). Row order: ObsIds in the order they appear in `metadata.folds[k].{train,test}_obs_ids`,
then per-image row order from the source parquet.

| Column | Type | Meaning |
|---|---|---|
| `obs_id`, `scale_idx`, `tile_size_px`, `ti`, `tj` | various | Join key (matches `y_*_fold{k}.parquet` row-for-row) |
| (everything else) | float / int / str | Feature columns -- see the Stage 4b feature-parquet schema above |

### `y_{train,test}_fold{k}.parquet`
Label side. Same tile-key columns + the label transforms + per-tile bound context:

| Column | Type | Meaning |
|---|---|---|
| `obs_id`, `scale_idx`, `tile_size_px`, `ti`, `tj` | various | Join key (identical to X) |
| `boulder_area`, `boulder_count`, `tile_area` | float / int | Base stats (Stage 4) |
| `fractional_area` | float | Primary regression target (Stage 4) |
| `binary_by_area`, `binary_by_count` | bool | Binary targets at the config thresholds |
| `count_density` | float | `boulder_count / tile_area` |
| `categorical` | Int64 | Emitted only when `labeling.categorical_bins` is non-empty (absent today) |
| `xmin`, `ymin`, `xmax`, `ymax`, `tile_size_m` | float | Per-tile bound context (handy for heatmap plotting; not a label) |

### `groups_{train,test}_fold{k}.npy`
Int32 array, one entry per row in the matching X/y parquet. Value = `obs_id`'s integer
code from `metadata.json::obs_to_int`. Use with `sklearn.model_selection.GroupKFold` etc.
for intra-train CV that respects the image-group structure.

### `all.parquet` (when `splits.emit_all_parquet=true`)
Consolidated view: every tile appears exactly once, tagged with the `fold_idx` of the
test fold it belongs to. Useful for ad-hoc analysis ("per-fold target distribution",
"per-fold-per-image variance", etc.) without repeated joins.

| Column | Type | Meaning |
|---|---|---|
| (all columns from the per-image label + feature join) | -- | Same shape as concatenating X and y on the join key |
| `fold_idx` | int | Which fold this tile lands in as the *test* tile. For a per-fold training set, filter by `fold_idx != k` |

### `metadata.json`
| Field | Type | Meaning |
|---|---|---|
| `name` | str | Scheme name |
| `split_hash` | str | Mirrors the split JSON's `split_hash`; mismatch indicates the package and split metadata are out of sync |
| `config_hash` | str | Provenance |
| `scale_filter` | list[int] or null | If non-null, only these `tile_size_px` values were included in the packaging |
| `emit_all_parquet` | bool | Whether `all.parquet` was written |
| `obs_to_int` | obj | `{obs_id: int}` mapping used by `groups_*.npy` |
| `per_fold` | list[obj] | One entry per fold: `{fold_idx, n_train_tiles, n_test_tiles, n_train_x_cols, n_y_cols, test_obs_ids}` |
| `all_parquet_path` | str or null | Absolute path of `all.parquet` if emitted |
| `written_at_iso` | str | UTC ISO timestamp |
