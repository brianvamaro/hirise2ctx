# Review area: features

- **Reviewed at commit:** da884c7
- **Date:** 2026-07-31
- **Verification:** self-refuted (single-agent pass; not independently verified)
- **Pass note:** `features-1` .. `features-5` are carried forward unchanged from the first
  self-refuted pass (they are the source of register entries **R27** / **R28**; do not
  renumber them). `features-6` and the extra items in *Refuted by my own check* /
  *Verified clean* are new in a second, independent pass that re-read all four modules and
  re-measured the empirical claims. The second pass **confirmed** every first-pass claim it
  re-tested and found no first-pass claim to retract.

## Findings

### features-1 — `lacunarity_shadow_b*` emits `0.0` — an out-of-range sentinel — on 21.2 % of S≥32 tiles, and Stage 6a averages it as if it were a real value
- **Severity:** medium
- **Liveness:** dead-closed for the shipped map (FM-embedding recipe is emb-only), but live for every GBM/W1 number ever reported off `dataset_v2/features/`
- **Confidence:** high
- **Where:** `src/features.py:422` (producer), `src/spatial_features.py:100-105,110-144` (consumer), `dataset/DATA_DICTIONARY.md:278-284` (documentation)

Gliding-box lacunarity satisfies `L = E[M²]/E[M]² ≥ 1` by Cauchy–Schwarz, so the whole
defined range is `[1, ∞)`. When a tile has no shadow pixels at all (`M1 == 0`) the code
returns `0.0` instead of `NaN`. Everywhere else in Stage 4b, "not computable" is encoded as
`NaN` (the S<32 gating at `src/features.py:392`, GLCM distance padding at `:457`), and
Stage 6a's neighbour aggregation is explicitly *NaN-aware but not sentinel-aware*
(`finite_vals = np.isfinite(vals)` at `src/spatial_features.py:100`). The `0.0` therefore
passes the finite test and is arithmetically averaged with genuine `≥1` values.

- **Failure scenario:** A shadow-free S=64 tile gets `lacunarity_shadow_b2 = 0.0`. Its
  8-neighbour window contains 3 shadow-free and 6 clustered-shadow tiles, so
  `nbr_mean_lacunarity_shadow_b2` returns a value that no real lacunarity mean can take.
  Measured on the committed caches:
  `dataset_v2/features/*.parquet` → 42 015 / 198 320 = **21.185 %** of S≥32 rows have
  `lacunarity_shadow_b2 == 0.0`; every single one of them (1774/1774 in
  `ESP_017355_2260`) has `shadow_fraction == 0`, the minimum non-zero value is exactly
  `1.0`, and **no** row lies in `(0, 1)` — i.e. `0.0` is provably a sentinel, not a
  computed value. Downstream, `dataset_v2_dev/features_nbr/ESP_055978_2270.parquet` has
  **12.61 %** of S≥32 rows with `nbr_mean_lacunarity_shadow_b2` in the impossible interval
  `(0, 1)` and a minimum of `-1.46e-13`. The base column is largely survivable for
  LightGBM (0.0 is separable by a split), but the Stage-6a mean/std columns are
  irrecoverably contaminated: no split can undo an average of a sentinel and a
  measurement.
- **Evidence:**
  ```
  src/features.py:420-422
              M1 = box_sums.mean()
              M2 = (box_sums ** 2).mean()
              out[col][i] = (M2 / (M1 ** 2)) if M1 > 0 else 0.0

  src/spatial_features.py:100-105
          finite_vals = np.isfinite(vals)
          valid = np.zeros((n_ti, n_tj), dtype=np.float64)
          valid[rr, cc] = finite_vals.astype(np.float64)
          grid = np.zeros((n_ti, n_tj), dtype=np.float64)
          grid[rr[finite_vals], cc[finite_vals]] = vals[finite_vals]

  dataset/DATA_DICTIONARY.md:280-284
  shadow pixels inside a b×b sliding box. L=1 means uniform shadow distribution; L>1 means
  clustered/gappy.
  | `lacunarity_shadow_b2`, `lacunarity_shadow_b4` | float | Lacunarity at gliding-box sizes 2 and 4 CTX pixels |
  ```
- **Self-refutation attempted:** (a) grepped `DECISIONS.md` for `lacunarity` — the only hits
  are the Stage-4b design note (`:591`, `:601`), the test list (`:665`) and the
  2026-06-10 dead-shadow finding (`:2720`); the `0.0`-vs-`NaN` choice is recorded nowhere,
  so it is not a deliberate decision. (b) Checked `tests/test_features.py:350-368` — both
  lacunarity tests use non-empty masks (`test_lacunarity_on_uniform_shadow_mask_equals_one`
  uses an **all-ones** mask, i.e. the opposite degenerate case), so nothing pins `0.0` as
  intended behaviour. (c) Checked whether it is unreachable — it is 21 % of the rows.
  (d) Checked whether it is out of scope because the shipped map is emb-only — but the
  column is in every packaged `X_*` parquet (`src/modeling/loaders.py:83-95` drops only tile
  keys and `patch_idx_S*`), so the banked GBM baseline and the W1 dossier consumed it.
- **Fix:** `out[col][i] = (M2 / (M1 ** 2)) if M1 > 0 else np.nan` (the array is already
  pre-filled with `np.nan` at `:392`, so the `else` branch can simply be dropped), and add
  the "NaN when the tile has no shadow pixels" case to `DATA_DICTIONARY.md:284`.
  Regenerating Stage 6a afterwards is required for the `nbr_*_lacunarity_*` columns.

### features-2 — Canny thresholds are a fixed fraction of the dtype range, not adaptive; the config says the opposite, and `edge_density` dies on low-contrast images
- **Severity:** medium
- **Liveness:** dead-closed for the shipped map; live for the GBM feature matrix and the W1 error atlas
- **Confidence:** high (mechanism), medium (magnitude of the effect on any single reported number)
- **Where:** `src/features.py:199-208`, `config.yaml:149-150`, `dataset/DATA_DICTIONARY.md:297`

`_compute_canny_window` passes `low_threshold=None, high_threshold=None` straight through to
`skimage.feature.canny`. I read the installed skimage (0.x in env `geospatial`) source: with
`use_quantiles=False` (the default, never overridden), `None` maps to the **constants
0.1 / 0.2** applied to the image after `img_as_float`, i.e. an absolute gradient threshold
in DN units — it is *not* derived from the image's gradient distribution. The config comment
asserts the opposite. Consequently `edge_density` / `edge_orientation_entropy` measure "how
much radiometric contrast this CTX frame happens to have" as much as "how rough this tile
is" — the exact per-frame-radiometry-into-features mechanism the striping programme spent
months on, never audited on the hand-crafted features.

- **Failure scenario:** A low-gain CTX frame (`ESP_068402_2240`, per-image mean
  `intensity_std` = 7.87 DN at S=64) produces `edge_density` mean 0.0254 and **33.9 %** of
  its S=64 tiles with *zero* Canny edge pixels in 4096 CTX pixels; a high-gain frame
  (`ESP_059686_2235`, `intensity_std` = 22.7) produces mean 0.3075 and 0 % dead tiles — a
  12× cohort spread with Spearman ρ = **0.894** between per-image `edge_density` and
  per-image `intensity_std` across the 38 v2 images. Two features are therefore
  identically-zero on a third of one image's tiles, which is precisely the "dead feature
  across a whole image" failure the project found and fixed for `shadow_fraction`
  (DECISIONS.md 2026-06-10, worth +0.249 / +0.127 meaningful AUC on the two affected
  images) — but the canny family was never checked for the same thing.
- **Evidence:**
  ```
  src/features.py:200-208
      from skimage.feature import canny
      edges = canny(
          arr,
          sigma=float(cfg["sigma"]),
          low_threshold=cfg.get("low_threshold"),
          high_threshold=cfg.get("high_threshold"),
      )

  config.yaml:149
      low_threshold: null            # None -> skimage chooses from gradient magnitude

  installed skimage/feature/_canny.py (read via inspect.getsource):
      if low_threshold is None:
          low_threshold = 0.1
      elif use_quantiles:
          ...
      else:
          low_threshold /= dtype_max
  ```
- **Self-refutation attempted:** (a) grepped `DECISIONS.md` for `canny` / `edge_density` —
  only the Stage-4b design description at `:587`/`:605`; nothing records the threshold
  semantics or a dead-feature audit. (b) Tried to kill the "it's the threshold, not the
  terrain" reading: the ρ=0.894 with `intensity_std` is confounded because rougher terrain
  is both genuinely edgier and higher-variance. That confound is real and I state it —
  but the *code* is decisive on the mechanism (a fixed absolute cut), and 34 % of tiles
  with literally zero edges in 4096 px is threshold starvation, not edge-free terrain.
  (c) Checked whether "Bet 1 (per-image feature standardization)" (DECISIONS 2026-06-11)
  already covers it — that bet z-scored the *emitted feature values* per image, which
  cannot resurrect a feature that is identically zero before standardization.
- **Fix:** Either pass `use_quantiles=True` with explicit quantile thresholds, or derive
  `low/high_threshold` per image from the window's gradient-magnitude percentiles (the same
  per-image-adaptive pattern `_compute_dn_thresholds` already uses), and record the chosen
  cuts in the provenance sidecar next to `dn_thresholds`. At minimum, fix
  `config.yaml:149` and `DATA_DICTIONARY.md:297` to state that the cut is absolute.

### features-3 — the degenerate-window fallback in `_compute_dn_thresholds` lacks every protection the main path has, and can still return `shadow = 0`
- **Severity:** low
- **Liveness:** live-shipped path (invariant 7: a new manifest row must flow end-to-end), unreached by the current cohort
- **Confidence:** high
- **Where:** `src/features.py:143-153`

The 2026-06-10 DN-clip fix hardened the main path in three ways: exclude `DN <= 1` from the
histogram (`:142`), and fall back to percentiles if the cut still lands at the clip floor
(`:159-165`). The `covered.size < 1000` branch above it has none of them: it reverts to
`arr.ravel()` (unfiltered — includes DN=0 mosaic nodata *and* the DN=1 bottom-clip spike the
fix was written to exclude), reports the **median** in a field named `mode`, and has no
`shadow <= _DN_CLIP_FLOOR` guard, so `shadow` can be `0` and `shadow_fraction`,
`shadow_fraction_strict` and both `lacunarity_shadow_b*` columns are then identically zero
for the entire image — the exact regression the fix targeted.

- **Failure scenario:** A new manifest row whose Stage-2 window fell back to the nominal
  footprint (the case the branch's own comment names) has a nearly-empty HiRISE mask. With
  the mask covering mostly bottom-clipped terrain, `covered.size < 1000` fires,
  `np.percentile(all_vals, 10)` returns 0 or 1 because DN=0/1 pixels dominate the window,
  and `shadow = 0` → `(arr < 0)` never fires → four feature columns silently dead
  image-wide. Stage 4b still exits 0 and writes a complete-looking parquet; the only signal
  is `"method": "image_percentile_fallback"` buried in the sidecar.
- **Evidence:**
  ```
  src/features.py:142-153
      covered = covered[covered > _DN_CLIP_FLOOR]
      if covered.size < 1000:
          # Degenerate window -- empty/near-empty HiRISE coverage; fall back to global pcts.
          all_vals = arr.ravel()
          mode = int(np.median(all_vals))
          shadow = int(max(0, np.percentile(all_vals, 10)))
          shadow_strict = int(max(0, np.percentile(all_vals, 5)))
          bright = int(min(255, np.percentile(all_vals, 95)))
          return {"mode": mode, "shadow": shadow, "shadow_strict": shadow_strict,
                  "bright": bright, "method": "image_percentile_fallback"}
  ```
- **Self-refutation attempted:** (a) Checked reachability on the real cohort — re-measured
  in pass 2 across **all three** caches: `dataset/features` 9/9, `dataset_v2/features`
  38/38, `dataset_v2_dev/features` 5/5 sidecars all report `method: dn_mode_offset`
  (v1 `shadow ∈ [57, 146]`, v2 `shadow ∈ [51, 149]`), so neither fallback branch is
  currently reached by any shipped cache. (b) Checked whether the one image that hits it is
  permanently excluded — `ESP_057469_2215` is in `EXCLUDED_FROM_SWEEP`
  (`src/features.py:91`), which is why I rate this low, not medium. (c) Checked
  `tests/test_features.py` — `test_dn_threshold_percentile_fallback_when_mode_is_dark`
  (`:279-291`) exercises the *second* fallback only; nothing exercises this branch, so the
  regression suite would not catch it. It survives because invariant 7 requires new
  manifest rows to work, and this is the branch they would land in.
- **Fix:** In the degenerate branch, compute the percentiles over
  `arr[arr > _DN_CLIP_FLOOR]` rather than `arr.ravel()`, apply the same
  `max(_DN_CLIP_FLOOR + 1, ...)` floor used at `:162-163`, and rename the returned `mode`
  key or set it to `None` so the sidecar does not label a median as a mode. While there,
  add the third method string `percentile_fallback_low_mode` (`src/features.py:165`) to
  `DATA_DICTIONARY.md`'s `dn_thresholds` row, which currently documents only two of the
  three possible values.

### features-4 — Stage 6b joins the SeamMap to the CTX window with no CRS check; the CRS it reads is discarded
- **Severity:** low
- **Liveness:** dead-closed (Stage 6b was a strict FAIL and was never promoted)
- **Confidence:** high
- **Where:** `src/ctx_source_illumination.py:346-356` and `:173`; `scripts/run_stage6b.py:104-119`

`load_window_metadata` reads and returns `window_crs_wkt`, and **nothing in the repo ever
reads it** (`grep -rn window_crs_wkt` returns exactly the one definition site). The spatial
join at `:173` is `seam_gdf[seam_gdf.intersects(bbox)]` where `bbox` is a bare shapely
`box` built from the window's affine — shapely carries no CRS, so geopandas performs no
check and emits no warning. The module docstring asserts "No reprojection required for the
spatial join" as an established fact; the code never verifies it at runtime, which
contradicts CLAUDE.md's VERIFY-AT-RUNTIME rule for exactly this class of assumption (and
DECISIONS.md:260 records that Murray tiles ship an *oblate* Mars_2015 CRS, not a pure
sphere).

- **Failure scenario:** A future Murray tile (or a re-extracted SeamMap) whose `.prj`
  differs from the cached window's CRS produces a silently wrong polygon subset. If the
  offset is large the subset is empty and all seven `ctx_*` columns come out NaN; if it is
  moderate, the wrong CTX source frames are attributed to each tile and
  `ctx_incidence_mean` / `ctx_n_sources` / `ctx_dominant_source_fraction` are quietly wrong
  — which is the substrate of the H3 anti-signal claim. The run still prints a healthy
  `n_sources_window` and exits 0.
- **Evidence:**
  ```
  src/ctx_source_illumination.py:350-356
      with rasterio.open(ctx_window_tif) as src:
          return {
              "window_transform": src.transform,
              "window_h": int(src.height),
              "window_w": int(src.width),
              "window_crs_wkt": src.crs.to_wkt() if src.crs else None,
          }

  src/ctx_source_illumination.py:173
      subset = seam_gdf[seam_gdf.intersects(bbox)].reset_index(drop=True)
  ```
- **Self-refutation attempted:** (a) Checked the real outputs — all 3 564 767 rows of
  `dataset_v2/features_ctx_illum/*.parquet` have finite `ctx_incidence_mean` and
  `ctx_n_sources >= 1`, so the assumption held for this cohort; that is why this is low and
  not medium. (b) Checked `tests/test_ctx_source_illumination.py` — the fixtures build the
  GeoDataFrame with `crs="EPSG:4326"` (`:52`) while the affine is a bare metre grid, i.e.
  the tests pass *with* a deliberately mismatched CRS, confirming nothing checks it.
  (c) Checked `run_stage6b.py`'s provenance — `window_crs_wkt` is not even recorded there,
  so there is no post-hoc audit trail either.
- **Fix:** In `add_ctx_source_illumination_features`, take the window CRS as an argument and
  raise if `seam_gdf.crs` is not equivalent to it (or reproject); record `window_crs_wkt`
  and `seam_crs_wkt` in the Stage 6b sidecar.

### features-5 — `inference.py`'s "the seam is genuinely clean" claim about Stage 4b is false in both halves
- **Severity:** low
- **Liveness:** dead-closed stub, but it is the written contract for any future off-HiRISE deployment of the feature-based model
- **Confidence:** high
- **Where:** `src/modeling/inference.py:32-35`, contradicted by `src/features.py:141,590`

The stub documents the off-HiRISE feature extractor as trivially separable because "Stage
4b's shadow detector already uses image-percentile thresholds (no label leak), so the seam
is genuinely clean." Both clauses are wrong. The primary method is `dn_mode_offset`
(percentiles are only the two fallbacks, and all 38 cohort sidecars report
`dn_mode_offset`), and the mode is computed over `arr[mask == 1]` — the **HiRISE coverage
mask**, which by definition does not exist off-HiRISE. `shadow_fraction`,
`shadow_fraction_strict`, `bright_cap_fraction` and both `lacunarity_shadow_b*` columns all
hang off that one number.

- **Failure scenario:** Someone implements the off-HiRISE extractor on the stated contract,
  computes the DN mode over the whole Murray window instead of a HiRISE-covered subregion,
  and every shadow-family feature shifts by the difference between the two supports'
  modal DN. The model is applied out of distribution with no error and no warning — the
  same train/deploy-mismatch class that made F pilot leg A fail (DECISIONS 2026-07-04).
- **Evidence:**
  ```
  src/modeling/inference.py:32-35
  src.features that operates on a CTX raster without consuming labels. Stage 4b's
  shadow detector already uses image-percentile thresholds (no label leak), so the
  seam is genuinely clean; what's missing is the wrapper script and a Murray Lab
  mosaic-tile iterator.

  src/features.py:141      covered = arr[mask == 1]
  src/features.py:590      dn_thresholds = _compute_dn_thresholds(arr, mask, cfg["shadow_fraction"])
  ```
- **Self-refutation attempted:** (a) Checked whether the stub is truly dead — yes,
  `predict_over_ctx_region` raises `NotImplementedError` and the shipped map goes through
  `src/mapping.py` + FM embeddings, so no current number is affected. (b) Checked whether
  the claim is defensible as "no *label* leak" — that half is fine; the false half is
  "genuinely clean", which asserts extractor parity that does not exist. It survives
  because the sentence is the only written spec for that seam.
- **Fix:** Replace the sentence with the actual constraint: the per-image DN mode must be
  computed over a support that is reproducible off-HiRISE (e.g. the whole window), and the
  training-side features must be regenerated under the same rule before any off-HiRISE
  deployment of the feature-based model.

### features-6 — Stage 4b locates every tile in the CTX window using an origin it copies from the *labels* sidecar and never checks against the window it actually opens
- **Severity:** low
- **Liveness:** live path (Stage 4b is the documented "re-run features only" seam, and the promoted `dataset_v2/features/` cache was produced by it); has not fired on any committed cache
- **Confidence:** high (mechanism + quantified tolerance), high (it has not fired)
- **Where:** `src/features.py:578-583`, `:653-663`; contrast `src/ctx_source_illumination.py:359-373` (`mosaic_origin_pixels`) + `scripts/run_stage6b.py:107-109`

`stage4b_one_image` takes `mosaic_row_origin` / `mosaic_col_origin` verbatim from
`dataset/labels/{ObsId}.json` and then converts every tile index to a window-pixel slice
with `r_win = ti*S - mosaic_row_origin`. It opens the window GeoTIFF but reads **only band
1** — `grep -n "transform" src/features.py` returns no hit outside docstrings, so the
window's own affine is never consulted. The sibling Stage-6b module solves exactly this
problem the safe way (`mosaic_origin_pixels(window_transform, mosaic_transform)`
"Mirrors the calculation in `src.labeling._compute_grid_alignment` so this module doesn't
depend on Stage 4 internals"), and Stages 6a/6b additionally hash their input parquet
(`source_sha256_short`). Stage 4b has neither a recomputation nor a hash nor a
shape/extent check, even though its own docstring makes "reads existing caches only" the
whole point of the stage.

- **Failure scenario:** Stage 2 is re-run for an ObsId (a changed `ctx_retrieve.buffer_m`,
  a detection-footprint change from a re-run Stage 1, or a real-vs-nominal footprint
  fallback flip) and produces a window whose mosaic origin moved by δ pixels, but Stage 4
  is *not* re-run — the documented and encouraged workflow, since Stage 4b exists precisely
  so features can be recomputed without Stages 1–4. Every feature is then read from mosaic
  row `ti*S + δ` while the label for that row still describes `ti*S`, i.e. the whole feature
  matrix is silently misregistered from its target by δ·5 m. The only guard is the
  bounds assertion at `:656-663`, which merely checks the *shifted* slice still lies inside
  the array; with `ctx_retrieve.buffer_m: 1000` (config.yaml:51 / config_v2.yaml:76) the
  window carries ~200 CTX pixels of margin beyond the footprint, so shifts up to ~**1 km**
  pass silently for interior tiles. The run exits 0 and the parquet looks complete; the
  symptom would appear only as an unexplained drop in every reported metric.
- **Evidence:**
  ```
  src/features.py:576-583
      labels_df = pd.read_parquet(labels_parquet)
      labels_prov = json.loads(labels_sidecar.read_text(encoding="utf-8"))
      mosaic_row_origin = int(labels_prov["mosaic_row_origin"])
      mosaic_col_origin = int(labels_prov["mosaic_col_origin"])
      ctx_window_tif = Path(labels_prov["ctx_window_tif"])
      mask_tif = Path(labels_prov["hirise_mask_tif"])
      arr, mask = _load_window_and_mask(ctx_window_tif, mask_tif)

  src/features.py:653-663
          r_win = (ti * S - mosaic_row_origin).astype(np.int64)
          c_win = (tj * S - mosaic_col_origin).astype(np.int64)
          # Sanity: Stage 4 guarantees tiles fit entirely inside the window.
          if not (
              (r_win >= 0).all() and (c_win >= 0).all()
              and (r_win + S <= H).all() and (c_win + S <= W).all()
          ):
              raise RuntimeError(...)

  src/features.py:108-120   (_load_window_and_mask -- band read only, no transform/crs)
      with rasterio.open(ctx_tif) as src:
          arr = src.read(1)
  ```
- **Self-refutation attempted:** (a) **Measured whether it has already fired.** For all 38
  `dataset_v2/labels/*.json` I recomputed the origin independently from the cached window
  GeoTIFF's own affine plus the matching `cache_v2/ctx_tiles/*.json` `inner_transform`
  (i.e. re-implemented `mosaic_origin_pixels`): **38/38 exact integer match, 0 mismatches**
  — so no committed feature number is affected, which is why this is low and not medium.
  (b) Checked whether the bounds assertion is actually sufficient — it is not: it constrains
  `r_win ∈ [0, H-S]`, which a δ-shift inside the 1 km buffer satisfies. (c) Checked whether
  the labels sidecar carries anything Stage 4b *could* cross-check without extra I/O — it
  records `finest_grid_cells` and the two origins but **not** the window's `height`/`width`
  or transform (`src/labeling.py:552-590`), so even a shape comparison is unavailable today;
  the labels parquet does however carry per-tile `xmin/ymin/xmax/ymax` in mosaic CRS
  metres, which makes the fix nearly free. (d) Grepped `DECISIONS.md` for
  `mosaic_row_origin` / `grid_anchor` — the convention is recorded, the absence of a
  verification is not recorded as a deliberate choice. (e) Checked the tests — the
  synthetic fixture writes labels and window together, so no test can exercise a
  disagreeing pair.
- **Fix:** In `stage4b_one_image`, after opening the window, recompute the origin from
  `src.transform` + the Murray tile's `inner_transform` (reuse
  `ctx_source_illumination.mosaic_origin_pixels`) — or, cheaper and dependency-free, assert
  that `window_transform.c + c_win[0]*px == labels_df["xmin"].iloc[0]` for one tile per
  scale — and raise if it disagrees with the labels sidecar. Also record the window's
  `height`/`width` (and ideally a short hash) in the Stage 4 sidecar and echo them in the
  Stage 4b provenance.

## Refuted by my own check

*(First-pass items, all re-checked and still refuted.)*

- **CTX nodata (DN=0) contaminating intensity/shadow/GLCM statistics.** Stage 4b applies no
  CTX-validity mask (only the HiRISE coverage mask), and `shadow_mask = arr < shadow` would
  count DN=0 nodata as shadow — but I measured every committed feature parquet
  (`dataset/`, `dataset_v2_dev/`): **zero** rows have `intensity_min == 0`, and the global
  minimum DN over all labelled tiles is 1. The Murray mosaic has no gaps inside these
  HiRISE footprints. (`src/mapping.py:78 own_tile_zero_fraction` guards the inference side
  separately.)
- **Stale pre-fix shadow thresholds in the reported cohort.** `dataset_v2_dev/features/
  ESP_064510_2260.json` still carries the pre-fix `{'mode': 1, 'shadow': 0}` (written
  2026-05-29) and its 98 736 rows have `shadow_fraction` identically 0 — but that is a
  superseded *dev* cache. Re-measured in pass 2: all 38 sidecars in `dataset_v2/features/`
  (`shadow ∈ [51, 149]`) and all 9 in `dataset/features/` (`shadow ∈ [57, 146]`) are
  post-fix. `ESP_064510_2260` in `dataset_v2_dev` is the **only** flagged sidecar anywhere.
  No reported number is stale.
- **`select_feature_columns` aggregating the target.** Stage 6a auto-selects every numeric
  column, which would create `nbr_mean_fractional_area` (catastrophic leakage) if the
  labels were present — but `run_stage6a.py:60,70` reads
  `dataset_dir/features/{obs}.parquet`, which is the label-free Stage 4b output. Clean.
- **GLCM `correlation` NaN→0 fill mangling constant tiles.** The docstring
  (`src/features.py:98-101`) and `DATA_DICTIONARY.md:244` both claim skimage emits NaN on
  constant tiles and that the column "can be 0" there. The installed skimage sets
  `correlation = 1` when the marginal std is < 1e-15; I verified directly
  (`graycoprops → [[1.]]`) and on real data (all 315 zero-contrast S=8 tiles in
  `ESP_017355_2260` have `glcm_correlation_d1 == 1.0` exactly). `_GLCM_NAN_FILL` is
  effectively dead for that case and the column is *correct*; only the two documentation
  lines are wrong. Cohort-wide `correlation == 0.0` is 0.025 % of S=8 rows and those tiles
  have non-zero contrast, i.e. they are genuine zeros.
- **`edge_orientation_entropy == 0` conflating "no edges" with "perfectly aligned edges".**
  Real, but small: 0.948 % of S≥16 rows have `entropy == 0` *and* `edge_density > 0`. Not
  worth a finding on its own; it is subsumed by features-2.
- **`valid_pixel_fraction` being a constant-1.0 column that reaches the model matrix.**
  It does (loaders keep it), and it is 1.0 on all 3 564 767 v2 rows — but
  `FeatureScaler.fit` guards with `self.sd[self.sd == 0] = 1.0`
  (`src/modeling/mlp_head.py:83`) and LightGBM ignores zero-variance features. No effect.
- **`ctx_n_sources` defaulting to `0` (a valid-looking value) where every other Stage-6b
  column defaults to NaN.** A genuine missing-as-zero conflation in
  `src/ctx_source_illumination.py:259` (and the `_aggregate_per_tile` docstring at `:247`
  says out-of-window tiles "get NaN", which is false for this one column), but zero rows in
  either `features_ctx_illum/` cache are affected (0 / 3 564 767 with `ctx_n_sources == 0`).
- **`mosaic_origin_pixels` drifting from `labeling._compute_grid_alignment`.** Compared
  line by line (`src/ctx_source_illumination.py:367-373` vs `src/labeling.py:133-139`) —
  identical arithmetic, including the `round()` and the `e`-sign convention.
- **`bright_cap_fraction` dying when `mode + 30 > 255`.** The `min(255, ...)` clamp makes
  `arr > 255` unsatisfiable, i.e. the same dead-feature class as the shadow bug with no
  guard — but the maximum modal DN across all 38 images is 169 (re-measured: `bright`
  spans 101–199 in v2, 107–196 in v1), so the branch needs `mode ≥ 226` and is unreachable
  in any plausible CTX scene.
- **HiRISE COLOR per-band scaling.** `parse_color_lbl` takes only the first token of
  `SCALING_FACTOR`/`OFFSET`, which would silently mis-scale if they were per-band tuples.
  I read `cache_v2/hirise_color/ESP_017355_2260_COLOR.LBL:135-136` — both are single
  scalars for the 3-band cube (`SCALING_FACTOR = 2.10297581310192e-04`,
  `OFFSET = 0.041517186439203`), so `mean(DN)·a + b = mean(I/F)` is exact and the band
  ratios in `run_stage7c_features.py:206-208` are genuine I/F ratios. `CORE_NULL = 0`
  confirms `region_means`' `> valid_min` nodata rule.
- **Colour tile-bbox over-approximation.** `ctx_bounds_to_source_bbox` is documented as
  over-approximating by "a few pixels" — in fact both CRSs are equirectangular, so the
  composite CTX→source map is affine and the four-corner bbox is *exact*. Docstring is
  conservative, code is right.
- **Context-patch index / row-order desync.** `_build_context_patches` assigns
  `len(stack)` before `append` and `stage4b_one_image:770` re-walks
  `sorted(tiles_by_scale)`, which is the same order as
  `labels_df.groupby("tile_size_px", sort=True)` used to build the frames. Verified
  consistent; `test_stage4b_context_patches_bundle_indices` pins the count.

*(New in pass 2.)*

- **`nbr_std_*` is silently NaN for some windows with exactly 2 valid neighbours.**
  `count_win` is reconstructed as `uniform_filter(valid, size=k)*k²`, which is **not**
  integer-exact: over 300 random 7×9 validity grids, 380 windows with a true count of 2
  produced `count_win < 2.0` (minimum `1.9999999999999978`), so
  `np.where(count_win >= 2.0, std_win, np.nan)` (`src/spatial_features.py:143`) drops them.
  Mechanism confirmed; impact measured and negligible — reconstructing exact neighbour
  counts from the `(ti, tj)` grids of the committed caches gives **1 affected row out of
  505 475** in `dataset_v2_dev/features_nbr` (stencil 3) and **0 out of 783 591** in the
  promoted `dataset_v2/features_nbr_s5` (stencil 5, the variant actually used for v2).
  Also noted: the docstring/comment says "sample std is undefined when only one valid
  neighbour exists" but the code computes the **population** std (`ex2 - mean²`, ddof=0),
  which is what `tests/test_spatial_features.py:75,98,127` pins — the comment is a
  mislabel, not a bug.
- **Stage-4b origin has already drifted on a real cache.** The empirical half of features-6:
  independently recomputed `mosaic_row_origin/col_origin` from each cached window's own
  affine + the Murray tile `inner_transform` for all 38 v2 images → 38/38 exact match. The
  hazard is latent, not realised.
- **GLCM `KeyError` at the Phase-C S=128 scale.** `levels_per_scale` / `distances_per_scale`
  are indexed directly by `S` (`src/features.py:730-731`) with no `.get`, so a scale present
  in `labeling.tile_sizes_px` but absent from the GLCM config would crash Stage 4b.
  `config_v2_dev.yaml:56,78,79` declares `128` in all three lists, so the Phase-C S=128 run
  is safe and `max_distances` stays 3 (no schema inflation). Refuted.
- **Half-tile / off-by-one between the labels grid and the features grid.**
  `src/labeling.py:165-168` (`r0_win = j*S_min - mosaic_row_origin`) and
  `_flatten_to_dataframe`'s `ti_abs` + `xmin = mx_origin_x + tj*S*px` both define tile
  `(ti, tj)` at scale `S` as mosaic rows/cols `[ti*S, (ti+1)*S)`; `src/features.py:653-654`
  uses the identical convention, and the ×2 ladder's `j_min_row // (S//S_min)`
  (`src/labeling.py:330-331`) keeps the nesting exact. No offset.
- **Context-patch geometry parity with the frozen FM recipe.** `_build_context_patches`
  centres the patch on `r_win + S//2` with top-left `− P//2`, so for `P == S` it returns
  exactly `arr[r_win:r_win+S, c_win:c_win+S]`. `scripts/probes/_w2_fang_embed.py:228-233`
  independently pins this at extraction time (`big[:, TILE_PX:2*TILE_PX, ...]` must equal
  the cached own-tile patch, asserted on 16 sampled rows per image) and `:198` asserts no
  `-1` rows reach the embedder. `src/modeling/loaders.py:414-432` also drops `-1` rows
  rather than letting `-1` index the last patch. The `-1` sentinel has no unguarded
  consumer in `src/`.
- **Band ratios computed on DN instead of I/F in Stage 7c.** Independently re-read
  `scripts/run_stage7c_features.py:186-208`: the per-band `mean_DN * scaling_factor +
  offset` conversion is applied *before* `ir/red`, `ir/bg`, `red/bg`. Dropping the
  `OFFSET` (0.0415, ~20 % of a typical I/F) would have biased every Stage-7c/7d ratio; it
  is not dropped. Clean.
- **`read_color_window` array-vs-transform sub-pixel offset.**
  `rasterio.windows.from_bounds` yields fractional `row_off/col_off`;
  `rasterio.windows.transform` keeps them fractional while `ds.read(window=...)` truncates
  them to int, so the returned `(arr, transform)` pair can be up to 1 colour pixel (0.25 m)
  out of register. Only `scripts/probes/_stage7_feasibility.py:146-149` rasterises a polygon
  mask into that grid (Test A's interior-vs-ring paired spectra); the effect there dilutes
  the interior/ring contrast, i.e. biases **toward the null**, so it cannot have manufactured
  the feasibility PASS. The shipped Stage 7c path passes an all-ones mask
  (`run_stage7c_features.py:176-182`), so it is unaffected. Not filed.
- **`read_color_window`'s documented `(None, None)` return on a non-intersecting window.**
  `Window.intersection` raises `WindowError`, it does not return an empty window, so the
  docstring at `src/colour.py:144-145` is wrong — but every caller in the repo goes through
  `windowed_colour_read`'s bbox short-circuit (`:236-238`) or pre-pads a bbox known to
  intersect, so the path is unreachable. Docstring-only.
- **`grad_dir_circvar == 1.0` on a zero-gradient tile.** `weights = mag/(mag.sum()+1e-12)`
  collapses to all-zero on a constant tile, giving `R = 0` and `circvar = 1.0` — the same
  "undefined encoded as an extreme in-range value" class as features-1, but 1.0 is inside
  the documented `[0, 1]` range and the population is the ~315 zero-contrast S=8 tiles per
  image, so it cannot move a metric. Not filed.
- **`run_stage6a.py` overwriting a variant run.** `--stats` and `--stencil-size` are not
  encoded in the output directory (only the optional `--output-suffix` is), so a stencil-5
  run without a suffix would overwrite the stencil-3 `features_nbr/`. Checked the real
  caches: every variant lives in its own directory and every sidecar records its
  `stencil_size` (`dataset_v2/features_nbr_s5` → 5 (the promoted v2 variant),
  `dataset_v2_dev/features_nbr` → 3, `_s5` → 5, `_max` → 3 with `stats: ['max']`). No
  collision has occurred, and artifact-path collisions are the `other-scripts` area's remit.
- **`_aggregate_per_tile` reusing the INCIDENCE finite-mask to index the other three angle
  rasters.** `rasterize_seam_map_window:186-190` filters `np.isfinite(v)` **per angle
  column**, so a SeamMap row with a finite INCIDENCE but a non-finite EMISSION would make
  `emi[valid]` contain NaN and `ctx_emission_mean` NaN (a loud-ish NaN, not a wrong value).
  Checked every Stage-6b sidecar's `finite_counts_per_column`: all four `*_mean` counts are
  equal in 38/38 v2 and 5/5 dev files, so no cohort polygon has a partially-finite angle
  row. Not filed.
- **`_deep_merge_defaults` merging a config's `levels_per_scale` instead of replacing it.**
  A config that lists only some scales silently inherits the default entries for the others.
  Harmless: lookups are by `S`, so inherited entries for absent scales are never read, and
  `max_distances` is unchanged for every shipped config (all declare `[1,2,3]` at every
  scale ≥ 16).
- **`_quantize_for_glcm` on a non-power-of-two `levels`.** `256 // levels` would give an
  uneven top bin (e.g. `levels=10` → values 0..10 clipped to 9). All three configs use only
  8 / 16 / 32, which divide 256 exactly. Latent only.

## Verified clean

- `_lacunarity_per_tile`'s integral-image box-sum algebra (padded `(H+1, W+1)` prefix sums,
  `r0 ∈ [r, r+S-b]`, the four-corner formula) — correct, and the max index `r+S ≤ H` never
  overruns.
- `_subtile_variance_per_tile`'s `reshape(n, 2, sub, 2, sub)` → `mean(axis=(2,4))` — the
  axis order really does give the four (S/2)² block means.
- `_quantize_for_glcm` for all configured level counts (8/16/32 all divide 256 exactly, so
  the integer division is exact and the `np.clip` is belt-and-braces).
- `_lbp_hist_per_tile` normalization (sums to 1 for `P=8, method='uniform'` → labels 0..9,
  `int8` cannot overflow at this P; `local_binary_pattern` returns exact integers so the
  `astype(np.int8)` truncation is lossless).
- `_intensity_stats_per_tile` skew/kurtosis `np.where(var > 0, ..., 0.0)` guard under
  `errstate` — no NaN escapes; float64 promotion before the moments; `intensity_iqr` really
  is `p75 − p25` from the same `np.percentile` axis-fold.
- `_canny_per_tile`'s `np.mod(d, pi)` fold onto `[0, π)` and the `0·log 0` handling
  (`np.histogram` with explicit `bin_edges` cannot drop a value, and `hist.sum()` is
  guarded non-zero by the `d.size == 0` early-continue).
- `_gradient_stats_per_tile`'s doubled-angle magnitude-weighted circular variance
  (`1e-12` denominator guard present).
- `_glcm_per_tile`'s distance↔column mapping: `graycoprops` returns
  `(n_distances, n_angles)`, `vals.mean(axis=1)` is indexed by `k_idx` in
  `enumerate(distances)`, so `glcm_*_d{d}` always carries the property at pixel distance
  `d`; the NaN-padded schema is stable across scales.
- `stage4b_one_image`'s tile-inside-window assertion (`:655-663`) — fails loudly on a
  gross Stage-2/Stage-4 cache mismatch rather than reading out of bounds (but see
  features-6 for what it does *not* catch).
- `_deep_merge_defaults` against both `config.yaml` and `config_v2.yaml` `features:` blocks
  — no key in either config is absent from `DEFAULT_FEATURES_CFG`, and the YAML integer
  scale keys match the dict lookups at `:730-731`.
- Feature column-order determinism: per-scale frames are concatenated in ascending
  `tile_size_px` (`groupby(sort=True)`), the packaged X/y split takes `x_cols` from
  `train_df` and indexes `test_df` with the same list (`src/dataset.py:663-664`), and
  `loaders._feature_columns` preserves the parquet order. No dict-iteration dependence.
  Stage 6a's appended columns are likewise deterministic
  (`itertools.product(stats, feature_cols)` insertion order, filled positionally).
- The empty-input paths: `labels_df` with no eligible tiles leaves `features_df` empty and
  the `per_scale_tile_counts` comprehension is short-circuited by the `if len(features_df)`
  guard; `_glcm_per_tile` returns the full NaN schema at `n == 0`; `_aggregate_one_group`
  returns zero-length arrays in the full schema.
- `src/spatial_features.py`'s NaN-as-gap semantics, edge padding, `count_win > 0` mean
  guard (exact at a true count of 0), `-inf` max sentinel, and positional row-order
  preservation — all correct and all pinned by `tests/test_spatial_features.py`.
- `src/colour.py` band constants, `geometry_mask` inversion polarity in `polygon_masks`
  (`~geometry_mask(..., invert=False)` = inside), the `outer.difference(inner)` annulus
  (metre buffers in a metre CRS), `region_means`' shared valid-mask across the three bands
  (so the ratios are consistent), and the SP1-corrected-CRS routing in
  `run_stage7c_features.py` (`corrected_source_crs` → `pyproj.Transformer` → source-CRS
  bounds), which matches the documented requirement in `read_color_window`'s docstring.
- Stage 4b consumes only `scale_idx / tile_size_px / ti / tj` from the labels parquet — no
  label column (`fractional_area`, `boulder_count`, …) touches any feature computation, so
  there is no label→feature leak at this seam.

## Coverage note

Read in full: `src/features.py` (872), `src/spatial_features.py` (213), `src/colour.py`
(267), `src/ctx_source_illumination.py` (373), `tests/test_features.py` (533),
`tests/test_spatial_features.py` (268), `tests/test_colour.py` (148),
`tests/test_ctx_source_illumination.py` (273), `scripts/run_stage4b.py`,
`scripts/run_stage6a.py`, `scripts/run_stage6b.py`, `scripts/run_stage7c_features.py`,
`src/modeling/inference.py`, `dataset/DATA_DICTIONARY.md` §Stage 4b, the `features:` and
`labeling:` blocks of `config.yaml`, `config_v2.yaml` and `config_v2_dev.yaml`. Pass 2
additionally read `src/labeling.py:88-410` (grid alignment, ×2 ladder, flattening — for the
labels↔features convention check), `src/modeling/loaders.py:29-105,392-432`,
`scripts/probes/_w2_fang_embed.py` (the context-patch consumer),
`scripts/probes/_stage7_feasibility.py:100-175`. Grepped (not read in full):
`DECISIONS.md` by term (`lacunarity`, `canny`, `edge_density`, `064510`, `oblate`,
`nodata`, `mosaic_row_origin`, `grid_anchor`), `src/dataset.py`, `src/mapping.py` (nodata
handling only), `notebooks/_build_07.py` / `_build_08.py` (patch-index use only).

Empirical checks were run **read-only over already-committed derived artifacts** — the
`dataset*/features*/**.parquet|json` and `dataset*/labels/*.json` caches, the
`cache_v2/ctx_tiles/*.json` sidecars, the *headers only* (`transform`/`height`/`width`, no
pixel read) of the cached `cache_v2/ctx_windows/*.tif`, and one `cache_v2/hirise_color/*.LBL`
text label — plus `inspect.getsource` on the installed skimage and two pure-numeric scipy
snippets (the `uniform_filter` exactness probe). No imagery pixels were decoded, no network
access, no notebook/sweep/training/map run.

Could **not** check: (1) whether any *reported* GBM metric actually moves when the
lacunarity sentinel becomes NaN or when the Canny threshold becomes adaptive — that needs a
re-sweep, which is out of scope here; (2) the two slow integration tests in
`tests/test_features.py` (`@pytest.mark.slow`, real Stage 2/4 caches); (3) `src/colour.py`'s
`polygon_masks` / `read_color_window` behaviour on a real COLOR.JP2 (the sub-pixel
transform/read offset above is reasoned from rasterio's semantics, not executed);
(4) whether a Stage-2 re-run can in practice move a window origin for an *unchanged*
manifest row — that is `geo-crs`'s `ctx_retrieve.py` remit, so features-6 is stated as a
missing verification rather than a demonstrated shift.
