# Review area: labeling

- **Reviewed at commit:** da884c7
- **Date:** 2026-07-31
- **Verification:** self-refuted (single-agent pass; not independently verified)

> **Note on provenance of this file.** An earlier pass in this same review round had already filed
> three findings here. This pass kept all three, and **replaced the mechanism of labeling-1 with a
> byte-level proof**: the losses are not "BoulderNet null-geometry records" (as `DECISIONS.md:1194`
> records) but **physically truncated `.shp` files**, and the project's own exclusion criterion was
> applied to only one of the four affected exports. labeling-4 is also new. Everything else is the
> earlier pass's work, preserved.

## Findings

### labeling-1 — Two in-cohort detection shapefiles are physically truncated; the pipeline silently absorbs the loss as "null geometry", so 11.7 % of the training/eval tiles have a multiplicatively depressed target
- **Severity:** blocker
- **Liveness:** live-shipped (the v2 labels are the training/eval basis of the frozen recipe and of the shipped mosaic map)
- **Confidence:** high (the truncation and the lost-record counts are proven from the file bytes; the size of the downstream label error is medium confidence)
- **Where:** `src/detections.py:112-127` (`drop_null_geometries`), called at `src/detections.py:202`;
  the natural integrity gate `src/manifest.py:73-93` (`find_shapefile`) checks only *how many* files
  match; consumed blind at `src/labeling.py:460`; provenance `src/labeling.py:554`;
  misattributed at `DECISIONS.md:1194-1201`; precedent at `DECISIONS.md:1190`

`DECISIONS.md:1190` excluded `ESP_028537_2270` from the manifest for exactly this defect — "`.dbf`/`.shp`
far smaller than the `.shx` record count implies" — and then, seven lines later, `DECISIONS.md:1194`
attributes the *identical* symptom in `ESP_017355_2260` and `ESP_068483_2280` to "**BoulderNet emits
many null-geometry records** at this density" and *fixes* it by dropping them. That attribution is
wrong. I scanned every `.shx` in both detection roots: **not one record in any file is a null shape**
(no record has content-length 2), and the `.dbf`s are complete (declared record count == `.shx` record
count in all 46 folders). The `.shp`s are simply cut off, and each one's **own header declares its
true length** (bytes 24-27, big-endian, in 16-bit words):

| ObsId | `.shx` records | `.shp` declares | `.shp` actual | complete records | lost | status |
|---|---|---|---|---|---|---|
| ESP_028537_2270 | 950,228 | 571,898,628 B | 58,490,431 B | 74,766 | 875,462 (**92.1 %**) | excluded from manifest ✔ |
| ESP_017355_2260 | 1,105,447 | 569,266,636 B | 214,884,317 B | 359,933 | 745,514 (**67.4 %**) | **IN COHORT** |
| ESP_046803_2325 | 658,290 | 323,962,020 B | 192,091,266 B | 367,140 | 291,150 (**44.2 %**) | excluded, unrelated reason |
| ESP_068483_2280 | 1,057,153 | 616,023,244 B | 443,015,777 B | 727,160 | 329,993 (**31.2 %**) | **IN COHORT** |
| other 36 v2 + all 7 v1 | — | == actual | — | all | **0** | clean |

My byte-level "complete records" counts are **exactly** the numbers `DECISIONS.md:1195-1197` records as
"real polygons" (359,933 and 727,160). That equality is the proof: the records geopandas reports as
null-geometry are precisely the records whose `.shx` offset lies past EOF. GDAL returns NULL for them
instead of erroring, so `drop_null_geometries` deletes them and the pipeline proceeds.

The `.dbf`s are **sorted by score descending** (record 0 → last: 0.9106 → 0.1000 for `ESP_017355_2260`),
so the lost tail is exactly the low-confidence tail. The cut lands at `score` 0.617257 for
`ESP_017355_2260` and 0.406698 for `ESP_068483_2280` — i.e. two of the cohort's 38 images are labelled
at an effective confidence floor of 0.62 and 0.41 while the other 36 sit at 0.10, and
`min_confidence: null` (`config_v2.yaml:104`) means no code equalises that.

- **Failure scenario:** on six unaffected images, the detection *area* retained above those cuts is
  22–40 % (median ≈ 0.29) at `≥0.617` and 58–74 % at `≥0.407`. So `ESP_017355_2260`'s
  `fractional_area` is plausibly **2.5–4.5× too low** and `ESP_068483_2280`'s **1.4–1.7× too low**
  relative to the cohort's basis. Those two images are 18,754 of the 161,005 S=32 tiles in
  `dataset_v2/labels` (**11.65 %**, verified), and `ESP_017355_2260` alone is 13,457 tiles — the
  largest observation in the cohort. `pr_auc@1e-2` and `precision@5%` are prevalence-dependent and the
  pooled rich share is 0.3598 (verified), so a multiplicative error on 11.65 % of tiles moves the
  reported surface; worse, the *per-image level* of the biggest image is wrong by a factor, which is
  the exact quantity the striping/F programme spent months measuring. Consistent with that,
  `ESP_017355_2260`'s `mosaic_ratio` in `reports/figures/fbuild_abort_level_vs_labels.csv` is 1.351
  (model over label) — the direction a depressed label predicts.
- **Evidence:**
  ```
  # measured from the files themselves (read-only, index/header bytes only):
  ESP_017355_2260: shp header declares 569266636 bytes; actual 214884317; deficit 354382319
     n_len2 (null-shape records) = 0        # every declared record has real geometry
     dbf nrec = 1105447 == shx nrec         # the .dbf is NOT truncated
     score by record index: 0 -> 0.910569 ... 359932 -> 0.617257 | 359933 -> 0.617257 ... 1105446 -> 0.100000
  ESP_069669_2220: shp header declares 17286228 bytes; actual 17286228; deficit 0      # healthy control

  DECISIONS.md:1190      "**`ESP_028537_2270` truncated** (`.dbf`/`.shp` far smaller than the `.shx`
                          record count implies; read fails). Unfixable upstream -> **excluded**"
  DECISIONS.md:1194-1201 "**BoulderNet emits many null-geometry records** at this density ...
                          `ESP_017355_2260` is 1.1M rows but only **359,933 real polygons** (745k null)
                          ... True per-image boulder counts span 9.6k -> 727k"   # treats kept == true

  src/detections.py:123      valid = ~(gdf.geometry.isna() | gdf.geometry.is_empty)
  src/manifest.py:82-93      matches = sorted(folder.glob(SHAPEFILE_GLOB))   # counts files, never
                             ...                                            # validates their integrity
  src/labeling.py:554        "n_polygons_stage1": int(gdf_pre_filter_n),     # post-drop, mis-named
  ```
- **Self-refutation attempted:** (a) I first tried the benign explanations — null shapes written
  deliberately, or the 2 GB `.shp` limit. Both die: there are **zero** null-shape records in any index,
  and the files are 0.2–0.4 GB. (b) NMS or a per-image detector setting: the filenames are byte-identical
  in the `ct-010-ss-256-is-1024-ov-020` parameters across all 40 folders, and a score-rank cut is not
  what IoU suppression produces. (c) A clean writer stop: `ESP_017355_2260`'s size is **odd**
  (214,884,317 B) while `.shp` records are word-aligned, so the file is cut *mid-record* — an
  interrupted write/copy. (d) Does it overturn the abort verdict? **No** — `sd(log10 mosaic_ratio)` over
  the 21 abort observations is 0.1744, 0.1755 without `ESP_017355_2260`, 0.1791 under a ×2.5 label
  correction. R10/R03 stand untouched. (e) Is `ESP_046803_2325` a third live case? No — already in
  `scripts/run_stage4.py:EXCLUDED_FROM_SWEEP`. (f) Would `min_confidence` have masked it? It is `null`
  in both shipped configs, and a *uniform* threshold cannot equalise a per-image floor of 0.62 vs 0.10.
  What survives: two images, 11.65 % of the tiles, whose label basis no code, sidecar or doc can
  distinguish from the rest — and the project's own written criterion ("truncated → exclude") was
  applied to only one of the four files that meet it, because that one happened to fail the read.
- **Fix:** three lines, in `manifest.find_shapefile` or `detections.read_detection_shapefile`: read
  `struct.unpack(">i", shp_header[24:28])[0] * 2` and raise unless it equals `os.path.getsize(shp)`;
  additionally assert `len(gdf) == (os.path.getsize(shx) - 100) // 8`. Then apply the
  `ESP_028537_2270` precedent to `ESP_017355_2260` and `ESP_068483_2280` — re-export or exclude — and
  record the decision in DECISIONS, correcting the `1194` "null-geometry" attribution. Keep
  `drop_null_geometries` (it is correct for genuinely empty geoms) but make it *warn* rather than
  silently absorb, since it is what hid this.

---

### labeling-2 — The Stage-3 co-registration shift moves the polygons but not the coverage mask, so a ~1-tile strip inside every swath edge is labelled zero by construction
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high (mechanism), medium (magnitude estimated analytically, not measured on the masks)
- **Where:** `src/labeling.py:474-478` and `src/labeling.py:85-93` (`_apply_coreg_shift`);
  mask producer `src/ctx_retrieve.py:459-531` (`build_hirise_coverage_mask`); docstring claim
  `src/labeling.py:24-28`

Stage 4 translates every detection polygon by the Stage-3 `(dx, dy)` and then gates eligibility with
the HiRISE coverage mask, which was reprojected from the **unshifted** HiRISE product. The shift is
the measured HiRISE↔CTX geolocation offset, so it applies to the whole HiRISE product — including
where HiRISE observed — not just to the boulders inside it. Because only the polygons move, a strip
of width `|shift|` on the receding side of each swath keeps `eligible = True` while no detection can
land in it any more.

Measured shifts (`cache_v2/coregistration/*.json`, 39 images): `|shift|` min 79.9 m, median 194.7 m,
max 327.3 m, with `dy` positive (northward) in 38 of 39. At S=32 (160 m tiles) the median shift is
1.2 tile rows, so roughly 1–2 rows of tiles along the southern boundary of each eligible region get
`fractional_area = 0` while their CTX texture is ordinary terrain — i.e. structured label noise
against a real feature vector, always in the same direction and always at the swath edge.

- **Failure scenario:** for a swath ~8.6 km wide (projected) with a 185 m northward shift, ≈1.6 km²
  of the ≈90 km² labelled area — ~2 % of tiles, ~60 S=32 tiles per image — is assigned a spurious
  zero. These tiles are systematically at the boundary, so they also bias any edge-vs-interior
  diagnostic and add a small negative bias to each image's label level.
- **Evidence:**
  ```
  src/labeling.py:474-478
      shift = coregister.load_shift(obs_id, cache_dir) if apply_coreg_shift else None
      gdf = _apply_coreg_shift(gdf, shift)          # polygons move

      with rasterio.open(mask_tif) as src:
          mask = src.read(1)                        # mask does not

  src/labeling.py:24-28  (docstring — states the choice but not this consequence)
      "- The Stage 3 (dx_m, dy_m) shift is **applied to the polygons** before rasterization ...
         The grid itself stays anchored to the CTX pixel origin (no resampling)."
  ```
- **Self-refutation attempted:** grepped `DECISIONS.md` for the 2026-05-23 shift decision and for
  `hirise_mask` — the decision to shift polygons rather than resample the grid is recorded, but the
  mask is never mentioned as needing the same treatment. Checked whether the mask is rebuilt after
  Stage 3: it is not — `build_hirise_coverage_mask` runs inside `stage2_one_image`, before Stage 3
  exists. Checked whether eligibility erosion would hide it: `_build_finest_stats` requires every
  mask pixel in a tile to be 1, which trims *partial* coverage but not this whole-swath translation.
  Checked whether the magnitude is negligible: at ~2 % of tiles it is small, which is why this is
  low and not medium.
- **Fix:** translate the mask by the same `(dx, dy)` before gating (a `scipy.ndimage.shift` by
  `(-dy/px_y, dx/px_x)` rounded to whole pixels, since the shifts are already quantised to CTX
  pixels), or — cheaper and strictly conservative — erode the eligible mask by
  `ceil(|shift| / px)` pixels so the affected strip is dropped instead of mislabelled.

---

### labeling-3 — The CLAUDE.md invariant-2 CRS gate has no production caller; 38 of the 39 v2 images were ingested without it
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `src/qa.py:45-116` (`assert_centroid_consistent`); the only callers are
  `tests/test_empty_shapefile.py:35`, `tests/test_sanity_residual_one_image.py:44` and
  `notebooks/01_detections_qa.ipynb` (single `OBS_ID`); `scripts/run_stage1.py:33-46` does not call it

CLAUDE.md's invariant 2 says the residual HiRISE↔CTX offset "must fail loudly" if it comes out in km,
and `PLAN_NewDetections.md:463-467` lists `qa.assert_centroid_consistent` as acceptance check #1 for
**each new/changed image** of the vClaire cohort. It was never wired into the driver that actually
ingested that cohort: `scripts/run_stage1.py` (added specifically for v2 — `DECISIONS.md:1166`) calls
`det.stage1_one_image` and prints the SP1 `correction.status`, nothing more. Notebook 01 does call the
check, but for one hard-coded `OBS_ID` (`ESP_069669_2220`, residual 1555 m). So 38 of 39 v2 images —
every label the frozen recipe and the shipped map are built on — went through Stage 1 with the
invariant-2 gate not executed.

- **Failure scenario:** a future manifest row (or a re-copied detection folder) whose `.prj` has a CRS
  pathology the `_suspect_sp1` fingerprint misses — e.g. `D_unnamed` present but SP1 within 15° of
  `CenterLat`, or a non-`D_unnamed` datum with a wrong radius — is reprojected to a wrong location and
  ingested silently; Stage 2 windows on the polygon bbox (so the window follows the error), Stage 4
  labels it happily, and the only remaining tripwire is the Stage-3 correlation peak.
- **Evidence:**
  ```
  scripts/run_stage1.py:33-46
      gdf_t, gpkg, correction = det.stage1_one_image(...)
      ...
      print(f"  {obs_id}: n_polys={len(gdf_t):>9}  {status}  [{dt:.1f}s]  -> {gpkg.name}")
      # no qa.assert_centroid_consistent anywhere in this file

  PLAN_NewDetections.md:464-467
      "1. **CRS residual O(200 m), not km** - `qa.assert_centroid_consistent` on the
         reprojected polygons vs manifest CenterLat/CenterLon (CLAUDE.md acceptance #1...)"
  ```
- **Self-refutation attempted:** grepped the whole repo for `assert_centroid_consistent` (9 hits: 3
  docs, 2 tests, the notebook, and the definition + its own docstrings) — no `src/` or `scripts/`
  caller. Tried to kill it on "the check is redundant": the Stage-3 block-median co-registration is a
  de-facto per-image O(200 m) residual check and it did lock on 38 of 39 images (peaks 0.58–0.85,
  `|shift|` 80–327 m), with the one failure (`ESP_046803_2325`, 0/210 blocks) correctly excluded.
  That is why this is **low** and not medium — the risk is a missing guard, not a demonstrated error.
  It survives because the guard is asserted in CLAUDE.md and in a plan's acceptance list as if it
  runs, and it does not.
- **Fix:** call `qa.assert_centroid_consistent` inside `det.stage1_one_image` (it already has the
  manifest row and the reprojected gdf), or at minimum in `scripts/run_stage1.py::_reproject_one`,
  and record `distance_m` in the Stage-1 sidecar so the residual is auditable per image.

---

### labeling-4 — `coregistration.enabled` is a required config key that no code reads; setting it to `false` (its shipped value) does not stop Stage 4 applying the shift
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `config.yaml:55-56` / `config_v2.yaml` (same block), `src/config.py:24`
  (`REQUIRED_TOP_LEVEL` includes `coregistration`), `src/labeling.py:474`,
  `scripts/run_stage4.py:89-92`

`config.yaml:56` reads `enabled: false   # nominal geolocation first; CLAUDE.md default`, and
`src/config.py` *requires* the `coregistration` block to exist — but validates nothing inside it and
nothing anywhere reads `enabled`. The only consumer of the block is `scripts/run_stage3.py:42`, which
pulls `fft_window_px` / `block_px` / `block_peak_min` / `min_confident_blocks`. Stage 4's actual
behaviour is "apply the shift iff a cached shift file exists" (`apply_coreg_shift=True` by default,
`coregister.load_shift` returns `None` only when the JSON is absent), and the real switch is the
undocumented-in-config CLI flag `--no-coreg-shift`. The shipped labels *are* co-registered
(`coreg_shift_applied: true` in every `dataset_v2/labels/*.json`), i.e. the opposite of what the
config states.

- **Failure scenario:** a collaborator reproducing the dataset reads `enabled: false`, believes the
  labels sit on nominal geolocation, and interprets a ~200 m HiRISE↔CTX residual as real
  mislocation — or conversely sets `enabled: true` expecting to turn co-registration *on* and sees no
  change, since Stage 3 must be run explicitly either way. Same class as the already-`DEPRECATED`
  `ctx_read` key at `config.yaml:40`, which is also still required by `src/config.py`.
- **Evidence:**
  ```
  config.yaml:55-56
      coregistration:
        enabled: false                  # nominal geolocation first; CLAUDE.md default

  # grep -rn '\["enabled"\]|get("enabled"' --include=*.py src/ scripts/ notebooks/
  #   -> 7 hits, ALL in the `features` block (src/config.py:157,165; src/features.py:567,749; ...)
  #   -> zero readers of coregistration.enabled

  src/labeling.py:474
      shift = coregister.load_shift(obs_id, cache_dir) if apply_coreg_shift else None
  ```
- **Self-refutation attempted:** grepped `coregistration` across `src/`, `scripts/`,
  `scripts/probes/`, `notebooks/_build_*.py` — the only config read is `run_stage3.py:42`, and it
  never touches `enabled`. Checked whether `run_stage3.py` early-returns on it (it does not) and
  whether Stage 4 gates on `peak_correlation` (it does not — any cached shift is applied verbatim).
  Kept at **low** because `--no-coreg-shift` does give real control and the shipped choice
  (co-register) is the scientifically correct one; the defect is that the config asserts the reverse.
- **Fix:** either honour it — `apply_coreg = not args.no_coreg_shift and bool(cfg["coregistration"].get("enabled", True))` — or delete the key and drop `coregistration` from
  `REQUIRED_TOP_LEVEL`. Same call for `ctx_read`.

## Refuted by my own check

- **`fa` is latitude-distorted by the plate-carrée CRS.** It is not: `fractional_area =
  boulder_pixel_count / (S·subpixel_factor)²` — numerator and denominator are stretched identically
  in x, so the ratio is the true ground fraction. (`src/labeling.py:274, 375`.)
- **`min_size_m: 1.4105` is applied to latitude-inflated projected areas, so the physical size floor
  varies with image latitude.** True (at 46.5°N the effective floor is ≈1.17 m, not 1.41 m), but this
  is already measured and accepted: `DECISIONS.md:2741-2751` ("true min-size floor 0.94 m vs
  1.16-1.36 … carry as a known systematic", probe `scripts/probes/_w1_latitude_distortion.py`). Same
  entry covers the `boulder_count ≥ 50` density threshold and the E-W ground-scale variation, so
  `boulder_area` / `count_density` / `tile_size_m` being *projected* rather than ground quantities is
  also on the record. Distinct mechanism from R03, but not a new finding.
- **The one-shot sign migration `scripts/probes/_w1_migrate_coreg_sign.py` could double-flip `dy` if
  re-run on freshly re-solved Stage-3 caches** (post-fix solves carry no `y_sign_fix_applied` marker,
  so the marker guard at `:21` would not skip them). It cannot: the `assert abs(rec["shift_m"]["dy"] +
  old_dy) < 1e-6` at `:27` fails loudly on an already-correct file. The migration is safe.
- **Stage 4 could read a mask whose shape/transform disagrees with the CTX window, silently
  misaligning eligibility** (`mask[r0:r1, c0:c1]` at `src/labeling.py:277` is unchecked). The producer
  forecloses it: `build_hirise_coverage_mask` allocates `np.zeros(ctx_shape)` from the window's own
  height/width and writes with the window's transform (`src/ctx_retrieve.py:505-527`). A too-small
  mask would raise on the subsequent `reshape`, not pass silently.
- **Stage 2 windows the polygon bbox, so sparse "boulder poor" images would only sample tiles near
  their boulders (selection on the outcome).** Checked all 49 `ctx_windows/*.json` sidecars: windows
  are 77–802 km² with `hirise_coverage_fraction` 0.43–0.67 even for `ESP_056165_2200` (26 polygons →
  185 km² window, 0.669 coverage). The detections span the swath in every image, so the bbox anchor
  is not selecting on the outcome. The one genuinely empty image (`ESP_065711_1545`) correctly falls
  back to `nominal_footprint_bounds`.
- **The window origin may not be an exact integer mosaic-pixel offset, desynchronising the
  window-anchored rasterisation from the mosaic-anchored centroid binning and tile bounds.**
  `_snap_bounds_to_pixel_grid` (`src/ctx_retrieve.py:340-358`) plus the integer window rounding in
  `extract_ctx_window` (`:428-432`) make the offset exact, so `int(round(...))` at
  `src/labeling.py:138-139` is lossless. (There is still no runtime assertion of it, but the
  producing path guarantees it.)
- **`inner_transform` in the Stage-2 tile sidecar might be a GDAL geotransform, making
  `mosaic_transform[2]/[5]` the wrong origin.** It is `list(src.transform)[:6]` =
  `(a, b, c, d, e, f)` (`src/ctx_retrieve.py:305`), so `[2]`/`[5]` are `c`/`f` = upper-left origin and
  `[0]`/`[4]` are the pixel sizes — exactly as `src/labeling.py:136-139, 487-495` uses them.
- **Overlapping observations would duplicate ground tiles across LOIO folds.** Only one pair of the
  39 v2 windows overlaps at all (`ESP_066634_2210`/`ESP_071093_2210`, 1.7 km² of bbox corner), and
  their emitted `(ti, tj)` sets at S=8 and S=32 are disjoint (0 shared keys). No duplicate tiles.
- **`_apply_detection_filters` silently no-ops `min_confidence` when the DBF has no `score` column
  (`src/labeling.py:106`), while the provenance still records the requested filter.** All 39 v2 GPKGs
  carry `score` with zero NaNs (the `.dbf` field list is
  `['score','cat_id','cat_name','isin_slice','is_at_edge','id']` in all 40 folders), so this is
  latent, not live — and `min_confidence` is `null` in both shipped configs.
- **Invalid/self-intersecting polygons could get a wrong `.area` and be dropped by `min_size_m` while
  still rasterising.** Possible in principle, immaterial in practice: `DECISIONS.md:1203-1205` records
  ~0 % of v2 polygons below the floor (pooled median diameter 3.4 m, p5 ≈ 1.9 m), so the filter drops
  essentially nothing.
- **BoulderNet slice-edge artifacts (`is_at_edge`, `isin_slice`) are ignored by the labeler.**
  Measured: `is_at_edge` is 0.07–0.87 % of polygons per image and `isin_slice == 0` is ≤ 5 rows per
  image. Negligible; `cat_name` is `'boulder'` for 100 % of rows, so there is no category leakage.
- **`binary_by_area` / `binary_by_count` degenerate to all-True under their defaults.** They would
  (`0.0` and `0`), but both configs set 0.005 / 1, and the config comment records both columns as
  diagnostic-only (two-stage modeling uses `fractional_area > 0` directly).
- **The `stage4_one_image` failure path in `scripts/run_stage4.py:52-60` swallows
  `RuntimeError`/`ValueError` and `main()` still returns 0.** Real, but it is the same shape as R04
  and it does print `FAILED` plus a `Skipped:` roll-up, so I did not file it separately.

## Verified clean

- **Nested ×2 ladder.** `_sum_up_ladder` (`src/labeling.py:288-336`) reduces by 2×2 with an explicit
  divisibility guard; `_compute_grid_alignment` aligns the working region to the *coarsest* scale
  first (`:146-167`), so `n_jr = (K_max_row + 1 - K_min_row) · S_max/S_min` is a multiple of 8 and
  every coarse tile has exactly 4 complete children; the per-scale absolute index
  `j_min_row // (S // S_min)` is the correct index at that scale. Totals are preserved (asserted in
  `tests/test_labeling.py:262-320`, and on real data in the slow
  `test_stage4_nested_consistency_on_real_data`).
- **"No coverage" is never conflated with "zero boulders" at the tile level.** Eligibility requires
  *every* CTX-pixel mask value inside the tile to be 1 (`src/labeling.py:277-279`), ineligible tiles
  are dropped rather than zero-filled (`_flatten_to_dataframe:358-360`), and ineligibility propagates
  upward through the ladder (`:325`). The strict `== 1.0` rule over `>= 0.95` is a recorded 2026-05-23
  decision with a correct rationale (fa is biased low under partial coverage).
- **Boulder double counting at tile borders.** Area is a union rasterisation (overlaps counted once,
  so `fa ≤ 1` by construction); count is centroid binning (each boulder in exactly one tile). The two
  rules are internally consistent and both are applied to the *same* post-filter, post-shift geometry
  set (`src/labeling.py:509-515`).
- **Filters apply identically to `fa`, `boulder_count` and `boulder_area`.** All three derive from the
  single filtered `gdf`; there is no path where one statistic sees a different polygon set.
- **Sub-pixel rasterisation transform.** `_rasterize_boulders_subpixel:201-206` builds the oversampled
  affine from the window transform and the working-region offsets, with `all_touched=False`; the 5×
  factor gives 1 m² granularity against a ~3.7 m² median boulder (recorded rationale).
- **Grid/feature registration.** `src/features.py:578-583` reads `mosaic_row_origin` /
  `mosaic_col_origin` and the two rasters straight out of the Stage-4 provenance sidecar, so features
  and labels are indexed off the same origin — no independent re-derivation to drift.
- **Runtime pixel-size guard.** `src/labeling.py:487-495` fails loudly if the window pixel size does
  not match the parent mosaic's, which is the precondition for the integer-nesting claim.
- **`src/manifest.py`.** `load_manifest` validates required columns and ObsId uniqueness;
  `find_shapefile` globs `{ObsId}/*-mask-nms.shp` and refuses to guess on 0 or >1 matches. Fully
  manifest-driven, no hardcoded image list. (Its one gap is file *integrity* — labeling-1.)
- **`src/config.py` label validation.** `tile_sizes_px` is checked to be a positive ×2 ladder
  (`:209-218`) — the precondition `_sum_up_ladder` relies on — and `grid_anchor` is re-checked at
  Stage 4 (`src/labeling.py:498-502`).
- **`src/qa.py` internals.** Reads the radius from the target CRS's own ellipsoid rather than
  hardcoding one, inverse-projects through `target_crs.geodetic_crs` (same sphere), normalises both
  longitudes to [-180, 180] before `Geod.inv`, and returns `None` (documented) rather than crashing on
  an empty gdf. The logic is right; see labeling-3 for the fact that nothing calls it.
- **`tests/test_labeling.py` asserts nothing scientifically wrong.** The centroid-ownership,
  mask-gating, ladder-total and mosaic-alignment tests all pin the behaviour this review considers
  correct. Two weaknesses worth knowing (not filed as findings): every synthetic fixture uses a
  *single* radius/datum and puts the window origin exactly on the mosaic origin
  (`tests/test_labeling.py:47-48`), so no test exercises a non-zero-phase window against a real
  mosaic sidecar; and `test_mask_gating…:357` computes an unused `ratio`.

## Coverage note

Read in full: `src/labeling.py` (604), `src/qa.py`, `src/manifest.py`, `src/config.py`, `config.yaml`,
`tests/test_labeling.py` (668), `tests/test_empty_shapefile.py`, `scripts/run_stage4.py`,
`scripts/run_stage1.py`, `src/detections.py`, `scripts/probes/_w1_migrate_coreg_sign.py`, and
`src/ctx_retrieve.py:290-545` (tile sidecar + window snapping + mask construction, as the Stage-4
inputs). Read partially: `src/features.py` (only the Stage-4b entry point, to check grid
registration), `src/coregister.py` (`load_shift`, the metre conversion and provenance),
`scripts/run_stage3.py`, `notebooks/01_detections_qa.ipynb` (the qa cell), `PLAN_NewDetections.md`
§1.1/§7/§8. Grepped `DECISIONS.md` by term for `min_size_m`, `1.4105`, `null-geometry`, `truncat`,
`ESP_028537_2270`, `ESP_017355_2260`, `distort`, `coverage_fraction`, `polygon_bbox`.

Measurements I ran (read-only; local provenance sidecars, shapefile `.shx`/`.shp`/`.dbf` **header and
index bytes only**, cached GPKG attribute tables, `dataset_v2/labels/*.parquet`, and the committed
`reports/figures/fbuild_abort_level_vs_labels.csv`; no imagery, no network): a `.shx`-vs-`.shp`
integrity scan of **all 46 detection folders** (both roots) including the `.shp` self-declared length,
the null-shape-record census and the last-complete-record index; `.dbf` header/field parse plus
seeked `score` reads at the truncation boundaries; per-image window/coverage audit; pairwise
window-overlap and shared-`(ti,tj)` test; score→count/area retention curves on six unaffected images;
per-image S=32 tile count / mean `fa` / rich share / zero share over all 38 images (161,005 tiles,
pooled rich share 0.3598); and the leave-one-out and counterfactual effect on `sd(log10 mosaic_ratio)`.

Could **not** check: whether BoulderNet's inference footprint equals the HiRISE image footprint
(would require the JP2s — the coverage mask is built from image validity, not detector coverage, so an
interior inference gap would still be labelled zero); the exact number of tiles affected by
labeling-2 (would require reading the cached `*_hirise_mask.tif` rasters — the estimate is analytic);
and whether the truncated `.shp`s can be recovered upstream (the `.shx`/`.dbf` survive intact, so a
partial recovery of the 359,933–727,160 present records is already what the pipeline reads; the
missing bytes are simply not on this disk).
