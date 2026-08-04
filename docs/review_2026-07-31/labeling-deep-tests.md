# Review area: labeling-deep-tests

- **Reviewed at commit:** 7bfedb8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified). Every claim below is
  backed by a **mutation experiment**: `src/` was copied into the session scratchpad, a defect was
  introduced into the *copy*, and the real `tests/test_labeling.py` was run against it. "SURVIVED"
  means the suite stayed green with the defect present. No file in `tests/` was modified.

> **Headline answer to the area's question.** The labelling tests do **not** pin wrong science — I
> found no assertion that defends a known defect. But they pin far less than they appear to: **16 of
> 20 seeded defects survive `pytest -m "not slow"` (the documented dev loop) and 12 of 20 survive the
> full suite including both slow real-data tests.** The root cause is a single fixture choice
> (`_make_window` puts the CTX window origin exactly on the mosaic origin) that **no production image
> shares — 0 of 47**. Separately, and more urgently, the two `slow` tests in this area *write into the
> live artifact tree* and I triggered that today.

---

## Findings

### labeling-deep-tests-1 — The two `slow` tests in this area regenerate live gitignored artifacts in `dataset/` and `cache/`; running the suite silently overwrites the provenance an audit reads, and the rewrite is **not** value-preserving
- **Severity:** high
- **Liveness:** live-shipped (the v1 `dataset/` + `cache/` trees; `dataset_v2/` and `cache_v2/` are untouched — verified)
- **Confidence:** high (reproduced: I caused four overwrites at 14:26 today, with mtimes, `config_hash` and a measured content diff)
- **Where:** `tests/test_labeling.py:576-596` (`output_dir=cfg.output_dir`), `tests/test_labeling.py:633`;
  `tests/test_empty_shapefile.py:16-28` (`cache_dir=cfg.cache_dir`); producers
  `src/labeling.py:543,591` and `src/detections.py:151-173`. Same shape at
  `tests/test_sanity_residual_one_image.py:35` and `tests/test_stage2_one_image.py:32,105`.

`stage4_one_image` and `stage1_one_image` have no dry-run mode: they write to
`{output_dir}/labels/{obs}.parquet|.json` and `{cache_dir}/reprojected_detections/{obs}.gpkg|.json`
from whatever paths the caller hands them. Both slow tests hand them the **live** `cfg.output_dir`
(`<repo>/dataset`) and `cfg.cache_dir` (`<repo>/cache`) rather than a `tmp_path`. So merely *running
the test suite* regenerates real project artifacts. These paths are gitignored, so git cannot restore
them, and there are no backups.

The rewrite is not a no-op. I compared the freshly regenerated
`dataset/labels/ESP_069669_2220.parquet` against `dataset/packaged/loio_9fold/y_test_fold6.parquet`
(mtime 2026-05-23 15:42, built from the original labels, **not** touched today). The tile *sets* are
identical at every scale, but the **values differ**: `max|Δfractional_area|` = 0.115 at S=8 / 0.029 at
S=32, and `max|Δboulder_count|` = 35 at S=8 / 115 at S=64. The v1 labels were last built 2026-05-23,
i.e. **before the 2026-06-10 coregistration y-sign fix**; today's code produces post-fix labels. So
the write did not "refresh a stale timestamp" — it silently migrated one of nine v1 images across a
known correctness boundary, leaving `dataset/labels/` **mixed-vintage** (8 files at
`config_hash=e9962e94…`, 1 at `958fdc25…`).

This is not new behaviour caused by the review: `cache/reprojected_detections/ESP_069669_2220.json`
already carries `config_hash=958fdc25…` with mtime **2026-06-10 17:32**, while its eight siblings carry
`e27f940c…` from 2026-05-20/21 — the signature of an earlier slow-suite run (via
`test_sanity_residual_one_image.py:35`). The schema differs too: the old sidecars have no
`n_polygons_raw`/`n_dropped_null_geometry` fields, the rewritten ones do. **The test suite has been
mutating the live v1 caches for months.**

There is a second-order consequence that is squarely the **R24 pattern** the brief asked me to hunt:
`test_stage4_nested_consistency_on_real_data` (`:619-668`) reads
`cfg.output_dir/labels/{OBS_ID}.parquet` — the file the *previous test in the same file* has just
overwritten. It therefore audits a fresh regeneration, never the shipped artifact. If the on-disk
labels were internally inconsistent, this test would replace them with consistent ones and then pass.
The defect it appears to cover is destroyed before the assertion runs.

- **Failure scenario:** a reviewer (or CI, or a routine `pytest` before a commit) runs the slow suite;
  four artifacts are rewritten; the drift evidence that `labeling-deep-artifact` exists to find is
  gone, and one image's labels now disagree with the eight others and with the packaged splits and
  every downstream number computed from them.
- **Evidence:**
  ```
  tests/test_labeling.py:586-590
      prov = stage4_one_image(
          OBS_ID, cache_dir=cache_dir,
          output_dir=cfg.output_dir,          # <-- <repo>/dataset, not tmp_path
          manifest_row=row, ... )
  src/labeling.py:541-543
      parquet_path = labels_dir / f"{obs_id}.parquet"
      df.to_parquet(parquet_path, index=False)

  tests/test_empty_shapefile.py:21-28
      gdf_t, gpkg, correction = detections.stage1_one_image(
          OBS_ID, ..., cache_dir=cfg.cache_dir, ... )   # <-- <repo>/cache

  $ find cache dataset cache_v2 dataset_v2 -newermt "2026-08-04 00:00" -type f
  cache/reprojected_detections/ESP_065711_1545.gpkg      14:26:46
  cache/reprojected_detections/ESP_065711_1545.json      14:26:46
  dataset/labels/ESP_069669_2220.json                    14:26:35
  dataset/labels/ESP_069669_2220.parquet                 14:26:35
  # (exactly four; git status shows no tracked file modified)

  # regenerated vs the untouched 2026-05-23 packaged vintage, same 96,354 rows, same (ti,tj):
  #   S=  8: n=72,821  max|dfa|=1.150e-01  max|dcount|=35
  #   S= 16: n=18,043  max|dfa|=7.062e-02  max|dcount|=79
  #   S= 32: n= 4,428  max|dfa|=2.898e-02  max|dcount|=99
  #   S= 64: n= 1,062  max|dfa|=9.268e-03  max|dcount|=115
  ```
- **Self-refutation attempted:** (a) Is `load_shift` the regenerating "load" helper? **No** —
  `src/coregister.py:511-516` is a pure read returning `None` if absent, and
  `cache/coregistration/ESP_069669_2220.json` still has its 2026-06-10 mtime after ~40 of my mutant
  runs. The writers are the two Stage-1/Stage-4 producers only. (b) Is the shipped basis affected?
  **No** — `dataset_v2/labels` newest file is 2026-06-10 18:26 and `cache_v2/reprojected_detections`
  is 2026-05-28 19:00; the frozen recipe, the deployable head and the regional map all read v2 and are
  untouched. That is what bounds this at high rather than blocker. (c) Is the content loss
  unrecoverable? **Not quite** — `dataset/packaged/loio_9fold/y_test_fold6.parquet` (untouched,
  2026-05-23 15:42) contains all 96,354 original rows with every label column
  (`obs_id, scale_idx, tile_size_px, ti, tj, boulder_area, boulder_count, tile_area, fractional_area,
  binary_by_area, binary_by_count, count_density, xmin, ymin, xmax, ymax, tile_size_m`); the only
  column missing is `config_hash`, whose original value is `e9962e94…` (all eight siblings). A faithful
  restoration is therefore possible from that file. (d) Is it deliberate? Grepped `DECISIONS.md` and
  the plans for a decision to point tests at live caches — nothing; the test docstrings say
  "auto-skips when Stage 2/3 caches are missing", i.e. they treat the live cache as a *read* dependency
  and are silent about writing to it.
- **Fix:** three lines. Give the slow tests `tmp_path`-backed output: `output_dir=tmp_path` in
  `test_stage4_runs_on_ESP_069669_2220` (it only reads inputs from `cache_dir`), and
  `cache_dir=tmp_path` in `test_stage1_handles_empty_shapefile` /
  `test_stage1_centroid_residual_under_threshold` (Stage 1 reads only `detections_root` and the PDS
  `.LBL` cache — copy the `.LBL` in, or pass a separate read cache). Then make
  `test_stage4_nested_consistency_on_real_data` read the **repo's** parquet explicitly, so it audits
  the shipped artifact rather than a fresh one. Additionally restore
  `dataset/labels/ESP_069669_2220.parquet` from `y_test_fold6.parquet` if the pre-y-sign-fix v1 vintage
  is wanted intact (see `labeling-deep-artifact`).

---

### labeling-deep-tests-2 — Every end-to-end fixture pins the mosaic grid phase to **zero**, a configuration 0 of 47 production images has; the entire grid-anchoring surface is unverified, and this is the *same* fixture defect that already produced the ~100 km `fgates` mis-key
- **Severity:** high
- **Liveness:** live-shipped (`xmin/ymin/xmax/ymax` flow into `dataset*/packaged` via
  `src/dataset.py:71` and are the join key for gates 5 & 6 at `src/fgates.py:237-245`)
- **Confidence:** high (mutation-proven, with measured displacement on the real image)
- **Where:** `tests/test_labeling.py:41-48` (`_make_window`), `:462-489`
  (`test_tile_bounds_align_with_mosaic_pixel_grid`), `:262-320` (all `_sum_up_ladder` tests pass
  `j_min_row=0, j_min_col=0`); code under test `src/labeling.py:135-179, 201-206, 264, 354-370`

`_make_window` sets `mosaic_transform = list(window_transform)[:6]` — "window IS the mosaic origin in
tests" (`:48`). Consequence: in **every** test that calls `stage4_one_image`,
`mosaic_row_origin = mosaic_col_origin = r0_win = c0_win = j_min_row = j_min_col = 0`, and the mosaic
origin is the CRS origin `(0, 0)`. In production, `mosaic_row_origin` ranges 894–43,790 and
`mosaic_col_origin` 183–41,945 across the 47 label sidecars, and **not one** image has
`r0_win == c0_win == 0`. The tests exercise a grid geometry that never occurs.

`test_tile_bounds_align_with_mosaic_pixel_grid` is the specific R19 instance: its docstring says the
check is *"the definition of 'anchored to CTX pixel origin'"*, but with `mx_origin_x = mx_origin_y = 0`
the assertion `xmin % (S·px) == 0` is satisfied identically by a grid anchored to the **CRS** origin.
Deleting `mx_origin_x`/`mx_origin_y` from the bounds computation (mutant **M5**) displaces every
emitted `ymin` on the real image by up to **2,608,087 m** and the full suite — including both slow
real-data tests — stays **green (20 passed)**.

`src/fgates.py:211-231` records that this exact fixture choice already cost the project a real result:
> *"gates 5 and 6 were pairing labels with predictions ~100 km apart and publishing plausible numbers
> (on E16_N44: pooled pr_auc 0.544 / Spearman −0.180 mis-keyed vs 0.939 / +0.791 correct). … the old
> unit test could not catch it because it **pinned `row0=col0=0` and re-derived the expectation from
> the same formula**."*

The consumer was fixed; the **upstream Stage-4 tests that produce `xmin/xmax` still pin `row0=col0=0`**,
and `cohort_tiles_to_global` now depends entirely on those bounds being right.

- **Failure scenario:** any refactor of `_compute_grid_alignment` / `_flatten_to_dataframe` /
  `_rasterize_boulders_subpixel` that drops or mis-signs the mosaic phase ships green. Labels are then
  computed from CTX pixels up to 63 px (315 m) away from the tile they are attributed to, or the
  emitted world bounds are off by the Murray tile origin — and the resulting numbers are, as the
  project already learned, *plausible*.
- **Evidence (mutation results; "SURVIVED" = suite green with the defect present):**
  ```
  fast suite only  (`pytest tests/test_labeling.py -m "not slow"`, the CLAUDE.md dev loop):
    M1 r0_win drops the row phase ................................. SURVIVED
    M2 c0_win drops the col phase ................................. SURVIVED
    M3 coarse ti offset not rescaled (j_min_row // (S//S_min)) ..... SURVIVED
    M4 coarse tj offset not rescaled .............................. SURVIVED
    M5 tile bounds anchored to CRS origin ......................... SURVIVED
    M6 sub-pixel raster origin drops r0/c0 ........................ SURVIVED
    M7 mask crop drops r0/c0 ...................................... SURVIVED
                                    (16 of 20 seeded defects survived in total)

  full suite incl. both slow real-data tests (20 passed baseline):
    M1, M2 ......... killed — but by `src/labeling.py:169`'s own internal
                     `assert 0 <= r0_win < r1_win <= window_h`, NOT by a test
                     assertion (verified: AssertionError (24192, 3462, 3484))
    M3, M4 ......... killed by test_stage4_nested_consistency_on_real_data (this
                     test earns its keep)
    M5 ............. SURVIVED | max |Δymin| on ESP_069669_2220 = 2,608,087 m
    M6 ............. SURVIVED
    M7 ............. SURVIVED
  ```
  ```
  tests/test_labeling.py:47-48
      window_transform = Affine(pixel_m, 0, origin_x, 0, -pixel_m, origin_y)
      mosaic_transform = list(window_transform)[:6]  # window IS the mosaic origin in tests
  # `_make_window` takes origin_x/origin_y parameters — no caller ever passes them.
  ```
- **Self-refutation attempted:** (a) *Isn't the offset case covered by `test_alignment_offset_window`
  (`:173-186`)?* Partly — it pins `mosaic_row/col_origin` and the `j_*` range, which is why a sign flip
  there dies. But it asserts **nothing** about `r0_win/c0_win/r1_win/c1_win`, and
  `test_alignment_aligned_window` asserts only their *differences* (`:169-170`). That is exactly the
  gap M1/M2/M6/M7 walk through. (b) *Do the slow tests cover it?* For M3/M4 yes; for M5/M6/M7 no — and
  the slow tests are deselected by the documented dev loop and skip on a fresh clone (see `tests-2`,
  `tests-3`). (c) *Is the code's internal `assert` sufficient?* No: it caught M1/M2 only because the
  induced offset (24,192 px) hugely exceeds the window height (3,484 px). It is a bounds check, not a
  correctness check. (d) *Did pass 1 file this?* `labeling.md`'s "Verified clean" mentions the
  single-origin fixture in one sentence as a weakness "not filed as a finding". I am filing it,
  because the mutation results and the `fgates` precedent make it a demonstrated hole rather than a
  stylistic one. (e) Grepped `DECISIONS.md` for a decision to test at zero phase — nothing.
- **Fix:** parameterise `_make_window` with a non-zero `(origin_x, origin_y)` — e.g. a mosaic origin at
  `(-2.3e6, 2.6e6)` and a window offset of `(3, 5)` mosaic pixels, mirroring
  `test_alignment_offset_window` — and run the five `stage4_one_image` tests over both phases. Assert
  the **absolute** `r0_win/c0_win`, and change `test_tile_bounds_align_with_mosaic_pixel_grid` to
  `np.mod(xmin - mx_origin_x, step) == 0` so it tests what its docstring claims.

---

### labeling-deep-tests-3 — `boulder_count` can be identically **zero on every tile of every image** and the whole labelling suite stays green
- **Severity:** medium
- **Liveness:** live-shipped (`boulder_count` is a packaged target — `src/dataset.py:61` — and was the
  target that lifted PR-AUC +22 % on dev; `count_density` derives from it)
- **Confidence:** high (measured on the real image)
- **Where:** `tests/test_labeling.py:231-256` (the only count test), `:598-616` and `:640-668` (the two
  slow tests, neither of which asserts a non-zero count); code
  `src/labeling.py:237-251` (`_count_centroids_per_finest_cell`)

Mutant **M23** makes the centroid binner use the *window* origin instead of the mosaic origin (the
same class of error the `fgates` docstring describes) — on real data every centroid then falls outside
`[j_min, j_max]` and is filtered by `in_range`. Result: `sum(boulder_count)` for ESP_069669_2220 drops
from **5,646 to 0**, `boulder_area` and `fractional_area` are untouched, and the suite reports
**20 passed**. The synthetic count test cannot see it because the fixture's origins coincide; the
slow tests never assert that any count is non-zero; and `test_stage4_nested_consistency_on_real_data`
compares sums of children against parents, which `0 == 0` satisfies.

Two further count-specific holes in the same test: the three boulders sit at cells (0,0) and (1,1) of
an 8×8 grid, so (i) the inclusive `j_max` boundary is never exercised — mutant **M8**
(`cell_row <= j_max` → `<`, which silently discards the last finest row and column of centroids in
every image) SURVIVES the full suite — and (ii) no centroid sits on a cell edge, so the module
docstring's central claim (`src/labeling.py:47-49`: *"unambiguous at tile borders (each boulder counts
exactly once in the tile owning its centroid)"*) is asserted nowhere.

- **Failure scenario:** a refactor of the centroid binning, or of the `in_range` bounds, ships a
  systematically depleted or empty `boulder_count`; the two-stage `boulder_count` head trains on it and
  reports a metric that looks merely "worse", not "broken".
- **Evidence:**
  ```
  BASELINE green | 20 passed | ESP_069669_2220: sum(boulder_count)=5,646
  M23     green | 20 passed | ESP_069669_2220: sum(boulder_count)=0
                              (sum(boulder_area) unchanged at 46,698 m2)
  M8      green | 20 passed  (`cell_row <= align["j_max_row"]` -> `<`)
  ```
- **Self-refutation attempted:** M24 (force `centroid_counts` to zeros unconditionally) **is** killed,
  by `test_count_centroids_per_finest_cell_assigns_to_owner` and
  `test_label_transforms_emit_expected_columns` — so the counting logic is not entirely unguarded. What
  survives is the *geometry* of the counting: any error whose effect is null at zero phase, or confined
  to the region boundary, is invisible. Checked whether a downstream test covers it: grepped `tests/`
  for `boulder_count` — hits are in `test_labeling.py`, `test_modeling_binary_target.py` (synthetic
  frames) and `test_modeling_group_leak.py` (column presence only). Nothing asserts a non-zero count
  on real labels.
- **Fix:** add `assert df["boulder_count"].sum() > 0` and a per-scale total-preservation assertion
  (`sum(count at S=8) == sum(count at S=64)` over the common eligible region) to the slow test; test the
  `j_max` boundary and an on-edge centroid in the unit test.

---

### labeling-deep-tests-4 — The size-floor filter — the mechanism behind R03 — is the least-tested code in the module: the fixture cannot tell **diameter from radius**, is exercised in a geographic CRS, and no end-to-end test wires a non-`None` filter at all
- **Severity:** medium
- **Liveness:** live-shipped (`min_size_m: 1.4105` is set in both `config.yaml` and `config_v2.yaml`)
- **Confidence:** high
- **Where:** `tests/test_labeling.py:404-413`, `:143` (`_labeling_cfg` defaults both filters to `None`);
  code `src/labeling.py:96-114`, called at `src/labeling.py:471`; config rationale `config.yaml:65-72`

Three independent gaps, all in the one filter that R03 is about:

1. **Diameter vs radius is unobservable.** The fixture's areas are `[1, 100, 1000]` and the threshold
   is `5.0`. Diameter `2√(A/π)` = `[1.128, 11.284, 35.682]` → keeps 2. Radius `√(A/π)` =
   `[0.564, 5.642, 17.841]` → **also keeps 2**. Mutant **M10** (drop the factor of 2) SURVIVES the full
   suite. `config.yaml:69-71` derives the shipped value *on the explicit premise* that the code uses a
   diameter — *"`_apply_detection_filters` uses equivalent-circle diameter: `2*sqrt(1.5625/pi) = 1.4105
   m` → matches that area threshold exactly"*. Under a radius interpretation the effective area floor
   would be 4× larger (6.25 m²) — which is precisely the 0.50 m/px cohort's floor, i.e. it would
   silently convert R03's confound into a different regime, with no test failing.
2. **Units are unpinned.** The fixture builds the GeoDataFrame in `pyproj.CRS.from_epsg(4326)`, so
   `gdf.geometry.area` is in **degrees²** and geopandas emits
   `UserWarning: Geometry is in a geographic CRS. Results from 'area' are likely incorrect`
   (visible in every run of the suite, `src/labeling.py:110`). The test therefore does not pin that the
   floor is applied in metres, which is the only reading under which `1.4105` means anything.
3. **The filter is never wired end-to-end.** `_labeling_cfg()` sets `min_confidence: None,
   min_size_m: None`, so all six `stage4_one_image` tests run with filtering disabled. Mutant **M16**
   (delete the `_apply_detection_filters` call from `stage4_one_image` entirely) SURVIVES the full
   suite.

- **Failure scenario:** the size floor changes meaning (units, radius/diameter, or is dropped) and the
  cohort's label basis shifts, with the *published* `fractional_area` still looking well-formed. Given
  R03 — the two pixel-scale cohorts already have disjoint detection floors — this is the one filter
  whose semantics the project cannot afford to have unpinned.
- **Evidence:**
  ```
  tests/test_labeling.py:405-413
      crs = pyproj.CRS.from_epsg(4326)          # geographic; .area is deg^2
      # Areas: 1, 100, 1000 -> diameters: 1.13, 11.28, 35.68
      gdf = gpd.GeoDataFrame({"score": [0.5]*3},
          geometry=[box(0,0,1,1), box(0,0,10,10), box(0,0,100,10)], crs=crs)
      out = _apply_detection_filters(gdf, {"min_confidence": None, "min_size_m": 5.0})
      assert len(out) == 2      # <-- same answer under radius OR diameter

  M10 (radius instead of diameter)  -> SURVIVED (20 passed)
  M16 (filter call deleted)         -> SURVIVED (20 passed)
  ```
- **Self-refutation attempted:** (a) *Does the min-size filter matter in practice?* `DECISIONS.md:2203-2205`
  records ~0 % of v2 polygons below the floor, so M16's *live* effect today is ≈0 — that is why this is
  medium, not high. The exposure is to a *change* in the floor, which is exactly what R03's recommended
  fix (per-cohort floors) would be. (b) *Is R03 pinned as intended anywhere?* **No** — `MapPixel`,
  `mpp` and `1.4105` appear nowhere in `tests/`; the only `min_size_m` occurrences are the three
  synthetic ones above. (c) *Is the boundary tested?* No: `min_confidence` `>=` → `>` (**M9**) and
  `binary_by_area` `>=` → `>` (**M18**) both SURVIVE, so no inclusive/exclusive boundary in the module
  is pinned.
- **Fix:** choose fixture areas that separate the two interpretations (e.g. area 20 m² → diam 5.05,
  radius 2.52, threshold 5.0) in a **projected metre** CRS; and add one `stage4_one_image` test with
  `min_size_m` / `min_confidence` set, asserting the emitted `boulder_area` and
  `n_polygons_after_filter` change accordingly.

---

### labeling-deep-tests-5 — `test_stage4_runs_on_ESP_069669_2220` is a runs-not-right test: six of its seven assertions cannot fail on a wrong labeller, and its one comment overclaims
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `tests/test_labeling.py:575-616`

| line | assertion | what it can actually catch |
|---|---|---|
| 597 | `len(df) > 0` | a total crash only |
| 600 | `sizes_present.issubset({8,16,32,64})` | a change to `config.yaml`'s ladder, not a labeller defect |
| 601 | `len(sizes_present) >= 1` | nothing beyond `:597` — the comment above it says *"Every tile size must appear if any tiles are eligible"*, which is **not** what is asserted |
| 603 | `0 <= fractional_area <= 1` | nothing — union rasterisation makes `fa ≤ 1` true by construction |
| 609-612 | `tile_area ≈ s*s*px_x*px_y` | nothing — it re-derives `_sum_up_ladder`'s exact formula (`src/labeling.py:305`) from the same GeoTIFF the code reads |
| 615 | `prov["coreg_shift_applied"] is True` | **substantive** — the cached shift was found and applied |
| 616 | `prov["coreg_shift_m"] is not None` | implied by `:615` |

This is the only test that runs Stage 4 on a real image with a real coverage mask, a real non-zero grid
phase and a real co-registration shift — i.e. the only place the production configuration is
reachable — and it checks essentially nothing about the values it produces. That is why M5/M6/M7/M8/M23
above survive it.

- **Failure scenario:** as in findings 2–4: it is the test that *would* have caught them.
- **Self-refutation attempted:** its sibling `test_stage4_nested_consistency_on_real_data` (`:619-668`)
  **is** substantive and did kill M3/M4 — so the slow pair is not worthless, and I have not filed the
  pair as a unit. But note the sibling is internal-consistency only (children sum to parents), which is
  invariant under every translation-class defect.
- **Fix:** replace the tautologies with a small golden-value check — e.g. assert this image's
  `sum(boulder_area)`, `sum(boulder_count)` and S=32 rich share against banked constants (today:
  46,698 m², 5,646, 0.0011), and assert `xmin/ymin` fall inside the parent Murray tile's world bounds.

---

### labeling-deep-tests-6 — `test_empty_shapefile.py`'s CRS assertion pins nothing about the CRS, and the file pins "zero detections ⇒ every covered tile is `fa = 0`" as intended
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `tests/test_empty_shapefile.py:30-42`; `tests/test_labeling.py:537-557`

```
tests/test_empty_shapefile.py:32
    assert gdf_t.crs is not None  # reprojection assigned the target CRS
```
The comment claims the target CRS was assigned; the assertion only excludes `None`. It would pass with
any CRS whatsoever — including the source per-image local-radius CRS, i.e. the invariant-1 failure mode
(`set_crs` where `to_crs` was meant). On an **empty** GeoDataFrame the two are indistinguishable
anyway, so this file structurally cannot test reprojection. It is also entirely `slow`, so it never
runs in the documented dev loop.

Second, both this file's docstring (*"the image contributes all-zero tiles to Stage 4 labeling"*) and
`test_stage4_handles_empty_polygons` (`:537-538`: *"An ObsId with zero detections should emit
all-eligible-tile rows with zero stats"*) **pin the detector-coverage / rock-abundance conflation as
intended**: absence of detections is asserted to be ground-truth zero abundance. That is a legitimate
recorded design choice for a genuinely empty image, but it is the exact semantics
`labeling-deep-footprint` is testing, and these are the tests that would have to change if that area
concludes the conflation is unsafe. Flagging so the two areas do not contradict each other.

- **Fix:** `assert gdf_t.crs == pyproj.CRS.from_user_input(target_wkt)`; and if
  `labeling-deep-footprint` finds interior detector gaps, re-word the two docstrings so they pin
  "no detections **and** validated detector coverage ⇒ zero", not "no detections ⇒ zero".

---

## Cross-check against the confirmed findings (as the brief required)

- **R23 / `labeling-1` (score-rank truncation of two cohort images).** **Not pinned anywhere.** No test
  in either file touches shapefile integrity, record counts, `.shx`/`.shp` lengths, or the score
  distribution. `grep -rn "truncat|\.shx|017355|068483|028537" tests/` returns nothing. `_write_polygons`
  defaults every score to 0.5, and the only score-aware test is the synthetic `min_confidence` unit
  test. The tests neither defend R23 nor could have detected it — R56 (the fix blocker) is unaffected by
  anything here.
- **`labeling-2` (swath-edge zero strip from shifting polygons but not the mask).** **Not pinned, and
  structurally undetectable by these tests.** All five synthetic `stage4_one_image` tests pass
  `apply_coreg_shift=False`; the only test that combines a real shift with a real coverage mask is
  `test_stage4_runs_on_ESP_069669_2220`, whose sole shift-related assertion is
  `prov["coreg_shift_applied"] is True`. The one test with a mask hole
  (`test_mask_gating_...:327`) explicitly disables the shift. So the configuration in which
  `labeling-2` manifests is never exercised with any value assertion.
- **R03 (pixel-scale label confound / the `min_size_m` floor).** **Not pinned as intended** — see
  finding 4. `MapPixel_mpp` appears nowhere in `tests/`; there is no notion of a pixel-scale cohort in
  the test suite at all, so a per-cohort floor could be introduced without breaking a single test
  (which is good news for R03's fix).
- **The R24 pattern** (*"the test NaNs only a fold that is filtered out before the assertion, so it does
  not exercise the defect it appears to cover"*). Found **two** instances:
  1. `test_stage4_nested_consistency_on_real_data` reads the parquet that the immediately preceding
     test has just **overwritten** — the artifact it appears to audit is replaced before the assertion
     (finding 1).
  2. `test_stage4_handles_empty_polygons:557` asserts `prov["n_polygons_after_filter"] == 0` in a run
     where both filters are `None`, so the quantity is zero for a reason unrelated to filtering.
     Mutant **M19** (report the *pre*-filter count in that provenance field) SURVIVES.
- **`test_sum_up_ladder_preserves_total_area_and_count`** is a near-miss of the same shape: it sets
  `eligible = np.ones(...)`, so the interaction between the ladder sum and eligibility — the thing that
  would reveal a double-count or an omission at partial coverage — is filtered out of the fixture. It
  is not a defect (the propagation is separately tested at `:283-300`), but it is worth knowing the
  totals claim holds only for the all-eligible case.

## What the tests genuinely pin (so the next session can stop re-reading them)

These survived every attempt to kill them, and the mutants they killed are named:

- **Eligibility semantics.** A single mask-0 CTX pixel drops the containing finest tile and every coarse
  tile above it. `M13` (`min` → `max`, i.e. "any covered pixel") and `M14` (`.all()` → `.any()` in the
  ladder) are both killed. `test_mask_gating_...:327-361` and
  `test_sum_up_ladder_coarse_ineligible_if_any_subtile_ineligible:283-300`. (Caveat: at S=64 that
  window holds exactly one tile, so at that scale the assertion is satisfied by an empty frame — 63/15/3/0
  rows are emitted at S=8/16/32/64. Substantive at the three finer scales.)
- **Sub-pixel area arithmetic.** A 5×5 m polygon rasterises to exactly 25 sub-pixels at `subpixel_factor=5`
  and yields `boulder_area = 25.0 m²`, `fractional_area = 25/1600` at S=8. `M20` (area silently ×0.25)
  and `M25` (area forced to zero) are killed. `:199-215`, `:452-459`.
- **The ×2 ladder is total-preserving and correctly grouped** for an all-eligible grid, at every scale,
  for both area and count. `M3`/`M4` (coarse index offsets not rescaled) are killed *on real data* by
  `test_stage4_nested_consistency_on_real_data`. `:262-320`, `:619-668`.
- **Centroid ownership.** Each boulder counts once, in the finest cell containing its centroid; two
  boulders in one cell give 2. `M24` (counts forced to zero) is killed. `:231-256`.
- **Co-registration shift sign and no-op.** `+dx` east, `+dy` north, applied to polygons; `None` is a
  no-op. `M15` (sign flip) is killed. `:368-385`.
- **Alignment arithmetic for a *unit-level* offset window**: `mosaic_row/col_origin` and the coarsest-
  aligned `j_*` range are correct for a window starting at mosaic pixel (3,5), and a window too small
  for one coarsest tile raises `ValueError("… cannot fit …")`. `:173-192`.
- **Idempotency.** Re-running Stage 4 with the same config hash reproduces every column bit-for-bit.
  `:496-530`.
- **Column contract.** All 17 emitted columns are present and `fa ∈ [0,1]`, `tile_area > 0`,
  `boulder_area ≥ 0`, `boulder_count ≥ 0`. `:436-450`.
- **Empty-input paths do not crash**: empty gdf → all-zero raster, all-zero label rows,
  `n_polygons_after_filter == 0`; empty shapefile → empty GPKG and `qa.assert_centroid_consistent`
  returns `None` rather than raising. `:217-224`, `:537-557`, `test_empty_shapefile.py:30-42`.

## Refuted by my own check

- **"`test_nested_consistency_matches_direct_coarse_compute:317` is tautological — it copies the
  implementation's `reshape(8,2,8,2).sum(axis=(1,3))` expression."** It duplicates the line, but the
  duplicate independently encodes the *correct* grouping, so an interleaved reshape
  (`reshape(2,ny,2,nx)`) in the implementation would still be caught. Weak, not a defect.
- **"`M17`: dropping `gdf.to_crs(window_crs)` at `src/labeling.py:468-469` survives, so the reprojection
  is untested."** It survives, but `src/labeling.py:461-467` documents that the sphere and oblate
  equirectangular definitions are numerically identical at these coordinates (0.000 m displacement,
  `DECISIONS.md` 2026-05-28). The mutation is a genuine no-op today, so its survival is not evidence of
  a hole. Not filed.
- **"`M11` (`count_density` in per-km² not per-m²) and `M12` (`tile_size_m` loses the pixel size)
  survive, so two published columns are unverified."** True, and both are consumed
  (`src/dataset.py:62,71`; `scripts/probes/_evidence_gapfill_map.py:132`;
  `_modeling_slim_panels.py:47`). But a pure scale factor on `count_density` leaves every
  ranking metric invariant, and `tile_size_m` is only ever read as a display/pixel-conversion constant.
  Real but low-consequence; folded into the coverage note rather than filed.
- **"No test feeds a null or empty geometry, so the `geom is not None and not geom.is_empty` guard at
  `src/labeling.py:210` — and its absence in `_count_centroids_per_finest_cell` — is unexercised."**
  True, but `detections.drop_null_geometries` (`src/detections.py:112-127`) removes both classes at
  Stage 1 before the GPKG is written, so no null can reach Stage 4 on any real path. Latent only.
- **"`np.array_equal(..., equal_nan=True)` on bool/int columns in the idempotency test would raise."**
  It does not; the test passes on this numpy. No finding.
- **"`test_mask_gating_...:357` computes an unused `ratio`."** Cosmetic; pass 1 already noted it.
- **"`M21`/`M22` (working region slid by 1 and by 8 CTX pixels) survive."** They do **not** — both are
  killed, but by `src/labeling.py:169`'s internal bounds assert firing in the synthetic fixtures (whose
  working region exactly fills the window), not by any test assertion. Honest partial credit to the
  code, none to the tests.

## Coverage note

**Read line by line, in full:** `tests/test_labeling.py` (668) and `tests/test_empty_shapefile.py` (43),
against `src/labeling.py` (605) read in full and `src/detections.py` (218) read in full. Also read:
`src/ctx_retrieve.py:459-531` (`build_hirise_coverage_mask` — confirms the real mask is 0/1 uint8, so
the fixture's `mask_fill=1` matches production), `src/coregister.py:420-440, 505-516`,
`src/fgates.py:210-245`, `src/dataset.py:55-72`, `tests/conftest.py`, `pyproject.toml`,
`config.yaml`'s `labeling` block, `config_v2.yaml:100-105`,
`tests/test_sanity_residual_one_image.py:24-50`, `tests/test_stage2_one_image.py:28-48`, and pass 1's
`labeling.md` and `tests.md`.

**Measurements run (all read-only w.r.t. the repo after the initial incident):**
1. `pytest tests/test_labeling.py tests/test_empty_shapefile.py -q` in the repo → 21 passed in 14.3 s
   (this is the run that overwrote the four artifacts; see finding 1 — I did not repeat it).
2. **Mutation testing**: `src/` copied to the scratchpad, 25 defects seeded into the *copy*, the real
   test file run against each. 20 mutants against the fast suite (16 survived); 16 re-run against the
   full suite incl. both slow real-data tests, with `cache_dir` pointed at the repo (read-only — Stage 4
   writes only to `output_dir`) and `output_dir` pointed at the scratchpad (12 survived); 5 further
   mutants in two follow-up rounds. Every "SURVIVED" above is a green 20-passed run.
3. Grid phase over all 47 label sidecars (`dataset/labels/*.json` + `dataset_v2/labels/*.json`):
   `mosaic_row_origin` 894–43,790, `mosaic_col_origin` 183–41,945, **0 of 47** with
   `r0_win == c0_win == 0`.
4. Damage assessment: mtimes and `config_hash` across all 9 v1 label sidecars and all 10 v1 Stage-1
   sidecars; `find … -newermt` over `cache/ dataset/ cache_v2/ dataset_v2/`; and the regenerated-vs-
   packaged label diff quoted in finding 1.
5. Per-scale row counts emitted by the mask-gating fixture (63/15/3/0), and the diameter-vs-radius
   arithmetic on the min-size fixture.

**Not checked:** the other 42 test files (out of scope; `tests.md` covers their marker/skip surface, and
`tests/test_stage2_one_image.py` / `test_coregister.py` / `test_sanity_residual_one_image.py` share
finding 1's live-path defect and deserve the same treatment). I did **not** re-run the repo's slow suite
after the first invocation, and I did not run `test_empty_shapefile.py` a second time. I did not attempt
to restore `dataset/labels/ESP_069669_2220.parquet` from `y_test_fold6.parquet` — that is a write, and
the decision belongs to Brian; the recipe is in finding 1.
