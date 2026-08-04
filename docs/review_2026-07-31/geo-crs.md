# Review area: geo-crs

- **Reviewed at commit:** da884c7
- **Date:** 2026-07-31
- **Verification:** self-refuted (single-agent pass; not independently verified)

## Findings

### geo-crs-1 — `extract_ctx_window` georeferences a silently-cropped read with the *un-cropped* window's transform, so a window overhanging its Murray tile is written kilometres off
- **Severity:** high
- **Liveness:** live-shipped (the code is the only Stage-2 window path for both `config.yaml` and
  `config_v2.yaml`; the one image that hit it is excluded, so no shipped number is wrong today)
- **Confidence:** high (reproduced from the on-disk artifacts)
- **Where:** `src/ctx_retrieve.py:433-434` (+ `:425-432`, `:442-447`), caller
  `src/ctx_retrieve.py:585`; documented-as-benign in `DECISIONS.md:410-412`

`src.read(window=…)` with `boundless=False` **crops** the window to the dataset
(`rasterio/_io.pyx:512-513` → `windows.crop()` at `windows.py:409-415`), but
`src.window_transform(window)` is called on the *original* window. When the requested window
overhangs the tile's north or west edge the crop moves the data's start while the transform keeps the
negative offset, so the output GeoTIFF contains real CTX pixels from *inside* the tile stamped with
the coordinates of the overhang. (An east/south-only overhang is not misregistered — it is silently
truncated instead, which is a lesser but still unflagged defect.) Nothing in `extract_ctx_window` or
`stage2_one_image` compares `actual_bounds` against the requested bounds, so the corruption is
recorded as provenance rather than raised.

- **Failure scenario:** `ESP_057469_2215` (v1 priority10) — a real, measured instance.
  `requested_bounds` `x ∈ [-9619.95, +1019.99]` against tile `E0_N40` whose origin is `x = 0`, so
  `from_bounds` returns `col_off = -1924`, `width = 2128`. `crop()` reads tile columns `0…204`
  (`x ∈ [0, 1020]`) while `window_transform` writes `c = -9619.95`. The cached window
  `cache/ctx_windows/ESP_057469_2215.tif` is 100 % non-zero real imagery (`mean DN 88.6`, range
  25-159) georeferenced **9 620 m (1 924 px) too far west**, and its HiRISE mask marks 917 pixels as
  covered — i.e. had the ObsId not been excluded, Stage 4 would have emitted tiles whose CTX texture
  comes from 9.6 km east of their labels. Generalised: an image straddling a tile boundary ~50/50
  keeps a normal-looking `hirise_coverage_fraction` (~0.5) and would pass every existing check while
  every tile in it is misregistered by kilometres. This is a **live invariant-7 hazard**: adding a
  manifest row whose footprint crosses a Murray tile edge silently yields wrong data, not an error.
- **Evidence:**
  ```
  src/ctx_retrieve.py:425-434
          window = from_bounds(*bounds, transform=src.transform)
          ...
          window = window.__class__(col_off=col_off, row_off=row_off, width=width, height=height)
          data = src.read(window=window)          # <- rasterio CROPS this window
          new_transform = src.window_transform(window)   # <- but the UNCROPPED one is used

  cache/ctx_windows/ESP_057469_2215.json
    "requested_bounds_target_crs": [-9619.95…, 2439362.55…, 1019.99…, 2456657.46…]
    "actual_bounds_target_crs":    [-9619.95…, 2439362.55…, -8599.96…, 2456657.46…]
    "actual_shape": [3459, 204]        # requested 3459 x 2128; 2128 - 1924 = 204
    "actual_transform": [4.99997…, 0, -9619.95…, 0, -4.99997…, 2456657.46…]

  # read back (204x3459 uint8 cache file):
  nonzero frac 1.0   mean 88.57   min/max 25 159      # NOT zeros
  ```
- **Self-refutation attempted:** (a) Checked whether the project already knows —
  `DECISIONS.md:397-419` records the incident, but its stated mechanism is **wrong**: "the strip is
  entirely WEST of x=0 so it's outside the E000_N40 tile and **reads as zero pixels**". The pixels are
  not zero (measured above); they are in-tile pixels with a falsified transform. Because the entry
  concluded the output was empty, the residual risk (a partial straddle stays plausible-looking) was
  never recognised, and the deferral decision ("not fixing this in Stage 2 now") was taken against a
  benign failure mode that does not exist. (b) Checked reachability across every cached window in the
  repo: **1 of 49** (all 39 v2 + 9 of 10 v1 are fully inside their tile), so no shipped v2 number is
  affected — this is why I did not rate it blocker. (c) Checked whether a test pins it: the
  containment assertion exists (`tests/test_stage2_one_image.py:96-99`) but only for the hardcoded
  good ObsId `ESP_069669_2220` and only under `-m slow`; `stage2_one_image` itself never asserts it.
  (d) Checked whether the same pattern is deliberate elsewhere — it is not: `src/colour.py:147-152`
  and `src/hirise_imagery.py:237-252` both clip first and derive the transform from the **clipped**
  window, so this site is an oversight, not a convention. (e) Checked the live map path
  (`src/mapping.py:60-67`) — same shape of code, but `scripts/map_region.py:96-115` caps offsets at
  `extent - win` and `predict_window` rebuilds the tile grid from `arr.shape`, so it cannot
  misregister.
- **Fix:** clip once and use the clipped window for both the read and the transform
  (`clipped = window.intersection(Window(0, 0, src.width, src.height))`, `data = src.read(window=clipped)`,
  `new_transform = src.window_transform(clipped)`), and raise if `clipped != window` so the deferred
  multi-tile case fails loudly instead of producing a misregistered window. Correct the
  `DECISIONS.md:410-412` mechanism at the same time.

### geo-crs-2 — Invariant 2's only automated km-scale guard cannot fail: the phase-correlation solve is bounded to ±640 m by construction, and the solved shift is applied to every polygon with no band check
- **Severity:** medium
- **Liveness:** live-shipped (all 39 v2 label sets were produced with the shift applied)
- **Confidence:** high (bound is analytic + confirmed in the installed skimage source)
- **Where:** `tests/test_coregister.py:254-258`; `src/coregister.py:216-221`, `:403`, docstring
  `:29-32`; `src/labeling.py:474-475` + `:85-93`; `config.yaml:58` (`search_radius_m`, dead);
  `scripts/run_stage3.py:54-58`, `:95-98`

CLAUDE.md invariant 2 requires a km-scale HiRISE↔CTX residual to "fail loudly". The only place that is
mechanised is `assert mag < 1000.0` in a slow-marked test on one hardcoded ObsId — and that assertion
is **unfalsifiable**. `phase_cross_correlation` folds its peak into `[-N/2, N/2]`
(`skimage/registration/_phase_cross_correlation.py:354-359`), so with `fft_window_px = block_px = 256`
at 5 m/px the returned shift can never exceed ~128.7 px = 643 m per axis, i.e.
`|shift| ≤ ~910 m < 1000 m` for *any* input, including a completely mis-registered image. A true
km-scale offset therefore **aliases** into a plausible sub-640 m shift, which
`labeling._apply_coreg_shift` then applies to every detection polygon verbatim — no magnitude band, no
`peak_correlation` floor. `coregistration.search_radius_m: 400` (present in all three configs) is read
by no code, so nothing bounds the solve either.

- **Failure scenario:** any image whose Stage-1/Stage-2 geometry is off by > ~640 m (e.g. the
  geo-crs-1 straddle, an un-corrected SP1 as in geo-crs-3, or a future local-radius regression) yields
  an aliased shift of a few hundred metres that looks exactly like the healthy cohort (measured range
  62-327 m), passes the test, and gets *added* to the polygons — moving the labels further from the
  CTX texture while every recorded diagnostic stays green. The one real km-scale misregistration in
  the project's history is the proof: `ESP_057469_2215`'s 9 620 m error never appeared as a shift at
  all. It surfaced only because `select_fft_window` happened to raise "no power-of-2 ≥ 64 fits" on the
  0.13 %-coverage mask — an *unrelated* guard — and `run_stage3.py:54-58` swallowed that `RuntimeError`
  and then printed the ObsId under `"Skipped (Stage 2 missing)"` (`:95-98`, which pools genuine
  Stage-2-absent skips with hard failures) while returning exit code 0.
- **Evidence:**
  ```
  tests/test_coregister.py:254-258
      mag = prov["shift_m"]["magnitude"]
      assert mag < 1000.0, (          # max attainable = hypot(643, 643) = 910 m
          f"{OBS_ID}: solved shift |{mag:.1f}| m is in km territory ...

  skimage/registration/_phase_cross_correlation.py:354-359
      midpoint = np.array([np.trunc(axis_size / 2) for axis_size in shape])
      shift[shift > midpoint] -= np.array(shape)[shift > midpoint]     # -> |shift| <= N/2

  src/labeling.py:474-475
      shift = coregister.load_shift(obs_id, cache_dir) if apply_coreg_shift else None
      gdf = _apply_coreg_shift(gdf, shift)        # no magnitude / peak gate

  src/coregister.py:29-32 (docstring)
      **No hard flag/fail thresholds applied** — the notebook 05 whole-image validation is
      where accept/flag thresholds are eyeballed.
  ```
- **Self-refutation attempted:** (a) This is adjacent to **R30** (`labeling-3`) but distinct and
  corrects it: R30's mitigation argument is "the de-facto backstop is the Stage-3 block-median
  correlation, which did lock on 38 of 39 images" — that backstop is structurally blind to exactly the
  km-scale errors invariant 2 names. `coregistration.enabled` is **already filed** as `labeling-4`, so
  I only cite the separate unread key `search_radius_m`. (b) Checked whether the guard lives elsewhere:
  `grep search_radius_m` → 3 config hits, 0 code hits; `qa.assert_centroid_consistent` (the 15 km
  gross gate) has no production caller (R30). (c) Checked whether it ever bit: measured all 48 cached
  coregistration JSONs — v2 spans 62-327 m (median 195 m), v1 126-269 m, all inside the O(200 m) band,
  so no shipped number is wrong; that is why this is medium, not high. (d) Checked whether the
  `upsample_factor` refinement can exceed the fold-over bound — it adds at most
  `±(⌈1.5·20⌉//2)/20 = ±0.75 px`, so the bound holds.
- **Fix:** gate in code, not in a test: after the solve, raise (or write
  `status: "REJECTED_out_of_band"` and refuse to apply) when
  `|shift| > coregistration.search_radius_m` or when `peak_correlation` is below a floor; make
  `_apply_coreg_shift` refuse an out-of-band shift; and separate the "FAILED" list from the
  "Stage 2 missing" list in `run_stage3.py` with a non-zero exit.

### geo-crs-3 — The SP1 correction is silently skipped for low-latitude images, and the tolerance is in degrees of latitude while the resulting ground error scales with longitude distance from the 180° central meridian
- **Severity:** low (latent; zero current impact — verified)
- **Liveness:** live-shipped code, dead for the current cohort
- **Confidence:** high on the mechanism, high on "no current image affected"
- **Where:** `src/detections.py:37`, `:40-51`, `:79-95`

`_suspect_sp1` declares a `.prj` buggy only if it carries the `D_unnamed` fingerprint **and**
`|SP1 - manifest CenterLat| > 15°`. The authoritative value — the PDS `.LBL` `CENTER_LATITUDE` — is
fetched only *after* that decision (`:82-83`), so it is never used as the test. In ESRI
`Equidistant_Cylindrical`, SP1 scales x by `cos(SP1)`; the ground error from mis-reading it is
`R·Δλ·(1 − cos SP1_true)` where `Δλ` is the distance from the file's central meridian (180° for every
BoulderNet export). The circum-Chryse cohort sits near lon 0, i.e. `Δλ ≈ 180°` — so an uncorrected
SP1 of 5° is already ≈40 km of x error and the 15° tolerance admits ≈360 km. The tolerance is
therefore not calibrated to the consequence, and a skip is recorded as `status: "trusted_prj"`, a
provenance claim that is then trusted by `hirise_imagery._corrected_source_crs` for the JP2 side too.

- **Failure scenario:** a new manifest row (PLAN_NewDetections is a live plan; invariant 7 requires
  new rows to flow end-to-end) at `CenterLat ≈ 12°` near lon 0 whose export carries the usual
  `D_unnamed` + `SP1 = 0` poisoning: `|0 − 12| = 12 < 15` → no correction, status `trusted_prj`,
  polygons and JP2 both placed ~hundreds of km east. Stage 2 then windows the *mis-located* bbox
  (possibly in the wrong Murray tile → geo-crs-1), the coverage mask collapses, and with R30's gate
  absent the only symptom is a low `hirise_coverage_fraction` that nothing asserts on.
- **Evidence:**
  ```
  src/detections.py:37
      _SP1_TOLERANCE_DEG = 15.0
  src/detections.py:49-51
      has_bad_datum = bool(_BAD_DATUM_FINGERPRINT.search(prj_text))
      far_from_image = abs(current_sp1 - image_lat_deg) > _SP1_TOLERANCE_DEG
      return (has_bad_datum and far_from_image), current_sp1
  src/detections.py:82-83   # truth fetched only inside the already-decided-buggy branch
      pds_labels.fetch_label(obs_id, manifest_row["LabelURL"], cache_dir)
      origin = pds_labels.projection_origin(obs_id, cache_dir)
  ```
- **Self-refutation attempted:** (a) Tried to find a current victim. Only `ESP_039820_1750`
  (`CenterLat = -4.9112`) is inside the tolerance window with the `D_unnamed` fingerprint, and it is
  **not** a victim: its cached PDS label says `CENTER_LATITUDE = 0.000 <DEG>` and
  `A_AXIS_RADIUS = 3396.19 <KM>`, matching its `.prj` `SPHEROID["unnamed",3396190.0]` — so `SP1 = 0`
  is genuinely correct and the code's decision to leave it alone is right. Every other cohort image is
  at `|lat| ≥ 20`, where the 15° tolerance separates the regimes cleanly (all 6 v1 + 32 v2 buggy files
  were caught: verified in the Stage-1 sidecars). (b) Checked whether DECISIONS pins the 15° value as
  deliberate — it does ("a generous margin that cleanly separates the two regimes"), but the
  justification is about separating the two *observed* regimes, not about bounding the resulting
  ground error, so the latent gap is unaddressed rather than accepted. (c) Checked whether the
  fingerprint alone would be a safer test — no: `ESP_039820_1750` shows `D_unnamed` on a correct file,
  which is exactly why the conjunction exists.
- **Fix:** fetch the `.LBL` whenever the fingerprint matches and compare `.prj` SP1 against
  `projection_origin()["center_lat_deg"]` directly (the PDS value is a multiple of 5°, so a ±2.5°
  tolerance suffices); record the comparison in the sidecar so `trusted_prj` means "verified against
  PDS", not "not obviously wrong".

### geo-crs-4 — `read_native_window`'s out-of-extent diagnostic is unreachable; rasterio raises first
- **Severity:** low
- **Liveness:** live (QA/zoom path only)
- **Confidence:** high
- **Where:** `src/hirise_imagery.py:240-250`

`Window.intersection` raises `WindowError` when the windows are disjoint
(`rasterio/windows.py:251-258`: `_intersection` returns `None` → `WindowError`), so the
`if clipped.width <= 0 or clipped.height <= 0` branch — written specifically to "fail loudly with a
helpful message instead of writing a 0x0 GeoTIFF deep in rasterio's C layer" — can never execute. The
call still fails loudly, but with rasterio's generic message, and the carefully-written diagnostic
naming the exact HiRISE PDS CRS pitfall ("bounds were computed after a reprojection round-trip while
the JP2's own embedded CRS metadata is wrong") is lost at the moment it is needed. `src/colour.py:149`
has the same dead test.

- **Failure scenario:** a notebook passes CTX-derived bounds to `read_native_window` for an
  SP1-affected image; the user gets `WindowError: windows do not intersect` instead of the message
  that tells them to read the shapefile in its native CRS, and the SP1 trap has to be re-diagnosed
  from scratch.
- **Evidence:**
  ```
  src/hirise_imagery.py:240-242
      clipped = window.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
      if clipped.width <= 0 or clipped.height <= 0:
          raise ValueError(                      # unreachable
  rasterio/windows.py:251-258
      coeffs = _compute_intersection(w1, w2)
      if coeffs[2] > 0 and coeffs[3] > 0: return Window(*coeffs)
      ... raise WindowError(...)
  ```
- **Self-refutation attempted:** checked whether a partial overlap can yield a zero-size window (no —
  `_compute_intersection` requires both dimensions `> 0` to return at all) and whether any caller
  catches `WindowError` and re-raises usefully (none do; the two callers are notebook/QA helpers).
- **Fix:** compute the intersection defensively
  (`clipped = _compute_intersection(...)` or wrap in `try/except WindowError`) and raise the existing
  `ValueError` from there.

## Refuted by my own check

- **pyproj drops `Standard_Parallel_1` when Stage 1 canonicalises the corrected ESRI WKT, so the JP2-side
  override is a no-op** (suggested by `hirise_imagery._crs_equal`'s own docstring). It does not: all 38
  `sp1_corrected_from_pds_label` sidecars in `cache/` + `cache_v2/` carry the corrected SP1 literally in
  `source_crs_wkt` (20/25/30/35/40/45/50/-25 as applicable). The override is live.
- **The buggy `.prj` files also carry the wrong sphere radius, so correcting SP1 alone leaves a ~1.6 km
  y error.** Checked 6 `.prj` files against their cached PDS labels: `ESP_047976_2020` 3393833.2607584
  vs `A_AXIS_RADIUS = 3393.8332607584 <KM>`, `ESP_054857_2270` 3386150.7470034 vs 3386.1507470034,
  `ESP_056165_2200` 3389574.3490888 vs 3389.5743490888, `ESP_065711_1545` 3392593.6110435 vs
  3392.5936110435 — exact to 1e-4 km. Only the datum/spheroid *names* and SP1 are poisoned; SP1-only
  correction is sufficient (and the measured 62-327 m Stage-3 shifts corroborate it).
- **`ESP_039820_1750` is an uncaught SP1 victim** (it has `D_unnamed` + `SP1 = 0` and was recorded
  `trusted_prj`). Its PDS label really says `CENTER_LATITUDE = 0.000` and `A_AXIS_RADIUS = 3396.19 KM`
  (= the equatorial radius, correct for lat 0), so the file is right as shipped.
- **The decimated HiRISE read scales the array but forgets the transform** (the classic bug the brief
  names). It does not: `hirise_imagery.py:194-199` sets `x_scale = a·(ds.width/out_w)`,
  `y_scale = e·(ds.height/out_h)` and preserves the origin, which is exactly GDAL's `out_shape`
  semantics. (The read is nearest-neighbour rather than averaged — an aliasing/quality issue for the
  correlation, not a georeferencing one.)
- **`ensure_jp2_local` can commit a truncated JP2 to the cache** (unlike `_download_to`, it has no size
  floor). It cannot in the realistic case: `shutil.copyfileobj` → `HTTPResponse.read(amt)` raises
  `IncompleteRead` on a premature EOF, and the exception propagates before `tmp.replace(out_path)`.
- **`read_full_footprint_decimated`'s cache key `f"{int(target_mpp)}mpp_full"` collides for
  non-integer resolutions.** True but unreachable: every caller passes `target_mpp=5.0`.
- **`ensure_tile_cached`'s `try: recorded_url = used_url / except NameError` is broken** when the zip
  pre-exists. It works — `UnboundLocalError` subclasses `NameError`.
- **Target CRS (IAU-2000 sphere, f = 0) vs the Murray tiles' oblate Mars-2015 CRS (1/f = 169.89) is a
  latitude-dependent y error.** It is not, for two independent reasons: PROJ's `eqc` is spherical-only
  and uses the semi-major axis, and the tiles' own affines confirm it (`E0_N40` origin
  `f = 2608086.69 = 3396190 × 44.0002°` in radians — pure `R·φ`); and no sphere→oblate transform is
  ever performed (bounds go straight from the target CRS into the tile's affine).
  `DECISIONS.md:262-273` already records the finding and the "sub-pixel" conclusion, which this check
  supports.
- **`stage1_one_image(manifest_row=None)` silently skips the SP1 correction** (flagged in the brief).
  All four call sites (`scripts/run_stage1.py:34`, two tests) pass a row; the default only exists for
  the `read_detection_shapefile` unit tests. Latent, not live.
- **Longitude seam / `clon_180`→`clon_0` wrapping.** No cohort image is within 25° of the ±180° seam
  (the nearest is `E152_N-8`); a seam-straddling bbox would produce a hemisphere-wide window whose
  cropped width is 0, which raises in `rasterio.open(..., width=0)` rather than passing silently.
- **`ctx_edr.frames_in_crop` joins SeamMap polygons to a crop box in a different CRS.**
  `striping.load_frames` reprojects (or assigns) to the abundance raster's CRS and caches that, and
  `frames_in_crop` builds the crop box from the same abundance transform — consistent.
- **`run_stage3.py` reporting a hard failure as "Skipped (Stage 2 missing)"** — real, but too small to
  file on its own; folded into geo-crs-2 as supporting evidence.

## Verified clean

- `shift_px_to_world_m` (`coregister.py:281-298`): row→world-y sign flip is correct and pinned by two
  regression tests (`test_coregister.py:101-123`), matching the DECISIONS 2026-06-10 W1 fix. The
  cohort-wide `dy > 0` pattern is expected (HiRISE sits north of the mosaic in 38/39 images).
- Per-image local-radius handling end to end: 6 distinct local radii observed across the priority10
  `.prj` files (3386150.75 … 3396190.0), each read from the file's own `.prj`; no radius is hardcoded
  anywhere in `coregister.py`, `ctx_retrieve.py`, `ctx_tiles.py`, `hirise_imagery.py`, `pds_labels.py`,
  `ctx_edr.py`.
- Murray Lab URL padding for negative longitude/latitude: all 24 cached tile sidecars resolved through
  the `_padded_manifest_form` fallback (`E-8_N32 → E-008_N32`, `E152_N-8 → E152_N-08`,
  `E0_N-28 → E000_N-28`), i.e. the 404-retry path is exercised by real data, not just unit tests.
- HiRISE coverage mask ↔ CTX window grid identity: `build_hirise_coverage_mask` allocates from the
  window's own `(height, width)` and writes with the window's transform/CRS
  (`ctx_retrieve.py:502-528`), asserted for real in `test_stage2_one_image.py:81-89`.
- `_snap_bounds_to_pixel_grid` outward snapping and its idempotence (tile-origin anchored, correct
  `e < 0` handling); accepts both `Affine` and the 6-list form stored in sidecars.
- Window-overhang audit of every cached window: 39/39 v2 and 9/10 v1 requested windows lie fully
  inside their Murray tile; only `ESP_057469_2215` overhangs (geo-crs-1).
- Stage-3 shift audit: 48 cached solves, all in 62-327 m — inside the invariant-2 O(200 m) band.
- `_warp_hirise_to_ctx_grid` / `build_hirise_coverage_mask` reproject with explicit `src_crs`/`dst_crs`
  and the documented `bilinear`-for-intensity vs `nearest`-for-mask split; `select_fft_window`
  intersects HiRISE coverage with `ctx_arr > 0` so mosaic nodata cannot enter the correlation.
- `src/mapping.py:60-67` + `scripts/map_region.py:96-115` (the live map path): offsets are
  non-negative and capped at `extent - win`, and the tile grid is derived from the array's own shape,
  so the geo-crs-1 pattern cannot misregister there.
- `src/colour.py:147-152` and `src/hirise_imagery.py:237-252` use the correct clip-then-transform
  order.

## Coverage note

Read in full: `src/coregister.py`, `src/ctx_retrieve.py`, `src/ctx_tiles.py`, `src/hirise_imagery.py`,
`src/pds_labels.py`, `src/ctx_edr.py`, `src/detections.py`, `src/qa.py`, `scripts/run_stage1.py`,
`scripts/run_stage3.py`, `src/mapping.py`, and the seven named test files. Read the relevant slices of
`src/labeling.py` (`_apply_coreg_shift`, Stage-4 wiring), `src/striping.py` (`load_frames`,
`lonlat_to_rc`), `src/colour.py`, `scripts/map_region.py`, and the installed `rasterio`
(`_io.pyx`, `windows.py`) and `skimage` (`_phase_cross_correlation.py`) sources for the crop and
fold-over semantics. Grepped `DECISIONS.md` by term (`057469`, `straddl`, SP1, sign, `search_radius`,
oblate) and checked `docs/CODE_REVIEW_2026-07-31.md` + the three existing area files so nothing here
duplicates R01/R13/R21/R29/R30 or `labeling-4`.

Measurements came from read-only inspection of committed/cached **sidecars** (`cache*/ctx_windows/*.json`,
`cache*/ctx_tiles/*.json`, `cache*/reprojected_detections/*.json`, `cache*/coregistration/*.json`),
the source `.prj` files, the cached PDS `.LBL`s, and one small cached window GeoTIFF
(204 × 3459 uint8, 407 KB) plus its mask — no multi-GB mosaic reads, no network, no notebooks, no
re-runs.

Could **not** check: (1) whether the Murray Lab URL forms still resolve or how `_download_to` behaves
on a live 404/timeout (no network); (2) the actual JP2 pixel geometry, so the claim that
SP1-corrected-CRS-over-original-transform places the imagery correctly rests on the 62-327 m Stage-3
peaks (0.58-0.88) rather than on direct inspection; (3) re-extraction of `ESP_057469_2215` to confirm
the crop from the source tile (inferred from `rasterio.windows.crop` semantics plus the on-disk
transform/shape/content, which agree exactly); (4) `discover_murray_lab_url_template` /
`read_ctx_tile_crs` (superseded per the module docstring, `target_crs` is now hardcoded — reviewed by
reading only).
