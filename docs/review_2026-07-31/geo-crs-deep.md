# Review area: geo-crs-deep

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-02
- **Verification:** self-refuted (single-agent pass; not independently verified)
- **Relation to pass 1:** second, deeper pass over the same files as `geo-crs.md` (R31 / `geo-crs-1..4`).
  Nothing here duplicates R31, `geo-crs-2/3/4`, R29, R30, R21 or R13.

## Findings

### geo-crs-deep-1 — `peak_correlation`, the co-registration's only per-image quality number, is truncated at its own screening threshold *and* scores a per-block shift model rather than the global shift that is applied
- **Severity:** medium (a real protocol defect; no shipped number is wrong today — verified)
- **Liveness:** live-shipped (the statistic accompanies every v2 label set; Stage 4 applies the shift by
  default, `scripts/run_stage4.py:100`)
- **Confidence:** high (mechanism is 4 lines of code; quantified on all 48 cached solves)
- **Where:** `src/coregister.py:226-238`, `:259-271`, `:384-397`, `:419`; consumed by
  `src/labeling.py:567-569`, `scripts/run_stage4.py:73`; **used as evidence** in
  `DECISIONS.md:2376-2378`, `:2386-2396`, `:2541-2542`; **defined wrongly** in
  `dataset/DATA_DICTIONARY.md:137`; **published** as Figure 3's x-axis, `docs/methods.md:601`, `:618-621`

Two independent defects in the same number.

**(a) It cannot fall below `block_peak_min`.** On the primary `block_median` path the reported
`peak_correlation` is `median(peaks[peaks >= block_peak_min])` — the median of exactly the blocks that
already cleared the 0.5 floor. So `peak_correlation >= 0.5` is true **by construction** for every
block-median solve. `DECISIONS.md:2376-2378` records a cohort screen on this quantity ("then ≥ 0.5 → all
38 cleared, so effectively no coreg filter") and `:2392-2394` concludes "None of the 38 images is below
an empirically reasonable noise floor (~0.3 – 0.5 …), so the cut at 0.5 effectively removes nothing."
That is a tautology, not a measurement: the 38 images *are* the block-median solves, and the cut is the
constraint that defines them. The untruncated companion statistic is in the same JSON and disagrees —
4 of 39 images have `single_window.peak < 0.5` (`ESP_049242_2115` **−0.058**, `ESP_076499_1160` 0.382,
`ESP_052576_2250` 0.447, `ESP_017355_2260` 0.487) yet all four report `peak_correlation` ≥ 0.586.

**(b) It measures the wrong model.** `phase_correlate_translation` computes its Pearson `peak` after
applying **that call's own** solved shift (`:230-236`). In `block_shift_field` each block therefore gets
a goodness-of-fit for *its own local* translation. The chosen global shift is the **median** over blocks
(`:266-267`), and it is never scored against anything. So `peak_correlation` is the fit quality of a
spatially-varying shift field, not of the single rigid translation that `labeling._apply_coreg_shift`
actually applies to every polygon. A fanned-out field — every block individually at peak 0.9 but shifts
scattered over ±20 px — would report `peak_correlation` 0.9 while the applied median fits no block.
The statistic that *would* catch that (`block_mad_px`) and the one that measures field density
(`n_confident_blocks / n_blocks`) are both computed and written to the sidecar, and neither is gated,
thresholded, or reported in `methods.md` / `DECISIONS.md`.

Consequences that are already in the record: `DECISIONS.md:2541-2542` ("The array-space solve itself was
verified correct (its own post-shift Pearson check, peaks 0.58-0.88)") uses (b) to certify the solve
during the W1 sign-error post-mortem — the conclusion was right but the cited evidence does not support
it. `dataset/DATA_DICTIONARY.md:137` defines the field as "Pearson correlation between CTX sub-window
and the shift-corrected HiRISE sub-window … bland-plains scenes produce low values regardless of the
true shift" — that describes the *fallback* path only; on the primary path bland scenes lose blocks, not
peak, so the documented diagnostic behaviour is inverted. Figure 3 (`docs/methods.md:618-621`) plots
"global `peak_correlation`" against "the fraction of 256 px blocks whose local peak ≥ 0.5" and reads the
result as "the global shift agrees with a dense, high-confidence block field" — but the x-axis is a
truncation of the same subset the y-axis counts (measured Spearman between the two across the 38
block-median solves: **ρ = +0.542, p = 4.4e-4**), and neither axis measures agreement with the global
shift.

- **Failure scenario:** a future image with real residual rotation/scale (or a partially mis-warped
  HiRISE, e.g. the R31 straddle) produces a block field with high individual peaks and mutually
  inconsistent shifts. The median is meaningless, `peak_correlation` comes out ~0.8, the ≥ 0.5 screen
  passes, `block_mad_px` (the only statistic that would show it) is written but never read, and every
  polygon in the image is translated by a shift that fits nothing. Same mechanism admits a solve resting
  on very few blocks: the acceptance is an absolute count (`min_confident_blocks = 6`) with no fraction
  floor, so 6 confident blocks out of 200 is accepted identically to 190 out of 200.
- **Evidence:**
  ```
  src/coregister.py:230-236   # peak uses THIS call's own (dy,dx)
      mov_shifted = nd_shift(moving.astype(np.float32), shift=(dy, dx), order=1, ...)
      a = reference.astype(np.float32)[margin:-margin, margin:-margin]
      b = mov_shifted[margin:-margin, margin:-margin]
      ...
      peak = float((a * b).sum() / denom) if denom > 0 else float("nan")

  src/coregister.py:259-271   # the reported number is a median over the ALREADY-SCREENED subset
      peaks = np.array([b["peak"] for b in field], dtype=np.float64)
      conf = peaks >= block_peak_min
      n_conf = int(conf.sum())
      if n_conf < min_confident_blocks:
          return None
      ...
      "median_block_peak": float(np.median(peaks[conf])),

  src/coregister.py:384-387
      if robust is not None:
          dy_px, dx_px, block_stats = robust
          peak = block_stats["median_block_peak"]      # <- becomes "peak_correlation"

  DECISIONS.md:2392-2394
      None of the 38 images is below an empirically reasonable noise floor (~0.3 –
      0.5 for cross-instrument HiRISE→CTX matching after decimation), so
      the cut at 0.5 effectively removes nothing.
  ```
  Measured over the 38 v2 block-median sidecars (`cache_v2/coregistration/*.json`):
  `peak_correlation` min **0.578**, max 0.875 (floor = `block_peak_min` = 0.5, so the observed minimum
  is 0.078 above a bound it cannot cross); `n_confident/n_blocks` min **0.46** (`ESP_068483_2280`, peak
  0.646), then 0.52 (`ESP_076499_1160`, 0.586), 0.56 (`ESP_055978_2270`, 0.626) — i.e. three images
  where barely half the blocks correlate report "confidence" ≥ 0.58. `block_mad_px` max 3.70 px
  (`ESP_064510_2260` = 18.5 m, the one image outside `docs/methods.md:585`'s "median residual … < ~15 m").
- **Self-refutation attempted:** (a) checked whether the code over-claims — it does not:
  `src/coregister.py:26-28` and `docs/methods.md:551-555` both state plainly that the recorded value is
  the median confident-block peak, so this is a *use* defect, not a coding lie; that is why the finding
  is anchored on the consumers. (b) checked whether the untruncated statistic is discarded — it is not
  (`single_window` is preserved at `:422-426`), so the fix is free; (c) checked whether the
  informative companions are used anywhere: `scripts/probes/_w1_coreg_vs_auc.py:73-77` does correlate
  `confident_frac` / `mad_dy_px` / `mad_dx_px` against AUC, so the *probe* is fine — the gap is in the
  cohort screen, the sidecar contract and the published figure; (d) checked whether any current image is
  actually mis-solved: no — `block_mad_px` ≤ 3.7 px cohort-wide and 38/39 solves have ≥ 0.46 of blocks
  confident, so no shipped label geometry is wrong. That bounds this to medium, not high. (e) grepped
  `DECISIONS.md` for `block_peak_min`, `peak_correlation`, "noise floor": `:1225-1238` documents the
  API and `:2386-2396` the distribution; nothing anywhere records the truncation or the
  per-block-vs-global distinction as known.
- **Fix:** report and gate the statistics that are not self-fulfilling — (i) add
  `global_peak`: re-run the Pearson check once with the *chosen* median shift over the full covered
  region and record it as `peak_correlation`, demoting the current value to
  `median_block_peak_conditioned`; (ii) gate on `block_mad_px` and on
  `n_confident_blocks / n_blocks` (a fraction floor beside the absolute `min_confident_blocks`), not on
  a conditioned median; (iii) correct `dataset/DATA_DICTIONARY.md:137` to state the primary path's
  definition and its lower bound, and annotate `DECISIONS.md:2386-2396` that the 0.5 screen was vacuous
  for the 38 block-median images.

### geo-crs-deep-2 — `ensure_jp2_local` commits a truncated HiRISE JP2 to a permanent cache: `HTTPResponse.read(amt)` provably does not raise on premature EOF, and the only integrity check is a 1 MB floor on files of 149 MB – 1.31 GB
- **Severity:** medium
- **Liveness:** live (every Stage 2 / Stage 3 run and every new manifest row; invariant 4/7)
- **Confidence:** high on the mechanism (CPython source), medium on the downstream symptom (GDAL's
  behaviour on a truncated JPEG2000 codestream could not be tested — no network)
- **Where:** `src/hirise_imagery.py:142`, `:146-148`, `:155`; sibling gap
  `src/ctx_retrieve.py:194`, `:207` — **corrects** `geo-crs.md:252-254`

`geo-crs.md` refuted this on a false premise: "`shutil.copyfileobj` → `HTTPResponse.read(amt)` raises
`IncompleteRead` on a premature EOF, and the exception propagates before `tmp.replace(out_path)`."
CPython does the opposite, and says so in a comment: `HTTPResponse.read(amt)` on a short read returns
`b""`, closes the connection and **deliberately does not raise** ("Ideally, we would raise
IncompleteRead if the content-length wasn't satisfied, but it might break compatibility").
`copyfileobj(resp, f, length=1<<20)` calls exactly that, so a dropped connection ends the copy loop
normally and `tmp.replace(out_path)` publishes the partial file. The only gate is
`out_path.stat().st_size > 1_000_000` — measured against the real cache, JP2s span
**149,019,114 – 1,306,649,437 bytes** (`cache/hirise_jp2/`), so the floor sits 150–1300× below the true
size and cannot fire for any truncation that matters. `_open_source:155` re-uses the same 1 MB test, so
the truncated file is preferred over `/vsicurl/` forever after.

Nothing recovers from it. `read_full_footprint_decimated` derives
`cache/hirise_decimated/{ObsId}_5mpp_full.tif` from whatever it read and the **only** staleness key on
that derived raster is a CRS comparison (`:181-186`) — not source size or mtime — so a later manual
re-download of a complete JP2 does not invalidate the corrupt decimated raster, nor the
`{ObsId}_hirise_mask.tif` and `hirise_coverage_fraction` computed from it, nor the Stage-3 solve.
`_download_to` (the CTX-tile sibling) *does* read `Content-Length` at `:194` but only forwards it to
`on_progress`; it never compares it to `downloaded`. That path is nevertheless safe in practice because
a truncated zip loses its central directory and `zipfile.ZipFile` at `:293` raises — the JP2 path has no
equivalent structural check.

- **Failure scenario:** a 400 MB HiRISE RED download drops at 60 %. `ensure_jp2_local` returns
  successfully; the JP2 is cached and preferred over the network from then on. GDAL then either (best
  case) raises on the missing codestream — a loud, confusing failure with a valid-looking cache — or
  (worst case, JPEG2000 being progressive/resolution-layered) decodes what is present and zero-fills the
  rest, in which case `build_hirise_coverage_mask` marks the missing part un-covered, Stage 4 silently
  emits no tiles there, `hirise_coverage_fraction` drops from ~0.55 to ~0.33, and nothing asserts on it
  (see R30 — the invariant-2 gate has no production caller). Stage 3 then solves its shift from the
  surviving fragment only.
- **Evidence:**
  ```
  src/hirise_imagery.py:142-148
      if out_path.exists() and out_path.stat().st_size > 1_000_000:  # > 1 MB sanity
          return out_path
      tmp = out_path.with_suffix(".JP2.partial")
      req = urllib.request.Request(jp2_url, headers={"User-Agent": "hirise2ctx/0.1"})
      with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
          shutil.copyfileobj(resp, f, length=1 << 20)  # 1 MB chunks
      tmp.replace(out_path)

  CPython http/client.py, HTTPResponse.read(amt) — verified in the running interpreter
  (Python 3.14.3, conda-forge):
          s = self.fp.read(amt)
          if not s and amt:
              # Ideally, we would raise IncompleteRead if the content-length
              # wasn't satisfied, but it might break compatibility.
              self._close_conn()

  src/ctx_retrieve.py:194                 # Content-Length read, never verified
              total = int(resp.headers.get("Content-Length") or 0)

  measured: cache/hirise_jp2/  min 149,019,114 B (ESP_045390_2215)
                               max 1,306,649,437 B (ESP_068483_2280)   vs floor 1,000,000 B
  ```
- **Self-refutation attempted:** (a) read the installed `http.client` source rather than trusting the
  pass-1 claim — the non-raising branch is explicit; (b) checked whether `chunked` transfer would save
  it (`_read_chunked` *does* raise `IncompleteRead`) — PDS serves `Content-Length`, and the code cannot
  rely on the encoding either way; (c) checked whether any caller validates the JP2 afterwards:
  `_warp_hirise_to_ctx_grid:70-73` and `build_hirise_coverage_mask:497-500` call
  `ensure_jp2_local` then read, with no size/geometry assertion; (d) checked for evidence it has already
  happened — no `.partial` leftovers and all 48 cached JP2s are ≥ 149 MB, so this is latent, which caps
  it at medium; (e) checked whether the sibling zip path is equally exposed — it is not (zip central
  directory + 50 MB floor), so this is a genuine asymmetry rather than a project-wide convention.
- **Fix:** compare bytes written against `Content-Length` before `tmp.replace(...)` and raise otherwise
  (three lines, mirroring `_download_to`'s floor); raise the 1 MB sanity floor to something related to
  the real distribution or drop it in favour of the length check; and add the source size/mtime to the
  `hirise_decimated` cache's staleness test so a corrected download invalidates its derivatives.

### geo-crs-deep-3 — `nominal_footprint_bounds` spends its `nominal_hirise_width_m` in *projected* metres of an equirectangular clon_0 CRS, so the window covers only `width_m·cos(lat)` of ground and clips the swath — measurably so on the one image that uses it
- **Severity:** medium (latent for the cohort; the one current user is measurably clipped but carries no
  labels and is excluded)
- **Liveness:** live-shipped code, live invariant-7 hazard (a new manifest row with an empty shapefile)
- **Confidence:** high (measured against the image's own PDS label)
- **Where:** `src/ctx_retrieve.py:376-401` (esp. `:398-400`), vs the polygon path's buffer at `:372`;
  `config.yaml:52-53` / `config_v2.yaml:77-78`; unused truth source `src/pds_labels.py:125-133`

The target CRS is spherical equirectangular with standard parallel 0 (`config_v2.yaml:16-31`), so
`x = R·Δλ` and one projected x-metre is `cos(φ)` ground metres. `nominal_footprint_bounds` builds the
fallback window as `half_w = width_m / 2.0` **in projected coordinates**, so a
`nominal_hirise_width_m: 6000` intended as the HiRISE swath width covers only `6000·cos(φ)` metres of
ground east-west. The y axis is unaffected (`y = R·φ`, 1 projected m = 1 ground m). Compounding it, this
branch — unlike `compute_window_bounds:372` — adds **no** `buffer_m`, so there is no margin to absorb
the shortfall.

Measured on the only image that takes this path, `ESP_065711_1545` (v1, `footprint_source:
"nominal_from_manifest"`, `CenterLat = -25.2993`): its own cached PDS label gives a footprint of
0.1045° longitude = 5.59 km on the ground = **6.19 km of clon_0 projected x**, against a window of
**6.005 km** projected x — the window is 185 m narrower than the image's own footprint, clipping ≈ 92 m
(18 CTX px) off each of the east and west swath edges. At the v2 cohort's typical latitude (≈ 44°) the
same 6000 m nominal would cover 4.32 km of ground against a ~5.6 km swath: ≈ 1.8 km clipped, ~180 px per
side.

The clipping is invisible to the one recorded diagnostic and in fact *improves* it: a narrower window
raises `hirise_coverage_fraction` (more of a smaller window is covered), so the sidecar's only
quality field moves the reassuring way when the failure gets worse.

- **Failure scenario:** invariant 7 — a new manifest row at lat ≈ 44° whose `*-mask-nms.shp` has zero
  detections (a genuinely boulder-free image, exactly the kind the target's zero-inflation needs) gets a
  CTX window ~1.8 km too narrow. The HiRISE coverage mask is cut to the window, so both swath edges are
  simply absent from the label grid, `hirise_coverage_fraction` reads *higher* than its neighbours, and
  no assertion exists between the manifest footprint and the extracted window.
- **Evidence:**
  ```
  src/ctx_retrieve.py:398-401
      half_w = width_m / 2.0
      half_l = length_m / 2.0
      bounds = (cx - half_w, cy - half_l, cx + half_w, cy + half_l)
      return _snap_bounds_to_pixel_grid(bounds, ctx_transform)

  src/ctx_retrieve.py:372          # the polygon path, for contrast — 1 km of slack
      expanded = (xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)

  config.yaml:52-53
      nominal_hirise_width_m: 6000     # used only when shapefile has 0 polygons
      nominal_hirise_length_m: 16000   # (ESP_065711_1545; diversity pick)

  measured (cache/pds_labels/ESP_065711_1545.LBL + cache/ctx_windows/ESP_065711_1545.json):
      PDS footprint  lon span 0.1045 deg -> 5.59 km ground E-W -> 6.19 km projected x
                     lat span 0.1403 deg -> 8.31 km ground N-S
      window         1201 x 3201 px -> 6.005 km projected x  (= 5.43 km ground), 16.005 km y
      hirise_coverage_fraction 0.4287
  ```
- **Self-refutation attempted:** (a) checked whether the current instance actually matters — it does
  not for any shipped number: `ESP_065711_1545` has zero detections and is excluded
  (`DECISIONS.md:1072`, "empty truth"), which is why this is medium rather than high; (b) checked
  whether pass 1 or `labeling.md` already covers it — `labeling.md:279-280` inspected exactly this image
  and concluded it "correctly falls back to `nominal_footprint_bounds`", i.e. the fallback was certified
  clean without checking its units, so this corrects that note; (c) checked whether `buffer_m` is
  applied later — it is not, `stage2_one_image:577-579` passes only the nominal sizes; (d) checked
  whether the config comment or docstring resolves projected-vs-ground — neither does
  (`:388` "builds a rectangle of size `width_m x length_m` around it"); (e) checked whether a correct
  footprint is available — `pds_labels.image_footprint` already parses the PDS
  MIN/MAX lat + E/WESTERNMOST lon and every label is cached for every row, but Stage 2 never calls it
  (`scripts/build_vclaire_manifest.py:261` is its only production caller).
- **Fix:** divide the east-west half-width by `cos(CenterLat)` (or, better, build the bounds from
  `pds_labels.image_footprint` reprojected into the target CRS), and add `buffer_m` on this branch as
  the polygon branch does; then assert the extracted window contains the PDS footprint.

### geo-crs-deep-4 — Stage 4's "runtime pixel-size guard" cannot fire and does not test the property it is cited as guaranteeing
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `src/labeling.py:484-495`; the producing path `src/ctx_retrieve.py:425-434`; cited as
  assurance in `docs/review_2026-07-31/labeling.md:339-340`

`extract_ctx_window` derives the window's transform from `src.window_transform(window)` of the parent
tile, which copies `a` and `e` bit-identically and only changes `c`/`f`. `mosaic_transform` is read from
`cache*/ctx_tiles/{tile}.json`, written from `list(src.transform)[:6]` of the *same* dataset. So
`abs(px_x - abs(mosaic_transform[0])) < 1e-6` is a comparison of a float with itself: no input reachable
through the pipeline can make it raise. Worse, the property it is documented as protecting — "the
integer-pixel alignment claim … the grid wouldn't be nested cleanly" — is about the *origin phase*
`((c_window - c_tile) / px_x) % 1 == 0`, which pixel-size equality does not test at all. That phase
relation is what R01 found broken in the merged regional mosaic, so the one class of failure this guard
is named for is precisely the one it cannot see. `labeling.md:339-340` lists it under **Verified clean**
as "the precondition for the integer-nesting claim", so the vacuity is currently on the record as
assurance.

- **Failure scenario:** no crash — the failure is that the guard's presence is used (in a review, and in
  its own comment) as evidence that the window↔mosaic grid nesting is checked at runtime, while a
  phase break would pass it silently. The comment at `src/labeling.py:485-486` and the `RuntimeError`
  text both name the alignment claim, not the pixel size, so a reader reasonably concludes it is
  verified.
- **Evidence:**
  ```
  src/labeling.py:484-490
      px_x = abs(window_transform.a)
      px_y = abs(window_transform.e)
      # Sanity: the window's pixel size must match the mosaic's. If not, the integer-pixel
      # alignment claim is wrong and the grid wouldn't be nested cleanly.
      if not (
          abs(px_x - abs(mosaic_transform[0])) < 1e-6
          and abs(px_y - abs(mosaic_transform[4])) < 1e-6
      ):

  src/ctx_retrieve.py:433-434     # a and e are copied from the tile, so they cannot differ
      data = src.read(window=window)
      new_transform = src.window_transform(window)
  ```
- **Self-refutation attempted:** (a) tried to construct a reachable failure — a stale/other-tile
  sidecar is the only candidate, and every Murray Lab tile shares the same 4.99997 m pixel, so even
  that passes; a replaced tile product of different resolution is the only input that fires, and it
  would also change `inner_shape`; (b) tried to show the phase can break instead, which would make the
  guard merely mis-aimed rather than vacuous — it cannot: `_snap_bounds_to_pixel_grid` plus the
  `int(round(...))` window rounding at `:428-432` force integer offsets, and all Murray tiles lie on one
  global 5 m grid, so `labeling.md:281-287`'s conclusion that the phase is exact stands. That is why
  this is low, not medium: the invariant holds, only the assurance is fake.
- **Fix:** replace the pixel-size comparison with the phase assertion it claims to make —
  `assert abs(((c_win - c_tile) / px_x) - round((c_win - c_tile) / px_x)) < 1e-6` on both axes (and keep
  the pixel-size test as a cheap second clause) — or delete it and remove the "verified clean" claim.

## Refuted by my own check

- **The Stage-3 sign convention is wrong somewhere** (a real historical bug, DECISIONS 2026-06-10c).
  Traced end to end and it is correct at every hop: skimage returns the translation to apply to
  `moving`; `nd_shift(moving, +shift)` is used consistently for the peak check; `shift_px_to_world_m`
  flips only the row component (`e < 0`); `labeling._apply_coreg_shift:85-93` *adds* `(dx_m, dy_m)` to
  HiRISE-derived polygons in the CTX CRS. `tests/test_coregister.py:111-123` is **not** circular — it
  builds the error geometrically ("terrain 50 m north appears 10 rows up") rather than from the module's
  own convention, and asserts the correction is `(-30, -50)` m. The cohort's uniform sign
  (`dy_px > 0` in 38/39 → `dy_m < 0`, labels pushed south) matches `DECISIONS.md:2543-2545`.
- **The decimated HiRISE read is a 400:1 nearest-neighbour point sample, so the Stage-3 reference is
  aliasing-dominated** (my strongest candidate for why no image reaches peak 0.9). Refuted: the cached
  JP2s expose wavelet resolution levels as GDAL overviews — `ESP_017355_2260` reports
  `overviews(1) = [2, 4, 8, 16, 32, 64, 128, 256, 511]` at 0.5 m/px — so `ds.read(out_shape=…)` reads a
  properly low-passed pyramid level and nearest-resamples over a small residual factor, not the full-res
  grid. The transform scaling at `hirise_imagery.py:194-199` is also correct (extent-preserving), as
  pass 1 found.
- **The JP2-side CRS override could substitute a wrong sphere radius or SP1** (it replaces the JP2's
  entire CRS with the shapefile's while keeping the JP2's transform, with no cross-check against the
  PDS `A_AXIS_RADIUS`). Checked **all 46** ObsIds with a cached `.LBL`: every Stage-1
  `source_crs_wkt` radius matches its PDS `A_AXIS_RADIUS` to < 0.5 m and every WKT `Standard_Parallel_1`
  equals the PDS `CENTER_LATITUDE` exactly. (Pass 1 verified 6; this extends it to the whole cohort.)
  The guard is still absent, but there is no discrepancy to find.
- **`cache_v2/hirise_decimated` is a junction to `cache/hirise_decimated` while
  `reprojected_detections` is not, so the two cohorts fight over one derived raster's CRS.** Confirmed
  the junctions (`ctx_tiles`, `hirise_decimated`, `hirise_jp2`, `pds_labels` are reparse points;
  `ctx_windows` and `reprojected_detections` are real directories), but the three shared ObsIds
  (`ESP_047976_2020`, `ESP_069669_2220`, `ESP_071093_2210`) have **byte-identical** `source_crs_wkt` in
  both cohorts, so `_crs_equal` never trips and no thrash or cross-cohort staleness occurs today.
- **`_crs_equal` may reject its own freshly-written cache forever** (if GDAL drops SP1 on the GeoTIFF
  round-trip while pyproj keeps it, every call would re-decimate a 200-500 MB JP2). Not reproducible
  without writing a raster, and pass 1's sidecar audit plus the presence of 55 stable files in
  `cache/hirise_decimated/` (including the SP1-corrected ObsIds) indicate the caches are being reused,
  not rebuilt.
- **`compute_window_bounds`'s empty-`gdf` `ValueError` is unreachable** — true (its only caller tests
  `len(gdf) > 0` first, `ctx_retrieve.py:573`), but it is a public-API message with no consequence.
- **The `/vsicurl/` HiRISE fallback has no GDAL SSL trust configured** (`truststore.inject_into_ssl()`
  patches Python's `ssl`, not libcurl; only `src/striping.py:168` and `src/validation_retrieve.py:59-63`
  set `GDAL_HTTP_UNSAFESSL`/`CURL_CA_BUNDLE`), so `hirise_imagery._open_source:157` and
  `ctx_retrieve.read_ctx_tile_crs:128` would fail on Windows. Unreachable in the pipeline —
  `ensure_jp2_local` always runs first in both production callers — and it fails loudly when reached.
- **`ctx_edr.COARSE_FACTOR = 32` is hardcoded against the abundance grid's tile size** — real coupling,
  but the module is dead-closed F-build machinery and the map is fixed at S=32.
- **`read_ctx_tile_crs` / `discover_murray_lab_url_template` cache to tile-independent filenames
  (`ctx_crs.wkt`, `ctx_url_template.txt`), so changing `probe_tile` returns a stale answer.** Both are
  on the superseded `target_crs: from_ctx_tile` branch (`ctx_retrieve.py:3-7`, `:152`), which no config
  selects.
- **A truncated CTX tile zip could be committed** — `_download_to`'s 50 MB floor plus the zip's
  end-of-file central directory (`zipfile.ZipFile` at `ctx_retrieve.py:293`) catch it loudly. Only the
  JP2 path is exposed (geo-crs-deep-2).
- **`pds_labels.read_label`'s first-occurrence-wins parse could pick a nested duplicate of
  `CENTER_LATITUDE` / `A_AXIS_RADIUS`.** Verified against all 46 cached labels via the radius/SP1
  cross-check above — every parse agrees with the shapefile-derived value, so no label has a
  conflicting earlier occurrence.

## Verified clean

- **Every affine composition in the six files, beyond R31.** `_snap_bounds_to_pixel_grid:350-358`
  (outward, tile-origin-anchored, idempotent, correct `e < 0` handling, accepts `Affine` or the 6-list
  sidecar form); `from_bounds(*bounds, …)` argument order (`bounds` is `(left, bottom, right, top)`);
  `read_full_footprint_decimated:194-199` (transform scaled by `width/out_w`, origin preserved — GDAL
  `out_shape` semantics); `read_native_window:251-252` (clip **then** `windows.transform(clipped, …)`);
  `build_hirise_coverage_mask:502-528` (allocates from the window's own `(height, width)`, writes with
  the window's own transform/CRS); `ctx_edr.frames_in_crop:52-57` (`t * (col, row)` is the correct
  rasterio order, `/COARSE_FACTOR` converts native→coarse pixels consistently on both corners).
- **The block-median solve's pixel↔metre conversion.** `px_x = abs(ctx_transform.a)`,
  `px_y = abs(ctx_transform.e)` (`:400-401`) taken from the CTX window itself, and the preserved
  `single_window` block (`:422-426`) applies the same `-dy·px_y` / `+dx·px_x` convention as
  `shift_px_to_world_m`, so the two records are directly comparable.
- **`< min_confident_blocks` handling.** `_robust_shift_from_field` returns `None` on an empty field or
  too few confident blocks and the caller falls back cleanly, recomputing `n_conf` for provenance
  (`:388-397`); NaN block peaks (produced when `margin*2 >= min(shape)`, `:227-228`) compare `False`
  against the floor and are excluded rather than poisoning the median.
- **The whole-image validation is not circular in its tiling.** `docs/methods.md:579` says the check
  re-tiles at 128 px, and the probes do (`scripts/probes/_diag_block_shift_field.py:33`,
  `_diag_fallback_explore.py:49`) while the solve uses `block_px: 256` — a genuinely different
  partition. (Its *summary statistic* is the problem, not its tiling — geo-crs-deep-1.)
- **Nodata handling into the correlation.** `select_fft_window` is fed
  `coverage_mask & (ctx_arr > 0)` (`:358-360`) and `block_shift_field` re-ands the same two conditions
  (`:489`), so neither CTX mosaic nodata nor HiRISE off-swath zeros can enter a correlated block; the
  `bilinear`-for-intensity vs `nearest`-for-mask split is deliberate and documented
  (`ctx_retrieve.py:473-476`).
- **Per-image local-radius CRS, cohort-wide.** 46/46 cached PDS labels agree with their Stage-1
  corrected WKT on both radius (< 0.5 m) and SP1 (exact); no Mars radius is hardcoded anywhere in
  `coregister.py`, `ctx_retrieve.py`, `ctx_tiles.py`, `hirise_imagery.py`, `pds_labels.py`,
  `ctx_edr.py`.
- **`_padded_manifest_form` / `manifest_to_murray` sign and padding logic**, and the 404-then-retry in
  `ensure_tile_cached:275-291` (only 404 is retried; the `UnboundLocalError ⊂ NameError` fallback at
  `:310-315` works when the zip pre-exists).
- **`pds_labels` platform TLS branch** (`:33-54`) and `_strip_units`' fail-loud behaviour on a
  non-numeric PDS value.

## Why the first pass found little

**Split verdict: the georeferencing arithmetic is genuinely sound; the module's *reported statistics*
were under-reviewed.**

The first pass was right about the hard part. I re-derived every affine composition, window rounding,
decimation-factor/transform pairing, sign flip and nodata mask in all 1,450 lines and found exactly one
defect in the geometry — R31, which pass 1 already has, measured correctly, and localised to the one
line where the array and its transform disagree. The classic traps the brief lists (decimate the array
and forget the transform; `set_crs` for `to_crs`; row/col vs x/y; clip-after-transform; a per-image CRS
read once and reused; a hardcoded radius) are all absent, and I extended pass 1's 6-file radius spot-check
to the whole 46-image cohort with zero discrepancies. For a subsystem this fiddly, one finding in the
geometry is a fair result, not an under-review.

What pass 1 missed is a different axis: it audited what the module **computes** and never audited what
the module **publishes**. `peak_correlation` is the only quality number Stage 3 emits, it accompanies
every shipped label set, and it is both (a) bounded below by the threshold it is screened against and
(b) a fit statistic for a different model than the one applied — yet pass 1 mentions it only inside
`geo-crs-2` as "no `peak_correlation` floor", i.e. it noticed the floor was *missing* without noticing
that adding one there would be vacuous. That is the same Pattern A pass 1 correctly identified in the
km-scale guard, one level up, and it is the more consequential instance because a decision
(`DECISIONS.md:2376-2396`, the cohort screen) and a published figure (`methods.md` Figure 3) rest on it.

Two smaller misses have a common cause: pass 1's refutations were reasoned from library semantics rather
than from the library source or the data. Its `IncompleteRead` refutation is contradicted by an explicit
comment in CPython, which killed a live silent-failure path; and its clean bill on the nominal-footprint
fallback (shared with `labeling.md`) checked the CRS and never the *units*, which the image's own cached
PDS label settles in one line. Both were cheap to check and neither needed the network. Conversely, my
own strongest new candidate (aliased 400:1 decimation) died on exactly this kind of check — the JP2s
carry overview pyramids — so the discipline cuts both ways.

Correctly skipped by pass 1: the dead Stage-0.5 probes, `ctx_edr`'s hardcoded coarse factor, and the
unreachable `compute_window_bounds` guard. Its `geo-crs-4` (the dead `clipped.width <= 0` branch) is the
right kind of find and I confirmed there is only one more of that shape in these files, in `labeling.py`
rather than here (geo-crs-deep-4).

## Coverage note

Read in full: `src/coregister.py` (560), `src/ctx_retrieve.py` (627), `src/hirise_imagery.py` (263),
`src/ctx_tiles.py` (51), `src/ctx_edr.py` (62), `src/pds_labels.py` (141), `tests/test_coregister.py`
(296). Read the relevant slices of `src/labeling.py` (`_apply_coreg_shift`, `_load_mosaic_transform`,
the Stage-4 wiring and pixel-size guard, the provenance writer), `scripts/run_stage3.py`,
`scripts/run_stage4.py`, `scripts/probes/_w1_coreg_vs_auc.py`, `docs/methods.md` §5.2-5.5,
`dataset/DATA_DICTIONARY.md` Stage-3 table, `config.yaml` / `config_v2.yaml` retrieval + coregistration
blocks, and the installed `http.client` source. Grepped `DECISIONS.md` by term (`peak_correlation`,
`block_peak_min`, `noise floor`, `ESP_046803_2325`, `ESP_065711_1545`, `36 of 38`,
`GDAL_HTTP_UNSAFESSL`, `image_footprint`) and cross-checked `docs/CODE_REVIEW_2026-07-31.md` §3-§5 plus
`geo-crs.md`, `labeling.md`, `features.md`, `fm-embeddings.md`, `other-scripts.md` so nothing here
duplicates R01/R13/R21/R29/R30/R31/R47 or `geo-crs-1..4` / `labeling-2..4`.

Measurements are from read-only inspection of cached **sidecars** (48 `coregistration/*.json`, 49
`ctx_windows/*.json`, 46 `reprojected_detections/*.json`, 46 `pds_labels/*.LBL`), a directory listing of
`cache/hirise_jp2` and `cache/hirise_decimated`, `Get-ChildItem` reparse-point flags on `cache*/`, and
one **header-only** `rasterio.open()` on a cached JP2 to enumerate its overview levels (no pixels read).
No network, no notebooks, no CTX/HiRISE pixel reads, no re-runs, no files written outside this one.

Could **not** check: (1) GDAL's actual behaviour on a truncated JPEG2000 codestream (needs a network
failure to reproduce), so geo-crs-deep-2's downstream symptom is bracketed rather than pinned;
(2) whether GDAL's `RasterIO` really selects a JP2 overview for the 0.5 m → 5 m read on this build (the
overviews exist and the observed peaks are consistent with a low-passed downsample, but confirming the
selection needs a `CPL_DEBUG` read); (3) `_crs_equal`'s GeoTIFF round-trip behaviour (would require
writing a raster, which the rules forbid); (4) whether the two `slow` Stage-3 integration tests
currently pass — they also **rewrite** `cache*/coregistration/ESP_069669_2220.json` when they run, which
is a `tests`-area concern I did not pursue.
