# Review area: labeling-deep-footprint

- **Reviewed at commit:** 7bfedb8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)

> **Headline answer to the sub-area's question: the "BoulderNet inference gap" hypothesis is
> REFUTED.** BoulderNet's detection footprint equals the HiRISE image footprint to within ~40 m on
> all four sides of all 38 in-cohort images; detection density *rises* to 1.51 × the image mean in
> the last 40 m before the coverage boundary rather than falling; and there is no chip-, stride-,
> or CCD-aligned periodicity in the density field. See **Refuted by my own check** for the
> measurements and the false-positive reasoning.
>
> **But the coverage mask is wrong in the other direction**, and that is finding 1. The mask is
> built from a DN threshold, not from image geometry, and it misclassifies deep-shadow pixels as
> "HiRISE did not observe here". Because eligibility requires *every* mask pixel in a tile to be 1,
> single 5 m shadow pixels silently delete whole 160 m tiles from the dataset — and the deleted
> tiles are **93.0 % rich against a 36.0 % base rate**. The label basis is censored on the outcome
> at the top of the distribution, not poisoned with false zeros at the bottom.

## Findings

### labeling-deep-footprint-1 — The HiRISE coverage mask calls deep-shadow pixels "no coverage"; because eligibility is `all(mask==1)`, this silently deletes 1.97 % of S=32 tiles that are 93 % rich and hold 7.70 % of all detected boulder area
- **Severity:** high
- **Liveness:** live-shipped (`dataset_v2/labels` is the training/eval basis of the frozen recipe, the deployed head `models/deployable/86c51a5dca220f63`, the banked calibrator, and the shipped mosaic map)
- **Confidence:** high (mechanism proven from the decimated HiRISE DN values; blast radius measured on all 38 images)
- **Where:** `src/ctx_retrieve.py:507` (`valid_src = (hi_arr > 0)`), docstring claim at
  `src/ctx_retrieve.py:470-472`; amplified by `src/labeling.py:276-279` and the ladder `.all()` at
  `src/labeling.py:325`; input produced by `src/hirise_imagery.py:192` (nearest-neighbour
  decimation); rule's rationale at `DECISIONS.md:301-327`

`build_hirise_coverage_mask` defines "HiRISE observed this ground" as `hi_arr > 0` on a 5 m/px
**nearest-neighbour** decimation of the RED product. HiRISE RDR DN is *not* gapped above zero — the
DN histogram runs continuously through 0 (DN = 1,2,3,… each with hundreds of pixels per image), so
0 is the bottom of the real radiometric distribution, not a reserved nodata sentinel. Genuinely
imaged, very dark pixels therefore become `mask = 0`. Stage 4 then requires **every** mask pixel in
a tile to be 1, and propagates ineligibility upward with `.all()`, so one 5 m pixel deletes the
40 m tile, the 80 m tile, the 160 m tile and the 320 m tile that contain it. The pixels that go
dark are the ones next to boulders, which is why the deleted tiles are the rockiest ones.

The `> 0` rule was intended for "rotated-rectangle corners, missing scans"
(`src/ctx_retrieve.py:472`) — i.e. for geometry. Interior zeros are not missing scans: in five
images I censused them and **99 % are isolated single pixels** (`ESP_076499_1160`: 2,130 interior
zero components, 2,115 of size 1 px, largest 2 px). A dropped scan or a CCD gap is a contiguous
line. And the 8-neighbour ring around each interior zero is systematically **darker** than the
image mean in 5 of 5 images (505 vs 610, 560 vs 618, 593 vs 688, 383 vs 503, 279 vs 389 DN) —
the signature of shadow, not of a random dropout.

- **Failure scenario:** a 160 m tile of boulder-strewn ground contains one 5 m cell whose sampled
  0.25 m HiRISE pixel sits in a boulder's shadow at DN 0. The mask marks that cell unobserved, the
  tile fails `mask_min == 1`, and the tile is dropped from `dataset_v2/labels` entirely — its
  features are never computed, it never enters a LOIO fold, and it never enters the calibrator's
  quantile grid. Across the cohort this removes **3,236 of 164,273 S=32 tiles (1.97 %)** whose
  mean `fractional_area` is **0.0613 vs 0.0148 for the tiles kept (4.15×)**, i.e. the pipeline
  systematically deletes the top of the abundance distribution. The deployed abundance layer is
  quantile-matched to a label distribution whose rich tail has been thinned, so the map's upper
  range is compressed by construction; the rich prevalence the recipe reports (0.3598) understates
  the cohort's true 0.3733; and every prevalence-dependent statistic (`pr_auc@1e-2`,
  `precision@5%`) is computed on a set from which the easiest-to-detect rich tiles were removed.
- **Evidence:**
  ```
  src/ctx_retrieve.py:470-472  (docstring — attributes interior zeros to geometry)
      "The mask is 1 where the decimated HiRISE (5 m/px) has a valid (non-zero) pixel after
       reprojection ... AND inside the swath where HiRISE itself has NaN/0 pixels
       (rotated-rectangle corners, missing scans)."
  src/ctx_retrieve.py:507
      valid_src = (hi_arr > 0).astype(np.uint8)

  src/hirise_imagery.py:192      arr = ds.read(1, out_shape=(out_h, out_w))   # nearest by default
      # -> one 0.25-0.5 m source pixel decides the validity of a whole 5 m cell

  src/labeling.py:277-279
      mask_crop = mask[r0:r1, c0:c1]
      mask_min  = mask_crop.reshape(n_jr, F, n_jc, F).min(axis=(1, 3))
      eligible  = (mask_min == 1)
  src/labeling.py:325
      eligible_k = prev["eligible"].reshape(ny, 2, nx, 2).all(axis=(1, 3))

  # --- measured (read-only; cache_v2/hirise_decimated 5 m/px cache, never the full-res JP2) ---
  ESP_076499_1160: DN hist 0..12 = [2570957, 356, 311, 273, 267, 243, 236, 264, 338, 407, 435, 376, 311]
                   interior zeros 2,145 px; components n=2130, 2115 of size 1 px, max 2 px
                   mean DN in ring around interior zeros 505  vs image mean(DN>0) 610
  ESP_017355_2260: interior zeros 1,185 px  ->  1,126 interior-ineligible S=8 tiles   (0.95 : 1)

  # --- blast radius, all 38 in-cohort images, S=32 (the frozen recipe's scale) ---
  interior tiles dropped by the mask : 3,236  (1.97 % of the 164,273-tile interior grid;
                                               median 21/image, max 1,127 = ESP_076499_1160;
                                               0 images unaffected)
  detection density in dropped vs kept: 3.04x  (count-weighted pooled over 38 images)
  rich (fa>1e-2) share, dropped tiles : 0.930   vs kept 0.362  (shipped labels 0.360 — my
                                                 area estimator reproduces it to 0.002)
  mean fa, dropped vs kept           : 0.0613 vs 0.0148  (4.15x)
  boulder area inside dropped tiles  : 5,075,686 m2 of 65,889,210 m2 = 7.70 %
                                        (per-image median 1.76 %, p90 10.08 %, max 37.29 %;
                                         17 of 38 images lose >2 % of their boulder area)
  ```
- **Self-refutation attempted:** (a) *"They are genuine HiRISE data gaps."* Dead — 99 % of interior
  zeros are isolated single pixels and the DN histogram is continuous through 0, so there is no
  sentinel gap below the data floor. (b) *"Then the pixels really are unusable."* Dead, and this is
  the decisive one: **BoulderNet detected boulders inside those tiles at 3.04× the average
  density**, so the underlying full-res HiRISE data is plainly present and interpretable; only the
  5 m nearest-neighbour sample of it is dark. (c) *"`binary_fill_holes` mis-labels swath-edge
  concavities as interior."* Checked independently — the interior-hole count matches the interior
  DN==0 census nearly 1:1 (`ESP_017355_2260`: 1,185 zero pixels → 1,126 holes), which is the exact
  prediction of the one-pixel-kills-one-tile mechanism. (d) *"Deliberate / documented."* Grepped
  `DECISIONS.md` for `hi_arr`, `valid_src`, `coverage mask`, `missing scan`, `DN 0`, `eligib`,
  `hirise_coverage_fraction`: the 2026-05-21 entry (`DECISIONS.md:301-327`) records the *intent*
  ("zero-inflation as a statistical property (real) and zero-inflation as a measurement artifact
  (avoidable)") and the `== 1.0` eligibility choice, but nothing anywhere anticipates that the DN
  proxy would misfire on dark scene content. The entry's own framing is the irony: the mask was
  added to remove one measurement artifact and introduces the opposite one. (e) *"A test pins it."*
  `tests/test_labeling.py:327-338`
  (`test_mask_gating_drops_tiles_with_any_uncovered_pixel`) pins the `all(mask==1)` **rule**, which
  is correct *given a correct mask*; every label test builds the mask as a synthetic
  `np.full(..., mask_fill)` fixture (`tests/test_labeling.py:81`), so no test ever exercises
  `build_hirise_coverage_mask` against a dark-but-valid pixel. The defect is unpinned. (f) *"Blast
  radius is one bad image."* No — all 38 images are affected, median 21 S=32 tiles each, and 17 of
  38 lose more than 2 % of their boulder area. (g) I could not determine the **sign** of the effect
  on `meaningful_auc`: the removed tiles are rich and probably visually rocky in CTX, so their
  absence could depress or inflate AUC. I am claiming the prevalence shift, the censored fa tail
  and the compressed map ceiling, not an AUC direction.
- **Fix:** one line at the producer. After building `mask` in
  `build_hirise_coverage_mask` (`src/ctx_retrieve.py:508-519`) and before writing, fill interior
  holes — `mask = ndimage.binary_fill_holes(mask).astype(np.uint8)`, optionally gated on component
  size so a genuine large interior data gap would still be honoured. Every interior hole measured
  is 1–2 px, so this recovers all 3,236 tiles. Better still, stop inferring coverage from DN: build
  the mask from the product's valid-data geometry (the source nodata/alpha mask, or the PDS
  footprint polygon) so radiometry cannot masquerade as absence. Then re-run Stage 4/4b; note that
  `hirise_coverage_fraction` in every `ctx_windows/*.json` and the `eligible_tiles_per_scale` in
  every `labels/*.json` change, so the label artifacts must be re-emitted, not patched.

---

### labeling-deep-footprint-2 — `labeling-2` measured, and materially corrected: the coreg-shift/mask mismatch puts 3.89 % of S=32 tiles on ground HiRISE did not image after alignment (pass 1 estimated ~2 %), it affects the *western* edge as well as the southern, and it pushes 82,210 detections out of the labelled area
- **Severity:** medium (pass 1 filed it as **low**)
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `src/labeling.py:474-478` and `src/labeling.py:85-93`; mask producer
  `src/ctx_retrieve.py:459-531`

This is pass 1's `labeling-2`; I am **not** re-filing the mechanism, I am replacing its analytic
estimate with a measurement and correcting two things it got wrong. Pass 1 said "roughly 1–2 rows
of tiles along the **southern** boundary … ~2 % of tiles, ~60 S=32 tiles per image". Measured: the
shift is +dy (north) in **38 of 38** images *and* +dx (east) in **30 of 38**, so the strip is an
**L along the southern *and* western** edges, and the affected count is **165 S=32 tiles per image
(median 114, max 953)**, 2.7× pass 1's figure.

I re-derived tile eligibility directly from the cached `*_hirise_mask.tif` rasters and the
`mosaic_row_origin`/`mosaic_col_origin` in each label sidecar; the recomputation reproduces the
shipped eligible tile set **exactly** on all 38 images (asserted, no mismatches), so the
counterfactual below is exact and not a model.

- **Failure scenario:** a tile at the southern edge of the swath is eligible because the *unshifted*
  mask covers it, but every detection that belonged there has been translated ~200 m north, so the
  tile is labelled `fractional_area = 0` while its CTX feature vector is ordinary terrain. 2,502
  S=32 tiles are currently labelled exactly 0 on ground that is not HiRISE-covered after alignment
  — **7.61 % of the 32,876 zero tiles at S=32**. These are the only genuine false zeros I found in
  the cohort, and they are structured: always at a swath edge, always on the same two sides.
- **Evidence:**
  ```
  src/labeling.py:474-478
      shift = coregister.load_shift(obs_id, cache_dir) if apply_coreg_shift else None
      gdf = _apply_coreg_shift(gdf, shift)          # polygons move
      with rasterio.open(mask_tif) as src:
          mask = src.read(1)                        # mask does not

  # pooled detection presence / image mean, at 40 m from the eligible-region edge,
  # split by which boundary is nearest (38 images, S=8 grid):
  #                        N              S              W              E
  #   UNSHIFTED     1.180+-0.083   1.273+-0.081   1.112+-0.051   1.142+-0.063
  #   AS LABELLED   1.098+-0.090   0.396+-0.063   0.553+-0.060   0.984+-0.071
  #   -> the S edge loses 60 % and the W edge 45 % of its detections; N and E are untouched.
  #      Unshifted there is NO edge deficit in any direction, which is what proves the
  #      deficit is the shift and not the detector.

  # counterfactual: eligibility recomputed against the mask translated by the same (dx,dy)
  S=32 : 6,269 of 161,037 eligible tiles (3.89 %) are not fully HiRISE-covered after alignment
         mean post-alignment coverage of such a tile 0.740; 1,464 are <50 % covered;
         345 are 0 % covered; 2,502 are currently labelled fa == 0  (7.61 % of the zero class)
         3,923 have fa <= 1e-2  = 3.72 % of the 105,459-tile "poor" class of fa_gt_1e-2
  S=8  : 67,970 of 2,700,653 (2.52 %); 55,879 labelled fa == 0 = 4.12 % of the zero class

  # the converse the brief asked about — detections pushed OUT of the labelled area:
  detections total 5,911,846; landing outside an eligible tile: 86,078 (1.46 %) unshifted
  -> 168,288 (2.85 %) after the shift.  The shift alone discards 82,210 detections (1.39 %).
  (Only 732 detections fall outside the CTX window at all, so window sizing is not the issue.)
  ```
- **Self-refutation attempted:** (a) *"The edge deficit is really a detector margin, not the
  shift."* Killed by the control: with the **unshifted** centroids the deficit is absent in all
  four directions and the edge is *enriched* (S: 1.273, W: 1.112). Only the shifted set shows it,
  and only on the two sides the shift recedes from. (b) *"It is the same finding, so do not
  report."* I am reporting it only because two of pass 1's specifics are wrong (one direction vs
  two, 2 % vs 3.89 %) and because the 82,210 silently discarded detections are a consequence pass 1
  did not identify. (c) *"Severity should stay low."* Raised to medium: the affected tiles are
  ~4 % of the frozen recipe's training/eval set, the error is one-directional, and it lands
  entirely at swath edges where the striping/F programme did its per-image level measurements.
  (d) Grepped `DECISIONS.md` for the 2026-05-23 shift decision: the choice to translate polygons
  rather than resample the grid is recorded; the mask is never mentioned. Unchanged from pass 1.
- **Fix:** as pass 1 — translate the mask by the same `(dx, dy)` before gating, or (strictly
  conservative, no resampling) erode the eligible mask by `ceil(|shift| / px)` pixels. Given
  finding 1 requires the mask to be rebuilt anyway, doing both in one pass is the economical route.

---

### labeling-deep-footprint-3 — Extension of R23 / `labeling-1`: the score-rank truncation does not just depress `fa`, it moves ~2,200 of `ESP_017355_2260`'s tiles into the wrong class of the frozen recipe's actual target
- **Severity:** medium (the parent finding R23 is already filed as **blocker** — this is its
  consequence on the reported surface, not a new defect)
- **Liveness:** live-shipped
- **Confidence:** medium (the flip rate is transferred from clean images; the transfer is
  prevalence-matched but is still a transfer)
- **Where:** consequence of `src/detections.py:112-127` / `DECISIONS.md:1194-1201`, materialising
  at `src/labeling.py:509-515`

R23 / `labeling-1` established that `ESP_017355_2260` and `ESP_068483_2280` are labelled at
effective confidence floors of 0.617 and 0.407 while the other 36 images sit at 0.10, and quantified
the damage as `fa` being "2.5–4.5×" and "1.4–1.7×" too low. The frozen recipe does not train on
`fa`; it trains on **`fa_gt_1e-2`**. I applied each truncation's score floor to the 36 clean images
and counted class flips at S=32. The flip rate is strongly prevalence-dependent
(Spearman ρ = −0.757 against the image's rich share), so a pooled rate would mislead; matching to
each affected image's richness gives:

| floor | matched cohort | rich→poor flip rate | boulder area retained |
|---|---|---|---|
| 0.6173 (`ESP_017355_2260`) | 8 images, rich share > 0.55 | **41.3 %** | 0.386 |
| 0.4067 (`ESP_068483_2280`) | 4 images, rich share > 0.75 | **5.2 %** | 0.408 |
| (pooled over all 36, for reference) | 36 images | 51.8 % / 19.8 % | 0.326 / 0.680 |

- **Failure scenario:** `ESP_017355_2260` ships 13,457 S=32 tiles at a rich share of 0.625 (8,408
  rich). Solving the transfer self-consistently (a richer true image flips at a lower rate) puts its
  true rich share near 0.79, i.e. **≈ 2,200 tiles — 16 % of the largest observation in the cohort,
  1.4 % of all 161 k tiles — are currently in the `poor` class of `fa_gt_1e-2` when they are truly
  rich.** They are false negatives in the target itself: the head is trained to call rocky CTX
  texture "poor" on the cohort's biggest image, and evaluated against the same wrong labels in its
  LOIO fold. `ESP_068483_2280`'s corresponding figure is ≈ 235 tiles (0.85 → 0.81 rich share) and is
  immaterial. The matched area-retention 0.386 also **narrows pass 1's "2.5–4.5×" fa depression to
  ≈ 2.6×** for `ESP_017355_2260`.
- **Evidence:**
  ```
  # per-image rich->poor flip rate at score >= 0.6173, S=32, 36 clean images (excerpt, sorted
  # by shipped rich share) — the rate collapses as the image gets richer, so pooling overstates it:
  #   ESP_047976_2020 rich 0.002 -> flip 1.000 | ESP_063429_2240 rich 0.544 -> flip 0.386
  #   ESP_054397_2105 rich 0.015 -> flip 0.902 | ESP_045139_2270 rich 0.724 -> flip 0.598
  #   ESP_069669_2220 rich 0.019 -> flip 0.711 | ESP_053989_2260 rich 0.797 -> flip 0.146
  #   ESP_046959_2225 rich 0.255 -> flip 0.496 | ESP_054622_2240 rich 0.975 -> flip 0.167
  #   Spearman(rich share, flip rate) = -0.757
  # matched-cohort flip rate at 0.6173 for rich-share ~0.625 images: 23,587/45,530 pooled -> 0.413 matched
  ESP_017355_2260: shipped 13,457 S=32 tiles, rich share 0.625 (8,408 rich)
  ESP_068483_2280: shipped  5,297 S=32 tiles, rich share 0.807 (4,275 rich)
  ```
- **Self-refutation attempted:** (a) *"This is just R23, do not re-file."* I am not re-filing the
  mechanism; R23's own damage estimate is stated in `fa` units, and the shipped model's target is a
  threshold on `fa`, so the class-flip count is the number a fix actually has to justify. (b)
  *"The pooled 51.8 % is the number."* No — I killed my own first pass at this. The flip rate is
  prevalence-dependent (ρ = −0.757) and applying the pooled rate to `ESP_017355_2260` produces an
  impossible true rich count (> 100 %), which is exactly the tell; the matched estimate is what
  survives. (c) *"Does it overturn the abort verdict?"* No, for the same reason pass 1 gave —
  `sd(log10 mosaic_ratio)` moves 0.1744 → 0.1755 without the image. R10 stands.
- **Fix:** as R23 — re-export or exclude the two images, per the `ESP_028537_2270` precedent, and
  add the `.shp` self-declared-length integrity assert. Nothing new here; this finding only sizes
  the consequence.

## Refuted by my own check

- **THE SUB-AREA'S PRIMARY HYPOTHESIS — "BoulderNet's inference footprint is smaller than the HiRISE
  image footprint, or has interior gaps, so tiles in those gaps enter training as genuine rock-free
  ground truth." REFUTED, on four independent tests.** Stating the false-positive reasoning
  explicitly, because a zero region is not evidence of a gap (the target is zero-inflated by design;
  pooled rich share 0.3598):
  1. **No crop.** The HiRISE-valid mask ends *inside* the CTX window on all four sides in **38 of
     38** images (`mask_touch_{top,bot,left,right}` all False), so the footprint comparison is never
     clipped by the window and is therefore measurable. The HiRISE-valid ground lying beyond the
     most extreme detection is a median of **21–29 m per side** (north 28.2, south 21.2, west 28.8,
     east 26.0; max 585 m). The detector reaches the image edge.
  2. **No margin.** Pooled over 38 images, **unshifted** detection *density* at 40 m from the
     eligible-region boundary is **1.506 ± 0.112 × the image mean**, decaying monotonically to
     1.124 at 320 m — the edge is *enriched*, not depleted, in every one of the four directions
     (N 1.180, S 1.273, W 1.112, E 1.142 on the presence metric). A real inference margin would
     have to be narrower than one 40 m cell to hide inside this. Geology cannot produce this
     signal: swath boundaries are arbitrary with respect to terrain, so pooling 38 independent
     observations averages geology out — which is precisely why the *shifted* version of the same
     statistic does show a deficit (finding 2). The test has demonstrated power.
  3. **No detector grid.** De-trended detection density, measured in ground metres relative to the
     swath edge using swath axes taken from the mask itself, shows a flat red-noise spectrum
     (908 cross-track bins, 1231 along-track, 38 images): the largest peak is only 1.4 × the median
     band amplitude and sits at an incoherent 89 m. At the physically motivated periods the
     amplitude is *below* median — cross-track **512 m (HiRISE CCD pitch) ratio 0.44**, 256 m
     (SAHI `ss-256` slice) 0.82, 204.8 m (stride at `ov-020`) 0.74, 1024 m (`is-1024`) 0.31;
     along-track 0.39 / 0.64 / 0.83 / 0.20. No chip, stride, or CCD seam exists in the density
     field.
  4. **No geometric holes.** Censusing every connected zero-detection component on the eligible
     grid: 150 components of ≥ 25 cells across 19 images have a ring ≥ 3 × denser than the image
     mean, but their **rectangularity (filled fraction of the bounding box) has median 0.42 and
     maximum 0.859** — none is a rectangle. The largest components are single amorphous blobs
     spanning up to 92 % of an image, i.e. the geology of a boulder-poor scene, and 137 of 150 do
     not even touch the grid edge yet are still amorphous. A skipped chip would be rectangular with
     rectangularity ≈ 1.0 and a size matching the slice grid; nothing of that shape exists.
  **Residual risk I am not claiming to have excluded:** an inference gap narrower than ~40 m, or
  one whose shape is neither rectangular nor periodic and which happens to coincide with
  boulder-free geology in every image. Both are unfalsifiable from the cached products and both are
  negligible in magnitude by construction. Per the brief's instruction, the verdict is REFUTED.
- **"Per-image dead rows/columns in the tile grid are detector gaps."** They are not. At S=8, 30 of
  38 images have ≤ 10 fully-empty rows and ≤ 10 empty columns out of 200–670, with a longest
  contiguous run of 3–7. The worst case (`ESP_055690_2200`, 53 empty rows of 461) has a longest run
  of 7 and a zero share of 0.774, and the empty rows are scattered, not contiguous. Under a
  detector-gap hypothesis they would be contiguous and full-width; under geology they are scattered,
  which is what they are.
- **"Detections falling outside the coverage mask are silently dropped."** Only 732 of 5,911,846
  detections fall outside the CTX window at all, so the polygon-bbox + 1 km window is not clipping
  the detection set. 1.46 % land in tiles the mask marks partially uncovered — those tiles are
  *dropped*, not zeroed, so no label is poisoned, and dropping them is the recorded 2026-05-23
  decision. The only excess above that baseline is the 1.39 % attributable to the coreg shift,
  filed above.
- **"`drop_null_geometries` removes detections in a spatially structured way."** The removals are a
  score-rank truncation, not a spatial one (R23 / `labeling-1` proved this from the `.dbf` byte
  order). I checked the spatial consequence rather than re-checking the mechanism: it materialises
  as a *class* bias, not a footprint bias — see finding 3.
- **"The mask could be misaligned with the CTX window, so eligibility is offset."** It is not. I
  recomputed tile eligibility from scratch out of the raw `*_hirise_mask.tif` plus each sidecar's
  `mosaic_row_origin`/`mosaic_col_origin`, and it reproduced the shipped eligible tile set exactly
  on all 38 images (hard assert, zero mismatches). Independently confirms pass 1's "Verified clean"
  entry on mask/window shape agreement, and pins the grid-alignment arithmetic on real data rather
  than on the synthetic fixtures the tests use.
- **"Nearest-neighbour reprojection could punch holes in the mask during the warp."** It cannot —
  `reproject(..., Resampling.nearest)` assigns every destination pixel from the nearest source
  pixel. I confirmed the holes originate upstream by censusing interior DN==0 pixels in the
  decimated HiRISE itself and matching the counts ~1:1 to the interior mask holes.

## Verified clean

- **Detection footprint vs image footprint** — equal to within ~40 m on all four sides of all 38
  images, by four independent tests (above). This closes pass 1's "Could not check" item.
- **CTX window sizing** — `footprint_source: polygon_bbox` + 1 km buffer never truncates the
  detection set (732 of 5.9 M detections outside, i.e. 1.2e-4), and never clips the HiRISE mask
  (0 of 38 images have the mask touching a window edge). Extends pass 1's window audit.
- **Grid alignment on real data** — the shipped eligible tile set is exactly reproducible from
  (mask raster, `mosaic_row_origin`, `mosaic_col_origin`, `tile_sizes_px`) on all 38 images. The
  `_compute_grid_alignment` integer arithmetic and the `mask[r0:r1, c0:c1]` crop are correct in
  production, not just in the fixtures.
- **`_count_centroids_per_finest_cell` binning** — my independent re-implementation of the centroid
  →cell mapping reproduces the shipped `boulder_area`-derived rich share to 0.002 (0.362 vs 0.360),
  so the floor/offset arithmetic at `src/labeling.py:240-250` is right.
- **`reproject` nearest-neighbour choice for the mask** (`src/ctx_retrieve.py:518`) — correct and
  correctly justified; bilinear would smear the swath boundary. The defect in finding 1 is the
  `> 0` predicate that feeds it, not the resampling of it.
- **The `all(mask == 1)` eligibility rule itself** — correct given a correct mask, correctly
  propagated upward by `.all()`, and correctly pinned by
  `tests/test_labeling.py:327-338`. I am not proposing to relax it; I am proposing to fix its input.
- **Shift magnitudes** — `|shift|` min 80 m, median 200 m, max 327 m over 38 images, all O(200 m)
  as invariant 2 requires; `dy > 0` in 38/38, `dx > 0` in 30/38.

## Coverage note

**Read in full:** `src/labeling.py` (604), `src/ctx_retrieve.py:440-531`
(`build_hirise_coverage_mask` + its callers), `src/hirise_imagery.py:160-211`
(`read_full_footprint_decimated`), `docs/review_2026-07-31/labeling.md`,
`docs/review_2026-07-31/_prompts.md` §1/§3, `_prompts_labeling_deep.md`.
**Read partially:** `tests/test_labeling.py` (the mask fixture at :43-105 and the gating test at
:327-338), `tests/test_stage2_one_image.py:40-80`, `config_v2.yaml`, `scripts/run_stage4.py`,
`scripts/probes/_w1_shadow_threshold_diag.py`, `scripts/probes/_w1_dn_clip_extent.py` (both are
about **CTX** DN, not HiRISE DN — neither covers finding 1).
**Grepped:** `DECISIONS.md` by term for `hi_arr`, `valid_src`, `coverage mask`, `missing scan`,
`shadow.*nodata`, `DN=0`, `DN 0`, `eligib`, `hirise_coverage_fraction`,
`Labels-only-on-HiRISE`; the whole repo for `build_hirise_coverage_mask` / `valid_src` /
`hirise_mask` call sites.

**Measurements I ran** (read-only; `dataset_v2/labels/*.parquet` + `*.json`,
`cache_v2/reprojected_detections/*.gpkg`, `cache_v2/ctx_windows/*_hirise_mask.tif`,
`cache_v2/coregistration/*.json`, `cache_v2/hirise_decimated/*_5mpp_full.tif` — **the 5 m/px
decimated cache only, never a full-res JP2**; no network, no notebooks, no training):
per-image detection-presence grids at S=8 with dead-row/column and run-length census (38 images);
cross-track and along-track presence profiles with swath axes from PCA on the coverage mask;
distance-to-eligible-edge presence *and* density profiles, per direction, for shifted and unshifted
detections; mask-vs-window-vs-detection extent audit; connected-component hole census with
ring-density and rectangularity tests; de-trended cross/along-track density spectra with explicit
checks at 512/256/204.8/128/1024 m; exact eligibility recomputation and a mask-shift counterfactual
at S=8 and S=32; interior-DN==0 census with component sizes and neighbourhood DN on 5 images;
rich-share and boulder-area accounting for mask-dropped tiles; and score-threshold class-flip rates
at S=32 on the 36 clean images with prevalence matching.

**Could NOT check:** whether an inference gap narrower than one 40 m cell exists (below the
resolution of every artifact on disk); whether the interior DN==0 pixels are boulder shadow
specifically versus isolated dead detector pixels (both give the same conclusion — real terrain
misclassified as unobserved — so I did not pursue it, and it would need the full-res JP2 to settle);
the S=64 magnitude of finding 1 (the `.all()` propagation means it is monotonically worse at coarser
scales than the measured S=8 → S=32 progression of 0.24 % → 1.97 %, but I measured only the frozen
recipe's S=32 and the finest S=8); and the sign of finding 1's effect on `meaningful_auc`, which
would require re-running the recipe on restored labels and is out of scope for a read-only pass.
