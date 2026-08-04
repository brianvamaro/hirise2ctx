# Review area: numerics

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (single-agent pass; not independently verified)

## Findings

### numerics-1 — A1's uint8 clip floor is `0`, which is the mosaic **nodata sentinel**, so dark valid CTX pixels become "nodata"
- **Severity:** high
- **Liveness:** live-shipped (A1 is the documented mitigation; its head, its LOIO cost and its η² payoff are all on record) — see R06 for the missing map
- **Confidence:** high on the mechanism and on reachability; the affected *native-pixel* count is unmeasured (imagery is out of bounds for this review)
- **Where:** `src/striping.py:251` (+ docstring `:243`, `:246-247`; `a1_stats` `:236`), duplicated at
  `scripts/striping_a1_map.py:93` and `scripts/striping_a1_infer_crop.py:88`; applied to the A1
  **training** embeddings at `scripts/probes/_w2_fang_embed.py:208-211`; consumed at
  `src/mapping.py:95` and `src/mapping.py:256-257`; η² footprint consequence at
  `scripts/striping_a1_infer_crop.py:98-101`.

`a1_apply` remaps DN by `(x − med)/iqr·27.7 + 125` and clips to **`[0, 255]`**. Everywhere else in this
repo the same kind of uint8 stretch clips to **`[1, 255]`** precisely because `0` is the Murray-mosaic
nodata value — including `scripts/f_pilot_crop.py:165`, which is the *same A1 formula* (`A1_M0=125`,
`A1_S0=27.7`) re-implemented for the F pilot with `1, 255`. Consequently any valid pixel with
`x ≤ med − (125/27.7)·iqr = med − 4.513·iqr` is written as `0` and is thereafter indistinguishable from
nodata: `own_tile_zero_fraction` counts it (`(box == 0).mean()`), and `predict_window` masks the whole
tile when that fraction exceeds `max_zero_fraction`. Inside surviving tiles the clip also collapses the
entire dark tail to one value, so the docstring's "within-frame texture is preserved" is false exactly
where the model's shadow signal lives.

- **Failure scenario:** a source frame with `median = 170`, `IQR = 19.7`
  (`B16_016011_2180_XI_38N012W`, row 2 of `reports/figures/striping_frame_radiometry.csv`) has a
  zero-clip threshold of **+81.3 DN**. Every 5 m pixel in that frame darker than DN 81 — i.e. exactly
  the boulder shadows the frozen recipe keys on — is written as 0. Tiles with enough of them are then
  dropped as "nodata", so the A1 arm scores on a *smaller, shadow-depleted, non-random* subset of tiles
  than the baseline arm. `scripts/striping_a1_infer_crop.py:98-101` builds `fb` and `fa` from each
  arm's own `np.isfinite(...)` mask and compares `eta2` across them, so the published "≈28 % η²
  reduction" mixes a coverage difference with a mitigation effect — the very confound
  `f_map_compare.quality_table` and gate 1 were later rewritten to eliminate. The same clip is baked
  into the A1 training embeddings, so the −0.024 LOIO AUC cost is measured through it too.
- **Evidence:**
  ```
  src/striping.py:240-253
      def a1_apply(arr, med, iqr, m0=A1_REF_MEDIAN, s0=A1_REF_IQR):
          """A1 normalization: remap CTX DN by robust offset+gain to the (m0, s0) reference,
          `(x - med)/iqr * s0 + m0`, clipped to [0,255] uint8. nodata (DN==0) stays 0.
          ... so within-frame texture is preserved and only the between-frame level/scale
          is removed."""
          ...
          out = np.clip((a - med) / iqr * s0 + m0, 0, 255)     # <-- floor is the nodata value
          out[arr == 0] = 0

  scripts/f_pilot_crop.py:165   # the SAME A1 map, re-implemented, floored at 1
          out[fin] = np.clip((arr[fin] - med) / iqr * A1_S0 + A1_M0, 1, 255).astype(np.uint8)
  scripts/f_region_stageb.py:76-77
          out[r0:r0 + row_block][fin] = np.clip((v - llo) / (lhi - llo) * 254.0 + 1.0, 1, 255)

  src/mapping.py:95              out[i] = float((box == 0).mean())
  src/mapping.py:256-257         zero_frac = own_tile_zero_fraction(arr, ...)
                                 usable = valid & (zero_frac <= max_zero_fraction)
  ```
  Measured over the committed `reports/figures/striping_frame_radiometry.csv` (380 source frames):
  `thr = median − 4.513·IQR` is **≥ 1 DN for 180 of 380 frames** (27.2 % pixel-weighted), with
  `median(thr) = −2.3`, `p95 = +81.3`, `max = +138.8` DN. That table is the **160 m coarsened**
  brightness, whose tail is far thinner than the native 5 m one, so it is a lower bound: at 160 m no
  frame has ≥ 5 % of cells below `thr`, but that says nothing about the 5 m pixels the embedder
  actually sees.
- **Aggravating interactions (not separate findings):** (a) the deploy path derives `(med, iqr)` from
  the **160 m** array and applies it to **native** pixels (that mismatch is R07) — the 160 m IQR is
  smaller, which pushes `thr` *up* and clips more; the training path (`_w2_fang_embed.py:209`) derives
  it natively, so train and deploy clip differently as well as scale differently. (b) `a1_stats`
  (`src/striping.py:236`) substitutes `iqr = 1.0` when the IQR is 0 (`... or 1.0`), which turns the
  remap into a ×27.7 gain and would clip essentially the whole window to `{0, 255}`;
  `striping_a1_map.frame_stats_160` guards `iqr > 0`, but `a1_normalize_window` (the training path)
  does not.
- **Self-refutation attempted:** (i) Is the floor intentional? The docstring does say "[0,255]", but no
  DECISIONS entry mentions it (`grep` for `a1_apply` / `A1_REF` / "clip" around the A1 entries finds
  none), the project's other four stretches all use `1, 255` with an explicit nodata rationale, and the
  same-formula F-pilot re-implementation uses `1, 255` — so it reads as an oversight, not a ruling.
  (ii) Is it unreachable? No: 180/380 frames have a positive threshold. (iii) Does a test pin it?
  `tests/test_striping.py:12-22` uses a synthetic `N(90, 12)` "dark frame" whose threshold sits at
  −6.1 σ, so no sample ever clips; `:26-30` uses a constant array; `:33-40`'s monotonicity assertion is
  `np.diff(...) >= 0`, which a flat clipped region satisfies. Nothing exercises the floor.
  (iv) Does R07/R08 already cover it? R07 is the 160 m-vs-native *statistic*; R08 is *un-labelled*
  pixels left at raw DN. Neither is the clip-floor/nodata collision, and neither implies the
  differential-footprint η² comparison.
- **Fix:** clip to `(1, 255)` in `src/striping.py:251` and in the two script copies (and have the
  scripts call `a1_apply` rather than re-inline it); make `a1_stats` return `nan` rather than `1.0` on a
  zero IQR; record the clipped-pixel fraction per frame; and re-score the A1 payoff η² on a common
  finite mask across the two arms (as `f_map_compare` already does).

### numerics-2 — The Stage-7d "per-image standardised" Spearman standardises the *feature* but not the *target*, understating the reported ρ by ~1.4–1.6×
- **Severity:** medium (record correctness; bias direction is conservative)
- **Liveness:** dead-closed (compositional programme), but the numbers are quoted in `DECISIONS.md` and `docs/compositional.md`
- **Confidence:** high (recomputed from the committed inputs)
- **Where:** `src/stage7d_pooled.py:344` and `:349-350` (vs the symmetric partial-dust branch at
  `:359-361`); claims at `DECISIONS.md:1992-1995` and `docs/compositional.md:779-784`

`run_spearman_tests` z-scores every colour feature within each image (`per_image_standardise`) and then
correlates the z-scored feature against the **raw pooled** `boulder_count`. The sibling
"partial dust" test residualises *both* sides per image, so the two rows in the same table are not the
same kind of statistic. Because the feature's per-image mean is forced to 0 while the target keeps its
large between-image level differences, the between-image variance enters only the denominator: the
result is neither the within-image correlation the docs describe nor the pooled raw one, and it is
attenuated toward zero.

- **Failure scenario:** `DECISIONS.md:1993` reports "all 6 standardised Spearman rhos (−0.123 to
  −0.172)" as Condition 3 of the Stage-7d PASS, and `docs/compositional.md:781-784` describes the
  standardised test as asking "how anomalous is this tile relative to its image" — which is true of the
  feature and false of the target. Recomputing on `dataset_v2/features_colour.parquet` +
  `dataset_v2/labels` with the shipped code reproduces −0.128 … −0.178; standardising the target the
  same way gives **−0.170 … −0.269**. Every published effect size for Condition 3 is ~35–45 % too small,
  and any future comparison against a properly-standardised number will look like a discrepancy.
- **Evidence:**
  ```
  src/stage7d_pooled.py:344   sub = per_image_standardise(df, feature_cols + [dust_col])
  src/stage7d_pooled.py:349       rho, p, n = spearman_with_p(sub[f"{feat}_z"].to_numpy(),
  src/stage7d_pooled.py:350                                   sub[target_col].to_numpy())   # RAW target
  src/stage7d_pooled.py:359       r_feat   = residualise_per_image(sub, y_col=feat,       x_col=dust_col)
  src/stage7d_pooled.py:360       r_target = residualise_per_image(sub, y_col=target_col, x_col=dust_col)  # both sides
  ```
  Recomputed (all 39 images, S=64, n = 9,860 paired tiles):

  | feature | ρ(feat_z, count_raw) — as shipped | ρ(feat_z, count_z) |
  |---|---|---|
  | IR_iof | −0.1779 | **−0.2687** |
  | RED_iof | −0.1711 | **−0.2618** |
  | BG_iof | −0.1436 | **−0.2346** |
  | IR_over_RED | −0.1730 | **−0.2400** |
  | IR_over_BG | −0.1442 | **−0.1926** |
  | dust_index_RED_over_BG | −0.1276 | **−0.1700** |
- **Self-refutation attempted:** the bias is toward the null, so the *sign-match* verdict that Condition 3
  actually rests on is unaffected — that is why this is medium and not high. I checked whether the
  asymmetry is a deliberate ruling: `DECISIONS.md:2020` records only "Spearman included for the §4.3
  continuous-monotonicity check", with no statement about which side is standardised, and the sibling
  partial-dust path in the same function standardises both. I also checked whether the docs quote the
  raw per-image Spearman instead — they do not; `docs/compositional.md:783` explicitly cites the
  *standardised* one.
- **Fix:** standardise `target_col` per image alongside the features in `run_spearman_tests` (or rename
  the `test_type` to `spearman_count_feature_standardised_only`), and restate the Condition-3 numbers in
  `DECISIONS.md:1993` and `docs/compositional.md:783`.

### numerics-3 — A two-stage hurdle that cannot fit its magnitude head silently predicts **all zeros**, and the docstring promises a different fallback
- **Severity:** medium
- **Liveness:** live code path (`LightGBMTwoStageBalanced` is the v2 GBM recipe); latent in every banked LOIO artifact
- **Confidence:** high on the code; **low** on whether it ever fired in the small-fold within-image sweeps (their boosters are not persisted)
- **Where:** `src/modeling/gbm.py:292-296` + `:332-334` (`LightGBMTwoStage`), and the identical branch in
  `_TwoStageBase` at `:548-550` + `:644-645`; downstream `src/modeling/evaluate.py:65` (`spearman_safe`)
  and `:390-394` (`mean_std` drops NaN folds)

When a training fold has fewer than 10 positive tiles the magnitude head is skipped and `predict`
returns `p_pos * np.zeros_like(p_pos)` — an identically-zero vector. The docstring says the opposite:
"predict falls back to mean-positive constant". An all-zero prediction is *constant*, so
`spearman_safe` returns NaN, and `mean_std` then **drops that fold from the mean** instead of scoring
it — the fold vanishes rather than being penalised. That is R24's optimistic-drop mechanism with a
second, previously unnamed producer.

- **Failure scenario:** a within-image quadrant fold on a boulder-poor image at a coarse scale leaves
  < 10 tiles with `fractional_area > 0` in the training set. `fit` returns after the presence head,
  `predict` emits zeros for every test tile, `per_fold_metrics` records `spearman_rho = NaN`,
  `meaningful_auc = NaN`, and `aggregate_fold_metrics` reports a mean over the *remaining* folds while
  advertising `n_real_folds` unchanged. The run looks healthy; the model produced nothing.
- **Evidence:**
  ```
  src/modeling/gbm.py:292-296
          if y_pos.size < 10:
              # Pathological: too few positives to fit a magnitude model. Skip; predict
              # falls back to mean-positive constant.          <-- it does not
              self._magnitude = None
              return
  src/modeling/gbm.py:332-334
          else:
              mag = np.zeros_like(p_pos)
          return p_pos * mag
  ```
- **Self-refutation attempted:** I tried to kill this by looking for evidence it ever fired. `save()`
  omits `magnitude.txt` when `_magnitude is None`, so the banked artifacts are a direct test: **0 of 188
  two-stage fold directories under `models/**/fold_*` lack `magnitude.txt`**, so no banked LOIO number
  is affected. It survives because (a) the code path is live and default-reachable, (b) the docstring is
  actively wrong about what happens, and (c) the within-image sweeps
  (`models/_sweep_within_image/*`) persist only aggregates, so the S=128 quadrant folds — the exact
  regime R24 documents as producing 15/20 constant predictions — cannot be cleared by this check.
- **Fix:** implement the documented fallback (`mag = mean(y_pos)` as a constant, or refuse to fit and
  raise), and have `evaluate` count constant-prediction folds separately (`n_degenerate_pred`) instead
  of sharing the NaN channel with genuinely-undefined folds (same fix as R24).

### numerics-4 — A missing Stage-4b features parquet is silently swallowed, producing an all-NaN feature block for that image instead of an error
- **Severity:** low (latent: 38/38 v2 and 9/9 v1 features parquets exist today)
- **Liveness:** live-shipped packaging path
- **Confidence:** high
- **Where:** `src/dataset.py:537-541`, consumed at `:660-671` and `:663` (`_split_columns(train_df)`)

`_join_one_image` takes `features_dir`, checks `feat_path.exists()`, and **just skips the merge** if it
does not — returning a labels-only frame. `package_split` then `pd.concat`s the per-image frames, which
takes the *union* of columns and fills the missing image's feature cells with NaN. LightGBM treats NaN
as "missing" and trains happily; the MLP `FeatureScaler` median-imputes them. Nothing raises, and the
only count `run_stage5.py` prints is `X_cols` from fold 0, which is unchanged because the other 37
images still supply the columns.

- **Failure scenario:** Stage 4b crashes on one ObsId (or its parquet is deleted / not copied to a new
  machine). Stage 5 re-packages, the log prints the same `X_cols=…`, and every subsequent LOIO run
  trains on that image as an all-missing row block and — on the fold where it is held out — scores a
  test set with no features at all. The per-image metric for that image is then a pure prior, silently
  dragging the cohort median. Pairs with **R04** (Stage-5 failures swallowed, stale packages
  undetectable).
- **Evidence:**
  ```
  src/dataset.py:537-541
      if features_dir is not None:
          feat_path = Path(features_dir) / f"{obs_id}.parquet"
          if feat_path.exists():
              features = pd.read_parquet(feat_path)
              labels = labels.merge(features, on=TILE_KEY_COLUMNS, suffixes=("", "_feat"))
      # (no else: a missing features parquet is indistinguishable from features_dir=None)
  ```
- **Self-refutation attempted:** I checked whether a caller guards it — `scripts/run_stage5.py:46-47`
  passes `features_dir` unconditionally and never verifies coverage, and `package_split` records no
  per-image feature-column count in `metadata.json` (`per_fold` carries only `n_train_x_cols` from the
  concatenated frame). I also checked whether it is currently firing: `dataset/labels` 9 / `dataset/features` 9
  and `dataset_v2/labels` 38 / `dataset_v2/features` 38, so no shipped number is affected — hence low.
- **Fix:** in `_join_one_image`, raise when `features_dir is not None` and the parquet is missing (the
  "no features at all" mode is already expressible as `features_dir=None`), or at minimum record
  per-image feature presence in the packaging `metadata.json`.

### numerics-5 — `precision@5%` is read out of a DataFrame row by **positional** `itertuples` field, so a column insertion silently swaps in a different metric
- **Severity:** low
- **Liveness:** dead-closed (§5.1 F-vs-mosaic comparison table)
- **Confidence:** high
- **Where:** `scripts/f_map_compare.py:186-190`

`precision@5%` is not a valid Python identifier, so `df.itertuples()` renames it to the positional
placeholder `_3`. The code reads it as `getattr(r, "_3", np.nan)`. `_3` is bound to *whatever column
sits in field position 3*, and the `getattr` default means a rename/reorder degrades to `NaN` or, worse,
to the neighbouring column's value with no error.

- **Failure scenario:** add or reorder one column in `reports/figures/f_h4_legb_summary.csv` before
  `precision@5%` and the §5.1 skill column silently reports `median_img_auc` (0.786) as
  `precision@5%` (0.9127) — both plausible numbers in the same range, printed into the comparison table
  the abort record cites.
- **Evidence:**
  ```
  scripts/f_map_compare.py:188-190
                  rows.append({"row": m[r.pipeline], "pooled_pr_auc": r.pooled_pr_auc,
                               "precision@5%": getattr(r, "_3", np.nan),
                               "n_img": r.n_img, "source": "f_h4_legb_summary.csv (36-img LOIO)"})
  ```
  Checked against the committed CSV: `fields = ('Index', 'pipeline', 'pooled_pr_auc', '_3',
  'median_img_auc', 'n_img')`, so `_3 → 0.9127` — **currently correct**. This is a latent
  silent-wrongness, not a live wrong number.
- **Self-refutation attempted:** I verified the current binding resolves to the intended column (above),
  and grepped the whole repo for other positional `itertuples` reads — this is the only one.
- **Fix:** `r._asdict()["precision@5%"]`, or index the DataFrame by column name instead of iterating
  tuples.

## Refuted by my own check

- **`lofo_offsets` averages neighbour offsets across different gauge components** (`src/leveling.py:484-487`
  has no cross-component guard, unlike `heldout_edge_cv:456`). Measured on the banked
  `reports/f_stagec/stagec_edges_min50.npz` (906 frames, 6,073 edges, **one** component): for **0 of 906**
  frames does removing that frame's edges split its neighbours across components. The residual gauge
  difference (median over 905 vs 906 frames) is one order statistic. Guard-3's LOFO numbers are sound.
- **`TileAccum.add_frame` uses buffered `+=` for `sum_logit`/`n_frames` but unbuffered `np.minimum.at` /
  `np.maximum.at` for `p_min`/`p_max`** (`src/fcompose.py:158-161`), which would undercount duplicate
  `(row, col)` pairs. Unreachable: `f_region_stageb.process_frame:174-180` deduplicates every frame's
  `(TI, TJ)` with `np.unique` + `np.add.reduceat` before writing the npz, and the global→tile map is an
  exact integer shift, so no frame can present a duplicate pixel.
- **`_GLCM_NAN_FILL = 0.0` is an in-range sentinel for `glcm_correlation_*`** (0 is a legal correlation),
  the same shape as R27. Measured over all 38 `dataset_v2/features/*.parquet`: **0 of 594,960** finite
  S ≥ 32 `glcm_correlation_d{1,2,3}` values are exactly 0.0 (689 are negative, none in the open
  interval that a sentinel would create). Never fires.
- **`compression_metrics["low_over"]` divides by `max(mean(y_true[y_true<=0]), 1e-9)`, i.e. by 1e-9**
  (`src/calibration.py:272`). Real, but already recorded in `calibration.md`'s coverage notes and
  explicitly dropped by `src/fgates.py:274,289`. Not re-filed.
- **`fa` / `boulder_area` are latitude-distorted by the plate-carrée CTX CRS.** `fa` is a ratio of two
  projected quantities, so the distortion cancels; the `min_size_m` and `boulder_area` consequences are
  already filed and quantified in `labeling.md`.
- **`banding_indices` (`src/striping.py:203-212`) has a shape-dependent null** (variance of column means
  ≈ tot/H under white noise), so a non-square field would make "vertical vs horizontal" incomparable.
  The map tiles are ~1481 × 1481 coarse cells (4° ≈ 237 km at 160 m), so the two nulls coincide, and the
  quoted V ≈ H ≈ 0.005–0.006 (`DECISIONS.md:4018-4019`) is ~8× the white-noise null and was in any case
  superseded by the km-scale reading at `:4052-4053`.
- **Stage-D writes `"ti_min": 1, "tj_min": 1` hard-coded into its per-tile sidecar**
  (`scripts/f_region_staged.py:266`) where the mosaic path writes the real values
  (`scripts/map_region.py:241`). No consumer reads the sidecar's `ti_min` — every user re-derives it
  from the raster or from `ti.min()`.
- **The abort table's headline statistic is mis-computed.** Recomputed from
  `reports/figures/fbuild_abort_level_vs_labels.csv`: population `sd(log10 ratio)` = mosaic **0.1702**,
  h1only 0.3282, resid **0.3710**, pfree **0.5318**, full 0.4119 over 19 (two `full_ratio == 0` rows are
  dropped by the `log10`). These match the quoted 0.170 / 0.371 / 0.532 exactly (`ddof=0`). The
  arithmetic is right; the interpretive problems are R10/R12/R33.
- **`plane_complement` would blow up on NaN `lon`/`lat`** (`src/leveling.py:257-258`). Unreachable:
  `f_region_stageb.py:226-228` skips any frame with no incidence row, so every frame that produces
  logits has finite `center_lon`/`center_lat`.
- **`f_region_stagec.py:484-486` computes `z_rad` with `nanmean`/`nanstd` and then `np.abs(z_rad) < 2.0`,
  so a frame with no `frame_median` is silently excluded from the guard-3 count** rather than flagged.
  Real but unreachable — Stage B writes `frame_median` for every frame it emits.
- **`k = max(1, int(0.05·n))` (floor) in `scripts/probes/_w2_fang_probe.py:164` vs
  `int(round(0.05·n))` in `src/fgates.py:263`**, whose docstring claims the conventions are copied
  verbatim. Real inconsistency, but it moves `k` by at most 1 tile out of thousands — immaterial.
- **`stripe_enhance` (`src/striping.py:280`) smooths `nan_to_num(det)` without renormalising by the
  validity mask** (unlike `detrend:199`), so coverage edges are diluted toward 0 over ~20 px. Figure-only
  quantity; no reported number depends on it.
- **`src/modeling/cnn.py:192,199,202` chain three `astype(np.int16)` truncations in the augmentation
  pipeline** (a systematic ≈−0.5 DN each). Dead-closed CNN work, and it is dominated by the ±15 %-of-DN
  brightness-jitter magnitude error already filed in R35.
- **`np.clip(..., 1, 255).astype(np.uint8)` truncates rather than rounds** in the F stretch
  (`f_region_stageb.py:76`, `f_pilot_crop.py:120/130/165`). Identical formula on both the training and
  the deploy side, so parity holds; the ≈0.5 DN floor bias is common-mode.

## Verified clean

- `src/leveling.py`: `pack_key`/`unpack_key` round-trip (negative `TJ` handled by the
  `((k + S/2) % S) − S/2` form), the `|TJ| ≥ 2^19` aliasing guard, `intersect_sorted`'s swap branch and
  its `clip`-then-`==` hit test, `regauge`'s defensive `.copy()` (no caller-array mutation),
  `normal_equations`' `np.add.at` assembly, and `plane_complement`'s rank-aware SVD.
- `scripts/f_region_stageb.py:174-180`: the per-frame `(TI, TJ)` dedup is correct
  (`np.add.reduceat` + `np.diff(np.append(first, len))` gives exact per-key means).
- `src/spatial_features.py`: the neighbour aggregation is genuinely NaN-aware — `count_win` is the
  window sum of a validity mask, `safe_count` only guards the division and the result is re-masked to
  NaN, `max` uses `−inf` fill and is rescued, and `std` is NaN below 2 valid neighbours.
- `src/modeling/evaluate.py`: `presence_auc`'s Mann-Whitney identity (`U/(n_pos·n_neg)`) is the
  tie-correct AUC; `rmse`, `rmse_log1p`, `per_bin_rmse`, `brier_score`, `expected_calibration_error` and
  `calibration_deciles` all guard empty inputs and empty bins.
- `src/coregister.py`: `select_fft_window`'s integral-image block sums, the Hann window + mean
  subtraction before `phase_cross_correlation`, the margin crop before the Pearson confidence, and the
  block-median MAD.
- `src/features.py`: `_intensity_stats_per_tile`'s `np.where(var > 0, …, 0.0)` under `errstate`,
  `_gradient_stats_per_tile`'s `+1e-12` weight denominator, `_canny_per_tile`'s zero-edge branch and
  `0·log 0` handling, and `_stack_tiles`' in-window assertion at `stage4b_one_image:656-663`.
- `src/labeling.py`: `_compute_grid_alignment` makes `j_min_row/col` an exact multiple of
  `S_max/S_min`, so `_sum_up_ladder`'s `j_min_row // (S // S_min)` is exact at every rung, and the
  ladder asserts even shapes before each ×2 reduction.
- `src/calibration.py`: `CalibrationLayer.save/load` round-trips knots without pickle;
  `IsotonicCalibrator.knots()` really does reproduce `predict` under `np.interp` (`out_of_bounds="clip"`).
- `src/reliability.py`: both novelty scorers mask all-NaN rows in and out, and floor the whitening
  variance.
- `reports/figures/fbuild_abort_level_vs_labels.csv` reproduces the published abort spreads exactly (see
  Refuted section).

## Coverage note

**Read in full:** `src/striping.py`, `src/calibration.py`, `src/mapping.py`, `src/spatial_features.py`,
`src/fcompose.py`, `src/fgates.py`, `src/reliability.py`, `src/stage7d_pooled.py`, `src/colour.py`,
`src/modeling/loaders.py` (fold loading), `scripts/f_region_stageb.py`, `scripts/striping_a1_map.py`,
`scripts/striping_a1_infer_crop.py`, `scripts/run_stage5.py`.
**Read in substantial part:** `src/features.py`, `src/labeling.py`, `src/leveling.py`, `src/dataset.py`,
`src/coregister.py`, `src/ctx_retrieve.py`, `src/ctx_source_illumination.py`, `src/validation_retrieve.py`,
`src/fm_embeddings.py`, `src/modeling/evaluate.py`, `src/modeling/gbm.py`, `src/modeling/mlp_head.py`,
`scripts/map_region.py`, `scripts/f_region_stagec.py`, `scripts/f_region_staged.py`,
`scripts/f_map_compare.py`, `scripts/f_pilot_crop.py`.
**Grepped only (hits triaged, not fully read):** every `except:` / `except Exception` (15 sites — all
either import-guards, I/O fallbacks that print, or the documented Cholesky→lstsq fallback at
`src/leveling.py:313`), every `nan_to_num` / `fillna` / `dropna` / `errstate` / `astype(uint8|int)` /
`np.log*` / `isclose` / `inplace=True` / mutable-default site in `src/` and top-level `scripts/`.
**Executed (read-only, on committed artifacts):** the abort-table recomputation, the Stage-7d Spearman
recomputation, the LOFO component check on the banked EdgeSet, the GLCM-sentinel census over
`dataset_v2/features/`, the A1 zero-clip threshold census over `striping_frame_radiometry.csv`, and the
`magnitude.txt` census over `models/**/fold_*`.
**Could not check:** (1) the *native-resolution* pixel fraction that numerics-1's clip destroys —
that needs the cached CTX zips, which the rules of engagement put out of bounds; the 160 m table gives
only a lower bound. (2) Whether numerics-3 ever fired in the within-image sweeps — those runs persist
aggregates only. (3) `scripts/probes/` (229 files) beyond `_w2_fang_embed.py` and
`_w2_fang_probe.py::verdict`; several DECISIONS numbers originate there and remain unswept for
numerics. (4) `src/modeling/cnn.py` and `src/modeling/binary_target.py` were only skimmed
(dead/covered by `modeling-heads`). (5) Notebook-resident arithmetic (`notebooks/_build_*.py`) is the
`notebooks` reviewer's area and I did not audit it.
