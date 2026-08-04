# Review area: leakage

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (single-agent pass; not independently verified)

## Findings

### leakage-1 — Stage-6a neighbour features are computed across the within-image quadrant cut, so the *treatment* arm's test tiles carry training-fold feature values — and only that arm does
- **Severity:** medium
- **Liveness:** dead-closed (the surviving Stage-6a decision was re-made at LOIO) **but the code path is still the default**
- **Confidence:** high (mechanism), high (magnitude — measured)
- **Where:** `src/spatial_features.py:149-161` + `:199-205`; `scripts/run_stage6a_repackage.py:1-8`;
  `scripts/probes/_sweep_stage6a.py:54-55`, `:224`; `src/dataset.py:212-217`; `config_v2.yaml:170`
  (`buffer_tiles: 0`); results `models/_sweep_stage6a/20260531T004356Z/result.md`

`add_neighbour_features` aggregates a `stencil_size × stencil_size` window over the **whole image's**
`(ti, tj)` grid, per `(obs_id, scale_idx)` — before any split exists. `run_stage6a_repackage.py` then
reuses the *existing* `within_image_4fold` split "verbatim -- same folds, same train/test ObsIds -- so
the only difference between the two packaged dirs is the X-matrix columns". Under the within-image
scheme the train/test boundary is a cut *inside* one image, and with `buffer_tiles: 0` nothing is
dropped at the cut (`src/dataset.py:215-216` returns an all-true keep mask). A test-quadrant tile
sitting on the cut therefore has `nbr_mean_*` / `nbr_max_*` / `nbr_std_*` values that are literal
arithmetic functions of **training-quadrant** rows. Crucially this is asymmetric: the baseline arm has
no `nbr_` columns at all (verified: `dataset_v2/packaged/within_image_4fold` has 60 columns, 0 of them
`nbr_`), so the *arm-vs-arm Δ* — the quantity Stage 6a's acceptance test is defined on — is inflated
in one direction only.

- **Failure scenario:** run `scripts/probes/_sweep_stage6a.py` with its defaults
  (`within_image_4fold` vs `within_image_4fold_nbr`). A Δ that is partly the model reading the
  training fold's own feature values is scored against the `Spearman Δ ≥ +0.05 AND PR-AUC Δ ≥ +0.03`
  bar. The banked S=32 run reports `meaningful AUC +0.6720 → +0.7436 (Δ +0.0716)`; a nonzero share of
  that Δ is the boundary band, not spatial context. Both banked within-image runs happened to FAIL, so
  no promotion rests on it today — but the same script rerun on a cohort where the Δ lands near the
  bar would PASS on the leak.
- **Evidence:**
  ```
  src/spatial_features.py:158-161
      Aggregation is per ``(obs_id, scale_idx)`` group: the ``(ti, tj)`` grid of each
      image-and-scale stands alone

  src/dataset.py:212-216
          if buffer_tiles > 0:
              in_buf = (np.abs(ti_sub - ti_mid) < buffer_tiles) | (np.abs(tj_sub - tj_mid) < buffer_tiles)
              keep_sub = ~in_buf
          else:
              keep_sub = np.ones(int(sel.sum()), dtype=bool)

  scripts/probes/_sweep_stage6a.py:54-55
  BASELINE_SCHEME = "within_image_4fold"
  NBR_SCHEME = "within_image_4fold_nbr"
  ```
  Measured share of test tiles whose stencil reaches across the cut, computed from
  `dataset_v2/splits/within_image_4fold.json` + `dataset_v2/labels/*.parquet` by re-deriving the
  code's own quadrant predicate (`2*(ti>=ti_mid) + (tj>=tj_mid)`):

  | scale | 3×3 stencil (`features_nbr`) | 5×5 stencil (`features_nbr_s5`) |
  |---|---|---|
  | S=8  | 0.79 % | 2.4 % |
  | S=16 | 1.6 %  | 4.7 % |
  | S=32 | 3.2 %  | 9.4 % |
  | S=64 | **6.5 %** | **18.8 %** |

- **Self-refutation attempted:** (a) the register already refutes *"`buffer_tiles: 0` invalidates
  within-image CV"* — but that claim was about label spatial autocorrelation as a general CV-validity
  argument. This is a different, mechanical claim: the test row's *feature vector* is a function of
  training rows, in one arm only. I am reporting the mechanism and its measured size, not re-litigating
  the general adjacency argument. (b) Does the final Stage-6a decision depend on it? No — `DECISIONS.md:2506`
  re-ran Stage 6a at full-v2 **LOIO** (`dataset_v2/packaged/loio_nfold_nbr_s5`), which is clean (neighbour
  reach ≤ 2 tiles = 640 m at S=64, and the nearest two image footprints in the cohort are 1,760 m apart
  — measured, see Verified clean). That is why this is medium and not high. (c) Is
  `within_image_4fold_nbr` even reachable? `dataset_v2/packaged/` has no `within_image_4fold_nbr`
  today, but `run_stage6a_repackage.py` regenerates it in one command and `_sweep_stage6a.py` defaults
  to it. (d) Does any script or doc warn about it? Grepped `buffer|leak|adjacen` across
  `scripts/probes/_sweep_stage6a.py` and `scripts/run_stage6a_repackage.py` — the only hit is
  `buffer_tiles` inside a hash-key tuple. Nothing warns.
- **Fix:** when repackaging an augmented split whose `kind == "within-image"`, either (i) set
  `buffer_tiles >= ceil(stencil_size/2)` in the cloned split JSON so the boundary band is dropped from
  both arms, or (ii) recompute the neighbour aggregation per fold using only that fold's training rows.
  Minimum viable: make `run_stage6a_repackage.py` refuse a `within-image` split whose `buffer_tiles` is
  smaller than the stencil reach.

---

### leakage-2 — The one cohort image whose CTX is featureless is excluded from labels *and* features, so it is absent from every reported per-image metric, on a criterion that includes the CTX↔abundance relation being scored
- **Severity:** medium
- **Liveness:** live-shipped (conditions the frozen recipe's headline per-image distribution and the map's skill claim)
- **Confidence:** medium-high on the mechanism; low on magnitude (bounded below, see below)
- **Where:** `src/features.py:86-91`; `scripts/run_stage4.py:31-38`, `:105-109`;
  `scripts/run_stage4b.py:86-90`; `scripts/run_stage6a.py:146-150`; `scripts/run_stage6b.py:206-210`;
  `config_v2.yaml:162`; rationale `DECISIONS.md:1258-1267`

`ESP_046803_2325` is in `EXCLUDED_FROM_SWEEP`, which gates Stage 4 (labels), Stage 4b (features) and
Stages 6a/6b. It therefore has no `dataset_v2/labels/*.parquet` and no
`dataset_v2/features/*.parquet`, so `discover_obs_ids` never sees it, no fold contains it, and it
appears in **no** reported metric — not the frozen recipe's median per-image AUC 0.7865, not
`frac_ge_0p7`, not the `dauc win 0.96`, not the A1/F per-image AUC medians. The stated rationale is
two-part: 0/210 co-registration blocks correlate (a legitimate *input-quality* criterion) **and** "it
is a high-target / no-input training example (featureless CTX paired with high abundance) that would
add label noise without teaching the CTX→abundance mapping". The second clause is a statement about
the joint (CTX texture, abundance) relationship — i.e. exactly the quantity every per-image metric
measures. Excluding on it is defensible for the *training* set; extending it to the *evaluation* set
makes the reported skill distribution conditional on the relation being estimated.

- **Failure scenario:** the shipped map is painted over ~26 Murray tiles of circum-Chryse that
  certainly include dust-mantled, low-texture CTX. The one cohort image of exactly that kind — with
  ~367k detected boulders, i.e. genuinely rock-rich under a featureless CTX surface — is the guaranteed
  near-0.5-AUC case, and it is not in the denominator of any quoted per-image statistic. A reader of
  `median per-image AUC 0.7865` / `frac AUC ≥ 0.7` cannot tell that the population was filtered on the
  input↔label relation.
- **Evidence:**
  ```
  src/features.py:86-91
  # Stage 4b deliberately recognises the same "drop this ObsId from --all sweeps" set that
  # Stage 4 already uses. ...
  #   ESP_046803_2325 — v2 vClaire: featureless CTX, 0/210 co-registration blocks correlate
  EXCLUDED_FROM_SWEEP = {"ESP_057469_2215", "ESP_046803_2325"}

  DECISIONS.md:1261-1263
  Despite ~367k detected boulders, it is a high-target / no-input training example (featureless CTX
  paired with high abundance) that would add label noise without teaching the CTX→abundance mapping.
  Brian's call: **drop**.
  ```
- **Self-refutation attempted:** (a) Is the exclusion purely input-quality? Partly — "0/210 blocks
  correlate" means Stage 3 has no registration lock, so the label *positions* are unverified, which
  alone justifies dropping it from training. That is why I rate this medium, not high. But the
  decisive sentence in the record is the outcome-flavoured one, and the drop was applied to evaluation
  too, which the input-quality argument does not require (an unregistered image could have been kept
  as a flagged specificity case, as `ESP_065711_1545` was). (b) Is the magnitude large? No: 39 → 38
  images, so medians move by at most one order statistic (≈0) and rate statistics such as `win 0.96`
  or `frac_ge_0p7` by ≈1/39 = 2.6 pp. I am reporting it because the *direction* is unambiguous and the
  record carries no caveat, not because it moves a headline number materially. (c) Is the second
  exclusion (`ESP_057469_2215`, 0.1 % swath coverage) the same problem? No — that is pure geometry,
  independent of the label relation. (d) Is it flagged anywhere? Grepped `DECISIONS.md`,
  `docs/modeling_results.md` and `ROADMAP.md`: the drop is recorded once, at the Stage-3 QA step, and
  never restated as a caveat on any per-image distribution.
- **Fix:** documentation, not code. Add a standing caveat wherever a per-image metric distribution is
  quoted: "38 of 39 manifest images; `ESP_046803_2325` (featureless CTX / high abundance, no
  co-registration lock) is excluded from labels and features and therefore from this distribution."
  If cheap, re-admit it as an evaluation-only, flagged image (mirroring the `is_specificity_only`
  treatment of `ESP_065711_1545`) so the reported distribution covers the low-texture regime.

---

### leakage-3 — Stage C picks λ as the argmin of the held-out-edge CV, and gate 2 then reports and tests that same statistic at that λ; the code also selects on the metric its own docstring says must not select λ
- **Severity:** low-medium
- **Liveness:** dead-closed (F build); direction is toward PASS, so the ABORT verdict is safe
- **Confidence:** high
- **Where:** `scripts/f_region_stagec.py:188-196` (docstring), `:234-240` (the argmin), `:390`,
  `:410`, `:527`; `src/fgates.py:146-148`, `:181-183`; artifacts
  `reports/figures/fbuild_stagec_lambda.csv`, `reports/figures/fbuild_gate2_edgecv.csv`

`lambda_sweep` scores every λ on `lv.heldout_edge_cv` and then selects
`best = df.sort_values([col, "lambda"]).iloc[0]` — the λ that **minimises** the held-out-edge CV.
Gate 2 (`fgates.edge_cv_for_offsets`) recomputes the same CV, with the same seed and fold fraction, at
that λ and reports it as `heldout_cv_dp` with `passes = cv < base`. The reported "held-out"
disagreement is therefore the minimum of the statistic over the λ grid, not an unbiased estimate of
it, and the pass/fail test is applied to the quantity that was optimised. The banked grid shows the
argmin at the boundary (λ = 0: `heldout_cv_dp` 0.0094, the smallest in the table, with
`railed_tile_frac` 0.5181 and `max_abs_offset` 21.3), which is the classic signature of selecting on a
statistic the search can game.

Second, smaller defect at the same site: the docstring states "λ\* is selected on the held-out
**logit** metric since 2026-07-29 (Brian). The pre-registered probability-space |Δp| … **cannot select
λ**", yet `:390` sets `lam_star = pick_dp["lam"]` and `:410` builds the shipped offset vector from
`solved[pick_dp["frac"]]` — i.e. selection on |Δp|, the metric the docstring forbids. On the banked
run both metrics picked λ = 0, so no number moved, but the code does the thing its documentation says
it must not.

- **Failure scenario:** a rerun on a graph where the two CV metrics disagree ships offsets chosen by
  |Δp| (which the project established is minimised by railing the sigmoid), while the docstring,
  `fbuild_stagec_offsets.csv` and the DECISIONS record all assert the logit-based choice. Independently,
  gate 2's `heldout_cv_dp` is quoted as a held-out generalisation figure in the abort record when it is
  a grid minimum.
- **Evidence:**
  ```
  scripts/f_region_stagec.py:236-240
      for key, col in (("lcv", "heldout_cv_dlogit"), ("dp", "heldout_cv_dp")):
          best = df.sort_values([col, "lambda"], ascending=[True, False]).iloc[0]
          pick[key] = {"lam": float(best["lambda"]), ...
                       "cv_dp": float(best["heldout_cv_dp"]),

  scripts/f_region_stagec.py:390,410
      lam_star, lam_lcv = pick_dp["lam"], pick_lcv["lam"]
      o_star, off_src = lv.patch_graph_holes(solved[pick_dp["frac"]], comp, deg, lon, lat)

  src/fgates.py:146-147,182-183
      base = float(np.median(metric_fn(edges, np.zeros(n))))
      ...
      return {**out, "heldout_cv_dp": cv, ..., "passes": bool(np.isfinite(cv) and cv < base)}

  reports/figures/fbuild_stagec_lambda.csv  (argmin at the grid boundary)
      0.0,0.0,0.0112,0.0094,1.0859,1.1198,0.5181,0,21.311,6.4431
      1.0,29122.0,0.1572,0.1609,1.0974,1.1431,0.0152,0,3.813,0.7561
  ```
- **Self-refutation attempted:** (a) The pathology of selecting on |Δp| **is** flagged — the docstring
  explains it and `:396-406` prints a runtime warning that the CV is "edge-LOCAL and blind to global
  drift". So I do not claim it is unrecognised; what is unrecognised is that the *gate* statistic is
  the argmin of the search, and that `lam_star` is still the dp pick. (b) Does it overturn the abort?
  No — the bias is toward PASS and the shipped `pfree` row already reads `passes=False`
  (`fbuild_gate2_edgecv.csv`). (c) Is this R11 or R19? No: R11 is guards 3/4 + the tautological trend
  guard; R19 is `edge_cv_for_offsets` mislabelling a fallback as `resid`/`pfree`. Neither touches λ
  selection or the `pick_dp`-vs-docstring contradiction. (d) Is `heldout_edge_cv` itself an honest
  instrument? The project already concluded it is "near in-sample on the over-determined graph"
  (`DECISIONS.md:4651`) and built LOFO as the honest alternative — so this compounds an acknowledged
  weakness rather than introducing a new one.
- **Fix:** report gate 2 at a λ fixed *before* the CV is looked at (or nest the λ search inside each
  CV fold), and record `heldout_cv_dp` at the selected λ as "selected-minimum, optimistic". Change
  `:390`/`:410` to `pick_lcv` to match the docstring, or correct the docstring.

---

### leakage-4 — The only GBM `eval_set` call site in the repo that is not the rotated inner-validation image early-stops on the held-out test fold, and prints per-fold Spearman/AUC beside it
- **Severity:** low
- **Liveness:** dead (probe, no banked artifact)
- **Confidence:** high
- **Where:** `scripts/probes/_smoke_gbm_one_fold.py:25`, `:27-29`

I audited every `eval_set=` call site (grep across the repo): `src/modeling/evaluate.py:635`,
`scripts/sweep.py:113`, `scripts/train_gbm.py:123-126`, `scripts/sweep_binary.py:121`,
`scripts/train_binary.py:131`, and the `_diag_tier2_*` probes all pass a mask over `X_train`. The one
exception feeds `f.X_test` as the early-stopping monitor and then prints `spearman` /
`rmse_log1p` / `auc` from `per_fold_metrics` on that same test fold. It is the pattern the harness
exists to prevent, sitting in a file a future session is likely to copy from.

- **Failure scenario:** anyone reusing this file as the one-fold template inherits early stopping on
  the evaluation fold; the printed Spearman/AUC are optimistic and carry no marker saying so.
- **Evidence:**
  ```
  scripts/probes/_smoke_gbm_one_fold.py:25-29
      model.fit(f.X_train, y_train, eval_set=(f.X_test, y_test))
      y_pred = model.predict(f.X_test)
      ...
      m = per_fold_metrics(y_test, y_pred, held_out_obs_ids=f.held_out_obs_ids)
      print(f"  spearman={m['spearman_rho']:+.4f}  rmse_log1p=...  auc=...")
  ```
- **Self-refutation attempted:** it is named `_smoke_*`, docstring says "Smoke-test one fold …
  end-to-end", writes nothing to disk, and I found no DECISIONS entry quoting its numbers. That is why
  it is low, not medium. It survives only as a latent copy-paste hazard and because the brief asked
  for an audit of all GBM `eval_set` provenance — this is the audit's one negative result.
- **Fix:** use the same inner-val rotation as `run_loio` (or `eval_set=None`), and/or print a
  `LEAKY — smoke only` banner.

## Refuted by my own check

- **"7,486 CTX tiles are shared by two `obs_id`s, so LOIO leaves a spatially-identical sibling in
  train."** Measured at S=32: 14,972 of 161,005 rows (9.3 %) share a `(ti, tj)` with another image,
  and the same 9.3 % at every scale. **Refuted:** `ti/tj` are anchored at the **parent Murray tile's**
  origin (`src/labeling.py:363-370`; `src/fgates.py:222-227` documents this explicitly), so `(ti, tj)`
  is not a global key. The four colliding pairs are ~12° of latitude apart (e.g. `ESP_045878_2235` vs
  `ESP_049242_2115`) and cannot overlap. Re-keying on the world bbox
  (`TI = round(y_c/159.9992)`, `TJ = round(x_c/159.9992)`) gives **161,005 distinct world tiles for
  161,005 rows — zero duplicates**. No group-key leak.
- **"The shipped Tier-2 quantile-match is fit on out-of-fold LOIO `P(rich)` (`meta.fit =
  pooled_loio_38`) but deployed on the all-data head (`recipe.json: n_train_images = 38`), so the input
  marginal is shifted and the reported LOIO bound (top_ratio 0.8573) does not describe the map."**
  **Refuted:** the LOIO predictions are themselves *out-of-sample* — each fold's head never saw the
  image it scored — so they are drawn from the same regime as deployment. The deployed head differs
  from each fold head by one training image out of 38, so there is no systematic in-sample sharpening
  at deploy time. And `bank_calibration.py`'s LOIO bound uses a 37-image calibrator, which is
  conservative relative to the deployed 38-image one, exactly as its docstring claims.
- **"The freeze gate's per-image dAUC is evaluated on a performance-selected subset (`validity_ok`)."**
  **Refuted:** `scripts/probes/_w1_build_dossier.py:40` defines
  `validity_ok = (n_neg >= 50) & (n_meaningful_positive >= 50)` — a pure class-count criterion over
  the *labels*, which is the standard AUC-definedness filter. Not outcome-dependent.
- **"`rescore_gain` / `best_offset` are an argmax over 25 label shifts scored on the same image."**
  True mechanically, but `scripts/probes/_w1_shift_rescore.py:6-9` explicitly states "Healthy images
  give the null distribution for the max-over-25-offsets inflation, so the anti-signal gains can be
  judged against chance". Acknowledged and controlled.
- **"`cv_ridge_then_logistic` leaks: `s = pred.std()` is the std of *all* out-of-fold predictions
  including the held-out row."** (`scripts/probes/_stage6c_gate_v2.py:131`.) **Refuted for the reported
  statistics:** the map `1/(1+exp(-(cutoff-pred)/(s/2)))` is monotone decreasing in `pred` with the
  same `s` for every row, so the reported `roc_auc` and `spearman_to_pr_auc` are invariant. Only the
  probability values shift.
- **"The `_diag_tier2_*` probes' inner-val is a random 10 % of training *tiles* across images
  (`inner_val(fold, frac=0.1)`), violating the group-aware-split rule."** True, but the sampling is
  entirely inside the training fold, so nothing from the held-out image reaches early stopping. It is a
  weak overfit detector applied identically to every arm, so it does not bias the Stage-2 comparisons
  optimistically (if anything the reverse).
- **"`temp->isotonic` in `_diag_tier1_isotonic.py:41-43` leaks: the isotonic reference set is
  temperature-calibrated predictions whose calibrators saw the held-out image."** Real but
  second-order (one parameter over 161k tiles), and that variant was not chosen — `isotonic` alone was
  (`DECISIONS.md:3843-3848`).
- **"`run_modeling_slim.py`'s `EXCLUDE_OBS` drops two images from the reported evaluation."**
  `BoulderLabel == 'unknown'` is a priori manifest metadata, not an outcome, so the exclusion is a
  scope definition rather than a leak. (It does remove the only far-southern image,
  `ESP_076499_1160` at ≈ −64°, from the reported cohort — a *generalisation-claim* narrowing, which
  belongs to `docs-consistency`, not here.)
- **"`_standardize_matrix_per_group` gets quadrant codes, not image codes, under the within-image
  scheme, so 'per-image' standardisation becomes per-quadrant on train and whole-fold on test."** The
  asymmetry is real in the code, but no caller combines the two: `scripts/sweep_within_image.py` never
  calls `standardize_fold_per_image` / `augment_fold_with_per_image` (grepped). Latent only, and no
  information crosses the boundary either way.

## Verified clean

- **The packaged LOIO folds themselves.** For all 38 folds of `dataset_v2/packaged/loio_nfold` I read
  the parquets and `groups_*.npy` and checked: train/test `obs_id` sets disjoint; train/test group-code
  sets disjoint; every group code decodes via `obs_to_int` onto the correct side; `len(X) == len(groups)`
  on both sides; `set(test obs_id) == metadata test_obs_ids`. **Zero problem folds.**
- **No cross-image spatial reach.** Nearest inter-image footprint gap in the v2 cohort is **1,760 m**
  (`ESP_066634_2210` / `ESP_071093_2210`), computed from the label bboxes; next are 3,840 / 6,240 /
  8,160 m. The 3×3 context box the frozen recipe embeds is 96 px = 480 m and the widest neighbour
  stencil is 5 × 320 m = 1,600 m, so neither the embedding input nor `nbr_*` can reach another image.
- **No label-derived column reaches X.** Intersecting the label-parquet schema with
  `X_train_fold0.parquet` gives exactly `{obs_id, scale_idx, tile_size_px, ti, tj}` — the tile keys.
  `boulder_area`/`boulder_count`/`fractional_area`/`binary_*`/`count_density`/`tile_area` and the world
  bbox are all on the y side only. `loaders._feature_columns` additionally drops `config_hash_feat` and
  `patch_idx_S*`.
- **GBM `eval_set` provenance, all production sites.** `src/modeling/evaluate.py:619-636` (with the
  `assert inner_val_code not in held_codes` guard), `scripts/sweep.py:104-114`,
  `scripts/train_gbm.py:114-127`, `scripts/sweep_binary.py:121`, `scripts/train_binary.py:131` all use
  the rotated training-image inner-val. The `LightGBMTwoStage` / `_TwoStageBase` /
  `LightGBMClassification` `fit` bodies only ever consume the `eval_set` they are handed (their stale
  comments say "test set", but the data is the inner-val).
- **Nested-scale fold coherence.** `_compute_quadrant_definitions` floor-snaps the finest-scale median
  to a multiple of `max(SCALE_TO_FACTOR_FROM_FINEST.values()) = 16` and then divides by each scale's
  factor, so a coarse tile with `ti_k >= ti_mid_k` covers only finest tiles with `ti >= ti_mid_finest`.
  Parent and child always land in the same quadrant — no nested-grid cross-fold split.
- **Scalers are train-only** (independently re-verified): `FeatureScaler.fit` (`mlp_head.py:72-84`)
  uses `nanmedian`/`mean`/`std` of the fit matrix and `apply` for eval/test; `MLPHead._fit_scaler` /
  `_StandardizedHead._fit_scaler` (`_w2_fang_heads.py:70-86`) the same;
  `_standardize_matrix_per_group` is called once per side with that side's own rows
  (`loaders.py:243-247`, `260-266`).
- **`DeployableHead` inner-val** comes from `np.unique(groups)` of the data handed to `fit`, which in
  every LOIO harness (`striping_a1_loio.py:94`, `f_leg_b_loio.py:86`, `run_loio`) is the training rows
  only.
- **Fang embeddings carry no fitted statistics.** `FangEmbedder` loads frozen pretrained weights
  strictly and normalises with fixed constants (`/255`, `(x-0.5)/0.5`); `load_fang_store` /
  `fang_columns_for_keys` / `EmbeddingBank.lookup` all join on `(obs_id, ti, tj)` with
  `validate="one_to_one"` and assert on a miss. No whitening/PCA is fit anywhere over all images.
- **Per-image DN thresholds are inference-compatible.** `features._compute_dn_thresholds` derives
  shadow/strict/bright from the *image's own* covered-pixel histogram (`src/features.py:141-165`), not
  from a cohort-pooled percentile.
- **Calibration honesty.** `loio_calibrate` (`src/calibration.py:292-300`) fits on `~held` and scores
  `held`. `bank_calibration.py:47-67` labels the in-sample numbers "[in-sample sanity] … NOT a
  deployment estimate" and reports the LOIO bound separately. `notebooks/_build_23.py:107-129`: every
  ECE column is LOIO; the two global-fit numbers are used only for the AUC-exactness argument and are
  labelled "deployment case".
- **Variant-selection bias is acknowledged.** The freeze entry (`DECISIONS.md:3402-3410`,
  `:3459-3462`) names the arch sweep's "forking-paths overfit" as the reason for keeping the
  mid-pack incumbent, and carries "post-hoc assembly (the freeze precedes the §3 pre-declared
  confirmation on cohort-expansion images)" as a standing caveat. In-sample gate numbers are labelled
  as such at `DECISIONS.md:4668` and `:5054`.
- **`sweep_select.pick_sweep`** selects by `sweep_meta.json`'s `dataset_dir` + mtime — a provenance
  choice, no metric involved.

## Coverage note

**Read in full:** `src/dataset.py`, `src/modeling/loaders.py`, `src/modeling/evaluate.py`,
`src/modeling/gbm.py`, `src/modeling/mlp_head.py`, `src/modeling/sweep_select.py`, `src/calibration.py`,
`src/fm_embeddings.py`, `src/spatial_features.py`, `src/fgates.py`, `src/mapping.py`,
`scripts/sweep.py`, `scripts/train_gbm.py`, `scripts/bank_calibration.py`,
`scripts/bank_calibration_f.py`, `scripts/f_leg_b_loio.py`, `scripts/striping_a1_loio.py`,
`scripts/f_region_staged.py`, `scripts/run_modeling_slim.py`, `scripts/parity_check.py` (head),
`scripts/probes/_w2_fang_probe.py`, `_w2_fang_heads.py`, `_diag_tier1_isotonic.py`,
`_w1_build_dossier.py` (head), `_smoke_gbm_one_fold.py`, `_fm_reliability_validation.py` (fold logic),
`_stage6c_gate_v2.py` (CV logic), `run_stage6a_repackage.py` (head).

**Grepped / spot-read only:** `src/modeling/cnn.py`, `src/reliability.py`, `src/stage7d_pooled.py`,
`src/leveling.py` (only `heldout_edge_cv`/`solve_offsets*` call sites), `scripts/sweep_binary.py`,
`scripts/train_binary.py`, `scripts/sweep_cnn.py`, `scripts/train_cnn.py`,
`scripts/train_deployable_head.py`, the remaining `scripts/probes/_diag_tier2_*` and `_w1_*` probes,
and `notebooks/_build_*.py` (only `_build_23`, and `_build_19`'s `VOK` line).

**Empirical checks run** (read-only, over `dataset_v2/labels/*.parquet`,
`dataset_v2/packaged/*/`, `models/deployable/*`, `reports/figures/*.csv`): the 38-fold LOIO
disjointness audit; the `(ti,tj)`-vs-world-bbox duplicate test; the inter-image footprint-gap matrix;
the label∩X column intersection; the quadrant-boundary band fractions; the banked calibration knots
and head recipe card.

**Could not check.** (1) I did not run the deployed `DeployableHead` or the Fang embedder, so I could
not *measure* any prediction-marginal claim — the refutation of the LOIO-vs-deploy calibrator concern
is analytic, not measured. (2) I did not run the fold-by-fold disjointness audit on
`dataset_v2/packaged/within_image_4fold` (152 folds) as I did for LOIO; the quadrant logic was verified
by reading plus the boundary-band computation from the split JSON. (3) `dataset_v2_dev` (the 5-image
Stage-6a dev cohort) is not on disk, so leakage-1's magnitude is computed on the full v2 within-image
split rather than the exact cohort those banked Δs were measured on. (4) Whether BoulderNet's inference
footprint equals the HiRISE footprint (an interior detector gap labelled as zero) is upstream of this
repo and untestable here; it would be a *label* defect, not a split defect. (5) I did not attempt to
quantify how much of Stage 6a's S=32 `Δ meaningful AUC = +0.0716` is the boundary band — that needs a
rerun with `buffer_tiles >= 2`, which is out of scope for a read-only review.
