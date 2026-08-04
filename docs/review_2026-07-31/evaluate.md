# Review area: evaluate

- **Reviewed at commit:** da884c7
- **Date:** 2026-07-31
- **Verification:** self-refuted (single-agent pass; not independently verified)

## Findings

### evaluate-1 — `aggregate_fold_metrics` silently drops NaN folds, so the S=128 dev Spearman (0.406, quoted as "20 folds") is a mean over 5 folds — and it is the "indirect evidence" that launched Stage 6a

- **Severity:** high
- **Liveness:** live-shipped (the aggregator is used by every sweep/train script); the number it corrupted belongs to a closed leg but is still asserted in a live doc
- **Confidence:** high (measured directly from the committed artifact + its `predictions.parquet`)
- **Where:** `src/modeling/evaluate.py:390-394` (`mean_std`), `:396`, `:403`, `:415-426`;
  `src/modeling/evaluate.py:63-68` (`spearman_safe` returns NaN on constant `y_pred`).
  Corrupted artifact: `models/lightgbm_two_stage/18c6431eb58ecc44/scale_S128_within/metrics.json`.
  Consumers: `docs/modeling_results.md:1068-1082`, `PLAN_ModelImprovement.md:145-146`,
  `PROMOTION_QUEUE.md:191-192`, `:653-654`, `:708-710`, `:727`, `DECISIONS.md:1605`.

`mean_std` filters out NaN fold values and returns `len(vals)`, but only the Spearman call site keeps
that count (`spearman_n`, `:407`); every other aggregated key throws it away (`_` at `:397-399`, and
`:424` discards it for all ten H1 keys). The dict emitted alongside them advertises `n_real_folds`
(`:403`), so a reader naturally reads `<metric>_mean` as a mean over `n_real_folds`. `spearman_safe`
returns NaN whenever the *model's* prediction is constant (`:65`), so the dropped folds are
model-dependent and are exactly the folds where the model had zero ranking skill — dropping them is
directly optimistic.

This is not hypothetical. At S=128 the within-image quadrants have only 41–101 test tiles, and the
two-stage hurdle emitted a **single constant prediction in 15 of 20 folds** (`uniq_pred == 1` in the
run's `predictions.parquet`). Those 15 folds are silently discarded; `spearman_rho_mean` is the mean
of the surviving 5.

- **Failure scenario:** `docs/modeling_results.md:1068` heads the table "dev (within-image, 20 folds)"
  and `:1077` reports **S=128 ρ = 0.406 ± 0.18**. The artifact says `spearman_n = 5`,
  `n_real_folds = 20`, `n_specificity_folds = 0`. Scoring the 15 constant-prediction folds at ρ = 0
  (the correct value for "no ranking information") gives **0.101**, i.e. *below* the S=8 value
  (0.118) — the entire monotone ladder collapses:

  | scale | doc ρ | folds actually averaged | ρ if degenerate folds = 0 |
  |---|---|---|---|
  | S=8 | 0.118 | 20/20 | 0.118 |
  | S=16 | 0.130 | 20/20 | 0.130 |
  | S=32 | 0.187 | 20/20 | 0.187 |
  | S=64 | 0.263 | 20/20 | 0.263 |
  | **S=128** | **0.406** | **5/20** | **0.101** |

  The claim at `docs/modeling_results.md:1079` — "Spearman 0.26 → 0.41 from S64 → S128" — and
  `PROMOTION_QUEUE.md:191-192` — "**Indirect evidence (going in)**: the S=128 scale study … jumped
  Spearman 0.26 → 0.41" — are therefore artifacts of the NaN drop. That "indirect evidence" is the
  stated justification for opening **Stage 6a** (spatial-context neighbour features), a whole
  research leg. The `bc_ge_1` column of the same table degrades the same way for a different reason
  (`n_specificity_folds` rises with scale): S=8 `auc_n`=20, S=16 20, S=32 17, S=64 12, **S=128 5** —
  all reported as "20 folds".
- **Evidence:**
  ```
  src/modeling/evaluate.py:390-394
      def mean_std(key: str, source: list[dict]) -> tuple[float, float, int]:
          vals = [f[key] for f in source if not np.isnan(f[key])]
          if not vals:
              return float("nan"), float("nan"), 0
          return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)

  src/modeling/evaluate.py:397-399   # the count is discarded at every non-Spearman call site
      rmse_log1p_mean, rmse_log1p_std, _ = mean_std("rmse_log1p", real)
      rmse_raw_mean,   rmse_raw_std,   _ = mean_std("rmse_raw", per_fold)
      auc_mean,        auc_std,        _ = mean_std("presence_auc", real)

  src/modeling/evaluate.py:423-426   # ...and for all ten H1 keys
          if real and key in real[0]:
              m, s, _ = mean_std(key, real)
              out[f"{key}_mean"] = m
              out[f"{key}_std"] = s

  src/modeling/evaluate.py:65
      if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
          return float("nan")
  ```
  ```
  models/lightgbm_two_stage/18c6431eb58ecc44/scale_S128_within/metrics.json
    aggregate: spearman_rho_mean 0.4056, spearman_rho_std 0.1786,
               spearman_n 5, n_real_folds 20, n_specificity_folds 0
  predictions.parquet, folds with exactly one distinct y_pred:
    2,3,4,5,7,8,11,12,13,14,15,16,17,18,19  (15 of 20)
  ```
  ```
  docs/modeling_results.md:1068  "Extending the ladder to **S=128 (640 m)** on dev
                                  (within-image, 20 folds; ...)"
  docs/modeling_results.md:1077  "| **S=128** | **640 m** | **0.406** | **0.573** |"
  PROMOTION_QUEUE.md:191-192     "**Indirect evidence (going in)**: the S=128 scale study
                                  jumped Spearman 0.26 -> 0.41 at S=64 -> S=128 dev within-image."
  ```
- **Self-refutation attempted:** (a) Maybe the NaNs are legitimate degenerate-truth folds — no:
  `n_specificity_folds = 0`, every fold has non-constant `y_true`; the NaN comes from constant
  `y_pred`. (b) Maybe the count is surfaced elsewhere — `spearman_n` *is* in `metrics.json` and in
  the sweep `aggregate.parquet`, but no doc or notebook reads it, and the doc asserts the wrong n
  outright. (c) Maybe the S=128 blast radius is limited to one dev run — the *other* four scales are
  20/20, so this run is the only one whose ladder position is fabricated, but that one run is the
  cited evidence for Stage 6a. (d) Maybe the full-v2 LOIO decisions are affected too — I checked all
  236 committed `metrics.json`: for the 38-fold `loio_nfold` runs `meaningful_auc`/`pr_auc` are
  defined on 37–38/38, so the S=32-vs-S=64 PR-AUC comparison at `DECISIONS.md:2507-2512` is over
  matched folds and survives. The damage is concentrated in the small-fold within-image sweeps.
  (e) `DECISIONS.md` was grepped for `S=128` / `spearman_n` / "20 folds" — the effective-n caveat is
  recorded once for a *different* metric (`docs/modeling_results.md:958`, bc_ge_1 at S=64 "computed
  on 26 of 38 images"), which shows the project knows the failure mode but did not apply it here.
- **Fix:** in `aggregate_fold_metrics` emit `f"{key}_n"` for every aggregated key (one line at
  `:426`, plus `rmse_log1p_n` / `rmse_raw_n` / `presence_auc_n`), and additionally count
  constant-prediction folds separately (`n_degenerate_pred`) rather than letting them share the NaN
  channel with genuinely undefined folds. Then correct `docs/modeling_results.md` §10.2 to state the
  per-cell n and retract the "0.26 → 0.41" claim (and the `PROMOTION_QUEUE.md` / `DECISIONS.md:1605`
  restatements of it).

---

### evaluate-2 — the classification aggregator computes `pr_auc` / `precision@5%` per fold and then throws them away, leaving ROC-AUC (= presence AUC on `bc_ge_1`) as the only aggregate discrimination metric

- **Severity:** medium
- **Liveness:** live-shipped (classification is the frozen recipe's task)
- **Confidence:** high
- **Where:** `src/modeling/evaluate.py:496-524` (`aggregate_fold_metrics_classification`) vs
  `:415-426` (the regression aggregator, which *does* aggregate them); per-fold values computed and
  discarded at `:469-474`. Reported surface: `scripts/sweep_within_image.py:58`
  (`BINARY_TARGET_ID = "bc_ge_1"`), `scripts/sweep_binary.py`, `docs/modeling_results.md:1071-1077`,
  `:552-555`.

`per_fold_metrics_classification` computes `pr_auc`, `normalised_lift` and
`precision/recall_at_top_{1,5,10}pct` for every real fold (`:469-474`), but
`aggregate_fold_metrics_classification` returns only `auc / brier / ece / lift_at_top_k` — the H1 loop
that exists at `:415-426` for regression has no counterpart. Consequently every classification run's
`metrics.json["aggregate"]` — including the frozen recipe's task — surfaces ROC-AUC as its sole
discrimination number, and CLAUDE.md's mandated `pr_auc@1e-2` / `precision@5%` are absent. When the
target is `bc_ge_1` (`boulder_count >= 1`), that ROC-AUC *is* presence AUC under a different name;
`src/modeling/binary_target.py:61-63` says so in a comment ("bc_ge_1 is presence … the wrong
operationalization") yet `scripts/sweep_within_image.py:58` still pins it. This is the reason the
frozen-recipe headline (`pooled_pr_auc 0.7832`) had to be recomputed by a bespoke `verdict()` in
`scripts/probes/_w2_fang_probe.py:157-195` instead of being read from the artifact — two independent
implementations of the same headline metric.

- **Failure scenario:** a future session opens `models/fang_probe/<hash>/metrics.json` (or any
  `sweep_binary` / `sweep_within_image` artifact) to recover a run's quality and finds only
  `auc_mean`. For the Stage-5c cells that number is presence AUC, and the pinned artifact
  (`models/_sweep_within_image/20260527T175437Z/summary.parquet`) predates the H1 metrics entirely —
  its columns are `presence_auc` (regression arm) and `auc` on `bc_ge_1` (classifier arm), i.e. two
  Mann-Whitney statistics against the *identical* label vector (`fa > 0` ≡ `boulder_count >= 1`).
  `docs/modeling_results.md:812-816` nevertheless reads them as corroborating framings: "Three
  independent target framings (regression, binary classification, within-image CV) at the same
  ceiling is the strongest evidence we have that the binding constraint is signal, not data
  quantity."
- **Evidence:**
  ```
  src/modeling/evaluate.py:512-524   (nothing after lift_at_top_k_std)
      return {
          "n_real_folds": len(real), "n_specificity_folds": len(spec),
          "auc_mean": auc_mean, "auc_std": auc_std, "auc_n": n_auc,
          "brier_mean": ..., "ece_mean": ..., "lift_at_top_k_mean": ..., }

  src/modeling/evaluate.py:469-474   (computed per fold, never aggregated)
      out["pr_auc"] = pr_auc(y_true_binary, y_pred_prob)
      out["normalised_lift"] = normalised_lift_at_top_k(y_true_binary, y_pred_prob)
      for k_frac, name in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")):
          p, r = precision_recall_at_k_frac(y_true_binary, y_pred_prob, k_frac)

  scripts/sweep_within_image.py:57-58
      # PLAN_Stage5c.md §5: the binary variant runs at bc_ge_1 (best Stage 5b cell).
      BINARY_TARGET_ID = "bc_ge_1"
  ```
- **Self-refutation attempted:** R02 already covers `presence_auc` in the *regression* pack, so I
  checked whether this is the same defect — it is not: different function, different key, different
  callers, and the missing-aggregation half has nothing to do with presence AUC. I also checked
  whether the presence framing is already disclaimed: `docs/modeling_results.md:28-45` ("Framing
  correction (2026-06-03)") disclaims the *"presence-AUC ceiling ~0.55–0.62"* headline, and `:1185`
  concedes "the binary reframing doesn't help" was at "the wrong threshold". That materially reduces
  the doc-side novelty (I have therefore *not* filed the §7 conclusion as a separate finding), but
  the code defect — the mandated metrics being computed and discarded, leaving only ROC-AUC in every
  classification artifact — is untouched by that disclaimer and is live.
- **Fix:** add the same H1 loop to `aggregate_fold_metrics_classification` (aggregate `pr_auc`,
  `normalised_lift`, `precision/recall_at_top_*` over `real`, with an `_n` per key per
  evaluate-1), and either retire `bc_ge_1` from `scripts/sweep_within_image.py` or rename its
  aggregate key to make the presence semantics unmissable.

---

### evaluate-3 — `precision_at_top_5pct` is hard-capped by the fold's base rate, and the unweighted fold-mean of it is reported as a quality level

- **Severity:** medium
- **Liveness:** live-shipped
- **Confidence:** high (ceiling measured on committed artifacts)
- **Where:** `src/modeling/evaluate.py:288-311` (`precision_recall_at_k_frac`), aggregated at
  `:419` / `:424-426`; contrast `:271-285` (`normalised_lift_at_top_k`, which fixes exactly this
  problem for the sibling metric). Quoted as a level at `DECISIONS.md:2507`.

`precision_recall_at_k_frac` takes `k = round(0.05·n)` and returns `tp / k`. If the fold has
`n_pos < k` the metric is mathematically capped at `n_pos / k = base_rate / 0.05` — a *perfect*
ranker on an image with a 1.3 % rich-tile base rate scores 0.26. `normalised_lift_at_top_k`'s
docstring diagnoses the mirror-image problem for lift ("Lift saturates at `1 / base_rate`, so
high-base-rate images can never reach raw lift above ~1.3 even with a perfect classifier …
Comparable across images with different base rates") and divides it out; no such correction exists
for precision@k, and `aggregate_fold_metrics` averages the raw values with equal weight per fold.

- **Failure scenario:** on the frozen scale/target the rich-tile base rate ranges from 0.0015 to 0.97
  across the 38 LOIO images, and **10 of 38 folds sit below 5 %**. For
  `models/fang_tier2/tier2_mlp_reg_emb_fractional_area_S32/1e01ad8b17447599/metrics.json` the
  reported `precision_at_top_5pct_mean = 0.5906`, but the mean attainable ceiling over the same folds
  is **0.8711**; the ceiling-normalised value is 0.6239. So ~30 % of the shortfall from 1.0 is base
  rate, not model quality — and any comparison of that level across scales, targets or cohorts (which
  change the base-rate mix) is confounded. Measured on three committed runs:
  `_sweep_stage6a/.../within_image_4fold/…/scale_S32` (10/20 folds under 5 %, mean 0.5160 vs ceiling
  0.7894), `lightgbm_log1p_huber/615bf0ebe05ac3d1/scale_S64_target_fractional_area` (11/38, 0.4776 vs
  0.8423), and the tier-2 cell above (10/38).
- **Evidence:**
  ```
  src/modeling/evaluate.py:299-311
      n = y_true_binary.size
      n_pos = int(y_true_binary.sum())
      if n_pos == 0:
          return float("nan"), float("nan")
      k = max(1, int(round(k_frac * n)))
      ...
      precision = tp / k          # <= n_pos / k  == base_rate / k_frac

  src/modeling/evaluate.py:274-279  (the fix that exists for lift but not for precision@k)
      """Lift saturates at `1 / base_rate`, so high-base-rate images can never reach
      raw lift above ~1.3 even with a perfect classifier.  Normalised lift is in
      [base_rate, 1] ... Comparable across images with different base rates."""
  ```
- **Self-refutation attempted:** (a) Deltas between two arms at the same scale/target share the same
  `y_true`, hence the same ceilings, so the *gate arithmetic* at `DECISIONS.md:2509` ("prec@5%
  +0.020") is unaffected — this survives only as a defect in the reported *level* and in any
  cross-scale/cross-target comparison. (b) The pooled headline `prec@5% 0.948` comes from
  `_w2_fang_probe.verdict()` over the concatenated 38-image vector, where the cohort base rate is
  ~0.35, well above 5 % — so the headline itself is not capped; only the per-fold-mean variant in
  `metrics.json` / `summary.parquet` is. (c) I grepped `DECISIONS.md` for an existing acknowledgement
  of the precision@k ceiling and found only the lift one.
- **Fix:** emit `precision_at_top_5pct_normalised = precision / min(1, base_rate / k_frac)` alongside
  the raw value (mirroring `normalised_lift_at_top_k`), or emit `meaningful_base_rate` into the
  aggregate so the ceiling is at least visible next to the mean.

---

### evaluate-4 — `per_bin_rmse`'s top bin is labelled `1e-2_to_max` but hard-coded to stop at 1.0, so on non-fractional-area targets it silently drops most of the data with no partition check

- **Severity:** low
- **Liveness:** live-shipped (latent — no current figure depends on it)
- **Confidence:** high (measured)
- **Where:** `src/modeling/evaluate.py:38-47` (labels + `POSITIVE_BIN_EDGES`), `:102-127` (the only
  assertion checks *edge count*, never coverage), `:354` (emitted into every regression
  `metrics.json`).

The bin table is built as `zero` plus half-open `(edge[i-1], edge[i]]` intervals with the last edge
fixed at `1.0`, while the label says `1e-2_to_max`. For `fractional_area` that is a complete
partition of `[0, 1]`. For `boulder_count` (a live target — `run_loio`'s docstring at `:583-587` and
`tests/test_evaluate_meaningful_threshold.py` exist precisely because count targets are used) every
tile with `count > 1` falls outside every bin and vanishes, silently and without warning. There is no
`assert sum(n_tiles) == y_true.size`.

- **Failure scenario:** across the 236 committed `metrics.json`, **40 runs** have per-bin tables that
  drop tiles — all count-target runs — losing **66–88 %** of them (e.g.
  `models/fang_tier2/tier2_mlp_reg_emb_boulder_count_S32/…`: 118,286 of 161,005 tiles absent;
  `models/lightgbm_two_stage/2a7d671cb7711753/scale_S64_target_boulder_count/…`: 32,760 of 37,315,
  87.8 %). Any reader who plots that table — `notebooks/_build_11.py:481`,
  `notebooks/_build_22.py:249-264`, `notebooks/_build_10.py:323` all consume `per_bin_rmse` from a
  `metrics.json` path chosen by a variable — sees a "top bin" that is really `1e-2 < count <= 1`,
  i.e. an empty set, and a "compression curve" built on the 12–34 % of tiles that happen to lie
  below 1.
- **Evidence:**
  ```
  src/modeling/evaluate.py:38-45
      ABUNDANCE_BIN_LABELS = ("zero", "0_to_1e-4", "1e-4_to_1e-3", "1e-3_to_1e-2", "1e-2_to_max")
      POSITIVE_BIN_EDGES: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1.0)

  src/modeling/evaluate.py:102-113   (only the edge COUNT is asserted)
      assert len(positive_edges) == n_pos_bins + 1, (...)
      ...
          lo, hi = positive_edges[i - 1], positive_edges[i]
          mask = (y_true > lo) & (y_true <= hi)
  ```
- **Self-refutation attempted:** (a) `scripts/probes/_fm_tier2_regression.py:68-69` documents the
  caveat ("per_bin_rmse still uses fractional_area bin edges, so for the count target read Spearman +
  meaningful_auc(@50), not the per-bin table") — so it is *known* for one probe, which is why I rank
  this low rather than medium. (b) The sweep scripts that use count targets all strip the table
  before writing their summaries (`_sweep_target_reformulation.py:261`, `_sweep_w0.py:217`,
  `_sweep_stage6a.py:267`, `_sweep_stage6b.py:281`, `sweep.py:178`,
  `sweep_within_image.py:148`), so no committed figure currently depends on a count-target per-bin
  table — I verified `notebooks/_build_22.py:263-264` calls `per_bin_curve` only on
  `..._fractional_area_S32` cells. (c) `src/fgates.py:290` calls `per_bin_rmse` on `fa_true` (≤ 1),
  which is safe. The defect is therefore latent, but it is a silent 88 % data loss in a function that
  writes to every regression artifact.
- **Fix:** take the upper edge from the data (`max(y_true)`, or `np.inf`) instead of the literal
  `1.0`, and add `assert int(df["n_tiles"].sum()) == y_true.size` so a mis-scaled target fails loudly
  instead of producing a plausible short table.

## Refuted by my own check

- **`presence_auc` / `meaningful_auc` via `stats.mannwhitneyu(..., alternative="greater")` returning
  the wrong U.** Checked in the project env (scipy 1.16.3): the returned statistic is U1 regardless of
  `alternative` (0.0 for all three alternatives on a separated pair), and ties get exactly 0.5 credit
  (U1 = 0.5 on `[1,2]` vs `[2,3]`). `U/(n_pos·n_neg)` is the correct tie-aware AUC. `src/modeling/evaluate.py:130-142`.
- **Cross-scale mis-join in `fang_columns_for_keys`** (`loaders.py:327-347`) — the join is on
  `(obs_id, ti, tj)` with no `scale_idx`/`tile_size_px` guard, so I expected a silent wrong-tile join
  when `px` and `scale_idx` disagree. Empirically it does not: joining S=64 fold keys onto the P96
  (S=32) store raises `AssertionError: Fang store is missing tiles present in the keys` at `:345`.
  Latent-only.
- **NaN embedding rows scored as real predictions.** `load_fang_store:318` NaNs out `valid == False`
  rows; I expected margin tiles to be median-imputed and then scored. The v2 P96 store has **0**
  invalid rows out of 161,005, and fold 0's test block has 0 all-NaN rows.
- **`_standardize_matrix_per_group` nulling partially-NaN columns.** `loaders.py:215-231` uses
  `mean`/`std`/`median`/`percentile` and `scipy.rankdata` (whose `nan_policy` in scipy 1.16 turns a
  column containing one NaN into all-NaN), so a partially-NaN feature column would be destroyed for
  that image. Measured on `dataset_v2/packaged/loio_nfold` at all four scales: **every** NaN column is
  *fully* NaN within every image (629 (image,col) pairs at S=8, 74 at S=16, 0 at S=32/S=64; partially
  NaN = 0 everywhere). No behaviour change.
- **`EMPTY_TRUTH_OBS_ID = "ESP_065711_1545"` hardcoded in `src/` (invariant 7).** It forces
  `is_specificity_only` by ObsId (`:329`, `:444`), which would wrongly exclude the fold if that image
  ever gained labels. It is not in `hirise_40_vclaire.csv` (the v2 cohort), the `np.unique(y_true).size
  < 2` fallback catches any *new* empty-truth image, and the choice is documented across
  `PLAN_modeling.md:473`, `PLAN_Stage5.md:89`, `config.yaml:167`. Deliberate; harmless today.
- **`run_loio`'s inner-val rotation colliding with the held-out group under the within-image scheme.**
  `unique_train[fold_idx % n]` (`:619`) with quadrant-coded groups: `src/dataset.py:747-752` writes
  quadrant indices 0–3 into the groups arrays and the test quadrant is excluded from `groups_train` by
  construction, so the assert at `:621` can never fire and the inner-val is always one of the 3
  non-test quadrants.
- **`precision@5%` k off-by-one.** `k = max(1, int(round(0.05·n)))` (`:303`) vs
  `k = max(1, int(0.05·n))` in `_w2_fang_probe.verdict():164` — two implementations, but at the
  cohort sizes in play (n ≥ 5,000) they agree, and both are "top 5 % of predictions" (the correct
  definition). Tie-breaking is `np.argpartition` (memory order) vs `np.argsort` (quicksort): arbitrary
  but deterministic, and predictions are continuous, so no observed effect.
- **Empty/1-row folds crashing or producing silent zeros.** A zero-row fold would make
  `is_specificity_only` true and every metric NaN, but `model.fit` on an empty `X_inner_train` raises
  first — loud.
- **`aggregate_fold_metrics`'s `if real and key in real[0]` only probing fold 0.** Both branches of
  `per_fold_metrics` (`:339-381`) set every H1 key unconditionally, so no fold can be missing one.
- **`write_run_artifacts` staleness.** It overwrites `predictions.parquet` / `metrics.json` /
  `snapshot.json` in a caller-chosen `config_hash` directory (`:722-753`). I looked for a *new*
  collision channel and did not find one beyond R04's ("packaged splits keyed on scheme name only, no
  `split_hash` check in `loaders.load_metadata`"): `sweep.py`'s snapshot includes variant, target,
  scheme, dataset_dir, scale and model params, so two genuinely different configs cannot collide. The
  residual risks are (i) stale `fold_<obs_id>/` sibling directories left behind if a re-run has a
  different cohort, and (ii) `json.dumps(default=float)` writing bare `NaN` literals (valid for
  Python's `json`, invalid for strict parsers) — neither reached the bar.

## Verified clean

- `presence_auc` / `meaningful_auc` AUC arithmetic and its single-class guards (`:138-142`,
  `:365-381`); `pr_auc`'s `n_pos`/`n_neg` guard (`:263-268`); `brier_score`; `expected_calibration_error`
  and `calibration_deciles` bin indexing (`np.digitize(p, edges[1:-1])` clipped to `[0, n_bins-1]`
  gives the correct 10 half-open bins with 1.0 landing in the last).
- `lift_at_top_k` / `normalised_lift_at_top_k` (`:223-285`) — `n_pos == 0 or n_pos == n` guard is
  right, and the normalisation identity `raw · base_rate = precision@n_pos` holds.
- `per_fold_metrics`' specificity flagging and the "always emit `per_bin_rmse`" choice; `rmse_log1p`'s
  clip-at-0 on both sides.
- `run_loio`'s inner-validation rotation is drawn from the *training* groups only and asserted
  disjoint from the held-out set (`:610-622`) — no early-stopping leak from this harness.
- `loaders.load_fold` feature-column determinism: `feat_cols` is computed once from the **train**
  frame and reused verbatim for `X_test` (`:142-146`), so train/test column order cannot permute;
  `_feature_columns` correctly excludes tile keys, `config_hash_feat` and `patch_idx_S*`.
- `standardize_fold_per_image` / `augment_fold_with_per_image` / `augment_fold_with_fang` each derive
  train and test statistics from their own side only — no split-boundary leak in `loaders.py`.
- `fang_columns_for_keys`' `validate="one_to_one"` + missing-row assert (`:342-345`).

## Coverage note

Read in full: `src/modeling/evaluate.py` (753), `src/modeling/loaders.py` (433),
`tests/test_modeling_evaluate.py`, `tests/test_evaluate_meaningful_threshold.py`,
`src/modeling/binary_target.py`, `scripts/sweep.py`, `scripts/sweep_within_image.py`.
Read in part: `scripts/probes/_w2_fang_probe.py` (`verdict`), `scripts/probes/_w2_fang_heads.py`,
`scripts/probes/_fm_tier2_regression.py`, `src/dataset.py:720-842`, `src/fgates.py:247-296`,
`src/modeling/mlp_head.py:290-305`, `notebooks/_build_10.py` (within-image section),
`notebooks/_build_11.py`/`_build_22.py` (per-bin consumers), `docs/modeling_results.md` §§7, 9.3,
10.2, 11.4 and the 2026-06-03 framing correction, `PROMOTION_QUEUE.md`, `PLAN_ModelImprovement.md`.
Grepped only: `DECISIONS.md` (by term: `pooled`, `prec@5`, `S=128`, `ESP_065711`, `precision_at_top`),
`PLAN_Stage5b/5c`, the remaining probe sweeps.

Numerical checks run (read-only, over committed artifacts): all 236 `models/**/metrics.json`
scanned for NaN-per-fold counts vs `n_real_folds` and for `per_bin_rmse` tile coverage; the S=128
within-image `predictions.parquet` inspected for per-fold prediction uniqueness; base-rate ceilings
computed for three runs; NaN structure of `dataset_v2/packaged/loio_nfold` X matrices at four scales;
the P96 embedding store's valid-row count and a deliberate cross-scale join attempt; scipy
`mannwhitneyu`/`rankdata` semantics in the `geospatial` env.

Not checked: `tests/test_modeling_evaluate_classification.py` and `tests/test_modeling_loaders.py`
were only skimmed for names, not audited for false assurance (that is the `tests` area's remit —
note `tests/test_modeling_evaluate.py` has **no** test that pins `spearman_n` against
`n_real_folds`, which is why evaluate-1 was invisible). I did not run `pytest`, any sweep, or any
model fit. `src/modeling/gbm.py`'s `eval_set` provenance, `sweep_select.py`'s selection bias, and the
calibration layer's consumption of these predictions are out of area (`modeling-heads`, `leakage`,
`calibration`). The pooled-vs-per-image question is only partly answered here: `run_loio` itself never
pools (it returns a per-fold concatenation), and the pooling that produces the headline `0.7832` lives
in `_w2_fang_probe.verdict()` and `src/fgates.pooled_skill` — that pooled ranking mixes 38 per-fold
models' output scales, but the project already knows and exploits this sensitivity
(`scripts/f_h4_legb.py` deliberately shifts per-obs logits and re-scores pooled PR-AUC), so I left it
to `stats-fallacies` rather than filing it.
