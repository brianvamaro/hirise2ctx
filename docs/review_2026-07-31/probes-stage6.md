# Review area: probes-stage6

- **Reviewed at commit:** `da884c7`
- **Date:** 2026-08-02
- **Verification:** self-refuted (single-agent pass; not independently verified). Every number below
  was re-derived from on-disk artifacts (`models/_sweep_stage6{a,b}/…/{summary,aggregate}.parquet`,
  `…/loio_nfold/8c7523615964f5cb/scale_S64/predictions.parquet`, `cache/stage6c/*.parquet`,
  `dataset_v2/features_ctx_illum/*.parquet`) with read-only pandas/scipy snippets. No probe was run.

**Headline.** The probes' arithmetic is right — I reproduced the pooled Strategy-B numbers to 4 dp
and the H3 correlation grid exactly. What does not survive is the *interpretation*: per-image
`pr_auc`, `normalised_lift_meaningful` and `precision_at_top_5pct` are rank-equivalent to the image's
**meaningful base rate** (Spearman +0.983 / +0.981 / +0.918 across the 38 images), so the Stage-6b
"mechanism EMPIRICALLY VALIDATED" claim and the Stage-6c gate's own target variable are measuring
prevalence, not model quality.

---

## Findings

### probes-stage6-1 — The "Stage 6e / CTX-source-heterogeneity mechanism is EMPIRICALLY VALIDATED" verdict is a per-image prevalence confound: partialling out the base rate kills 10 of the 12 `p < 0.05` cells it rests on

- **Severity:** high
- **Liveness:** dead-closed programme, but the claim is quoted as a *live* validated mechanism in
  `ROADMAP.md:30`, `README.md:70`, `PLAN_ModelUsability.md:35` and six places in the reader-facing
  `docs/modeling.md`
- **Confidence:** high (measured; the raw correlations reproduce exactly, and the partials are a
  three-line calculation on the same table)
- **Where:** `scripts/probes/_diag_stage6b_h3_check.py:133-145` (the 4×5 correlation grid),
  `:80-92` (`_spearman_block`); consumers `PROMOTION_QUEUE.md:145-156`, `:511-533`,
  `docs/modeling_results.md:1398-1423`, `docs/modeling.md:421`, `:490-491`, `PLAN_ModelUsability.md:35`,
  `ROADMAP.md:30`, `README.md:70`

The probe's pre-declared test — `rho(per-image AUC, mean_ctx_incidence) < -0.30, p < 0.05`
(`_diag_stage6b_h3_check.py:6-7`, restated as half of the Stage-6b acceptance criterion at
`_sweep_stage6b.py:19-24`) — **failed** (+0.050, p = 0.765 on `pr_auc`). The record then substitutes a
post-hoc 4-feature × 5-metric grid the same probe prints, harvests the 12 cells at `p < 0.05`, and
declares the mechanism validated. But four of the five metrics in that grid are essentially the
image's positive base rate re-expressed: measured over the 38 images,
`Spearman(pr_auc, meaningful_base_rate) = +0.983`, `(normalised_lift_meaningful, base_rate) = +0.981`,
`(precision_at_top_5pct, base_rate) = +0.918` — and the ctx features themselves correlate with the
base rate (`std_ctx_incidence` −0.371, `mean_n_sources` −0.299, `dominant_source_frac` +0.338). Once
the base rate is partialled out, the grid collapses.

- **Failure scenario:** a reader of `docs/modeling.md:490-491` or `PLAN_ModelUsability.md:35` takes
  "when a HiRISE footprint is stitched from many CTX sources the model performs worse on that image"
  as an established model-degradation mechanism and spends effort on a seam-based fix. What the data
  actually support is "images stitched from more CTX sources contain a smaller fraction of
  boulder-rich tiles" — a statement about where CTX coverage is fragmented relative to terrain, with
  no model defect in it. The one metric that is *not* base-rate-driven (per-image Spearman ρ, which
  correlates only +0.317 with base rate) retains a signal, but at p ≈ 0.035–0.049 among 20 tests.
- **Evidence:**
  ```
  scripts/probes/_diag_stage6b_h3_check.py:133-142
      h3_metrics = [m for m in (
          "presence_auc", "pr_auc", "spearman_rho",
          "normalised_lift_meaningful", "precision_at_top_5pct",
      ) if m in join.columns]
      corr = _spearman_block(
          join,
          features=["mean_ctx_incidence", "std_ctx_incidence", "mean_n_sources",
                    "dominant_source_frac_mean"],
          metrics=h3_metrics,
      )

  PROMOTION_QUEUE.md:526-529
  **Stage 6e mechanism (CTX-source heterogeneity / mosaic stitching) is EMPIRICALLY VALIDATED**:
  `mean_n_sources` and `std_ctx_incidence` correlate negatively (p < 0.05) with every
  operational metric; `dominant_source_frac_mean` correlates positively.
  ```
  Raw vs base-rate-partialled Spearman (n = 38; partial = Pearson on rank residuals after
  regressing out `rank(meaningful_base_rate)`; source `models/_sweep_stage6b/20260531T020308Z/summary.parquet`
  scheme `loio_nfold`, scale_idx 3, joined to `dataset_v2/features_ctx_illum/*.parquet`):

  | feature | metric | raw ρ (p) | partial ρ (p) |
  |---|---|---|---|
  | `std_ctx_incidence` | `pr_auc` | **−0.370 (0.022)** | −0.031 (0.852) |
  | `std_ctx_incidence` | `normalised_lift_meaningful` | **−0.400 (0.013)** | −0.202 (0.224) |
  | `std_ctx_incidence` | `precision_at_top_5pct` | **−0.361 (0.026)** | −0.058 (0.731) |
  | `std_ctx_incidence` | `spearman_rho` | **−0.342 (0.036)** | −0.254 (0.123) |
  | `mean_n_sources` | `pr_auc` | **−0.326 (0.046)** | −0.182 (0.275) |
  | `mean_n_sources` | `normalised_lift_meaningful` | **−0.342 (0.036)** | −0.259 (0.116) |
  | `mean_n_sources` | `precision_at_top_5pct` | **−0.357 (0.028)** | −0.218 (0.188) |
  | `mean_n_sources` | `spearman_rho` | **−0.405 (0.012)** | **−0.343 (0.035)** |
  | `dominant_source_frac_mean` | `pr_auc` | **+0.361 (0.026)** | +0.168 (0.313) |
  | `dominant_source_frac_mean` | `normalised_lift_meaningful` | **+0.376 (0.020)** | +0.245 (0.139) |
  | `dominant_source_frac_mean` | `precision_at_top_5pct` | **+0.393 (0.015)** | +0.223 (0.178) |
  | `dominant_source_frac_mean` | `spearman_rho` | **+0.394 (0.014)** | **+0.321 (0.049)** |

  **12 of 20 raw cells are p < 0.05; 2 of 20 partials are** — and those two are the same underlying
  quantity (`dominant_source_fraction` ≈ 1/`n_sources`) against the one base-rate-free metric, both
  sitting on the 0.05 boundary with 20 tests run and no correction.

  Two further defects in the same table as published:
  1. **`PROMOTION_QUEUE.md:517` bolds `std_ctx_incidence ↔ presence_auc = −0.340` as `p < 0.05`
     under a caption that says `n = 38` (`:511`).** That column has **n = 26** — `_spearman_block`
     drops NaN pairs per cell (`_diag_stage6b_h3_check.py:84`) and `presence_auc` is undefined on 12
     single-class images. At n = 26, ρ = −0.340 gives **p = 0.089**. The probe's own writeup prints
     `n=26` for that column (`_diag_stage6b_h3_check.md:9`); the hand-transcribed table in
     PROMOTION_QUEUE drops the `n` and adds a significance mark the data do not support. It is also a
     presence-AUC column (invariant 8) used as one of the "every operational metric" it is claimed to
     validate against.
  2. **`normalised_lift_meaningful` is not the prevalence-corrected metric it is documented to be.**
     `src/modeling/evaluate.py:271-285` returns `lift_at_top_k × base_rate`, and
     `lift_at_top_k` (`:223-243`) uses `k = n_pos`, so normalised lift is exactly **precision at
     k = n_pos** (R-precision) — floored at the image's base rate for a random ranker. Its docstring
     claims "Comparable across images with different base rates"; measured, it correlates +0.981 with
     the base rate across this cohort. This **materially corrects R26**, which asserts that
     `normalised_lift_at_top_k` "fixes exactly this for the sibling metric".
- **Self-refutation attempted:** (a) *Is prevalence a legitimate part of "model reliability"?* No —
  average precision's random-ranker baseline **is** the base rate, so ρ = 0.983 with base rate means
  per-image PR-AUC in this cohort (base rates 0.027–0.998) is dominated by its own floor, not by
  discrimination. (b) *Is the mechanism separately established?* Partly, and I am careful not to
  overclaim: the later striping work independently demonstrated CTX per-frame radiometry does affect
  the embedder (notebooks 24–25, DECISIONS 2026-06-18d). That does **not** rescue *this* statistical
  validation, which is what four documents cite. (c) *Is this just R26?* No — R26 is about comparing
  `precision@5%` *levels* across populations and explicitly exempts arm-vs-arm deltas; this is a
  cross-image *correlation* used as mechanism evidence, a different use, and it additionally shows
  R26's exemption for `normalised_lift` is wrong. (d) *Was the confound noted anywhere?* Grepped
  `DECISIONS.md`, `PROMOTION_QUEUE.md`, `docs/modeling*.md` for `base_rate` / "base rate" near the
  Stage-6b entries — the only mention is the unrelated §9.3 saturation caveat.
- **Fix:** re-state the Stage-6b/6e conclusion as base-rate-conditional: report the partial
  correlations (or restrict the claim to per-image Spearman ρ, the only base-rate-free metric, and
  note it is 2 of 20 tests). Correct `PROMOTION_QUEUE.md:511-521` — add the per-cell `n`, drop the
  `**` on the `presence_auc` column, and drop the presence column entirely per invariant 8. Amend
  `ROADMAP.md:30` ("6b strict-FAIL but mechanism validated"), `README.md:70`,
  `PLAN_ModelUsability.md:35` and `docs/modeling.md:421,490-491,558,590`. Fix the
  `normalised_lift_at_top_k` docstring and R26's characterisation of it.

---

### probes-stage6-2 — Stage 6c's "bad image" label is the per-image base rate, so its strict acceptance test is passed by a prevalence oracle that contains no anti-signal information — and the implemented criterion is not the pre-declared one

- **Severity:** high
- **Liveness:** dead-closed (Stage 6c never shipped) but quoted as a verdict in `ROADMAP.md:30`
  ("6c soft-PASS") and as a validated procedure in `docs/modeling_results.md §14`
- **Confidence:** high
- **Where:** `scripts/probes/_stage6c_gate.py:60-63` (`BAD_PR_AUC_THRESHOLD`), `:161` / `:194` / `:228`
  (`y_bin = baseline_pr_auc < 0.55`), `:134` (`meaningful_base_rate` loaded and never used),
  `:407-411` (`STRICT`), `:250-283`; consumers `docs/modeling_results.md:1508-1525`,
  `PROMOTION_QUEUE.md:170-186`, `:248`

The gate is trained to predict `baseline_pr_auc < 0.55`. Given ρ(`baseline_pr_auc`,
`meaningful_base_rate`) = **+0.983**, that binary label is, to within rank noise, "this image has few
boulder-rich tiles". Every acceptance criterion is then built from the same base-rate-driven
quantities (`kept_pr_auc_mean`, `delta_norm_lift`). Consequence: **the test cannot distinguish an
anti-signal gate from a prevalence gate.** I checked directly — dropping the K images with the
*lowest true meaningful base rate* (an oracle that knows nothing about CTX sources, illumination or
model behaviour):

| K dropped by true base rate | retained mean PR-AUC (bar 0.65) | tile_kept_frac (bar 0.70) | Δ norm lift (bar +0.10) | strict? |
|---:|---:|---:|---:|---|
| 6 | 0.635 | 0.821 | +0.090 | ✗ |
| **10** | **0.698** | **0.704** | **+0.150** | **✓ PASS** |
| 12 | 0.733 | 0.644 | +0.186 | ✗ (tiles) |

So the criterion that Stage 6c is recorded as having *failed* is one that a pure prevalence sort
*passes*. Had the CTX features been slightly better proxies for prevalence, Stage 6c would have been
promoted on a gate with no mechanism in it.

Separately, **the implemented criterion is not the pre-declared one.** `PROMOTION_QUEUE.md:248`
(written before the run) declares: "held-out per-image **AUC** for 'good' images cleared ≥ 0.65 mean
(vs **0.61** baseline) AND retained tile fraction ≥ 70 %" — two criteria, on ROC-AUC, whose banked
baseline is 0.6149 (`PROMOTION_QUEUE.md:493`). The probe implements `kept_pr_auc_mean ≥ 0.65` against
a **0.543** PR-AUC baseline and adds a third, undeclared criterion (`delta_norm_lift ≥ 0.10`). The
same absolute number 0.65 therefore became a +0.107 ask instead of a +0.04 ask, on a different metric.

- **Failure scenario:** a future session re-opens the anti-signal gate (`docs/modeling_results.md:1576-1579`
  explicitly invites this with "more LOIO images or HiRISE-side priors"), reruns the probe on a larger
  cohort, clears `kept_pr_auc_mean ≥ 0.65` at ≥ 70 % tiles, and promotes a gate that is only sorting
  images by predicted rock abundance.
- **Evidence:**
  ```
  scripts/probes/_stage6c_gate.py:60-63
  # Binary "bad image" cutoff. The full-set baseline PR-AUC mean is 0.543 (from
  # aggregate.parquet); we call an image "bad" if held-out PR-AUC is below
  # this average. Threshold tunable via __main__.
  BAD_PR_AUC_THRESHOLD = 0.55

  scripts/probes/_stage6c_gate.py:128-135   # the one column that would have shown the confound
      ].rename(columns={
          "pr_auc": "baseline_pr_auc",
          ...
          "meaningful_base_rate": "baseline_base_rate",
      })
  ```
  `baseline_base_rate` is loaded into the table, cached into
  `cache/stage6c/predictor_table.parquet`, and **never referenced again** — not in `FEATURES`, not in
  `acceptance`, not in `write_markdown`. One `stats.spearmanr(table["baseline_pr_auc"],
  table["baseline_base_rate"])` would have returned 0.983. This is the same "the diagnostic stopped
  one column short" pattern the register records for **R23**.
- **Self-refutation attempted:** (a) *Is dropping low-prevalence images a legitimate deployment
  strategy?* It can be — but then the deliverable is "predict per-image abundance and reorder", not
  "detect CTX anti-signal", and the CTX features are not needed (`mean_pred` per image is already
  available). The defect is that the acceptance test has no *specificity* for the hypothesis it is
  named after. (b) *Does the oracle result depend on my choice of K?* It passes at exactly one K
  (10 of 38) with tile_kept_frac 0.7038 against a 0.70 bar — marginal, and I say so; but the actual
  gate never passed at any K, so the comparison stands. (c) *Is the criterion drift already filed?*
  Grepped the register: `other-scripts`/`docs-consistency` cover README/SHERLOCK commands and DATA
  DICTIONARY drift, not PROMOTION_QUEUE acceptance criteria. Not filed.
- **Fix:** any future image-level gate must be scored on a base-rate-free instrument (per-image
  Spearman ρ, or PR-AUC *excess over the image's base rate*), and must report the prevalence-oracle
  control alongside the gate as the null. Record in `PROMOTION_QUEUE.md` that the Stage-6c criterion
  as run differs from the one declared at `:248`.

---

### probes-stage6-3 — The "+0.056 pooled-global Strategy B" that the record calls "the deliverable" sits at the 95th percentile of a permutation null nobody computed, and v2's headline is the max over 20 configurations

- **Severity:** medium
- **Liveness:** dead-closed, but it is the one *positive* Stage-6 result and it is quoted as a soft
  PASS in `ROADMAP.md:30` and as a "documented procedure" in `docs/modeling_results.md:1556-1558`
- **Confidence:** high (baseline and all three Strategy-B values reproduce to 4 dp; the null is a
  500-draw permutation I ran on the same artifact)
- **Where:** `scripts/probes/_stage6c_gate.py:347-371` (`pooled_global_with_strategy_b`),
  `:577-586` (the writeup table); `scripts/probes/_stage6c_gate_v2.py:308-311`, `:355-360`
  (`best_pooled_b` = argmax over the sweep, printed as the headline); consumers
  `docs/modeling_results.md:1544-1558`, `PROMOTION_QUEUE.md:170-176`

`docs/modeling_results.md:1544` titles the section "clean +0.04–0.06 PR-AUC" and `:1556` says "The
+0.056 pooled-global lift **is the deliverable**". No uncertainty is attached anywhere. Strategy B
multiplies each image's predictions by a per-image constant `(1 − p_bad)` and re-pools; the
appropriate null is therefore *some other per-image reweighting with the same value distribution*,
not "no reweighting". Reproduced and tested (500 permutations of the same `p_bad` vector across the
38 held-out images, seed 0):

| gate | observed Δ | null mean | null sd | null p95 | one-sided perm p |
|---|---:|---:|---:|---:|---:|
| ridge (the quoted one) | **+0.0563** | −0.0409 | 0.0497 | +0.0393 | 0.030 |
| logreg | +0.0368 | −0.0036 | 0.0290 | +0.0440 | 0.086 |
| `mean_n_sources > median` rule | +0.0352 | −0.0585 | 0.0460 | +0.0187 | 0.034 |

Family-wise (max over the three v1 gates, 400 draws, seed 1): observed **+0.0563** vs null-max p95
**+0.0510**, **p = 0.035**. So the "clean" headline is a marginal result at n = 38 images, selected
post hoc as the best of three gates (`docs/modeling_results.md:1556` "The **v1 ridge gate** wins"),
with a null spread (sd ≈ 0.05) *as large as the effect*. In v2 the selection is explicit: the probe
tracks `best_pooled_b` across 5 gates × 4 cutoffs and prints the argmax (+0.0414) as the headline,
while the same sweep's pooled-B deltas span **−0.044 to +0.041** (`_stage6c_gate_v2.md:23-42`) — i.e.
the reported maximum is inside the spread of the configurations it was chosen from. The gates'
own LOIO ROC-AUCs are 0.28–0.61, i.e. chance to barely-better, which is the internal signal that this
should have been null-tested.

- **Failure scenario:** a future session reads `ROADMAP.md:30` "6c soft-PASS" plus "the deliverable"
  and implements the per-image confidence weight in the deployment path, on an effect that a
  1-in-30 permutation draw reproduces.
- **Evidence:**
  ```
  scripts/probes/_stage6c_gate.py:361-363
      p_bad_per_tile = per_tile["fold_held_out_obs_id"].map(p_bad_by_obs).to_numpy()
      p_bad_per_tile = np.nan_to_num(p_bad_per_tile, nan=0.0)
      y_pred_adj = per_tile["y_pred"].to_numpy() * (1.0 - p_bad_per_tile)

  scripts/probes/_stage6c_gate_v2.py:308-311
              # Track best pooled-B.
              cand_b = (g.name, cutoff, delta_b, g)
              if best_pooled_b is None or cand_b[2] > best_pooled_b[2]:
                  best_pooled_b = cand_b
  ```
  Reproduction: baseline pooled AP = **0.6086** (37,315 tiles, 17,920 positive at
  `y_true >= 50`, base rate 0.480) → ridge Strategy B **0.6649**, Δ **+0.0563** — matching
  `_stage6c_gate.md:126-131` exactly.
- **Self-refutation attempted:** (a) *Is the permutation null too harsh?* It preserves the exact
  multiset of `p_bad` values and only breaks their assignment to images — the minimum needed to
  destroy the gate's information while keeping the treatment magnitude identical. (b) *Is p = 0.03
  a pass?* Uncorrected and post-hoc-selected, no: 3 v1 gates + 20 v2 combos were tried. (c) *Is this
  just R41?* R41 is about the striping/F programme's ±0.02 tolerances and says no gated statistic has
  a sampling spread; this is a different programme, a different statistic, and an actual measured
  null — it extends R41's pattern rather than restating it. (d) *Is the Strategy-B mechanism itself
  invalid?* No — it is a legitimate operation, and the rank-invariance argument at
  `_stage6c_gate.py:562-566` is correct. The defect is the missing null.
- **Fix:** re-report the Strategy-B delta against the permutation null (`Δ − null_mean` with the null
  p95 beside it), and label v2's headline as an argmax over 20 configurations. If the effect is to be
  kept, pre-register one gate and one cutoff and test that one.

---

### probes-stage6-4 — `std_ctx_incidence` is two different statistics in the two probes, and the record says Stage 6c reuses the Stage-6b-validated feature

- **Severity:** medium
- **Liveness:** dead-closed
- **Confidence:** high (the two probes report different ρ for the "same" feature against the same
  metric: −0.370 vs −0.331)
- **Where:** `scripts/probes/_diag_stage6b_h3_check.py:57` vs `scripts/probes/_stage6c_gate.py:88`
  (and `_stage6c_gate_v2.py:75`); consumers `PROMOTION_QUEUE.md:248`, `:158-160`,
  `docs/modeling_results.md:1490-1493`

```
scripts/probes/_diag_stage6b_h3_check.py:57
    "std_ctx_incidence": float(df64["ctx_incidence_mean"].std(ddof=0)),   # between-tile spread

scripts/probes/_stage6c_gate.py:88
    "std_ctx_incidence": float(sub["ctx_incidence_std"].mean()),          # mean within-tile spread
```

The first is the dispersion of the per-tile *mean* incidence across an image; the second is the
average *within-tile* incidence spread. `PROMOTION_QUEUE.md:248` and `:158-160` say Stage 6c trains
its gate on "the now-empirically-validated features (`mean_n_sources`, `std_ctx_incidence`,
`dominant_source_fraction`)" — but one of the three is a different quantity from the one Stage 6b
validated.

- **Failure scenario:** anyone re-deriving the Stage-6c predictor table from the Stage-6b definition
  (or vice versa) gets different feature values and a different gate, with no error and no note; and
  the "reuse the validated features" provenance claim is false for one of three.
- **Evidence:** measured against `baseline_pr_auc` (n = 38): 6b's definition ρ = −0.370 (p = 0.022),
  6c's ρ = −0.331 (p = 0.043) — reproducing `_diag_stage6b_h3_check.md:23` and
  `_stage6c_gate.md:29` respectively, so both writeups are internally correct and only the shared
  name is wrong.
- **Self-refutation attempted:** (a) *Does it change a verdict?* No — both variants are marginally
  significant raw and both die under the base-rate partial (finding 1: −0.031 vs −0.008 on `pr_auc`),
  so the Stage-6c outcome is unaffected. That is why this is medium, not high. (b) *Is one of them a
  typo for the other?* No: `ctx_incidence_std` is a real emitted column
  (`dataset_v2/features_ctx_illum/*.parquet`), so both expressions are deliberate.
- **Fix:** rename to `between_tile_incidence_sd` and `mean_within_tile_incidence_sd`, and correct the
  "reuses the validated features" sentences in `PROMOTION_QUEUE.md`.

---

### probes-stage6-5 — The gate-comparison table's `Spearman(p_bad, baseline_pr_auc)` column reports the opposite sign for the winning row

- **Severity:** low
- **Liveness:** dead-closed
- **Confidence:** high (exactly reproduced: −0.116 vs the published +0.116)
- **Where:** `scripts/probes/_stage6c_gate.py:216` vs `:182` and `:241`; emitted at `:508-511`;
  published at `scripts/probes/_stage6c_gate.md:34-38` (tracked file)

```
_stage6c_gate.py:182  (logreg)  spearman_to_pr_auc=float(stats.spearmanr(p_bad,     y_cont)...)
_stage6c_gate.py:216  (ridge)   spearman_to_pr_auc=float(stats.spearmanr(pred_cont, y_cont)...)
_stage6c_gate.py:241  (rule)    spearman_to_pr_auc=float(stats.spearmanr(p_bad,     y_cont)...)
_stage6c_gate.py:508  lines.append("| model | ROC-AUC (binary) | Spearman(p_bad, baseline_pr_auc) |")
```

`p_bad` for the ridge gate is a *decreasing* logistic of `pred_cont` (`:207`), so
`Spearman(p_bad, y)` = −`Spearman(pred_cont, y)` exactly. The published table therefore reads
`logreg −0.120 | ridge **+0.116** | rule −0.300` under one header, where the correct value for ridge
is **−0.116** (verified from `cache/stage6c/gate_cv.parquet`). A reader comparing the three rows sees
the winning gate as the only one that ranks images the *right* way when in fact all three rank them
the same (weakly correct) way.

- **Failure scenario:** the ridge row is the one the record promotes; anyone using this column to
  sanity-check the gate's direction concludes the ridge gate is inverted relative to the other two, or
  that the other two are.
- **Self-refutation attempted:** the number is not quoted in any doc (`docs/modeling_results.md §14.3`
  quotes ROC-AUC only), so nothing downstream is wrong — hence `low`. But `_stage6c_gate.md` is a
  **tracked** file linked from `PROMOTION_QUEUE.md:183` and `docs/modeling_results.md:1584`.
- **Fix:** compute `spearman_to_pr_auc` from `p_bad` in all three constructors (one-line change at
  `:216`), or rename the ridge row's column.

---

### probes-stage6-6 — Stage 6a's single dev-PASS clears its bar by 0.13 of a standard error, and the fold-variance probe that exists to answer "is this noise?" is hardcoded to the *other* sweep

- **Severity:** low
- **Liveness:** dead-closed (re-tested at full-v2 LOIO and STRICT-FAILED, `DECISIONS.md:2509-2517`)
- **Confidence:** high (paired test computed from `models/_sweep_stage6a/20260531T004356Z/summary.parquet`)
- **Where:** `scripts/probes/_diag_stage6a_followup_compare.py:89-96` (verdict logic),
  `scripts/probes/_diag_stage6a_fold_variance.py:18` (hardcoded source),
  `scripts/probes/_sweep_stage6a.py:205-212`; consumers `PROMOTION_QUEUE.md:623-670`,
  `docs/modeling_results.md:1278-1335`, `ROADMAP.md:30` ("6a dev-PASS deferred")

The promoted cell is `nbr_s5 @ S=32`: Δ Spearman **+0.0534** against a **+0.05** bar — a margin of
0.0034. The folds are correctly paired (I verified the two arms use identical fold indices and have
identical NaN patterns), so a paired test is available and gives: sd of the per-fold delta 0.1179 over
20 folds → SEM **0.0264**, 95 % CI **[+0.0017, +0.1051]**, paired-t **p = 0.057**, **13 of 20** folds
win. It is 1 PASS among 6 (variant × scale) cells with no multiplicity adjustment, and the margin over
the bar is 0.13 SEM. `_diag_stage6a_fold_variance.py` is precisely the probe that would have shown
this — it reports per-fold Δ sd and win/loss counts — but line 18 pins it to
`models/_sweep_stage6a/20260530T213424Z/summary.parquet`, the *earlier* two-scheme run, and
`PROMOTION_QUEUE.md:662` cites it "for the default-variant run". The promoted cell's spread was never
characterised.

- **Failure scenario:** bounded. The dev-PASS was deferred, then re-tested at full-v2 LOIO where it
  STRICT-FAILED (`DECISIONS.md:2509-2517`: ρ +0.072 PASS, PR-AUC +0.0166 FAIL), so no live number
  depends on it. It is filed because the same probe pair is the template for future feature-promotion
  sweeps and because `leakage-1` independently confounds this arm — two reasons the same PASS should
  not be re-used as evidence.
- **Self-refutation attempted:** (a) *Is the Δ unpaired or over different fold sets (the R24
  mechanism)?* No — checked: `spearman_n = 20` in both arms and both arms drop the same 5 folds from
  `pr_auc` at S=32 / 3 at S=64 (the NaN folds are label-determined, not model-determined). So the
  R24 defect does **not** bite here; see "Verified clean". (b) *Is it just `leakage-1`?* No —
  `leakage-1` is about neighbour features crossing the quadrant cut; this is about the PASS margin
  being inside the fold-level noise, which would hold even with a clean split.
- **Fix:** parameterise `_diag_stage6a_fold_variance.py` on `--sweep-dir` and require the paired
  spread (or a paired test) beside any Δ-vs-bar verdict in `_diag_stage6a_followup_compare.py` and
  `_sweep_stage6a._build_result_md`.

---

## Refuted by my own check

- **"The Stage-6a Δ is a mean over different fold counts in the two arms" (the R24 pattern).**
  Checked `models/_sweep_stage6a/20260531T004356Z/summary.parquet`: all four schemes have
  `spearman_n = 20`, `n_real_folds = 20`, and identical NaN counts per scale (5 at S=32, 3 at S=64)
  because `pr_auc` is NaN exactly on the single-class folds, a property of the labels. The arms are
  paired. (Residual nit, not filed: `pr_auc_mean` at S=32 is a mean over 15 of 20 folds while the
  aggregate advertises `n_real_folds = 20` and emits no `pr_auc_n` — that *is* R24's fix, already
  filed.)
- **"Strategy B leaks the held-out image's identity."** It does not: `p_bad` is an out-of-fold LOIO
  prediction from CTX-source features that are computable at inference, and
  `pooled_global_with_strategy_b` keys on `fold_held_out_obs_id`, which is the right key. The mild
  transductive element (`s = pred_cont.std()` at `_stage6c_gate.py:206` is computed over all 38
  out-of-fold predictions, and `BAD_PR_AUC_THRESHOLD = 0.55` was set from the full-set mean) only
  rescales a monotone transform and cannot change the pooled ranking's *within-image* order; it is
  not worth a finding.
- **"`_sweep_stage6a`/`_sweep_stage6b`'s `meaningful_threshold` monkeypatch is neutralised."**
  Both use the `default-argument` patch that `modeling-heads-3` (R42) covers; already filed, not
  re-reported.
- **"`_diag_within_image_deltas.py` is a second, doc-cited implementation of the H5 within-vs-LOIO
  instrument with the same quadrant-vs-whole-image mis-pairing as R45."** True (it averages 4
  quadrant AUCs per image at `:66-68` and pairs that against a whole-image LOIO AUC at `:52-60`), and
  it produces the `docs/modeling_results.md:960-985` table whose "every CI brackets zero" is R45's
  target. But R45 already owns the defect and its magnitude; re-filing the second call site adds
  nothing but the pointer, which I record in the load-bearing map instead. Same for its use of
  `presence_auc` / `bc_ge_1` ROC-AUC — §9 is openly labelled a presence-AUC section, and R02/R25 own
  the surface.
- **`ESP_068483_2280` has `mean_ctx_incidence = 4.276°` while all 37 others are 40–62°**
  (`_diag_stage6b_h3_check.md:47`) — physically implausible and worth a look, but it is a
  `src/ctx_source_illumination.py` / SeamMap-join question owned by the `features` area
  (cf. `features-4`), and because every statistic built on it here is a Spearman it moves nothing.
  Noted, not filed.

## Verified clean

- **All arithmetic I could reproduce, reproduces.** Baseline pooled AP 0.6086 (37,315 tiles / 17,920
  positive / base rate 0.480), Strategy-B 0.6454 / 0.6649 / 0.6438 for logreg / ridge / rule, and all
  20 H3 grid correlations, match the published tables to the printed precision.
- **`_sweep_stage6b.py` reports its own verdict honestly.** Both halves of the pre-declared Stage-6b
  criterion (`:19-24`) failed and are reported as FAIL; the docstring correctly warns that the n=5 dev
  LOIO cannot support the mechanism check and defers it to full v2. The dev sweep is a clean
  apples-to-apples pair (same variant, target, params, fold definitions).
- **`_stage6c_gate.py`'s rank-invariance argument** (`:562-566`, "per-fold PR-AUC is rank-invariant
  within a single held-out image, so Strategies B/C only show up in a pooled ranking") is correct, and
  is the reason the pooled metric is the right place to look.
- **`_stage6c_gate.py`'s LOIO loops** (`:164-173`, `:197-202`, `:230-233`) fit the scaler and the model
  on the training folds only, including the per-fold median in `threshold_rule`. The `merge` guard at
  `:612` asserts no rows are dropped.
- **`_diag_stage6b_h3_check.py`'s per-image delta alignment** (`:155-159`) indexes the illum arm by
  `ObsId` before subtracting, so the per-image Δ table is correctly paired.
- **`_sweep_perimage_std.py`** declares its promotion criteria in advance (paired Wilcoxon on
  `meaningful_auc` at p < 0.05 AND pooled PR-AUC Δ ≥ −0.01, `:9-12`), uses the banked baseline of the
  same recipe identity, and the recorded verdict (NOT PROMOTED, `DECISIONS.md:2810-2853`) follows from
  the numbers it reports. Per-image standardisation of the held-out image using its own statistics is
  legitimate at inference and is not a leak.
- **Import order:** every probe in this area that touches LightGBM imports `src.modeling` before
  numpy/pandas (`_sweep_stage6a.py:41`, `_sweep_stage6b.py:49`, `_diag_stage6b_h3_check.py:32`,
  `_sweep_perimage_std.py:31`, `_diag_within_image_deltas.py:21`). Invariant 9 satisfied.
  `_stage6c_gate*.py` do not use torch.

## Load-bearing map

| probe | cited by | number it produced | verdict |
|---|---|---|---|
| `_diag_stage6b_h3_check.py` (235) | `PROMOTION_QUEUE.md:147,510-533`; `docs/modeling_results.md:1398-1423`; `README.md:70`; `ROADMAP.md:30`; `PLAN_ModelUsability.md:35`; `docs/modeling.md:185,395,421,490,558,590`; tracked `.md` | 4×5 Spearman grid; "H3 falsified, Stage 6e mechanism EMPIRICALLY VALIDATED"; per-image 6b deltas | ρ reproduce exactly; **conclusion confounded by per-image base rate** → `probes-stage6-1`; one bolded `p<0.05` is p=0.089 at n=26 |
| `_stage6c_gate.py` (661) | `docs/modeling_results.md:1495,1584` §14; `PROMOTION_QUEUE.md:172-186`; `ROADMAP.md:30`; tracked `.md`; writes `cache/stage6c/{predictor_table,gate_cv}.parquet` | pooled PR-AUC 0.6086→0.6649 (**+0.056**); gate ROC-AUC 0.503/0.533/0.606; Strategy-A τ and top-K sweeps | numbers reproduce; **target = prevalence** (`-2`), **no null on the +0.056** (`-3`), **sign error in the §3 table** (`-5`) |
| `_stage6c_gate_v2.py` (381) | `docs/modeling_results.md:1500,1585`; `PROMOTION_QUEUE.md:184`; tracked `.md` | "strict FAIL across 5 gates × 4 cutoffs"; "best pooled-B **+0.0414**" | FAIL verdict sound; **headline is an argmax over 20 combos spanning −0.044…+0.041** → `probes-stage6-3` |
| `_sweep_stage6b.py` (330) | `PROMOTION_QUEUE.md:482,492`; `PLAN_ModelUsability.md:161`; `docs/modeling_results.md:1378,1488` | the `models/_sweep_stage6b/20260531T020308Z/` sweep every 6b/6c number reads; dev Δ tables | clean — both pre-declared halves fail and are reported as FAIL |
| `_sweep_stage6a.py` (303) | `docs/modeling.md:639`; `PROMOTION_QUEUE.md:617`; `docs/modeling_results.md:1278,1292`; `DECISIONS.md:2525` | the 6-cell Δ grid, `models/_sweep_stage6a/{2 timestamps}/` | arithmetic correct, arms paired; PASS margin 0.13 SEM (`-6`); dev arm confounded by `leakage-1`; superseded by the LOIO STRICT FAIL at `DECISIONS.md:2509-2517` |
| `_diag_stage6a_followup_compare.py` (122) | `PROMOTION_QUEUE.md:625`; tracked `.md` | the "5×5 @ S=32 is the only clean pass" verdict | reproduces `aggregate.parquet` exactly; no spread or multiplicity reported → `probes-stage6-6` |
| `_diag_stage6a_fold_variance.py` (92) | `PROMOTION_QUEUE.md:662`; `docs/modeling_results.md:1330`; tracked `.md` | per-fold Δ sd + win/loss (Spearman Δ sd 0.2234, 12/8) | correct, but hardcoded to the **earlier** sweep so it never covers the promoted cell → `probes-stage6-6` |
| `_diag_within_image_deltas.py` (106) | `docs/modeling_results.md:967` §9.4; `DECISIONS.md:1335`; `PLAN_ModelImprovement.md:165`; `PLAN_NewDetections.md:514` | the v2 within-vs-LOIO Δ table ("every CI brackets zero") | second implementation of the instrument **R45**/`notebooks-4` refutes (quadrant-vs-whole-image pairing); metric is presence AUC / `bc_ge_1` ROC-AUC. Not re-filed |
| `_sweep_perimage_std.py` (160) | `DECISIONS.md:2815,2841`; artifacts `models/_sweep_perimage_std/` | the 4-method table and the NOT-PROMOTED verdict | clean (pre-declared paired Wilcoxon; verdict follows) |
| `_diag_within_image_smoke.py` (48) | nowhere | prints per-fold AUC for 4 folds | throwaway harness smoke test; nothing depends on it |
| `_inspect_stage6b_output.py` (36) | nowhere | column-population sanity print for one dev image | throwaway; nothing depends on it |

## Coverage note

**Read in full:** `_stage6c_gate.py` (661), `_stage6c_gate_v2.py` (381), `_diag_stage6b_h3_check.py`
(235), `_diag_stage6a_followup_compare.py` (122), `_diag_stage6a_fold_variance.py` (92),
`_diag_within_image_deltas.py` (106), `_inspect_stage6b_output.py` (36), plus the three tracked
writeups `_stage6c_gate.md`, `_stage6c_gate_v2.md`, `_diag_stage6b_h3_check.md` and
`_diag_stage6a_fold_variance.md` / `_diag_stage6a_followup_compare.md`. **Read in full:**
`_sweep_stage6a.py` (303). **Read the docstring, acceptance logic and result-md emitter, skimmed the
run plumbing:** `_sweep_stage6b.py` (330), `_sweep_perimage_std.py` (160). **Skimmed:**
`_diag_within_image_smoke.py` (48).

**Reproduced numerically** (read-only, conda env `geospatial`): the pooled Strategy-A/B PR-AUCs from
`models/_sweep_stage6b/20260531T020308Z/loio_nfold/8c7523615964f5cb/scale_S64/predictions.parquet` +
`cache/stage6c/gate_cv.parquet`; the permutation nulls (500 and 400 draws); the full H3 grid and its
base-rate partials from `dataset_v2/features_ctx_illum/*.parquet` +
`models/_sweep_stage6b/20260531T020308Z/summary.parquet`; the prevalence-oracle Strategy-A sweep; the
paired Stage-6a fold tests from `models/_sweep_stage6a/20260531T004356Z/summary.parquet`.

**Could not check:** (a) the Stage-6a **dev** cohort's own `dataset_v2_dev` packaged splits — present
on disk but I did not re-derive the quadrant assignment (`leakage-1` already measured the cross-cut
share); (b) whether the ridge/logreg gates would still beat the permutation null on a re-run — that
needs re-running `_stage6c_gate.py`, which is out of scope; (c) the physical validity of
`ctx_incidence_*` (the `ESP_068483_2280` = 4.3° outlier) — a `features`-area question about
`src/ctx_source_illumination.py`; (d) `models/_sweep_stage6{a,b}` and `cache/stage6c` are **not
git-tracked**, so my reproductions are against the working-tree artifacts, which match the published
tables but cannot be pinned to a commit; (e) I did not attempt to quantify how much of Stage 6a's
S=32 Δ is `leakage-1`'s boundary band (needs a rerun with `buffer_tiles >= 2`).
