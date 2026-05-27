# PLAN — Stage 5b: binary-classification reframing on the Stage 5 dataset

**Status:** **planned, not yet implemented (2026-05-26)**. Will update with the
shipped commit + DECISIONS.md entry once shipped. Three AskUserQuestion answers
already pinned (§10).

This plan extends [PLAN_Stage5](PLAN_Stage5.md) — Stage 5 packaged the
`fractional_area` regression target into `dataset/packaged/{scheme}/`; Stage 5b
adds a parallel **binary classification** path on the same packaged data, with
three configurable presence thresholds and a dedicated LightGBM classifier
variant. No dataset re-build is required; binarisation happens at fit/eval time.

**Why this matters** — Stage-5-shipped regression baselines (the Week 3 sweep
documented in [docs/modeling_results.md](docs/modeling_results.md)) showed that
the model's discriminating power is at the presence threshold, not in
regression magnitude. Across all 12 (variant, scale) configurations, presence
AUC is above 0.5 with binomial-test *p* = 0.0002, while the Spearman ρ signal
is small (mean +0.016) and the model effectively predicts a near-constant
near-zero value regardless of true abundance (modeling_results.md §3.1). The
hypothesis under test in Stage 5b is that **reframing the task as binary
classification will surface a usable signal that the regression formulation
buries**. If true, the binary classifier becomes the primary deliverable; if
false (even the simplified problem stays mediocre), this is the strongest
evidence yet that the binding constraint is data quantity, not loss design or
target framing.

**Scope** — Stage 5b is a *modeling* extension, not a data-packaging change.
The Stage 5 `dataset/packaged/` layout is unchanged. What changes:

- a new `LightGBMClassification` model variant in [src/modeling/gbm.py](src/modeling/gbm.py),
- a new `BinaryTarget` abstraction defining the three thresholds (§3),
- a new fan-out script `scripts/sweep_binary.py` mirroring
  [scripts/sweep.py](scripts/sweep.py),
- new evaluation metrics appropriate for classification (AUC, Brier,
  calibration, lift-at-top-k) integrated into the existing
  [src/modeling/evaluate.py](src/modeling/evaluate.py) LOIO harness,
- a new section in [notebooks/10_modeling_qa.ipynb](notebooks/10_modeling_qa.ipynb)
  rendering the binary-sweep diagnostics alongside the existing regression ones,
- a "Binary reframing" section in [docs/modeling_results.md](docs/modeling_results.md)
  that either updates the verdict (binary clearly working) or strengthens it
  (even simplified problem stays mediocre).

---

## 1. Inputs

| Input | Path | Used for |
|---|---|---|
| Stage 5 packaged folds | `dataset/packaged/loio_9fold/X_{train,test}_fold{k}.parquet`, `y_{train,test}_fold{k}.parquet`, `groups_{train,test}_fold{k}.npy` | Feature matrices + continuous-target labels; same data as the regression sweep consumes |
| Stage 5 metadata | `dataset/packaged/loio_9fold/metadata.json` | ObsId↔int code map, per-fold tile counts, held-out ObsIds, provenance |
| `binary_by_count` column | embedded in `y_*.parquet` (Stage 5) | One of the three threshold sources (the `boulder_count ≥ 1` target); already in the dataset |
| `fractional_area` column | embedded in `y_*.parquet` (Stage 5) | Source for the other two thresholds (`fa > 1e-3` and `fa > 1e-2`), derived at fit/eval time |

No new caches, no Stage 4 / 4b / 5 re-runs.

## 2. Outputs

```
models/lightgbm_classification/<config_hash>/scale_S{n}_t{threshold_id}/
    classifier.txt              # LightGBM booster
    predictions.parquet         # per-tile (key cols + y_true_binary + y_pred_prob)
    metrics.json                # per-fold + aggregate metrics (AUC, Brier, calibration, lift_at_k)
    snapshot.json               # params + threshold spec + scheme + scale + config_hash
    fold_<obs_id>/classifier.txt
models/_sweep_binary/<ts>/
    summary.parquet             # one row per (threshold_id, scale_idx, fold)
    aggregate.parquet           # one row per (threshold_id, scale_idx)
```

`threshold_id` is one of `"bc_ge_1"`, `"fa_gt_1e-3"`, `"fa_gt_1e-2"` (§3).
Path layout mirrors the existing regression artifacts so the notebook + future
sweeps can navigate both with the same `*/scale_S{n}_*` glob pattern (cf. the
[`config_hash`-per-scale gotcha](MEMORY-cross-ref `sweep_vs_train_gbm_artifacts.md`)).

## 3. Binary-target specification

Three thresholds, fixed by AskUserQuestion 2026-05-26:

| `threshold_id` | Definition | Positives at S=64 | Scientific meaning |
|---|---|---:|---|
| `bc_ge_1`     | `boulder_count >= 1`     | 28.0 % | "Any boulder visible" — same target the existing two-stage hurdle uses for its presence head |
| `fa_gt_1e-3`  | `fractional_area > 1e-3` | 3.4 %  | "Some boulder coverage" — excludes single-boulder tiles |
| `fa_gt_1e-2`  | `fractional_area > 1e-2` | 0.17 % | "Boulder-rich tile" — per-tile analogue of the manifest-level `Boulder rich` label |

Encoded in a single dataclass module:

```python
# src/modeling/binary_target.py
@dataclass(frozen=True)
class BinaryTarget:
    id: str             # e.g. "bc_ge_1"
    source_col: str     # "boulder_count" or "fractional_area"
    threshold: float
    comparison: str     # ">=" or ">"
    label: str          # human-readable, used in plots/tables

    def binarize(self, y_df: pd.DataFrame) -> np.ndarray:
        col = y_df[self.source_col].to_numpy()
        op = operator.ge if self.comparison == ">=" else operator.gt
        return op(col, self.threshold).astype(np.int8)

BINARY_TARGETS: list[BinaryTarget] = [
    BinaryTarget("bc_ge_1",    "boulder_count",   1.0,  ">=", "boulder_count ≥ 1"),
    BinaryTarget("fa_gt_1e-3", "fractional_area", 1e-3, ">",  "fractional_area > 1e-3"),
    BinaryTarget("fa_gt_1e-2", "fractional_area", 1e-2, ">",  "fractional_area > 1e-2"),
]
```

Single source of truth — notebook, tests, sweep script, classifier all import
from here. Adding a fourth threshold is a one-line append.

## 4. Model: `LightGBMClassification`

New class in [src/modeling/gbm.py](src/modeling/gbm.py) implementing the
existing `Model` Protocol (same `fit / predict / save / load / model_hash`
interface as the three regression variants). Wraps a single `lgb.Booster`
trained with `objective="binary"`, `metric=["auc", "binary_logloss"]`.
`predict()` returns probabilities in [0, 1] (not 0/1 hard labels — the
operating point is chosen downstream from the calibration plot).

**Class-imbalance handling: `scale_pos_weight = neg_count / pos_count` per
fold, computed automatically.** Without this, the `fa_gt_1e-2` model (0.17 %
positives at S=64) collapses to predicting "no" everywhere, which is exactly
the constant-predictor failure mode the regression sweep exhibited. The
auto-weighting matches LightGBM's documented recommendation for imbalanced
binary tasks and is the textbook fix; it does not change the AUC ranking but
it makes the predicted probabilities calibratable. Documented as a
non-overridable default; expose `scale_pos_weight=None` as a debug-only
override.

Inner-validation early-stopping uses the same per-fold rotation as the
regression variants (PLAN_modeling.md §4: never use the test fold as
`eval_set`). The classifier's early-stopping metric is `binary_logloss`, not
`auc`, because AUC is non-decomposable and produces noisier early-stopping
behaviour on small inner-validation sets.

Registered as variant `"lightgbm_classification"` in `VARIANT_CONSTRUCTORS`.
Booster artifacts save to `classifier.txt` (mirroring the single-booster
regression variants).

## 5. Evaluation: classification mode in the LOIO harness

[src/modeling/evaluate.py](src/modeling/evaluate.py) currently computes
regression metrics (Spearman, log1p-RMSE, per-bin RMSE, presence AUC). Stage
5b adds a `task: Literal["regression", "classification"] = "regression"`
parameter to `run_loio`. When `task="classification"`:

- skip Spearman and the per-bin RMSE table (undefined for binary);
- report **AUC** (already computed, becomes primary metric),
- **Brier score** = mean squared error between predicted probability and
  binary truth — the canonical proper scoring rule for probabilistic
  classification,
- **Per-decile calibration** — bin tiles by predicted probability into
  deciles and report (mean predicted, mean true) per decile; a perfectly
  calibrated model has these equal,
- **Lift at top-k** — sort tiles by predicted probability descending, take
  top-k where k = number of true positives, report fraction of true positives
  captured. Base-rate-normalised: lift = (positives in top-k) / k divided by
  the global positive rate. A random classifier has lift = 1; a perfect
  classifier has lift = 1 / positive_rate.

Aggregate (`aggregate_fold_metrics`) handles classification: mean ± std over
folds for AUC and Brier; calibration concatenates across folds before
re-binning; lift is averaged.

The existing empty-truth fold (`ESP_065711_1545`) handling carries over: it
is a specificity-only fold (truth is constant zero — no positives, no AUC).
Reported separately as `n_specificity_folds` with the false-positive count
at a chosen probability threshold (default 0.5).

## 6. Sweep script

`scripts/sweep_binary.py` (new, ~120 LOC) mirrors [sweep.py](scripts/sweep.py):

```
Usage:
    python scripts/sweep_binary.py                  # all 3 thresholds, all 4 scales = 12 runs
    python scripts/sweep_binary.py --thresholds bc_ge_1 --scales 0 1
```

Per (threshold, scale):
1. binarise y at fit time using `BINARY_TARGET[id].binarize(y_train_df)`,
2. compute per-fold `scale_pos_weight` from the binarised train labels,
3. fit, eval, write artifacts via `write_run_artifacts()`,
4. second-pass per-fold booster persistence (mirrors `train_gbm.py` /
   `sweep.py` regression pattern — same idempotent `config_hash` mechanism).

Aggregates to `models/_sweep_binary/<ts>/{summary,aggregate}.parquet`.

A companion `scripts/train_binary.py` provides single-(threshold, scale)
invocation for debugging.

## 7. Notebook 10 extension

New section appended to `notebooks/10_modeling_qa.ipynb` (built by
`notebooks/_build_10.py`):

- **Binary sweep AUC bar chart** — mirrors the regression Spearman bar
  chart. Three coloured bars per scale (one per threshold), `mean ± std`
  across the 8 real-truth folds.
- **Per-fold AUC by held-out BoulderLabel** — mirrors the existing regression
  per-fold-by-label scatter. Quick visual on whether the binary signal is
  class-conditional.
- **Calibration plot for the best (threshold, scale)** — one panel per
  threshold, deciles on the x-axis, mean predicted vs mean true. Identity
  diagonal annotated.
- **Lift at top-k vs regression AUC table** — side-by-side comparison
  of "what does the binary classifier give us at threshold T" vs "what
  did the regression model give us on the same threshold."
- **Per-feature importance for the best classifier** — does the binary
  classifier key on the same features (`shadow_fraction` dominance) as the
  regression GBMs?

Existing regression section is unchanged. The two sections coexist so the
"is this method working?" comparison is one-notebook.

## 8. Tests

New `tests/test_modeling_binary.py` (~120 LOC):

- `test_binarize_boulder_count_ge_1_on_synthetic_y` — three rows, count=[0,1,5], expect [0,1,1].
- `test_binarize_fractional_area_gt_1e_3_on_synthetic_y`.
- `test_binary_target_registry_has_three_thresholds` — guard against accidental dedupe.
- `test_lightgbm_classification_fit_predict_returns_probabilities_in_unit_interval`.
- `test_lightgbm_classification_save_load_roundtrip`.
- `test_lightgbm_classification_uses_scale_pos_weight_on_imbalanced_synthetic` —
  fit on 99% negative synthetic data and assert the model predicts some
  positives (without the weight, it predicts all zeros).
- `test_run_loio_classification_mode_returns_auc_brier_calibration_lift` —
  end-to-end on a small synthetic fold.
- `test_lift_at_top_k_on_perfect_classifier_equals_inverse_base_rate`.
- `test_aggregate_fold_metrics_handles_classification_task`.
- `test_sweep_binary_smoke_one_threshold_one_scale` — slow-marked integration
  test against the real packaged dataset.

Target: +10–12 tests; pytest 159 → ~170.

## 9. Sequencing relative to Stage 5

Stage 5 is shipped (commit `aa6cd74`). Stage 5b is purely additive:
no Stage 5 code is modified, no `dataset/packaged/` files are touched. Stage
5b can be reverted in isolation if the binary reframing turns out
uninformative.

## 10. Decisions already pinned (AskUserQuestion 2026-05-26)

| Question | Decision | Rationale |
|---|---|---|
| Which binary target? | **All three** (`bc_ge_1` + `fa_gt_1e-3` + `fa_gt_1e-2`) | Most informative about where the model's discriminating power actually lives. Sweep cost is small (12 runs vs the regression sweep's 12). |
| Refactor scope | **Add a new `lightgbm_classification` variant** | Cleanest separation. Goes through the existing LOIO harness; runs side-by-side with the regression variants under one notebook. Smaller blast radius than promoting the two-stage hurdle's presence head to a first-class result. |
| Feature-name figure fix | **Defer** (leave as `Column_X` for now) | The modeling_results.md text tables already have real names; figure fix can ride along with a future model-artifact regeneration. |

## 11. Open questions to resolve at implementation time

1. **Should the existing two-stage hurdle's presence head be evaluated alongside
   the new dedicated classifier on `bc_ge_1`?** The two-stage already trains a
   binary presence head on `fractional_area > 0` (≈ `boulder_count ≥ 1` per the
   2026-05-27 probe). Comparing the dedicated classifier to the hurdle-first-stage
   on the same target answers "is dedicated training better than embedded?"
   Cheap addition — read the existing `fold_*/presence.txt` artifacts at
   notebook time, no re-train.
2. **Calibration: report per-decile means or expected calibration error (ECE)?**
   Per-decile is more visual + diagnostic; ECE is a scalar suitable for
   aggregate tables. Recommendation: both, ECE in the aggregate table and
   per-decile in the calibration plot.
3. **Operating-point selection for "use the classifier".** AUC is
   threshold-free, but any practical deployment needs a decision threshold.
   Three candidates: Youden's J (max TPR − FPR), max-F1, equal-error-rate.
   Defer to a follow-up — pick after seeing the calibration shape.
4. **`fa_gt_1e-2` at the finest scale (S=8) is essentially infeasible.** Only
   0.015 % of S=8 tiles are positive — roughly 73 positives in the entire
   648,554-tile dataset, distributed across 9 images. A LOIO fold's training
   set might have as few as 50 positives; the test set as few as 5. Plan
   anyway, but expect AUC noise and decide at write-up time whether to drop
   the `fa_gt_1e-2 × S=8` cell from the headline table.
5. **Does the notebook get a binary-vs-regression head-to-head?** The
   regression GBM's `presence AUC` column (already computed) provides the
   direct comparison: classifier `bc_ge_1` AUC vs regression two-stage's
   presence AUC, side by side at each scale. Yes — include in the notebook
   extension §7.

## 12. Out of scope (carry forward)

- **CNN binary baseline.** A CNN with `objective="binary"` (log-loss) on the
  patch stacks would test whether the patch CNN's failure
  (modeling_results.md §3.3) is fundamentally a regression-loss problem or a
  patch-signal problem. Worth doing, but not in Stage 5b — needs a
  class-balanced PyTorch loader and is its own scoped effort.
- **Within-image cross-validation.** modeling_results.md §5 identifies this
  as the cheapest decisive next experiment for the regression problem. It
  applies equally to binary classification but is a separate methodological
  change (new split scheme + harness invariant relaxation) and deserves its
  own plan.
- **Threshold calibration via the data.** All three thresholds in §3 are
  chosen by reasoning about the truth distribution; data-driven thresholds
  (e.g. the value that maximises the Youden statistic for an external
  reference like THEMIS) would be a follow-up.
- **THEMIS validation.** CLAUDE.md §10 future work. Binary classifier
  outputs may be easier to validate against the THEMIS rock-abundance map
  than fractional-area predictions; flag for Week 4+.

## 13. Time estimate

| Task | Est. |
|---|---|
| `binary_target.py` + tests | 15 min |
| `LightGBMClassification` + tests | 20 min |
| `evaluate.py` classification mode + tests | 25 min |
| `sweep_binary.py` + `train_binary.py` | 20 min |
| Run the 12-cell sweep | 5 min |
| Notebook 10 extension + execution | 25 min |
| docs/modeling_results.md update | 20 min |
| **Total** | **~130 min** |

Implementation order: `binary_target.py` → `LightGBMClassification` →
`evaluate.py` classification mode → `train_binary.py` (single-cell smoke) →
`sweep_binary.py` (full 12-cell) → notebook → docs update. Tests written
alongside each component.
