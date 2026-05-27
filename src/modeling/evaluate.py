"""LOIO cross-validation runner + zero-inflation-aware metric helpers.

Per PLAN_modeling.md §5:
  - Primary metric: Spearman rho between predicted and true `fractional_area`,
    reported as mean +/- std across the 9 LOIO folds.
  - Per-abundance-bin RMSE table is the secondary diagnostic (the CLAUDE.md
    "not a single RMSE dominated by near-zero tiles" requirement).
  - The empty-truth image ESP_065711_1545 is a specificity stress test: Spearman
    is undefined when truth is constant, so its fold is tagged `specificity_only`
    and reports `n_pred_above_threshold` and `mean_pred` instead.
  - Per-fold predictions are cached to a parquet so metric re-aggregation never
    requires re-training.

This module is stateless: `run_loio` is a pure function of `(loader_factory,
model_factory)`. It does not pin a target column -- the caller passes
`target_col` so the same runner serves both `fractional_area` regression and
binary classification (the latter just uses a model whose `predict` returns
probabilities, and bin-RMSE collapses to log-loss / AUC at the analysis stage).
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.modeling.base import Model
from src.modeling.loaders import Fold, iter_loio_folds

# Per-abundance-bin RMSE table (PLAN_modeling.md §5 starting cuts).
# The "zero" bin is a literal point (y_true == 0). The N positive bins are half-open
# (edge_i, edge_{i+1}] intervals over the N+1 positive edges.
ABUNDANCE_BIN_LABELS: tuple[str, ...] = (
    "zero",
    "0_to_1e-4",
    "1e-4_to_1e-3",
    "1e-3_to_1e-2",
    "1e-2_to_max",
)
POSITIVE_BIN_EDGES: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1.0)
# ABUNDANCE_BIN_EDGES kept as a public alias for backward compatibility / readability.
ABUNDANCE_BIN_EDGES = POSITIVE_BIN_EDGES

# Stage-4 special-case ObsId: the empty-truth image. Used only to tag specificity-only folds.
EMPTY_TRUTH_OBS_ID = "ESP_065711_1545"


# ============================================================================
# Per-fold metrics
# ============================================================================


def spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rho with NaN-safe handling for degenerate (constant) inputs.

    Returns NaN if either side has zero variance (no rank ordering possible).
    """
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_log1p(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE on log1p-stabilised target (PLAN_modeling.md §5 secondary metric)."""
    if y_true.size == 0:
        return float("nan")
    yt = np.log1p(np.clip(y_true, 0.0, None))
    yp = np.log1p(np.clip(y_pred, 0.0, None))
    return rmse(yt, yp)


def per_bin_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_edges: tuple[float, ...] = POSITIVE_BIN_EDGES,
    bin_labels: tuple[str, ...] = ABUNDANCE_BIN_LABELS,
) -> pd.DataFrame:
    """Per-abundance-bin RMSE table. Bins are by *true* abundance.

    Bin layout (N positive bins + 1 zero bin = N+1 total labels; N+1 positive_edges):
      * "zero"           y_true == 0
      * label[i] (i>=1)  positive_edges[i-1] < y_true <= positive_edges[i]
    Empty bins return NaN RMSE with n=0.
    """
    n_bins = len(bin_labels)
    n_pos_bins = n_bins - 1
    assert len(positive_edges) == n_pos_bins + 1, (
        f"need {n_pos_bins + 1} positive_edges for {n_bins} labels, got {len(positive_edges)}"
    )

    rows = []
    for i, label in enumerate(bin_labels):
        if i == 0:
            mask = y_true == 0.0
            lo, hi = 0.0, 0.0
        else:
            lo, hi = positive_edges[i - 1], positive_edges[i]
            mask = (y_true > lo) & (y_true <= hi)
        n = int(mask.sum())
        rmse_bin = rmse(y_true[mask], y_pred[mask]) if n > 0 else float("nan")
        mean_true = float(y_true[mask].mean()) if n > 0 else float("nan")
        mean_pred = float(y_pred[mask].mean()) if n > 0 else float("nan")
        rows.append({
            "bin": label,
            "lo": float(lo),
            "hi": float(hi),
            "n_tiles": n,
            "rmse": rmse_bin,
            "mean_true": mean_true,
            "mean_pred": mean_pred,
        })
    return pd.DataFrame(rows)


def presence_auc(y_true_positive: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC of binary presence detection (`y_true > 0`) vs the regression output.

    Uses scipy.stats.mannwhitneyu / U-statistic equivalence to avoid the sklearn dep
    just for this. Returns NaN when one class is missing.
    """
    pos = y_pred[y_true_positive]
    neg = y_pred[~y_true_positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U is equivalent to AUC via U / (n_pos * n_neg).
    u, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u / (pos.size * neg.size))


# ============================================================================
# Binary-classification metrics (Stage 5b)
# ============================================================================


def brier_score(y_true_binary: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Brier score = mean squared error between predicted probability and binary truth.

    Lower is better. The canonical proper scoring rule for probabilistic
    classification (PLAN_Stage5b.md §5).
    """
    if y_true_binary.size == 0:
        return float("nan")
    return float(np.mean((y_pred_prob - y_true_binary.astype(np.float64)) ** 2))


def expected_calibration_error(
    y_true_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """ECE = sum_b (n_b / N) * |mean_pred_b - mean_true_b|.

    A scalar summary of how far the model is from being perfectly calibrated.
    0 = perfect calibration, larger = more miscalibrated. PLAN_Stage5b.md §11 q2.
    """
    if y_true_binary.size == 0:
        return float("nan")
    # Equal-width bins over the predicted-probability domain [0, 1].
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred_prob, bin_edges[1:-1]), 0, n_bins - 1)
    total_err = 0.0
    n = y_true_binary.size
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        mean_pred = float(y_pred_prob[mask].mean())
        mean_true = float(y_true_binary[mask].mean())
        total_err += (mask.sum() / n) * abs(mean_pred - mean_true)
    return float(total_err)


def calibration_deciles(
    y_true_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict]:
    """Per-decile calibration table -- one row per equal-width predicted-prob bin.

    Each row: {bin_idx, lo, hi, n, mean_pred, mean_true}. A perfectly calibrated
    model has mean_pred == mean_true in every bin. Empty bins return n=0 and NaN.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred_prob, bin_edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n > 0:
            mean_pred = float(y_pred_prob[mask].mean())
            mean_true = float(y_true_binary[mask].mean())
        else:
            mean_pred = float("nan")
            mean_true = float("nan")
        rows.append({
            "bin_idx": b,
            "lo": float(bin_edges[b]),
            "hi": float(bin_edges[b + 1]),
            "n": n,
            "mean_pred": mean_pred,
            "mean_true": mean_true,
        })
    return rows


def lift_at_top_k(y_true_binary: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Base-rate-normalised precision at k, where k = number of true positives.

    Take the k tiles with highest predicted probability (k = sum(y_true)).
    Precision@k = positives_in_top_k / k. Lift = precision@k / base_rate.

    A random classifier has lift == 1; a perfect classifier has lift ==
    1 / base_rate (= n / n_pos). Returns NaN when there are no positives or
    no negatives.

    PLAN_Stage5b.md §5.
    """
    n = y_true_binary.size
    n_pos = int(y_true_binary.sum())
    if n_pos == 0 or n_pos == n:
        return float("nan")
    base_rate = n_pos / n
    # Argsort descending by predicted probability; take the top n_pos indices.
    top_k_idx = np.argpartition(-y_pred_prob, n_pos - 1)[:n_pos]
    precision_at_k = float(y_true_binary[top_k_idx].mean())
    return float(precision_at_k / base_rate)


def per_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    held_out_obs_ids: list[str],
) -> dict:
    """All per-fold metrics in one dict, with the specificity-only flag set when due."""
    is_empty_truth = (len(held_out_obs_ids) == 1 and held_out_obs_ids[0] == EMPTY_TRUTH_OBS_ID)

    out: dict = {
        "n_tiles": int(y_true.size),
        "held_out_obs_ids": list(held_out_obs_ids),
        "is_specificity_only": bool(is_empty_truth or np.unique(y_true).size < 2),
        "mean_true": float(y_true.mean()) if y_true.size else float("nan"),
        "mean_pred": float(y_pred.mean()) if y_pred.size else float("nan"),
    }

    if not out["is_specificity_only"]:
        out["spearman_rho"] = spearman_safe(y_true, y_pred)
        out["rmse_raw"] = rmse(y_true, y_pred)
        out["rmse_log1p"] = rmse_log1p(y_true, y_pred)
        out["presence_auc"] = presence_auc(y_true > 0, y_pred)
    else:
        # Spearman / AUC undefined; report what we can.
        out["spearman_rho"] = float("nan")
        out["rmse_raw"] = rmse(y_true, y_pred)
        out["rmse_log1p"] = rmse_log1p(y_true, y_pred)
        out["presence_auc"] = float("nan")
        out["pred_above_1e-4"] = float((y_pred > 1e-4).mean()) if y_pred.size else float("nan")

    # Per-abundance-bin RMSE: always emitted (the zero bin alone is meaningful even on
    # the empty-truth fold).
    out["per_bin_rmse"] = per_bin_rmse(y_true, y_pred).to_dict(orient="records")
    return out


def aggregate_fold_metrics(per_fold: list[dict]) -> dict:
    """Mean +/- std of fold-level metrics, ignoring specificity-only folds for Spearman / AUC."""
    real = [f for f in per_fold if not f["is_specificity_only"]]
    spec = [f for f in per_fold if f["is_specificity_only"]]

    def mean_std(key: str, source: list[dict]) -> tuple[float, float, int]:
        vals = [f[key] for f in source if not np.isnan(f[key])]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)

    spearman_mean, spearman_std, n_spearman = mean_std("spearman_rho", real)
    rmse_log1p_mean, rmse_log1p_std, _ = mean_std("rmse_log1p", real)
    rmse_raw_mean, rmse_raw_std, _ = mean_std("rmse_raw", per_fold)  # raw RMSE meaningful on all folds
    auc_mean, auc_std, _ = mean_std("presence_auc", real)

    return {
        "n_real_folds": len(real),
        "n_specificity_folds": len(spec),
        "spearman_rho_mean": spearman_mean,
        "spearman_rho_std": spearman_std,
        "spearman_n": n_spearman,
        "rmse_log1p_mean": rmse_log1p_mean,
        "rmse_log1p_std": rmse_log1p_std,
        "rmse_raw_mean": rmse_raw_mean,
        "rmse_raw_std": rmse_raw_std,
        "presence_auc_mean": auc_mean,
        "presence_auc_std": auc_std,
    }


def per_fold_metrics_classification(
    y_true_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    *,
    held_out_obs_ids: list[str],
    decision_threshold: float = 0.5,
) -> dict:
    """Per-fold classification metrics. `y_true_binary` is int8 0/1, `y_pred_prob` in [0, 1].

    A fold whose y_true is constant (e.g. the empty-truth ESP_065711_1545 fold)
    is tagged `is_specificity_only`: AUC / Brier / lift are undefined on a
    single-class fold, but the false-positive count at `decision_threshold`
    is the meaningful diagnostic.
    """
    is_empty_truth = (len(held_out_obs_ids) == 1 and held_out_obs_ids[0] == EMPTY_TRUTH_OBS_ID)
    n_pos = int(y_true_binary.sum())
    n_neg = int(y_true_binary.size - n_pos)
    is_spec = bool(is_empty_truth or n_pos == 0 or n_neg == 0)

    out: dict = {
        "n_tiles": int(y_true_binary.size),
        "held_out_obs_ids": list(held_out_obs_ids),
        "is_specificity_only": is_spec,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "base_rate": float(n_pos / y_true_binary.size) if y_true_binary.size else float("nan"),
        "mean_pred_prob": float(y_pred_prob.mean()) if y_pred_prob.size else float("nan"),
    }

    if not is_spec:
        # AUC via Mann-Whitney U (same machinery as presence_auc).
        pos_scores = y_pred_prob[y_true_binary == 1]
        neg_scores = y_pred_prob[y_true_binary == 0]
        u, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        out["auc"] = float(u / (pos_scores.size * neg_scores.size))
        out["brier"] = brier_score(y_true_binary, y_pred_prob)
        out["ece"] = expected_calibration_error(y_true_binary, y_pred_prob)
        out["lift_at_top_k"] = lift_at_top_k(y_true_binary, y_pred_prob)
    else:
        out["auc"] = float("nan")
        out["brier"] = brier_score(y_true_binary, y_pred_prob)  # MSE still defined
        out["ece"] = float("nan")
        out["lift_at_top_k"] = float("nan")
        # On the empty-truth fold, the FP rate at the decision threshold is the
        # diagnostic that maps to "does the classifier hallucinate?".
        out["false_positive_rate_at_threshold"] = (
            float((y_pred_prob >= decision_threshold).mean())
            if y_pred_prob.size else float("nan")
        )

    out["calibration_deciles"] = calibration_deciles(y_true_binary, y_pred_prob)
    return out


def aggregate_fold_metrics_classification(per_fold: list[dict]) -> dict:
    """Mean +/- std of classification fold metrics, ignoring specificity-only folds for AUC."""
    real = [f for f in per_fold if not f["is_specificity_only"]]
    spec = [f for f in per_fold if f["is_specificity_only"]]

    def mean_std(key: str, source: list[dict]) -> tuple[float, float, int]:
        vals = [f[key] for f in source if not np.isnan(f[key])]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)

    auc_mean, auc_std, n_auc = mean_std("auc", real)
    brier_mean, brier_std, _ = mean_std("brier", per_fold)  # Brier defined on all folds (even all-zero truth)
    ece_mean, ece_std, _ = mean_std("ece", real)
    lift_mean, lift_std, _ = mean_std("lift_at_top_k", real)

    return {
        "n_real_folds": len(real),
        "n_specificity_folds": len(spec),
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_n": n_auc,
        "brier_mean": brier_mean,
        "brier_std": brier_std,
        "ece_mean": ece_mean,
        "ece_std": ece_std,
        "lift_at_top_k_mean": lift_mean,
        "lift_at_top_k_std": lift_std,
    }


# ============================================================================
# LOIO runner
# ============================================================================


@dataclass(frozen=True)
class RunResult:
    """Container returned by `run_loio` -- the full per-fold predictions + metrics."""

    predictions: pd.DataFrame  # one row per test tile across all folds
    per_fold_metrics: list[dict]
    aggregate: dict
    snapshot: dict  # config + model + protocol provenance


ModelFactory = Callable[[], Model]
FoldIterator = Callable[[], Iterable[Fold]]


def _default_fold_iter(scheme: str, scale_idx: int | None) -> FoldIterator:
    def _it() -> Iterable[Fold]:
        return iter_loio_folds(scheme, scale_idx=scale_idx)

    return _it


def run_loio(
    model_factory: ModelFactory,
    *,
    target_col: str = "fractional_area",
    binarize: Callable[[pd.DataFrame], np.ndarray] | None = None,
    task: Literal["regression", "classification"] = "regression",
    scheme: str = "loio_9fold",
    scale_idx: int | None = None,
    fold_iter: FoldIterator | None = None,
    snapshot: dict | None = None,
    verbose: bool = True,
) -> RunResult:
    """Run LOIO CV. Returns predictions + per-fold + aggregated metrics.

    The caller supplies `model_factory` -- a zero-arg callable that produces a fresh
    `Model` per fold. The harness fits each on `(X_train, y_train[target_col])`,
    predicts on `X_test`, and aggregates predictions into one dataframe.

    `scale_idx=None` uses all scales concatenated; pass an int to train per-scale
    models (PLAN_modeling.md §6 Option A).

    Stage 5b classification mode: pass `task="classification"` plus a `binarize`
    callable mapping a y dataframe to a 0/1 int8 array (typically
    `src.modeling.binary_target.BinaryTarget.binarize`). The harness then
    bypasses `target_col`, feeds the binarised y to `model.fit`, treats
    `model.predict` output as probabilities in [0, 1], and computes
    classification metrics (AUC, Brier, ECE, lift_at_top_k) instead of the
    regression set.
    """
    if task == "classification" and binarize is None:
        raise ValueError("task='classification' requires a binarize callable")
    fold_iter = fold_iter if fold_iter is not None else _default_fold_iter(scheme, scale_idx)

    pred_rows: list[pd.DataFrame] = []
    per_fold: list[dict] = []

    for fold in fold_iter():
        if binarize is not None:
            y_train_full = binarize(fold.y_train).astype(np.int8, copy=False)
            y_test = binarize(fold.y_test).astype(np.int8, copy=False)
        else:
            y_train_full = fold.y_train[target_col].to_numpy(dtype=np.float64, copy=False)
            y_test = fold.y_test[target_col].to_numpy(dtype=np.float64, copy=False)

        # PLAN_modeling.md §4: "use one of the 8 training images as the early-stopping
        # monitor, rotate which one across folds." Picking the training image whose code
        # is (held_out_code + 1) mod N_total gives a deterministic, fold-dependent
        # rotation that never uses the held-out fold as a validation set.
        train_codes = fold.groups_train
        unique_train = np.unique(train_codes)
        # Held-out code is whatever's in groups_test (one element for LOIO).
        held_codes = set(np.unique(fold.groups_test).tolist())
        # Rotation: the unique training code at position (fold_idx % n).
        inner_val_code = int(unique_train[fold.fold_idx % unique_train.size])
        # Defensive: if it ever overlapped with the held-out set, the splitter is broken.
        assert inner_val_code not in held_codes, "inner-val code collided with held-out"
        inner_val_mask = train_codes == inner_val_code
        inner_train_mask = ~inner_val_mask
        X_inner_train = fold.X_train[inner_train_mask]
        y_inner_train = y_train_full[inner_train_mask]
        X_inner_val = fold.X_train[inner_val_mask]
        y_inner_val = y_train_full[inner_val_mask]
        train_inner_groups = train_codes[inner_train_mask]

        model = model_factory()
        # eval_set is the rotated inner-validation image, not the held-out test fold.
        model.fit(
            X_inner_train, y_inner_train,
            groups=train_inner_groups,
            eval_set=(X_inner_val, y_inner_val),
        )
        y_pred = np.asarray(model.predict(fold.X_test), dtype=np.float64)
        y_presence = model.predict_presence_prob(fold.X_test)

        # Per-fold metric pack
        if task == "classification":
            m = per_fold_metrics_classification(
                y_test.astype(np.int8, copy=False), y_pred,
                held_out_obs_ids=fold.held_out_obs_ids,
            )
        else:
            m = per_fold_metrics(y_test, y_pred, held_out_obs_ids=fold.held_out_obs_ids)
        m["fold_idx"] = fold.fold_idx
        m["scale_idx"] = fold.scale_idx if fold.scale_idx is not None else -1
        m["model_name"] = getattr(model, "name", type(model).__name__)
        m["model_hash"] = model.model_hash()
        per_fold.append(m)

        # Append to predictions dataframe
        block = fold.keys_test.copy()
        block["fold_held_out_obs_id"] = fold.held_out_obs_ids[0] if fold.held_out_obs_ids else ""
        block["fold_idx"] = fold.fold_idx
        block["y_true"] = y_test
        block["y_pred"] = y_pred
        if y_presence is not None:
            block["y_pred_presence_prob"] = np.asarray(y_presence, dtype=np.float64)
        else:
            block["y_pred_presence_prob"] = np.nan
        block["model_hash"] = model.model_hash()
        pred_rows.append(block)

        if verbose:
            if task == "classification":
                tag = "spec" if m["is_specificity_only"] else f"auc={m['auc']:+.4f}"
                print(
                    f"  fold {m['fold_idx']}: {fold.held_out_obs_ids[0]:>20s}  "
                    f"n_test={m['n_tiles']:>6d}  n_pos={m['n_positive']:>5d}  {tag}  "
                    f"brier={m['brier']:.4g}  lift={m['lift_at_top_k']:.3f}"
                )
            else:
                tag = "spec" if m["is_specificity_only"] else f"rho={m['spearman_rho']:+.4f}"
                print(
                    f"  fold {m['fold_idx']}: {fold.held_out_obs_ids[0]:>20s}  "
                    f"n_test={m['n_tiles']:>6d}  {tag}  "
                    f"rmse_log1p={m['rmse_log1p']:.4g}  auc={m['presence_auc']:.3f}"
                )

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    aggregate = (
        aggregate_fold_metrics_classification(per_fold)
        if task == "classification"
        else aggregate_fold_metrics(per_fold)
    )

    snap = dict(snapshot or {})
    snap.setdefault("target_col", target_col)
    snap.setdefault("task", task)
    snap.setdefault("scheme", scheme)
    snap.setdefault("scale_idx", scale_idx)
    snap.setdefault("written_at_iso", _dt.datetime.now(_dt.timezone.utc).isoformat())

    if verbose:
        if task == "classification":
            print(
                f"  AGG: auc={aggregate['auc_mean']:+.4f} +/- "
                f"{aggregate['auc_std']:.4f}  brier={aggregate['brier_mean']:.4g}  "
                f"lift={aggregate['lift_at_top_k_mean']:.3f}  "
                f"(n={aggregate['auc_n']} real folds, "
                f"{aggregate['n_specificity_folds']} specificity)"
            )
        else:
            print(
                f"  AGG: spearman={aggregate['spearman_rho_mean']:+.4f} +/- "
                f"{aggregate['spearman_rho_std']:.4f}  (n={aggregate['spearman_n']} real folds, "
                f"{aggregate['n_specificity_folds']} specificity)"
            )

    return RunResult(predictions=predictions, per_fold_metrics=per_fold, aggregate=aggregate, snapshot=snap)


# ============================================================================
# Artifact persistence
# ============================================================================


def write_run_artifacts(result: RunResult, out_dir: Path | str) -> dict:
    """Write predictions.parquet, metrics.json, snapshot.yaml-equivalent JSON.

    Caller is responsible for `out_dir = models/{name}/{config_hash}/[scale_S/]`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.parquet"
    result.predictions.to_parquet(pred_path, index=False)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "per_fold": result.per_fold_metrics,
                "aggregate": result.aggregate,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    snap_path = out_dir / "snapshot.json"
    snap_path.write_text(json.dumps(result.snapshot, indent=2, default=str), encoding="utf-8")

    return {
        "predictions": str(pred_path),
        "metrics": str(metrics_path),
        "snapshot": str(snap_path),
    }
