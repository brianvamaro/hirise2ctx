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
from typing import Callable, Iterable

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
    """
    fold_iter = fold_iter if fold_iter is not None else _default_fold_iter(scheme, scale_idx)

    pred_rows: list[pd.DataFrame] = []
    per_fold: list[dict] = []

    for fold in fold_iter():
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
            tag = "spec" if m["is_specificity_only"] else f"rho={m['spearman_rho']:+.4f}"
            print(
                f"  fold {m['fold_idx']}: {fold.held_out_obs_ids[0]:>20s}  "
                f"n_test={m['n_tiles']:>6d}  {tag}  "
                f"rmse_log1p={m['rmse_log1p']:.4g}  auc={m['presence_auc']:.3f}"
            )

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    aggregate = aggregate_fold_metrics(per_fold)

    snap = dict(snapshot or {})
    snap.setdefault("target_col", target_col)
    snap.setdefault("scheme", scheme)
    snap.setdefault("scale_idx", scale_idx)
    snap.setdefault("written_at_iso", _dt.datetime.now(_dt.timezone.utc).isoformat())

    if verbose:
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
