"""Post-hoc calibration & de-compression of the boulder-abundance outputs.

Both products compress toward the middle (PLAN_Calibration.md): the Tier-1 rich/poor
probability is mis-calibrated, and the Tier-2 abundance regressor is two-sided
regression-to-the-mean (over-predicts the low end / floors above true zero,
under-predicts the high tail). This module holds the *post-hoc, ranking-preserving*
calibrators that sit AFTER the frozen recipe — so they de-compress the outputs
without reopening the freeze.

All calibrators are monotone (rank-preserving): they change the *values*, never the
ordering, so AUC / Spearman / NDCG are invariant by construction. Fit them
LOIO-honestly (`loio_calibrate`): for each held-out image, fit on the OTHER images
and apply to the held-out one — never fit on the data you score.

Primitives
----------
- ``reliability_curve`` / ``expected_calibration_error`` — Tier-1 calibration diagnosis.
- ``TemperatureScaler`` — 1-parameter probability calibration; AUC-exact.
- ``IsotonicCalibrator`` — monotone pred→E[true] fit (de-compresses toward the mean).
- ``quantile_match`` — map the prediction distribution onto the truth distribution
  (histogram/quantile transfer); recovers the full marginal incl. the tail.
- ``compression_metrics`` — the scorecard (top/low ratios, near-zero share, marginal L1).
- ``loio_calibrate`` — the honest fit-on-others / apply-to-held-out protocol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "reliability_curve", "expected_calibration_error",
    "TemperatureScaler", "IsotonicCalibrator", "quantile_match",
    "compression_metrics", "loio_calibrate",
]


# ============================================================================
# Tier-1 probability calibration diagnosis
# ============================================================================


def reliability_curve(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10):
    """Binned (mean confidence, empirical accuracy, count) for a reliability diagram.

    `y_true` is 0/1; `p_pred` in [0, 1]. Equal-width bins on the probability axis.
    Empty bins are dropped.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_pred, dtype=np.float64), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf.append(p[m].mean()); acc.append(y_true[m].mean()); cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """ECE: count-weighted mean |accuracy - confidence| over probability bins."""
    conf, acc, cnt = reliability_curve(y_true, p_pred, n_bins)
    if cnt.sum() == 0:
        return float("nan")
    return float(np.sum(cnt * np.abs(acc - conf)) / cnt.sum())


# ============================================================================
# Monotone calibrators (rank-preserving by construction)
# ============================================================================

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class TemperatureScaler:
    """One-parameter probability calibration: p' = sigmoid(logit(p) / T).

    Fit T>0 by minimizing binary cross-entropy. T>1 softens (spreads toward 0.5),
    T<1 sharpens. Strictly monotone in p, so ROC-AUC / ranking are unchanged — it
    only fixes the *confidence*, the canonical safe Tier-1 calibrator.
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, p_pred: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar

        z = _logit(p_pred)
        y = np.asarray(y_true, dtype=np.float64)

        def nll(logT):
            t = np.exp(logT)
            q = 1.0 / (1.0 + np.exp(-z / t))
            q = np.clip(q, _EPS, 1 - _EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def predict(self, p_pred: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-_logit(p_pred) / self.T))


class IsotonicCalibrator:
    """Monotone pred -> E[true] fit (sklearn isotonic, clipped at the train range).

    Minimizes squared error to the truth under a monotonicity constraint, so it
    de-compresses toward the conditional mean and preserves ranking. (Fits the
    *mean* at each level, so it recovers the tail less aggressively than
    ``quantile_match`` but adds no variance.)
    """

    def __init__(self) -> None:
        self._iso = None

    def fit(self, pred: np.ndarray, true: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._iso.fit(np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64))
        return self

    def predict(self, pred: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.predict before fit")
        return self._iso.predict(np.asarray(pred, dtype=np.float64))


def quantile_match(pred: np.ndarray, ref_pred: np.ndarray, ref_true: np.ndarray) -> np.ndarray:
    """Map `pred` through the monotone function that carries the ref-prediction
    distribution onto the ref-truth distribution (histogram / quantile transfer).

    By construction the calibrated marginal matches the truth marginal, so the high
    tail and the true-zero mass are both recovered — directly attacking compression
    — while monotonicity preserves ranking. Equivalent to sorting both references
    and interpolating the i-th sorted prediction to the i-th sorted truth.
    """
    sp = np.sort(np.asarray(ref_pred, dtype=np.float64))
    st = np.sort(np.asarray(ref_true, dtype=np.float64))
    # common quantile grid (references may differ in length)
    q = np.linspace(0, 1, min(len(sp), 4000))
    xp = np.quantile(sp, q)
    fp = np.quantile(st, q)
    return np.interp(np.asarray(pred, dtype=np.float64), xp, fp)


# ============================================================================
# Compression scorecard
# ============================================================================


def compression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        *, rich_threshold: float = 1e-2) -> dict:
    """The de-compression scorecard for an abundance prediction.

    - ``spearman``       : rank skill (must-not-regress under calibration)
    - ``top_ratio``      : mean_pred/mean_true for true > rich_threshold (1.0 = calibrated; <1 = high tail squashed)
    - ``low_over``       : mean_pred/mean_true in the truly-zero bin (>>1 = low-end over-prediction)
    - ``near_zero_pred`` : share of predictions < 1e-4 (compare to the true exact-zero share)
    - ``near_zero_true`` : share of truth exactly zero
    - ``marginal_l1``    : mean |quantile(true) - quantile(pred)| over the CDF (0 = matched marginal)
    """
    from scipy.stats import spearmanr

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    top = yt > rich_threshold
    zero = yt <= 0
    q = np.linspace(0, 1, 101)
    return {
        "spearman": float(spearmanr(yt, yp).correlation),
        "top_ratio": float(yp[top].mean() / yt[top].mean()) if top.any() else float("nan"),
        "low_over": float(yp[zero].mean() / max(yt[zero].mean(), 1e-9)) if zero.any() else float("nan"),
        "near_zero_pred": float(np.mean(yp < 1e-4)),
        "near_zero_true": float(np.mean(zero)),
        "marginal_l1": float(np.mean(np.abs(np.quantile(yt, q) - np.quantile(yp, q)))),
    }


# ============================================================================
# LOIO-honest application
# ============================================================================


def loio_calibrate(df: pd.DataFrame, fit_apply, *, group_col: str = "obs_id",
                   pred_col: str = "y_pred", true_col: str = "y_true") -> np.ndarray:
    """Apply a calibrator leave-one-image-out: fit on the other images, score the held-out.

    `fit_apply(ref_pred, ref_true, held_pred) -> held_pred_calibrated`. Returns the
    calibrated prediction aligned to `df`'s row order. This is the deployment-honest
    protocol — the calibrator never sees the image it is scoring.
    """
    out = np.full(len(df), np.nan, dtype=np.float64)
    groups = df[group_col].to_numpy()
    pred = df[pred_col].to_numpy(dtype=np.float64)
    true = df[true_col].to_numpy(dtype=np.float64)
    for g in np.unique(groups):
        held = groups == g
        ref = ~held
        out[held] = fit_apply(pred[ref], true[ref], pred[held])
    return out
