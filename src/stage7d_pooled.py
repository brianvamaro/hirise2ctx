"""Stage 7d -- pooled cross-image boulder-rich vs boulder-poor colour test.

Joins per-tile colour features (``dataset_v2/features_colour.parquet``) to the
per-image labels at S=64, partitions tiles into boulder-rich vs boulder-poor under
two rules (P4 = ``fractional_area >= 1e-2``; P2 = ``boulder_count > 50``), and
runs three pooled statistical tests per colour feature:

  * raw                  -- pooled Mann-Whitney U + Cohen's d on the raw I/F or ratio
  * per-image standardised -- subtract per-image mean / divide by per-image std,
                              then pool, MW + Cohen's d on the z-scored values
                              (the headline cross-image test per PLAN_Compositional.md §4.2)
  * partial dust         -- residualise the feature on ``dust_index_RED_over_BG``
                              per image, then pool, MW + Cohen's d on the residuals
                              (the dust-confound discriminator per PLAN §5.2)

Plus the §4.3 continuous-target Spearman check between each feature and
``boulder_count`` (pooled standardised, pooled partial-dust, per-image).

Per-image inclusion rule: an image contributes to the pooled tests only if it has
>= ``min_per_class`` rich AND >= ``min_per_class`` poor tiles under the partition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats


SCALE_IDX_S64 = 3

# PLAN_Compositional.md §4 + HANDOFF_NEXT_SESSION.md spec the two partitions
P4_AREA_THRESHOLD = 1e-2
P2_COUNT_THRESHOLD = 50

COLOUR_FEATURES: tuple[str, ...] = (
    "IR_iof", "RED_iof", "BG_iof",
    "IR_over_RED", "IR_over_BG", "dust_index_RED_over_BG",
)
DUST_COL = "dust_index_RED_over_BG"

PartitionRule = Literal["P4_area", "P2_count"]
_PARTITION_COLS: dict[PartitionRule, str] = {
    "P4_area": "is_rich_P4",
    "P2_count": "is_rich_P2",
}


# ---------------------------------------------------------------------------
# Data loading + partitioning
# ---------------------------------------------------------------------------
def load_joined(
    features_path: Path | str,
    labels_dir: Path | str,
    scale_idx: int = SCALE_IDX_S64,
) -> pd.DataFrame:
    """Inner-join the colour features parquet with per-image label parquets.

    Returns one row per colour-covered tile at ``scale_idx``, carrying both the
    colour features and the labelling base stats (``boulder_area``,
    ``boulder_count``, ``fractional_area``).
    """
    features_path = Path(features_path)
    labels_dir = Path(labels_dir)

    feats = pd.read_parquet(features_path)
    feats = feats[feats["scale_idx"] == scale_idx].copy()

    label_cols = [
        "obs_id", "scale_idx", "ti", "tj",
        "boulder_area", "boulder_count", "tile_area",
        "fractional_area", "binary_by_area", "binary_by_count",
    ]
    frames: list[pd.DataFrame] = []
    for obs_id in feats["obs_id"].unique():
        lab_path = labels_dir / f"{obs_id}.parquet"
        if not lab_path.exists():
            continue
        lab = pd.read_parquet(lab_path)
        lab = lab[lab["scale_idx"] == scale_idx][label_cols]
        frames.append(lab)
    labels = pd.concat(frames, ignore_index=True)
    return feats.merge(labels, on=["obs_id", "scale_idx", "ti", "tj"], how="inner")


def add_partitions(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_rich_P4`` and ``is_rich_P2`` boolean columns."""
    out = df.copy()
    out["is_rich_P4"] = out["fractional_area"] >= P4_AREA_THRESHOLD
    out["is_rich_P2"] = out["boulder_count"] > P2_COUNT_THRESHOLD
    return out


# ---------------------------------------------------------------------------
# Stats primitives
# ---------------------------------------------------------------------------
def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-variance Cohen's d for ``x`` vs ``y`` (positive = x > y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def mann_whitney_with_effect(rich: np.ndarray, poor: np.ndarray) -> dict:
    """Two-sided MW-U + Cohen's d + summary stats. NaNs are dropped."""
    rich = np.asarray(rich, dtype=float)
    poor = np.asarray(poor, dtype=float)
    rich = rich[~np.isnan(rich)]
    poor = poor[~np.isnan(poor)]
    n_rich, n_poor = len(rich), len(poor)

    def _stat(a, fn, ddof=0):
        if len(a) == 0 or (ddof and len(a) <= ddof):
            return float("nan")
        return float(fn(a) if ddof == 0 else fn(a, ddof=ddof))

    base = dict(
        n_rich=n_rich, n_poor=n_poor, n_total=n_rich + n_poor,
        mean_rich=_stat(rich, np.mean), mean_poor=_stat(poor, np.mean),
        median_rich=_stat(rich, np.median), median_poor=_stat(poor, np.median),
        std_rich=_stat(rich, np.std, ddof=1), std_poor=_stat(poor, np.std, ddof=1),
    )
    if n_rich < 2 or n_poor < 2:
        return {**base, "statistic": float("nan"), "p_value": float("nan"),
                "effect_size": float("nan"), "effect_size_type": "cohen_d"}
    U, p = stats.mannwhitneyu(rich, poor, alternative="two-sided")
    return {**base, "statistic": float(U), "p_value": float(p),
            "effect_size": cohen_d(rich, poor), "effect_size_type": "cohen_d"}


def spearman_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Spearman rho + two-sided p, dropping NaN pairs. Returns (rho, p, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def per_image_standardise(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    image_col: str = "obs_id",
    suffix: str = "_z",
) -> pd.DataFrame:
    """Subtract per-image mean and divide by per-image std for each column.

    Adds ``{col}{suffix}`` columns. Constant-within-image features (std=0) become
    NaN under that image.
    """
    out = df.copy()
    grouped = out.groupby(image_col)
    for col in feature_cols:
        mu = grouped[col].transform("mean")
        sd = grouped[col].transform("std")
        z = (out[col] - mu) / sd.replace(0, np.nan)
        out[f"{col}{suffix}"] = z
    return out


def residualise_per_image(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    image_col: str = "obs_id",
) -> pd.Series:
    """Linear residuals of ``y`` on ``x``, fit per image.

    For images with < 3 tiles or zero variance in ``x``, the per-image residuals are
    NaN. Returns a Series aligned to ``df.index``.
    """
    residuals = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, idx in df.groupby(image_col).groups.items():
        sub = df.loc[idx]
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3 or np.var(x[mask]) == 0:
            continue
        slope, intercept, *_ = stats.linregress(x[mask], y[mask])
        residuals.loc[idx[mask]] = y[mask] - (intercept + slope * x[mask])
    return residuals


# ---------------------------------------------------------------------------
# Test orchestration
# ---------------------------------------------------------------------------
def eligible_images(
    df: pd.DataFrame,
    partition_rule: PartitionRule,
    min_per_class: int = 5,
) -> pd.Index:
    """Images with >= ``min_per_class`` rich AND >= ``min_per_class`` poor tiles."""
    rich_col = _PARTITION_COLS[partition_rule]
    counts = df.groupby("obs_id")[rich_col].agg(["sum", "count"])
    counts["n_rich"] = counts["sum"].astype(int)
    counts["n_poor"] = (counts["count"] - counts["sum"]).astype(int)
    mask = (counts["n_rich"] >= min_per_class) & (counts["n_poor"] >= min_per_class)
    return counts.index[mask]


def _empty_row_skeleton() -> dict:
    return dict(
        level=None, obs_id=None, partition_rule=None, feature=None, test_type=None,
        controls_for=None, n_images_pooled=None,
        n_rich=None, n_poor=None, n_total=None,
        mean_rich=None, mean_poor=None, median_rich=None, median_poor=None,
        std_rich=None, std_poor=None,
        statistic=None, p_value=None,
        effect_size=None, effect_size_type=None,
    )


def run_pooled_binary_tests(
    df: pd.DataFrame,
    partition_rule: PartitionRule,
    feature_cols: Iterable[str] = COLOUR_FEATURES,
    dust_col: str = DUST_COL,
    min_per_class: int = 5,
) -> pd.DataFrame:
    """Pooled MW + Cohen's d under raw / standardised / partial-dust transforms."""
    feature_cols = list(feature_cols)
    rich_col = _PARTITION_COLS[partition_rule]
    keep = eligible_images(df, partition_rule, min_per_class)
    sub = df[df["obs_id"].isin(keep)].copy()
    sub = per_image_standardise(sub, feature_cols + [dust_col])
    n_images = len(keep)

    rows: list[dict] = []
    for feat in feature_cols:
        for test_type, value_col, controls_for in [
            ("mann_whitney_raw", feat, None),
            ("mann_whitney_standardised", f"{feat}_z", None),
        ]:
            rich = sub.loc[sub[rich_col], value_col].to_numpy()
            poor = sub.loc[~sub[rich_col], value_col].to_numpy()
            r = mann_whitney_with_effect(rich, poor)
            rows.append({**_empty_row_skeleton(),
                         "level": "pooled", "partition_rule": partition_rule,
                         "feature": feat, "test_type": test_type,
                         "controls_for": controls_for,
                         "n_images_pooled": n_images, **r})
        if feat == dust_col:
            continue
        resid = residualise_per_image(sub, y_col=feat, x_col=dust_col)
        rich = resid[sub[rich_col]].to_numpy()
        poor = resid[~sub[rich_col]].to_numpy()
        r = mann_whitney_with_effect(rich, poor)
        rows.append({**_empty_row_skeleton(),
                     "level": "pooled", "partition_rule": partition_rule,
                     "feature": feat, "test_type": "mann_whitney_partial_dust",
                     "controls_for": dust_col,
                     "n_images_pooled": n_images, **r})
    return pd.DataFrame(rows)


def run_per_image_binary_tests(
    df: pd.DataFrame,
    partition_rule: PartitionRule,
    feature_cols: Iterable[str] = COLOUR_FEATURES,
    min_per_class: int = 5,
) -> pd.DataFrame:
    """Per-image MW + Cohen's d on the raw features (no standardisation)."""
    feature_cols = list(feature_cols)
    rich_col = _PARTITION_COLS[partition_rule]
    rows: list[dict] = []
    for obs_id, sub in df.groupby("obs_id"):
        n_rich = int(sub[rich_col].sum())
        n_poor = int((~sub[rich_col]).sum())
        if n_rich < min_per_class or n_poor < min_per_class:
            continue
        for feat in feature_cols:
            rich = sub.loc[sub[rich_col], feat].to_numpy()
            poor = sub.loc[~sub[rich_col], feat].to_numpy()
            r = mann_whitney_with_effect(rich, poor)
            rows.append({**_empty_row_skeleton(),
                         "level": "per_image", "obs_id": obs_id,
                         "partition_rule": partition_rule, "feature": feat,
                         "test_type": "mann_whitney_raw",
                         "n_images_pooled": 1, **r})
    return pd.DataFrame(rows)


def run_spearman_tests(
    df: pd.DataFrame,
    feature_cols: Iterable[str] = COLOUR_FEATURES,
    target_col: str = "boulder_count",
    dust_col: str = DUST_COL,
    min_image_tiles: int = 5,
) -> pd.DataFrame:
    """Spearman rho of each feature vs ``target_col`` — pooled std, partial-dust, per-image."""
    feature_cols = list(feature_cols)
    sub = per_image_standardise(df, feature_cols + [dust_col])
    n_images = sub["obs_id"].nunique()

    rows: list[dict] = []
    for feat in feature_cols:
        rho, p, n = spearman_with_p(sub[f"{feat}_z"].to_numpy(),
                                    sub[target_col].to_numpy())
        rows.append({**_empty_row_skeleton(),
                     "level": "pooled", "partition_rule": None, "feature": feat,
                     "test_type": "spearman_count_standardised",
                     "controls_for": None, "n_images_pooled": n_images,
                     "n_total": n, "statistic": rho, "p_value": p,
                     "effect_size": rho, "effect_size_type": "spearman_rho"})
        if feat == dust_col:
            continue
        r_feat = residualise_per_image(sub, y_col=feat, x_col=dust_col)
        r_target = residualise_per_image(sub, y_col=target_col, x_col=dust_col)
        rho_r, p_r, n_r = spearman_with_p(r_feat.to_numpy(), r_target.to_numpy())
        rows.append({**_empty_row_skeleton(),
                     "level": "pooled", "partition_rule": None, "feature": feat,
                     "test_type": "spearman_count_partial_dust",
                     "controls_for": dust_col, "n_images_pooled": n_images,
                     "n_total": n_r, "statistic": rho_r, "p_value": p_r,
                     "effect_size": rho_r, "effect_size_type": "spearman_rho"})

    for obs_id, gsub in df.groupby("obs_id"):
        if len(gsub) < min_image_tiles:
            continue
        for feat in feature_cols:
            rho, p, n = spearman_with_p(gsub[feat].to_numpy(),
                                        gsub[target_col].to_numpy())
            rows.append({**_empty_row_skeleton(),
                         "level": "per_image", "obs_id": obs_id,
                         "partition_rule": None, "feature": feat,
                         "test_type": "spearman_count_raw",
                         "controls_for": None, "n_images_pooled": 1,
                         "n_total": n, "statistic": rho, "p_value": p,
                         "effect_size": rho, "effect_size_type": "spearman_rho"})
    return pd.DataFrame(rows)


def run_all(
    df: pd.DataFrame,
    feature_cols: Iterable[str] = COLOUR_FEATURES,
    dust_col: str = DUST_COL,
    min_per_class: int = 5,
) -> pd.DataFrame:
    """Run the full Stage 7d suite. ``df`` must already have partition columns."""
    feature_cols = list(feature_cols)
    parts: list[pd.DataFrame] = []
    for rule in ("P4_area", "P2_count"):
        parts.append(run_pooled_binary_tests(df, rule, feature_cols, dust_col, min_per_class))
        parts.append(run_per_image_binary_tests(df, rule, feature_cols, min_per_class))
    parts.append(run_spearman_tests(df, feature_cols, dust_col=dust_col))
    return pd.concat(parts, ignore_index=True)
