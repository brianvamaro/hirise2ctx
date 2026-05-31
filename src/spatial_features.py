"""Stage 6a -- spatial-context neighbour features.

For each ``(obs_id, scale_idx)`` group in a Stage 4b feature parquet, this module
computes ``stencil_size x stencil_size`` neighbour aggregations (mean, max, std) on
the ``(ti, tj)`` grid for every existing numeric feature column.  Boulder fields are
spatially coherent (crater ejecta, fluvial deposits, rockfall debris), so a tile in
a real cluster differs from an isolated false-positive texture even when their
per-tile features look identical.  The neighbour aggregations give the model that
local context without leaving the per-tile resolution.

Output columns are named ``nbr_<stat>_<feature>`` and appended to the input frame.
The aggregation is NaN-aware: image-edge gaps and Stage-4-eligibility gaps in the
``(ti, tj)`` grid are excluded from the window; if every neighbour (including self)
is missing, the aggregation result is NaN.  ``std`` is also NaN when fewer than two
valid neighbours are in the window (no sample variance defined).  LightGBM handles
NaN natively, so missing aggregations propagate through without any special handling
on the modelling side.

This is a post-processing step that reads existing Stage 4b parquets and writes
augmented parquets to a separate output directory; the original Stage 4b cache is
not modified.  Promotion-queue spec: ``PROMOTION_QUEUE.md`` Stage 6a.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, uniform_filter

# Identifiers + the per-Stage-4b config hash are not eligible for aggregation.
# `valid_pixel_fraction` is an eligibility marker (Stage 4 only emits tiles where it
# exceeds threshold), not a texture feature; aggregating it would just mirror the
# eligibility-mask shape.  Loaders._feature_columns has a similar rule but is
# duplicated here so this module is self-contained for tests.
_NON_FEATURE_COLUMNS = frozenset({
    "obs_id", "scale_idx", "tile_size_px", "ti", "tj",
    "valid_pixel_fraction",
    "config_hash", "config_hash_feat",
})

STATS_SUPPORTED: tuple[str, ...] = ("mean", "max", "std")
DEFAULT_STENCIL_SIZE = 3


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Auto-select the numeric per-tile feature columns to aggregate.

    Skips identifiers, ``valid_pixel_fraction``, ``config_hash*``, ``patch_idx_S*``
    (context-patch row indices, not predictive), and any non-numeric column.
    """
    cols: list[str] = []
    for c in df.columns:
        if c in _NON_FEATURE_COLUMNS:
            continue
        if c.startswith("patch_idx_S"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def _aggregate_one_group(
    sub: pd.DataFrame,
    feature_cols: list[str],
    stats: tuple[str, ...],
    stencil_size: int,
) -> dict[str, np.ndarray]:
    """Compute neighbour aggregations for one ``(obs_id, scale_idx)`` group.

    Returns a dict mapping ``nbr_<stat>_<feature>`` to a 1-D array aligned with
    ``sub`` row order.  The intermediate dense grid is bounded by the group's own
    ``(ti, tj)`` bounding box so memory stays small regardless of the global grid
    extent.
    """
    n_rows = len(sub)
    if n_rows == 0:
        return {
            f"nbr_{stat}_{col}": np.zeros(0, dtype=np.float64)
            for stat in stats for col in feature_cols
        }

    ti = sub["ti"].to_numpy(dtype=np.int64)
    tj = sub["tj"].to_numpy(dtype=np.int64)
    ti_min = int(ti.min())
    tj_min = int(tj.min())
    n_ti = int(ti.max()) - ti_min + 1
    n_tj = int(tj.max()) - tj_min + 1
    rr = ti - ti_min
    cc = tj - tj_min

    win_area = float(stencil_size * stencil_size)

    out: dict[str, np.ndarray] = {}

    for col in feature_cols:
        vals = sub[col].to_numpy(dtype=np.float64)
        # Feature-level validity: a tile is valid if it exists in this group AND the
        # source feature value is finite (lacunarity / GLCM / subtile_var are NaN at
        # scales below their min_tile_size_px).
        finite_vals = np.isfinite(vals)
        valid = np.zeros((n_ti, n_tj), dtype=np.float64)
        valid[rr, cc] = finite_vals.astype(np.float64)

        grid = np.zeros((n_ti, n_tj), dtype=np.float64)
        grid[rr[finite_vals], cc[finite_vals]] = vals[finite_vals]

        # Window-sum of valid mask is the count of valid neighbours in the stencil.
        # `uniform_filter` returns the mean over the window; multiply by area to get
        # the sum.  `mode="constant", cval=0` zero-pads outside the bounding box.
        count_win = (
            uniform_filter(valid, size=stencil_size, mode="constant", cval=0.0) * win_area
        )
        # Avoid div-by-zero; we will mask the result below.
        safe_count = np.where(count_win > 0.0, count_win, 1.0)
        sum_win = (
            uniform_filter(grid, size=stencil_size, mode="constant", cval=0.0) * win_area
        )
        mean_win = sum_win / safe_count
        mean_win = np.where(count_win > 0.0, mean_win, np.nan)

        if "mean" in stats:
            out[f"nbr_mean_{col}"] = mean_win[rr, cc]

        if "max" in stats:
            # Send invalid cells to -inf so they never win the max; rescue tiles with
            # an empty window back to NaN at the end.
            grid_for_max = np.where(valid > 0.0, grid, -np.inf)
            max_win = maximum_filter(
                grid_for_max, size=stencil_size, mode="constant", cval=-np.inf,
            )
            max_win = np.where(count_win > 0.0, max_win, np.nan)
            out[f"nbr_max_{col}"] = max_win[rr, cc]

        if "std" in stats:
            sq_sum_win = (
                uniform_filter(grid * grid, size=stencil_size, mode="constant", cval=0.0)
                * win_area
            )
            ex2 = sq_sum_win / safe_count
            var_win = np.clip(ex2 - mean_win * mean_win, 0.0, None)
            std_win = np.sqrt(var_win)
            # Sample std is undefined when only one valid neighbour exists in the window.
            std_win = np.where(count_win >= 2.0, std_win, np.nan)
            out[f"nbr_std_{col}"] = std_win[rr, cc]

    return out


def add_neighbour_features(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    stats: tuple[str, ...] = STATS_SUPPORTED,
    stencil_size: int = DEFAULT_STENCIL_SIZE,
) -> pd.DataFrame:
    """Return a copy of ``df`` with neighbour-aggregated columns appended.

    Aggregation is per ``(obs_id, scale_idx)`` group: the ``(ti, tj)`` grid of each
    image-and-scale stands alone (different scales nest on the same mosaic origin
    but use different tile_size_px integer indices, so a neighbour at S=64 is *not*
    a neighbour at S=32 in this stencil).

    Args:
        df: A Stage 4b feature frame.  Must contain columns ``obs_id``,
            ``scale_idx``, ``ti``, ``tj`` plus the numeric feature columns selected
            via ``feature_cols`` (defaults to all numeric non-identifier columns).
        feature_cols: Override the auto-selected feature list.  Use ``None`` to keep
            the default (``select_feature_columns``).
        stats: Subset of ``("mean", "max", "std")`` to compute.
        stencil_size: Odd integer ``>= 3``.  Default 3 = standard 8-neighbour + self
            stencil.

    Returns:
        A new dataframe with the original columns followed by ``nbr_<stat>_<col>``
        columns in ``itertools.product(stats, feature_cols)`` order.
    """
    if stencil_size % 2 != 1 or stencil_size < 3:
        raise ValueError(f"stencil_size must be odd and >= 3, got {stencil_size}")
    for s in stats:
        if s not in STATS_SUPPORTED:
            raise ValueError(f"unsupported stat {s!r}; choose from {STATS_SUPPORTED}")
    if feature_cols is None:
        feature_cols = select_feature_columns(df)

    new_cols: dict[str, np.ndarray] = {
        f"nbr_{stat}_{col}": np.full(len(df), np.nan, dtype=np.float64)
        for stat in stats for col in feature_cols
    }

    if len(df) == 0:
        augmented = df.copy()
        for col_name, vals in new_cols.items():
            augmented[col_name] = vals
        return augmented

    # Preserve incoming row order: take a positional view via reset_index, build a
    # parallel positional array, then attach to the original index at the end.
    df_pos = df.reset_index(drop=True)
    for (_obs_id, _scale_idx), sub in df_pos.groupby(
        ["obs_id", "scale_idx"], sort=False,
    ):
        pos = sub.index.to_numpy()  # positional indices into df_pos
        per_group = _aggregate_one_group(sub, feature_cols, stats, stencil_size)
        for col_name, vals in per_group.items():
            new_cols[col_name][pos] = vals

    augmented = df.copy()
    # Align by positional order (df.copy() preserves the original index; new_cols was
    # filled by positional indices, which match df_pos.reset_index, which matches the
    # original row order of df).
    for col_name in new_cols:
        augmented[col_name] = new_cols[col_name]
    return augmented
