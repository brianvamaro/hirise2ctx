"""Tests for `src.spatial_features.add_neighbour_features` (Stage 6a).

Synthetic-data only -- exercises NaN handling, edge padding, scale isolation, and
constant-feature sanity without touching the cache or any real Stage 4b parquet.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.spatial_features import (
    STATS_SUPPORTED,
    add_neighbour_features,
    select_feature_columns,
)


def _dense_grid_df(
    obs_id: str = "ESP_TEST_0001",
    scale_idx: int = 3,
    tile_size_px: int = 64,
    n_ti: int = 4,
    n_tj: int = 5,
    feature_value: float | np.ndarray = 1.0,
    feature_name: str = "intensity_mean",
) -> pd.DataFrame:
    """Build a fully-dense `(ti, tj)` rectangle for one image x scale group."""
    ti, tj = np.meshgrid(np.arange(n_ti), np.arange(n_tj), indexing="ij")
    n = ti.size
    if np.isscalar(feature_value):
        vals = np.full(n, float(feature_value))
    else:
        vals = np.asarray(feature_value, dtype=np.float64).ravel()
        assert vals.size == n, f"feature_value length {vals.size} != grid {n}"
    return pd.DataFrame({
        "obs_id": [obs_id] * n,
        "scale_idx": [scale_idx] * n,
        "tile_size_px": [tile_size_px] * n,
        "ti": ti.ravel(),
        "tj": tj.ravel(),
        feature_name: vals,
    })


def test_constant_feature_yields_constant_mean_and_max_zero_std():
    """A grid where every tile has feature value k:
       - nbr_mean and nbr_max should be k everywhere (interior and edges).
       - nbr_std should be 0 everywhere (no variation across the window).
    """
    df = _dense_grid_df(n_ti=5, n_tj=5, feature_value=3.14)
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])

    assert "nbr_mean_intensity_mean" in out.columns
    assert "nbr_max_intensity_mean" in out.columns
    assert "nbr_std_intensity_mean" in out.columns
    np.testing.assert_allclose(out["nbr_mean_intensity_mean"], 3.14)
    np.testing.assert_allclose(out["nbr_max_intensity_mean"], 3.14)
    # uniform_filter introduces ~1e-8 float noise on E[X^2] - E[X]^2 cancellation; this
    # is well below any modelling tolerance and the value is effectively zero.
    np.testing.assert_allclose(out["nbr_std_intensity_mean"], 0.0, atol=1e-6)


def test_interior_tile_mean_equals_arithmetic_mean_over_3x3():
    """For an interior tile on a 5x5 dense grid, nbr_mean should equal the plain mean
    of its 9-cell window (no NaN handling kicks in)."""
    # Build a deterministic grid: feature_value[i, j] = 10*i + j.
    n = 5
    raw = np.arange(n * n, dtype=np.float64).reshape(n, n)
    # Interior tile at (2, 2): the 3x3 window is rows 1..3 and cols 1..3.
    expected_mean = raw[1:4, 1:4].mean()
    expected_max = raw[1:4, 1:4].max()
    expected_std = raw[1:4, 1:4].std(ddof=0)

    df = _dense_grid_df(n_ti=n, n_tj=n, feature_value=raw.ravel())
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])
    interior_row = out[(out["ti"] == 2) & (out["tj"] == 2)].iloc[0]
    assert math.isclose(interior_row["nbr_mean_intensity_mean"], expected_mean)
    assert math.isclose(interior_row["nbr_max_intensity_mean"], expected_max)
    assert math.isclose(interior_row["nbr_std_intensity_mean"], expected_std, abs_tol=1e-12)


def test_edge_tile_aggregates_only_present_neighbours():
    """A corner tile on a 3x3 dense grid sees only 4 neighbours (self + 3 in-grid),
    not 9. The mean / std should reflect those 4 cells, not 9 with 0-fills."""
    n = 3
    raw = np.arange(n * n, dtype=np.float64).reshape(n, n)  # 0..8 row-major
    df = _dense_grid_df(n_ti=n, n_tj=n, feature_value=raw.ravel())
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])
    # Corner tile at (0, 0): 3x3 window is rows -1..1 and cols -1..1; the in-grid
    # subset is {(0,0), (0,1), (1,0), (1,1)} -> values {0, 1, 3, 4}.
    corner = out[(out["ti"] == 0) & (out["tj"] == 0)].iloc[0]
    in_window = np.array([0.0, 1.0, 3.0, 4.0])
    assert math.isclose(corner["nbr_mean_intensity_mean"], in_window.mean())
    assert math.isclose(corner["nbr_max_intensity_mean"], in_window.max())
    assert math.isclose(corner["nbr_std_intensity_mean"], in_window.std(ddof=0))


def test_single_tile_group_has_self_only_and_nan_std():
    """A group with one tile: mean = max = self; std = NaN (count < 2)."""
    df = _dense_grid_df(n_ti=1, n_tj=1, feature_value=7.0)
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])
    row = out.iloc[0]
    assert row["nbr_mean_intensity_mean"] == 7.0
    assert row["nbr_max_intensity_mean"] == 7.0
    assert math.isnan(row["nbr_std_intensity_mean"])


def test_sparse_grid_with_missing_centre_excludes_gap_from_window():
    """A 3x3 dense grid with the centre tile (1, 1) removed: each remaining tile's
    window must NOT count the gap as a 0-fill. The 4 corner tiles see their own
    2x2 sub-window (4 cells, none at the centre); the 4 edge tiles see a 3-cell
    sub-window."""
    n = 3
    raw = np.arange(n * n, dtype=np.float64).reshape(n, n)  # values 0..8
    df = _dense_grid_df(n_ti=n, n_tj=n, feature_value=raw.ravel())
    df = df[~((df["ti"] == 1) & (df["tj"] == 1))].reset_index(drop=True)
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])

    # Corner (0, 0): present neighbours {(0,0), (0,1), (1,0)} = {0, 1, 3}.
    corner = out[(out["ti"] == 0) & (out["tj"] == 0)].iloc[0]
    expected = np.array([0.0, 1.0, 3.0])
    assert math.isclose(corner["nbr_mean_intensity_mean"], expected.mean())
    assert math.isclose(corner["nbr_max_intensity_mean"], expected.max())
    assert math.isclose(corner["nbr_std_intensity_mean"], expected.std(ddof=0))
    # Edge (0, 1): present neighbours {(0,0), (0,1), (0,2), (1,0), (1,2)} = {0, 1, 2, 3, 5}.
    edge = out[(out["ti"] == 0) & (out["tj"] == 1)].iloc[0]
    expected_edge = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    assert math.isclose(edge["nbr_mean_intensity_mean"], expected_edge.mean())


def test_nan_source_value_is_excluded_from_aggregation():
    """If a source feature is NaN at some tiles (e.g. lacunarity at S < min_tile_size_px),
    those tiles should NOT contribute to the window mean / max / std for surrounding
    tiles -- treated as a gap, not a 0 value."""
    n = 3
    vals = np.array([1.0, 2.0, 3.0,
                     4.0, np.nan, 6.0,
                     7.0, 8.0, 9.0])
    df = _dense_grid_df(n_ti=n, n_tj=n, feature_value=vals,
                        feature_name="lacunarity_shadow_b2")
    out = add_neighbour_features(df, feature_cols=["lacunarity_shadow_b2"])
    # Corner (0, 0): window in-grid is {(0,0),(0,1),(1,0),(1,1)}; (1,1) is NaN.
    # Effective values: {1, 2, 4}.
    corner = out[(out["ti"] == 0) & (out["tj"] == 0)].iloc[0]
    effective = np.array([1.0, 2.0, 4.0])
    assert math.isclose(corner["nbr_mean_lacunarity_shadow_b2"], effective.mean())
    assert math.isclose(corner["nbr_max_lacunarity_shadow_b2"], effective.max())
    # The NaN tile itself should still get its nbr_mean computed over its valid window
    # neighbours (everything except itself).
    centre = out[(out["ti"] == 1) & (out["tj"] == 1)].iloc[0]
    expected_centre_mean = vals[~np.isnan(vals)].mean()
    assert math.isclose(centre["nbr_mean_lacunarity_shadow_b2"], expected_centre_mean)


def test_scales_are_isolated():
    """A neighbour at scale_idx=2 must NOT bleed into the aggregation at scale_idx=3
    even if they share the same (obs_id, ti, tj). Different scales = different grids."""
    df_a = _dense_grid_df(scale_idx=2, tile_size_px=32, n_ti=3, n_tj=3,
                          feature_value=100.0)
    df_b = _dense_grid_df(scale_idx=3, tile_size_px=64, n_ti=3, n_tj=3,
                          feature_value=1.0)
    df = pd.concat([df_a, df_b], ignore_index=True)
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])
    np.testing.assert_allclose(
        out.loc[out["scale_idx"] == 2, "nbr_mean_intensity_mean"], 100.0,
    )
    np.testing.assert_allclose(
        out.loc[out["scale_idx"] == 3, "nbr_mean_intensity_mean"], 1.0,
    )


def test_obs_ids_are_isolated():
    """Two different ObsIds with the same (ti, tj) grid stay independent."""
    df_a = _dense_grid_df(obs_id="ESP_AAA", n_ti=3, n_tj=3, feature_value=10.0)
    df_b = _dense_grid_df(obs_id="ESP_BBB", n_ti=3, n_tj=3, feature_value=20.0)
    df = pd.concat([df_a, df_b], ignore_index=True)
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])
    np.testing.assert_allclose(
        out.loc[out["obs_id"] == "ESP_AAA", "nbr_mean_intensity_mean"], 10.0,
    )
    np.testing.assert_allclose(
        out.loc[out["obs_id"] == "ESP_BBB", "nbr_mean_intensity_mean"], 20.0,
    )


def test_row_order_is_preserved():
    """The augmented frame's row order must match the input's."""
    n = 3
    vals = np.arange(n * n, dtype=np.float64) * 1.7
    df = _dense_grid_df(n_ti=n, n_tj=n, feature_value=vals)
    # Shuffle deterministically.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    df_shuffled = df.iloc[perm].reset_index(drop=True)
    out = add_neighbour_features(df_shuffled, feature_cols=["intensity_mean"])
    # The (obs_id, ti, tj) ordering should match df_shuffled exactly.
    assert (out["ti"].to_numpy() == df_shuffled["ti"].to_numpy()).all()
    assert (out["tj"].to_numpy() == df_shuffled["tj"].to_numpy()).all()


def test_select_feature_columns_excludes_identifiers_and_patch_indices():
    """Smoke test: identifiers, config hash, patch indices are not selected."""
    df = pd.DataFrame({
        "obs_id": ["x"], "scale_idx": [3], "tile_size_px": [64], "ti": [0], "tj": [0],
        "valid_pixel_fraction": [1.0], "config_hash": ["abc"],
        "intensity_mean": [10.0], "lbp_hist_0": [0.1],
        "patch_idx_S32": [0], "patch_idx_S64": [-1],
    })
    feats = select_feature_columns(df)
    assert "intensity_mean" in feats
    assert "lbp_hist_0" in feats
    for excluded in (
        "obs_id", "scale_idx", "tile_size_px", "ti", "tj",
        "valid_pixel_fraction", "config_hash",
        "patch_idx_S32", "patch_idx_S64",
    ):
        assert excluded not in feats


def test_invalid_stencil_size_raises():
    df = _dense_grid_df()
    with pytest.raises(ValueError):
        add_neighbour_features(df, feature_cols=["intensity_mean"], stencil_size=2)
    with pytest.raises(ValueError):
        add_neighbour_features(df, feature_cols=["intensity_mean"], stencil_size=4)


def test_unknown_stat_raises():
    df = _dense_grid_df()
    with pytest.raises(ValueError):
        add_neighbour_features(df, feature_cols=["intensity_mean"], stats=("mean", "median"))


def test_default_stats_emit_all_three_per_feature():
    df = _dense_grid_df(n_ti=2, n_tj=2, feature_value=1.0)
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])
    new_cols = [c for c in out.columns if c.startswith("nbr_")]
    assert sorted(new_cols) == sorted([
        "nbr_mean_intensity_mean",
        "nbr_max_intensity_mean",
        "nbr_std_intensity_mean",
    ])


def test_subset_stats_emit_only_requested():
    df = _dense_grid_df(n_ti=2, n_tj=2, feature_value=1.0)
    out = add_neighbour_features(df, feature_cols=["intensity_mean"], stats=("mean",))
    new_cols = [c for c in out.columns if c.startswith("nbr_")]
    assert new_cols == ["nbr_mean_intensity_mean"]


def test_empty_frame_produces_empty_columns_in_full_schema():
    """Edge case: empty input still yields the full output schema with 0 rows."""
    df = pd.DataFrame({
        "obs_id": pd.Series([], dtype=str),
        "scale_idx": pd.Series([], dtype=np.int64),
        "tile_size_px": pd.Series([], dtype=np.int64),
        "ti": pd.Series([], dtype=np.int64),
        "tj": pd.Series([], dtype=np.int64),
        "intensity_mean": pd.Series([], dtype=np.float64),
    })
    out = add_neighbour_features(df, feature_cols=["intensity_mean"])
    for stat in STATS_SUPPORTED:
        assert f"nbr_{stat}_intensity_mean" in out.columns
    assert len(out) == 0
