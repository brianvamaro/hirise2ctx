"""Tests for `src.ctx_source_illumination` (Stage 6b).

Synthetic-data only -- no SeamMap or Murray-tile cache dependency. Exercises:

  * Per-tile aggregation: vectorized mean / std are correct for known angle grids.
  * Out-of-bounds tiles return NaN aggregates and n_sources=0.
  * Source-count and dominant-source-fraction reflect the rasterized SOURCE_ID.
  * Rasterize-then-aggregate round-trip on a hand-built 2-polygon GDF.
  * Empty SeamMap subset produces all-NaN windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.ctx_source_illumination import (
    ANGLE_COLUMNS,
    OUTPUT_COLUMNS,
    _aggregate_per_tile,
    add_ctx_source_illumination_features,
    rasterize_seam_map_window,
)


def _affine(px: float = 5.0, x0: float = 0.0, y0: float = 0.0):
    """Build a north-up rasterio Affine with the CTX-mosaic sign convention."""
    from rasterio.transform import Affine
    return Affine(px, 0.0, x0, 0.0, -px, y0)


def _seam_gdf(polygons_with_values):
    """Build a SeamMap-shaped GeoDataFrame from [(polygon, INCIDENCE, EMISSION,
    PHASE, SB_SLR_AZ, PRODUCT_ID), ...]."""
    import geopandas as gpd
    cols = {
        "INCIDENCE": [],
        "EMISSION": [],
        "PHASE": [],
        "SB_SLR_AZ": [],
        "PRODUCT_ID": [],
        "geometry": [],
    }
    for poly, inc, emi, pha, saz, pid in polygons_with_values:
        cols["INCIDENCE"].append(inc)
        cols["EMISSION"].append(emi)
        cols["PHASE"].append(pha)
        cols["SB_SLR_AZ"].append(saz)
        cols["PRODUCT_ID"].append(pid)
        cols["geometry"].append(poly)
    return gpd.GeoDataFrame(cols, crs="EPSG:4326")


# ============================================================================
# Per-tile aggregation -- synthetic angle arrays
# ============================================================================

def _ones_arrays(h: int, w: int, *, inc: float, emi: float, pha: float,
                 saz: float, sid: int):
    return {
        "INCIDENCE": np.full((h, w), inc, dtype=np.float32),
        "EMISSION": np.full((h, w), emi, dtype=np.float32),
        "PHASE": np.full((h, w), pha, dtype=np.float32),
        "SB_SLR_AZ": np.full((h, w), saz, dtype=np.float32),
        "SOURCE_ID": np.full((h, w), sid, dtype=np.uint16),
    }


def test_uniform_window_constant_mean_zero_std_one_source():
    arrays = _ones_arrays(64, 64, inc=42.0, emi=5.0, pha=47.0, saz=160.0, sid=7)
    df = pd.DataFrame({
        "obs_id": ["ESP_TEST"] * 4,
        "scale_idx": [3] * 4,
        "tile_size_px": [32] * 4,
        "ti": [0, 0, 1, 1],
        "tj": [0, 1, 0, 1],
    })
    out = _aggregate_per_tile(df, arrays, mosaic_row_origin=0, mosaic_col_origin=0)
    np.testing.assert_allclose(out["ctx_incidence_mean"], 42.0)
    np.testing.assert_allclose(out["ctx_emission_mean"], 5.0)
    np.testing.assert_allclose(out["ctx_phase_mean"], 47.0)
    np.testing.assert_allclose(out["ctx_subsolar_az_mean"], 160.0)
    np.testing.assert_allclose(out["ctx_incidence_std"], 0.0, atol=1e-6)
    np.testing.assert_array_equal(out["ctx_n_sources"], [1, 1, 1, 1])
    np.testing.assert_allclose(out["ctx_dominant_source_fraction"], 1.0)


def test_tile_mean_matches_manual_calculation():
    """Tile (0,0) at S=4 covers rows 0..3, cols 0..3. Mean over that block
    should equal the arithmetic mean of those 16 pixels."""
    raw = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    arrays = {
        "INCIDENCE": raw.copy(),
        "EMISSION": raw.copy() + 100,
        "PHASE": raw.copy() + 200,
        "SB_SLR_AZ": raw.copy() + 300,
        "SOURCE_ID": np.ones((8, 8), dtype=np.uint16),
    }
    df = pd.DataFrame({
        "obs_id": ["X"], "scale_idx": [1], "tile_size_px": [4],
        "ti": [0], "tj": [0],
    })
    out = _aggregate_per_tile(df, arrays, mosaic_row_origin=0, mosaic_col_origin=0)
    expected_inc = raw[0:4, 0:4].mean()
    expected_emi = (raw + 100)[0:4, 0:4].mean()
    expected_std = raw[0:4, 0:4].std(ddof=0)
    assert out["ctx_incidence_mean"][0] == pytest.approx(expected_inc, rel=1e-5)
    assert out["ctx_emission_mean"][0] == pytest.approx(expected_emi, rel=1e-5)
    assert out["ctx_incidence_std"][0] == pytest.approx(expected_std, rel=1e-5)


def test_two_sources_split_evenly():
    """Half the window has source A (incidence=40), half has source B (incidence=60).
    A single S=4 tile covering the whole 4x4 window should report mean=50,
    std=10, n_sources=2, dominant_fraction=0.5."""
    arr = np.empty((4, 4), dtype=np.float32)
    arr[:, :2] = 40.0
    arr[:, 2:] = 60.0
    sid = np.empty((4, 4), dtype=np.uint16)
    sid[:, :2] = 1
    sid[:, 2:] = 2
    arrays = {
        "INCIDENCE": arr,
        "EMISSION": np.zeros_like(arr),
        "PHASE": np.zeros_like(arr),
        "SB_SLR_AZ": np.zeros_like(arr),
        "SOURCE_ID": sid,
    }
    df = pd.DataFrame({
        "obs_id": ["X"], "scale_idx": [1], "tile_size_px": [4],
        "ti": [0], "tj": [0],
    })
    out = _aggregate_per_tile(df, arrays, mosaic_row_origin=0, mosaic_col_origin=0)
    assert out["ctx_incidence_mean"][0] == pytest.approx(50.0)
    assert out["ctx_incidence_std"][0] == pytest.approx(10.0)
    assert out["ctx_n_sources"][0] == 2
    assert out["ctx_dominant_source_fraction"][0] == pytest.approx(0.5)


def test_out_of_bounds_tile_returns_nan():
    arrays = _ones_arrays(8, 8, inc=50.0, emi=5.0, pha=55.0, saz=160.0, sid=3)
    df = pd.DataFrame({
        "obs_id": ["X"] * 2,
        "scale_idx": [3] * 2,
        "tile_size_px": [4] * 2,
        # First tile is inside; second tile is beyond the window edge.
        "ti": [0, 5], "tj": [0, 0],
    })
    out = _aggregate_per_tile(df, arrays, mosaic_row_origin=0, mosaic_col_origin=0)
    assert np.isfinite(out["ctx_incidence_mean"][0])
    assert np.isnan(out["ctx_incidence_mean"][1])
    assert out["ctx_n_sources"][1] == 0
    assert np.isnan(out["ctx_dominant_source_fraction"][1])


def test_nan_pixels_excluded_from_aggregate():
    """If half the pixels are NaN, the mean should come from the finite half only."""
    arr = np.full((4, 4), np.nan, dtype=np.float32)
    arr[:, :2] = 40.0
    sid = np.zeros((4, 4), dtype=np.uint16)
    sid[:, :2] = 1
    arrays = {
        "INCIDENCE": arr,
        "EMISSION": arr.copy(),
        "PHASE": arr.copy(),
        "SB_SLR_AZ": arr.copy(),
        "SOURCE_ID": sid,
    }
    df = pd.DataFrame({
        "obs_id": ["X"], "scale_idx": [1], "tile_size_px": [4],
        "ti": [0], "tj": [0],
    })
    out = _aggregate_per_tile(df, arrays, mosaic_row_origin=0, mosaic_col_origin=0)
    # Mean of the 8 finite pixels = 40; std = 0 (all 40s among the finite ones).
    assert out["ctx_incidence_mean"][0] == pytest.approx(40.0)
    assert out["ctx_incidence_std"][0] == pytest.approx(0.0)
    # n_sources counts SOURCE_ID > 0 even where the angle is NaN -- mirrors the
    # SeamMap convention where SOURCE_ID is 0 outside any polygon. Here all 8
    # nonzero-SID pixels carry source 1, so n_sources = 1.
    assert out["ctx_n_sources"][0] == 1
    assert out["ctx_dominant_source_fraction"][0] == pytest.approx(0.5)


# ============================================================================
# Rasterization -- 2-polygon SeamMap, hand-built CRS-free
# ============================================================================

def test_rasterize_two_polygon_seam_map():
    """Cover the left half of a 4x4 window with source A (incidence=40) and the
    right half with source B (incidence=60)."""
    transform = _affine(px=1.0, x0=0.0, y0=4.0)
    gdf = _seam_gdf([
        (box(0, 0, 2, 4), 40.0, 1.0, 41.0, 150.0, "A"),
        (box(2, 0, 4, 4), 60.0, 2.0, 62.0, 160.0, "B"),
    ])
    arrays = rasterize_seam_map_window(gdf, transform, window_h=4, window_w=4)
    np.testing.assert_array_equal(arrays["INCIDENCE"][:, :2], 40.0)
    np.testing.assert_array_equal(arrays["INCIDENCE"][:, 2:], 60.0)
    np.testing.assert_array_equal(arrays["SOURCE_ID"][:, :2], 1)
    np.testing.assert_array_equal(arrays["SOURCE_ID"][:, 2:], 2)


def test_rasterize_subsets_to_window_bbox():
    """Polygons outside the window bbox should not appear in the output (a perf
    optimisation that should also be observable)."""
    transform = _affine(px=1.0, x0=0.0, y0=4.0)
    gdf = _seam_gdf([
        (box(0, 0, 4, 4), 50.0, 5.0, 55.0, 160.0, "INSIDE"),
        (box(100, 100, 200, 200), 80.0, 20.0, 100.0, 170.0, "OUTSIDE"),
    ])
    arrays = rasterize_seam_map_window(gdf, transform, window_h=4, window_w=4)
    assert (arrays["SOURCE_ID"] == 1).all()  # only INSIDE survives


def test_rasterize_empty_intersection_returns_all_nan():
    transform = _affine(px=1.0, x0=0.0, y0=4.0)
    gdf = _seam_gdf([
        (box(100, 100, 200, 200), 80.0, 20.0, 100.0, 170.0, "FAR"),
    ])
    arrays = rasterize_seam_map_window(gdf, transform, window_h=4, window_w=4)
    assert np.isnan(arrays["INCIDENCE"]).all()
    assert (arrays["SOURCE_ID"] == 0).all()


# ============================================================================
# End-to-end via the public API
# ============================================================================

def test_add_features_end_to_end():
    transform = _affine(px=1.0, x0=0.0, y0=8.0)
    gdf = _seam_gdf([
        (box(0, 0, 4, 8), 40.0, 1.0, 41.0, 150.0, "A"),
        (box(4, 0, 8, 8), 60.0, 2.0, 62.0, 160.0, "B"),
    ])
    df = pd.DataFrame({
        "obs_id": ["ESP_TEST"] * 4,
        "scale_idx": [1] * 4,
        "tile_size_px": [4] * 4,
        "ti": [0, 0, 1, 1],
        "tj": [0, 1, 0, 1],
        "intensity_mean": [1.0, 2.0, 3.0, 4.0],
    })
    out = add_ctx_source_illumination_features(
        df, seam_gdf=gdf, window_transform=transform,
        window_h=8, window_w=8,
        mosaic_row_origin=0, mosaic_col_origin=0,
    )
    for c in OUTPUT_COLUMNS:
        assert c in out.columns
    # Left column of tiles ((0,0) and (1,0)) is fully in source A (incidence=40);
    # right column ((0,1) and (1,1)) is fully in source B (incidence=60).
    left = out[(out["tj"] == 0)]
    right = out[(out["tj"] == 1)]
    np.testing.assert_allclose(left["ctx_incidence_mean"], 40.0)
    np.testing.assert_allclose(right["ctx_incidence_mean"], 60.0)
    np.testing.assert_array_equal(out["ctx_n_sources"], [1, 1, 1, 1])
    # Original columns must be preserved unchanged.
    np.testing.assert_array_equal(out["intensity_mean"], [1.0, 2.0, 3.0, 4.0])


def test_output_columns_constant_advertises_full_set():
    """Guard against drift: OUTPUT_COLUMNS should be exactly what
    add_ctx_source_illumination_features produces."""
    expected = {
        "ctx_incidence_mean", "ctx_emission_mean", "ctx_phase_mean",
        "ctx_subsolar_az_mean", "ctx_incidence_std",
        "ctx_n_sources", "ctx_dominant_source_fraction",
    }
    assert set(OUTPUT_COLUMNS) == expected
    assert {name for _, name in ANGLE_COLUMNS} == {
        "INCIDENCE", "EMISSION", "PHASE", "SB_SLR_AZ",
    }
