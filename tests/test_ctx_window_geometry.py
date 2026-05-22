"""Unit tests for Stage 2 window-bounds geometry (no network).

Synthetic polygons + a synthetic CTX affine transform let us pin down the pixel-snap
behavior without touching the real 1.5 GB tile zips. The real-tile case is covered by
the slow integration test `test_stage2_one_image.py`.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pyproj
import pytest
from rasterio.transform import Affine
from shapely.geometry import Polygon

from src.ctx_retrieve import (
    _snap_bounds_to_pixel_grid,
    compute_window_bounds,
    nominal_footprint_bounds,
)

# Murray Lab CTX mosaic v01 is 5 m/pixel, north-up. Pick a synthetic origin far from zero
# so a "bounds at the origin" bug couldn't accidentally pass.
PX = 5.0
TILE_ORIGIN_X = 1_000_000.0
TILE_ORIGIN_Y = 2_500_000.0
TILE_TRANSFORM = Affine(PX, 0.0, TILE_ORIGIN_X, 0.0, -PX, TILE_ORIGIN_Y)

# Lift the project's target CRS (Mars 2000 equirectangular) out of config.yaml so the
# nominal_footprint_bounds projection step uses the real CRS, not a stand-in.
from src.config import load_config

_CFG = load_config("config.yaml")
TARGET_CRS = _CFG["target_crs"]


def _polygon_gdf(polys):
    return gpd.GeoDataFrame(geometry=polys, crs=TARGET_CRS)


def test_snap_bounds_grows_outward_and_aligns_to_grid():
    # Input bounds that are intentionally off-grid by fractional pixels.
    raw = (
        TILE_ORIGIN_X + 7.3,        # xmin: between cols 1 and 2
        TILE_ORIGIN_Y - 22.4,       # ymin: between rows 4 and 5 (from top)
        TILE_ORIGIN_X + 18.6,       # xmax: between cols 3 and 4
        TILE_ORIGIN_Y - 6.1,        # ymax: between rows 1 and 2
    )
    snapped = _snap_bounds_to_pixel_grid(raw, TILE_TRANSFORM)
    xmin, ymin, xmax, ymax = snapped
    # Each snapped value is an integer multiple of PX away from the tile origin
    assert ((xmin - TILE_ORIGIN_X) / PX).is_integer()
    assert ((xmax - TILE_ORIGIN_X) / PX).is_integer()
    assert ((TILE_ORIGIN_Y - ymin) / PX).is_integer()
    assert ((TILE_ORIGIN_Y - ymax) / PX).is_integer()
    # And the snapped bbox always contains the raw bbox
    assert xmin <= raw[0] and ymin <= raw[1]
    assert xmax >= raw[2] and ymax >= raw[3]
    # Snap distance is < 1 pixel in each direction
    assert (raw[0] - xmin) < PX
    assert (xmax - raw[2]) < PX
    assert (raw[1] - ymin) < PX
    assert (ymax - raw[3]) < PX


def test_snap_bounds_is_idempotent_on_already_snapped_input():
    raw = (TILE_ORIGIN_X + 5.0, TILE_ORIGIN_Y - 50.0, TILE_ORIGIN_X + 100.0, TILE_ORIGIN_Y - 10.0)
    once = _snap_bounds_to_pixel_grid(raw, TILE_TRANSFORM)
    twice = _snap_bounds_to_pixel_grid(once, TILE_TRANSFORM)
    assert once == twice


def test_compute_window_bounds_polygon_bbox_plus_buffer_snapped():
    # Two small polygons; bbox spans ~30 m E-W, ~50 m N-S
    polys = [
        Polygon([
            (TILE_ORIGIN_X + 100.0, TILE_ORIGIN_Y - 200.0),
            (TILE_ORIGIN_X + 110.0, TILE_ORIGIN_Y - 200.0),
            (TILE_ORIGIN_X + 110.0, TILE_ORIGIN_Y - 210.0),
            (TILE_ORIGIN_X + 100.0, TILE_ORIGIN_Y - 210.0),
        ]),
        Polygon([
            (TILE_ORIGIN_X + 130.0, TILE_ORIGIN_Y - 240.0),
            (TILE_ORIGIN_X + 132.5, TILE_ORIGIN_Y - 240.0),  # off-grid intentionally
            (TILE_ORIGIN_X + 132.5, TILE_ORIGIN_Y - 252.5),
            (TILE_ORIGIN_X + 130.0, TILE_ORIGIN_Y - 252.5),
        ]),
    ]
    gdf = _polygon_gdf(polys)
    buffer_m = 1000.0
    bounds = compute_window_bounds(gdf, buffer_m, TILE_TRANSFORM)
    xmin, ymin, xmax, ymax = bounds
    raw_xmin, raw_ymin, raw_xmax, raw_ymax = gdf.total_bounds
    # The window must contain the buffered bbox
    assert xmin <= raw_xmin - buffer_m
    assert ymin <= raw_ymin - buffer_m
    assert xmax >= raw_xmax + buffer_m
    assert ymax >= raw_ymax + buffer_m
    # And be pixel-aligned to the tile origin
    assert ((xmin - TILE_ORIGIN_X) / PX).is_integer()
    assert ((xmax - TILE_ORIGIN_X) / PX).is_integer()
    assert ((TILE_ORIGIN_Y - ymin) / PX).is_integer()
    assert ((TILE_ORIGIN_Y - ymax) / PX).is_integer()


def test_compute_window_bounds_rejects_empty_gdf():
    empty = gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    with pytest.raises(ValueError, match="empty"):
        compute_window_bounds(empty, 1000.0, TILE_TRANSFORM)


def test_nominal_footprint_bounds_centered_on_manifest_point():
    # Use an arbitrary manifest-ish point; nominal HiRISE footprint 6 x 16 km.
    row = pd.Series({"CenterLat": 41.6915, "CenterLon_180": 0.829})
    width_m = 6000.0
    length_m = 16000.0
    # Use a "tile transform" whose origin is somewhere reasonable; doesn't have to align.
    transform = Affine(PX, 0.0, 0.0, 0.0, -PX, 3_000_000.0)
    bounds = nominal_footprint_bounds(row, TARGET_CRS, width_m, length_m, transform)
    xmin, ymin, xmax, ymax = bounds

    # Width / length match (post-snap, within 1 px each side)
    assert abs((xmax - xmin) - width_m) <= 2 * PX
    assert abs((ymax - ymin) - length_m) <= 2 * PX

    # Center is within 1 pixel of the projected manifest point
    target = pyproj.CRS.from_user_input(TARGET_CRS)
    tx = pyproj.Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    cx_expected, cy_expected = tx.transform(row["CenterLon_180"], row["CenterLat"])
    cx_actual = (xmin + xmax) / 2.0
    cy_actual = (ymin + ymax) / 2.0
    assert abs(cx_actual - cx_expected) <= PX
    assert abs(cy_actual - cy_expected) <= PX

    # And pixel-aligned to the supplied transform
    assert ((xmin - transform.c) / PX).is_integer()
    assert ((transform.f - ymax) / PX).is_integer()


def test_compute_window_bounds_accepts_list_transform():
    """Sidecar JSON stores the transform as a 6-element list; helper must accept that form."""
    list_transform = [PX, 0.0, TILE_ORIGIN_X, 0.0, -PX, TILE_ORIGIN_Y]
    polys = [
        Polygon([
            (TILE_ORIGIN_X + 100.0, TILE_ORIGIN_Y - 200.0),
            (TILE_ORIGIN_X + 110.0, TILE_ORIGIN_Y - 200.0),
            (TILE_ORIGIN_X + 110.0, TILE_ORIGIN_Y - 210.0),
            (TILE_ORIGIN_X + 100.0, TILE_ORIGIN_Y - 210.0),
        ]),
    ]
    gdf = _polygon_gdf(polys)
    bounds_via_list = compute_window_bounds(gdf, 500.0, list_transform)
    bounds_via_affine = compute_window_bounds(gdf, 500.0, TILE_TRANSFORM)
    assert bounds_via_list == bounds_via_affine
