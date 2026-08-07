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


# ----------------------------------------------------------------------------
# R31 — extract_ctx_window must not stamp a cropped read with the un-cropped
# transform. Measured on a synthetic tile (DECISIONS 2026-08-06q): a 300 px west
# overhang put the origin exactly 1500 m too far west and a 200 px north overhang
# 1000 m too far north, while east/south overhang georeferenced correctly but
# truncated silently. These use a real (tiny) GeoTIFF in a zip — no network, no
# live root, ~40 ms.
# ----------------------------------------------------------------------------
import zipfile

import numpy as np
import rasterio

from src.ctx_retrieve import extract_ctx_window

# Deliberately NON-SQUARE so a crop(height, width) / crop(width, height) argument
# swap cannot pass by symmetry.
_TILE_H, _TILE_W = 300, 400


def _make_tile_zip(tmp_path):
    """A 400x300 px, 5 m/px tile whose pixel value encodes its own (row, col)."""
    inner = "tile.tif"
    tif = tmp_path / inner
    rows, cols = np.mgrid[0:_TILE_H, 0:_TILE_W]
    data = (rows * 1000 + cols).astype("uint32")
    transform = Affine(PX, 0.0, TILE_ORIGIN_X, 0.0, -PX, TILE_ORIGIN_Y)
    with rasterio.open(
        tif, "w", driver="GTiff", height=_TILE_H, width=_TILE_W, count=1,
        dtype="uint32", crs=TARGET_CRS, transform=transform,
    ) as dst:
        dst.write(data, 1)
    zp = tmp_path / "tile.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(tif, inner)
    return zp, inner


def _bounds_px(c0, r0, w, h):
    """Window in tile-pixel units -> world bounds (left, bottom, right, top)."""
    return (
        TILE_ORIGIN_X + c0 * PX,
        TILE_ORIGIN_Y - (r0 + h) * PX,
        TILE_ORIGIN_X + (c0 + w) * PX,
        TILE_ORIGIN_Y - r0 * PX,
    )


def test_extract_ctx_window_is_exact_on_a_fully_interior_window(tmp_path):
    zp, inner = _make_tile_zip(tmp_path)
    out = extract_ctx_window(zp, inner, _bounds_px(50, 40, 100, 80), tmp_path / "w.tif")
    with rasterio.open(out) as src:
        assert (src.height, src.width) == (80, 100)
        # The pixel at output (0,0) must be the source pixel the transform claims.
        assert int(src.read(1)[0, 0]) == 40 * 1000 + 50
        assert src.transform.c == pytest.approx(TILE_ORIGIN_X + 50 * PX)
        assert src.transform.f == pytest.approx(TILE_ORIGIN_Y - 40 * PX)


@pytest.mark.parametrize(
    "name,c0,r0,w,h",
    [
        ("west", -300, 40, 400, 80),    # origin would be stamped 1500 m too far west
        ("north", 50, -200, 100, 280),  # origin would be stamped 1000 m too far north
        ("east", 350, 40, 300, 80),     # correct origin, silently short
        ("south", 50, 250, 100, 300),   # correct origin, silently short
        ("nw_corner", -200, -300, 400, 400),
    ],
)
def test_extract_ctx_window_refuses_an_overhanging_window(tmp_path, name, c0, r0, w, h):
    zp, inner = _make_tile_zip(tmp_path)
    with pytest.raises(ValueError, match="overhangs the source raster"):
        extract_ctx_window(zp, inner, _bounds_px(c0, r0, w, h), tmp_path / f"{name}.tif")


def test_west_overhang_would_otherwise_be_misgeoreferenced(tmp_path):
    """The defect itself, via the test-only escape hatch.

    This is the assertion that pins the fix: with the clip allowed, the transform must
    still describe the pixels actually written. Revert to `src.window_transform(window)`
    on the un-cropped window and the origin lands 1500 m west of the truth, and this
    fails.
    """
    zp, inner = _make_tile_zip(tmp_path)
    out = extract_ctx_window(
        zp, inner, _bounds_px(-300, 40, 400, 80), tmp_path / "w.tif",
        _allow_partial_tile=True,
    )
    with rasterio.open(out) as src:
        assert (src.height, src.width) == (80, 100)      # cropped 400 -> 100 cols
        first = int(src.read(1)[0, 0])
        assert first == 40 * 1000 + 0                    # source col 0 landed at output 0
        # ...so the origin must be the tile's own left edge, NOT 300 px west of it.
        assert src.transform.c == pytest.approx(TILE_ORIGIN_X)
        assert src.transform.c != pytest.approx(TILE_ORIGIN_X - 300 * PX)


def test_north_overhang_would_otherwise_be_misgeoreferenced(tmp_path):
    zp, inner = _make_tile_zip(tmp_path)
    out = extract_ctx_window(
        zp, inner, _bounds_px(50, -200, 100, 280), tmp_path / "n.tif",
        _allow_partial_tile=True,
    )
    with rasterio.open(out) as src:
        assert (src.height, src.width) == (80, 100)      # cropped 280 -> 80 rows
        assert int(src.read(1)[0, 0]) == 0 * 1000 + 50   # source row 0 landed at output 0
        assert src.transform.f == pytest.approx(TILE_ORIGIN_Y)
        assert src.transform.f != pytest.approx(TILE_ORIGIN_Y + 200 * PX)


def test_production_never_passes_the_escape_hatch():
    """`_allow_partial_tile` is test-only; nothing in src/ or scripts/ may set it."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    # Look for it being PASSED (`_allow_partial_tile=...`), not merely named — the
    # parameter declaration and the docstring that warns about it both mention it.
    hits = [
        f"{p.relative_to(repo)}:{i}"
        for d in ("src", "scripts")
        for p in (repo / d).rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "_allow_partial_tile=" in line and not line.lstrip().startswith("#")
    ]
    assert hits == [], f"production code sets the test-only escape hatch: {hits}"
