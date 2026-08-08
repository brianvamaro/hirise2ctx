"""Stage 4 (label generation) tests.

Unit tests use synthetic CTX windows, masks, and polygons -- no caches, no downloads.
The integration test on ESP_069669_2220 is marked `slow` and auto-skips when Stage 2/3
caches are missing.
"""
from __future__ import annotations

import json
import warnings
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, box

from src.labeling import (
    LABELS_SUBDIR,
    _apply_coreg_shift,
    _apply_detection_filters,
    _compute_grid_alignment,
    _count_centroids_per_finest_cell,
    _flatten_to_dataframe,
    _rasterize_boulders_subpixel,
    _sum_up_ladder,
    stage4_one_image,
)


# ----------------------------------------------------------------------------
# Synthetic helpers
# ----------------------------------------------------------------------------

TILE_SIZES = [8, 16, 32, 64]

# --- R78: the fixtures must never pin the mosaic grid phase to (0, 0) --------------
# Real geometry, read off disk: dataset_v2/labels/ESP_042964_2160.json carries
# mosaic_row_origin = 894 / mosaic_col_origin = 12645, and its parent Murray tile
# cache_v2/ctx_tiles/E-8_N32.json carries inner_transform origin
# (-474197.58018644986, 2133889.110839024).  0 of the 52 production label sidecars has
# either origin equal to 0 (they span 894-43,790 and 183-41,945), yet every fixture here
# used to set both to 0 -- the same fixture defect src/fgates.py:211-231 records as the
# cause of the ~100 km gate mis-key.  With the phase at zero, `ti*S - origin`, `ti*S +
# origin` and `ti*S` are the same expression and the whole grid-anchoring surface is
# untested.
MOSAIC_ORIGIN_XY = (-474197.58018644986, 2133889.110839024)   # E-8_N32 inner_transform
MOSAIC_ROW_ORIGIN = 894                                        # ESP_042964_2160 sidecar
MOSAIC_COL_ORIGIN = 12645                                      # ESP_042964_2160 sidecar

# The finest-grid index of the first coarsest-aligned cell for that phase:
#   ceil(894 / 64) * 8 = 112   and   ceil(12645 / 64) * 8 = 1584
_LADDER_RATIO = TILE_SIZES[-1] // TILE_SIZES[0]
J_MIN_ROW = math.ceil(MOSAIC_ROW_ORIGIN / TILE_SIZES[-1]) * _LADDER_RATIO
J_MIN_COL = math.ceil(MOSAIC_COL_ORIGIN / TILE_SIZES[-1]) * _LADDER_RATIO


def mars_crs_wkt(standard_parallel_deg: float = 0.0) -> str:
    """A projected, metre-unit Mars equirectangular CRS for fixtures.

    R80: the size floor is an equivalent-circle **diameter in metres**, so any fixture that
    exercises it must live in a projected metre frame. The old fixture used EPSG:4326, where
    `geopandas.area` is degrees² and emits `UserWarning: Geometry is in a geographic CRS` —
    the units were simply unpinned. `standard_parallel_deg` is exposed because the realised
    physical floor scales with the projection's standard parallel, which is the mechanism
    behind R03's mixed size floor.
    """
    return (
        'PROJCRS["TestMars",BASEGEOGCRS["GCS_TestMars",DATUM["D_TestMars",'
        'ELLIPSOID["TestMars",3396190.0,0.0,LENGTHUNIT["metre",1]]],'
        'PRIMEM["Reference_Meridian",0,ANGLEUNIT["Degree",0.0174532925199433]]],'
        'CONVERSION["EquidistantCylindrical",'
        'METHOD["Equidistant Cylindrical (Spherical)",ID["EPSG",1029]],'
        f'PARAMETER["Latitude of 1st standard parallel",{standard_parallel_deg},'
        'ANGLEUNIT["Degree",0.0174532925199433]],'
        'PARAMETER["Longitude of natural origin",0,ANGLEUNIT["Degree",0.0174532925199433]],'
        'PARAMETER["False easting",0,LENGTHUNIT["metre",1]],'
        'PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],'
        'CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1]],'
        'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1]]]'
    )


TEST_MARS_CRS_WKT = mars_crs_wkt()


def _make_window(tmp_path: Path, *, height: int, width: int, pixel_m: float = 5.0,
                 mosaic_origin_xy: tuple[float, float] = MOSAIC_ORIGIN_XY,
                 row_origin: int = MOSAIC_ROW_ORIGIN,
                 col_origin: int = MOSAIC_COL_ORIGIN,
                 mask_fill: int = 1) -> tuple[Path, Path, dict, str]:
    """Write a synthetic CTX window + mask + mosaic tile sidecar. Returns the four pieces.

    R78: the window's upper-left sits at mosaic pixel ``(row_origin, col_origin)`` of a
    mosaic whose own CRS origin is ``mosaic_origin_xy`` -- both non-zero, mirroring
    production.  `info` carries the resulting `align` dict plus `work_origin_xy`, the world
    coordinate of working-region pixel (0, 0), so tests place geometry against the real
    working region instead of assuming it starts at the window (and CRS) origin.
    """
    # The "mosaic" extends from mosaic_origin_xy covering more than the window; the window
    # upper-left lies at integer mosaic-pixel offsets (row_origin, col_origin) inside it.
    mx_origin_x, mx_origin_y = mosaic_origin_xy
    origin_x = mx_origin_x + col_origin * pixel_m
    origin_y = mx_origin_y - row_origin * pixel_m
    window_transform = Affine(pixel_m, 0, origin_x, 0, -pixel_m, origin_y)
    mosaic_transform = [pixel_m, 0.0, mx_origin_x, 0.0, -pixel_m, mx_origin_y]
    crs = pyproj.CRS.from_user_input(mars_crs_wkt())
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    (cache_dir / "ctx_windows").mkdir(parents=True, exist_ok=True)
    (cache_dir / "ctx_tiles").mkdir(parents=True, exist_ok=True)
    (cache_dir / "reprojected_detections").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_id = "SYN_000000_0000"
    ctx_tif = cache_dir / "ctx_windows" / f"{obs_id}.tif"
    mask_tif = cache_dir / "ctx_windows" / f"{obs_id}_hirise_mask.tif"

    # Synthetic CTX intensities (don't care for label tests; uint8 zero is fine).
    ctx_arr = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(
        ctx_tif, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs=crs, transform=window_transform,
    ) as dst:
        dst.write(ctx_arr, 1)

    mask = np.full((height, width), mask_fill, dtype=np.uint8)
    with rasterio.open(
        mask_tif, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs=crs, transform=window_transform, nodata=0,
    ) as dst:
        dst.write(mask, 1)

    # Mosaic sidecar -- inner_transform is what labeling._load_mosaic_transform reads.
    mosaic_tile_name = "SYN_TILE"
    (cache_dir / "ctx_tiles" / f"{mosaic_tile_name}.json").write_text(
        json.dumps({
            "murray_tile": mosaic_tile_name,
            "inner_transform": mosaic_transform,
            "inner_shape": [47_420, 47_420],   # a real Murray tile is 47420 px square
            "inner_dtype": "uint8",
        }),
        encoding="utf-8",
    )

    align = _compute_grid_alignment(
        window_transform, mosaic_transform, height, width, TILE_SIZES,
    )
    work_origin_xy = window_transform * (align["c0_win"], align["r0_win"])

    info = {
        "obs_id": obs_id,
        "cache_dir": cache_dir,
        "out_dir": out_dir,
        "ctx_tif": ctx_tif,
        "mask_tif": mask_tif,
        "window_transform": window_transform,
        "mosaic_transform": mosaic_transform,
        "crs": crs,
        "murray_tile": mosaic_tile_name,
        "align": align,
        "work_origin_xy": work_origin_xy,
    }
    return cache_dir, out_dir, info, obs_id


def _work_box(info: dict, dx: float, dy: float, half: float):
    """A square of side ``2*half`` centred ``(dx, dy)`` metres from working-region pixel (0,0).

    R78: the working region no longer starts at the CRS origin, so synthetic geometry is
    positioned relative to it rather than at absolute (0, 0).
    """
    x0, y0 = info["work_origin_xy"]
    return box(x0 + dx - half, y0 + dy - half, x0 + dx + half, y0 + dy + half)


def _write_polygons(info: dict, polygons: list[Polygon], scores=None) -> None:
    """Save a synthetic Stage 1 GeoPackage for the test ObsId."""
    if scores is None:
        scores = [0.5] * len(polygons)
    gdf = gpd.GeoDataFrame(
        {"score": scores, "id": list(range(len(polygons)))},
        geometry=polygons,
        crs=info["crs"],
    )
    out = info["cache_dir"] / "reprojected_detections" / f"{info['obs_id']}.gpkg"
    gdf.to_file(out, driver="GPKG", layer="detections")


def _make_manifest_row(info: dict):
    return pd.Series({
        "ObsId": info["obs_id"],
        "CTX_TileName": "E000_N00",   # value irrelevant; we patch murray_tile_for_manifest_row
        "CenterLat": 0.0, "CenterLon_180": 0.0, "CenterLon_360": 0.0,
    })


def _labeling_cfg(**overrides):
    cfg = {
        "grid_anchor": "ctx_pixel_origin",
        "tile_sizes_px": list(TILE_SIZES),
        "label_type": "fractional_area",
        "binary_area_threshold": 0.005,
        "binary_count_threshold": 1,
        "categorical_bins": [],
        "detection_filters": {"min_confidence": None, "min_size_m": None},
        "context_patch_px": None,
        "features": [],
    }
    cfg.update(overrides)
    return cfg


# ----------------------------------------------------------------------------
# Grid alignment
# ----------------------------------------------------------------------------

def test_alignment_aligned_window():
    """Window origin exactly at mosaic origin: range starts at 0 cells, fits perfectly."""
    transform = Affine(5, 0, 0, 0, -5, 0)
    align = _compute_grid_alignment(transform, list(transform)[:6], 128, 192, TILE_SIZES)
    # mosaic_row/col_origin should be 0
    assert align["mosaic_row_origin"] == 0
    assert align["mosaic_col_origin"] == 0
    # With S_max=64 and 128 rows, K_max_row = 128/64 - 1 = 1 -> 2 coarsest rows
    # With 192 cols, K_max_col = 192/64 - 1 = 2 -> 3 coarsest cols
    # Finest range: rows 0..15, cols 0..23
    assert align["j_min_row"] == 0
    assert align["j_max_row"] == 15
    assert align["j_min_col"] == 0
    assert align["j_max_col"] == 23
    assert (align["r1_win"] - align["r0_win"]) == 128
    assert (align["c1_win"] - align["c0_win"]) == 192


def test_alignment_offset_window():
    """Window starts at mosaic-pixel (3, 5): coarsest-aligned region starts at K=1."""
    # mosaic at (0, 0), window upper-left at mosaic-pixel (3, 5): origin_x=25, origin_y=-15
    mosaic_transform = [5.0, 0.0, 0.0, 0.0, -5.0, 0.0]
    window_transform = Affine(5, 0, 25, 0, -5, -15)
    align = _compute_grid_alignment(window_transform, mosaic_transform, 130, 130, TILE_SIZES)
    assert align["mosaic_row_origin"] == 3
    assert align["mosaic_col_origin"] == 5
    # K_min_row = ceil(3/64) = 1; K_max_row = (3+130)//64 - 1 = 1 -> 1 coarsest row
    # K_min_col = ceil(5/64) = 1; K_max_col = (5+130)//64 - 1 = 1 -> 1 coarsest col
    # Finest range: rows 8..15, cols 8..15 (8 cells per axis)
    assert align["j_min_row"] == 8 and align["j_max_row"] == 15
    assert align["j_min_col"] == 8 and align["j_max_col"] == 15


def test_alignment_raises_when_window_too_small():
    """Window smaller than one coarsest tile must raise."""
    transform = Affine(5, 0, 0, 0, -5, 0)
    with pytest.raises(ValueError, match="cannot fit"):
        _compute_grid_alignment(transform, list(transform)[:6], 50, 50, TILE_SIZES)


# ----------------------------------------------------------------------------
# Rasterization + sub-pixel area
# ----------------------------------------------------------------------------

def test_rasterize_single_boulder_sub_pixel_count(tmp_path):
    """A 5x5 m boulder (one full CTX pixel) at the working-region origin should rasterize to
    25 sub-pixels at subpixel_factor=5 -> 25 m^2 -- exactly the polygon area."""
    # R78: 128 px (not 64) because at the real mosaic phase (894, 12645) a 64-px window
    # cannot contain a single 64-px coarsest tile.
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=128, width=128)
    # Polygon covering the first CTX pixel of the working region.
    poly = _work_box(info, 2.5, -2.5, 2.5)
    _write_polygons(info, [poly])
    gdf = gpd.read_file(
        cache_dir / "reprojected_detections" / f"{obs}.gpkg", layer="detections",
    )
    align = info["align"]
    raster = _rasterize_boulders_subpixel(gdf, info["window_transform"], align, subpixel_factor=5)
    # Sub-pixel size is 1 m; a 5 m x 5 m polygon should fill 25 sub-pixels.
    assert int(raster.sum()) == 25
    # And it must land in the raster's top-left cell -- i.e. the sub-pixel transform is
    # offset by the working-region origin, not by the window (or CRS) origin.
    assert int(raster[:5, :5].sum()) == 25


def test_rasterize_returns_zero_for_empty_gdf(tmp_path):
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=128, width=128)
    gdf = gpd.GeoDataFrame({"score": []}, geometry=[], crs=info["crs"])
    align = info["align"]
    raster = _rasterize_boulders_subpixel(gdf, info["window_transform"], align, subpixel_factor=5)
    assert raster.sum() == 0


# ----------------------------------------------------------------------------
# Centroid counting
# ----------------------------------------------------------------------------

def test_count_centroids_per_finest_cell_assigns_to_owner(tmp_path):
    # R78: 128 px and geometry placed relative to the working-region origin, which at the
    # real mosaic phase (894, 12645) is window pixel (2, 27), not (0, 0).
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=128, width=128)
    # Three boulders, offsets measured from working-region pixel (0, 0):
    #   (20, -20) m  -> local finest cell (0, 0)  (S_min=8 -> 40 m cells)
    #   (60, -60) m  -> local finest cell (1, 1)
    #   (50, -50) m  -> local finest cell (1, 1)
    polys = [
        _work_box(info, 20, -20, 2),
        _work_box(info, 60, -60, 2),
        _work_box(info, 50, -50, 2),
    ]
    _write_polygons(info, polys)
    gdf = gpd.read_file(
        cache_dir / "reprojected_detections" / f"{obs}.gpkg", layer="detections",
    )
    align = info["align"]
    # The absolute finest-cell indices these land in are J_MIN_ROW/J_MIN_COL-based, so the
    # j_min subtraction inside the counter is exercised (it was a no-op at phase 0).
    assert (align["j_min_row"], align["j_min_col"]) == (J_MIN_ROW, J_MIN_COL)
    counts = _count_centroids_per_finest_cell(
        gdf, info["mosaic_transform"], align, finest_size_px=8, px_x=5.0, px_y=5.0,
    )
    # Cell (0, 0) gets 1, cell (1, 1) gets 2, every other cell gets 0
    assert counts[0, 0] == 1
    assert counts[1, 1] == 2
    assert counts.sum() == 3


# ----------------------------------------------------------------------------
# Nested grid -- the core consistency requirement
# ----------------------------------------------------------------------------

def test_sum_up_ladder_preserves_total_area_and_count():
    """For an all-eligible grid, summing finest stats up the ladder must preserve totals."""
    rng = np.random.default_rng(0)
    n = 16  # divisible by 8, the ratio of S_max/S_min
    finest = {
        "boulder_area": rng.random((n, n)) * 100.0,
        "boulder_count": rng.integers(0, 5, size=(n, n)).astype(np.int64),
        "eligible": np.ones((n, n), dtype=bool),
    }
    # R78: real non-zero grid phase (see J_MIN_ROW/J_MIN_COL), not the (0, 0) no case has.
    scales = _sum_up_ladder(
        finest, TILE_SIZES, px_x=5.0, px_y=5.0, j_min_row=J_MIN_ROW, j_min_col=J_MIN_COL,
    )
    base_area = float(finest["boulder_area"].sum())
    base_count = int(finest["boulder_count"].sum())
    for sc in scales:
        # All cells eligible -> all-eligible at every scale
        assert sc["eligible"].all()
        assert float(sc["boulder_area"].sum()) == pytest.approx(base_area, rel=1e-10)
        assert int(sc["boulder_count"].sum()) == base_count
    # The absolute index offset must be rescaled by the ladder ratio at each scale --
    # at phase 0 every one of these was 0 and the rescaling was untested.
    # ratios 1, 2, 4, 8 -> 112/1584, 56/792, 28/396, 14/198
    assert [(sc["j_min_row"], sc["j_min_col"]) for sc in scales] == [
        (112, 1584), (56, 792), (28, 396), (14, 198),
    ]


def test_sum_up_ladder_coarse_ineligible_if_any_subtile_ineligible():
    """A single ineligible finest tile must drop the containing 16-, 32-, and 64-px tile."""
    n = 8
    finest = {
        "boulder_area": np.zeros((n, n)),
        "boulder_count": np.zeros((n, n), dtype=np.int64),
        "eligible": np.ones((n, n), dtype=bool),
    }
    finest["eligible"][0, 0] = False
    # R78: real non-zero grid phase, not (0, 0).
    scales = _sum_up_ladder(
        finest, TILE_SIZES, px_x=5.0, px_y=5.0, j_min_row=J_MIN_ROW, j_min_col=J_MIN_COL,
    )
    # At every coarser scale, the (0, 0) coarse tile must be ineligible.
    for sc in scales[1:]:
        assert not sc["eligible"][0, 0]
    # The coarsest scale here is 1x1 (n=8 finest -> 8/8=1 at S=64), so this catches everything.
    assert scales[-1]["boulder_area"].shape == (1, 1)
    assert not scales[-1]["eligible"][0, 0]


def test_nested_consistency_matches_direct_coarse_compute():
    """The reshape-and-sum path must equal a hand-rolled coarse aggregation."""
    n = 16
    area = np.arange(n * n, dtype=np.float64).reshape(n, n)
    count = np.arange(n * n, dtype=np.int64).reshape(n, n) % 7
    finest = {
        "boulder_area": area,
        "boulder_count": count,
        "eligible": np.ones((n, n), dtype=bool),
    }
    # R78: real non-zero grid phase, not (0, 0).
    scales = _sum_up_ladder(
        finest, TILE_SIZES, px_x=5.0, px_y=5.0, j_min_row=J_MIN_ROW, j_min_col=J_MIN_COL,
    )
    # S=16 (scale 1): 2x2 sums of finest
    expected_16 = area.reshape(8, 2, 8, 2).sum(axis=(1, 3))
    assert np.array_equal(scales[1]["boulder_area"], expected_16)
    expected_count_16 = count.reshape(8, 2, 8, 2).sum(axis=(1, 3))
    assert np.array_equal(scales[1]["boulder_count"], expected_count_16)


# ----------------------------------------------------------------------------
# Mask gating
# ----------------------------------------------------------------------------

def test_mask_gating_drops_tiles_with_any_uncovered_pixel(tmp_path):
    """Even a single mask=0 pixel inside a finest tile drops the tile and every coarse tile
    that contains it."""
    # R78: 192 px at the real mosaic phase (894, 12645) reproduces the old 2x2-coarsest /
    # 16x16-finest working region; it starts at window pixel (r0_win, c0_win) = (2, 27), so
    # the mask hole must be punched relative to it, not at window (0, 0).
    cache_dir, out_dir, info, obs = _make_window(
        tmp_path, height=192, width=192, mask_fill=1,
    )
    align = info["align"]
    # Punch a single mask=0 pixel into the working region's first finest tile.
    mask_tif = info["mask_tif"]
    with rasterio.open(mask_tif, "r+") as src:
        mask = src.read(1)
        mask[align["r0_win"] + 3, align["c0_win"] + 3] = 0
        src.write(mask, 1)
    _write_polygons(info, [])
    row = _make_manifest_row(info)

    from unittest.mock import patch

    with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
        prov = stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(),
            labeling_cfg=_labeling_cfg(),
            config_hash="test",
            apply_coreg_shift=False,
        )
    df = pd.read_parquet(out_dir / LABELS_SUBDIR / f"{obs}.parquet")
    # The working region's first finest tile must be missing at every scale (the
    # ineligibility propagates upward through the nested grid). R78: at the real phase the
    # index is J_MIN_ROW/J_MIN_COL rescaled per scale, not (0, 0) -- so this now also pins
    # the absolute-index arithmetic.
    for s in TILE_SIZES:
        scaled = df[df["tile_size_px"] == s]
        ratio = s // TILE_SIZES[0]
        ti0, tj0 = J_MIN_ROW // ratio, J_MIN_COL // ratio
        assert not ((scaled["ti"] == ti0) & (scaled["tj"] == tj0)).any(), (
            f"tile_size_px={s}: ({ti0},{tj0}) tile should have been dropped"
        )
        # ... and the surviving tiles must actually be indexed from that offset, so a
        # mutant that emits 0-based indices fails here rather than passing vacuously.
        assert int(scaled["ti"].min()) == ti0 and int(scaled["tj"].min()) == tj0


# ----------------------------------------------------------------------------
# Coreg shift application
# ----------------------------------------------------------------------------

def test_apply_coreg_shift_translates_polygons():
    crs = pyproj.CRS.from_epsg(4326)
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=crs)
    shift = {"shift_m": {"dx": 10.0, "dy": -5.0}}
    out = _apply_coreg_shift(gdf, shift)
    xmin, ymin, xmax, ymax = out.total_bounds
    assert xmin == pytest.approx(10.0)
    assert xmax == pytest.approx(11.0)
    assert ymin == pytest.approx(-5.0)
    assert ymax == pytest.approx(-4.0)


def test_apply_coreg_shift_no_op_on_none():
    crs = pyproj.CRS.from_epsg(4326)
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=crs)
    out = _apply_coreg_shift(gdf, None)
    assert out.total_bounds[0] == 0.0
    assert out.total_bounds[2] == 1.0


# ----------------------------------------------------------------------------
# Detection filters
# ----------------------------------------------------------------------------

def test_detection_filter_min_confidence_drops_low_scores():
    crs = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)
    gdf = gpd.GeoDataFrame(
        {"score": [0.1, 0.4, 0.9]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
        crs=crs,
    )
    out = _apply_detection_filters(gdf, {"min_confidence": 0.3, "min_size_m": None})
    assert len(out) == 2
    assert set(out["score"]) == {0.4, 0.9}


def test_detection_filter_min_size_is_an_equivalent_circle_DIAMETER_in_metres():
    """R80. The old fixture (areas 1/100/1000 in EPSG:4326, threshold 5.0) could not
    distinguish diameter from radius, or from three other plausible size metrics, and
    measured area in degrees². This one separates five interpretations at once:

    | id | shape           | area  | diameter | radius  | max bbox side | perim/pi |
    |----|-----------------|-------|----------|---------|---------------|----------|
    | 0  | 2 x 2 box       |     4 |   2.2568 |  1.1284 |             2 |    2.546 |
    | 1  | 0.5 x 10 box    |     5 |   2.5231 |  1.2616 |            10 |    6.685 |
    | 2  | 5 x 5 box       |    25 |   5.6419 |  2.8209 |             5 |    6.366 |
    | 3  | 20 x 20 box     |   400 |  22.5676 | 11.2838 |            20 |   25.465 |
    | 4  | 4.5 x 4.5 box   | 20.25 |   5.0771 |  2.5386 |           4.5 |    5.730 |
    | 5  | concave L       |    16 |   4.5135 |  2.2568 |             5 |    5.729 |

    At a 5.0 m floor: diameter keeps {2,3,4}; radius keeps {3}; max-bbox-side keeps
    {1,2,3,5}; perimeter/pi keeps {1,2,3,4,5}; area-as-size keeps {1,2,3,4,5}; sqrt(area)
    (equivalent-square side) keeps {2,3} — id 4 is what separates it from diameter.

    **id 5 is not an axis-aligned rectangle, and that is load-bearing.** For a
    `shapely.box`, polygon area == envelope area == convex-hull area, so a fixture made
    only of boxes cannot tell `geometry.area` from `geometry.envelope.area` — and that
    substitution survives the whole suite otherwise. It matters on real data: for
    `ESP_046328_2180` (138,373 polygons, none axis-aligned) the median bbox/polygon area
    ratio is 1.2673, and measuring bbox area would silently retain 3,466 of the 6,360
    polygons the production floor drops. The L is a 5x5 square with a 3x3 corner removed,
    so its three areas straddle the floor: polygon 16 (d=4.51, **dropped**), convex hull
    20.5 (d=5.11, kept), bounding box 25 (d=5.64, kept).
    """
    crs = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)
    assert crs.is_projected and crs.axis_info[0].unit_name == "metre"
    ell = Polygon([(400, 0), (405, 0), (405, 2), (402, 2), (402, 5), (400, 5)])
    gdf = gpd.GeoDataFrame(
        {"score": [0.5] * 6, "id": [0, 1, 2, 3, 4, 5]},
        geometry=[
            box(0, 0, 2, 2),            # A=4
            box(100, 0, 100.5, 10),     # A=5   -- long and thin: bbox/perimeter trap
            box(200, 0, 205, 5),        # A=25
            box(300, 0, 320, 20),       # A=400
            box(350, 0, 354.5, 4.5),    # A=20.25 -- separates sqrt(area) from diameter
            ell,                        # A=16   -- separates polygon from hull and bbox
        ],
        crs=crs,
    )
    assert list(gdf.geometry.area) == pytest.approx([4.0, 5.0, 25.0, 400.0, 20.25, 16.0])
    assert gdf.geometry.convex_hull.area.iloc[5] == pytest.approx(20.5)
    assert gdf.geometry.envelope.area.iloc[5] == pytest.approx(25.0)
    # The fixture must contain at least one shape whose bbox/hull area differ from its own,
    # or the polygon-area interpretation silently stops being pinned.
    assert not np.allclose(gdf.geometry.envelope.area, gdf.geometry.area)
    assert not np.allclose(gdf.geometry.convex_hull.area, gdf.geometry.envelope.area)
    with warnings.catch_warnings():
        # A geographic CRS would emit "Geometry is in a geographic CRS" here. Making that
        # an error is what pins "measured in metres" -- it is the R80 units hole itself.
        warnings.simplefilter("error", UserWarning)
        out = _apply_detection_filters(gdf, {"min_confidence": None, "min_size_m": 5.0})
    assert list(out["id"]) == [2, 3, 4]


def test_detection_filter_size_floor_is_inclusive():
    """`>=`, not `>`. Derived with the production expression so the comparison is
    bit-exact rather than approximately-at-the-boundary."""
    crs = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)
    gdf = gpd.GeoDataFrame(
        {"score": [0.5], "id": [0]}, geometry=[box(0, 0, 5, 5)], crs=crs,
    )
    exact = float(2.0 * np.sqrt(gdf.geometry.area.to_numpy()[0] / np.pi))
    out = _apply_detection_filters(gdf, {"min_confidence": None, "min_size_m": exact})
    assert len(out) == 1, "a polygon exactly at the floor must be kept"


def test_detection_filter_confidence_floor_is_inclusive():
    crs = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)
    gdf = gpd.GeoDataFrame(
        {"score": [0.29, 0.30, 0.31], "id": [0, 1, 2]},
        geometry=[box(i * 10, 0, i * 10 + 5, 5) for i in range(3)],
        crs=crs,
    )
    out = _apply_detection_filters(gdf, {"min_confidence": 0.30, "min_size_m": None})
    assert list(out["id"]) == [1, 2], "a score exactly at the floor must be kept"


def test_size_floor_realised_in_the_ctx_frame_is_looser_than_the_configured_metres():
    """R80/R03 characterisation — the executable half of "retain and document".

    The floor is applied to polygons already reprojected into the clon_0 CTX frame, where
    easting is stretched by 1/cos(phi) relative to the source frame's own standard
    parallel. So the realised *physical* floor is `min_size_m * sqrt(cos phi)`, looser than
    the configured metres, and it differs image to image. Measured over the 39 v2 images
    the realised floor spans 0.993-1.367 m against a configured 1.4105 m.

    If someone later moves the filter before the reprojection, or divides by the scale
    inside it, this test fails -- which is correct: that would redefine the target rather
    than document it.
    """
    phi = 60.0
    src_crs = pyproj.CRS.from_user_input(mars_crs_wkt(phi))
    ctx_crs = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)   # standard parallel 0

    # A rock sized to straddle the floor: side 4 m -> d_src = 4.51 m (below a 5 m floor),
    # d_ctx = 4.51 * 1/sqrt(cos 60) = 6.38 m (above it). The window is (3.13, 4.43) m.
    rock_src = gpd.GeoDataFrame(
        {"score": [0.9], "id": [0]}, geometry=[box(0, 0, 4.0, 4.0)], crs=src_crs,
    )
    rock_ctx = rock_src.to_crs(ctx_crs)

    d_src = float(2.0 * np.sqrt(rock_src.geometry.area.iloc[0] / np.pi))
    d_ctx = float(2.0 * np.sqrt(rock_ctx.geometry.area.iloc[0] / np.pi))
    # Area inflates by 1/cos(phi), so the equivalent-circle diameter by 1/sqrt(cos phi).
    assert d_ctx / d_src == pytest.approx(1.0 / np.sqrt(np.cos(np.radians(phi))), rel=2e-3)

    floor = 5.0
    assert len(_apply_detection_filters(rock_src, {"min_confidence": None,
                                                   "min_size_m": floor})) == 0
    assert len(_apply_detection_filters(rock_ctx, {"min_confidence": None,
                                                   "min_size_m": floor})) == 1, (
        "the same rock clears the floor once reprojected -- this is the R03 mechanism"
    )


# ----------------------------------------------------------------------------
# Label transforms + tile bounds
# ----------------------------------------------------------------------------

def test_label_transforms_emit_expected_columns(tmp_path):
    # R78: 192 px at the real mosaic phase (894, 12645) -> 16x16 finest / 2x2 coarsest.
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=192, width=192)
    # Single 25 m^2 boulder in the working region's first finest tile.
    poly = _work_box(info, 2.5, -2.5, 2.5)
    _write_polygons(info, [poly], scores=[0.7])
    row = _make_manifest_row(info)

    from unittest.mock import patch
    with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
        stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(),
            labeling_cfg=_labeling_cfg(binary_area_threshold=0.0001, binary_count_threshold=1),
            config_hash="test",
            apply_coreg_shift=False,
        )
    df = pd.read_parquet(out_dir / LABELS_SUBDIR / f"{obs}.parquet")
    for col in [
        "obs_id", "scale_idx", "tile_size_px", "tile_size_m", "ti", "tj",
        "xmin", "ymin", "xmax", "ymax",
        "boulder_area", "boulder_count", "tile_area",
        "fractional_area", "binary_by_area", "binary_by_count", "count_density",
        "config_hash",
    ]:
        assert col in df.columns, f"missing column {col!r}"

    # fractional_area must be in [0, 1] and tile_area > 0
    assert (df["fractional_area"] >= 0).all() and (df["fractional_area"] <= 1).all()
    assert (df["tile_area"] > 0).all()
    assert (df["boulder_area"] >= 0).all()
    assert (df["boulder_count"] >= 0).all()

    # The working region's first finest tile -- absolute index (J_MIN_ROW, J_MIN_COL) at the
    # real mosaic phase, R78 -- must have boulder_area = 25 m^2, count = 1, frac = 25/1600.
    finest_00 = df[
        (df["tile_size_px"] == 8) & (df["ti"] == J_MIN_ROW) & (df["tj"] == J_MIN_COL)
    ]
    assert len(finest_00) == 1
    assert finest_00["boulder_area"].iloc[0] == pytest.approx(25.0)
    assert int(finest_00["boulder_count"].iloc[0]) == 1
    assert finest_00["fractional_area"].iloc[0] == pytest.approx(25.0 / 1600.0)
    assert bool(finest_00["binary_by_area"].iloc[0])
    assert bool(finest_00["binary_by_count"].iloc[0])
    # Its emitted world bounds must be the mosaic-anchored ones for that absolute index.
    mx_origin_x, mx_origin_y = info["mosaic_transform"][2], info["mosaic_transform"][5]
    assert finest_00["xmin"].iloc[0] == pytest.approx(mx_origin_x + J_MIN_COL * 8 * 5.0)
    assert finest_00["ymax"].iloc[0] == pytest.approx(mx_origin_y - J_MIN_ROW * 8 * 5.0)


def test_tile_bounds_align_with_mosaic_pixel_grid(tmp_path):
    """Tile bounds (xmin, xmax, ymin, ymax) must lie on integer multiples of (S * px) from the
    mosaic origin -- the definition of 'anchored to CTX pixel origin'."""
    # R78: 192 px at the real mosaic phase (894, 12645) with a real Murray-tile CRS origin.
    # This is the test the review found could not detect the failure its own docstring names:
    # with mx_origin_x = mx_origin_y = 0 the assertion below was satisfied identically by a
    # grid anchored to the *CRS* origin, so dropping the mosaic origin from the bounds
    # (which displaces real ymin by 2,608 km) left the suite green.
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=192, width=192)
    _write_polygons(info, [])
    row = _make_manifest_row(info)

    from unittest.mock import patch
    with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
        stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(),
            labeling_cfg=_labeling_cfg(),
            config_hash="test",
            apply_coreg_shift=False,
        )
    df = pd.read_parquet(out_dir / LABELS_SUBDIR / f"{obs}.parquet")
    assert len(df) > 0
    # Pixel size = 5 m. Every tile bound must be an integer multiple of (tile_size_px * 5 m)
    # *from the mosaic origin* -- measure the offset from mx_origin, not from (0, 0).
    mx_origin_x, mx_origin_y = info["mosaic_transform"][2], info["mosaic_transform"][5]
    # Guard the guard: the mosaic origin is not itself on the coarsest tile lattice, so
    # anchoring to the CRS origin instead would be detected.
    assert not np.isclose(np.mod(mx_origin_x, TILE_SIZES[-1] * 5.0), 0, atol=1e-6)
    assert not np.isclose(np.mod(mx_origin_y, TILE_SIZES[-1] * 5.0), 0, atol=1e-6)
    for s in TILE_SIZES:
        sub = df[df["tile_size_px"] == s]
        assert len(sub) > 0
        step = s * 5.0
        for col, origin in (("xmin", mx_origin_x), ("xmax", mx_origin_x),
                            ("ymin", mx_origin_y), ("ymax", mx_origin_y)):
            # Offset from the mosaic origin, in tiles: must be a whole number of tiles.
            # (`np.mod` is unusable here -- a 1e-11 negative float error wraps to ~step.)
            offsets = (sub[col].to_numpy() - origin) / step
            resid = np.abs(offsets - np.round(offsets)) * step
            assert (resid < 1e-6).all(), (
                f"{col} not aligned at scale {s}: max residual {resid.max():.6g} m"
            )


# ----------------------------------------------------------------------------
# Idempotency
# ----------------------------------------------------------------------------

def test_stage4_sidecar_binds_to_the_exact_upstream_bytes(tmp_path):
    """R74. Stage 4's `mask_min == 1` rule means the coverage mask decides which tiles exist
    at all, so a pre-R74 and a post-R74 label table differ in *row set* while sharing a
    config hash — the Pattern-D provenance failure PENDING_REBUILD.md exists to control.
    The sidecar must therefore record what it actually read, by content.
    """
    from src.ctx_retrieve import CTX_WINDOWS_SUBDIR, file_sha256

    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=192, width=192)
    _write_polygons(info, [])
    row = _make_manifest_row(info)
    window_tif = cache_dir / CTX_WINDOWS_SUBDIR / f"{obs}.tif"
    mask_tif = cache_dir / CTX_WINDOWS_SUBDIR / f"{obs}_hirise_mask.tif"

    from unittest.mock import patch

    def _run():
        with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
            return stage4_one_image(
                obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
                target_crs=info["crs"].to_wkt(), labeling_cfg=_labeling_cfg(),
                config_hash="hash_v1", apply_coreg_shift=False,
            )

    prov = _run()
    inputs = prov["inputs"]
    assert inputs["ctx_window_sha256"] == file_sha256(window_tif)
    assert inputs["hirise_mask_sha256"] == file_sha256(mask_tif)

    # Now change the mask the way an algorithm change would: re-mark one interior pixel as
    # covered. The config hash is untouched; the recorded identity must move anyway.
    with rasterio.open(mask_tif) as src:
        prof, mask = src.profile, src.read(1)
    mask[mask.shape[0] // 2, mask.shape[1] // 2] ^= 1
    with rasterio.open(mask_tif, "w", **prof) as dst:
        dst.write(mask, 1)

    prov2 = _run()
    assert prov2["config_hash"] == prov["config_hash"], "fixture drifted: config must be equal"
    assert prov2["inputs"]["hirise_mask_sha256"] != inputs["hirise_mask_sha256"], (
        "the sidecar cannot distinguish two coverage-mask generations — recording only "
        "pathnames and a YAML config hash is exactly the R74 provenance gap"
    )


def test_stage4_is_idempotent(tmp_path):
    # R78: 192 px at the real mosaic phase; polygons placed in the working region.
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=192, width=192)
    polys = [_work_box(info, 25, -15, 5), _work_box(info, 87.5, -72.5, 7.5)]
    _write_polygons(info, polys, scores=[0.6, 0.8])
    row = _make_manifest_row(info)

    from unittest.mock import patch
    with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
        stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(),
            labeling_cfg=_labeling_cfg(),
            config_hash="hash_v1",
            apply_coreg_shift=False,
        )
        df1 = pd.read_parquet(out_dir / LABELS_SUBDIR / f"{obs}.parquet")
        stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(),
            labeling_cfg=_labeling_cfg(),
            config_hash="hash_v1",
            apply_coreg_shift=False,
        )
        df2 = pd.read_parquet(out_dir / LABELS_SUBDIR / f"{obs}.parquet")

    # Numeric columns: exact match; bool/int also exact.
    for col in df1.columns:
        if df1[col].dtype.kind in ("f", "i", "b", "u"):
            assert np.array_equal(df1[col].to_numpy(), df2[col].to_numpy(), equal_nan=True), (
                f"column {col} not idempotent"
            )
        else:
            assert (df1[col].to_numpy() == df2[col].to_numpy()).all(), (
                f"column {col} not idempotent"
            )


# ----------------------------------------------------------------------------
# Empty-shapefile case (ESP_065711_1545 analog)
# ----------------------------------------------------------------------------

def test_stage4_handles_empty_polygons(tmp_path):
    """An ObsId with zero detections should emit all-eligible-tile rows with zero stats."""
    # R78: 192 px at the real mosaic phase (894, 12645), not the (0, 0) no image has.
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=192, width=192)
    _write_polygons(info, [])
    row = _make_manifest_row(info)

    from unittest.mock import patch
    with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
        prov = stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(),
            labeling_cfg=_labeling_cfg(),
            config_hash="test",
            apply_coreg_shift=False,
        )
    df = pd.read_parquet(out_dir / LABELS_SUBDIR / f"{obs}.parquet")
    assert len(df) > 0
    assert (df["boulder_area"] == 0).all()
    assert (df["boulder_count"] == 0).all()
    assert (df["fractional_area"] == 0).all()
    assert prov["n_polygons_after_filter"] == 0


# ----------------------------------------------------------------------------
# Stage 4 integration on ESP_069669_2220 (slow)
# ----------------------------------------------------------------------------

OBS_ID = "ESP_069669_2220"


def _stage2_and_3_ready(cache_dir: Path) -> bool:
    return (
        (cache_dir / "ctx_windows" / f"{OBS_ID}.tif").exists()
        and (cache_dir / "ctx_windows" / f"{OBS_ID}_hirise_mask.tif").exists()
        and (cache_dir / "reprojected_detections" / f"{OBS_ID}.gpkg").exists()
    )


@pytest.mark.slow
def test_stage4_runs_on_ESP_069669_2220(cfg, tmp_path):
    # R77: writes MUST go to tmp_path, never cfg.output_dir. On 2026-08-04 this test
    # overwrote the live gitignored dataset/labels/ESP_069669_2220.{parquet,json} and,
    # because the v1 tree predates the 2026-06-10 y-sign fix, migrated one of nine
    # images across a correctness boundary. git cannot restore these paths.
    cache_dir = cfg.cache_dir
    if not _stage2_and_3_ready(cache_dir):
        pytest.skip(f"{OBS_ID}: Stage 2 caches missing")

    from src import manifest as M

    df_manifest = M.load_manifest(cfg.manifest_path)
    row = df_manifest.set_index("ObsId").loc[OBS_ID]

    out_dir = tmp_path / "dataset"
    prov = stage4_one_image(
        OBS_ID,
        cache_dir=cache_dir,
        output_dir=out_dir,
        manifest_row=row,
        target_crs=cfg["target_crs"],
        labeling_cfg=cfg["labeling"],
        config_hash=cfg.hash,
        apply_coreg_shift=True,
    )
    df = pd.read_parquet(out_dir / LABELS_SUBDIR / f"{OBS_ID}.parquet")
    assert len(df) > 0
    # Every tile size must appear if any tiles are eligible.
    sizes_present = set(int(s) for s in df["tile_size_px"].unique())
    assert sizes_present.issubset({8, 16, 32, 64})
    assert len(sizes_present) >= 1
    # fractional_area in [0, 1].
    assert (df["fractional_area"] >= 0).all() and (df["fractional_area"] <= 1).all()
    # tile_area: tile_size_px**2 * px_x * px_y. Murray Lab CTX pixels are 4.99997 m
    # (DECISIONS.md 2026-05-21), not exactly 5 m.
    with rasterio.open(cache_dir / "ctx_windows" / f"{OBS_ID}.tif") as src:
        px_x = abs(src.transform.a)
        px_y = abs(src.transform.e)
    for s in sizes_present:
        sub = df[df["tile_size_px"] == s]
        expected_area = s * s * px_x * px_y
        assert np.allclose(sub["tile_area"].to_numpy(), expected_area, rtol=1e-9)

    # Stage 3 shift was applied -> provenance records it.
    assert prov["coreg_shift_applied"] is True
    assert prov["coreg_shift_m"] is not None


@pytest.mark.slow
def test_stage4_nested_consistency_on_real_data(cfg):
    """Aggregate finest-scale base stats up the ladder by sibling-quartet and compare
    against the coarser-scale rows the pipeline emitted.

    A coarse tile (ti, tj) at scale S aggregates the four finest tiles at scale S/2
    with indices (2*ti, 2*tj), (2*ti, 2*tj+1), (2*ti+1, 2*tj), (2*ti+1, 2*tj+1).
    Only coarse tiles where all four siblings are present in the finest table should
    be compared; partial-coverage coarse tiles are by construction missing.
    """
    cache_dir = cfg.cache_dir
    if not _stage2_and_3_ready(cache_dir):
        pytest.skip(f"{OBS_ID}: Stage 2 caches missing")

    parquet = cfg.output_dir / LABELS_SUBDIR / f"{OBS_ID}.parquet"
    if not parquet.exists():
        pytest.skip(f"{OBS_ID}: Stage 4 output missing -- run scripts/run_stage4.py first")

    df = pd.read_parquet(parquet)
    sizes = sorted(int(s) for s in df["tile_size_px"].unique())

    for fine_s, coarse_s in zip(sizes, sizes[1:]):
        if coarse_s != fine_s * 2:
            continue
        fine = df[df["tile_size_px"] == fine_s].set_index(["ti", "tj"])
        coarse = df[df["tile_size_px"] == coarse_s]
        mismatches = 0
        compared = 0
        for _, row in coarse.iterrows():
            ti, tj = int(row["ti"]), int(row["tj"])
            sib_idx = [(2 * ti, 2 * tj), (2 * ti, 2 * tj + 1),
                       (2 * ti + 1, 2 * tj), (2 * ti + 1, 2 * tj + 1)]
            sub_rows = [fine.loc[idx] for idx in sib_idx if idx in fine.index]
            if len(sub_rows) != 4:
                # Coarse tile only emitted if all 4 sub-tiles eligible -- ensure that.
                # (If a sibling is missing here, eligibility math is wrong.)
                mismatches += 1
                continue
            sum_area = sum(float(r["boulder_area"]) for r in sub_rows)
            sum_count = sum(int(r["boulder_count"]) for r in sub_rows)
            if not np.isclose(sum_area, float(row["boulder_area"]), rtol=0, atol=1e-9):
                mismatches += 1
            if sum_count != int(row["boulder_count"]):
                mismatches += 1
            compared += 1
        assert compared > 0, f"no comparable {coarse_s}-px tiles found"
        assert mismatches == 0, (
            f"nested consistency broke for {fine_s}->{coarse_s} px on {compared} tiles "
            f"({mismatches} mismatches)"
        )


# ----------------------------------------------------------------------------
# R29 / R75 — the coverage mask must move with the polygons.
# ----------------------------------------------------------------------------
from src.labeling import _shift_coverage_mask


def _shift_dict(dx, dy):
    return {"shift_m": {"dx": dx, "dy": dy, "magnitude": (dx ** 2 + dy ** 2) ** 0.5}}


def test_mask_shift_is_a_noop_without_a_stage3_shift():
    m = np.ones((6, 6), dtype=np.uint8)
    out, prov = _shift_coverage_mask(m, None, 5.0, 5.0)
    assert out is m
    assert prov["applied"] is False


def test_mask_shift_moves_north_as_decreasing_row():
    """+dy is northward, and north is a SMALLER row index in a north-up raster.

    This is the assertion that pins the sign. Flip it and the mask moves the wrong way,
    doubling the misalignment instead of removing it.
    """
    m = np.zeros((6, 6), dtype=np.uint8)
    m[3:5, 1:3] = 1                                  # a 2x2 block at rows 3-4
    out, prov = _shift_coverage_mask(m, _shift_dict(0.0, 10.0), 5.0, 5.0)  # 10 m north
    assert prov["shift_px"] == {"drow": -2, "dcol": 0}
    assert out[1:3, 1:3].all()                       # moved up two rows
    assert not out[3:5, 1:3].any()                   # vacated
    assert out.sum() == m.sum()                      # nothing lost off-array here


def test_mask_shift_moves_east_as_increasing_column():
    m = np.zeros((6, 6), dtype=np.uint8)
    m[1:3, 0:2] = 1
    out, prov = _shift_coverage_mask(m, _shift_dict(15.0, 0.0), 5.0, 5.0)  # 15 m east
    assert prov["shift_px"] == {"drow": 0, "dcol": 3}
    assert out[1:3, 3:5].all()
    assert not out[1:3, 0:2].any()


def test_mask_shift_fills_vacated_area_as_INELIGIBLE():
    """The whole point: vacated area must not stay eligible."""
    m = np.ones((8, 8), dtype=np.uint8)
    out, _ = _shift_coverage_mask(m, _shift_dict(0.0, 10.0), 5.0, 5.0)
    assert out[-2:, :].sum() == 0                    # southern strip vacated -> 0
    assert out[:-2, :].all()                         # the rest survives
    assert out.sum() < m.sum()


def test_mask_shift_rounds_subpixel_shifts_and_reports_the_residual():
    """Real shifts are quantised to 1/20 px, not whole px (measured 0/39 integer)."""
    m = np.ones((10, 10), dtype=np.uint8)
    _, prov = _shift_coverage_mask(m, _shift_dict(0.0, 182.999), 5.0, 5.0)
    assert prov["shift_px"]["drow"] == -37           # 182.999/5 = 36.6 -> 37 rows north
    assert abs(prov["residual_m"]["dy"]) <= 2.5 + 1e-9
    assert prov["applied"] is True


def test_mask_shift_records_the_eligibility_it_removed():
    m = np.ones((10, 10), dtype=np.uint8)
    out, prov = _shift_coverage_mask(m, _shift_dict(5.0, 10.0), 5.0, 5.0)
    assert prov["n_eligible_px_before"] == 100
    assert prov["n_eligible_px_after"] == int((out == 1).sum())
    assert prov["n_eligible_px_after"] < prov["n_eligible_px_before"]


def test_mask_shift_larger_than_the_array_empties_it():
    m = np.ones((4, 4), dtype=np.uint8)
    out, prov = _shift_coverage_mask(m, _shift_dict(0.0, 1000.0), 5.0, 5.0)
    assert out.sum() == 0
    assert prov["n_eligible_px_after"] == 0


# ----------------------------------------------------------------------------
# R68 — the window origin must sit on the parent mosaic's integer pixel lattice.
# The guard this replaces compared the window's pixel size against the mosaic's,
# which Stage 2 makes bit-identical by construction, so it could never fire.
# ----------------------------------------------------------------------------
from src.labeling import GRID_PHASE_TOL_PX

_MOSAIC = Affine(5.0, 0.0, -474197.5, 0.0, -5.0, 2133889.0)
_LADDER = [8, 16, 32, 64]


def _win(dcol, drow):
    """Window transform offset from the mosaic origin by (dcol, drow) mosaic pixels."""
    return _MOSAIC * Affine.translation(dcol, drow)


def test_grid_alignment_accepts_an_integer_offset_window():
    align = _compute_grid_alignment(_win(1280, 640), list(_MOSAIC)[:6], 512, 512, _LADDER)
    assert align["mosaic_col_origin"] == 1280
    assert align["mosaic_row_origin"] == 640


def test_grid_alignment_tolerates_float_noise_but_not_a_real_phase():
    """Pins the tolerance from BOTH sides.

    The real cohort's worst phase residual is 1.38e-10 px -- non-zero -- so a mutant that
    demands an exact 0 would break on live data. And a genuine sub-pixel phase must be
    rejected. A fixture that only exercises an exactly-zero residual kills neither mutant.
    """
    px = 5.0
    # ~2e-9 px of noise: three orders below the 1e-6 px tolerance, above bit-zero.
    noisy = Affine(px, 0.0, _MOSAIC.c + 1280 * px + 1e-8,
                   0.0, -px, _MOSAIC.f - 640 * px - 1e-8)
    col_f = (noisy.c - _MOSAIC.c) / px
    assert 0 < abs(col_f - round(col_f)) < GRID_PHASE_TOL_PX   # genuinely non-zero
    align = _compute_grid_alignment(noisy, list(_MOSAIC)[:6], 512, 512, _LADDER)
    assert (align["mosaic_col_origin"], align["mosaic_row_origin"]) == (1280, 640)


@pytest.mark.parametrize("dcol,drow", [(0.5, 0.0), (0.0, 0.5), (0.25, 0.25), (0.0, 0.2)])
def test_grid_alignment_rejects_a_fractional_phase(dcol, drow):
    with pytest.raises(RuntimeError, match="NOT on its parent mosaic's pixel lattice"):
        _compute_grid_alignment(_win(1280 + dcol, 640 + drow), list(_MOSAIC)[:6],
                                512, 512, _LADDER)


def test_grid_alignment_reports_the_phase_in_both_px_and_m():
    with pytest.raises(RuntimeError) as ei:
        _compute_grid_alignment(_win(1280.5, 640.0), list(_MOSAIC)[:6], 512, 512, _LADDER)
    msg = str(ei.value)
    assert "0.5" in msg and "2.5" in msg      # 0.5 px = 2.5 m at 5 m/px


def test_grid_alignment_rejects_a_pixel_size_mismatch():
    other = Affine(4.0, 0.0, _MOSAIC.c, 0.0, -4.0, _MOSAIC.f)
    with pytest.raises(RuntimeError, match="does not match its parent mosaic"):
        _compute_grid_alignment(other, list(_MOSAIC)[:6], 512, 512, _LADDER)


@pytest.mark.parametrize("dcol,drow", [(-1924, 0), (0, -8), (-1, -1)])
def test_grid_alignment_rejects_a_window_origin_outside_its_tile(dcol, drow):
    """R31 defence in depth: v1's ESP_057469_2215 is -1924 px (9.6 km) west today.

    Its phase is still integer, so only the bounds clause catches it.
    """
    with pytest.raises(RuntimeError, match="outside the parent tile"):
        _compute_grid_alignment(_win(dcol, drow), list(_MOSAIC)[:6], 512, 512, _LADDER)


def test_row_axis_sign_is_not_the_column_axis_sign():
    """The row quotient uses (f_mosaic - f_window); e < 0 flips it. A copy-paste of the
    column form onto the row axis negates the origin, which this catches."""
    align = _compute_grid_alignment(_win(0, 640), list(_MOSAIC)[:6], 512, 512, _LADDER)
    assert align["mosaic_row_origin"] == +640      # not -640


def test_stage4_applies_the_size_and_confidence_floors_end_to_end(tmp_path):
    """R80 gate: the filter is never wired end-to-end anywhere today.

    Every existing Stage-4 test passes `detection_filters` as `{None, None}`, so a mutant
    that deletes the `_apply_detection_filters` call from `stage4_one_image` survives the
    entire suite. Four polygons on the 1 m sub-pixel grid so rasterised areas are exact:
    one dropped by size, one dropped by score, two kept (25 + 400 = 425 m2).
    """
    from unittest.mock import patch

    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=192, width=192)
    polys = [
        _work_box(info, 21.0, -21.0, 1.0),     # 2x2  = 4 m2, d=2.26 -> dropped by size
        _work_box(info, 52.5, -52.5, 2.5),     # 5x5  = 25 m2, d=5.64 -> kept
        _work_box(info, 110.0, -110.0, 10.0),  # 20x20 = 400 m2 -> kept
        _work_box(info, 205.0, -205.0, 5.0),   # 10x10, big enough, but score 0.10
    ]
    _write_polygons(info, polys, scores=[0.9, 0.9, 0.9, 0.10])
    row = _make_manifest_row(info)
    cfg = _labeling_cfg(detection_filters={"min_confidence": 0.5, "min_size_m": 5.0})

    with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
        prov = stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(), labeling_cfg=cfg,
            config_hash="hash_r80", apply_coreg_shift=False,
        )

    assert prov["n_polygons_stage1"] == 4
    assert prov["n_polygons_after_filter"] == 2, "exactly one dropped by size, one by score"
    assert prov["detection_filters"] == {"min_confidence": 0.5, "min_size_m": 5.0}

    df = pd.read_parquet(out_dir / "labels" / f"{obs}.parquet")
    fin = df[df.tile_size_px == TILE_SIZES[0]]
    assert fin["boulder_area"].sum() == pytest.approx(425.0, rel=1e-9), (
        "425 = 25 + 400; a radius reading of the floor would drop the 25 and give 400"
    )
    assert int(fin["boulder_count"].sum()) == 2


def test_stage4_records_the_realised_size_basis(tmp_path):
    """R80(d): the realised SIZE floor, the analogue of `realised_label_basis`.

    `detection_filters` records the configured floor and is byte-identical across all 38
    v2 sidecars, so exactly as with the confidence floor it cannot express the mixture
    R03 found. This records what was actually measured.
    """
    from unittest.mock import patch

    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=192, width=192)
    polys = [
        _work_box(info, 21.0, -21.0, 1.0),     # 2x2,     d=2.26 -> dropped by size
        _work_box(info, 52.5, -52.5, 2.5),     # 5x5,     d=5.64 -> kept
        _work_box(info, 110.0, -110.0, 10.0),  # 20x20,   d=22.57 -> kept
        # 4.5x4.5: d=5.08, so it CLEARS the size floor but is dropped by score. It is
        # deliberately the smallest size-survivor, so computing the surviving-diameter
        # statistic from the combined mask instead of the size mask alone changes the
        # answer (5.077 -> 5.642) and is caught.
        _work_box(info, 205.0, -205.0, 2.25),
    ]
    _write_polygons(info, polys, scores=[0.9, 0.9, 0.9, 0.10])
    row = _make_manifest_row(info)
    cfg = _labeling_cfg(detection_filters={"min_confidence": 0.5, "min_size_m": 5.0})

    with patch("src.ctx_tiles.murray_tile_for_manifest_row", return_value=info["murray_tile"]):
        prov = stage4_one_image(
            obs, cache_dir=cache_dir, output_dir=out_dir, manifest_row=row,
            target_crs=info["crs"].to_wkt(), labeling_cfg=cfg,
            config_hash="hash_r80b", apply_coreg_shift=False,
        )

    sb = prov["realised_size_basis"]
    assert sb["convention"] == "mixed_per_image_size_floor"
    assert sb["size_metric"] == "equivalent_circle_diameter_2sqrt_area_over_pi"
    assert sb["configured_min_size_m"] == 5.0
    assert sb["measured_in_crs_is_projected"] is True
    assert sb["measured_in_crs"] == "TestMars", "must be a short name, not a WKT blob"
    # Each floor is attributed separately -- a polygon failing both must not be counted twice.
    assert sb["n_dropped_by_size"] == 1
    assert sb["n_dropped_by_confidence"] == 1
    assert sb["size_floor_was_binding"] is True
    # Survivors of the SIZE floor ALONE: the 5x5 (5.642), the 20x20 (22.57) and the
    # 4.5x4.5 (5.077) that only the confidence floor removed -- NOT the 2x2. Using the
    # combined mask here would report 5.642 and quietly describe a different population.
    assert sb["min_surviving_diameter_ctx_frame_m"] == pytest.approx(5.0771, rel=1e-3)
    assert sb["area_dropped_by_size_m2"] == pytest.approx(4.0, rel=1e-9)


def test_realised_size_basis_records_the_per_image_physical_floor(tmp_path):
    """The number the block exists to carry.

    The floor is enforced on CTX-frame areas, so the *physical* floor it corresponds to
    depends on each image's own source projection. Two images with the same configured
    `min_size_m` must therefore record different physical floors -- otherwise the block is
    as byte-identical as the `detection_filters` snapshot it was written to replace.
    """
    from src.labeling import _describe_realised_size_basis

    ctx = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)             # standard parallel 0
    filters = {"min_confidence": None, "min_size_m": 1.4105}
    diag = {"n_in": 10, "n_dropped_by_size": 0, "n_dropped_by_confidence": 0}

    at_equator = _describe_realised_size_basis(diag, ctx, filters, mars_crs_wkt(0.0))
    at_60 = _describe_realised_size_basis(diag, ctx, filters, mars_crs_wkt(60.0))

    # Source frame == CTX frame: scale is exactly 1 and NOTHING is loosened. This is a real
    # case -- v1's ESP_039820_1750 has source lat_ts=0 on the same sphere.
    assert at_equator["source_to_target_diameter_scale"] == pytest.approx(1.0, abs=1e-12)
    assert at_equator["realised_physical_min_size_m"] == pytest.approx(1.4105, rel=1e-9)
    assert at_equator["realised_floor_is_looser_than_configured"] is False

    # A 60-degree source standard parallel inflates area by 1/cos(60) = 2, diameter by
    # sqrt(2), so the physical floor enforced is 1.4105/sqrt(2).
    assert at_60["source_to_target_diameter_scale"] == pytest.approx(2 ** 0.5, rel=1e-9)
    assert at_60["realised_physical_min_size_m"] == pytest.approx(1.4105 / 2 ** 0.5, rel=1e-9)
    assert at_60["realised_floor_is_looser_than_configured"] is True

    assert at_equator["realised_physical_min_size_m"] != at_60["realised_physical_min_size_m"]


def test_realised_size_basis_says_unknown_when_the_source_crs_is_missing():
    """Absence must never read as "checked and equal"."""
    from src.labeling import _describe_realised_size_basis

    ctx = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)
    out = _describe_realised_size_basis(
        {"n_in": 1}, ctx, {"min_confidence": None, "min_size_m": 1.4105}, None,
    )
    assert out["source_crs_available"] is False
    assert out["realised_floor_is_looser_than_configured"] is None
    assert "source_to_target_diameter_scale" not in out


@pytest.mark.parametrize("crs_in,expect_projected", [
    (TEST_MARS_CRS_WKT, True),
    ("EPSG:4326", False),
    (None, None),
])
def test_realised_size_basis_reports_the_measurement_frame_honestly(crs_in, expect_projected):
    """A hardcoded `True` here would pass a fixture that only ever uses a metric CRS."""
    from src.labeling import _describe_realised_size_basis

    crs = pyproj.CRS.from_user_input(crs_in) if crs_in else None
    out = _describe_realised_size_basis({"n_in": 0}, crs, {"min_size_m": None}, None)
    assert out["measured_in_crs_is_projected"] is expect_projected


def test_diagnostics_distinguish_configured_from_applied(tmp_path):
    """`_apply_detection_filters` silently skips the confidence floor when there is no
    `score` column, and applies neither floor on an empty frame. Both must be legible."""
    crs = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)

    no_score = gpd.GeoDataFrame(geometry=[box(0, 0, 5, 5)], crs=crs)
    diag: dict = {}
    _apply_detection_filters(no_score, {"min_confidence": 0.5, "min_size_m": None},
                             diagnostics=diag)
    assert diag["confidence_floor_configured"] is True
    assert diag["confidence_floor_applied"] is False
    assert diag["n_dropped_by_confidence"] == 0
    assert "confidence_floor_note" in diag

    empty = gpd.GeoDataFrame({"score": []}, geometry=[], crs=crs)
    diag2: dict = {}
    _apply_detection_filters(empty, {"min_confidence": 0.5, "min_size_m": 1.4105},
                             diagnostics=diag2)
    assert diag2["size_floor_configured"] is True, "configured is knowable even with no rows"
    assert diag2["size_floor_applied"] is False
    # Nothing was measured, so no measured-looking zero may be emitted.
    assert "area_total_m2" not in diag2


def test_diagnostics_do_not_emit_unmeasured_areas():
    """A seeded `area_total_m2: 0.0` on a non-empty image is a false claim, not a gap."""
    crs = pyproj.CRS.from_user_input(TEST_MARS_CRS_WKT)
    gdf = gpd.GeoDataFrame(
        {"score": [0.5] * 3}, geometry=[box(i * 20, 0, i * 20 + 10, 10) for i in range(3)],
        crs=crs,
    )
    diag: dict = {}
    _apply_detection_filters(gdf, None, diagnostics=diag)
    assert diag["n_in"] == 3
    assert "area_total_m2" not in diag, "300 m2 of polygons must not be recorded as 0.0"
