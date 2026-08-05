"""Stage 4 (label generation) tests.

Unit tests use synthetic CTX windows, masks, and polygons -- no caches, no downloads.
The integration test on ESP_069669_2220 is marked `slow` and auto-skips when Stage 2/3
caches are missing.
"""
from __future__ import annotations

import json
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


def _make_window(tmp_path: Path, *, height: int, width: int, pixel_m: float = 5.0,
                 origin_x: float = 0.0, origin_y: float = 0.0,
                 mask_fill: int = 1) -> tuple[Path, Path, dict, str]:
    """Write a synthetic CTX window + mask + mosaic tile sidecar. Returns the four pieces."""
    # The "mosaic" extends from (origin_x, origin_y) covering more than the window so window
    # (0,0) lies at integer pixel offsets of the mosaic (here, exactly at the mosaic origin).
    window_transform = Affine(pixel_m, 0, origin_x, 0, -pixel_m, origin_y)
    mosaic_transform = list(window_transform)[:6]  # window IS the mosaic origin in tests
    crs = pyproj.CRS.from_user_input(
        'PROJCRS["TestMars",BASEGEOGCRS["GCS_TestMars",DATUM["D_TestMars",'
        'ELLIPSOID["TestMars",3396190.0,0.0,LENGTHUNIT["metre",1]]],'
        'PRIMEM["Reference_Meridian",0,ANGLEUNIT["Degree",0.0174532925199433]]],'
        'CONVERSION["EquidistantCylindrical",'
        'METHOD["Equidistant Cylindrical (Spherical)",ID["EPSG",1029]],'
        'PARAMETER["Latitude of 1st standard parallel",0,ANGLEUNIT["Degree",0.0174532925199433]],'
        'PARAMETER["Longitude of natural origin",0,ANGLEUNIT["Degree",0.0174532925199433]],'
        'PARAMETER["False easting",0,LENGTHUNIT["metre",1]],'
        'PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],'
        'CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1]],'
        'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1]]]'
    )
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
            "inner_shape": [10_000, 10_000],
            "inner_dtype": "uint8",
        }),
        encoding="utf-8",
    )

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
    }
    return cache_dir, out_dir, info, obs_id


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
    """A 5x5 m boulder (one full CTX pixel) at the origin should rasterize to 25 sub-pixels
    at subpixel_factor=5 -> 25 m^2 -- exactly the polygon area."""
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=64, width=64)
    # Polygon covering CTX pixel (0,0): x in [0, 5], y in [-5, 0]
    poly = box(0.0, -5.0, 5.0, 0.0)
    _write_polygons(info, [poly])
    gdf = gpd.read_file(
        cache_dir / "reprojected_detections" / f"{obs}.gpkg", layer="detections",
    )
    align = _compute_grid_alignment(
        info["window_transform"], info["mosaic_transform"], 64, 64, TILE_SIZES,
    )
    raster = _rasterize_boulders_subpixel(gdf, info["window_transform"], align, subpixel_factor=5)
    # Sub-pixel size is 1 m; a 5 m x 5 m polygon should fill 25 sub-pixels.
    assert int(raster.sum()) == 25


def test_rasterize_returns_zero_for_empty_gdf(tmp_path):
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=64, width=64)
    gdf = gpd.GeoDataFrame({"score": []}, geometry=[], crs=info["crs"])
    align = _compute_grid_alignment(
        info["window_transform"], info["mosaic_transform"], 64, 64, TILE_SIZES,
    )
    raster = _rasterize_boulders_subpixel(gdf, info["window_transform"], align, subpixel_factor=5)
    assert raster.sum() == 0


# ----------------------------------------------------------------------------
# Centroid counting
# ----------------------------------------------------------------------------

def test_count_centroids_per_finest_cell_assigns_to_owner(tmp_path):
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=64, width=64)
    # Three boulders:
    #   centroid (20, -20) m -> mosaic-pixel (4, 4) -> finest cell (0, 0)  (S_min=8)
    #   centroid (60, -60) m -> mosaic-pixel (12, 12) -> finest cell (1, 1)
    #   centroid (50, -50) m -> mosaic-pixel (10, 10) -> finest cell (1, 1)
    polys = [
        box(18, -22, 22, -18),
        box(58, -62, 62, -58),
        box(48, -52, 52, -48),
    ]
    _write_polygons(info, polys)
    gdf = gpd.read_file(
        cache_dir / "reprojected_detections" / f"{obs}.gpkg", layer="detections",
    )
    align = _compute_grid_alignment(
        info["window_transform"], info["mosaic_transform"], 64, 64, TILE_SIZES,
    )
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
    scales = _sum_up_ladder(
        finest, TILE_SIZES, px_x=5.0, px_y=5.0, j_min_row=0, j_min_col=0,
    )
    base_area = float(finest["boulder_area"].sum())
    base_count = int(finest["boulder_count"].sum())
    for sc in scales:
        # All cells eligible -> all-eligible at every scale
        assert sc["eligible"].all()
        assert float(sc["boulder_area"].sum()) == pytest.approx(base_area, rel=1e-10)
        assert int(sc["boulder_count"].sum()) == base_count


def test_sum_up_ladder_coarse_ineligible_if_any_subtile_ineligible():
    """A single ineligible finest tile must drop the containing 16-, 32-, and 64-px tile."""
    n = 8
    finest = {
        "boulder_area": np.zeros((n, n)),
        "boulder_count": np.zeros((n, n), dtype=np.int64),
        "eligible": np.ones((n, n), dtype=bool),
    }
    finest["eligible"][0, 0] = False
    scales = _sum_up_ladder(
        finest, TILE_SIZES, px_x=5.0, px_y=5.0, j_min_row=0, j_min_col=0,
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
    scales = _sum_up_ladder(
        finest, TILE_SIZES, px_x=5.0, px_y=5.0, j_min_row=0, j_min_col=0,
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
    cache_dir, out_dir, info, obs = _make_window(
        tmp_path, height=64, width=64, mask_fill=1,
    )
    # Punch a single mask=0 pixel into finest tile (0, 0).
    mask_tif = info["mask_tif"]
    with rasterio.open(mask_tif, "r+") as src:
        mask = src.read(1)
        mask[3, 3] = 0
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
    # Finest tile (ti=0, tj=0) must be missing at every scale (the ineligibility propagates
    # upward through the nested grid).
    for s in TILE_SIZES:
        scaled = df[df["tile_size_px"] == s]
        ratio = s // TILE_SIZES[0]
        # The (0, 0) coarse tile would have its top-left finest sub-tile at (0, 0) -- ineligible.
        assert not ((scaled["ti"] == 0) & (scaled["tj"] == 0)).any(), (
            f"tile_size_px={s}: (0,0) tile should have been dropped"
        )


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
    crs = pyproj.CRS.from_epsg(4326)
    gdf = gpd.GeoDataFrame(
        {"score": [0.1, 0.4, 0.9]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
        crs=crs,
    )
    out = _apply_detection_filters(gdf, {"min_confidence": 0.3, "min_size_m": None})
    assert len(out) == 2
    assert set(out["score"]) == {0.4, 0.9}


def test_detection_filter_min_size_drops_small_diameters():
    crs = pyproj.CRS.from_epsg(4326)
    # Areas: 1, 100, 1000 -> diameters: 1.13, 11.28, 35.68
    gdf = gpd.GeoDataFrame(
        {"score": [0.5] * 3},
        geometry=[box(0, 0, 1, 1), box(0, 0, 10, 10), box(0, 0, 100, 10)],
        crs=crs,
    )
    out = _apply_detection_filters(gdf, {"min_confidence": None, "min_size_m": 5.0})
    assert len(out) == 2  # diam 1.13 drops, 11.28 and 35.68 stay


# ----------------------------------------------------------------------------
# Label transforms + tile bounds
# ----------------------------------------------------------------------------

def test_label_transforms_emit_expected_columns(tmp_path):
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=128, width=128)
    # Single 25 m^2 boulder centered in finest tile (0,0).
    poly = box(0.0, -5.0, 5.0, 0.0)
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

    # Finest tile (0, 0) must have boulder_area = 25 m^2, boulder_count = 1, frac = 25/1600.
    finest_00 = df[(df["tile_size_px"] == 8) & (df["ti"] == 0) & (df["tj"] == 0)]
    assert len(finest_00) == 1
    assert finest_00["boulder_area"].iloc[0] == pytest.approx(25.0)
    assert int(finest_00["boulder_count"].iloc[0]) == 1
    assert finest_00["fractional_area"].iloc[0] == pytest.approx(25.0 / 1600.0)
    assert bool(finest_00["binary_by_area"].iloc[0])
    assert bool(finest_00["binary_by_count"].iloc[0])


def test_tile_bounds_align_with_mosaic_pixel_grid(tmp_path):
    """Tile bounds (xmin, xmax, ymin, ymax) must lie on integer multiples of (S * px) from the
    mosaic origin -- the definition of 'anchored to CTX pixel origin'."""
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=128, width=128)
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
    # Mosaic origin is (0, 0); pixel size = 5 m. Every tile bound must be an integer multiple
    # of (tile_size_px * 5 m) from the origin -- i.e. xmin / (S*5) is an integer.
    for s in TILE_SIZES:
        sub = df[df["tile_size_px"] == s]
        step = s * 5.0
        for col in ("xmin", "xmax"):
            mods = np.mod(sub[col].to_numpy(), step)
            assert np.allclose(mods, 0, atol=1e-6), f"{col} not aligned at scale {s}"
        for col in ("ymin", "ymax"):
            mods = np.mod(sub[col].to_numpy(), step)
            assert np.allclose(mods, 0, atol=1e-6), f"{col} not aligned at scale {s}"


# ----------------------------------------------------------------------------
# Idempotency
# ----------------------------------------------------------------------------

def test_stage4_is_idempotent(tmp_path):
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=128, width=128)
    polys = [box(20, -20, 30, -10), box(80, -80, 95, -65)]
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
    cache_dir, out_dir, info, obs = _make_window(tmp_path, height=128, width=128)
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
