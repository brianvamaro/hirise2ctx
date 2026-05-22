"""End-to-end Stage 2 integration test on ESP_069669_2220.

Marked `slow` because it touches the local CTX tile cache. Skipped automatically if the
required tile zip hasn't been downloaded yet — so a fresh checkout / CI run isn't ambushed
by a 1.5 GB download. Once `cache/ctx_tiles/E000_N40.zip` exists locally, this test runs
in seconds (windowed read is ~5 MB).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from src import detections, manifest as M
from src.ctx_retrieve import CTX_TILES_SUBDIR, CTX_WINDOWS_SUBDIR, stage2_one_image

OBS_ID = "ESP_069669_2220"
# Cache files use the Murray Lab form (manifest_to_murray() output) regardless of which
# URL form (padded vs bare) Murray Lab actually serves. The 2026-05-21 fetch confirmed
# E000_N40 manifest -> "E0_N40" murray form -> cache file `E0_N40.zip`.
TILE_NAME = "E0_N40"


def _tile_zip_exists(cache_dir: Path) -> bool:
    return (cache_dir / CTX_TILES_SUBDIR / f"{TILE_NAME}.zip").exists()


@pytest.mark.slow
def test_stage2_window_for_ESP_069669_2220(cfg):
    cache_dir = cfg.cache_dir
    if not _tile_zip_exists(cache_dir):
        pytest.skip(
            f"{TILE_NAME}.zip not in {cache_dir / CTX_TILES_SUBDIR}; "
            "run Stage 2 once interactively to populate the tile cache."
        )

    df = M.load_manifest(cfg.manifest_path)
    row = df.set_index("ObsId").loc[OBS_ID]
    cfg_retrieve = cfg["ctx_retrieve"]

    prov = stage2_one_image(
        OBS_ID,
        cache_dir=cache_dir,
        manifest_row=row,
        target_crs=cfg["target_crs"],
        url_template=cfg["ctx_mosaic"]["url_template"],
        buffer_m=float(cfg_retrieve["buffer_m"]),
        nominal_width_m=float(cfg_retrieve["nominal_hirise_width_m"]),
        nominal_length_m=float(cfg_retrieve["nominal_hirise_length_m"]),
        config_hash=cfg.hash,
    )

    out_tif = cache_dir / CTX_WINDOWS_SUBDIR / f"{OBS_ID}.tif"
    mask_tif = cache_dir / CTX_WINDOWS_SUBDIR / f"{OBS_ID}_hirise_mask.tif"
    assert out_tif.exists(), f"Stage 2 did not write {out_tif}"
    assert mask_tif.exists(), f"Stage 2 did not write HiRISE coverage mask {mask_tif}"
    assert prov["footprint_source"] == "polygon_bbox"
    assert prov["source_murray_tile"] == "E0_N40"
    # HiRISE coverage of the polygon-bbox window is typically 0.3-0.8 (diagonal swath).
    cov = prov["hirise_coverage_fraction"]
    assert 0.2 < cov < 0.9, f"HiRISE coverage fraction {cov} outside expected 0.2-0.9"

    # Confirm the cached GeoTIFF round-trips through rasterio and the CRS matches target.
    with rasterio.open(out_tif) as src:
        assert src.crs is not None
        # CTX mosaic is 5 m/px
        assert abs(src.transform.a - 5.0) < 1e-3
        assert abs(src.transform.e + 5.0) < 1e-3
        # Reasonable shape: HiRISE footprint ~6 km x ~16 km + 1 km buffer -> ~8 x ~18 km
        # at 5 m/px -> ~1600 x ~3600 px. Allow a wide envelope.
        h, w = src.height, src.width
        assert 1000 <= w <= 4000, f"width {w} px outside expected 1000-4000"
        assert 1000 <= h <= 5000, f"height {h} px outside expected 1000-5000"
        ctx_shape = (h, w)
        ctx_transform = src.transform
        ctx_crs = src.crs

    # HiRISE mask must match CTX window exactly: same shape, same transform, same CRS.
    with rasterio.open(mask_tif) as src:
        assert (src.height, src.width) == ctx_shape
        assert src.transform.almost_equals(ctx_transform)
        assert src.crs == ctx_crs
        m = src.read(1)
        assert m.dtype == np.uint8
        # Some "inside" and some "outside" pixels — neither degenerate.
        assert (m == 0).any() and (m == 1).any()
        assert abs(float(m.mean()) - cov) < 0.001

    # Window must contain the polygon bbox + buffer on every side.
    gdf = detections.load_reprojected(OBS_ID, cache_dir)
    poly_xmin, poly_ymin, poly_xmax, poly_ymax = gdf.total_bounds
    win_xmin, win_ymin, win_xmax, win_ymax = prov["actual_bounds_target_crs"]
    buffer_m = float(cfg_retrieve["buffer_m"])
    assert win_xmin <= poly_xmin - buffer_m + 5.0  # +5 m tolerance for pixel snap
    assert win_ymin <= poly_ymin - buffer_m + 5.0
    assert win_xmax >= poly_xmax + buffer_m - 5.0
    assert win_ymax >= poly_ymax + buffer_m - 5.0


@pytest.mark.slow
def test_stage2_is_idempotent(cfg):
    """Re-running Stage 2 must overwrite the window cache without erroring or growing the tile cache."""
    cache_dir = cfg.cache_dir
    if not _tile_zip_exists(cache_dir):
        pytest.skip(f"{TILE_NAME}.zip not present")

    df = M.load_manifest(cfg.manifest_path)
    row = df.set_index("ObsId").loc[OBS_ID]
    cfg_retrieve = cfg["ctx_retrieve"]

    tile_zip = cache_dir / CTX_TILES_SUBDIR / f"{TILE_NAME}.zip"
    before_mtime = tile_zip.stat().st_mtime
    before_size = tile_zip.stat().st_size

    prov1 = stage2_one_image(
        OBS_ID,
        cache_dir=cache_dir,
        manifest_row=row,
        target_crs=cfg["target_crs"],
        url_template=cfg["ctx_mosaic"]["url_template"],
        buffer_m=float(cfg_retrieve["buffer_m"]),
        nominal_width_m=float(cfg_retrieve["nominal_hirise_width_m"]),
        nominal_length_m=float(cfg_retrieve["nominal_hirise_length_m"]),
        config_hash=cfg.hash,
    )
    prov2 = stage2_one_image(
        OBS_ID,
        cache_dir=cache_dir,
        manifest_row=row,
        target_crs=cfg["target_crs"],
        url_template=cfg["ctx_mosaic"]["url_template"],
        buffer_m=float(cfg_retrieve["buffer_m"]),
        nominal_width_m=float(cfg_retrieve["nominal_hirise_width_m"]),
        nominal_length_m=float(cfg_retrieve["nominal_hirise_length_m"]),
        config_hash=cfg.hash,
    )

    # Tile zip untouched
    assert tile_zip.stat().st_mtime == before_mtime
    assert tile_zip.stat().st_size == before_size
    # Bounds + transform stable across runs
    assert prov1["actual_bounds_target_crs"] == prov2["actual_bounds_target_crs"]
    assert prov1["actual_transform"] == prov2["actual_transform"]
