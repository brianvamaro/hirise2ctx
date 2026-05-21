"""Empty-shapefile handling: ESP_065711_1545 has 0 detections. Stage 1 must still
reproject + cache it cleanly (the image contributes all-zero tiles to Stage 4 labeling),
and the sanity check must short-circuit instead of crashing on a missing centroid.
"""
from __future__ import annotations

import pytest

from src import detections, manifest as M, qa
from src.ctx_retrieve import resolve_target_crs

OBS_ID = "ESP_065711_1545"


@pytest.mark.slow
def test_stage1_handles_empty_shapefile(cfg):
    df = M.load_manifest(cfg.manifest_path)
    row = df.set_index("ObsId").loc[OBS_ID]
    target_wkt = resolve_target_crs(cfg)

    gdf_t, gpkg, correction = detections.stage1_one_image(
        OBS_ID,
        detections_root=cfg.detections_root,
        target_crs=target_wkt,
        cache_dir=cfg.cache_dir,
        config_hash=cfg.hash,
        manifest_row=row,
    )
    # Empty in, empty out — but everything else should still work.
    assert len(gdf_t) == 0
    assert gpkg.exists()
    assert gdf_t.crs is not None  # reprojection assigned the target CRS

    # Sanity check must not crash on empty input; returns None instead of asserting.
    result = qa.assert_centroid_consistent(
        gdf_t,
        obs_id=OBS_ID,
        manifest_lat_deg=float(row["CenterLat"]),
        manifest_lon_deg=float(row["CenterLon_180"]),
        max_km=float(cfg["sanity"]["centroid_max_km"]),
    )
    assert result is None
