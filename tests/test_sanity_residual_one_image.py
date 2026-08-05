"""End-to-end Stage 0-1 integration test on ESP_047976_2020.

Runs the actual pipeline against the real BoulderNet shapefile, performs the Stage 0.5
CTX CRS probe (small network call, cached), reprojects, and asserts that the residual
between the reprojected polygon centroid and the manifest's published center is well
under the configured threshold. A failure here is the working version of the "fail loudly"
requirement in CLAUDE.md §3.3.
"""
from __future__ import annotations

import pytest

from src import detections, manifest as M, qa
from src.ctx_retrieve import resolve_target_crs

# Probe image switched from ESP_047976_2020 -> ESP_069669_2220 on 2026-05-20 after
# discovering that ESP_047976_2020 (and 3 other manifest images) have BoulderNet outputs
# whose polygon geometry is mis-located by hundreds of km from the manifest's published
# image center. See DECISIONS.md. ESP_069669_2220 is verified-good: 1462 polygons,
# residual ~6 km, in the dense E000_N40 region that covers 3 manifest images.
OBS_ID = "ESP_069669_2220"


@pytest.mark.slow
def test_stage1_centroid_residual_under_threshold(cfg, tmp_path):
    # R77: cache_dir MUST be tmp_path. Stage 1 writes cache/reprojected_detections/;
    # DECISIONS 2026-08-04 records this test silently rewriting the live copy for this
    # ObsId on 2026-06-10. Stage 1's inputs come from detections_root, not cache_dir,
    # so a bare tmp_path is sufficient here.
    df = M.load_manifest(cfg.manifest_path)
    row = df.set_index("ObsId").loc[OBS_ID]

    target_wkt = resolve_target_crs(cfg)

    gdf_t, gpkg, correction = detections.stage1_one_image(
        OBS_ID,
        detections_root=cfg.detections_root,
        target_crs=target_wkt,
        cache_dir=tmp_path,
        config_hash=cfg.hash,
        manifest_row=row,
    )
    assert gpkg.exists()
    assert len(gdf_t) == 1462  # matches summary.csv n_detections for this ObsId
    # ESP_069669_2220 has a correctly-labelled .prj; SP1 correction should NOT trigger.
    assert correction["status"] == "trusted_prj"

    result = qa.assert_centroid_consistent(
        gdf_t,
        obs_id=OBS_ID,
        manifest_lat_deg=float(row["CenterLat"]),
        manifest_lon_deg=float(row["CenterLon_180"]),
        max_km=float(cfg["sanity"]["centroid_max_km"]),
    )

    # Verified-good probe: residual is well under the 15 km threshold (was 6.3 km when
    # measured on 2026-05-20). Hundreds-of-km offsets (the bad images) would loud-fail.
    assert result.distance_m < float(cfg["sanity"]["centroid_max_km"]) * 1000.0
