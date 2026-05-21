"""Manifest loader + shapefile-glob tests."""
from __future__ import annotations

from src import manifest as M


def test_manifest_loads_with_required_columns(cfg):
    df = M.load_manifest(cfg.manifest_path)
    assert len(df) == 10
    for col in M.REQUIRED_COLUMNS:
        assert col in df.columns
    assert df["ObsId"].is_unique


def test_manifest_iter_rows_returns_dataclass(cfg):
    df = M.load_manifest(cfg.manifest_path)
    rows = list(M.iter_rows(df))
    assert len(rows) == 10
    r = rows[0]
    assert r.obs_id.startswith("ESP_")
    assert r.product_id.endswith("_RED")
    assert isinstance(r.center_lat, float)


def test_all_shapefiles_resolve(cfg):
    """Every manifest ObsId must resolve to exactly one *-mask-nms.shp under detections_root."""
    df = M.load_manifest(cfg.manifest_path)
    resolved = M.resolve_all_shapefiles(df, cfg.detections_root)
    assert set(resolved) == set(df["ObsId"])
    for obs_id, shp in resolved.items():
        assert shp.exists(), f"missing shapefile: {shp}"
        assert shp.suffix == ".shp"
        assert (shp.with_suffix(".prj")).exists(), f"missing .prj sidecar for {obs_id}"
