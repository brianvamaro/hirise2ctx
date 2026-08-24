"""The SeamMap cache must work on a clone with no cache directory yet.

DECISIONS 2026-08-23e. `SEAM_DIR` was `cache/ctx_tiles` while `CTX_ZIP_DIR` was
`cache_v2/ctx_tiles`. On the dev laptop those are the SAME directory -- `cache_v2/ctx_tiles`
is an NTFS junction into `cache/` -- so the split was invisible. On Linux there is no
junction and a fresh Sherlock clone has no `cache/` at all (gitignored), so `load_frames`
would fetch the SeamMap over vsicurl and then die writing its GeoPackage into a directory
that does not exist. That is the whole A1 arm of step 11, on the first tile.
"""
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from src import striping


def test_seam_dir_and_zip_dir_share_one_root():
    """Two roots for one concept only worked because of a Windows junction."""
    assert striping.SEAM_DIR == striping.CTX_ZIP_DIR, (
        "SeamMaps and tile zips must resolve to one directory on every platform; "
        f"got {striping.SEAM_DIR} vs {striping.CTX_ZIP_DIR}")


def test_load_frames_creates_its_cache_root(tmp_path, monkeypatch):
    """The write must not assume the cache directory already exists."""
    root = tmp_path / "cache_v2" / "ctx_tiles"     # deliberately NOT created
    assert not root.exists()
    monkeypatch.setattr(striping, "SEAM_DIR", root)
    monkeypatch.setattr(striping, "CTX_ZIP_DIR", root)

    g = gpd.GeoDataFrame(
        {"PRODUCT_ID": ["A", "B"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 2), (3, 2), (3, 3)])],
        crs="EPSG:4326")
    monkeypatch.setattr(striping, "_tile_crs", lambda t: g.crs)
    monkeypatch.setattr(striping, "find_seam_shp", lambda t: "dummy.shp")
    monkeypatch.setattr(gpd, "read_file", lambda *a, **k: g)

    out = striping.load_frames("E0_N40", dissolve=True)
    assert len(out) == 2
    assert (root / "_frames_E0_N40.gpkg").is_file(), "cache root was not created on write"


def test_a_second_call_reads_the_cache_it_just_wrote(tmp_path, monkeypatch):
    root = tmp_path / "cache_v2" / "ctx_tiles"
    monkeypatch.setattr(striping, "SEAM_DIR", root)
    monkeypatch.setattr(striping, "CTX_ZIP_DIR", root)
    g = gpd.GeoDataFrame({"PRODUCT_ID": ["A"]},
                         geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326")
    monkeypatch.setattr(striping, "_tile_crs", lambda t: g.crs)
    monkeypatch.setattr(striping, "find_seam_shp", lambda t: "dummy.shp")
    real_read = gpd.read_file
    monkeypatch.setattr(gpd, "read_file", lambda *a, **k: g)
    striping.load_frames("E4_N44", dissolve=True)
    monkeypatch.setattr(gpd, "read_file", real_read)      # force the cache path
    again = striping.load_frames("E4_N44", dissolve=True)
    assert len(again) == 1
