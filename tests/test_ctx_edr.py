"""Unit tests for src/ctx_edr.py — SeamMap fields -> live CTX EDR URL (the F resolver).

Template correctness is offline; the frame-geometry tests use the cached E8_N44 SeamMap and
skip when the cache is absent (CI/pristine checkouts). No network.
"""
from pathlib import Path

import pytest

from src.ctx_edr import EDR_URL_TEMPLATE, edr_url

E8_SEAMMAP = Path("cache/ctx_tiles/_seammap_E8_N44")
E8_ABUNDANCE = Path("reports/map_region/E8_N44_abundance.tif")


def test_edr_url_template():
    url = edr_url("MROX_1097", "B20_017408_2244_XN_44N351W")
    assert url == ("https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/"
                   "mrox_1097/data/B20_017408_2244_XN_44N351W.IMG")


def test_edr_url_lowercases_volume_only():
    # SeamMap stores VOLUME_ID upper-case; the archive tree is lower-case. PRODUCT_ID
    # case must be preserved (files are named with the original mixed-case id).
    url = edr_url("MROX_0009", "P01_001414_2164_XI_36N008W")
    assert "/mrox_0009/" in url
    assert url.endswith("P01_001414_2164_XI_36N008W.IMG")


def test_template_is_the_relocated_jpl_tree():
    # Guard against silently reverting to the stale PDS_IMG path (mars_reconnaissance_orbiter/
    # ctx — 404 since the post-2024 reorganization; DECISIONS 2026-07-02).
    assert "/mro/ctx/" in EDR_URL_TEMPLATE
    assert "mars_reconnaissance_orbiter" not in EDR_URL_TEMPLATE


@pytest.mark.skipif(not (E8_SEAMMAP.exists() and E8_ABUNDANCE.exists()),
                    reason="needs the cached E8_N44 SeamMap + abundance raster")
def test_frames_in_crop_matches_a1_payoff_site():
    from src.ctx_edr import frames_in_crop

    g = frames_in_crop("E8_N44", 1504, 8992, 15008)
    # the A1 payoff crop is the documented ~8-frame site; slivers must be filtered out
    assert 5 <= len(g) <= 12
    assert (g["overlap_frac"] >= 0.01).all()
    assert g["overlap_frac"].is_monotonic_decreasing
    assert g["edr_url"].str.endswith(".IMG").all()
    assert g["PRODUCT_ID"].is_unique
