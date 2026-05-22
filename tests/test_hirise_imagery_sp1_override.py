"""SP1 fix on the JP2-imagery side.

The upstream HiRISE PDS bug (DECISIONS.md 2026-05-20) writes `Standard_Parallel_1=0` to
both the BoulderNet `.prj` AND the matching JP2 metadata for 4 of 10 priority10 images.
Stage 1 fixes the shapefile side via `src/pds_labels.py` + `src/detections.py`. These
tests cover the symmetric fix on the imagery side: `hirise_imagery._corrected_source_crs`
reads the Stage 1 sidecar's `source_crs_wkt` and `read_full_footprint_decimated` /
`read_native_window` apply it as a CRS override, so the decimated/native caches don't
inherit the buggy JP2 CRS.

Pure-unit: builds a synthetic Stage 1 sidecar JSON next to a temp cache dir; no JP2 IO.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyproj

from src.hirise_imagery import _corrected_source_crs, _crs_equal


def _write_sidecar(cache_dir: Path, obs_id: str, source_wkt: str) -> None:
    out_dir = cache_dir / "reprojected_detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{obs_id}.json").write_text(
        json.dumps({"obs_id": obs_id, "source_crs_wkt": source_wkt}),
        encoding="utf-8",
    )


# A corrected-CRS WKT in the same style detections.py writes (pyproj-canonicalized).
# Standard parallel 20° matches what we'd recover for ESP_047976_2020 from its PDS .LBL.
_CORRECTED_WKT = pyproj.CRS.from_user_input(
    'PROJCS["Equirectangular_MARS",'
    'GEOGCS["GCS_MARS",DATUM["D_MARS",'
    'SPHEROID["MARS_localRadius",3393833.2607584,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Equidistant_Cylindrical"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",180.0],'
    'PARAMETER["Standard_Parallel_1",20.0],'
    'UNIT["Meter",1.0]]'
).to_wkt()


def test_corrected_source_crs_returns_none_when_no_sidecar(tmp_path: Path):
    assert _corrected_source_crs("ESP_001234_2020", tmp_path) is None


def test_corrected_source_crs_returns_pyproj_crs_when_sidecar_exists(tmp_path: Path):
    _write_sidecar(tmp_path, "ESP_001234_2020", _CORRECTED_WKT)
    crs = _corrected_source_crs("ESP_001234_2020", tmp_path)
    assert crs is not None
    # Same projection latitude as the source WKT (20°).
    pp = pyproj.CRS.from_user_input(crs.to_wkt())
    assert pp.equals(pyproj.CRS.from_user_input(_CORRECTED_WKT))


def test_crs_equal_canonicalizes_whitespace(tmp_path: Path):
    """`_crs_equal` should ignore formatting differences between two equivalent WKTs."""
    a = pyproj.CRS.from_user_input(_CORRECTED_WKT)
    # Re-serialize via from_wkt to get a re-formatted but equivalent WKT
    b = pyproj.CRS.from_user_input(a.to_wkt(version="WKT2_2019"))
    assert _crs_equal(a, b)


def test_crs_equal_distinguishes_different_sp1():
    """A corrected SP1=20 CRS must NOT compare equal to a buggy SP1=0 CRS, even though
    pyproj's `.equals()` treats them as equal because the spherical-Equirectangular
    canonical form drops SP1 from its equality check. The literal SP1 parse in
    `_crs_equal` is what catches the stale-cache case in `read_full_footprint_decimated`.

    Build both CRSs from the canonical EPSG-name WKT so the replace actually fires (the
    earlier draft replaced the ESRI key in an already-canonicalized WKT, which was a
    no-op — and the test silently failed to exercise the staleness path).
    """
    a = pyproj.CRS.from_user_input(_CORRECTED_WKT)  # canonical form, SP1 = 20
    buggy_wkt = _CORRECTED_WKT.replace(
        '"Latitude of 1st standard parallel",20',
        '"Latitude of 1st standard parallel",0',
    )
    assert buggy_wkt != _CORRECTED_WKT, "replace did not fire — test setup broken"
    b = pyproj.CRS.from_user_input(buggy_wkt)
    assert not _crs_equal(a, b), (
        "An SP1=0 buggy CRS must not compare equal to the SP1=20 corrected CRS — "
        "otherwise the cache-staleness check in read_full_footprint_decimated would "
        "miss stale caches built before the SP1 JP2-side fix."
    )
