"""Unit tests for src/validation_retrieve.py (PLAN_RegionalMap §3, phase 1).

No network: the geometry helpers (seam detection/split, geographic->CRS bbox, target grid),
the reproject-onto-grid, the windowed read, and the hillshade are exercised on synthetic
in-memory rasters and Mars proj4 CRSs. The real fetch is covered by the figure step, not here.
"""
import math

import numpy as np
import pytest

from src import validation_retrieve as vr

# Mars IAU-sphere CRSs (decoupled from config; R matches the pipeline target_crs).
R = 3396190.0
GEO = f"+proj=longlat +R={R} +no_defs"
EQC = f"+proj=eqc +R={R} +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs"
DEG2M = R * math.pi / 180.0  # metres per degree on the eqc sphere at the equator


def test_seam_lon_from_central_meridian():
    cm180 = f"+proj=eqc +R={R} +lat_ts=0 +lon_0=180 +x_0=0 +y_0=0 +units=m +no_defs"
    assert vr.seam_lon(EQC) == pytest.approx(-180.0)     # cm=0 (MOLA/TES) -> seam +/-180
    assert vr.seam_lon(cm180) == pytest.approx(0.0)      # cm=180 (THEMIS night-IR) -> seam 0


def test_split_bounds_at_seam():
    # Not crossing the seam -> single part unchanged.
    assert vr.split_bounds_at_seam((0, 40, 20, 48), seam=-180.0) == [(0, 40, 20, 48)]
    # Circum-Chryse over a cm=180 source: seam=0 inside [-12,20] -> two halves split at 0.
    parts = vr.split_bounds_at_seam((-12, 32, 20, 48), seam=0.0)
    assert len(parts) == 2
    (w0, _, e0, _), (w1, _, e1, _) = parts
    assert w0 == -12 and e1 == 20
    assert e0 == pytest.approx(0.0, abs=1e-3) and w1 == pytest.approx(0.0, abs=1e-3)
    assert e0 < 0 < w1                                   # nudged off the seam either side


def test_bounds_lonlat_to_crs_projects_to_metres():
    left, bottom, right, top = vr.bounds_lonlat_to_crs((0, 40, 20, 48), EQC)
    assert left == pytest.approx(0.0, abs=1.0)
    assert right == pytest.approx(20 * DEG2M, rel=1e-6)
    assert bottom == pytest.approx(40 * DEG2M, rel=1e-6)
    assert top == pytest.approx(48 * DEG2M, rel=1e-6)
    assert left < right and bottom < top


def test_bounds_lonlat_to_crs_geographic_identity():
    # A degree-tagged source CRS returns the lon/lat box essentially unchanged.
    left, bottom, right, top = vr.bounds_lonlat_to_crs((0, 40, 20, 48), GEO)
    assert (left, bottom, right, top) == pytest.approx((0, 40, 20, 48), abs=1e-6)


def test_build_target_grid_covers_bounds():
    transform, width, height = vr.build_target_grid((0, 40, 20, 48), EQC, res_m=1000.0)
    assert transform.a == pytest.approx(1000.0) and transform.e == pytest.approx(-1000.0)
    assert width >= int(20 * DEG2M / 1000.0) and height >= int(8 * DEG2M / 1000.0)
    # north-up: origin at the top-left (max y), pixels march south/east.
    assert transform.f == pytest.approx(48 * DEG2M, rel=1e-6)
    assert transform.c == pytest.approx(0.0, abs=1.0)


def _memraster(crs, transform, arr, nodata=None):
    import rasterio
    from rasterio.io import MemoryFile

    mf = MemoryFile()
    ds = mf.open(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                 dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata)
    ds.write(arr, 1)
    return mf, ds


def test_windowed_read_returns_covering_subwindow():
    from rasterio.transform import Affine

    # Global-ish eqc raster, 1 deg/px, covering lon[-30,30] lat[30,60].
    res = DEG2M
    transform = Affine(res, 0, -30 * DEG2M, 0, -res, 60 * DEG2M)
    arr = (np.arange(60 * 30).reshape(30, 60)).astype(np.float32)  # 30 rows x 60 cols
    mf, ds = _memraster(EQC, transform, arr)
    try:
        data, win_tr = vr.windowed_read(ds, (0, 40, 20, 48), buffer_deg=0.5)
    finally:
        ds.close(); mf.close()
    # Sub-window is much smaller than the full raster but covers the request.
    assert data.shape[0] < 30 and data.shape[1] < 60
    left = win_tr.c
    right = win_tr.c + data.shape[1] * win_tr.a
    top = win_tr.f
    bottom = win_tr.f + data.shape[0] * win_tr.e
    assert left <= 0 * DEG2M + 1 and right >= 20 * DEG2M - 1
    assert bottom <= 40 * DEG2M + 1 and top >= 48 * DEG2M - 1


def test_windowed_read_raises_off_raster():
    from rasterio.transform import Affine

    transform = Affine(DEG2M, 0, 0, 0, -DEG2M, 10 * DEG2M)
    arr = np.zeros((10, 10), np.float32)  # lon[0,10] lat[0,10]
    mf, ds = _memraster(EQC, transform, arr)
    try:
        with pytest.raises(ValueError):
            vr.windowed_read(ds, (100, 80, 110, 85), buffer_deg=0.1)
    finally:
        ds.close(); mf.close()


def test_reproject_to_grid_preserves_values_same_crs():
    from rasterio.transform import Affine

    # Source ramp in eqc; reproject onto a coarser eqc grid over a sub-extent.
    res = DEG2M
    src_tr = Affine(res, 0, 0, 0, -res, 20 * DEG2M)
    src = np.tile(np.arange(20, dtype=np.float32), (20, 1))  # value == column == lon degree
    dst_tr = Affine(2 * res, 0, 2 * DEG2M, 0, -2 * res, 18 * DEG2M)
    out = vr.reproject_to_grid(src, src_tr, EQC, dst_crs_wkt=EQC, dst_transform=dst_tr,
                               dst_shape=(8, 8), resampling="nearest")
    assert out.shape == (8, 8)
    assert np.all(np.isfinite(out))
    # Column 0 of dst sits at lon~2, column 7 at lon~16 -> values track lon.
    assert out[0, 0] == pytest.approx(2, abs=1.5)
    assert out[0, 7] > out[0, 0]


def test_reproject_to_grid_nan_outside_coverage():
    from rasterio.transform import Affine

    res = DEG2M
    src_tr = Affine(res, 0, 0, 0, -res, 5 * DEG2M)
    src = np.ones((5, 5), np.float32)  # lon[0,5] lat[0,5]
    # dst extends east of the source -> right columns must be NaN.
    dst_tr = Affine(res, 0, 0, 0, -res, 5 * DEG2M)
    out = vr.reproject_to_grid(src, src_tr, EQC, dst_crs_wkt=EQC, dst_transform=dst_tr,
                               dst_shape=(5, 10), resampling="nearest")
    assert np.all(np.isfinite(out[:, :5]))
    assert np.all(np.isnan(out[:, 6:]))


def test_hillshade_flat_and_range():
    flat = np.zeros((8, 8))
    hs = vr.hillshade(flat, res_m=463.0, altitude_deg=45.0)
    assert np.allclose(hs, math.sin(math.radians(45.0)), atol=1e-6)
    assert hs.min() >= 0.0 and hs.max() <= 1.0
    # NaN propagates from the DEM.
    dem = np.zeros((5, 5)); dem[2, 2] = np.nan
    assert np.isnan(vr.hillshade(dem)[2, 2])
