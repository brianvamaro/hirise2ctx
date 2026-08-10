"""R14 — a killed write must not leave a product that looks finished.

Three kill signatures were measured on real `reports/map_region` rasters, and the obvious
checks catch different subsets of them:

  1. **truncated** — `rasterio.open` succeeds at 10/50/90/99/99.99 % truncation and reports the
     correct shape and dtype (the first IFD sits at byte 8); a decode raises.
  2. **valid but all-nodata** — opens, decodes, last block reads fine. NaN *is* this product's
     nodata, so nothing structural separates it from a legitimately masked tile.
  3. **half the blocks written** — opens, decodes, last block fine; finite fraction 0.5193.

Signatures 2 and 3 are visible *only* to a finite-count check, which is why `expect_finite`
is load-bearing rather than decoration. Everything here is synthetic and lives in `tmp_path`.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import Affine                                      # noqa: E402

from src.mapping import file_sha256, verify_geotiff, write_geotiff        # noqa: E402

_T = Affine(160.0, 0.0, -711036.0, 0.0, -160.0, 2844945.0)
_CRS = ("PROJCS[\"m\",GEOGCS[\"g\",DATUM[\"d\",SPHEROID[\"Mars_2015\",3396190,169.8944472]],"
        "PRIMEM[\"Reference_Meridian\",0],UNIT[\"degree\",0.0174532925199433]],"
        "PROJECTION[\"Equirectangular\"],UNIT[\"metre\",1]]")


def _field(n=64, finite=True):
    a = np.arange(n * n, dtype=np.float64).reshape(n, n) / (n * n)
    return a if finite else np.full((n, n), np.nan)


def test_a_good_raster_verifies(tmp_path):
    """Guards against a verify so strict it rejects correct output."""
    a = _field()
    p = write_geotiff(tmp_path / "ok.tif", a, _T, _CRS)
    assert verify_geotiff(p, expect_shape=a.shape, expect_finite=a.size) is None


def test_signature_1_truncation_is_rejected(tmp_path):
    """`rasterio.open` alone proves nothing — it succeeds at every truncation fraction."""
    a = _field(128)
    p = write_geotiff(tmp_path / "t.tif", a, _T, _CRS)
    full = p.read_bytes()
    for frac in (0.10, 0.50, 0.90, 0.99):
        p.write_bytes(full[: int(len(full) * frac)])
        with rasterio.open(p) as src:                     # still opens, still right shape
            assert (src.height, src.width) == a.shape
        assert verify_geotiff(p, expect_shape=a.shape, expect_finite=a.size) is not None


def test_signature_2_all_nodata_is_rejected_only_by_the_finite_count(tmp_path):
    """The mutant the register's own fix bullet does not kill: a cleanly closed, fully
    readable, 100 %-NaN raster. Shape, dtype and a last-block read all pass."""
    a = _field(finite=False)
    p = write_geotiff(tmp_path / "nan.tif", a, _T, _CRS, verify=False)
    assert verify_geotiff(p, expect_shape=a.shape) is None            # structure is perfect
    why = verify_geotiff(p, expect_shape=a.shape, expect_finite=64 * 64)
    assert why is not None and "finite" in why


def test_signature_3_half_written_blocks_is_rejected(tmp_path):
    """Cleanly closed, fully readable, and silently half empty."""
    a = _field(512)
    half = a.copy()
    half[256:] = np.nan
    p = write_geotiff(tmp_path / "half.tif", half, _T, _CRS)
    assert verify_geotiff(p, expect_shape=a.shape) is None
    assert verify_geotiff(p, expect_shape=a.shape, expect_finite=a.size) is not None


def test_a_post_commit_byte_flip_is_caught_by_sha256(tmp_path):
    """All 78 shipped rasters share one mtime — they were bulk-copied off Sherlock, the same
    transport vector that produced R23's truncated shapefiles. Size alone can miss it."""
    p = write_geotiff(tmp_path / "c.tif", _field(), _T, _CRS)
    digest, size = file_sha256(p), p.stat().st_size
    assert verify_geotiff(p, expect_bytes=size, expect_sha256=digest) is None
    b = bytearray(p.read_bytes())
    b[-32] ^= 0xFF
    p.write_bytes(bytes(b))
    assert p.stat().st_size == size                        # same size, different content
    assert "sha256" in verify_geotiff(p, expect_bytes=size, expect_sha256=digest)


def test_a_failed_write_does_not_destroy_the_existing_good_raster(tmp_path):
    """The real R14 harm mode, and the strongest pin in this file.

    `rasterio.open(path, "w")` truncates the destination *immediately*, so a re-run that dies
    mid-write destroys the good tile it was going to replace. Staging to a `.tmp` sibling means
    a failure leaves the original byte-identical.
    """
    p = tmp_path / "keep.tif"
    write_geotiff(p, _field(), _T, _CRS)
    before, size = file_sha256(p), p.stat().st_size

    bad = _field(128)                                      # different shape -> verify fails
    import src.mapping as m
    real = m.verify_geotiff
    try:
        m.verify_geotiff = lambda *a, **k: "forced failure"
        with pytest.raises(OSError, match="failed post-write verification"):
            write_geotiff(p, bad, _T, _CRS)
    finally:
        m.verify_geotiff = real

    assert p.exists(), "the existing raster was destroyed by a failed write"
    assert file_sha256(p) == before and p.stat().st_size == size
    assert not list(tmp_path.glob("*.tmp")), "a .tmp sibling was left behind"


def test_expect_finite_is_computed_on_the_float32_cast(tmp_path):
    """`write_geotiff` casts to float32, and the cast can turn a finite float64 into inf
    (1e300 -> inf). Computing the expectation from the float64 input would make the function
    reject its own correct output."""
    a = np.array([[1e300, 1.0], [2.0, 3.0]], dtype=np.float64)
    assert np.isfinite(a).sum() == 4
    with np.errstate(over="ignore"):
        assert np.isfinite(a.astype(np.float32)).sum() == 3
    p = write_geotiff(tmp_path / "cast.tif", a, _T, _CRS)   # must not raise
    assert verify_geotiff(p, expect_finite=3) is None


def test_verify_reports_a_missing_file_rather_than_raising(tmp_path):
    assert verify_geotiff(tmp_path / "nope.tif") == "missing"
