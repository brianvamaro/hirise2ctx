"""Unit tests for `src.colour` -- Stage 7 colour primitives.

Covers the helpers introduced for Stage 7c (windowed colour read in the source
CRS via tile-bounds reprojection) without requiring any real HiRISE COLOR.JP2
on disk: a tiny synthetic 3-band rasterio MemoryFile stands in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.transform
from rasterio.io import MemoryFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import colour  # noqa: E402


def _make_synth_color(
    width: int = 200,
    height: int = 200,
    px_size: float = 0.5,
    crs: str = "EPSG:32633",  # any projected CRS with metre units
):
    """Build an in-memory 3-band uint16 raster (IR/RED/BG order), with a known pattern.

    Band 1 (IR)  = 1000 + col index
    Band 2 (RED) = 800  + col index
    Band 3 (BG)  = 200  + col index
    All bands rescaled to uint16. Pad column 0 with literal 0s in all bands -- the
    `region_means` valid-mask test should drop it as nodata.
    """
    rng_x = np.arange(width, dtype=np.uint16)
    bands = np.stack(
        [
            np.tile(1000 + rng_x, (height, 1)),
            np.tile(800 + rng_x, (height, 1)),
            np.tile(200 + rng_x, (height, 1)),
        ]
    ).astype(np.uint16)
    bands[:, :, 0] = 0  # mark column 0 as nodata for the valid-mask test
    transform = rasterio.transform.from_origin(west=1000.0, north=2000.0,
                                                xsize=px_size, ysize=px_size)
    mem = MemoryFile()
    ds = mem.open(
        driver="GTiff", height=height, width=width, count=3,
        dtype="uint16", crs=crs, transform=transform,
    )
    ds.write(bands)
    return mem, ds


def test_region_means_drops_nodata_pixels():
    """region_means should ignore pixels that are 0 in any band (HiRISE nodata pad)."""
    mem, ds = _make_synth_color(width=10, height=10)
    arr = ds.read()
    mask = np.ones(arr.shape[1:], dtype=bool)
    out = colour.region_means(arr, mask, min_pixels=1)
    # Column 0 is nodata in all bands; valid pixels are columns 1-9, 10 rows = 90.
    assert out is not None
    assert out["n_pixels"] == 9 * 10
    # IR mean = mean(1001..1009) = 1005.0
    assert abs(out["IR"] - 1005.0) < 1e-9
    assert abs(out["RED"] - 805.0) < 1e-9
    assert abs(out["BG"] - 205.0) < 1e-9
    mem.close()


def test_region_means_returns_none_below_threshold():
    """region_means should refuse to compute a mean from too-few valid pixels."""
    mem, ds = _make_synth_color(width=5, height=5)
    arr = ds.read()
    mask = np.zeros(arr.shape[1:], dtype=bool)
    mask[0, 1] = True  # single valid pixel
    out = colour.region_means(arr, mask, min_pixels=8)
    assert out is None
    mem.close()


def test_lambertian_correct_scales_by_cos_i():
    """I/F_corrected = I/F_observed / cos(incidence_deg)."""
    arr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    out = colour.lambertian_correct(arr, incidence_deg=60.0)  # cos(60°) = 0.5
    expected = arr / 0.5
    assert np.allclose(out, expected)


def test_lambertian_correct_rejects_dark_geometry():
    """incidence > 90° (cos(i) <= 0) means the surface isn't illuminated -- raise."""
    with pytest.raises(ValueError, match="non-illuminated"):
        colour.lambertian_correct(np.array([0.1]), incidence_deg=95.0)


def test_ctx_bounds_to_source_bbox_is_axis_aligned():
    """ctx_bounds_to_source_bbox returns the bbox of the 4 transformed corners."""
    # Identity transformer so we can predict the answer exactly.
    class _Identity:
        @staticmethod
        def transform(xs, ys):
            return list(xs), list(ys)
    bbox = colour.ctx_bounds_to_source_bbox((1.0, 2.0, 3.0, 4.0), _Identity())
    assert bbox == (1.0, 2.0, 3.0, 4.0)


def test_windowed_colour_read_returns_none_outside_swath():
    """A tile fully outside the JP2 bounds should short-circuit to (None, None)."""
    mem, ds = _make_synth_color(width=10, height=10, px_size=1.0)
    # The synthetic raster covers x=[1000, 1010], y=[1990, 2000]. Pick bounds outside it.
    class _Identity:
        @staticmethod
        def transform(xs, ys):
            return list(xs), list(ys)
    arr, transform = colour.windowed_colour_read(
        ds, (5000.0, 5000.0, 5001.0, 5001.0),
        transformer=_Identity(), jp2_bounds=tuple(ds.bounds),
    )
    assert arr is None and transform is None
    mem.close()


def test_windowed_colour_read_returns_window_when_inside():
    """A tile inside the JP2 bounds returns a 3-band window + its transform."""
    mem, ds = _make_synth_color(width=10, height=10, px_size=1.0)
    # The synthetic raster covers x in [1000, 1010], y in [1990, 2000]. Pick a sub-window.
    class _Identity:
        @staticmethod
        def transform(xs, ys):
            return list(xs), list(ys)
    arr, transform = colour.windowed_colour_read(
        ds, (1002.0, 1995.0, 1005.0, 1998.0),
        transformer=_Identity(), jp2_bounds=tuple(ds.bounds),
    )
    assert arr is not None and arr.shape[0] == 3  # 3 bands
    assert arr.shape[1] > 0 and arr.shape[2] > 0
    mem.close()


def test_color_band_constants_are_stable():
    """The IR/RED/BG band-index constants are part of the public API; pin them so a
    typo or accidental reorder in src.colour is caught by tests rather than silently
    swapping band identity downstream."""
    assert colour.BAND_IR == 1
    assert colour.BAND_RED == 2
    assert colour.BAND_BG == 3
    assert colour.BAND_NAMES == ("IR", "RED", "BG")
