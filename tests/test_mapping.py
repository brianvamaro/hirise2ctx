"""Unit tests for the map-inference geometry/raster helpers (PLAN_FM §2.6).

Pure-numpy helpers only (no checkpoint, no GeoTIFF read). The (ti,tj)->raster
placement, the 32x-coarsened affine, and own-tile nodata fraction are the seams
that decide whether predictions land in the right geo location, so they get
exact assertions on hand-built inputs.
"""
import numpy as np
import pytest

from src.fm_embeddings import tile_grid_for_window
from src.mapping import (
    CtxWindow, coarsened_transform, own_tile_zero_fraction, predict_window,
    tile_origin_transform, tiles_to_raster,
)


class _FakeEmbedder:
    """Returns a per-tile embedding whose column 0 is a 0.1..0.9 ramp; all valid."""
    def embed_window(self, arr, ti, tj, *, tile_px, row0, col0, pool, batch):
        n = ti.size
        emb = np.zeros((n, 4), np.float32)
        emb[:, 0] = np.linspace(0.1, 0.9, n)
        return emb, np.ones(n, dtype=bool)


class _FakeHead:
    """P(rich) = embedding column 0 (so raw prob is the known ramp)."""
    def predict(self, emb):
        return emb[:, 0].astype(np.float64)


def _fit_layer():
    from src.calibration import CalibrationLayer
    rng = np.random.default_rng(0)
    pr = rng.uniform(0, 1, 3000)
    fa = np.clip(pr * 0.05, 0, 0.3); fa[rng.random(3000) < 0.18] = 0.0
    return CalibrationLayer.fit(pr, (fa > 1e-2).astype(int), pr, fa)


def _window():
    data = np.full((32, 32), 100, dtype=np.uint8)   # non-zero -> no nodata masking
    return CtxWindow(data=data, row_off=0, col_off=0,
                     transform=(5.0, 0.0, 0.0, 0.0, -5.0, 0.0), crs_wkt="LOCAL")


def test_predict_window_raw_is_backward_compatible():
    pred = predict_window(_window(), _FakeEmbedder(), _FakeHead(), tile_px=8)
    assert pred.calibrated is False
    assert pred.prob_raw is None and pred.abundance is None and pred.abundance_raster is None
    assert np.nanmax(pred.prob) <= 0.9 + 1e-9         # raw ramp, uncalibrated


def test_predict_window_calibration_applies_both_maps():
    layer = _fit_layer()
    pred = predict_window(_window(), _FakeEmbedder(), _FakeHead(), tile_px=8, calibrator=layer)
    assert pred.calibrated is True
    u = np.isfinite(pred.prob_raw)
    assert u.any()
    # rich/poor raster is isotonic(raw); abundance is qmatch(raw) — both off the SAME raw P(rich)
    assert np.allclose(pred.prob[u], layer.calibrate_prob(pred.prob_raw[u]))
    assert np.allclose(pred.abundance[u], layer.calibrate_abundance(pred.prob_raw[u]))
    assert pred.abundance_raster.shape == pred.raster.shape
    assert np.all(pred.abundance[u] >= 0)            # fractional_area is non-negative


def test_predict_window_isotonic_toggle():
    layer = _fit_layer()
    on = predict_window(_window(), _FakeEmbedder(), _FakeHead(), tile_px=8, calibrator=layer)
    off = predict_window(_window(), _FakeEmbedder(), _FakeHead(), tile_px=8, calibrator=layer,
                         apply_isotonic=False)
    u = np.isfinite(on.prob_raw)
    # isotonic off -> rich/poor raster is raw P(rich); abundance (qmatch) still applied either way
    assert np.allclose(off.prob[u], off.prob_raw[u])
    assert np.allclose(on.abundance[u], off.abundance[u])
    assert off.calibrated is True


def test_tiles_to_raster_places_by_index():
    ti = np.array([10, 10, 11, 11])
    tj = np.array([4, 5, 4, 5])
    vals = np.array([0.1, 0.2, 0.3, 0.4])
    raster, ti_min, tj_min = tiles_to_raster(ti, tj, vals)
    assert (ti_min, tj_min) == (10, 4)
    assert raster.shape == (2, 2)
    np.testing.assert_array_equal(raster, [[0.1, 0.2], [0.3, 0.4]])


def test_tiles_to_raster_fills_missing_with_nan():
    # A gap in the (ti,tj) coverage stays NaN.
    ti = np.array([0, 0, 1])
    tj = np.array([0, 1, 0])  # (1,1) missing
    vals = np.array([1.0, 2.0, 3.0])
    raster, _, _ = tiles_to_raster(ti, tj, vals)
    assert raster.shape == (2, 2)
    assert np.isnan(raster[1, 1])


def test_coarsened_transform_origin_and_scale():
    # North-up tile: 5 m/px, origin (c, f). Tile_px=32 -> 160 m output pixels,
    # output (0,0) at tile (ti_min, tj_min) top-left.
    a, e = 5.0, -5.0
    c, f = -711296.37, 2370987.90
    tile_transform = (a, 0.0, c, 0.0, e, f)
    ti_min, tj_min, tile_px = 100, 40, 32
    tr = coarsened_transform(tile_transform, ti_min, tj_min, tile_px)
    assert tr.a == a * tile_px and tr.e == e * tile_px
    # Origin: x at col tj_min*tile_px, y at row ti_min*tile_px.
    assert tr.c == pytest.approx(c + tj_min * tile_px * a)
    assert tr.f == pytest.approx(f + ti_min * tile_px * e)


def test_coarsened_transform_pixel_maps_to_tile_topleft():
    # The geo of output pixel (row=r, col=cc) must equal the parent-tile geo of
    # tile (ti_min+r, tj_min+cc)'s top-left.
    a, e = 5.0, -5.0
    c, f = 1000.0, 5000.0
    tile_transform = (a, 0.0, c, 0.0, e, f)
    ti_min, tj_min, tile_px = 7, 3, 32
    tr = coarsened_transform(tile_transform, ti_min, tj_min, tile_px)
    r, cc = 2, 4
    x, y = tr * (cc, r)  # rasterio Affine: (col, row) -> (x, y) of pixel top-left
    exp_x = c + (tj_min + cc) * tile_px * a
    exp_y = f + (ti_min + r) * tile_px * e
    assert x == pytest.approx(exp_x) and y == pytest.approx(exp_y)


def test_tile_origin_transform_inverts_window_offset():
    # A window read at (row_off, col_off) carries an offset origin; recovering the
    # tile origin must return the original tile affine.
    a, e = 5.0, -5.0
    c, f = 237099.0, 2370987.9
    tile_transform = (a, 0.0, c, 0.0, e, f)
    row_off, col_off = 43790, 10828
    win_c = c + col_off * a
    win_f = f + row_off * e
    recovered = tile_origin_transform((a, 0.0, win_c, 0.0, e, win_f), row_off, col_off)
    assert recovered == pytest.approx(tile_transform)


def test_window_placement_not_double_counted():
    # Regression for the §2.6 georef bug: building the coarsened transform from the
    # WINDOW affine + global (ti,tj) double-counts the read offset. Going through
    # tile_origin_transform, the first output pixel must sit just inside the window
    # (within ~2 tiles of its left/top edge), not ~2x the offset away.
    a, e, tile_px = 5.0, -5.0, 32
    c, f = 237099.0, 2370987.9
    row_off, col_off, win = 43790, 10828, 3000
    win_c, win_f = c + col_off * a, f + row_off * e
    win_transform = (a, 0.0, win_c, 0.0, e, win_f)

    ti, tj = tile_grid_for_window((win, win), row_off, col_off, tile_px)
    ti_min, tj_min = int(ti.min()), int(tj.min())
    tr = coarsened_transform(tile_origin_transform(win_transform, row_off, col_off),
                             ti_min, tj_min, tile_px)
    # Output origin within [window edge, window edge + 2 tiles].
    assert win_c <= tr.c <= win_c + 2 * tile_px * a
    assert win_f + 2 * tile_px * e <= tr.f <= win_f


def test_own_tile_zero_fraction():
    tile_px, row0, col0 = 4, 0, 0
    window = np.ones((20, 20), dtype=np.uint8)
    # Zero out the own tile of (ti=2, tj=2): window rows/cols [8,12).
    window[8:12, 8:12] = 0
    ti = np.array([1, 2])
    tj = np.array([1, 2])
    zf = own_tile_zero_fraction(window, ti, tj, tile_px=tile_px, row0=row0, col0=col0)
    assert zf[0] == 0.0 and zf[1] == 1.0
