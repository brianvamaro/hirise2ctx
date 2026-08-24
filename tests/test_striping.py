"""Unit tests for src/striping.py — the regional-map frame-block artifact analysis + A1 fix.

All synthetic, no downloads. Covers the A1 normalization (the mitigation), eta^2 (the decisive
frame-coherence statistic), and detrend.
"""
import numpy as np
import pytest

from src import striping as st


def test_a1_apply_maps_to_reference():
    rng = np.random.default_rng(0)
    arr = np.clip(rng.normal(90, 12, size=(200, 200)), 1, 255).astype(np.uint8)  # a "dark" frame
    med, iqr = st.a1_stats(arr)
    out = st.a1_apply(arr, med, iqr, m0=125.0, s0=27.7)
    v = out[out > 0].astype(float)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    # after normalization the robust center/scale match the reference
    assert abs(np.median(v) - 125.0) <= 2.0
    assert abs(np.subtract(*np.percentile(v, [75, 25])) - 27.7) <= 3.0


def test_a1_preserves_nodata():
    arr = np.full((50, 50), 100, np.uint8)
    arr[:10] = 0  # nodata band
    out = st.a1_normalize_window(arr)
    assert (out[:10] == 0).all()           # nodata stays nodata
    assert (out[10:] > 0).all()


def test_a1_monotonic_preserves_within_frame_order():
    # A1 is a monotonic remap, so within-frame pixel ordering (texture) is preserved.
    arr = np.arange(1, 201, dtype=np.uint8).reshape(10, 20)
    out = st.a1_normalize_window(arr)
    flat_in = arr.ravel().astype(float)
    flat_out = out.ravel().astype(float)
    order_in = np.argsort(flat_in, kind="stable")
    assert np.all(np.diff(flat_out[order_in]) >= 0)


def test_eta2_high_when_groups_differ():
    # three frames with distinct mean levels -> high eta^2
    labels = np.repeat(np.arange(3), 100).reshape(30, 10)
    vals = labels.astype(float) * 5.0 + np.random.default_rng(1).normal(0, 0.1, labels.shape)
    finite = np.ones_like(labels, bool)
    assert st.eta2(vals, labels, finite) > 0.95


def test_eta2_low_when_label_is_noise():
    rng = np.random.default_rng(2)
    vals = rng.normal(0, 1, (40, 40))
    labels = rng.integers(0, 8, (40, 40))   # labels unrelated to values
    finite = np.ones_like(labels, bool)
    assert st.eta2(vals, labels, finite) < 0.1


def test_eta2_nan_safe_under_roll():
    vals = np.random.default_rng(3).normal(0, 1, (30, 30))
    vals[:5] = np.nan
    labels = np.repeat(np.arange(3), 300).reshape(30, 30) % 3
    finite = np.isfinite(vals)
    rolled = np.roll(vals, (7, 3), (0, 1))
    assert np.isfinite(st.eta2(rolled, labels, finite))   # roll drags NaN in; must not crash/NaN


def test_detrend_removes_trend_keeps_small_scale():
    y, x = np.mgrid[0:120, 0:120]
    trend = 5.0 + 0.05 * x + 0.03 * y                       # large-scale (geology)
    bumps = 0.5 * np.sin(x * 1.5) * np.sin(y * 1.5)         # small-scale (boulders)
    resid, finite = st.detrend((trend + bumps).astype(float), sig=20.0)
    core = resid[30:90, 30:90]                              # interior, avoid edge/reflect effects
    assert finite.all()
    assert abs(np.nanmean(core)) < 0.2                      # large-scale trend removed
    # small-scale structure preserved: resid tracks the bumps
    r = np.corrcoef(core.ravel(), bumps[30:90, 30:90].ravel())[0, 1]
    assert r > 0.9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --- frame_labels_on bbox prefilter (R06) ------------------------------------------------------
# The prefilter is a pure speed fix, so the contract is *bit-identity* with the unfiltered
# rasterize, not "close enough". See DECISIONS 2026-08-24b: without it the step-11 A1 array
# timed out at 10 h with one core pegged, because the map driver calls this 144x per tile.

def _synthetic_frames(n=80, *, seed=0):
    """`n` dissolved-frame-like strips spread over a 40 km x 40 km extent, in a GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    rng = np.random.default_rng(seed)
    polys = []
    for i in range(n):
        x0 = float(rng.uniform(0, 38_000))
        y0 = float(rng.uniform(0, 38_000))
        w = float(rng.uniform(1_500, 6_000))
        h = float(rng.uniform(1_500, 6_000))
        polys.append(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]))
    return gpd.GeoDataFrame({"pid": [f"F{i:03d}" for i in range(n)]}, geometry=polys, crs=None)


def _unfiltered_labels(transform, shape, frames, dtype="int32"):
    """`frame_labels_on` as it was before the prefilter: every frame, every time."""
    from rasterio.features import rasterize
    return rasterize([(g, i) for i, g in enumerate(frames.geometry)], out_shape=tuple(shape),
                     transform=transform, fill=-1, dtype=dtype, all_touched=False)


def test_frame_labels_on_prefilter_is_identical():
    from rasterio.transform import from_origin

    frames = _synthetic_frames()
    shape = (512, 512)
    n_touched = 0
    # walk a grid of windows across the extent, incl. windows that touch no frame at all
    for r in range(0, 40_000, 5_000):
        for c in range(0, 40_000, 5_000):
            tr = from_origin(float(c), 40_000.0 - float(r), 10.0, 10.0)
            got = st.frame_labels_on(tr, shape, frames)
            want = _unfiltered_labels(tr, shape, frames)
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(got, want)
            n_touched += int((got >= 0).any())
    assert n_touched > 10, "degenerate fixture: almost no window saw a frame"


def test_frame_labels_on_prefilter_identical_off_extent_and_empty():
    from rasterio.transform import from_origin

    frames = _synthetic_frames(n=12, seed=3)
    shape = (64, 64)
    # a window nowhere near any frame: must be all -1, and match the unfiltered path
    tr = from_origin(-500_000.0, -500_000.0, 10.0, 10.0)
    got = st.frame_labels_on(tr, shape, frames)
    assert (got == -1).all()
    np.testing.assert_array_equal(got, _unfiltered_labels(tr, shape, frames))
    # zero frames: still the -1 fill, not a rasterize crash
    empty = frames.iloc[:0]
    tr2 = from_origin(0.0, 40_000.0, 10.0, 10.0)
    assert (st.frame_labels_on(tr2, shape, empty) == -1).all()


def test_frame_labels_on_prefilter_preserves_overlap_precedence():
    """Overlapping frames: the higher index must still win, as `rasterize` order dictates."""
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import box

    # frame 0 is far away (must be prefiltered out); 1 and 2 overlap inside the window
    frames = gpd.GeoDataFrame(geometry=[box(-9e5, -9e5, -8e5, -8e5),
                                        box(0, 0, 600, 600),
                                        box(300, 300, 900, 900)], crs=None)
    tr = from_origin(0.0, 1_000.0, 10.0, 10.0)
    got = st.frame_labels_on(tr, (100, 100), frames)
    np.testing.assert_array_equal(got, _unfiltered_labels(tr, (100, 100), frames))
    assert 0 not in np.unique(got)                  # the distant frame contributes nothing
    assert set(np.unique(got).tolist()) == {-1, 1, 2}


def test_frame_labels_on_prefilter_is_faster():
    """Guards the actual point of the change; generous factor so it is not flaky on CI."""
    import time

    from rasterio.transform import from_origin

    frames = _synthetic_frames(n=80, seed=1)
    tr = from_origin(20_000.0, 22_000.0, 5.0, 5.0)
    shape = (1024, 1024)                            # one window: sees a handful of 80 frames
    bounds = st.frame_bounds(frames)
    assert len(st.frames_hitting(bounds, tr, shape)) < 80

    st.frame_labels_on(tr, shape, frames, bounds=bounds)     # warm
    _unfiltered_labels(tr, shape, frames)
    t0 = time.perf_counter()
    for _ in range(3):
        st.frame_labels_on(tr, shape, frames, bounds=bounds)
    t_new = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(3):
        _unfiltered_labels(tr, shape, frames)
    t_old = time.perf_counter() - t0
    assert t_new < t_old, f"prefilter not faster: {t_new:.3f}s vs {t_old:.3f}s"


def test_frame_bounds_matches_geometry_order():
    frames = _synthetic_frames(n=5, seed=7)
    b = st.frame_bounds(frames)
    assert b.shape == (5, 4)
    for i, g in enumerate(frames.geometry):
        np.testing.assert_allclose(b[i], np.array(g.bounds))
    assert st.frame_bounds(frames.iloc[:0]).shape == (0, 4)


def test_frame_labels_on_accepts_a_bare_transform_tuple():
    """`CtxWindow.transform` is a plain 6-tuple in places, and `rasterize` accepts either."""
    frames = _synthetic_frames(n=20, seed=11)
    tr = (10.0, 0.0, 5_000.0, 0.0, -10.0, 25_000.0)
    got = st.frame_labels_on(tr, (128, 128), frames)
    np.testing.assert_array_equal(got, _unfiltered_labels(tr, (128, 128), frames))
    assert st.window_extent(tr, (128, 128)) == (5_000.0, 23_720.0, 6_280.0, 25_000.0)


def test_window_extent_covers_a_rotated_transform():
    """Two corners would land on the wrong diagonal and under-cover the window."""
    from affine import Affine

    tr = Affine.rotation(30.0) * Affine.scale(10.0, -10.0)
    x0, y0, x1, y1 = st.window_extent(tr, (100, 100))
    corners = [tr * (c, r) for c, r in ((0, 0), (100, 0), (0, 100), (100, 100))]
    assert x0 <= min(p[0] for p in corners) and x1 >= max(p[0] for p in corners)
    assert y0 <= min(p[1] for p in corners) and y1 >= max(p[1] for p in corners)
