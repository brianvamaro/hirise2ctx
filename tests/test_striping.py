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
