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


# ============================================================================
# R01 — the globally anchored coarse grid. See DECISIONS 2026-08-06x.
# ============================================================================
import math

from src.mapping import (
    COARSE_GRID_ID, MURRAY_NATIVE_M, MURRAY_PPD, MURRAY_RADIUS_M,
    assert_murray_sphere, assert_shared_lattice, global_cell_transform,
    global_native_origin, tile_grid_phase,
)

_CELL = 32 * MURRAY_NATIVE_M
_MURRAY_WKT = ('PROJCS["Mars_2015",GEOGCS["GCS",DATUM["D",'
               'SPHEROID["Mars_2015",3396190,169.894447223612]]]]')


def _tile_transform(lon_deg: int, lat_deg: int):
    """The affine of a real Murray V01 tile: origin on the lon0/lat0 native lattice."""
    a = MURRAY_NATIVE_M
    return (a, 0.0, lon_deg * MURRAY_PPD * a, 0.0, -a, (lat_deg + 4) * MURRAY_PPD * a)


def test_the_native_constant_matches_the_murray_definition():
    assert MURRAY_PPD * 4 == 47420
    assert MURRAY_NATIVE_M == pytest.approx(math.pi * MURRAY_RADIUS_M / 180.0 / MURRAY_PPD)


def test_global_native_origin_is_integral_on_real_tile_geometry():
    for lon, lat in [(0, 0), (-12, 44), (8, 40), (152, -8)]:
        gr, gc = global_native_origin(_tile_transform(lon, lat))
        assert gc == lon * MURRAY_PPD
        assert gr == -(lat + 4) * MURRAY_PPD


def test_global_native_origin_rejects_an_off_lattice_raster():
    a = MURRAY_NATIVE_M
    with pytest.raises(ValueError, match="off the Murray global native lattice"):
        global_native_origin((a, 0.0, 12345.6, 0.0, -a, 98765.4))


def test_the_phase_walks_four_native_px_per_tile_and_is_never_constant():
    """The defect itself: gcd(47420, 32) = 4, so 47420 % 32 = 28 = -4 mod 32."""
    assert 47420 % 32 == 28
    assert math.gcd(47420, 32) == 4
    phases = [tile_grid_phase(_tile_transform(lon, 40))[1] for lon in range(-12, 20, 4)]
    assert len(set(phases)) > 1, "if every tile shared a phase there would be no R01"
    # adjacent tiles differ by exactly 4 native px = 20 m
    for p, q in zip(phases, phases[1:]):
        assert (q - p) % 32 == 4


def test_tile_grid_phase_uses_the_pinned_convention_not_its_complement():
    """M6: `(-gr) % 32` vs `gr % 32`. At N44 both are 16, so a test written only on an
    N44 tile is blind to the mutant — these latitudes are chosen to separate them."""
    got = {lat: tile_grid_phase(_tile_transform(0, lat))[0] for lat in (44, 40, 36, 32)}
    assert got == {44: 16, 40: 20, 36: 24, 32: 28}
    complement = {lat: (-v) % 32 for lat, v in got.items()}
    assert complement == {44: 16, 40: 12, 36: 8, 32: 4}
    assert got != complement, "the two conventions must be distinguishable in this fixture"


def test_the_phase_lands_the_first_cell_on_a_global_boundary():
    """The runtime invariant the drivers assert: origin + phase is cell-aligned."""
    for lon, lat in [(-12, 44), (-8, 40), (0, 36), (8, 32), (16, 44)]:
        t = _tile_transform(lon, lat)
        gr, gc = global_native_origin(t)
        pr, pc = tile_grid_phase(t)
        assert (gr + pr) % 32 == 0
        assert (gc + pc) % 32 == 0


def test_global_cell_transform_is_bit_identical_across_tiles():
    """Why the canonical constant: the cached sidecars carry FOUR distinct pixel sizes and
    none equals the exact value, so per-tile `a` would re-import that spread."""
    a = global_cell_transform(-100, 250)
    b = global_cell_transform(-100, 999)
    assert a.a == b.a and a.e == b.e
    assert a.c == 250 * _CELL and a.f == 100 * _CELL
    # offsets between any two cells are exactly integral in cells
    assert (b.c - a.c) / _CELL == pytest.approx(749, abs=1e-9)


def test_assert_shared_lattice_accepts_the_global_grid():
    assert_shared_lattice([global_cell_transform(i, j)
                           for i, j in [(-17781, -4444), (-17781, 1000), (-12000, 7407)]])


def test_assert_shared_lattice_rejects_a_half_cell_phase():
    good = global_cell_transform(-100, 250)
    from rasterio.transform import Affine
    bad = Affine(good.a, 0.0, good.c + _CELL / 2, 0.0, good.e, good.f)
    with pytest.raises(ValueError, match="not on murray_v01"):
        assert_shared_lattice([good, bad])


def test_assert_shared_lattice_rejects_the_real_20_m_tile_phase():
    """20 m is the measured adjacent-tile offset — an eighth of a cell. A tolerance loose
    enough to accept it would accept the whole defect."""
    good = global_cell_transform(-100, 250)
    from rasterio.transform import Affine
    bad = Affine(good.a, 0.0, good.c + 20.0, 0.0, good.e, good.f)
    with pytest.raises(ValueError, match="phase"):
        assert_shared_lattice([good, bad])


def test_assert_murray_sphere_measures_rather_than_assumes():
    assert assert_murray_sphere(_MURRAY_WKT) == MURRAY_RADIUS_M
    assert "R3396190" in COARSE_GRID_ID, "the id asserts the radius, so it must be checked"
    with pytest.raises(ValueError, match="does not describe this product"):
        assert_murray_sphere(_MURRAY_WKT.replace("3396190", "3389500"))   # IAU mean radius
    with pytest.raises(ValueError, match="no SPHEROID"):
        assert_murray_sphere('PROJCS["nope"]')


def test_predict_window_global_grid_produces_global_cells_and_a_global_affine():
    win = _window()
    legacy = predict_window(win, _FakeEmbedder(), _FakeHead(), tile_px=8)
    glob = predict_window(win, _FakeEmbedder(), _FakeHead(), tile_px=8,
                          global_grid=(-17781, -4444, 0, 0))
    # Same predictions, relabelled onto the global lattice.
    assert np.allclose(np.nan_to_num(legacy.prob), np.nan_to_num(glob.prob))
    assert glob.ti.min() == legacy.ti.min() - 17781
    assert glob.tj.min() == legacy.tj.min() - 4444
    expected = global_cell_transform(glob.ti_min, glob.tj_min, 8)
    assert glob.transform.c == pytest.approx(expected.c)
    assert glob.transform.f == pytest.approx(expected.f)


def test_predict_window_global_grid_does_not_break_window_indexing():
    """The ordering trap: promoting ti/tj to global BEFORE embed_window /
    own_tile_zero_fraction drives the slice origin to ~-521,600, so `valid` goes all-False
    and every prob is NaN. A large negative cell origin must still predict."""
    glob = predict_window(_window(), _FakeEmbedder(), _FakeHead(), tile_px=8,
                          global_grid=(-16300, -4444, 0, 0))
    assert np.isfinite(glob.prob).all(), "global cell indices leaked into the window slicer"
    assert glob.n_valid == glob.ti.size


def test_predict_window_legacy_path_is_untouched():
    a = predict_window(_window(), _FakeEmbedder(), _FakeHead(), tile_px=8)
    b = predict_window(_window(), _FakeEmbedder(), _FakeHead(), tile_px=8, global_grid=None)
    assert a.transform == b.transform and np.array_equal(a.ti, b.ti)


# ---------------------------------------------------------------------------
# load_regional_mosaic — the read-first consumer seam
# ---------------------------------------------------------------------------
#
# `scripts/map_mosaics.py` is the sole producer of `regional_{layer}_mosaic.tif`, and the
# tags it stamps (SIZE_FLOOR_* + MOSAIC_*) are the only record of the product's units and
# provenance. Notebook 24 used to rebuild those files with `mosaic_geotiffs(out_path=...)`,
# which replaces the tagged product with an untagged look-alike — and notebooks are not
# covered by the test-side write guard. These tests pin the three properties that matter:
# it reads the tagged file when present, it never writes, and the fallback is visibly
# distinguishable from the real thing.

def _write_tile(path, arr, transform, *, tags=None):
    from src.mapping import MURRAY_RADIUS_M, write_geotiff
    wkt = (f'PROJCS["m",GEOGCS["g",DATUM["d",SPHEROID["s",{MURRAY_RADIUS_M},0]],'
           'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
           'PROJECTION["Equirectangular"],UNIT["metre",1]]')
    write_geotiff(path, arr, transform, wkt, tags=tags)


def _two_adjacent_tiles(tmp_path, layer="abundance", n=8):
    """Two side-by-side rasters on the one global coarse lattice, plus their merged truth."""
    from src.mapping import global_cell_transform
    a = np.arange(n * n, dtype=np.float32).reshape(n, n)
    b = a + 1000.0
    _write_tile(tmp_path / f"E0_N0_{layer}.tif", a, global_cell_transform(0, 0))
    _write_tile(tmp_path / f"E4_N0_{layer}.tif", b, global_cell_transform(0, n))
    return np.hstack([a, b])


def test_load_regional_mosaic_prefers_the_tagged_file_over_re_merging(tmp_path):
    """A prebuilt mosaic wins even when it disagrees with the tiles — it is the product."""
    from src.mapping import global_cell_transform, load_regional_mosaic

    _two_adjacent_tiles(tmp_path)
    sentinel = np.full((8, 16), -7.0, dtype=np.float32)
    _write_tile(tmp_path / "regional_abundance_mosaic.tif", sentinel,
                global_cell_transform(0, 0),
                tags={"SIZE_FLOOR_M": "1.5", "MOSAIC_N_TILES": "2",
                      "MOSAIC_BUILT_BY": "scripts/map_mosaics.py"})

    arr, _, _, meta = load_regional_mosaic(tmp_path, "abundance")
    assert meta["source"] == "prebuilt"
    assert meta["n_tiles"] == 2
    assert meta["tags"]["SIZE_FLOOR_M"] == "1.5"
    assert np.all(arr == -7.0), "re-merged the tiles instead of reading the shipped mosaic"


def test_load_regional_mosaic_never_writes_the_mosaic(tmp_path):
    """The whole point: reading must not create or touch `regional_*_mosaic.tif`."""
    from src.mapping import load_regional_mosaic

    truth = _two_adjacent_tiles(tmp_path)
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
              for p in sorted(tmp_path.iterdir())}
    arr, _, _, meta = load_regional_mosaic(tmp_path, "abundance")
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(tmp_path.iterdir())}

    assert after == before, f"the loader wrote to disk: {set(after) ^ set(before) or 'mtime'}"
    assert not (tmp_path / "regional_abundance_mosaic.tif").exists()
    assert meta["source"] == "merged_in_memory"
    np.testing.assert_array_equal(arr, truth)


def test_load_regional_mosaic_fallback_reports_no_tags(tmp_path):
    """An in-memory merge has no size-floor basis, and must not look like it does."""
    from src.mapping import load_regional_mosaic

    _two_adjacent_tiles(tmp_path)
    _, _, _, meta = load_regional_mosaic(tmp_path, "abundance")
    assert meta["tags"] == {} and meta["path"] is None
    assert meta["n_tiles"] == 2


def test_load_regional_mosaic_can_refuse_to_fall_back(tmp_path):
    from src.mapping import load_regional_mosaic

    _two_adjacent_tiles(tmp_path)
    with pytest.raises(FileNotFoundError, match="map_mosaics"):
        load_regional_mosaic(tmp_path, "abundance", allow_build=False)


def test_load_regional_mosaic_raises_when_nothing_is_there(tmp_path):
    from src.mapping import load_regional_mosaic

    with pytest.raises(FileNotFoundError, match="prob_raw"):
        load_regional_mosaic(tmp_path, "prob_raw")


def test_load_regional_mosaic_ignores_the_mosaic_when_globbing_tiles(tmp_path):
    """`*_abundance.tif` also matches a stray `regional_abundance.tif`-style sibling."""
    from src.mapping import global_cell_transform, load_regional_mosaic

    truth = _two_adjacent_tiles(tmp_path)
    _write_tile(tmp_path / "regional_prob_abundance.tif",
                np.zeros((8, 16), np.float32), global_cell_transform(0, 0))
    arr, _, _, meta = load_regional_mosaic(tmp_path, "abundance")
    assert meta["n_tiles"] == 2
    np.testing.assert_array_equal(arr, truth)


def test_load_regional_mosaic_dtype_is_selectable(tmp_path):
    """562 MB per layer at float64; the plotting callers ask for float32."""
    from src.mapping import global_cell_transform, load_regional_mosaic

    _write_tile(tmp_path / "regional_abundance_mosaic.tif",
                np.zeros((4, 4), np.float32), global_cell_transform(0, 0))
    assert load_regional_mosaic(tmp_path, "abundance")[0].dtype == np.float64
    assert load_regional_mosaic(tmp_path, "abundance", dtype="float32")[0].dtype == np.float32
