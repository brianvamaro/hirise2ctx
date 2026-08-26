"""Tests for `src/fcompose.py` — the PLAN_FBuild Stage D composite core.

The two things most likely to ship as silent bugs are (a) the TI axis, which increases NORTHWARD so a
naive `raster[TI - TI_min]` gives a vertically mirrored map, and (b) the exact-160 m global lattice vs
the 159.9991835 m mosaic grid. Both are pinned here, the second against the real map tiles when they
are on disk.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import fcompose as fc
from src import leveling as lv

REF_PITCH = 159.9991835298017          # the Murray coarse pitch (32 x 4.9999744853063 m/px)


def grid(h=40, w=50, c=-711136.371096145, f=2133729.111655494, pitch=REF_PITCH) -> fc.TileGrid:
    Kj, Ki, dx, dy, tie = fc.tile_index_map((pitch, 0.0, c, 0.0, -pitch, f), h, w, "SYNTH")
    return fc.TileGrid("SYNTH", (pitch, 0.0, c, 0.0, -pitch, f), h, w, "", Kj, Ki, dx, dy, tie)


# --------------------------------------------------------------------------- the index map
def test_index_map_is_a_constant_integer_shift():
    g = grid()
    col = np.arange(g.width)
    row = np.arange(g.height)
    a, _, c, _, e, f = g.transform
    assert np.array_equal(g.cols_of_TJ(np.round((c + (col + 0.5) * a) / lv.TILE_M).astype(int)), col)
    assert np.array_equal(g.rows_of_TI(np.round((f + (row + 0.5) * e) / lv.TILE_M).astype(int)), row)


def test_ti_increases_northward_so_row_zero_is_the_HIGHEST_ti():
    """The single easiest Stage-D bug: TI is a y index, not a raster row."""
    g = grid()
    ti_lo, ti_hi = g.TI_range()
    assert g.rows_of_TI(np.array([ti_hi]))[0] == 0
    assert g.rows_of_TI(np.array([ti_lo]))[0] == g.height - 1


def test_index_map_rejects_a_grid_that_is_not_a_pure_translation():
    # a 2x-coarser pitch cannot be a constant shift of a 160 m lattice
    with pytest.raises(ValueError, match="not constant"):
        fc.tile_index_map((320.0, 0.0, 0.0, 0.0, -320.0, 0.0), 10, 10, "BAD")


def test_index_map_reports_the_subpixel_translation():
    g = grid()
    assert 0.0 <= g.dx_m <= lv.TILE_M / 2 and 0.0 <= g.dy_m <= lv.TILE_M / 2
    assert g.tie_margin_m == pytest.approx(lv.TILE_M / 2 - g.dx_m)


def test_tj_ti_ranges_cover_exactly_the_raster():
    g = grid()
    tj0, tj1 = g.TJ_range()
    ti0, ti1 = g.TI_range()
    assert tj1 - tj0 + 1 == g.width and ti1 - ti0 + 1 == g.height


def test_bbox_intersection_prescreen():
    g = grid()
    ti0, ti1 = g.TI_range()
    tj0, tj1 = g.TJ_range()
    assert fc.bbox_intersects_tile((ti0, ti1, tj0, tj1), g)
    assert fc.bbox_intersects_tile((ti1, ti1 + 50, tj1, tj1 + 50), g)      # touches one corner
    assert not fc.bbox_intersects_tile((ti1 + 1, ti1 + 50, tj0, tj1), g)   # just north
    assert not fc.bbox_intersects_tile((ti0, ti1, tj1 + 1, tj1 + 50), g)   # just east
    assert not fc.bbox_intersects_tile((0, -1, 0, -1), g)                  # empty frame


# --------------------------------------------------------------------------- accumulation
def _add(acc, g, TI, TJ, prob, off=0.0, idx=0, inc=45.0, src=0):
    rows, cols = fc.frame_rows_cols(g, np.asarray(TI), np.asarray(TJ))
    return acc.add_frame(rows, cols, lv.logit(np.asarray(prob, float)) + off,
                         frame_idx=idx, incidence=inc, src_code=src)


def test_composite_is_the_mean_of_leveled_logits_then_one_sigmoid():
    g = grid(4, 4)
    ti_hi = g.TI_range()[1]
    tj_lo = g.TJ_range()[0]
    acc = fc.TileAccum.zeros(g.shape)
    # deliberately ASYMMETRIC probabilities: for a symmetric pair (e.g. 0.2/0.8) mean-of-logits and
    # mean-of-probabilities coincide at 0.5 and the test could not tell the two rules apart.
    _add(acc, g, [ti_hi], [tj_lo], [0.10], off=0.5, idx=0)
    _add(acc, g, [ti_hi], [tj_lo], [0.60], off=-0.5, idx=1)
    res = acc.finish()
    want = lv.sigmoid((lv.logit(0.10) + 0.5 + lv.logit(0.60) - 0.5) / 2)
    assert res["prob_raw"][0, 0] == pytest.approx(want, abs=1e-6)
    assert res["n_frames"][0, 0] == 2
    # and it is NOT the mean of the probabilities (0.35) — logit-domain averaging pulls it down
    assert res["prob_raw"][0, 0] == pytest.approx(0.2898, abs=1e-3)
    assert abs(float(res["prob_raw"][0, 0]) - 0.35) > 0.05


def test_uncovered_pixels_stay_nan():
    g = grid(4, 4)
    acc = fc.TileAccum.zeros(g.shape)
    _add(acc, g, [g.TI_range()[1]], [g.TJ_range()[0]], [0.5])
    res = acc.finish()
    assert np.isfinite(res["prob_raw"]).sum() == 1
    assert np.isnan(res["prob_raw"][1, 1]) and np.isnan(res["n_frames"][1, 1])


def test_overlap_dp_is_exactly_the_pairwise_max_not_an_approximation():
    """max over frame PAIRS |p_i - p_j| == max_f p_f - min_f p_f, so the O(k) running min/max IS
    the O(k^2) quantity the H6 overlap-QA layer asks for."""
    rng = np.random.default_rng(0)
    g = grid(3, 3)
    ti_hi, tj_lo = g.TI_range()[1], g.TJ_range()[0]
    acc = fc.TileAccum.zeros(g.shape)
    probs = rng.uniform(0.05, 0.95, 6)
    for k, p in enumerate(probs):
        _add(acc, g, [ti_hi], [tj_lo], [p], idx=k)
    res = acc.finish()
    brute = max(abs(a - b) for a in probs for b in probs)
    assert res["overlap_dp"][0, 0] == pytest.approx(brute, abs=1e-6)


def test_overlap_dp_is_nan_for_single_frame_pixels():
    g = grid(3, 3)
    acc = fc.TileAccum.zeros(g.shape)
    _add(acc, g, [g.TI_range()[1]], [g.TJ_range()[0]], [0.4])
    assert np.isnan(acc.finish()["overlap_dp"][0, 0])


def test_primary_frame_is_the_best_illuminated_contributor():
    g = grid(3, 3)
    ti_hi, tj_lo = g.TI_range()[1], g.TJ_range()[0]
    acc = fc.TileAccum.zeros(g.shape)
    _add(acc, g, [ti_hi], [tj_lo], [0.4], idx=7, inc=55.0)
    _add(acc, g, [ti_hi], [tj_lo], [0.6], idx=3, inc=41.0)      # lower incidence wins
    _add(acc, g, [ti_hi], [tj_lo], [0.5], idx=9, inc=49.0)
    res = acc.finish()
    assert res["primary_frame"][0, 0] == 3
    assert res["incidence"][0, 0] == pytest.approx(41.0)


def test_offset_provenance_takes_the_WORST_contributor():
    g = grid(3, 3)
    ti_hi, tj_lo = g.TI_range()[1], g.TJ_range()[0]
    acc = fc.TileAccum.zeros(g.shape)
    _add(acc, g, [ti_hi], [tj_lo], [0.4], idx=0, src=fc.OFFSET_SOURCE_CODE["solved"])
    _add(acc, g, [ti_hi], [tj_lo], [0.4], idx=1, src=fc.OFFSET_SOURCE_CODE["interpolated"])
    assert acc.finish()["offset_source"][0, 0] == fc.OFFSET_SOURCE_CODE["interpolated"]


def test_out_of_bounds_tiles_are_dropped_not_wrapped():
    g = grid(4, 4)
    ti_hi, tj_lo = g.TI_range()[1], g.TJ_range()[0]
    acc = fc.TileAccum.zeros(g.shape)
    n = _add(acc, g, [ti_hi + 10, ti_hi], [tj_lo - 10, tj_lo], [0.3, 0.7])
    assert n == 1
    assert np.isfinite(acc.finish()["prob_raw"]).sum() == 1


def test_leveling_cancels_a_planted_per_frame_bias():
    """Two frames viewing one shared truth field through opposite biases; the H4 offsets are the
    negated biases, so the leveled composite must recover the truth exactly."""
    rng = np.random.default_rng(3)
    g = grid(10, 10)
    ti_hi, tj_lo = g.TI_range()[1], g.TJ_range()[0]
    ti = np.repeat(np.arange(ti_hi - 9, ti_hi + 1), 10)
    tj = np.tile(np.arange(tj_lo, tj_lo + 10), 10)
    truth_logit = rng.normal(-1.0, 1.0, ti.size)
    bias = np.array([+0.7, -0.4])
    acc = fc.TileAccum.zeros(g.shape)
    for k, b in enumerate(bias):
        _add(acc, g, ti, tj, lv.sigmoid(truth_logit + b), off=-b, idx=k)
    res = acc.finish()
    rows, cols = fc.frame_rows_cols(g, ti, tj)
    assert np.allclose(res["prob_raw"][rows, cols], lv.sigmoid(truth_logit), atol=1e-5)


# --------------------------------------------------------------------------- scoring support
def test_partition_composite_takes_each_pixels_owner_frame():
    labels = np.array([[0, 1], [1, -1]], dtype=np.int32)
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    per_frame = {0: (rows, cols, np.array([0.1, 0.2, 0.3, 0.4], np.float32)),
                 1: (rows, cols, np.array([0.9, 0.8, 0.7, 0.6], np.float32))}
    out = fc.partition_composite(per_frame, labels)
    assert out[0, 0] == pytest.approx(0.1)      # owner 0
    assert out[0, 1] == pytest.approx(0.8)      # owner 1
    assert out[1, 0] == pytest.approx(0.7)      # owner 1
    assert np.isnan(out[1, 1])                  # unowned


def test_windows_over_grid_covers_the_tile_without_overlap():
    g = grid(100, 100)
    wins = list(fc.windows_over_grid(g, 40, min_frac=0.0))
    area = sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in wins)
    assert area == 100 * 100
    seen = np.zeros(g.shape, int)
    for r0, r1, c0, c1 in wins:
        seen[r0:r1, c0:c1] += 1
    assert (seen == 1).all()


def test_windows_over_grid_drops_slivers():
    g = grid(100, 100)
    assert all((r1 - r0) * (c1 - c0) >= 0.5 * 40 * 40
               for r0, r1, c0, c1 in fc.windows_over_grid(g, 40, min_frac=0.5))


# --------------------------------------------------------------------------- against the real map
# TWO generations of shipped map now exist on disk, and they sit on different lattices, so the
# pin has to name which one it is measuring (step 12, 2026-08-25):
#
#   reports/map_region      the PROMOTED product, rendered on COARSE_GRID_ID. One lattice: the
#                           per-tile origins share a single sub-cell phase.
#   reports/map_region_g1   the ARCHIVED pre-R01 product. **26 distinct sub-cell phases** — every
#                           tile on its own — which is the R01 defect itself.
#
# Keeping both pinned is the point: the archived row is the historical record the 2026-07-28
# measurement was made against, and the promoted row is what any consumer reads today. A single
# unlabelled pin would have silently become a claim about whichever directory happened to exist.
REAL_KJ_KI = {
    "map_region": {"E-12_N32": (-4443, 13334), "E0_N44": (1, 17780), "E8_N44": (2965, 17780),
                   "E16_N44": (5929, 17780)},
    "map_region_g1": {"E-12_N32": (-4444, 13335), "E0_N44": (1, 17781), "E8_N44": (2965, 17781),
                      "E16_N44": (5929, 17781)},
}


@pytest.mark.parametrize("generation,tile,expect",
                         [(gen, t, e) for gen, d in sorted(REAL_KJ_KI.items())
                          for t, e in sorted(d.items())])
def test_real_map_tiles_have_the_measured_lattice_shift(repo_root, generation, tile, expect):
    """Regression-pins the measured global-lattice shifts per shipped generation (both map dirs
    are gitignored, so this skips on a fresh clone)."""
    ref = repo_root / "reports" / generation / f"{tile}_prob_raw.tif"
    if not ref.exists():
        pytest.skip(f"{ref} not on disk")
    g = fc.tile_grid_from_raster(ref, tile)
    assert (g.Kj, g.Ki) == expect
    assert g.height == g.width == 1479
    assert g.dx_m <= lv.TILE_M / 2 + 1e-6 and g.dy_m <= lv.TILE_M / 2 + 1e-6


def test_the_promoted_arms_share_one_lattice_and_the_archived_one_does_not(repo_root):
    """The R01 defect, stated as a property of the products rather than of one tile's indices.

    `rasterio.merge` floors each tile's fractional destination offset, so a per-tile sub-cell
    phase becomes a whole-cell displacement. The promoted arms must therefore present ONE phase
    and the archived product many — this is what `mosaic_geotiffs(require_shared_lattice=True)`
    enforces at build time, checked here against what is actually on disk.
    """
    import rasterio

    def phases(d):
        out = set()
        for p in (repo_root / "reports" / d).glob("*_prob_raw.tif"):
            if p.name.startswith("regional_"):
                continue
            with rasterio.open(p) as ds:
                t = ds.transform
            out.add((round(t.c % t.a, 3), round(t.f % abs(t.e), 3)))
        return out

    if not (repo_root / "reports" / "map_region").is_dir():
        pytest.skip("no map arms on disk")
    for arm in ("map_region", "map_a1"):
        # a phase of ~0 or ~cell-size are the same lattice (float modulo lands either side)
        got = {tuple(0.0 if (v < 1e-3 or abs(v - 160.0) < 1.0) else v for v in ph)
               for ph in phases(arm)}
        assert got == {(0.0, 0.0)}, f"{arm} is not on one lattice: {sorted(phases(arm))}"
    archived = repo_root / "reports" / "map_region_g1"
    if archived.is_dir():
        assert len(phases("map_region_g1")) > 1, (
            "the archived pre-R01 product should show many sub-cell phases — that defect is "
            "the whole reason the rebuild re-rendered these tiles")


def test_the_e0_column_is_flagged_as_a_rounding_tie(repo_root):
    """E0's map pixel centres sit ~80 m (a half cell) from the global node they map to, i.e. right on
    the cell boundary — an irreducible <=80 m placement ambiguity that must stay visible.

    Survives the step-12 promotion unchanged: the tie is a property of E0's longitude relative to
    the clon_0 origin, not of which generation rendered it."""
    ref = repo_root / "reports" / "map_region" / "E0_N44_prob_raw.tif"
    if not ref.exists():
        pytest.skip(f"{ref} not on disk")
    g = fc.tile_grid_from_raster(ref, "E0_N44")
    assert g.tie_margin_m < 1.0
    assert g.dx_m > 79.0
