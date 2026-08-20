"""R13 — the context-window nodata gate, and the fact that it is RECORDED.

The finding, restated executably: `own_tile_zero_fraction` tests the central `tile_px²` only.
At the frozen S=32 that is 1024 of the 9216 pixels `slice_context_boxes` hands the ViT, so
**88.9 % of what the embedder sees was never checked** — a tile whose own 32² is spotless can
sit against a mosaic gap, be embedded almost entirely black, and be predicted anyway. Measured
through the real `predict_window` before the fix: own zero-fraction 0.00, context zero-fraction
**0.8889**, and a finite probability 0.0845 emitted.

Impact, real frozen ViT + real shipped head, against the shipped E4_N44 IQR of 0.152: one
blackened 32-block in the ring moves p90 |ΔP| by 0.45 (≈3× IQR); 92 *scattered* black pixels by
0.70 (≈4.6×). Rarity is not the defence — on the shipped 26-tile map only ~1.5e-05 of cells are
affected, but each affected cell is wrong by several times the map's own spread.

Mutants these kill, all of which are green against the pre-R13 suite:
  M1  drop `& (zero_frac <= max_zero_fraction)` — the entire nodata gate disappears
  M2  the own-tile signature default 0.3 -> 1.0, or the context default 0.0 -> 1.0
  M3  `n_masked_nodata=0`
  M4  enforce the context gate but keep ONE conflated counter
  M5  enforce it but omit it from the sidecar — precisely the register's "Record" half
  M6  sum the per-window counters instead of de-duplicating the cell sets
"""
import inspect
import json

import numpy as np
import pytest

from src.fm_embeddings import slice_context_boxes
from src.mapping import (
    CONTEXT_ZERO_HIST_EDGES, CtxWindow, context_zero_fraction, context_zero_histogram,
    own_tile_zero_fraction, predict_window,
)


class _NullSrc:
    """Stand-in for the per-tile rasterio handle the drivers now hold open.

    The sweep drivers `open_tile(...)` once and pass the handle to `read_tile_window`
    (DECISIONS 2026-08-18c). Every test here fakes `read_tile_window`, so the handle is
    never read from -- it only has to exist and close.
    """

    def close(self):
        pass

class _FakeEmbedder:
    """Valid everywhere the real box geometry is valid, so the gate is what is under test."""
    def embed_window(self, arr, ti, tj, *, tile_px, row0, col0, pool, batch):
        _, valid = slice_context_boxes(arr, ti, tj, tile_px, row0, col0)
        emb = np.zeros((ti.size, 4), np.float32)
        emb[:, 0] = 0.5
        return emb, valid


class _FakeHead:
    def predict(self, emb):
        return emb[:, 0].astype(np.float64)


def _window(data, row_off=0, col_off=0):
    return CtxWindow(data=data, row_off=row_off, col_off=col_off,
                     transform=(5.0, 0.0, 0.0, 0.0, -5.0, 0.0), crs_wkt="LOCAL")


# ---------------------------------------------------------------- the helper (T1)

@pytest.mark.parametrize("row0,col0", [(0, 0), (7, 3), (-5, 11), (32, 32), (13, -29)])
def test_context_zero_fraction_matches_the_boxes_actually_sliced(row0, col0):
    """T1. The gate must measure the box the embedder consumes, bit for bit.

    Non-tile-aligned origins are the point: the lattice-block implementation crops the window
    to `(-row0) % tile_px`, so an off-by-one there is invisible at origin (0, 0) and wrong
    everywhere else. Kills a helper that measures the own tile, a 2× box, or forgets the
    `- tile_px` that centres it.
    """
    rng = np.random.default_rng(0)
    tile_px = 8
    window = rng.integers(0, 3, size=(97, 89), dtype=np.uint8)   # ~1/3 nodata, deliberately dense
    ti = np.arange(-2, 14, dtype=np.int64).repeat(16)
    tj = np.tile(np.arange(-2, 14, dtype=np.int64), 16)
    ti = ti + (row0 // tile_px)
    tj = tj + (col0 // tile_px)

    got = context_zero_fraction(window, ti, tj, tile_px=tile_px, row0=row0, col0=col0)
    boxes, valid = slice_context_boxes(window, ti, tj, tile_px, row0, col0)
    want_count = (boxes == 0).sum(axis=(1, 2))

    assert valid.any(), "fixture must produce some valid boxes or it tests nothing"
    assert want_count.sum() > 0, "and some of them must actually contain nodata"
    # exact on the COUNT (the helper returns float32, like `own_tile_zero_fraction`, so the
    # fraction itself is only float32-exact — the underlying integer must be dead on)
    assert np.array_equal(np.rint(got[valid].astype(np.float64) * (3 * tile_px) ** 2),
                          want_count)
    assert np.all(got[~valid] == 1.0), "a box that spills the window must not read as clean"
    assert np.all(np.isin(np.where(~valid)[0], np.where(got == 1.0)[0]))


def test_context_zero_fraction_is_not_the_own_tile_fraction():
    """The whole finding in one assertion: clean centre, filthy ring."""
    tile_px = 8
    window = np.zeros((3 * tile_px, 3 * tile_px), dtype=np.uint8)
    window[tile_px:2 * tile_px, tile_px:2 * tile_px] = 200          # own tile clean only
    ti = tj = np.array([1])
    assert own_tile_zero_fraction(window, ti, tj, tile_px=tile_px, row0=0, col0=0)[0] == 0.0
    assert context_zero_fraction(window, ti, tj, tile_px=tile_px, row0=0, col0=0)[0] == \
        pytest.approx(8 / 9)


def test_context_zero_fraction_handles_an_empty_and_a_too_small_window():
    small = np.full((10, 10), 5, dtype=np.uint8)
    assert context_zero_fraction(small, np.array([], np.int64), np.array([], np.int64),
                                 tile_px=8, row0=0, col0=0).size == 0
    # 10 px cannot hold a 24-px context box: every cell must read 1.0, not crash
    out = context_zero_fraction(small, np.array([1]), np.array([1]), tile_px=8, row0=0, col0=0)
    assert out.tolist() == [1.0]


def test_context_zero_histogram_counts_strictly_above_each_edge():
    frac = np.array([0.0, 0.01, 0.06, 0.4, 0.9])
    valid = np.array([True, True, True, True, False])
    got = context_zero_histogram(frac, valid, edges=(0.0, 0.05, 0.3))
    assert got.tolist() == [3, 2, 1]        # the invalid 0.9 cell is excluded from every bin
    assert len(context_zero_histogram(frac, valid)) == len(CONTEXT_ZERO_HIST_EDGES)


# ---------------------------------------------------------------- the gate (T2, T3)

def test_predict_window_masks_a_single_context_nodata_pixel():
    """T2 — THE defect test: it fails on pre-R13 code.

    5×5 tiles, ONE DN-0 pixel at (8, 8). Cell (2,2)'s own 32² is spotless and its context box
    is 1/576 nodata, so before the fix it was predicted with the gate reporting nothing. The
    same shape reproduced through the real `predict_window` at context fraction 0.889 emitted
    a finite 0.0845.

    Note the own-tile gate is genuinely blind here, not merely lenient: cell (1,1) *owns* that
    pixel and its own zero-fraction is 1/64 = 0.0156, comfortably inside the 0.3 threshold.
    Kills M1 and M4.
    """
    tile_px = 8
    data = np.full((5 * tile_px, 5 * tile_px), 200, dtype=np.uint8)
    data[tile_px, tile_px] = 0
    pred = predict_window(_window(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px)

    centre = np.where((pred.ti == 2) & (pred.tj == 2))[0]
    assert centre.size == 1
    assert np.isnan(pred.prob[centre[0]]), "a dirty context must not be predicted"
    assert pred.n_masked_nodata == 0, "no own tile is 30 % nodata; the old gate sees nothing"
    # the pixel lies in the context box of exactly these four enumerated cells
    assert sorted(map(tuple, pred.masked_context_cells.tolist())) == \
        [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_predict_window_counters_do_not_conflate_the_two_gates():
    """T3. One own-tile drop and one context drop must read as (1, 1), never (2, 0).

    Kills M4 — computing `usable` with the context check while keeping the single
    `(valid & ~usable)` counter, which would leave the sidecar under-reporting exactly as it
    does today while looking fixed.
    """
    tile_px = 8
    data = np.full((6 * tile_px, 6 * tile_px), 200, dtype=np.uint8)
    data[2 * tile_px:3 * tile_px, 2 * tile_px:3 * tile_px] = 0     # tile (2,2)'s OWN tile
    data[3 * tile_px, 5 * tile_px - 1] = 0                          # tile (4,4)'s ring only

    pred = predict_window(_window(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px,
                          max_zero_fraction=0.3, max_context_zero_fraction=0.0)
    own = pred.masked_own_cells
    ctx = pred.masked_context_cells
    assert own.tolist() == [[2, 2]], f"own-tile drops were {own.tolist()}"
    assert [2, 2] not in ctx.tolist(), "a cell must be attributed to ONE gate, not both"
    assert pred.n_masked_nodata == 1
    assert pred.n_masked_context_nodata == len(ctx) >= 1
    # every enumerated cell here has full context, so the two counters must account for
    # every NaN — a conflated counter would leave n_masked_context_nodata at 0
    assert pred.ti.size == pred.n_valid
    assert pred.n_masked_nodata + pred.n_masked_context_nodata == int(np.isnan(pred.prob).sum())


def test_a_clean_window_masks_nothing_and_costs_nothing():
    """The 0.0 default must not reject a legitimate tile: 8 of the 9 cached shipped Murray
    tiles contain literally zero DN-0 pixels, and 0.0 must be free on those."""
    tile_px = 8
    data = np.full((5 * tile_px, 5 * tile_px), 200, dtype=np.uint8)
    pred = predict_window(_window(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px)
    assert pred.n_masked_nodata == 0 and pred.n_masked_context_nodata == 0
    assert np.isfinite(pred.prob).all()
    assert pred.n_valid == pred.ti.size
    assert pred.context_zero_hist.tolist() == [0] * len(CONTEXT_ZERO_HIST_EDGES)


def test_the_gate_survives_the_global_grid_promotion():
    """R01 x R13: masked cells are reported in GLOBAL cell indices, while the gate is still
    computed against the LOCAL window origin. Promoting too early made `valid` all-False, so
    "the same drops, relabelled" is the assertion that separates the two."""
    tile_px = 8
    data = np.full((5 * tile_px, 5 * tile_px), 200, dtype=np.uint8)
    data[tile_px, tile_px] = 0
    local = predict_window(_window(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px)
    glob = predict_window(_window(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px,
                          global_grid=(-17781, -4444, 0, 0))
    assert glob.n_masked_context_nodata == local.n_masked_context_nodata > 0
    assert np.array_equal(glob.masked_context_cells,
                          local.masked_context_cells + np.array([-17781, -4444]))


# ---------------------------------------------------------------- recording (T4, T6, T7)

def test_predict_window_records_the_thresholds_it_used():
    """T4. Kills M5 at the seam: a product must carry the policy that made it."""
    pred = predict_window(_window(np.full((40, 40), 200, np.uint8)), _FakeEmbedder(),
                          _FakeHead(), tile_px=8, max_zero_fraction=0.25,
                          max_context_zero_fraction=0.125)
    assert pred.max_zero_fraction == 0.25
    assert pred.max_context_zero_fraction == 0.125
    assert len(pred.context_zero_hist) == len(CONTEXT_ZERO_HIST_EDGES)


def test_the_defaults_are_pinned_so_loosening_them_is_a_reviewed_edit():
    """T6 + T7, and T7 is a real regression: `max_zero_fraction`'s signature default was 0.5
    while every production driver passed 0.3, and `scripts/parity_check.py` took the
    signature default — so the one cross-machine gate ran a threshold nothing shipped with.

    Kills M2 in both parameters.
    """
    p = inspect.signature(predict_window).parameters
    assert p["max_zero_fraction"].default == 0.3
    assert p["max_context_zero_fraction"].default == 0.0, (
        "0.0 is the training distribution: 0 nodata pixels in 161,005 training context boxes")


# ---------------------------------------------------------------- the product boundary (T5)

def test_gate_summary_deduplicates_cells_shared_by_two_windows():
    """M6. Read windows overlap in pixels, and at grid phase 0 consecutive windows share one
    cell per axis seam — so SUMMING per-window counters over a tile double-counts. The
    re-validation criterion for this fix ("the context gate drops exactly N additional
    cells") is only checkable if the sidecar reports a de-duplicated set.
    """
    from scripts.map_region import gate_summary

    hist = np.zeros(len(CONTEXT_ZERO_HIST_EDGES), dtype=np.int64)
    shared = np.array([[10, 20]], dtype=np.int32)
    loaded = [
        {"_masked_own": np.zeros((0, 2), np.int32), "_masked_ctx": shared, "_ctx_hist": hist},
        {"_masked_own": np.zeros((0, 2), np.int32), "_masked_ctx": shared, "_ctx_hist": hist},
        {"_masked_own": np.array([[1, 2]], np.int32),
         "_masked_ctx": np.array([[10, 21]], np.int32), "_ctx_hist": hist},
    ]
    rec = gate_summary(loaded, max_zero_fraction=0.3, max_context_zero_fraction=0.0)
    assert rec["n_masked_context_nodata"] == 2, "the shared cell was counted twice"
    assert rec["n_masked_nodata"] == 1
    assert rec["max_context_zero_fraction"] == 0.0


def test_gate_summary_reports_unknown_rather_than_zero_for_pre_r13_partials():
    """Absence must never read as "checked and clean" — the failure mode this audit has now
    caught several times."""
    from scripts.map_region import gate_summary

    rec = gate_summary([{"ti": np.array([1])}],
                       max_zero_fraction=0.3, max_context_zero_fraction=0.0)
    assert rec["n_masked_context_nodata"] is None and rec["n_masked_nodata"] is None
    assert "predate the R13 gate record" in rec["gate_counts_note"]


def test_the_tile_sidecar_records_the_gate(tmp_path, monkeypatch):
    """T5 + M3 + M5 at the product boundary: drive the REAL `map_one_tile` over a synthetic
    tile with a mosaic gap and read the committed sidecar.

    The shipped `reports/map_region/E4_N44.json` carries no threshold and no mask counts, so
    a raster rendered with the gate off is indistinguishable from one rendered with it on.
    """
    import importlib.util
    import sys
    from pathlib import Path

    pytest.importorskip("rasterio")
    spec = importlib.util.spec_from_file_location(
        "_r01_grid_tests_r13", Path(__file__).with_name("test_map_region_global_grid.py"))
    g = importlib.util.module_from_spec(spec)
    sys.modules["_r01_grid_tests_r13"] = g
    spec.loader.exec_module(g)

    import scripts.map_region as mr
    from src.mapping import CtxWindow

    tile, transform, extent = g._synthetic_tile(tmp_path)
    a, b, c, d, e, f = transform
    # one small square gap, well inside the tile, so a handful of cells are gated
    gap = (500, 560, 500, 560)

    def fake_read(zip_path, inner_tif, row_off, col_off, size, **_kw):
        h, w = min(size, extent - row_off), min(size, extent - col_off)
        arr = np.full((h, w), 200, dtype=np.uint8)
        r0, r1 = max(gap[0] - row_off, 0), min(gap[1] - row_off, h)
        c0, c1 = max(gap[2] - col_off, 0), min(gap[3] - col_off, w)
        if r1 > r0 and c1 > c0:
            arr[r0:r1, c0:c1] = 0
        return CtxWindow(data=arr, row_off=row_off, col_off=col_off,
                         transform=(a, b, c + col_off * a, d, e, f + row_off * e), crs_wkt=g._WKT)

    monkeypatch.setattr(mr, "open_tile", lambda *a_, **k_: _NullSrc())
    monkeypatch.setattr(mr, "read_tile_window", fake_read)
    args = g._fake_args(tmp_path, out_dir=str(tmp_path), win_px=256,
                        max_zero_fraction=0.3, max_context_zero_fraction=0.0)
    assert mr.map_one_tile(tile, g._StubEmbedder(), g._StubHead(), None,
                           args=args)["status"] == "done"

    rec = json.loads((tmp_path / f"{tile}.json").read_text(encoding="utf-8"))
    gate = rec["nodata_gate"]
    assert gate["max_zero_fraction"] == 0.3
    assert gate["max_context_zero_fraction"] == 0.0
    assert gate["n_masked_nodata"] > 0, "the fixture's gap must trip the own-tile gate"
    assert gate["n_masked_context_nodata"] > 0, "and its halo must trip the context gate"
    assert gate["context_zero_hist_edges"] == [float(e) for e in CONTEXT_ZERO_HIST_EDGES]
    assert len(gate["context_zero_hist"]) == len(CONTEXT_ZERO_HIST_EDGES)
    # the histogram is monotone non-increasing in the threshold, and its first bin ("any
    # context nodata at all") must be at least the number of cells the gate actually dropped
    assert gate["context_zero_hist"] == sorted(gate["context_zero_hist"], reverse=True)
    assert gate["context_zero_hist"][0] >= gate["n_masked_context_nodata"]
    assert gate["context_zero_hist_is_window_sum"] is True

    # the counts must be a de-duplicated CELL SET, so they cannot exceed the cells that exist
    assert gate["n_masked_nodata"] + gate["n_masked_context_nodata"] < rec["n_unique_cells"]

    # ... and the raster really is holed where the gate fired
    import rasterio
    with rasterio.open(tmp_path / f"{tile}_prob.tif") as ds:
        assert int(np.isnan(ds.read(1)).sum()) >= gate["n_masked_context_nodata"]


def test_the_context_threshold_is_a_resume_match_field(tmp_path, monkeypatch):
    """R13 x R14 — the coupling that would otherwise bite silently.

    `max_zero_fraction` is already a resume-match field. If the context threshold did not join
    it, a post-R13 resume would reuse pre-R13 partials computed under no context gate at all
    and assemble the mixture without a word.
    """
    pytest.importorskip("rasterio")
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_r01_grid_tests_r13b", Path(__file__).with_name("test_map_region_global_grid.py"))
    g = importlib.util.module_from_spec(spec)
    sys.modules["_r01_grid_tests_r13b"] = g
    spec.loader.exec_module(g)

    import scripts.map_region as mr
    from src.mapping import CtxWindow

    tile, transform, extent = g._synthetic_tile(tmp_path)
    a, b, c, d, e, f = transform

    def fake_read(zip_path, inner_tif, row_off, col_off, size, **_kw):
        h, w = min(size, extent - row_off), min(size, extent - col_off)
        return CtxWindow(data=np.full((h, w), 200, dtype=np.uint8), row_off=row_off,
                         col_off=col_off,
                         transform=(a, b, c + col_off * a, d, e, f + row_off * e), crs_wkt=g._WKT)

    monkeypatch.setattr(mr, "open_tile", lambda *a_, **k_: _NullSrc())
    monkeypatch.setattr(mr, "read_tile_window", fake_read)
    args = g._fake_args(tmp_path, out_dir=str(tmp_path), win_px=256)
    mr.map_one_tile(tile, g._StubEmbedder(), g._StubHead(), None, args=args)

    rec = json.loads((tmp_path / f"{tile}.json").read_text(encoding="utf-8"))
    assert "max_context_zero_fraction" in rec["run"]
    loosened = dict(rec["run"], max_context_zero_fraction=0.25)
    why = mr.tile_is_reusable(tmp_path, tile, loosened)
    assert why and "max_context_zero_fraction" in why

    # and a partial directory whose _sweep.json predates the field is refused, not resumed
    sweep = tmp_path / "partials" / tile / "_sweep.json"
    stale = json.loads(sweep.read_text(encoding="utf-8"))
    stale.pop("max_context_zero_fraction")
    sweep.write_text(json.dumps(stale), encoding="utf-8")
    for p in tmp_path.glob(f"{tile}*.tif"):
        p.unlink()
    (tmp_path / f"{tile}.json").unlink()
    with pytest.raises(SystemExit, match="max_context_zero_fraction"):
        mr.map_one_tile(tile, g._StubEmbedder(), g._StubHead(), None,
                        args=g._fake_args(tmp_path, out_dir=str(tmp_path), win_px=256))


def test_both_arms_now_gate_the_context_identically_because_r38_landed():
    """R13 x R38 ordering — this test used to pin the A1 gate DISABLED, and the flip is the
    record that the ordering constraint was honoured rather than forgotten.

    Until 2026-08-10 A1 clipped to [0, 255], so a legal dark pixel was written as the nodata
    sentinel. Measured on the 38 training windows as a deploy proxy, the share of
    own-tile-passing cells carrying >=1 "nodata" context pixel went 0.00 % (raw mosaic) ->
    2.67 % (native A1 statistic) -> ~13 % (160 m statistic), so a zero-tolerance gate would have
    deleted a large slice of the map for a radiometric reason dressed as a data gap.

    R38 removed the collision at the root, so the two arms now agree. What keeps that honest is
    asserted next door in `tests/test_a1_clip_floor.py`: the floor moved off the sentinel AND
    the driver passes an explicit mask instead of inferring one from A1's output. Flooring alone
    would have made the damaged pixels invisible rather than safe.
    """
    import scripts.map_region as mr
    import scripts.striping_a1_map as a1
    from src.striping import A1_VALID_FLOOR

    for p in (a1.build_parser(), mr.build_parser()):
        assert p.get_default("max_context_zero_fraction") == 0.0
        assert p.get_default("max_zero_fraction") == 0.3

    assert A1_VALID_FLOOR > 0, (
        "the A1 arm may only run a zero-tolerance context gate while DN 0 in its output means "
        "nodata and nothing else; a floor of 0 puts back the collision this gate cannot see")


def test_parity_check_run_window_accepts_what_its_own_call_sites_pass():
    """Found while wiring R13's thresholds through: `run_window` declared seven parameters
    and both call sites passed eight positional arguments, so `scripts/parity_check.py`
    raised `TypeError` on every invocation — emit and check alike. The one gate meant to
    prove a GPU box reproduces the laptop had not been runnable, and nothing noticed because
    no test imports it.
    """
    from pathlib import Path

    import scripts.parity_check as pc

    inspect.signature(pc.run_window).bind(
        "E4_N44", 20000, 20000, 512, Path("model"), "calibration.npz", 96,
        "ctx_tiles", 0.3, 0.0)
