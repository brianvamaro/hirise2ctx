"""R14 — resume must not accept work it cannot vouch for.

The driver used to resume on `path.exists()`: no size, no read, no provenance, and keyed on
the FIRST of four artifacts written. Measured consequences, all reproduced on real geometry:

  * a kill between artifacts 1 and 4 left a tile permanently "done" with **no abundance
    raster**, and abundance is the deliverable;
  * the assembly gate was `len(present) < len(grid)`, a COUNT over a glob — on the reachable
    stale state (a completed `--win-px 2048` sweep, then `--win-px 4096 --force`) that is 719
    files against 144 expected, the gate passes, and the emitted raster is the right shape with
    63.1 % of its finite pixels from the stale run;
  * a partial truncated by a wall-clock kill was skipped, then blew up at assembly, so every
    re-run crashed at the same point forever.

These tests drive the REAL `map_one_tile` / `process_tile` with inference stubbed.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

# the sibling module owns the synthetic-tile fixture; import it by path so this works
# regardless of pytest's import mode
_spec = importlib.util.spec_from_file_location(
    "_r01_grid_tests", Path(__file__).with_name("test_map_region_global_grid.py"))
_g = importlib.util.module_from_spec(_spec)
sys.modules["_r01_grid_tests"] = _g
_spec.loader.exec_module(_g)
_WKT, _StubEmbedder, _StubHead = _g._WKT, _g._StubEmbedder, _g._StubHead
_fake_args, _synthetic_tile = _g._fake_args, _g._synthetic_tile

TILE_PX = 32


class _NullSrc:
    """Stand-in for the per-tile rasterio handle the drivers now hold open.

    The sweep drivers `open_tile(...)` once and pass the handle to `read_tile_window`
    (DECISIONS 2026-08-18c). Every test here fakes `read_tile_window`, so the handle is
    never read from -- it only has to exist and close.
    """

    def close(self):
        pass

def _uncommit(tmp_path, tile):
    """Remove the tile-level commit so the next call reaches the PARTIAL gates."""
    for p in tmp_path.glob(f"{tile}*.tif"):
        p.unlink()
    (tmp_path / f"{tile}.json").unlink(missing_ok=True)


def _drive(monkeypatch, tmp_path, out_dir=None, **kw):
    """Render one synthetic tile through the real driver; returns (module, tile, args)."""
    import scripts.map_region as mr
    from src.mapping import CtxWindow

    tile, transform, extent = _synthetic_tile(tmp_path)
    a, b, c, d, e, f = transform

    def fake_read(zip_path, inner_tif, row_off, col_off, size, **_kw):
        h, w = min(size, extent - row_off), min(size, extent - col_off)
        return CtxWindow(data=np.full((h, w), 200, dtype=np.uint8),
                         row_off=row_off, col_off=col_off,
                         transform=(a, b, c + col_off * a, d, e, f + row_off * e),
                         crs_wkt=_WKT)

    monkeypatch.setattr(mr, "open_tile", lambda *a_, **k_: _NullSrc())
    monkeypatch.setattr(mr, "read_tile_window", fake_read)
    args = _fake_args(tmp_path, out_dir=str(out_dir or tmp_path), win_px=256, **kw)
    return mr, tile, args


# ------------------------------------------------------------------ the assembly gate

def test_extra_partials_from_another_sweep_are_refused(tmp_path, monkeypatch):
    """A COUNT gate is satisfied by a superset; set equality is not."""
    mr, tile, args = _drive(monkeypatch, tmp_path)
    assert mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                           args=args)["status"] == "done"

    # plant orphans with names no sweep of this win_px produces, as a 2048 run would leave
    pdir = tmp_path / "partials" / tile
    pdir.mkdir(parents=True, exist_ok=True)
    good = next(pdir.glob("*.npz")) if list(pdir.glob("*.npz")) else None
    if good is None:                       # --clean-partials off, so partials survive
        pytest.skip("no partials retained")
    for off in (1, 2, 3):
        (pdir / f"{off:06d}_{off:06d}.npz").write_bytes(good.read_bytes())

    args2 = _fake_args(tmp_path, out_dir=str(tmp_path), win_px=256, force=True)
    with pytest.raises(SystemExit, match="not part of this sweep"):
        mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args2)


def test_partials_from_a_different_win_px_are_refused_before_any_gpu_time(tmp_path, monkeypatch):
    """The sweep manifest names the field that differs, and it fires at scan time."""
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)
    _uncommit(tmp_path, tile)                    # reach the PARTIAL gate, not the tile gate

    args2 = _fake_args(tmp_path, out_dir=str(tmp_path), win_px=512)
    with pytest.raises(SystemExit, match="different sweep") as exc:
        mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args2)
    assert "win_px" in str(exc.value)


def test_partials_from_a_different_head_are_refused(tmp_path, monkeypatch):
    """Identical filenames, identical geometry, different head — structurally undetectable."""
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)
    _uncommit(tmp_path, tile)
    sweep = tmp_path / "partials" / tile / "_sweep.json"
    rec = json.loads(sweep.read_text(encoding="utf-8"))
    rec["head_digest"] = "0" * 64
    sweep.write_text(json.dumps(rec), encoding="utf-8")

    with pytest.raises(SystemExit, match="head_digest"):
        mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                        args=_fake_args(tmp_path, out_dir=str(tmp_path), win_px=256))


def test_this_small_sweep_assigns_every_cell_to_exactly_one_window(tmp_path, monkeypatch):
    """Windows overlap in PIXELS (so every cell has context); on THIS sweep they also
    partition the CELLS, so the overlap costs no duplicated inference.

    ⚠ **Do not generalise this to the shipped sweep.** It used to be pinned here as the reason
    the disagreement check "cannot false-positive", and that inference was wrong: whether cells
    partition depends on the window geometry, and the 144-window / 4096 px sweep the drivers
    actually ship duplicates 62,559-80,570 cells per Murray tile. See
    `test_a_larger_sweep_really_does_duplicate_cells` and `overlap_disagreement`.
    """
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)
    parts = sorted((tmp_path / "partials" / tile).glob("*.npz"))
    ti = np.concatenate([mr.read_partial(p)["ti"] for p in parts]).astype(np.int64)
    tj = np.concatenate([mr.read_partial(p)["tj"] for p in parts]).astype(np.int64)
    assert np.unique(ti * (2 ** 21) + tj).size == ti.size, "a cell was computed twice"
    c = mr.overlap_disagreement(ti, tj, np.full(ti.size, 0.5))
    assert (c.n_dup, c.n_disagree, c.fraction, c.max_abs) == (0, 0, 0.0, 0.0)
    assert not c.refuse


def test_a_larger_sweep_really_does_duplicate_cells():
    """The measurement the old guard's premise was missing.

    `window_offsets` on a Murray-scale extent yields windows whose cell sets overlap, so the
    "cells are a partition, therefore within one run disagreement is 0 by construction" claim
    only ever held for small sweeps. This walks the real geometry and asserts the duplicates
    exist, so the premise cannot quietly come back.
    """
    import scripts.map_region as mr
    extent, win, overlap = 47420, 4096, 3 * TILE_PX
    offs = mr.window_offsets(extent, win, overlap, TILE_PX, tile_aligned=False)
    assert len(offs) ** 2 == 144, f"geometry changed: {len(offs)}^2 windows"
    cells = set()
    dup = 0
    for r0 in offs:
        for c0 in offs:
            # the cells a window contributes: its interior ring, on the global lattice
            rr = range(r0 + TILE_PX, r0 + win - TILE_PX, TILE_PX)
            cc = range(c0 + TILE_PX, c0 + win - TILE_PX, TILE_PX)
            for r in rr:
                for c in cc:
                    if (r, c) in cells:
                        dup += 1
                    cells.add((r, c))
    assert dup > 10_000, f"expected tens of thousands of duplicated cells, got {dup}"


def test_partials_from_two_runs_disagree_and_are_refused(tmp_path, monkeypatch):
    """The value-level failure no structural check can see: every file is perfect, the shape
    is right, and 63 % of the pixels came from another run.

    Both runs compute the same cell set on the same lattice, so a surviving stale partial
    collides cell-for-cell with the current run's.
    """
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)

    pdir = tmp_path / "partials" / tile
    parts = sorted(pdir.glob("*.npz"))
    z = mr.read_partial(parts[0])
    z["prob"] = np.full(z["prob"].shape, 0.9, dtype=np.float32)   # a "stale head" wrote these
    stale = pdir / "_stale_from_an_earlier_run.keep"
    with open(stale, "wb") as fh:
        np.savez_compressed(fh, **z)

    grid_geom = mr.tile_global_grid(
        tuple(json.loads((tmp_path / "ctx" / f"{tile}.json").read_text())["inner_transform"]),
        _WKT, TILE_PX)
    with pytest.raises(SystemExit, match="duplicated cells disagree"):
        mr.write_tile_geotiffs(tile, parts + [stale], grid_geom, _WKT, None, args)


# ------------------------------------------------------- the fraction gate (R14, 2026-08-24d)

def _check(vals_a, vals_b):
    """One duplicated cell block: the same `n` cells written twice, with `a` then `b`."""
    import scripts.map_region as mr
    n = len(vals_a)
    ti = np.concatenate([np.arange(n), np.arange(n)])
    tj = np.zeros(2 * n, dtype=np.int64)
    return mr.overlap_disagreement(ti, tj, np.concatenate([vals_a, vals_b]))


def test_fp16_noise_on_a_fraction_of_a_percent_is_not_refused():
    """The case that stalled E0_N36 and E0_N44: 242 of 79,059 duplicated cells differ."""
    n = 79_059
    a = np.full(n, 0.5)
    b = a.copy()
    b[:242] += 5.3e-4                       # measured magnitude, measured count
    c = _check(a, b)
    assert (c.n_dup, c.n_disagree) == (n, 242)
    assert c.fraction == pytest.approx(242 / n)
    assert not c.refuse, "float noise must not block a correctly rendered tile"


def test_a_stale_partial_mixture_is_refused():
    """The case R14 exists for: 63.1 % of cells came from the wrong run."""
    n = 79_059
    k = int(0.631 * n)
    a = np.full(n, 0.5)
    b = a.copy()
    b[:k] = 0.9                             # a different head wrote these
    c = _check(a, b)
    assert c.n_disagree == k
    assert c.fraction == pytest.approx(0.631, abs=1e-3)
    assert c.refuse


def test_a_tiny_disagreeing_count_is_not_refused_even_at_a_high_fraction():
    """The absolute floor. On a sweep with a handful of duplicated cells the fraction is a
    very noisy estimator of "two runs", so both conditions must hold."""
    c = _check(np.full(4, 0.5), np.array([0.9, 0.9, 0.9, 0.9]))
    assert (c.n_dup, c.n_disagree, c.fraction) == (4, 4, 1.0)
    assert not c.refuse                      # 4 <= OVERLAP_DISAGREE_MIN_CELLS
    big = _check(np.full(40, 0.5), np.full(40, 0.9))
    assert big.refuse                        # 40 > 16, fraction 1.0


def test_a_cell_written_three_times_counts_once():
    import scripts.map_region as mr
    ti = np.array([7, 7, 7, 8])
    tj = np.zeros(4, dtype=np.int64)
    c = mr.overlap_disagreement(ti, tj, np.array([0.5, 0.5, 0.6, 0.1]))
    assert (c.n_dup, c.n_disagree) == (1, 1)   # cell 7 only, not 2 pairs
    assert c.max_abs == pytest.approx(0.1)


def test_a_one_sided_nan_is_a_disagreement():
    """One run masked the cell and the other did not -- that is a mixture, not float noise."""
    n = 100
    a = np.full(n, 0.5)
    b = a.copy()
    b[:50] = np.nan
    c = _check(a, b)
    assert c.n_disagree == 50 and not np.isfinite(c.max_abs)
    assert c.refuse
    both = _check(np.full(n, np.nan), np.full(n, np.nan))
    assert both.n_disagree == 0 and not both.refuse   # both nan = agreement


def test_check_overlap_gates_on_prob_raw_and_records_every_layer():
    """Isotonic collapses most prob_raw noise and amplifies the survivors, so `prob` is not
    the layer to judge on -- but it, and abundance, must still be recorded."""
    import scripts.map_region as mr
    n = 5_000
    ti = np.concatenate([np.arange(n), np.arange(n)])
    tj = np.zeros(2 * n, dtype=np.int64)

    def dup(vals_a, vals_b):
        return np.concatenate([vals_a, vals_b])

    raw_a = np.full(n, 0.5)
    raw_b = raw_a.copy()
    raw_b[:20] += 5e-4                       # 0.4 % of cells: noise
    prob_a = np.full(n, 0.3)
    prob_b = prob_a.copy()
    prob_b[0] += 6e-3                        # one amplified survivor
    rec = mr.check_overlap("T", ti, tj, {"prob_raw": dup(raw_a, raw_b),
                                         "prob": dup(prob_a, prob_b),
                                         "abundance": None})
    assert rec["gate_layer"] == "prob_raw"
    assert rec["prob_raw"]["n_disagree"] == 20
    assert rec["prob"]["n_disagree"] == 1
    assert "abundance" not in rec            # None layers are skipped, not recorded as 0

    # with no calibrator there is no prob_raw, so the gate falls back to prob
    rec2 = mr.check_overlap("T", ti, tj, {"prob_raw": None, "prob": dup(prob_a, prob_b)})
    assert rec2["gate_layer"] == "prob"


# ------------------------------------------------------------------ partial integrity

def test_a_corrupt_partial_is_recomputed_not_skipped_forever(tmp_path, monkeypatch):
    """A truncated `.npz` used to be skipped by the exists() check and then crash assembly, so
    every re-run failed at the same point forever. It must be recomputed instead."""
    mr, tile, args = _drive(monkeypatch, tmp_path)
    assert mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                           args=args)["status"] == "done"
    pdir = tmp_path / "partials" / tile
    victim = sorted(pdir.glob("*.npz"))[0]
    victim.write_bytes(victim.read_bytes()[: 40])            # truncate
    assert mr.partial_grid_id(victim) is None

    for p in tmp_path.glob(f"{tile}*.tif"):                  # force a re-assembly
        p.unlink()
    (tmp_path / f"{tile}.json").unlink()
    assert mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                           args=args)["status"] == "done"
    mr.read_partial(victim)                                  # regenerated and readable


def test_a_partial_never_appears_under_its_final_name_until_it_round_trips(tmp_path, monkeypatch):
    """`np.savez_compressed` writes the zip in place, so the final name must not be the
    write target."""
    mr, tile, args = _drive(monkeypatch, tmp_path)
    real = mr.read_partial
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("simulated bad CRC on the staged file")
        return real(path)

    monkeypatch.setattr(mr, "read_partial", flaky)
    with pytest.raises(ValueError, match="simulated bad CRC"):
        mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)
    pdir = tmp_path / "partials" / tile
    assert list(pdir.glob("*.npz")) == [], "a partial was committed despite failing its check"
    assert list(pdir.glob("*.tmp")) == [], "a .tmp was left behind"


# ------------------------------------------------------------------ the resume predicate

def test_a_pre_r14_sidecar_is_not_reusable_without_an_explicit_opt_in(tmp_path):
    """The 26 shipped tiles have sidecars with no `rasters` block, so their contents cannot be
    verified. "Unverifiable" must not read as "fine"."""
    import scripts.map_region as mr

    (tmp_path / "E4_N40.json").write_text(json.dumps(
        {"murray_tile": "E4_N40", "tile_px": 32, "raster_shape": [1479, 1479]}),
        encoding="utf-8")
    why = mr.tile_is_reusable(tmp_path, "E4_N40", None)
    assert why and "predates R14" in why
    assert mr.tile_is_reusable(tmp_path, "E4_N40", None, trust_existing=True) is None


def test_reuse_requires_the_run_to_match(tmp_path, monkeypatch):
    """Content alone cannot see a raster that is structurally perfect and made the wrong way."""
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)
    rec = json.loads((tmp_path / f"{tile}.json").read_text(encoding="utf-8"))
    assert mr.tile_is_reusable(tmp_path, tile, rec["run"]) is None

    wrong = dict(rec["run"], max_zero_fraction=0.123)
    why = mr.tile_is_reusable(tmp_path, tile, wrong)
    assert why and "max_zero_fraction" in why


def test_a_raw_run_does_not_mark_a_calibrated_tile_done(tmp_path, monkeypatch):
    """`--raw` writes ONLY `{tile}_prob.tif` — exactly the old sentinel — so a raw run made a
    later calibrated run report skipped_done and the region shipped a raw tile inside a map
    whose manifest said calibrated: true."""
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)   # calibrator=None
    rec = json.loads((tmp_path / f"{tile}.json").read_text(encoding="utf-8"))
    assert rec["run"]["calibrated"] is False
    assert [r["kind"] for r in rec["rasters"]] == ["prob"]

    calibrated_run = dict(rec["run"], calibrated=True)
    why = mr.tile_is_reusable(tmp_path, tile, calibrated_run)
    assert why and "calibrated" in why


# ------------------------------------------------------------------ the manifest

def test_the_region_manifest_merges_across_runs_instead_of_clobbering(tmp_path):
    """The shipped manifest lists 4 tiles while 26 tiles' rasters are on disk, because every
    run overwrote it — so 22 of 26 have no record and `win_px` is unknown for them."""
    import scripts.map_region as mr

    m = tmp_path / "region_manifest.json"
    mr.write_json_atomic(m, {"tiles": ["A", "B"], "runs": [{"win_px": 2048}],
                             "results": [{"tile": "A", "status": "done"},
                                         {"tile": "B", "status": "done"}]})
    prev = json.loads(m.read_text(encoding="utf-8"))
    by_tile = {r["tile"]: r for r in prev["results"]}
    by_tile.update({"C": {"tile": "C", "status": "done"}})
    mr.write_json_atomic(m, {"tiles": sorted(by_tile), "runs": prev["runs"] + [{"win_px": 4096}],
                             "results": [by_tile[t] for t in sorted(by_tile)]})

    out = json.loads(m.read_text(encoding="utf-8"))
    assert out["tiles"] == ["A", "B", "C"]
    assert [r["win_px"] for r in out["runs"]] == [2048, 4096]


# ----------------------------------------------- per-tile isolation of the array stride

def test_a_failing_tile_does_not_forfeit_the_rest_of_the_stride():
    """Step 11 lost 3 renderable tiles to a `SystemExit` on a sibling. It must not recur."""
    import scripts.map_region as mr

    seen = []

    def fn(tile):
        seen.append(tile)
        if tile == "BAD":
            raise SystemExit("cells were written twice")
        return {"tile": tile, "status": "done"}

    rows = [mr.run_tile_isolated(t, lambda t=t: fn(t)) for t in ("A", "BAD", "B")]
    assert seen == ["A", "BAD", "B"], "the stride stopped at the failing tile"
    assert [r["status"] for r in rows] == ["done", "failed", "done"]
    assert "cells were written twice" in rows[1]["error"]


def test_isolation_does_not_swallow_a_cancel():
    """`scancel` / Ctrl-C means stop now, not 'record and carry on'."""
    import scripts.map_region as mr

    def boom():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        mr.run_tile_isolated("T", boom)


# ------------------------------------- device provenance (2026-08-24e, soft resume field)

def test_a_recorded_device_change_is_a_mismatch():
    """Partials from an RTX 2080 Ti and a TITAN Xp are two fp16 regimes, not one run."""
    import scripts.map_region as mr

    want = {"win_px": 4096, "device": "NVIDIA GeForce RTX 2080 Ti"}
    assert mr.sweep_mismatch(dict(want), want) is None
    why = mr.sweep_mismatch({"win_px": 4096, "device": "NVIDIA TITAN Xp"}, want)
    assert why is not None and "device" in why


def test_an_absent_device_is_unknown_not_a_mismatch():
    """The field was added mid-rebuild, after 7 A1 tiles and 255 windows were banked without
    it. Treating those as mismatched would discard real work by fiat."""
    import scripts.map_region as mr

    want = {"win_px": 4096, "device": "NVIDIA GeForce RTX 2080 Ti"}
    assert mr.sweep_mismatch({"win_px": 4096}, want) is None          # key absent
    assert mr.sweep_mismatch({"win_px": 4096, "device": None}, want) is None   # recorded null
    # but a HARD field is still refused when absent
    assert "win_px" in mr.sweep_mismatch({"device": want["device"]}, want)
    assert mr.SWEEP_SOFT_FIELDS == frozenset({"device"})


def test_the_sweep_manifest_carries_the_device(tmp_path, monkeypatch):
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)
    rec = json.loads((tmp_path / f"{tile}.json").read_text(encoding="utf-8"))
    assert "device" in rec["run"]


def test_compute_device_name_never_raises_on_a_stub():
    """Provenance must not be able to kill a render."""
    import scripts.map_region as mr

    class NoDevice:
        pass

    assert isinstance(mr.compute_device_name(NoDevice()), str)
    assert isinstance(mr.compute_device_name(None), str)


# ------------------------------------------------------------- the cost projection line

def test_project_tile_cost_reports_the_hours_the_allocation_implies():
    """The line that would have flagged the A1 timeout at minute four."""
    import scripts.map_region as mr

    # TITAN Xp: 202 s/window over 144 windows = 8.1 h, against a 10 h wall for 5 tiles
    msg = mr.project_tile_cost("E-12_N36", 202.0, 144)
    assert "8.08 h" in msg and "144 remaining" in msg
    # RTX 2080 Ti baseline: 17.9 s/window = 0.72 h
    assert "0.72 h" in mr.project_tile_cost("E-12_N36", 17.9, 144)
    # a resumed tile projects only what is LEFT
    assert "48 remaining" in mr.project_tile_cost("E0_N32", 202.0, 144, 96)


def test_require_device_refuses_an_unbudgeted_gpu():
    """A TITAN Xp costs 8.08 h/tile against a wall sized for 0.72 h. Refuse in seconds."""
    import scripts.map_region as mr

    class Stub:
        pass

    monkey = mr.compute_device_name
    try:
        mr.compute_device_name = lambda _e: "NVIDIA TITAN Xp"
        with pytest.raises(SystemExit, match="REFUSING to render"):
            mr.require_device(Stub(), ["2080 Ti"])
        mr.compute_device_name = lambda _e: "NVIDIA GeForce RTX 2080 Ti"
        assert mr.require_device(Stub(), ["2080 ti"]) == "NVIDIA GeForce RTX 2080 Ti"
        assert mr.require_device(Stub(), ["P100", "2080 Ti"]) is not None
        # unconstrained must keep working -- CPU smoke tests and laptop runs depend on it
        mr.compute_device_name = lambda _e: "cpu"
        assert mr.require_device(Stub(), None) == "cpu"
        assert mr.require_device(Stub(), []) == "cpu"
    finally:
        mr.compute_device_name = monkey
