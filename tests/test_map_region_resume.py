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


def test_the_sweep_assigns_every_cell_to_exactly_one_window(tmp_path, monkeypatch):
    """Windows overlap in PIXELS (so every cell has context) but partition the CELLS.

    Worth pinning: it is why the cross-run disagreement check cannot false-positive, and it
    means the overlap costs no duplicated inference.
    """
    mr, tile, args = _drive(monkeypatch, tmp_path)
    mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None, args=args)
    parts = sorted((tmp_path / "partials" / tile).glob("*.npz"))
    ti = np.concatenate([mr.read_partial(p)["ti"] for p in parts]).astype(np.int64)
    tj = np.concatenate([mr.read_partial(p)["tj"] for p in parts]).astype(np.int64)
    assert np.unique(ti * (2 ** 21) + tj).size == ti.size, "a cell was computed twice"
    assert mr.overlap_disagreement(ti, tj, np.full(ti.size, 0.5)) == (0, 0.0)


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
    with pytest.raises(SystemExit, match="DIFFERENT values"):
        mr.write_tile_geotiffs(tile, parts + [stale], grid_geom, _WKT, None, args)


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
