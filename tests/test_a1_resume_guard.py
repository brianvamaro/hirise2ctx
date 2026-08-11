"""R14 on the A1 arm — the resume guard `scripts/striping_a1_map.py` never had.

Found 2026-08-10 while landing R13: the driver deleted a `partials/<tile>/_sweep.json` in its
`--clean-partials` path but **never wrote one** and never called `sweep_manifest`. So R14's
sweep-identity protection covered `scripts/map_region.py` only, and on the A1 arm the *only* thing
between a resumed run and a two-run raster was the `grid_id` check.

`grid_id` is a **lattice** check. Two runs with different heads, window sizes, masking thresholds or
A1 statistics all share it — and R01 made both drivers use the same lattice deliberately, so it is
guaranteed to match. Their partials also have colliding filenames (`{row:06d}_{col:06d}.npz`). Every
structural check downstream then passes: each file is a perfect `.npz`, the set-equality gate is
satisfied, and the assembled raster is the right shape. R14 measured that exact state on the
baseline arm: 63.1 % of finite pixels came from the stale run.

A1 needs *more* identity than the baseline, not less, because its input is **derived**: two runs can
agree on window geometry and head and still have normalised the DN differently. Hence `norm_arm`
(R07's statistic definition), `a1_ref` + `a1_clip_floor` (R38 moved the floor, which changes pixels),
`a1_min_frame_px` (R08's ratified fallback boundary) and `a1_seammap_digest` (the frame partition).

Mutants these kill:
  M1  `process_tile` never writes `_sweep.json`  (the defect as found)
  M2  the A1 manifest drops `norm_arm` / `a1_ref` / `a1_clip_floor` / `a1_min_frame_px`
  M3  the A1 manifest drops `a1_seammap_digest`
  M4  `seammap_digest` hashes only the `.shp`, ignoring the load-bearing `.prj`
  M5  the tile-level reuse check is still passed `None` instead of the sweep manifest
"""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rasterio")

import scripts.striping_a1_map as a1
from scripts.map_region import sweep_mismatch
from src.striping import A1_ARM, A1_MIN_FRAME_PX, A1_REF_IQR, A1_REF_MEDIAN, A1_VALID_FLOOR


def _grid_geom():
    return SimpleNamespace(cell_row0=-10000, cell_col0=-20000, phase_r=20, phase_c=4)


def _args(**kw):
    a = SimpleNamespace(win_px=4096, max_zero_fraction=0.3, max_context_zero_fraction=0.0,
                        no_isotonic=True, head="models/deployable_a1/x", calibration=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _manifest(tile="E4_N44", **kw):
    return a1.a1_sweep_manifest(_grid_geom(), [0, 4000], [0, 4000], _args(**kw),
                                extent=(47420, 47420), tile=tile,
                                head_digest="deadbeef", calibration_digest=None)


# ---------------------------------------------------------------- the identity

def test_the_a1_manifest_carries_what_makes_a1_a1():
    """M2/M3. The shared fields are necessary but not sufficient on this arm."""
    m = _manifest()
    # inherited from map_region -- geometry, head, thresholds
    for k in ("grid_id", "tile_px", "win_px", "overlap", "extent", "n_windows",
              "max_zero_fraction", "max_context_zero_fraction", "head_digest"):
        assert k in m, f"{k} must stay a resume-match field on the A1 arm too"
    # A1-specific: the input was DERIVED, so geometry+head agreement is not enough
    assert m["variant"] == "A1"
    assert m["norm_arm"] == A1_ARM
    assert m["a1_ref"] == [A1_REF_MEDIAN, A1_REF_IQR]
    assert m["a1_clip_floor"] == A1_VALID_FLOOR
    assert m["a1_min_frame_px"] == A1_MIN_FRAME_PX
    assert "a1_seammap_digest" in m


@pytest.mark.parametrize("field,bad", [
    ("norm_arm", "a1_some_other_arm"),
    ("a1_ref", [125.0, 40.0]),
    ("a1_clip_floor", 0),
    ("a1_min_frame_px", 500),
    ("a1_seammap_digest", "0" * 64),
    ("max_context_zero_fraction", 1.0),
    ("win_px", 2048),
    ("head_digest", "0" * 64),
])
def test_each_field_alone_refuses_the_resume(field, bad):
    """A mismatch on ANY of these must be caught and must name itself.

    Parametrised rather than lumped: a guard that only fires when several fields differ at once
    is the one that lets the realistic single-knob re-run through.
    """
    want = _manifest()
    have = dict(want)
    have[field] = bad
    why = sweep_mismatch(have, want)
    assert why and field in why, f"a changed {field} was not caught ({why!r})"


def test_a_pre_guard_partial_directory_counts_as_a_mismatch():
    """Absence must not read as agreement. Every A1 partial ever written predates this block,
    so an empty/missing `_sweep.json` has to fail rather than pass by default."""
    want = _manifest()
    why = sweep_mismatch({}, want)
    assert why and "absent" in why


def test_an_identical_sweep_still_resumes():
    """The guard must not make resume impossible -- that is the whole feature it protects."""
    assert sweep_mismatch(_manifest(), _manifest()) is None


# ---------------------------------------------------------------- the seammap digest

def test_seammap_digest_covers_the_prj_not_just_the_shp(tmp_path, monkeypatch):
    """M4, and it is the CLAUDE.md gotcha in miniature.

    The frames are reprojected into the tile CRS before rasterization, so a changed `.prj`
    silently changes which pixels belong to which frame -- and therefore every per-frame
    statistic -- without touching one coordinate in the `.shp`.
    """
    shp = tmp_path / "seam.shp"
    shp.write_bytes(b"geometry bytes")
    prj = tmp_path / "seam.prj"
    prj.write_text("PROJCS[radius 3396190]", encoding="utf-8")

    monkeypatch.setattr(a1, "find_seam_shp", lambda tile: shp)
    before = a1.seammap_digest("T")
    prj.write_text("PROJCS[radius 3393833]", encoding="utf-8")   # the local-radius gotcha
    after = a1.seammap_digest("T")

    assert before and after
    assert before != after, "a changed .prj must change the digest; the CRS defines the frames"


def test_seammap_digest_is_none_when_there_is_no_seammap(monkeypatch):
    """None must propagate as 'unknown', not crash and not silently look like a match."""
    monkeypatch.setattr(a1, "find_seam_shp", lambda tile: None)
    assert a1.seammap_digest("T") is None


# ---------------------------------------------------------------- the wiring

def _process_tile_ast():
    return ast.parse(inspect.getsource(a1.process_tile))


def test_process_tile_actually_writes_the_sweep_manifest():
    """M1 — the defect as found: it deleted a `_sweep.json` it never wrote.

    Scans the AST rather than the text, so a docstring describing the guard cannot satisfy it.
    """
    tree = _process_tile_ast()
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "a1_sweep_manifest" in called, "the A1 sweep identity is never built"
    assert "write_json_atomic" in called, "the sweep manifest is never persisted"
    assert "sweep_mismatch" in called, "an existing partial dir is never checked against it"


def test_the_tile_reuse_check_is_given_the_sweep_not_none():
    """M5. `tile_is_reusable(out_dir, tile, None)` skips the run comparison entirely, so a tile
    normalised by a different A1 arm reads as reusable on content alone."""
    tree = _process_tile_ast()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "tile_is_reusable"):
            third = node.args[2]
            assert not (isinstance(third, ast.Constant) and third.value is None), (
                "tile_is_reusable is still passed None as `want_run`")
            return
    pytest.fail("tile_is_reusable is not called in process_tile")


def test_the_sweep_is_built_before_the_expensive_statistics_pass():
    """Ordering, and it is worth pinning: `a1_sweep_manifest` digests A1's *inputs* rather than
    the derived per-frame statistics precisely so an already-committed tile can be skipped
    without paying for the ~3 min streaming read it does not need."""
    src = inspect.getsource(a1.process_tile)
    assert src.index("a1_sweep_manifest(") < src.index("frame_stats_native("), (
        "the sweep identity must be cheap enough to build before the streaming pass")
    assert src.index("tile_is_reusable(") < src.index("frame_stats_native("), (
        "a committed tile must be skipped before the streaming pass, not after")
