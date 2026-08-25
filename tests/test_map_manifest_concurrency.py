"""The manifest index must survive concurrency and mid-stride death. 2026-08-25.

Step 11 shipped 26/26 tiles on both arms with the INDEX damaged: `region_manifest.json` listed
21 of 26, `a1_manifest.json` 1 of 26. No raster was affected — the sidecar is the authority and
all 52 were intact — but run-level provenance recorded nowhere else (`win_px`) was lost.

Three causes, covered in the three sections below:
  * a fixed `<path>.tmp` staging name, so two concurrent writers collided and the loser CRASHED;
  * a read-modify-write merge, so an overtaken or killed task vanished from the index;
  * the A1 driver not merging at all — a bare `write_text` of only its own results.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import scripts.map_region as mr  # noqa: E402

SRC = Path(__file__).parents[1] / "scripts"


# ---------------------------------------------------------------- atomic write, concurrently

def test_the_staging_name_is_per_process():
    """A shared `<path>.tmp` is what let one writer delete another's staging file."""
    src = (Path(__file__).parents[1] / "src" / "map_manifest.py").read_text(encoding="utf-8")
    i = src.index("def write_json_atomic")
    body = src[i:src.index("\ndef ", i + 10)]
    assert "os.getpid()" in body, "the staging name is not per-process any more"
    assert 'path.name + ".tmp"' not in body, "the fixed staging name is back"


def test_map_region_still_re_exports_the_moved_helpers():
    """They moved to a stdlib-only module; every existing caller must keep working."""
    for name in ("write_json_atomic", "merge_manifest", "tile_result_rows", "tile_sidecars"):
        assert callable(getattr(mr, name)), f"map_region no longer exposes {name}"


def test_two_writers_do_not_destroy_each_others_staging_file(tmp_path):
    """Reproduces the E0_N36 crash on the old scheme, then shows the fix.

    Old behaviour: both writers stage to the same name; whoever renames first leaves the other
    renaming a file that no longer exists -> FileNotFoundError, *after* its tile was committed.
    """
    target = tmp_path / "region_manifest.json"
    shared = target.with_name(target.name + ".tmp")

    shared.write_text('{"writer": "A"}', encoding="utf-8")   # writer A stages
    shared.replace(target)                                   # writer B renames it away
    with pytest.raises(FileNotFoundError):                   # A now renames nothing
        shared.replace(target)

    # the fixed version: each writer owns its staging file, so every write lands
    target.unlink()
    stale = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    stale.write_text("{}", encoding="utf-8")                 # a stale sibling must not matter
    mr.write_json_atomic(target, {"writer": "A"})
    mr.write_json_atomic(target, {"writer": "B"})
    assert json.loads(target.read_text())["writer"] == "B"   # last writer wins, no crash
    assert not list(tmp_path.glob("*.tmp")), "staging files leaked"


def test_a_failed_write_leaves_no_staging_file(tmp_path):
    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        mr.write_json_atomic(tmp_path / "m.json", {"bad": Unserialisable()})
    assert not list(tmp_path.glob("*")), "a failed write left debris"


# ---------------------------------------------------------------- sidecar-derived index

def _sidecar(d: Path, tile: str, *, grid="G1", n_windows=144, cells=2187441):
    (d / f"{tile}.json").write_text(json.dumps({
        "murray_tile": tile, "grid_id": grid, "raster_shape": [1479, 1479],
        "n_unique_cells": cells, "elapsed_s": 2723.0,
        "run": {"n_windows": n_windows, "win_px": 4096},
        "rasters": [{"name": f"{tile}_prob.tif", "kind": "prob", "bytes": 1, "sha256": "x"}],
    }), encoding="utf-8")


def test_tile_result_rows_discovers_sidecars_and_ignores_manifests(tmp_path):
    for t in ("E0_N36", "E4_N44"):
        _sidecar(tmp_path, t)
    (tmp_path / "region_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a1_manifest.json").write_text("{}", encoding="utf-8")
    rows = mr.tile_result_rows(tmp_path)
    assert [r["tile"] for r in rows] == ["E0_N36", "E4_N44"]
    assert rows[0]["windows"] == 144 and rows[0]["status"] == "done"


def test_tiles_is_a_filter_not_the_source_of_truth(tmp_path):
    """None must index whatever is on disk, so a non-BLOCK_TILES footprint still gets a
    complete index."""
    for t in ("WEIRD_TILE", "E0_N36"):
        _sidecar(tmp_path, t)
    assert len(mr.tile_result_rows(tmp_path)) == 2
    assert [r["tile"] for r in mr.tile_result_rows(tmp_path, ["E0_N36"])] == ["E0_N36"]


def test_the_index_self_heals_a_manifest_that_lost_entries(tmp_path):
    """The measured 21-of-26 case: the manifest under-reports, one later write repairs it."""
    tiles = ["E0_N32", "E0_N36", "E0_N40", "E0_N44"]
    for t in tiles:
        _sidecar(tmp_path, t)
    m = tmp_path / "region_manifest.json"
    mr.write_json_atomic(m, {"grid_id": "G1", "tiles": ["E0_N32"], "runs": [{"win_px": 2048}],
                             "results": [{"tile": "E0_N32", "status": "done"}]})
    doc = mr.merge_manifest(m, out_dir=tmp_path, grid_id="G1", run_record={"win_px": 4096})
    assert doc["tiles"] == tiles, "the index did not heal from the sidecars"
    assert [r["win_px"] for r in doc["runs"]] == [2048, 4096], "run history not preserved"


def test_a_corrupt_previous_manifest_does_not_block_the_rebuild(tmp_path):
    _sidecar(tmp_path, "E0_N36")
    m = tmp_path / "region_manifest.json"
    m.write_text("{ this is not json", encoding="utf-8")
    doc = mr.merge_manifest(m, out_dir=tmp_path, grid_id="G1", run_record=None)
    assert doc["tiles"] == ["E0_N36"] and doc["runs"] == []


def test_a_failed_tile_has_no_sidecar_so_it_is_folded_in_from_results(tmp_path):
    """A `failed` row exists precisely because no sidecar was written; it must not vanish."""
    _sidecar(tmp_path, "E0_N36")
    doc = mr.merge_manifest(
        tmp_path / "region_manifest.json", out_dir=tmp_path, grid_id="G1", run_record=None,
        results=[{"tile": "E0_N36", "status": "done"},
                 {"tile": "E4_N32", "status": "failed", "error": "SystemExit: GPU is lost"}])
    by = {r["tile"]: r for r in doc["results"]}
    assert by["E4_N32"]["status"] == "failed"
    assert by["E0_N36"]["status"] == "done" and by["E0_N36"]["windows"] == 144


def test_a_sidecar_beats_a_stale_failed_row(tmp_path):
    """If the tile has since rendered, the sidecar wins over an old `failed` row."""
    _sidecar(tmp_path, "E4_N32")
    doc = mr.merge_manifest(tmp_path / "m.json", out_dir=tmp_path, grid_id="G1",
                            run_record=None,
                            results=[{"tile": "E4_N32", "status": "failed", "error": "old"}])
    assert doc["results"][0]["status"] == "done"


# ---------------------------------------------------------------- the A1 driver's clobber

def test_the_a1_driver_merges_instead_of_clobbering():
    """It wrote only its own results, with a bare non-atomic `write_text`: 1 of 26 recorded."""
    src = (SRC / "striping_a1_map.py").read_text(encoding="utf-8")
    assert 'a1_manifest.json").write_text' not in src, "the clobbering write is back"
    i = src.index('"a1_manifest.json"')
    assert "merge_manifest" in src[max(0, i - 400):i + 200]


# ---------------------------------------------------------------- the repair script

def _rebuild_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rebuild_map_manifest", SRC / "rebuild_map_manifest.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_rebuild_map_manifest"] = m
    spec.loader.exec_module(m)
    return m


def test_the_repair_script_recovers_tiles_without_a_driver(tmp_path):
    rb = _rebuild_module()
    for t in ("E0_N32", "E0_N36", "E0_N40"):
        _sidecar(tmp_path, t)
    m = tmp_path / "region_manifest.json"
    mr.write_json_atomic(m, {"grid_id": "G1", "tiles": ["E0_N32"], "runs": [{"win_px": 4096}],
                             "results": [{"tile": "E0_N32", "status": "done"}]})
    before, after = rb.rebuild(tmp_path, dry_run=True, note=None)
    assert (before, after) == (1, 3)
    assert json.loads(m.read_text())["tiles"] == ["E0_N32"], "--dry-run wrote anyway"

    rb.rebuild(tmp_path, dry_run=False, note="repaired out of band")
    doc = json.loads(m.read_text())
    assert doc["tiles"] == ["E0_N32", "E0_N36", "E0_N40"]
    assert doc["runs"][0]["win_px"] == 4096                  # history preserved
    assert doc["runs"][-1]["rebuilt_from_sidecars"] is True  # repair is visible, not silent


def test_the_repair_script_refuses_two_lattices(tmp_path):
    rb = _rebuild_module()
    _sidecar(tmp_path, "E0_N32", grid="G1")
    _sidecar(tmp_path, "E0_N36", grid="G2")
    with pytest.raises(SystemExit, match="grid_id"):
        rb.rebuild(tmp_path, dry_run=True, note=None)


def test_the_repair_script_picks_the_right_manifest_name(tmp_path):
    rb = _rebuild_module()
    a1 = tmp_path / "map_a1_g2"
    a1.mkdir()
    assert rb.manifest_path(a1).name == "a1_manifest.json"
    base = tmp_path / "map_region_g2"
    base.mkdir()
    assert rb.manifest_path(base).name == "region_manifest.json"
    # an existing file wins over the name guess
    (a1 / "region_manifest.json").write_text("{}", encoding="utf-8")
    assert rb.manifest_path(a1).name == "region_manifest.json"


# ---------------------------------------------------------------- the tools must stay portable

TOOLS = ("rebuild_map_manifest.py", "verify_map_download.py")


def _imported_roots(path: Path) -> set[str]:
    """Top-level module names this file imports, from the AST -- comments and docstrings are
    allowed to *mention* numpy; what matters is whether it imports it."""
    import ast

    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


STDLIB_OK = {"__future__", "argparse", "ast", "hashlib", "importlib", "json", "os",
             "pathlib", "sys", "typing"}


@pytest.mark.parametrize("name", TOOLS)
def test_the_standalone_tools_import_no_third_party_module(name):
    """A repair tool must not share the heavy dependencies of the thing it repairs.

    `rebuild_map_manifest.py` used to load `map_region.py`, which does `import src.modeling` --
    the torch/OpenMP bootstrap -- so moving a few JSON keys around required CUDA-capable torch,
    and on a Sherlock login node it never even got that far.
    """
    roots = _imported_roots(SRC / name)
    assert roots <= STDLIB_OK | {"src"}, f"{name} imports {roots - STDLIB_OK - {'src'}}"
    src = (SRC / name).read_text(encoding="utf-8")
    assert "from src.map_manifest import" in src
    # `src.*` is only allowed to be the stdlib-only module, never the torch bootstrap
    import ast as _ast
    for node in _ast.walk(_ast.parse(src)):
        mod = getattr(node, "module", None) or ""
        if isinstance(node, _ast.ImportFrom) and mod.startswith("src"):
            assert mod == "src.map_manifest", f"{name} imports from {mod}"
        if isinstance(node, _ast.Import):
            for a in node.names:
                assert not a.name.startswith("src."), f"{name} imports {a.name}"


@pytest.mark.parametrize("name", TOOLS)
def test_the_standalone_tools_are_ascii_and_say_so_under_python2(name):
    """A `sys.version_info` check is useless if the file will not parse.

    Run bare on a Sherlock login node the default `python` is 2.7, and a non-ASCII docstring
    gave `SyntaxError: Non-ASCII character '\xe2' ... no encoding declared` -- which says
    nothing about the actual problem. ASCII-only source lets the guard below actually run.
    """
    raw = (SRC / name).read_bytes()
    raw.decode("ascii")                      # raises if any byte is non-ASCII
    src = raw.decode("ascii")
    assert "sys.version_info < (3, 8)" in src, f"{name} lost its interpreter guard"
    assert "ml python/3.12.1" in src, f"{name} does not say how to fix it"
    # the guard must precede every other import, or it cannot fire first
    assert src.index("sys.version_info") < src.index("\nimport argparse")


def test_src_map_manifest_is_standard_library_only():
    """The whole point of the module. Keep it importable under any Python 3."""
    roots = _imported_roots(Path(__file__).parents[1] / "src" / "map_manifest.py")
    assert roots <= STDLIB_OK, f"src/map_manifest.py imports {roots - STDLIB_OK}"
