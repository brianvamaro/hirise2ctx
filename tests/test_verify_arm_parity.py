"""Tests for scripts/verify_arm_parity.py -- the cross-arm comparability gate.

`verify_map_download.py` proves each arm is internally sound. Nothing checked that the two arms
could be DIFFERENCED, which is what step 12 does to them and what section 5.1's
one-common-footprint rule requires. These are the failures that gate has to catch, all of which
leave every individual raster perfect.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_verify_arm_parity", Path(__file__).parents[1] / "scripts" / "verify_arm_parity.py")
ap = importlib.util.module_from_spec(_spec)
sys.modules["_verify_arm_parity"] = ap
_spec.loader.exec_module(ap)

TOOL = Path(__file__).parents[1] / "scripts" / "verify_arm_parity.py"


def _arm(root: Path, name: str, tiles, *, grid="G1", ti_min=0, floor="d0", shape=(1479, 1479)):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for t in tiles:
        (d / f"{t}.json").write_text(json.dumps({
            "murray_tile": t, "grid_id": grid, "ti_min": ti_min, "tj_min": 0,
            "raster_shape": list(shape), "tile_px": 32, "grid_cell_m": 160.0,
            "size_floor_basis_digest": floor,
        }), encoding="utf-8")
    return d


TILES = ["E0_N32", "E0_N36", "E0_N40"]


def test_two_aligned_arms_pass(tmp_path):
    a = _arm(tmp_path, "map_region_g2", TILES)
    b = _arm(tmp_path, "map_a1_g2", TILES)
    assert ap.check([a, b], 3) == []


def test_a_second_lattice_is_refused(tmp_path):
    """R01. Each raster is individually perfect; only the grid_id shows it."""
    a = _arm(tmp_path, "map_region_g2", TILES, grid="G1")
    b = _arm(tmp_path, "map_a1_g2", TILES, grid="G2")
    bad = ap.check([a, b], 3)
    assert any("lattices" in p for p in bad)


def test_a_cell_offset_between_arms_is_refused(tmp_path):
    """Same lattice, same tiles, shifted origin -- the rows would silently mis-difference."""
    a = _arm(tmp_path, "map_region_g2", TILES, ti_min=0)
    b = _arm(tmp_path, "map_a1_g2", TILES, ti_min=1)
    bad = ap.check([a, b], 3)
    assert len([p for p in bad if "not co-registered" in p]) == 3
    assert all("ti_min" in p for p in bad if "not co-registered" in p)


def test_a_shape_mismatch_is_refused(tmp_path):
    a = _arm(tmp_path, "map_region_g2", TILES)
    b = _arm(tmp_path, "map_a1_g2", TILES, shape=(1479, 1478))
    assert any("raster_shape" in p for p in ap.check([a, b], 3))


def test_two_size_floor_bases_are_refused(tmp_path):
    """R84: the rows are differenced, so they must count the same boulders."""
    a = _arm(tmp_path, "map_region_g2", TILES, floor="d0")
    b = _arm(tmp_path, "map_a1_g2", TILES, floor="d1")
    assert any("size-floor" in p for p in ap.check([a, b], 3))


def test_differing_tile_sets_are_named(tmp_path):
    a = _arm(tmp_path, "map_region_g2", TILES)
    b = _arm(tmp_path, "map_a1_g2", TILES[:2] + ["E16_N44"])
    bad = ap.check([a, b], 0)
    assert any("tile sets differ" in p and "E0_N40" in p and "E16_N44" in p for p in bad)


def test_a_leftover_partials_dir_is_flagged(tmp_path):
    a = _arm(tmp_path, "map_region_g2", TILES)
    b = _arm(tmp_path, "map_a1_g2", TILES)
    (b / "partials" / "E0_N32").mkdir(parents=True)
    assert any("leftover partials" in p for p in ap.check([a, b], 3))


def test_an_under_indexed_manifest_is_flagged_with_the_remedy(tmp_path):
    a = _arm(tmp_path, "map_region_g2", TILES)
    b = _arm(tmp_path, "map_a1_g2", TILES)
    (a / "region_manifest.json").write_text(
        json.dumps({"tiles": ["E0_N32"], "runs": []}), encoding="utf-8")
    bad = ap.check([a, b], 3)
    assert any("indexes 1 tiles" in p and "rebuild_map_manifest" in p for p in bad)


def test_the_expected_count_is_enforced_and_can_be_waived(tmp_path):
    a = _arm(tmp_path, "map_region_g2", TILES)
    b = _arm(tmp_path, "map_a1_g2", TILES)
    assert any("expected 26" in p for p in ap.check([a, b], 26))
    assert ap.check([a, b], 0) == []


# --- same portability contract as the other standalone tools

def test_the_tool_is_ascii_stdlib_only_and_guards_the_interpreter():
    import ast

    src = TOOL.read_bytes().decode("ascii")           # raises if non-ASCII
    tree = ast.parse(src, feature_version=(3, 6))     # must parse for the guard to run
    assert not [n for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert src.splitlines()[0] == "#!/usr/bin/env python3"
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert roots <= {"argparse", "ast", "json", "pathlib", "sys", "src"}, roots
    i = src.index("sys.version_info")
    for other in ("import argparse", "import json", "from pathlib", "from src."):
        assert i < src.index(other), f"{other!r} precedes the guard"
