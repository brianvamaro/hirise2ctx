"""Tests for scripts/verify_map_download.py -- the sidecar-vs-file integrity check.

R14 gave every tile sidecar a `rasters[]` commit record (name, bytes, sha256) so content could
be checked without re-deriving it. Nothing consumed it until 2026-08-25, which meant a transfer
off Sherlock was trusted on file count alone.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_verify_map_download", Path(__file__).parents[1] / "scripts" / "verify_map_download.py")
vd = importlib.util.module_from_spec(_spec)
sys.modules["_verify_map_download"] = vd
_spec.loader.exec_module(vd)


def test_the_tile_list_matches_the_driver():
    """`verify_map_download` duplicates BLOCK_TILES so it can run without CUDA. Pin them equal."""
    src = (Path(__file__).parents[1] / "scripts" / "map_region.py").read_text(encoding="utf-8")
    i = src.index("BLOCK_TILES")
    lit = src[src.index("[", i):src.index("]", i) + 1]
    from_driver = [t.strip().strip('"\'') for t in lit.strip("[]").split(",") if t.strip()]
    assert from_driver == vd.BLOCK_TILES, "the duplicated tile list has drifted"
    assert len(vd.BLOCK_TILES) == 26


def _tile(d: Path, tile: str, *, payload=b"raster-bytes", overlap=None, corrupt=False):
    """Write one synthetic tile: a raster plus a sidecar whose record describes it."""
    import hashlib

    p = d / f"{tile}_prob.tif"
    p.write_bytes(payload)
    rec = {"murray_tile": tile, "grid_id": "G1",
           "rasters": [{"name": p.name, "kind": "prob", "bytes": len(payload),
                        "sha256": hashlib.sha256(
                            b"WRONG" if corrupt else payload).hexdigest()}]}
    if overlap is not None:
        rec["overlap"] = overlap
    (d / f"{tile}.json").write_text(json.dumps(rec), encoding="utf-8")
    return p


def test_a_clean_directory_reports_no_problems(tmp_path):
    _tile(tmp_path, "T1")
    assert vd.verify_dir(tmp_path, expect=["T1"], quick=False) == []


def test_a_corrupted_raster_is_caught_by_hash_but_not_by_size(tmp_path):
    """The whole point: right name, right size, wrong content."""
    _tile(tmp_path, "T1", corrupt=True)
    assert vd.verify_dir(tmp_path, expect=["T1"], quick=True) == []       # size-only misses it
    bad = vd.verify_dir(tmp_path, expect=["T1"], quick=False)
    assert len(bad) == 1 and "sha256" in bad[0]


def test_a_truncated_raster_is_caught(tmp_path):
    p = _tile(tmp_path, "T1")
    p.write_bytes(b"trunc")
    bad = vd.verify_dir(tmp_path, expect=["T1"], quick=False)
    assert len(bad) == 1 and "size" in bad[0]


def test_a_missing_raster_and_a_missing_tile_are_distinguished(tmp_path):
    p = _tile(tmp_path, "T1")
    p.unlink()
    bad = vd.verify_dir(tmp_path, expect=["T1", "T2"], quick=False)
    assert any("MISSING" in b for b in bad)
    assert any("NO sidecar" in b and "T2" in b for b in bad)


def test_two_lattices_in_one_directory_are_refused(tmp_path):
    _tile(tmp_path, "T1")
    _tile(tmp_path, "T2")
    rec = json.loads((tmp_path / "T2.json").read_text())
    rec["grid_id"] = "G2"                      # R01: a different lattice
    (tmp_path / "T2.json").write_text(json.dumps(rec), encoding="utf-8")
    bad = vd.verify_dir(tmp_path, expect=["T1", "T2"], quick=False)
    assert any("grid_id" in b for b in bad)


def test_manifests_are_not_mistaken_for_tiles(tmp_path):
    _tile(tmp_path, "T1")
    for name in ("region_manifest.json", "a1_manifest.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert vd.verify_dir(tmp_path, expect=["T1"], quick=False) == []


# --- the three sidecar schema generations must be reported DIFFERENTLY, never averaged

def test_overlap_line_names_the_pre_floor_schema_as_raw():
    line = vd.overlap_line({"overlap": {
        "gate_layer": "prob_raw",
        "prob_raw": {"n_dup": 61596, "n_disagree": 482, "fraction": 0.007825,
                     "max_abs": 2.09e-07}}})
    assert "pre-floor" in line and "RAW fraction" in line and "0.7825 %" in line


def test_overlap_line_reports_both_counts_for_the_floor_schema():
    line = vd.overlap_line({"overlap": {
        "gate_layer": "prob_raw",
        "prob_raw": {"n_dup": 61596, "n_disagree": 482, "n_significant": 0,
                     "fraction": 0.0, "fraction_raw": 0.007825, "max_abs": 2.09e-07}}})
    assert "[floor]" in line and "0/61596 significant" in line and "482 at any" in line


def test_overlap_line_handles_the_oldest_scalar_only_sidecars():
    """The 21 baseline tiles rendered before any of this carry only the scalar."""
    assert "scalar only" in vd.overlap_line({"overlap_disagreements": 0})
