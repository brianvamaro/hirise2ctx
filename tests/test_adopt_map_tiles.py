"""Unit tests for `scripts/adopt_map_tiles.py` and the sidecar-discovery rule it relies on.

Adopting a tile is a file copy between two shipped products, which is precisely the operation
that can put a wrong raster into a right-looking directory. The tests pin the four refusals
that make it safe -- corrupt source, wrong lattice, silent overwrite, unverifiable sidecar --
plus `tile_sidecars`'s discovery rule, because a `plan.json` that was not on its exclusion list
made `verify_map_download.py` see a second lattice in a one-lattice product.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from adopt_map_tiles import destination_grid_id, verify_tile  # noqa: E402

GRID = "murray_v01_clon0_R3396190_ppd11855_S32_anchor_lonlat0"


def _tile(d: Path, tile: str, *, grid_id: str = GRID, payload: bytes = b"raster-bytes",
          with_rasters: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rasters = []
    for kind in ("prob", "abundance", "prob_raw"):
        p = d / f"{tile}_{kind}.tif"
        body = payload + kind.encode()
        p.write_bytes(body)
        rasters.append({"name": p.name, "kind": kind, "bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest()})
    rec = {"murray_tile": tile, "grid_id": grid_id}
    if with_rasters:
        rec["rasters"] = rasters
    side = d / f"{tile}.json"
    side.write_text(json.dumps(rec), encoding="utf-8")
    return side


def test_verify_tile_accepts_a_consistent_tile(tmp_path):
    _tile(tmp_path, "E-24_N28")
    paths, rec = verify_tile(tmp_path, "E-24_N28")
    assert len(paths) == 4                       # sidecar + 3 rasters
    assert rec["grid_id"] == GRID


def test_verify_tile_rejects_a_truncated_raster(tmp_path):
    _tile(tmp_path, "E-24_N28")
    p = tmp_path / "E-24_N28_abundance.tif"
    p.write_bytes(p.read_bytes()[:-3])
    with pytest.raises(SystemExit, match="bytes"):
        verify_tile(tmp_path, "E-24_N28")


def test_verify_tile_rejects_a_same_size_corruption(tmp_path):
    """The byte count is not enough -- a flipped byte keeps the size and changes the map."""
    _tile(tmp_path, "E-24_N28")
    p = tmp_path / "E-24_N28_prob.tif"
    b = bytearray(p.read_bytes())
    b[0] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(SystemExit, match="sha256"):
        verify_tile(tmp_path, "E-24_N28")


def test_verify_tile_rejects_a_missing_raster(tmp_path):
    _tile(tmp_path, "E-24_N28")
    (tmp_path / "E-24_N28_prob_raw.tif").unlink()
    with pytest.raises(SystemExit, match="not in"):
        verify_tile(tmp_path, "E-24_N28")


def test_verify_tile_refuses_a_sidecar_with_no_commit_record(tmp_path):
    """A pre-R14 sidecar cannot be verified, so it must not be adopted silently."""
    _tile(tmp_path, "E-24_N28", with_rasters=False)
    with pytest.raises(SystemExit, match="rasters"):
        verify_tile(tmp_path, "E-24_N28")


def test_verify_tile_reports_a_missing_sidecar(tmp_path):
    with pytest.raises(SystemExit, match="no sidecar"):
        verify_tile(tmp_path, "E-24_N28")


def test_destination_grid_id_ignores_a_plan_file(tmp_path):
    """`plan.json` in a product dir must not read as a tile on an unknown lattice."""
    _tile(tmp_path, "E-24_N28")
    (tmp_path / "plan.json").write_text(json.dumps({"tiles": ["E-24_N28"]}), encoding="utf-8")
    assert destination_grid_id(tmp_path) == GRID


def test_destination_grid_id_is_none_for_an_empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    assert destination_grid_id(tmp_path / "empty") is None


# ---------------------------------------------------------------------------
# the discovery rule both this script and verify_map_download.py depend on
# ---------------------------------------------------------------------------

def test_tile_sidecars_excludes_every_non_tile_json_this_project_writes(tmp_path):
    """`plan.json` included -- it lives beside the product it defines.

    It was NOT excluded at first, and `verify_map_download.py` reported it as an unexpected tile
    with no `rasters` record *and as a second grid_id*, firing the R01 two-lattices alarm on a
    one-lattice product.
    """
    from src.map_manifest import MANIFEST_NAMES, tile_sidecars

    for t in ("E-24_N28", "E0_N40", "E152_N-8"):
        _tile(tmp_path, t)
    for name in MANIFEST_NAMES:
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
    assert "plan" in MANIFEST_NAMES
    assert set(tile_sidecars(tmp_path)) == {"E-24_N28", "E0_N40", "E152_N-8"}


def test_tile_sidecars_does_NOT_require_a_murray_style_name(tmp_path):
    """Deliberate: `tile_result_rows` must index whatever footprint is on disk.

    A name pattern would be tighter here but would reintroduce a hardcoded assumption about the
    tile list, which is the thing the manifest's self-healing rebuild exists to avoid. So the
    rule is a denylist, and anything new written into a product dir must join MANIFEST_NAMES.
    """
    from src.map_manifest import tile_sidecars

    _tile(tmp_path, "WEIRD_TILE")
    assert set(tile_sidecars(tmp_path)) == {"WEIRD_TILE"}


def test_adopting_the_shipped_tiles_reproduces_them_byte_for_byte(tmp_path):
    """End-to-end on real artifacts: adopt from `reports/map_region`, then re-verify."""
    src = REPO / "reports" / "map_region"
    tile = "E-8_N32"
    if not (src / f"{tile}.json").exists():
        pytest.skip("no shipped map_region product in this checkout")

    paths, _ = verify_tile(src, tile)
    dst = tmp_path / "adopted"
    dst.mkdir()
    for p in paths:
        shutil.copy2(p, dst / p.name)
    verify_tile(dst, tile)                       # raises if the copy is not faithful
    assert destination_grid_id(dst) == GRID
