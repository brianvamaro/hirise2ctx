"""Unit tests for `scripts/map_union.py` -- the read surface all five PLAN_MapValidation
notebooks depend on.

The union's whole reason to exist is that `reports/map_region` and `reports/map_extended`
share 8 tiles, so pooling the two per-arm mosaics double-counts 15% of the footprint. That
makes the *dedup* the load-bearing part, and the failure mode is silent: a wrong union still
loads, still computes, and is simply a statement about the wrong ground. So the tests pin the
refusals (sha mismatch, mixed size-floor basis, writing into a source arm, a layer-dependent
footprint) rather than just the happy path, plus the `MANIFEST_NAMES` entry -- because
`plan.json` not being on that list once made `verify_map_download.py` see a second lattice in
a one-lattice product.

Everything runs on tiny synthetic rasters under `tmp_path`; the one test that touches the
shipped products only *reads* them, and skips if they are absent.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from src.mapping import COARSE_GRID_ID, write_geotiff  # noqa: E402

import map_union  # noqa: E402

LAYERS = ("abundance", "prob", "prob_raw")

SIZE_FLOOR = {"SIZE_FLOOR_BASIS_VERSION": "v2_mixed_floor_2",
              "SIZE_FLOOR_GLOBAL_MIN_SIZE_M": "1.4105"}


def _proj_crs():
    """The map's real projected CRS, taken from a shipped tile when one is available.

    `assert_murray_sphere` reads the sphere radius out of the WKT, so the tests need a WKT it
    accepts. Building one by hand is how a test starts asserting against a CRS the product
    does not use, so prefer the product's own.
    """
    for arm in ("map_region", "map_extended"):
        for p in (REPO / "reports" / arm).glob("*_abundance.tif"):
            with rasterio.open(p) as ds:
                return ds.crs.to_wkt()
    return None


@pytest.fixture(scope="module")
def crs_wkt():
    wkt = _proj_crs()
    if wkt is None:
        pytest.skip("no shipped map product to take a CRS from")
    return wkt


def _tile_transform(lon_i: int, lat_i: int, n: int):
    """A transform on the real global coarse lattice, built by the project's own helper.

    Hand-rolling this is how a test ends up asserting against a lattice the product does not
    use -- and `mosaic_geotiffs(require_shared_lattice=True)` is the R01 gate the union relies
    on, so the synthetic tiles have to pass it for real. Each 4-degree tile id maps to its own
    `n`-cell block, so the tiles abut without overlapping.
    """
    from src.mapping import global_cell_transform

    return global_cell_transform((-lat_i // 4) * n, (lon_i // 4) * n)


def make_tile(d: Path, tile: str, crs_wkt: str, *, n: int = 8, seed: int = 0,
              tags: dict | None = None, nodata_cells: int = 0, layers=LAYERS) -> None:
    """Write a `{tile}_{layer}.tif` trio on the global lattice, deterministic from `seed`."""
    import re

    d.mkdir(parents=True, exist_ok=True)
    m = re.fullmatch(r"E(-?\d+)_N(-?\d+)", tile)
    tr = _tile_transform(int(m.group(1)), int(m.group(2)), n)
    for k, layer in enumerate(layers):
        rng = np.random.default_rng(seed * 100 + k)
        a = rng.random((n, n)).astype(np.float32)
        if nodata_cells:
            flat = a.ravel()
            flat[:nodata_cells] = np.nan
        t = dict(SIZE_FLOOR)
        t.update(tags or {})
        write_geotiff(d / f"{tile}_{layer}.tif", a, tr, crs_wkt, tags=t)


# --------------------------------------------------------------------- dedup, the core rule
def test_resolve_union_dedups_a_shared_tile_and_records_both_arms(tmp_path, crs_wkt):
    a, b = tmp_path / "arm_a", tmp_path / "arm_b"
    make_tile(a, "E-12_N32", crs_wkt, seed=1)
    make_tile(b, "E-12_N32", crs_wkt, seed=1)          # same seed -> same bytes
    make_tile(b, "E-16_N32", crs_wkt, seed=2)

    chosen, origin = map_union.resolve_union([a, b], "abundance")
    assert sorted(chosen) == ["E-12_N32", "E-16_N32"]
    assert chosen["E-12_N32"].parent == a              # first source wins, deterministically
    assert origin["E-12_N32"] == ["arm_a", "arm_b"]
    assert origin["E-16_N32"] == ["arm_b"]


def test_resolve_union_REFUSES_a_shared_tile_whose_bytes_differ(tmp_path, crs_wkt):
    """The failure that must never be merged: two arms rendered the same ground differently."""
    a, b = tmp_path / "arm_a", tmp_path / "arm_b"
    make_tile(a, "E-12_N32", crs_wkt, seed=1)
    make_tile(b, "E-12_N32", crs_wkt, seed=99)

    with pytest.raises(SystemExit, match="differs between"):
        map_union.resolve_union([a, b], "abundance")


def test_union_footprint_counts_a_shared_tile_ONCE(tmp_path, crs_wkt):
    """The bug the union exists to prevent, measured: 3 distinct tiles, not 4 rasters."""
    a, b = tmp_path / "arm_a", tmp_path / "arm_b"
    out = tmp_path / "union"
    make_tile(a, "E-12_N32", crs_wkt, n=8, seed=1)
    make_tile(a, "E-8_N32", crs_wkt, n=8, seed=2)
    make_tile(b, "E-12_N32", crs_wkt, n=8, seed=1)     # the shared tile
    make_tile(b, "E-16_N32", crs_wkt, n=8, seed=3)

    chosen, origin = map_union.resolve_union([a, b], "abundance")
    rec = map_union.build(chosen, origin, out, "abundance", seams=False)
    assert rec["n_tiles"] == 3
    assert rec["n_adopted_tiles"] == 1
    assert rec["n_finite"] == 3 * 8 * 8                # footprint closes on 3 tiles
    assert rec["footprint_closes"] is True


def test_union_footprint_closes_with_intra_tile_nodata(tmp_path, crs_wkt):
    """A closed account, not a plausible percentage: named nodata on a named tile."""
    a = tmp_path / "arm_a"
    out = tmp_path / "union"
    make_tile(a, "E-12_N32", crs_wkt, n=8, seed=1, nodata_cells=5)
    make_tile(a, "E-8_N32", crs_wkt, n=8, seed=2)

    chosen, origin = map_union.resolve_union([a], "abundance")
    rec = map_union.build(chosen, origin, out, "abundance", seams=False)
    assert rec["intra_tile_nodata"] == {"E-12_N32": 5}
    assert rec["n_finite"] == 2 * 8 * 8 - 5


# ------------------------------------------------------------------ the honesty guardrails
def test_scan_tiles_refuses_a_mixed_size_floor_basis(tmp_path, crs_wkt):
    """`abundance` has no meaning without one basis, so a mixed union must not be reportable."""
    a, b = tmp_path / "arm_a", tmp_path / "arm_b"
    make_tile(a, "E-12_N32", crs_wkt, seed=1)
    make_tile(b, "E-16_N32", crs_wkt, seed=2,
              tags={"SIZE_FLOOR_BASIS_VERSION": "v3_something_else"})

    chosen, _ = map_union.resolve_union([a, b], "abundance")
    with pytest.raises(SystemExit, match="SIZE_FLOOR"):
        map_union.scan_tiles(chosen, "abundance")


def test_check_out_dir_refuses_writing_into_a_source_arm(tmp_path):
    a, b = tmp_path / "arm_a", tmp_path / "arm_b"
    with pytest.raises(SystemExit, match="also a --source"):
        map_union.check_out_dir(a, [a, b])


def test_check_out_dir_refuses_a_duplicated_source(tmp_path):
    a = tmp_path / "arm_a"
    with pytest.raises(SystemExit, match="same directory twice"):
        map_union.check_out_dir(tmp_path / "union", [a, a])


def test_main_refuses_a_layer_dependent_footprint(tmp_path, crs_wkt):
    """Ruling 3's three targets must describe the SAME cells, so this is a defect."""
    a = tmp_path / "arm_a"
    make_tile(a, "E-12_N32", crs_wkt, seed=1)
    make_tile(a, "E-16_N32", crs_wkt, seed=2, layers=("abundance", "prob"))  # no prob_raw

    with pytest.raises(SystemExit, match="do not cover the same tiles"):
        map_union.main(["--source", str(a), "--out", str(tmp_path / "union")])


def test_main_reports_a_missing_source_dir(tmp_path):
    with pytest.raises(SystemExit, match="no such source dir"):
        map_union.main(["--source", str(tmp_path / "nope"),
                        "--out", str(tmp_path / "union")])


def test_arm_tiles_skips_the_arms_own_regional_mosaic(tmp_path, crs_wkt):
    """`regional_abundance_mosaic.tif` is a product, not a tile; folding it in would
    double-count the entire arm."""
    a = tmp_path / "arm_a"
    make_tile(a, "E-12_N32", crs_wkt, seed=1)
    (a / "regional_abundance.tif").write_bytes(b"not a tile")
    assert map_union.arm_tiles(a, "abundance") == ["E-12_N32"]


# ------------------------------------------------------------------------- end to end
def test_main_end_to_end_writes_mosaics_a_manifest_and_union_tags(tmp_path, crs_wkt):
    a, b = tmp_path / "arm_a", tmp_path / "arm_b"
    out = tmp_path / "union"
    make_tile(a, "E-12_N32", crs_wkt, n=8, seed=1)
    make_tile(b, "E-12_N32", crs_wkt, n=8, seed=1)
    make_tile(b, "E-16_N32", crs_wkt, n=8, seed=2)

    assert map_union.main(["--source", str(a), str(b), "--out", str(out)]) == 0

    for layer in LAYERS:
        assert (out / f"regional_{layer}_mosaic.tif").exists()
    with rasterio.open(out / "regional_abundance_mosaic.tif") as ds:
        tags = ds.tags()
    assert tags["UNION_N_TILES"] == "2"
    assert tags["MOSAIC_ARM"] == "union"
    assert tags["MOSAIC_GRID_ID"] == COARSE_GRID_ID
    assert tags["UNION_ADOPTED_TILES"] == "E-12_N32"
    assert json.loads(tags["UNION_TILE_ORIGIN"]) == {"E-12_N32": "arm_a",
                                                     "E-16_N32": "arm_b"}
    # the size-floor basis is carried forward, not dropped
    assert tags["SIZE_FLOOR_BASIS_VERSION"] == "v2_mixed_floor_2"

    man = json.loads((out / "union_manifest.json").read_text(encoding="utf-8"))
    assert man["grid_id"] == COARSE_GRID_ID
    assert set(man["layers"]) == set(LAYERS)
    assert man["layers"]["abundance"]["shared_tiles"] == ["E-12_N32"]


def test_a_new_source_dir_joins_with_no_code_change(tmp_path, crs_wkt):
    """The plan-driven convention: growing the map edits one argument, not the script."""
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "round2"
    make_tile(a, "E-12_N32", crs_wkt, seed=1)
    make_tile(b, "E-16_N32", crs_wkt, seed=2)
    make_tile(c, "E-20_N32", crs_wkt, seed=3)

    out = tmp_path / "union"
    assert map_union.main(["--source", str(a), str(b), "--out", str(out)]) == 0
    with rasterio.open(out / "regional_abundance_mosaic.tif") as ds:
        assert ds.tags()["UNION_N_TILES"] == "2"

    out2 = tmp_path / "union2"
    assert map_union.main(["--source", str(a), str(b), str(c), "--out", str(out2)]) == 0
    with rasterio.open(out2 / "regional_abundance_mosaic.tif") as ds:
        assert ds.tags()["UNION_N_TILES"] == "3"


def test_union_manifest_is_not_mistaken_for_a_tile_sidecar():
    """⚠ CLAUDE.md: an unlisted JSON in a map-output dir reads as a corrupt tile on a second
    lattice. `union_manifest.json` lives in one, so it must be on the denylist."""
    from src.map_manifest import MANIFEST_NAMES

    assert "union_manifest" in MANIFEST_NAMES


# --------------------------------------------------- the real products (read-only, skippable)
@pytest.mark.slow
def test_the_shipped_arms_share_only_byte_identical_tiles():
    r"""The property the union is built on, checked against whatever is on disk.

    ⚠ **This deliberately asserts no tile COUNT.** It first read `len(chosen) == 53` and
    `arm_tiles(extended) == 35`, which was true for about a day: round 2 rendered, the
    extension went 35 -> 104, the union went 53 -> 122, and the test failed on a change that
    was entirely correct. That is a snapshot of one rendering round masquerading as an
    invariant -- the same defect this same commit fixed in
    `test_plan_map_extent.test_the_shipped_plan_pins_the_rebuild_head`.

    What must hold for *any* round: the union is the set union, the shared tiles are exactly
    the intersection, every shared tile is byte-identical on every layer, and all three layers
    cover the same tiles. The actual counts belong in `union_manifest.json`, which records what
    was built, not in an assertion that has to be edited whenever the map grows.
    """
    from src.map_manifest import file_sha256

    region = REPO / "reports" / "map_region"
    extended = REPO / "reports" / "map_extended"
    if not (region.is_dir() and extended.is_dir()):
        pytest.skip("both shipped arms not present in this checkout")

    per_layer = {}
    for layer in LAYERS:
        a = set(map_union.arm_tiles(region, layer))
        b = set(map_union.arm_tiles(extended, layer))
        assert a and b, f"{layer}: an arm has no tiles"
        chosen, origin = map_union.resolve_union([region, extended], layer)  # asserts equality
        shared = {t for t in chosen if len(origin[t]) > 1}
        assert set(chosen) == a | b                  # the union is the set union
        assert shared == a & b                       # ...and shared is exactly the overlap
        assert len(chosen) == len(a) + len(b) - len(shared)
        for t in sorted(shared):
            assert (file_sha256(region / f"{t}_{layer}.tif")
                    == file_sha256(extended / f"{t}_{layer}.tif")), f"{t}/{layer}"
        per_layer[layer] = frozenset(chosen)

    # the three targets must describe the same tiles, or ruling 3 is not comparable
    assert len(set(per_layer.values())) == 1, {k: len(v) for k, v in per_layer.items()}


@pytest.mark.slow
def test_the_shipped_arms_carry_ONE_size_floor_basis():
    region = REPO / "reports" / "map_region"
    extended = REPO / "reports" / "map_extended"
    if not (region.is_dir() and extended.is_dir()):
        pytest.skip("both shipped arms not present in this checkout")
    chosen, _ = map_union.resolve_union([region, extended], "abundance")
    tags, _ = map_union.scan_tiles(chosen, "abundance")      # raises on a mixed basis
    assert tags["SIZE_FLOOR_BASIS_VERSION"] == "v2_mixed_floor_2"
