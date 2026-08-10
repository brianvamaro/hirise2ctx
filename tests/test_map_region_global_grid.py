"""R01 part 2 — the *drivers* render onto the one global coarse lattice.

Part 1 added the grid vocabulary and proved it in isolation; every mutant that mattered
lived in the drivers, which still anchored each tile to its own pixel origin. These tests
pin the wiring itself:

  * the read-window sweep computes **every** cell at **every** phase (it did not: 11 lost
    per axis at all seven non-zero phases, and each half of the fix alone still loses some);
  * `map_one_tile` / `process_tile` actually pass `global_grid`, and the assembled affine
    comes from the global lattice rather than the parent tile;
  * per-window partials carry `grid_id`, and a resumed run refuses to mix lattices;
  * adjacent Murray tiles claim disjoint, correctly-spaced global cells across the seam;
  * the shipped products are checked against the lattice as an acceptance gate.

All read-only and GPU-free: real tile geometry comes from the cached sidecars, inference is
a stub. Nothing here writes outside `tmp_path`.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.mapping import (COARSE_GRID_ID, MURRAY_NATIVE_M, assert_coregistered,
                         assert_shared_lattice, global_cell_transform, tile_global_grid,
                         uncovered_cells, window_offsets)

REPO = Path(__file__).resolve().parents[1]
CTX_TILES = REPO / "cache_v2" / "ctx_tiles"
MAP_DIR = REPO / "reports" / "map_region"

TILE_PX = 32
EXTENT = 47420          # a Murray V01 tile, native px
WIN = 4096              # the shipped --win-px
PHASES = list(range(0, TILE_PX, 4))   # gcd(47420,32)=4, so only multiples of 4 occur


# ---------------------------------------------------------------- the window sweep

@pytest.mark.parametrize("phase", PHASES)
def test_the_sweep_computes_every_cell_at_every_phase(phase):
    """The configuration the drivers use must lose nothing, at any tile's phase."""
    offs = window_offsets(EXTENT, WIN, 3 * TILE_PX, TILE_PX, tile_aligned=False)
    assert uncovered_cells(offs, EXTENT, WIN, TILE_PX, phase=phase) == []


@pytest.mark.parametrize("phase", PHASES)
def test_each_half_of_the_sweep_fix_alone_still_loses_cells(phase):
    """Kills the two mutants independently — neither edit is redundant.

    Measured over the real tile size: 11 lost per axis in the shipped configuration, 1 with
    only the overlap widened, 10 with only the final offset unpinned, 0 with both. Phase 0
    loses nothing in every configuration, which is exactly why this never bit before.
    """
    def lost(tile_aligned, overlap):
        offs = window_offsets(EXTENT, WIN, overlap, TILE_PX, tile_aligned=tile_aligned)
        return len(uncovered_cells(offs, EXTENT, WIN, TILE_PX, phase=phase))

    expected = {0: (0, 0, 0, 0)}.get(phase, (11, 1, 10, 0))
    assert (lost(True, 2 * TILE_PX), lost(True, 3 * TILE_PX),
            lost(False, 2 * TILE_PX), lost(False, 3 * TILE_PX)) == expected


def test_widening_the_overlap_does_not_cost_a_single_extra_window():
    """The fix is free — if it ever stops being free, that is a budget decision, not a silent one."""
    shipped = window_offsets(EXTENT, WIN, 2 * TILE_PX, TILE_PX, tile_aligned=True)
    fixed = window_offsets(EXTENT, WIN, 3 * TILE_PX, TILE_PX, tile_aligned=False)
    assert len(fixed) == len(shipped) == 12


def test_legacy_tile_aligned_offsets_are_unchanged():
    """`f_region_stageb` sweeps ISIS cubes on their own phase-0 grid and keeps the default;
    changing the shared helper must not move its windows or its partial filenames."""
    offs = window_offsets(EXTENT, WIN, 2 * TILE_PX, TILE_PX)
    assert offs == [0, 4032, 8064, 12096, 16128, 20160, 24192, 28224, 32256, 36288, 40320, 43296]
    assert all(o % TILE_PX == 0 for o in offs)


def test_uncovered_cells_reports_the_holes_not_just_their_count():
    offs = window_offsets(EXTENT, WIN, 2 * TILE_PX, TILE_PX, tile_aligned=True)
    holes = uncovered_cells(offs, EXTENT, WIN, TILE_PX, phase=4)
    assert holes[:4] == [4036, 8068, 12100, 16132]
    assert all(h % TILE_PX == 4 for h in holes)


# ---------------------------------------------------------------- real tile geometry

def _sidecars():
    if not CTX_TILES.exists():
        return []
    out = []
    for p in sorted(CTX_TILES.glob("*.json")):
        info = json.loads(p.read_text(encoding="utf-8"))
        if info.get("inner_transform") and info.get("inner_crs_wkt"):
            out.append((p.stem, info))
    return out


@pytest.mark.skipif(not _sidecars(), reason="no cached Murray tile sidecars")
def test_every_cached_tile_places_on_the_global_lattice_with_exact_division():
    for name, info in _sidecars():
        g = tile_global_grid(info["inner_transform"], info["inner_crs_wkt"], TILE_PX)
        a, b, c, d, e, f = [float(v) for v in info["inner_transform"][:6]]
        # the first global cell's north-west corner, reconstructed two independent ways
        from_tile_px = (c + g.phase_c * MURRAY_NATIVE_M, f - g.phase_r * MURRAY_NATIVE_M)
        from_global = global_cell_transform(g.cell_row0, g.cell_col0, TILE_PX)
        assert from_global.c == pytest.approx(from_tile_px[0], abs=1e-6), name
        assert from_global.f == pytest.approx(from_tile_px[1], abs=1e-6), name


@pytest.mark.skipif(not _sidecars(), reason="no cached Murray tile sidecars")
def test_a_tile_yields_1479_cells_per_axis_at_every_phase():
    """The corrected product keeps the shipped per-tile shape — the grid moves, the size does not."""
    for name, info in _sidecars():
        g = tile_global_grid(info["inner_transform"], info["inner_crs_wkt"], TILE_PX)
        H, W = info["inner_shape"]
        for extent, phase in ((H, g.phase_r), (W, g.phase_c)):
            offs = window_offsets(extent, WIN, 3 * TILE_PX, TILE_PX, tile_aligned=False)
            assert uncovered_cells(offs, extent, WIN, TILE_PX, phase=phase) == [], name
            n = len(range(TILE_PX + phase, extent - 2 * TILE_PX + 1, TILE_PX))
            assert n == 1479, f"{name}: {n} cells at phase {phase}"


@pytest.mark.skipif(len(_sidecars()) < 2, reason="need two adjacent cached tiles")
def test_adjacent_tiles_claim_disjoint_correctly_spaced_global_cells():
    """Kills an off-by-one in `cell_row0`/`cell_col0`: the seam must be 2 or 3 empty cells,
    and no global cell may be claimed by both neighbours."""
    have = dict(_sidecars())
    pairs = [("E4_N40", "E8_N40"), ("E4_N44", "E8_N44"), ("E12_N44", "E16_N44")]
    tested = 0
    for west, east in pairs:
        if west not in have or east not in have:
            continue
        gw = tile_global_grid(have[west]["inner_transform"], have[west]["inner_crs_wkt"], TILE_PX)
        ge = tile_global_grid(have[east]["inner_transform"], have[east]["inner_crs_wkt"], TILE_PX)
        last_w = gw.cell_col0 + 1479          # tj_local runs 1..1479
        first_e = ge.cell_col0 + 1
        assert first_e > last_w, f"{west}/{east} overlap: {last_w} >= {first_e}"
        assert first_e - last_w - 1 in (2, 3), f"{west}/{east} seam = {first_e - last_w - 1}"
        tested += 1
    if not tested:
        pytest.skip("no adjacent pair cached")


# ---------------------------------------------------------------- driver wiring

class _StubEmbedder:
    """Embeds each cell as its own window-local slice origin, so a wiring error is visible."""
    def embed_window(self, arr, ti, tj, *, tile_px, row0, col0, pool, batch):
        from src.fm_embeddings import slice_context_boxes
        boxes, valid = slice_context_boxes(arr, ti, tj, tile_px, row0, col0)
        emb = np.full((valid.size, 4), np.nan, dtype=np.float32)
        emb[np.where(valid)[0], 0] = 0.5
        return emb, valid


class _StubHead:
    def predict(self, emb):
        return np.full(emb.shape[0], 0.5, dtype=np.float64)


def _fake_args(tmp_path, **kw):
    import argparse
    a = argparse.Namespace(
        out_dir=str(tmp_path), win_px=256, batch=4, max_zero_fraction=0.9,
        no_isotonic=True, force=False, limit_windows=None, clean_partials=False,
        calibration="", ctx_tiles=str(tmp_path / "ctx"), _model_dir=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


_WKT = ('PROJCS["Mars_2015_Equidistant_Cylindrical_clon0",'
        'GEOGCS["Mars_2015",DATUM["Mars_2015",'
        'SPHEROID["Mars_2015",3396190,169.894447223612]],'
        'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
        'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH]]')

# A synthetic tile far from the anchor, so a global cell index can never be mistaken for a
# tile-local one: cell_row0 = -10000, cell_col0 = -20000. Its extent is 1052 px, chosen
# `== 28 (mod 32)` like the real 47,420 — with a round extent, `tile_aligned=True` and
# `extent - win` coincide and the driver's final-offset mutant is invisible.
_G_ROW, _G_COL = -10000 * TILE_PX - 20, -20000 * TILE_PX - 4      # phases 20 and 4


def _synthetic_tile(tmp_path, tile="E4_N40", extent=1052):
    """A tiny Murray-like tile whose origin sits on the global native lattice with a phase."""
    ctx = tmp_path / "ctx"
    ctx.mkdir(exist_ok=True)
    transform = [MURRAY_NATIVE_M, 0.0, _G_COL * MURRAY_NATIVE_M,
                 0.0, -MURRAY_NATIVE_M, -_G_ROW * MURRAY_NATIVE_M]
    (ctx / f"{tile}.json").write_text(json.dumps({
        "inner_tif": "inner.tif", "inner_transform": transform,
        "inner_crs_wkt": _WKT, "inner_shape": [extent, extent]}), encoding="utf-8")
    (ctx / f"{tile}.zip").write_bytes(b"")     # existence check only; reads are monkeypatched
    return tile, transform, extent


def test_map_one_tile_renders_on_the_global_lattice(tmp_path, monkeypatch):
    """End to end through the real driver with inference stubbed: the written GeoTIFF must
    sit on the global lattice, and its cells must be global indices.

    This is the test that would have caught R01 itself. Every piece of the grid vocabulary
    can be correct while the driver never calls it — which is exactly what part 1 shipped.
    """
    rasterio = pytest.importorskip("rasterio")
    import scripts.map_region as mr
    from src.mapping import CtxWindow

    tile, transform, extent = _synthetic_tile(tmp_path)
    a, b, c, d, e, f = transform

    def fake_read(zip_path, inner_tif, row_off, col_off, size):
        h, w = min(size, extent - row_off), min(size, extent - col_off)
        return CtxWindow(data=np.full((h, w), 200, dtype=np.uint8),
                         row_off=row_off, col_off=col_off,
                         transform=(a, b, c + col_off * a, d, e, f + row_off * e),
                         crs_wkt=_WKT)

    monkeypatch.setattr(mr, "read_tile_window", fake_read)
    status = mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                             args=_fake_args(tmp_path, win_px=256))
    assert status["status"] == "done", status

    with rasterio.open(tmp_path / f"{tile}_prob.tif") as ds:
        assert_shared_lattice([ds.transform], tile_px=TILE_PX)   # THE acceptance property
        got, shape = ds.transform, (ds.height, ds.width)
    side = json.loads((tmp_path / f"{tile}.json").read_text(encoding="utf-8"))
    assert side["grid_id"] == COARSE_GRID_ID
    assert side["grid_tile_px"] == TILE_PX
    assert side["grid_radius_m"] == 3396190.0
    assert side["grid_phase_px"] == [20, 4]
    # the affine must be reconstructible from the recorded global cell indices alone
    expect = global_cell_transform(side["ti_min"], side["tj_min"], TILE_PX)
    assert (got.a, got.c, got.e, got.f) == (expect.a, expect.c, expect.e, expect.f)
    # GLOBAL, not tile-local: the first cell of this tile is (-10000+1, -20000+1)
    assert (side["ti_min"], side["tj_min"]) == (-9999, -19999)
    # no holes: every cell the tile can support is present
    n = len(range(TILE_PX + 20, extent - 2 * TILE_PX + 1, TILE_PX))
    assert shape == (n, n)
    assert side["n_predicted_tiles"] >= n * n


def test_the_driver_refuses_a_sweep_that_would_leave_holes(tmp_path, monkeypatch):
    """The coverage guard must actually fire — with the right constants it never does, so
    without this the guard could be deleted and every test would still pass.

    Drops one interior offset, which is exactly the shape of the defect: the product still
    renders, still looks plausible, and has a one-cell stripe of nodata through it.
    """
    import scripts.map_region as mr
    from src.mapping import CtxWindow, window_offsets as real_offsets

    tile, transform, extent = _synthetic_tile(tmp_path)
    a, b, c, d, e, f = transform

    def holey(*args, **kw):
        offs = real_offsets(*args, **kw)
        return offs[:2] + offs[3:] if len(offs) > 3 else offs

    monkeypatch.setattr(mr, "window_offsets", holey)
    monkeypatch.setattr(mr, "read_tile_window", lambda *a_, **k_: CtxWindow(
        data=np.full((256, 256), 200, dtype=np.uint8), row_off=0, col_off=0,
        transform=(a, b, c, d, e, f), crs_wkt=_WKT))
    with pytest.raises(SystemExit, match="uncomputable"):
        mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                        args=_fake_args(tmp_path, win_px=256))


def test_the_grid_reads_the_sphere_radius_out_of_the_tile_rather_than_assuming_it():
    """`COARSE_GRID_ID` asserts `R3396190`. If `tile_global_grid` filled that in from the
    module constant instead of the tile's own CRS, the id would be an assertion nothing
    measures — the failure class caught four times on this project."""
    tf = [MURRAY_NATIVE_M, 0.0, _G_COL * MURRAY_NATIVE_M,
          0.0, -MURRAY_NATIVE_M, -_G_ROW * MURRAY_NATIVE_M]
    # a radius inside the tolerance but not equal to the constant: it must be REPORTED,
    # which is only possible if it was parsed
    off_by_half = _WKT.replace("3396190,", "3396190.5,")
    assert tile_global_grid(tf, off_by_half, TILE_PX).radius_m == 3396190.5
    # and a genuinely different sphere must be refused outright
    with pytest.raises(ValueError, match="does not describe this product"):
        tile_global_grid(tf, _WKT.replace("3396190,", "3389500,"), TILE_PX)
    with pytest.raises(ValueError, match="no CRS WKT"):
        tile_global_grid(tf, "", TILE_PX)


def test_a_partial_from_another_lattice_stops_the_run(tmp_path):
    """A resumed Sherlock run is the realistic way the old lattice comes back: same
    filenames, different `(ti, tj)` meaning, no visible error."""
    import scripts.map_region as mr

    pdir = tmp_path / "partials" / "E4_N40"
    pdir.mkdir(parents=True)
    np.savez_compressed(pdir / "000000_000000.npz",           # pre-R01: no grid_id at all
                        ti=np.arange(3, dtype=np.int32), tj=np.arange(3, dtype=np.int32),
                        prob=np.zeros(3, dtype=np.float32))
    assert mr.partial_grid_id(pdir / "000000_000000.npz") is None

    with pytest.raises(SystemExit, match="different coarse lattice"):
        mr.reject_foreign_partials(pdir, _fake_args(tmp_path))

    # --force discards them rather than silently mixing
    mr.reject_foreign_partials(pdir, _fake_args(tmp_path, force=True))
    assert list(pdir.glob("*.npz")) == []


def test_an_existing_old_lattice_product_is_not_skipped_as_done(tmp_path, monkeypatch):
    """Existence is not completeness once the lattice has moved.

    Every tile of the pre-R01 product is on disk. A bare `prob_tif.exists()` skip would
    therefore skip all 26, write a manifest stamped with the NEW grid_id, and print
    "26/26 tiles complete" — a rebuild that rendered nothing and then certified itself.
    """
    pytest.importorskip("rasterio")
    import scripts.map_region as mr
    from src.mapping import coarsened_transform, write_geotiff

    tile, transform, extent = _synthetic_tile(tmp_path)
    # the pre-R01 affine: parent-tile anchored, so it carries this tile's sub-cell phase
    old = coarsened_transform(transform, 1, 1, TILE_PX)
    write_geotiff(tmp_path / f"{tile}_prob.tif", np.zeros((4, 4)), old, _WKT)
    assert mr.existing_product_off_lattice(tmp_path / f"{tile}_prob.tif")

    with pytest.raises(SystemExit, match="is NOT on"):
        mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                        args=_fake_args(tmp_path, win_px=256))

    # a product genuinely on the grid, with a sidecar that says so, still skips
    good = global_cell_transform(-9999, -19999, TILE_PX)
    write_geotiff(tmp_path / f"{tile}_prob.tif", np.zeros((4, 4)), good, _WKT)
    (tmp_path / f"{tile}.json").write_text(json.dumps({"grid_id": COARSE_GRID_ID}),
                                           encoding="utf-8")
    assert mr.existing_product_off_lattice(tmp_path / f"{tile}_prob.tif") is None
    assert mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                           args=_fake_args(tmp_path, win_px=256))["status"] == "skipped_done"


def test_an_on_lattice_raster_without_a_sidecar_is_still_refused(tmp_path):
    """Absence of provenance must not read as "checked and clean" — the failure class this
    project has now been bitten by five times."""
    pytest.importorskip("rasterio")
    import scripts.map_region as mr
    from src.mapping import write_geotiff

    p = tmp_path / "E4_N40_prob.tif"
    write_geotiff(p, np.zeros((4, 4)), global_cell_transform(-9999, -19999, TILE_PX), _WKT)
    assert "no sidecar" in mr.existing_product_off_lattice(p)


def test_a_corrupt_partial_cannot_escape_the_gate_or_block_force(tmp_path):
    """`np.savez_compressed` writes the zip in place, so a killed job leaves a truncated
    `.npz`. Letting BadZipFile escape would make even --force unable to clear it."""
    import scripts.map_region as mr

    pdir = tmp_path / "partials" / "E4_N40"
    pdir.mkdir(parents=True)
    (pdir / "000000_000000.npz").write_bytes(b"PK\x03\x04 truncated")
    (pdir / "000000_004000.npz").write_bytes(b"")
    assert mr.partial_grid_id(pdir / "000000_000000.npz") is None
    assert mr.partial_grid_id(pdir / "000000_004000.npz") is None

    with pytest.raises(SystemExit, match="different coarse lattice"):
        mr.reject_foreign_partials(pdir, _fake_args(tmp_path))
    mr.reject_foreign_partials(pdir, _fake_args(tmp_path, force=True))
    assert list(pdir.glob("*.npz")) == []


def test_a_current_partial_is_kept_on_resume(tmp_path):
    import scripts.map_region as mr

    pdir = tmp_path / "partials" / "E4_N40"
    pdir.mkdir(parents=True)
    np.savez_compressed(pdir / "000000_000000.npz", ti=np.arange(3, dtype=np.int32),
                        tj=np.arange(3, dtype=np.int32), prob=np.zeros(3, dtype=np.float32),
                        grid_id=np.array(COARSE_GRID_ID))
    mr.reject_foreign_partials(pdir, _fake_args(tmp_path))     # must not raise
    assert len(list(pdir.glob("*.npz"))) == 1


class _OneFrame:
    """Minimal stand-in for `load_frames`: one dissolved source frame covering everything."""
    def __init__(self):
        from shapely.geometry import box
        self.geometry = [box(-1e9, -1e9, 1e9, 1e9)]

    def __len__(self):
        return 1


def test_a1_renders_on_the_same_lattice_as_the_baseline(tmp_path, monkeypatch):
    """The A1 row must land cell-for-cell on the baseline's lattice.

    Source inspection is not enough here: the two drivers can both *mention* the grid and
    still diverge in the one line that builds the affine, or in one of the two axes of the
    sweep. This runs A1's real `process_tile` with inference, CTX I/O and the SeamMap stubbed
    and compares its affine against the baseline's, which is the property the whole
    same-commit mandate exists to protect.
    """
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("shapely")
    import scripts.map_region as mr
    import scripts.striping_a1_map as a1
    from src.mapping import CtxWindow

    tile, transform, extent = _synthetic_tile(tmp_path)
    a, b, c, d, e, f = transform
    side = json.loads((tmp_path / "ctx" / f"{tile}.json").read_text(encoding="utf-8"))

    def fake_read(zip_path, inner_tif, row_off, col_off, size):
        h, w = min(size, extent - row_off), min(size, extent - col_off)
        return CtxWindow(data=np.full((h, w), 200, dtype=np.uint8),
                         row_off=row_off, col_off=col_off,
                         transform=(a, b, c + col_off * a, d, e, f + row_off * e),
                         crs_wkt=_WKT)

    monkeypatch.setattr(a1, "CTX_ZIP_DIR", tmp_path / "ctx")
    monkeypatch.setattr(a1, "load_tile_sidecar", lambda t, *a_, **k_: side)
    monkeypatch.setattr(a1, "_inner_tif_name", lambda z: "inner.tif")
    monkeypatch.setattr(a1, "frame_stats_160", lambda t: ({0: (100.0, 20.0)}, 1))
    monkeypatch.setattr(a1, "load_frames", lambda t: _OneFrame())
    monkeypatch.setattr(a1, "read_tile_window", fake_read)
    monkeypatch.setattr(mr, "read_tile_window", fake_read)

    base_dir, a1_dir = tmp_path / "base", tmp_path / "a1"
    base_dir.mkdir()
    a1_dir.mkdir()
    assert mr.map_one_tile(tile, _StubEmbedder(), _StubHead(), None,
                           args=_fake_args(tmp_path, out_dir=str(base_dir),
                                           win_px=256))["status"] == "done"
    a1_args = _fake_args(tmp_path, out_dir=str(a1_dir), win_px=256,
                         head=str(tmp_path / "nohead"), calibration=None)
    assert a1.process_tile(tile, _StubEmbedder(), _StubHead(), None,
                           a1_args)["status"] == "done"

    with rasterio.open(base_dir / f"{tile}_prob.tif") as b_ds, \
            rasterio.open(a1_dir / f"{tile}_prob.tif") as a_ds:
        assert a_ds.transform == b_ds.transform, "A1 landed on a different lattice"
        assert (a_ds.height, a_ds.width) == (b_ds.height, b_ds.width)
        assert_shared_lattice([a_ds.transform, b_ds.transform], tile_px=TILE_PX)
    b_side = json.loads((base_dir / f"{tile}.json").read_text(encoding="utf-8"))
    a_side = json.loads((a1_dir / f"{tile}.json").read_text(encoding="utf-8"))
    for k in ("grid_id", "grid_cell_m", "grid_tile_px", "cell_row0", "cell_col0",
              "ti_min", "tj_min"):
        assert a_side[k] == b_side[k], k


def test_a1_refuses_a_baseline_reference_on_the_old_lattice(tmp_path, monkeypatch):
    """The ordering constraint as a gate, not a sentence: A1 reads its per-frame
    normalisation off the baseline product's grid, so an old-lattice baseline must abort."""
    rasterio = pytest.importorskip("rasterio")
    import scripts.striping_a1_map as a1
    from src.mapping import write_geotiff

    # a per-tile raster with the shipped kind of sub-cell phase (0.875 cell)
    cell = TILE_PX * MURRAY_NATIVE_M
    from rasterio.transform import Affine
    bad = Affine(cell, 0.0, 17.5 * cell + 0.875 * cell, 0.0, -cell, -9.0 * cell)
    write_geotiff(tmp_path / "E4_N40_abundance.tif",
                  np.zeros((4, 4), dtype=np.float64), bad, _WKT)
    monkeypatch.setattr(a1, "MAP_DIR", tmp_path)

    def _must_not_run(*a_, **k_):
        raise AssertionError("the lattice gate did not fire: A1 went on to read the CTX")

    monkeypatch.setattr(a1, "read_ctx_on_grid", _must_not_run)
    with pytest.raises(SystemExit, match="must be re-rendered"):
        a1.frame_stats_160("E4_N40")
    assert rasterio  # (import guard only)


def test_both_map_drivers_share_one_grid_and_one_sweep():
    """A1 and the baseline must move together — separately is the failure the 2026-08-06
    product decision forbids, and it would be invisible until the two rasters are compared."""
    import inspect

    import scripts.map_region as mr
    import scripts.striping_a1_map as a1

    for mod, fn in ((mr, mr.map_one_tile), (a1, a1.process_tile)):
        src = inspect.getsource(fn)
        assert "tile_global_grid(" in src, f"{mod.__name__} never builds the global grid"
        assert "global_grid=grid_geom.as_tuple" in src, f"{mod.__name__} never passes it"
        assert "3 * TILE_PX" in src, f"{mod.__name__} kept the 2-cell overlap"
        assert "tile_aligned=False" in src, f"{mod.__name__} kept tile-aligned offsets"
        assert "uncovered_cells(" in src, f"{mod.__name__} has no coverage guard"
    for fn in (mr.write_tile_geotiffs, a1.write_tile):
        src = inspect.getsource(fn)
        assert "grid_geom.transform(" in src
        assert "coarsened_transform" not in src, "parent-tile affine + global indices = ~2,600 km"


# ---------------------------------------------------------------- acceptance gate

def test_the_real_themis_offset_is_caught_by_the_coregistration_guard():
    """The exact post-rebuild THEMIS trap: same shape, origin off by +100 m E / -80 m S.

    Equal shapes are what makes this silent — notebook 24 leg 1 indexes both arrays with the
    same boolean mask, so nothing raises and the Spearman is computed on cells 0.625 of a
    cell apart. Uses the measured shipped and corrected mosaic origins, not invented numbers.
    """
    from rasterio.transform import Affine
    cell = TILE_PX * MURRAY_NATIVE_M
    shipped = Affine(cell, 0.0, -711136.3711, 0.0, -cell, 2845025.4819)
    corrected = Affine(cell, 0.0, -711036.3716, 0.0, -cell, 2844945.4823)
    shape = (5925, 11852)
    assert_coregistered(corrected, corrected, shape_a=shape, shape_b=shape)   # must not raise
    with pytest.raises(ValueError, match=r"offset \(\-100"):
        assert_coregistered(corrected, shipped, shape_a=shape, shape_b=shape,
                            name_a="abundance mosaic", name_b="THEMIS night-IR")


def _products_claiming_the_grid():
    """Every rendered raster whose sidecar claims `COARSE_GRID_ID`.

    Gating on the *claim* rather than on mere existence is what keeps this a gate instead of
    a permanent red mark: the pre-R01 product on disk claims nothing (its sidecars predate
    the field), so it is skipped and its pending re-render is tracked in
    docs/PENDING_REBUILD.md. The moment a product asserts the grid, this test measures it.

    Driven from the **sidecar**, not from a raster glob. Keying on `*_abundance.tif` looked
    equivalent and was not: `striping_a1_map.write_tile` only writes an abundance raster when
    a calibrator is supplied, and `--calibration` defaults to `None` (the A1 row is scored on
    raw `P(rich)`), so the shipped A1 configuration emits `_prob.tif` + `_prob_raw.tif` and a
    sidecar that *does* claim the grid. The gate would have inspected zero A1 files while its
    own docstring promised it spanned both rows.
    """
    out = []
    for d in (MAP_DIR, REPO / "reports" / "map_a1"):
        if not d.exists():
            continue
        for side in sorted(d.glob("*.json")):
            try:
                claim = json.loads(side.read_text(encoding="utf-8")).get("grid_id")
            except (ValueError, OSError):
                continue
            if claim != COARSE_GRID_ID:
                continue
            out += [t for t in (d / f"{side.stem}_prob.tif", d / f"{side.stem}_abundance.tif",
                                d / f"{side.stem}_prob_raw.tif") if t.exists()]
    return out


@pytest.mark.skipif(not _products_claiming_the_grid(),
                    reason="no rendered product claims COARSE_GRID_ID yet (rebuild pending)")
def test_every_product_that_claims_the_grid_is_actually_on_it():
    """R01's acceptance gate on the rebuilt map, and the standing anti-drift check.

    A sidecar that says `grid_id` while the raster sits elsewhere is precisely the
    "provenance that asserts instead of measures" failure this project has now been bitten
    by four times. Pure geometry, ~10 ms — cheaper than one window of inference, so it can
    gate the rebuild before any GPU time is spent. Spans the baseline and the A1 rows
    together, which is what makes them comparable.
    """
    rasterio = pytest.importorskip("rasterio")
    transforms = []
    for p in _products_claiming_the_grid():
        with rasterio.open(p) as ds:
            transforms.append(ds.transform)
    assert_shared_lattice(transforms, tile_px=TILE_PX)
