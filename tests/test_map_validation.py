"""Unit tests for `src.map_validation` -- the module all five PLAN_MapValidation notebooks call.

Two classes of behaviour are pinned here, and they matter for different reasons:

* **The refusals.** Being handed a single-arm mosaic instead of the union is a *silent*
  50%-coverage bug -- everything loads, everything computes, and the answer is about half the
  footprint. `load_union` must refuse it. Same for a cross-lattice mosaic (R01).
* **The arithmetic.** `zonal_cells` and `radial_annuli` are windowed for speed over a
  169-million-cell mosaic, and a windowing bug is invisible in a plot. So both are checked
  against hand-computable geometry, and `zonal_cells` is checked to return the *same* cells
  for every target so the ruling-3 triple stays comparable.

Also pinned: `cluster_bootstrap_ci` takes its n from the group count, not the cell count --
which is the whole point of ruling 5.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import map_validation as mv  # noqa: E402
from src.mapping import COARSE_GRID_ID, write_geotiff  # noqa: E402

LAYERS = ("abundance", "prob", "prob_raw")


def _proj_crs():
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


def _fake_union(d: Path, crs_wkt: str, *, n: int = 10, n_tiles: int = 2,
                union_tags: bool = True, grid_id: str = COARSE_GRID_ID,
                values: dict | None = None) -> Path:
    """A tiny stand-in for `reports/map_union`: three mosaics with the union's own tags."""
    d.mkdir(parents=True, exist_ok=True)
    tr = Affine(160.0, 0.0, 0.0, 0.0, -160.0, 0.0)
    for k, layer in enumerate(LAYERS):
        if values and layer in values:
            a = np.asarray(values[layer], dtype=np.float32)
        else:
            a = (np.arange(n * n, dtype=np.float32).reshape(n, n) / (n * n) + k * 0.01)
        tags = {"SIZE_FLOOR_BASIS_VERSION": "v2_mixed_floor_2",
                "MOSAIC_ARM": "union", "MOSAIC_LAYER": layer,
                "MOSAIC_GRID_ID": grid_id}
        if union_tags:
            tiles = [f"E-{4 * (i + 1)}_N32" for i in range(n_tiles)]
            tags.update(UNION_N_TILES=str(n_tiles), UNION_TILES=",".join(tiles),
                        UNION_TILE_ORIGIN='{"E-4_N32": "map_region"}',
                        UNION_ADOPTED_TILES="E-4_N32")
        write_geotiff(d / f"regional_{layer}_mosaic.tif", a, tr, crs_wkt, tags=tags)
    return d


# ------------------------------------------------------------------------ load_union
def test_load_union_reads_the_union_and_parses_its_provenance(tmp_path, crs_wkt):
    d = _fake_union(tmp_path / "map_union", crs_wkt, n_tiles=3)
    arr, transform, wkt, meta = mv.load_union("abundance", union_dir=d)
    assert arr.shape == (10, 10)
    assert meta["source"] == "prebuilt"
    assert meta["n_union_tiles"] == 3
    assert meta["union_tiles"] == ["E-4_N32", "E-8_N32", "E-12_N32"]
    assert meta["adopted_tiles"] == ["E-4_N32"]
    assert meta["size_floor"]["SIZE_FLOOR_BASIS_VERSION"] == "v2_mixed_floor_2"


def test_load_union_REFUSES_a_single_arm_mosaic(tmp_path, crs_wkt):
    """The silent 50%-coverage bug: an arm mosaic loads fine and answers the wrong question."""
    d = _fake_union(tmp_path / "map_region", crs_wkt, union_tags=False)
    with pytest.raises(ValueError, match="not a union mosaic"):
        mv.load_union("abundance", union_dir=d)
    # ...but it can be read deliberately, for a like-for-like arm comparison
    arr, _, _, meta = mv.load_union("abundance", union_dir=d, require_union_tags=False)
    assert arr.shape == (10, 10)
    assert meta["n_union_tiles"] == 0


def test_load_union_refuses_a_mosaic_on_another_lattice(tmp_path, crs_wkt):
    d = _fake_union(tmp_path / "map_union", crs_wkt, grid_id="some_other_lattice")
    with pytest.raises(ValueError, match="lattice"):
        mv.load_union("abundance", union_dir=d)


def test_load_union_points_at_the_producer_when_the_mosaic_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/map_union.py"):
        mv.load_union("abundance", union_dir=tmp_path / "nothing_here")


def test_load_union_never_builds(tmp_path, crs_wkt):
    """A consumer that re-merges and writes replaces a tagged product with a look-alike."""
    d = tmp_path / "map_union"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        mv.load_union("abundance", union_dir=d)
    assert list(d.iterdir()) == []


def test_union_tiles_comes_from_the_tags_not_a_hardcoded_list(tmp_path, crs_wkt):
    d = _fake_union(tmp_path / "map_union", crs_wkt, n_tiles=5)
    assert mv.union_tiles(union_dir=d) == [f"E-{4 * (i + 1)}_N32" for i in range(5)]


# ---------------------------------------------------------------------- three_targets
def test_three_targets_share_one_finite_mask(tmp_path, crs_wkt):
    """A cell missing in ANY layer is missing in all three, so contrasts are comparable."""
    n = 6
    ab = np.arange(n * n, dtype=np.float32).reshape(n, n) / (n * n)
    praw = ab.copy()
    prob = ab.copy()
    ab[0, 0] = np.nan
    praw[1, 1] = np.nan
    prob[2, 2] = np.nan
    d = _fake_union(tmp_path / "map_union", crs_wkt, n=n,
                    values={"abundance": ab, "prob": prob, "prob_raw": praw})

    t = mv.three_targets(union_dir=d)
    assert t.finite.sum() == n * n - 3
    for name, arr in t.as_dict().items():
        assert np.isfinite(arr).sum() == n * n - 3, name
        assert not np.isfinite(arr[0, 0]) and not np.isfinite(arr[1, 1]), name
        assert not np.isfinite(arr[2, 2]), name


def test_three_targets_rich_is_prob_ge_half(tmp_path, crs_wkt):
    n = 4
    prob = np.linspace(0.0, 1.0, n * n, dtype=np.float32).reshape(n, n)
    d = _fake_union(tmp_path / "map_union", crs_wkt, n=n, values={"prob": prob})
    t = mv.three_targets(union_dir=d)
    assert np.array_equal(t.rich, prob >= 0.5)
    rich = t.as_dict()["rich"]
    assert np.nanmean(rich) == pytest.approx(float((prob >= 0.5).mean()))
    assert mv.RICH_PROB == 0.5                      # ruling 4 / notebook 24's convention


def test_three_targets_rich_threshold_is_overridable_but_recorded(tmp_path, crs_wkt):
    n = 4
    prob = np.linspace(0.0, 1.0, n * n, dtype=np.float32).reshape(n, n)
    d = _fake_union(tmp_path / "map_union", crs_wkt, n=n, values={"prob": prob})
    t = mv.three_targets(union_dir=d, rich_prob=0.8)
    assert t.meta["rich_prob"] == 0.8
    assert np.array_equal(t.rich, prob >= 0.8)


def test_target_names_are_the_ruling_3_triple():
    assert mv.TARGET_NAMES == ("abundance", "prob_raw", "rich")


# ------------------------------------------------------------------------ zonal_cells
def _grid(n=10, cell=160.0):
    """`n x n` array whose value equals its flat index, on a `cell`-metre north-up grid."""
    arr = np.arange(n * n, dtype=np.float64).reshape(n, n)
    return arr, Affine(cell, 0.0, 0.0, 0.0, -cell, 0.0)


def test_zonal_cells_returns_exactly_the_covered_cells(crs_wkt=None):
    arr, tr = _grid(10)
    # cells (row, col) with centre inside x in (0, 320), y in (-320, 0) -> rows 0-1, cols 0-1
    vals = mv.zonal_cells(box(0.0, -320.0, 320.0, 0.0), arr, tr)
    assert sorted(vals.tolist()) == [0.0, 1.0, 10.0, 11.0]


def test_zonal_cells_drops_nan_cells_but_keeps_the_rest():
    arr, tr = _grid(10)
    arr[0, 0] = np.nan
    vals = mv.zonal_cells(box(0.0, -320.0, 320.0, 0.0), arr, tr)
    assert sorted(vals.tolist()) == [1.0, 10.0, 11.0]
    raw = mv.zonal_cells(box(0.0, -320.0, 320.0, 0.0), arr, tr, finite_only=False)
    assert raw.size == 4 and np.isnan(raw).sum() == 1


def test_zonal_cells_returns_the_SAME_cells_for_every_target():
    """One mask, three distributions -- otherwise a per-polygon contrast is not comparable."""
    arr, tr = _grid(10)
    arrays = {"abundance": arr, "prob_raw": arr * 2.0, "rich": (arr > 50).astype(float)}
    out = mv.zonal_cells(box(0.0, -320.0, 320.0, 0.0), arrays, tr)
    assert set(out) == set(arrays)
    assert {v.size for v in out.values()} == {4}
    assert sorted(out["prob_raw"].tolist()) == [0.0, 2.0, 20.0, 22.0]


def test_zonal_cells_is_empty_for_a_geometry_off_the_raster():
    arr, tr = _grid(10)
    vals = mv.zonal_cells(box(1e7, 1e7, 1.1e7, 1.1e7), arr, tr)
    assert vals.size == 0
    out = mv.zonal_cells(box(1e7, 1e7, 1.1e7, 1.1e7), {"a": arr}, tr)
    assert out["a"].size == 0


def test_zonal_cells_windowing_matches_a_whole_array_mask():
    """The windowing is for speed; a windowing bug would be invisible in a plot."""
    from rasterio.features import geometry_mask

    rng = np.random.default_rng(0)
    arr = rng.random((40, 40))
    tr = Affine(160.0, 0.0, -1000.0, 0.0, -160.0, 500.0)
    geom = box(-500.0, -2000.0, 2500.0, 100.0)
    inside = ~geometry_mask([geom], out_shape=arr.shape, transform=tr, all_touched=False)
    expect = np.sort(arr[inside].ravel())
    got = np.sort(mv.zonal_cells(geom, arr, tr))
    assert got.size == expect.size
    assert np.allclose(got, expect)


# ----------------------------------------------------------------------- radial_annuli
def test_radial_annuli_bins_by_crater_radii_not_metres():
    arr, tr = _grid(21)          # 160 m cells, so a 480 m radius is 3 cells
    arr[:] = 1.0
    cx, cy = 10.5 * 160.0, -10.5 * 160.0        # centre of cell (10, 10)
    out = mv.radial_annuli(cx, cy, 480.0, arr, tr, edges_R=(0.0, 1.0, 2.0))
    assert len(out) == 2
    # inner annulus r < 1R = 480 m, i.e. cell-centre offsets with dx**2 + dy**2 < 9 cells:
    # 1 + 4 + 4 + 4 + 8 + 4 = 25 cells (the distance-3 ring is excluded, r < R is strict)
    assert out[0].size == 25
    assert out[1].size > out[0].size             # the 1-2R annulus has more area


def test_radial_annuli_edges_are_half_open_so_no_cell_is_double_counted():
    arr, tr = _grid(31)
    arr[:] = 1.0
    cx, cy = 15.5 * 160.0, -15.5 * 160.0
    edges = (0.0, 1.0, 1.5, 2.0, 3.0)
    out = mv.radial_annuli(cx, cy, 800.0, arr, tr, edges_R=edges)
    total = sum(a.size for a in out)
    disc = mv.radial_annuli(cx, cy, 800.0, arr, tr, edges_R=(0.0, 3.0))[0]
    assert total == disc.size


def test_radial_annuli_returns_the_same_annuli_for_every_target():
    arr, tr = _grid(21)
    arr[:] = 1.0
    arrays = {"abundance": arr, "prob_raw": arr * 3.0}
    cx, cy = 10.5 * 160.0, -10.5 * 160.0
    out = mv.radial_annuli(cx, cy, 480.0, arrays, tr, edges_R=(0.0, 1.0, 2.0))
    assert set(out) == {"abundance", "prob_raw"}
    assert [a.size for a in out["abundance"]] == [a.size for a in out["prob_raw"]]
    assert np.allclose(out["prob_raw"][0], 3.0)


def test_radial_annuli_rejects_bad_edges_and_radius():
    arr, tr = _grid(10)
    with pytest.raises(ValueError, match="increasing"):
        mv.radial_annuli(0.0, 0.0, 480.0, arr, tr, edges_R=(0.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        mv.radial_annuli(0.0, 0.0, 0.0, arr, tr)


def test_radial_annuli_is_empty_off_the_raster():
    arr, tr = _grid(10)
    out = mv.radial_annuli(1e7, 1e7, 480.0, arr, tr, edges_R=(0.0, 1.0, 2.0))
    assert [a.size for a in out] == [0, 0]


# ------------------------------------------------------- effective n / inference (ruling 5)
def test_cluster_bootstrap_ci_takes_its_n_from_the_GROUPS():
    rng = np.random.default_rng(0)
    groups = [rng.normal(loc=m, scale=0.1, size=500) for m in (0.0, 0.5, 1.0, 1.5)]
    out = mv.cluster_bootstrap_ci(groups, n_boot=300, seed=1)
    assert out["n_groups"] == 4
    assert out["n_cells"] == 2000
    assert out["lo"] < out["point"] < out["hi"]


def test_cluster_bootstrap_ci_is_WIDER_with_fewer_groups_at_equal_cell_count():
    """The behaviour ruling 5 demands: 57 million correlated cells are not 57 million samples."""
    rng = np.random.default_rng(2)
    means = rng.normal(size=32)
    many = [rng.normal(loc=m, scale=0.1, size=60) for m in means]
    few = [np.concatenate(many[i::4]) for i in range(4)]      # same cells, 4 groups
    assert sum(g.size for g in many) == sum(g.size for g in few)
    w_many = mv.cluster_bootstrap_ci(many, n_boot=400, seed=3)
    w_few = mv.cluster_bootstrap_ci(few, n_boot=400, seed=3)
    assert (w_few["hi"] - w_few["lo"]) > (w_many["hi"] - w_many["lo"])


def test_cluster_bootstrap_ci_refuses_to_imply_certainty_from_one_group():
    out = mv.cluster_bootstrap_ci([np.arange(10_000.0)], n_boot=50)
    assert out["n_groups"] == 1
    assert np.isnan(out["lo"]) and np.isnan(out["hi"])
    assert "undefined" in out["note"]


def test_cluster_bootstrap_ci_handles_no_data():
    out = mv.cluster_bootstrap_ci([np.array([np.nan]), np.array([])], n_boot=10)
    assert out["n_groups"] == 0 and np.isnan(out["point"])


def test_cluster_bootstrap_ci_is_deterministic_for_a_seed():
    rng = np.random.default_rng(0)
    groups = [rng.normal(size=50) for _ in range(6)]
    a = mv.cluster_bootstrap_ci(groups, n_boot=200, seed=7)
    b = mv.cluster_bootstrap_ci(groups, n_boot=200, seed=7)
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])


def test_variance_decomposition_separates_between_from_within():
    a = np.full(100, 0.0)
    b = np.full(100, 1.0)
    out = mv.variance_decomposition([a, b])
    assert out["eta2"] == pytest.approx(1.0)         # no within-group variance at all
    rng = np.random.default_rng(0)
    noisy = [rng.normal(loc=0.0, size=500), rng.normal(loc=0.0, size=500)]
    assert mv.variance_decomposition(noisy)["eta2"] < 0.05


def test_variance_decomposition_is_undefined_for_one_group():
    out = mv.variance_decomposition([np.arange(10.0)])
    assert np.isnan(out["eta2"]) and "undefined" in out["note"]


def test_frame_effective_n_rejects_a_bad_on_missing():
    with pytest.raises(ValueError, match="on_missing"):
        mv.frame_effective_n(tiles=[], on_missing="ignore")


def test_frame_effective_n_dedups_frames_across_tiles(monkeypatch):
    """A CTX frame straddles Murray tiles, so summing per-tile counts overcounts."""
    import pandas as pd

    from src import striping

    fake = {"E-4_N32": ["P1", "P2", "P3"], "E-8_N32": ["P3", "P4"]}
    monkeypatch.setattr(striping, "load_frames",
                        lambda tile, dissolve=True: pd.DataFrame(
                            {"PRODUCT_ID": fake[tile]}))
    out = mv.frame_effective_n(tiles=list(fake))
    assert out["n_frames"] == 4                  # P1..P4, not 5
    assert out["sum_per_tile"] == 5
    assert out["per_tile"] == {"E-4_N32": 3, "E-8_N32": 2}
    assert out["failed"] == {}
    assert "overcounts" in out["note"]


def test_frame_effective_n_can_skip_a_failing_tile_and_says_it_is_a_lower_bound(monkeypatch):
    import pandas as pd

    from src import striping

    def loader(tile, dissolve=True):
        if tile == "E-8_N32":
            raise OSError("SeamMap unreachable")
        return pd.DataFrame({"PRODUCT_ID": ["P1", "P2"]})

    monkeypatch.setattr(striping, "load_frames", loader)
    with pytest.raises(OSError):
        mv.frame_effective_n(tiles=["E-4_N32", "E-8_N32"])
    out = mv.frame_effective_n(tiles=["E-4_N32", "E-8_N32"], on_missing="skip")
    assert out["n_frames"] == 2
    assert "E-8_N32" in out["failed"]
    assert "LOWER BOUND" in out["note"]


# ---------------------------------------------------------------------------- the caveat
def test_caveat_names_all_four_standing_caveats_and_the_uncorrected_artifact():
    """One string, quoted by all five notebooks, so the caveats cannot drift or soften."""
    c = mv.CAVEAT_MD
    assert "UNCORRECTED" in c
    assert "UPPER BOUND" in c
    assert "size-floor-referenced" in c
    assert "extrapolation" in c
    assert "resample, never index-match" in c
    assert "presence AUC" in c


# --------------------------------------------------- notebook 30: geology (SIM3292)
def test_stratigraphic_rank_parses_every_unit_in_the_union():
    """The 14 real Tanaka codes over the 122-tile union. An unparsed code would silently
    move a unit along the very age axis the abundance-vs-age conclusion rests on."""
    expect = {                       # unit: (rank, spans, terrain)
        "eNh": (1.0, False, "h"), "mNh": (2.0, False, "h"), "Nhu": (2.0, False, "hu"),
        "lNh": (3.0, False, "h"), "HNt": (3.5, True, "t"), "ANa": (3.5, True, "a"),
        "eHh": (4.0, False, "h"), "eHt": (4.0, False, "t"), "Hto": (4.5, False, "to"),
        "lHl": (5.0, False, "l"), "lHt": (5.0, False, "t"),
        "AHi": (5.5, True, "i"), "AHv": (5.5, True, "v"), "mAl": (7.0, False, "l"),
    }
    for unit, (rank, spans, terrain) in expect.items():
        got = mv.stratigraphic_rank(unit)
        assert got["rank"] == rank, (unit, got)
        assert got["spans"] is spans, (unit, got)
        assert got["terrain"] == terrain, (unit, got)


def test_stratigraphic_rank_orders_noachian_before_hesperian_before_amazonian():
    order = ["eNh", "mNh", "lNh", "eHt", "lHt", "mAl"]
    ranks = [mv.stratigraphic_rank(u)["rank"] for u in order]
    assert ranks == sorted(ranks)
    assert ranks[0] < ranks[-1]                  # older -> younger, increasing


def test_stratigraphic_rank_reports_an_unknown_code_rather_than_guessing():
    got = mv.stratigraphic_rank("Zzz")
    assert np.isnan(got["rank"]) and got["epoch"] is None and got["label"] == "unparsed"


def test_stratigraphic_rank_flags_two_epoch_units():
    """An AHi unit is not '5.5 old' -- it is undated within a ~3 Gyr window."""
    spanning = [u for u in ("AHi", "AHv", "ANa", "HNt")]
    assert all(mv.stratigraphic_rank(u)["spans"] for u in spanning)
    assert not any(mv.stratigraphic_rank(u)["spans"] for u in ("mNh", "lHl", "Hto", "mAl"))


def test_bounds_lonlat_is_exact_on_the_clon0_sphere():
    deg = np.pi * 3396190.0 / 180.0
    got = mv.bounds_lonlat((-56 * deg, 16 * deg, 20 * deg, 48 * deg))
    assert got == pytest.approx((-56.0, 16.0, 20.0, 48.0))


def test_min_cells_unit_is_the_ruled_floor():
    """50,000 cells = 1,280 km2 at 160 m (ruled 2026-08-29 with the distributions in hand)."""
    assert mv.MIN_CELLS_UNIT == 50_000
    assert mv.MIN_CELLS_UNIT * (mv.PX_M / 1000.0) ** 2 == pytest.approx(1280.0)


@pytest.mark.slow
def test_load_geology_closes_the_partition_and_produces_no_nonfinite_geometry():
    """SIM3292 is a complete partition of Mars, so the clip must tile the window exactly.

    This is the gate on the Robinson recipe: a shortfall means the intermediate cut ate area
    that belongs inside the window.
    """
    import rasterio
    from rasterio.transform import array_bounds

    union = REPO / "reports" / "map_union" / "regional_abundance_mosaic.tif"
    if not (union.exists() and Path(mv.SIM3292_ZIP).exists()):
        pytest.skip("union mosaic or SIM3292 download not present in this checkout")
    with rasterio.open(union) as ds:
        bounds = array_bounds(ds.height, ds.width, ds.transform)
        wkt = ds.crs.to_wkt()

    geo, rep = mv.load_geology(bounds, wkt)
    assert rep["source_polygons"] == 1311 and rep["source_units"] == 44
    assert rep["source_crs"].startswith("Robinson")
    assert rep["source_invalid"] == 0 and rep["source_nonfinite"] == 0
    assert rep["nonfinite_after_source_clip"] == 0
    assert rep["nonfinite_after_reprojection"] == 0      # the whole point of the recipe
    assert rep["invalid_after_repair"] == 0
    assert rep["partition_closure"] == pytest.approx(1.0, abs=1e-9)
    assert len(geo) == rep["polygons"] and rep["units"] == geo["Unit"].nunique()
    assert {"Unit", "UnitDesc", "SphArea_km", "area_km2", "geometry"} <= set(geo.columns)
    assert all(np.isfinite(mv.stratigraphic_rank(u)["rank"]) for u in geo["Unit"].unique())


@pytest.mark.slow
def test_the_naive_reprojection_really_does_produce_nonfinite_geometry():
    """Pins the trap `load_geology` exists for, so a library upgrade that changes it is noticed.

    All 1311 SIM3292 polygons are valid in Robinson, but the INVERSE Robinson overflows for 62
    of them -- and `.intersects()` on a non-finite geometry returns garbage behind only a
    RuntimeWarning, so the naive path yields a plausible, wrong polygon set.
    """
    import geopandas as gpd
    import shapely

    if not Path(mv.SIM3292_ZIP).exists():
        pytest.skip("SIM3292 download not present in this checkout")
    src = gpd.read_file(mv.SIM3292_GDB, layer=mv.SIM3292_LAYER, engine="pyogrio")
    assert int((~src.geometry.is_valid).sum()) == 0          # valid at source

    geog = src.to_crs(src.crs.geodetic_crs)
    nonfinite = sum(not np.isfinite(np.asarray(shapely.get_coordinates(g))).all()
                    for g in geog.geometry)
    assert nonfinite > 0, ("the inverse Robinson no longer overflows -- re-measure the "
                          "load_geology recipe before trusting this docstring")


# ------------------------------------- moment-based helpers (exact over 265.8 M cells)
def test_cluster_bootstrap_ratio_ci_is_exact_for_a_mean():
    """The ratio bootstrap must give the same POINT estimate as pooling the cells."""
    rng = np.random.default_rng(0)
    groups = [rng.random(rng.integers(50, 500)) for _ in range(12)]
    counts = [g.size for g in groups]
    sums = [g.sum() for g in groups]
    got = mv.cluster_bootstrap_ratio_ci(counts, sums, n_boot=200, seed=1)
    assert got["point"] == pytest.approx(np.concatenate(groups).mean())
    assert got["n_groups"] == 12
    assert got["n_cells"] == sum(counts)
    assert got["lo"] < got["point"] < got["hi"]


def test_cluster_bootstrap_ratio_ci_matches_the_cellwise_bootstrap_for_a_mean():
    """Same estimator, same resampling -- so the cheap path must agree with the honest one."""
    rng = np.random.default_rng(3)
    groups = [rng.normal(loc=m, scale=0.2, size=200) for m in rng.normal(size=15)]
    a = mv.cluster_bootstrap_ci(groups, stat=np.mean, n_boot=400, seed=5)
    b = mv.cluster_bootstrap_ratio_ci([g.size for g in groups], [g.sum() for g in groups],
                                      n_boot=400, seed=5)
    assert b["point"] == pytest.approx(a["point"])
    assert b["lo"] == pytest.approx(a["lo"], rel=0.05)
    assert b["hi"] == pytest.approx(a["hi"], rel=0.05)


def test_cluster_bootstrap_ratio_ci_handles_a_fraction():
    """Rich fraction = count(prob>=0.5) / n_cells, which is a ratio of sums."""
    counts = [1000, 2000, 500]
    rich = [100, 600, 50]
    got = mv.cluster_bootstrap_ratio_ci(counts, rich, n_boot=200, seed=0)
    assert got["point"] == pytest.approx(750 / 3500)


def test_cluster_bootstrap_ratio_ci_drops_empty_groups_and_guards_one_group():
    got = mv.cluster_bootstrap_ratio_ci([0, 0, 100], [0.0, 0.0, 25.0], n_boot=50)
    assert got["n_groups"] == 1 and got["point"] == pytest.approx(0.25)
    assert np.isnan(got["lo"]) and "undefined" in got["note"]
    empty = mv.cluster_bootstrap_ratio_ci([0, 0], [0.0, 0.0], n_boot=10)
    assert empty["n_groups"] == 0 and np.isnan(empty["point"])


def test_cluster_bootstrap_ratio_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        mv.cluster_bootstrap_ratio_ci([1, 2, 3], [1.0, 2.0])


def test_nested_variance_decomposition_closes_and_matches_the_cellwise_split():
    rng = np.random.default_rng(7)
    labels, counts, sums, sumsqs, cells_by_unit = [], [], [], [], {}
    for unit, unit_mean in (("A", 0.0), ("B", 1.0), ("C", 2.0)):
        cells_by_unit[unit] = []
        for _ in range(4):
            g = rng.normal(loc=unit_mean + rng.normal(scale=0.3), scale=0.5, size=300)
            labels.append(unit)
            counts.append(g.size)
            sums.append(g.sum())
            sumsqs.append((g ** 2).sum())
            cells_by_unit[unit].append(g)
    out = mv.nested_variance_decomposition(labels, counts, sums, sumsqs)
    assert out["n_units"] == 3 and out["n_polygons"] == 12 and out["n_cells"] == 3600
    # the three components add to the total by construction
    assert out["closure_residual_relative"] < 1e-9
    shares = (out["eta2_between_unit"] + out["eta2_within_unit_between_polygon"]
              + out["eta2_within_polygon"])
    assert shares == pytest.approx(1.0)
    # and the between-unit share agrees with the flat cellwise computation
    flat = mv.variance_decomposition([np.concatenate(v) for v in cells_by_unit.values()])
    assert out["eta2_between_unit"] == pytest.approx(flat["eta2"])


def test_nested_variance_decomposition_detects_within_unit_dominance():
    """The publishable-negative case PLAN_MapValidation named in advance."""
    rng = np.random.default_rng(11)
    labels, counts, sums, sumsqs = [], [], [], []
    for unit in ("A", "B"):                     # units identical in mean...
        for k in range(5):                      # ...but polygons wildly different
            g = rng.normal(loc=10.0 * k, scale=0.1, size=200)
            labels.append(unit)
            counts.append(g.size)
            sums.append(g.sum())
            sumsqs.append((g ** 2).sum())
    out = mv.nested_variance_decomposition(labels, counts, sums, sumsqs)
    assert out["eta2_within_unit_between_polygon"] > 0.9
    assert out["eta2_between_unit"] < 0.01


def test_nested_variance_decomposition_rejects_mismatched_lengths_and_handles_empty():
    with pytest.raises(ValueError, match="same length"):
        mv.nested_variance_decomposition(["A", "B"], [1, 2, 3], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    out = mv.nested_variance_decomposition(["A"], [0], [0.0], [0.0])
    assert out["n_polygons"] == 0 and np.isnan(out["eta2_between_unit"])
