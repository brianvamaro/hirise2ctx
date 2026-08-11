"""R03 / R83 / R84 — the deployed abundance layer's size-floor basis (PLAN_RegionalMap leg 4).

`fractional_area` is not size-independent rock abundance. It is the area share of boulders large
enough for BoulderNet to have detected them *in that HiRISE image*, and across the v2 cohort that
qualifier varies by ~3.6x in area. The deployed raster is quantile-matched onto a pool that **mixes**
those conventions, and the product recorded none of it: `write_geotiff` wrote no tags at all.

Measured on the real cohort 2026-08-11 and used as fixtures below:

  * pool = **161,005** S=32 tiles over 38 images; **78.3914 %** at 0.50 m/px, **21.6086 %** at
    0.25 m/px — R84's previously-unverified 78.4 / 21.6, independently re-derived;
  * image share is **68.4 / 31.6**, a *different* quantity the audit warned against conflating;
  * effective floors: **1.5626 m² for all 12 fine images** (the global Stage-4 filter) vs
    **2.9652–5.5719 m² across the 26 coarse** — so the coarse cohort is the internally
    heterogeneous one, which is R83's correction to R03;
  * `calibration.npz` `t2_y` max == pool max `fa` == 0.293242 exactly, R84's proof of which pool.

Mutants these kill:
  M1  `effective_floor_m2` returns the natural floor, ignoring the Stage-4 filter (R03's error)
  M2  the basis reports image share where tile share is meant
  M3  `product_tags` omits the mixture, or `write_geotiff` drops tags again
  M4  a basis is constructible over an empty pool, so it states a mixture it never counted
  M5  `map_scale_from_pds_label` silently returns None instead of the cached MAP_SCALE
"""
import json

import numpy as np
import pytest

from src.size_floor import (DEFAULT_MIN_SIZE_M, SIZE_FLOOR_BASIS_VERSION, SizeFloorBasis,
                            area_to_diameter, diameter_to_area, effective_floor_m2,
                            map_scale_from_pds_label)

FILTER_AREA = diameter_to_area(DEFAULT_MIN_SIZE_M)          # 1.5626 m²


def _cohort():
    """A miniature of the real cohort: 2 fine below the filter, 2 coarse above it."""
    per_image = [
        {"obs_id": "FINE_A", "map_scale_mpp": 0.25, "natural_floor_m2": 0.830, "n_polygons": 10},
        {"obs_id": "FINE_B", "map_scale_mpp": 0.25, "natural_floor_m2": 1.156, "n_polygons": 10},
        {"obs_id": "COARSE_A", "map_scale_mpp": 0.5, "natural_floor_m2": 2.9652, "n_polygons": 10},
        {"obs_id": "COARSE_B", "map_scale_mpp": 0.5, "natural_floor_m2": 5.5719, "n_polygons": 10},
    ]
    # tile counts chosen to reproduce the real 78.4 / 21.6 tile share
    counts = {"FINE_A": 17000, "FINE_B": 17791, "COARSE_A": 63000, "COARSE_B": 63214}
    return per_image, counts


# ---------------------------------------------------------------- the effective floor

def test_the_global_filter_is_the_floor_for_the_fine_cohort_and_nothing_for_the_coarse():
    """M1, and it is R83's correction to R03 in one assertion.

    Stage 4 applies `min_size_m` AFTER Stage 1, so the Stage-1 polygon minimum is not the floor.
    The filter (1.5626 m²) sits above every fine image's natural floor and below every coarse
    one's, so it binds on exactly one cohort. Reading the raw minima as "the floor" understates
    the fine cohort's by ~2x and makes the fine cohort look heterogeneous when it is uniform.
    """
    assert effective_floor_m2(0.830) == pytest.approx(FILTER_AREA)    # fine -> the filter
    assert effective_floor_m2(1.156) == pytest.approx(FILTER_AREA)    # fine -> the filter
    assert effective_floor_m2(2.9652) == pytest.approx(2.9652)        # coarse -> its own
    assert effective_floor_m2(5.5719) == pytest.approx(5.5719)        # coarse -> its own
    assert FILTER_AREA == pytest.approx(1.5626, abs=1e-4)


def test_the_filter_floor_round_trips_through_the_diameter_convention():
    """`min_size_m` is an equivalent-circle DIAMETER; getting that wrong scales the floor by ~4x."""
    assert area_to_diameter(diameter_to_area(1.4105)) == pytest.approx(1.4105)
    assert area_to_diameter(FILTER_AREA) == pytest.approx(DEFAULT_MIN_SIZE_M)


def test_a_bigger_global_filter_would_bind_on_both_cohorts():
    """The floor is a function of the config in force, not a constant of the cohort."""
    assert effective_floor_m2(2.9652, min_size_m=3.0) == pytest.approx(diameter_to_area(3.0))


# ---------------------------------------------------------------- the mixture

def test_the_basis_reproduces_the_measured_tile_share():
    """M2. Tile share (78.4 / 21.6) is what R84 means; image share (68.4 / 31.6) is a different
    number and quoting one for the other is wrong by ten points."""
    basis = SizeFloorBasis.from_records(*_cohort())
    assert basis.n_tiles == 161_005
    assert basis.tile_share_by_scale["0.5"] == pytest.approx(0.783914, abs=1e-5)
    assert basis.tile_share_by_scale["0.25"] == pytest.approx(0.216086, abs=1e-5)
    # ... and the image share is carried separately, precisely so they cannot be confused
    assert basis.image_share_by_scale["0.5"] == pytest.approx(0.5)
    assert basis.tile_share_by_scale != basis.image_share_by_scale


def test_the_basis_counts_distinct_floors_not_distinct_images():
    """The fine cohort collapses onto ONE floor (the filter); the coarse keeps its own."""
    basis = SizeFloorBasis.from_records(*_cohort())
    assert basis.n_images == 4
    assert basis.n_distinct_floors == 3, "two fine images share the filter floor"
    assert basis.floor_min_m2 == pytest.approx(FILTER_AREA)
    assert basis.floor_max_m2 == pytest.approx(5.5719)


def test_the_mean_floor_is_tile_weighted_not_image_weighted():
    """An image-weighted mean would over-represent the fine cohort 1.5x here, and the deployed
    layer is calibrated per tile, not per image."""
    per_image, counts = _cohort()
    basis = SizeFloorBasis.from_records(per_image, counts)
    floors = np.array([FILTER_AREA, FILTER_AREA, 2.9652, 5.5719])
    w = np.array([counts[r["obs_id"]] for r in per_image], dtype=float)
    assert basis.floor_tile_weighted_mean_m2 == pytest.approx((floors * w).sum() / w.sum())
    assert basis.floor_tile_weighted_mean_m2 != pytest.approx(floors.mean())


def test_an_image_with_no_pool_tiles_contributes_no_floor():
    """It is in the cohort but not in the product, so its floor must not widen the stated range."""
    per_image, counts = _cohort()
    per_image.append({"obs_id": "ORPHAN", "map_scale_mpp": 0.5,
                      "natural_floor_m2": 99.0, "n_polygons": 1})
    counts["ORPHAN"] = 0
    basis = SizeFloorBasis.from_records(per_image, counts)
    assert basis.n_distinct_floors == 3, "a zero-tile image must not add a floor"
    assert basis.tile_share_by_scale["0.5"] == pytest.approx(0.783914, abs=1e-5)


def test_an_empty_pool_refuses_to_become_a_basis():
    """M4. Absence must not read as a measured mixture."""
    per_image, _ = _cohort()
    with pytest.raises(ValueError, match="states nothing"):
        SizeFloorBasis.from_records(per_image, {r["obs_id"]: 0 for r in per_image})


# ---------------------------------------------------------------- what the product carries

def test_product_tags_state_the_mixture_and_refuse_the_size_independent_reading():
    """M3 at the seam. A reader must be able to answer 'what size boulders is this counting?'
    from the raster alone."""
    tags = SizeFloorBasis.from_records(*_cohort()).product_tags()
    assert json.loads(tags["SIZE_FLOOR_TILE_SHARE_BY_MPP"])["0.5"] == pytest.approx(0.783914,
                                                                                    abs=1e-5)
    assert tags["SIZE_FLOOR_N_DISTINCT"] == "3"
    assert tags["SIZE_FLOOR_N_POOL_TILES"] == "161005"
    assert "NOT size-independent rock abundance" in tags["SIZE_FLOOR_SUMMARY"]
    assert all(isinstance(v, str) for v in tags.values()), "GDAL metadata is string-valued"


def test_the_basis_round_trips_and_refuses_a_foreign_version(tmp_path):
    basis = SizeFloorBasis.from_records(*_cohort())
    p = basis.to_json(tmp_path / "size_floor_basis.json")
    assert SizeFloorBasis.load(p).product_tags() == basis.product_tags()

    d = json.loads(p.read_text(encoding="utf-8"))
    d["version"] = "v1_something_else"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="re-measure rather than mixing"):
        SizeFloorBasis.load(p)
    assert SIZE_FLOOR_BASIS_VERSION in basis.product_tags()["SIZE_FLOOR_BASIS_VERSION"]


def test_write_geotiff_actually_persists_the_tags(tmp_path):
    """M3 at the product boundary. `write_geotiff` wrote no metadata at all, so a shipped raster
    could not state what its abundance number counts -- R84's whole complaint."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    from src.mapping import write_geotiff

    tags = SizeFloorBasis.from_records(*_cohort()).product_tags()
    p = write_geotiff(tmp_path / "a.tif", np.zeros((4, 4)), Affine(160, 0, 0, 0, -160, 0),
                      "", tags=tags)
    with rasterio.open(p) as ds:
        got = ds.tags()
    for k, v in tags.items():
        assert got.get(k) == v, f"{k} did not survive the write"


def test_write_geotiff_without_tags_is_unchanged(tmp_path):
    """The legacy call must keep working -- every existing caller passes no tags."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    from src.mapping import write_geotiff

    p = write_geotiff(tmp_path / "b.tif", np.zeros((4, 4)), Affine(160, 0, 0, 0, -160, 0), "")
    with rasterio.open(p) as ds:
        assert not [k for k in ds.tags() if k.startswith("SIZE_FLOOR")]


# ---------------------------------------------------------------- the pixel scale

def test_an_unset_basis_yields_no_tags_rather_than_loading_the_working_directory(tmp_path):
    """A trap the suite caught: `Path("")` is `.`, and `Path(".").exists()` is **True**, so an
    unset `--size-floor-basis` sailed past an `exists()` guard and tried to parse the working
    directory as JSON. Absent must mean untagged, never fabricated and never a crash."""
    from types import SimpleNamespace

    import scripts.map_region as mr

    for raw in ("", None):
        assert mr.size_floor_tags(SimpleNamespace(size_floor_basis=raw)) == {}
    assert mr.size_floor_tags(SimpleNamespace()) == {}
    # a DIRECTORY at the path is not a basis either
    assert mr.size_floor_tags(SimpleNamespace(size_floor_basis=str(tmp_path))) == {}


def test_a_banked_basis_is_stamped_by_both_map_drivers(tmp_path):
    """The two rows are compared cell for cell, so they must count the same boulders."""
    from types import SimpleNamespace

    import scripts.map_region as mr
    import scripts.striping_a1_map as a1

    p = SizeFloorBasis.from_records(*_cohort()).to_json(tmp_path / "basis.json")
    args = SimpleNamespace(size_floor_basis=str(p))
    tags = mr.size_floor_tags(args)
    assert tags["SIZE_FLOOR_N_POOL_TILES"] == "161005"
    assert tags["SIZE_FLOOR_BASIS_PATH"] == str(p)
    # the A1 driver reuses the identical helper, so the two arms cannot drift apart
    assert a1.size_floor_tags is mr.size_floor_tags


def test_map_scale_is_read_from_the_pds_label(tmp_path):
    """M5. The manifest sources `MapPixel_mpp` from the label spreadsheet, which is why two cohort
    rows are blank; the `.LBL` is the authoritative source and is already cached."""
    lbl = tmp_path / "ESP_X.LBL"
    lbl.write_text("OBJECT = IMAGE_MAP_PROJECTION\n"
                   "  MAP_SCALE   = 0.5 <METERS/PIXEL>\n"
                   "END_OBJECT\n", encoding="utf-8")
    assert map_scale_from_pds_label(lbl) == 0.5
    assert map_scale_from_pds_label(tmp_path / "nope.LBL") is None


def test_a_malformed_map_scale_returns_none_rather_than_guessing(tmp_path):
    lbl = tmp_path / "bad.LBL"
    lbl.write_text("MAP_SCALE = UNKNOWN\n", encoding="utf-8")
    assert map_scale_from_pds_label(lbl) is None
