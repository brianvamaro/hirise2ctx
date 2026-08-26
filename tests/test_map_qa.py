"""Tests for `src/map_qa.py` — the step-12 sidecar/mosaic QA primitives.

The load-bearing behaviour is **not reading an absent measurement as a passing one**. The
shipped sidecars come in three schema generations, and 28 of the 52 carry only the legacy
scalar `overlap_disagreements: 0` — a number counted on the *calibrated* layer, where
isotonic collapses raw fp16 disagreements onto shared knots. A QA table that treats that 0
as "no disagreement" reports a fiction on more than half its rows, so these tests pin the
distinction between `pass` and `unknown_on_gate_layer` rather than just the arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import map_qa


# --------------------------------------------------------------------------- fixtures
def g1_sidecar(scalar=0):
    """Pre-2026-08-24d: no `overlap` block at all, only the calibrated-layer scalar."""
    return {"murray_tile": "E0_N36", "overlap_disagreements": scalar,
            "run": {"grid_id": "g"}, "rasters": []}


def g2_sidecar(n_dup=70000, n_disagree=242, fraction=0.00346, max_abs=8.4e-4, device=None):
    ov = {"gate_layer": "prob_raw",
          "prob_raw": {"n_dup": n_dup, "n_disagree": n_disagree, "fraction": fraction,
                       "max_abs": max_abs},
          "prob": {"n_dup": n_dup, "n_disagree": 1, "fraction": 1.3e-05, "max_abs": 6e-3}}
    run = {"grid_id": "g"}
    if device:
        run["device"] = device
    return {"murray_tile": "E0_N36", "overlap": ov, "overlap_disagreements": n_disagree,
            "run": run, "rasters": []}


def g3_sidecar(n_dup=70000, n_disagree=242, n_significant=4, fraction=5.7e-05):
    ov = {"gate_layer": "prob_raw",
          "prob_raw": {"n_dup": n_dup, "n_disagree": n_disagree, "fraction_raw": 0.00346,
                       "n_significant": n_significant, "fraction": fraction,
                       "significant_abs": 1e-6, "max_abs": 8.4e-4}}
    return {"murray_tile": "E0_N36", "overlap": ov, "run": {"grid_id": "g"}, "rasters": []}


# --------------------------------------------------------------------------- generations
def test_generation_is_keyed_off_structure_not_values():
    assert map_qa.sidecar_generation(g1_sidecar()) == map_qa.GEN_G1
    assert map_qa.sidecar_generation(g2_sidecar()) == map_qa.GEN_G2
    assert map_qa.sidecar_generation(g3_sidecar()) == map_qa.GEN_G3
    # a g1 sidecar whose scalar is nonzero is still g1: the value never decides the generation
    assert map_qa.sidecar_generation(g1_sidecar(scalar=99)) == map_qa.GEN_G1


def test_overlap_key_present_but_not_a_dict_is_g1():
    """A null `overlap` must not be indexed into. This is the shape on disk for g1 tiles
    once they round-trip through JSON with an explicit null."""
    assert map_qa.sidecar_generation({"overlap": None}) == map_qa.GEN_G1


# --------------------------------------------------------------------------- the trap
def test_g1_scalar_zero_is_unknown_never_pass():
    """The regression this module exists for: a missing measurement is not a passing one."""
    row = map_qa.overlap_status(g1_sidecar(scalar=0))
    assert row["verdict"] == "unknown_on_gate_layer"
    assert row["fraction"] is None and row["n_dup"] is None
    # the legacy number is still carried, so nothing is hidden -- it just does not vote
    assert row["scalar_overlap_disagreements"] == 0
    assert "not evidence" in row["note"]


def test_g2_is_labelled_an_upper_bound_and_gated_on_the_raw_fraction():
    row = map_qa.overlap_status(g2_sidecar(fraction=0.00346, n_disagree=242))
    assert row["verdict"] == "pass"
    assert row["gate_layer"] == "prob_raw"
    assert row["fraction"] == pytest.approx(0.00346)
    assert "UPPER BOUND" in row["note"]


def test_g2_over_the_fraction_gate_fails():
    row = map_qa.overlap_status(g2_sidecar(fraction=0.0631, n_disagree=4400))
    assert row["verdict"] == "fail"


def test_absolute_floor_rescues_a_noisy_fraction_on_few_duplicated_cells():
    """The fraction is a noisy estimator when few cells duplicate, so <16 disagreeing cells
    passes regardless. Without this the 1 % gate had only 1.28x margin at 23 tiles."""
    row = map_qa.overlap_status(g2_sidecar(n_dup=100, n_disagree=15, fraction=0.15))
    assert row["verdict"] == "pass"
    row = map_qa.overlap_status(g2_sidecar(n_dup=100, n_disagree=16, fraction=0.16))
    assert row["verdict"] == "fail"


def test_g3_gates_on_the_significant_count_not_the_raw_one():
    """Post-floor, a tile with many raw disagreements but few significant ones passes."""
    row = map_qa.overlap_status(g3_sidecar(n_disagree=4400, n_significant=3, fraction=4e-05))
    assert row["verdict"] == "pass"
    assert row["n_significant"] == 3
    assert "post-1e-6-floor" in row["note"]


# --------------------------------------------------------------------------- device
def test_recorded_device_is_never_marked_inferred():
    row = map_qa.device_status(g2_sidecar(device="NVIDIA GeForce RTX 2080 Ti"), arm="a1")
    assert row["device_inferred"] is False
    assert row["device"] == "NVIDIA GeForce RTX 2080 Ti"


def test_absent_device_is_arm_conditional_and_always_flagged_inferred():
    """Absence identifies "predates the field", NOT a card — and the two arms' pre-field
    tiles ran on different hardware, so the same absence means different things."""
    a1 = map_qa.device_status(g1_sidecar(), arm="a1")
    base = map_qa.device_status(g1_sidecar(), arm="baseline")
    assert a1["device"] != base["device"]
    assert "Pascal" in a1["device"] and "2080 Ti" in base["device"]
    assert a1["device_inferred"] is True and base["device_inferred"] is True
    assert "run logs" in a1["device_evidence"]


# --------------------------------------------------------------------------- mosaic QA
def test_mosaic_footprint_counts_nodata_and_reports_value_range():
    a = np.full((10, 10), 0.5)
    a[0, :] = np.nan          # a fully-nodata row
    a[:, 0] = np.nan          # a fully-nodata column
    f = map_qa.mosaic_footprint(a)
    assert f["shape"] == [10, 10]
    assert f["n_finite"] == 81 and f["n_nodata"] == 19
    assert f["rows_all_nodata"] == 1 and f["cols_all_nodata"] == 1
    assert f["value_min"] == f["value_max"] == pytest.approx(0.5)


def test_mosaic_footprint_on_an_all_nodata_array_omits_value_stats():
    f = map_qa.mosaic_footprint(np.full((4, 4), np.nan))
    assert f["n_finite"] == 0
    assert "value_min" not in f          # nanmin of all-NaN is not a number to report


def test_seam_widths_counts_only_interior_runs():
    """Leading and trailing nodata are the L-corner and the outside margin, not seams."""
    a = np.array([[np.nan, 1.0, np.nan, np.nan, 1.0, np.nan]])
    assert map_qa.seam_widths(a) == {2: 1}


def test_seam_widths_buckets_a_wide_run_and_reports_the_widest():
    """A real hole must not hide inside the seam histogram — it lands in gt_max."""
    row = np.concatenate([[1.0], np.full(20, np.nan), [1.0]])
    got = map_qa.seam_widths(row[None, :], max_width=8)
    assert got["gt_max"] == 1 and got["widest"] == 20


def test_seam_widths_ignores_an_all_nodata_row():
    a = np.array([[np.nan, np.nan, np.nan], [1.0, np.nan, 1.0]])
    assert map_qa.seam_widths(a) == {1: 1}


# --------------------------------------------------------------------------- difference
def test_difference_stats_is_b_minus_a_on_the_common_mask():
    a = np.array([[1.0, 2.0, np.nan]])
    b = np.array([[1.5, 2.0, 5.0]])
    d = map_qa.difference_stats(a, b)
    assert d["n_common"] == 2
    assert d["only_b"] == 1 and d["only_a"] == 0
    assert d["mean"] == pytest.approx(0.25)
    assert d["max_abs"] == pytest.approx(0.5)
    assert d["frac_nonzero"] == pytest.approx(0.5)


def test_difference_stats_refuses_mismatched_shapes():
    """Broadcasting two differently-shaped arms would silently produce a number."""
    with pytest.raises(ValueError, match="not.*differenceable|shape mismatch"):
        map_qa.difference_stats(np.zeros((2, 2)), np.zeros((2, 3)))


def test_difference_stats_with_no_overlap_returns_no_stats():
    a = np.array([[1.0, np.nan]])
    b = np.array([[np.nan, 2.0]])
    d = map_qa.difference_stats(a, b)
    assert d["n_common"] == 0 and "mean" not in d


# ------------------------------------------------------- cross-generation comparison
def _smooth_field(h=240, w=240, seed=0):
    """A smooth spatially-correlated field, like an abundance map at coarse scale."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(0, 1, (h, w)), 6.0)
    return f / f.std()


def test_difference_character_separates_a_shift_from_a_level_change():
    """The discriminator notebook 29 §1d rests on: a translation is high-frequency and
    gradient-following; a regional re-levelling is neither."""
    f = _smooth_field()
    shifted = np.roll(f, 3, axis=1) - f                 # pure displacement
    # a broad regional offset: half the map raised by a constant
    level = np.zeros_like(f)
    level[:, : f.shape[1] // 2] = 0.3

    d_shift = map_qa.difference_character(shifted, f, smooth_px=30)
    d_level = map_qa.difference_character(level, f, smooth_px=30)

    # a shift errs where the field is steep; a flat regional offset does not
    assert d_shift["gradient_rho"] > 0.5
    assert d_level["gradient_rho"] < d_shift["gradient_rho"]
    # and a shift loses its variance under smoothing while a regional offset keeps it
    assert d_shift["smooth_variance_share"] < 0.2
    assert d_level["smooth_variance_share"] > 0.8


def test_difference_character_reports_its_smoothing_scale_and_sample_size():
    f = _smooth_field(80, 80)
    got = map_qa.difference_character(np.roll(f, 2, 0) - f, f, smooth_px=10)
    assert got["smooth_px"] == 10
    assert got["n"] == f.size
    assert got["sd_total"] > 0 and got["sd_smoothed"] >= 0


def test_difference_character_declines_to_answer_on_a_tiny_sample():
    """Better to return n than a spearman rho over nine cells."""
    got = map_qa.difference_character(np.ones((3, 3)), np.ones((3, 3)))
    assert got == {"n": 9}


def test_quantile_table_reports_zero_fraction_and_the_full_range():
    a = np.array([0.0, 0.0, 1.0, 2.0, 3.0, np.nan])
    got = map_qa.quantile_table({"x": a})["x"]
    assert got["n"] == 5                      # the NaN is excluded
    assert got["zero_fraction"] == pytest.approx(0.4)
    assert got["p0"] == 0.0 and got["p100"] == 3.0
    assert got["mean"] == pytest.approx(1.2)


def test_quantile_table_handles_an_all_nan_array_without_raising():
    got = map_qa.quantile_table({"empty": np.full(4, np.nan)})["empty"]
    assert got == {"n": 0}


def test_displacement_sensitivity_of_a_flat_field_is_zero(tmp_path):
    """A shift can only create a difference where the field varies -- so on a constant field
    the geometry bound must be exactly zero, not merely small."""
    import rasterio
    from rasterio.transform import from_origin

    from src.mapping import write_geotiff

    p = tmp_path / "flat.tif"
    write_geotiff(p, np.full((64, 64), 0.25, dtype=np.float32),
                  from_origin(0.0, 10_000.0, 160.0, 160.0), "")
    got = map_qa.displacement_sensitivity(p, -160.0, 0.0)
    assert got["max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert got["dx_m"] == -160.0 and got["dy_m"] == 0.0


def test_displacement_sensitivity_grows_with_the_offset(tmp_path):
    from rasterio.transform import from_origin

    from src.mapping import write_geotiff

    p = tmp_path / "ramp.tif"
    # a linear ramp: |delta| under a shift is proportional to the shift
    ramp = np.tile(np.linspace(0, 1, 128, dtype=np.float32), (128, 1))
    write_geotiff(p, ramp, from_origin(0.0, 20_000.0, 160.0, 160.0), "")
    small = map_qa.displacement_sensitivity(p, -160.0, 0.0)
    large = map_qa.displacement_sensitivity(p, -480.0, 0.0)
    assert large["sd"] > small["sd"]


def test_raster_onto_lands_a_raster_on_another_grid(tmp_path):
    """The old product is not co-registered with the new one, so this warp is the only honest
    comparison path -- it must return the REFERENCE grid's shape, not the source's."""
    import rasterio
    from rasterio.transform import from_origin

    from src.mapping import write_geotiff

    src = tmp_path / "src.tif"
    ref = tmp_path / "ref.tif"
    write_geotiff(src, np.full((40, 40), 0.5, dtype=np.float32),
                  from_origin(0.0, 6_400.0, 160.0, 160.0), "")
    write_geotiff(ref, np.zeros((20, 30), dtype=np.float32),
                  from_origin(160.0, 6_240.0, 160.0, 160.0), "")
    got = map_qa.raster_onto(src, ref)
    assert got.shape == (20, 30)
    inner = got[np.isfinite(got)]
    assert inner.size and np.allclose(inner, 0.5, atol=1e-5)
