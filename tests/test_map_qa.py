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
