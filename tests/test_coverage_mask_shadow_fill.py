"""R74 — the coverage mask must not call deep shadow "no coverage".

HiRISE RDR DN is continuous through zero, so a pixel in a boulder's shadow reads 0 and is
indistinguishable *by value* from unimaged ground. Combined with nearest-neighbour
decimation and Stage 4's unanimous `mask_min == 1` eligibility, a single shadowed pixel
used to delete the 40/80/160/320 m tiles containing it — and because dark pixels sit next
to boulders, the deleted tiles were the rockiest ones (3,236 of 164,273 S=32 tiles, 93 %
rich, 7.70 % of all detected boulder area).

`_fill_interior_shadow_holes` separates shadow from geometry *topologically*: only regions
fully enclosed by valid data are candidates, and only ones no larger than the threshold.
The fix shipped 2026-08-05 with no direct tests; these are them. Pure synthetic arrays —
no cache, no producer, no imagery.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.ctx_retrieve import (
    COVERAGE_MASK_METHOD,
    COVERAGE_MASK_VERSION,
    DEFAULT_MAX_INTERIOR_HOLE_PX,
    _fill_interior_shadow_holes,
)


def _valid_field(h: int = 24, w: int = 24) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def test_small_enclosed_hole_is_refilled():
    """The case the finding is about: a 1-2 px shadow puddle inside the swath."""
    valid = _valid_field()
    valid[10, 10] = 0
    valid[15, 4:6] = 0  # a 2 px hole as well
    out, n_filled = _fill_interior_shadow_holes(valid, max_hole_px=DEFAULT_MAX_INTERIOR_HOLE_PX)
    assert n_filled == 3
    assert out[10, 10] == 1
    assert (out[15, 4:6] == 1).all()
    assert out.sum() == valid.size


def test_hole_larger_than_the_threshold_is_left_alone():
    """A genuine enclosed dropout must survive — that is what the size bound is for."""
    valid = _valid_field()
    valid[6:12, 6:12] = 0  # 36 px, over the 16 px threshold
    out, n_filled = _fill_interior_shadow_holes(valid, max_hole_px=DEFAULT_MAX_INTERIOR_HOLE_PX)
    assert n_filled == 0
    assert (out[6:12, 6:12] == 0).all()
    np.testing.assert_array_equal(out, valid)


def test_threshold_boundary_is_inclusive():
    """`<= max_hole_px` fills, one pixel more does not. Pins the comparison, not the number."""
    exact = _valid_field()
    exact[4:8, 4:8] = 0                       # 16 px, exactly at the threshold
    out, n = _fill_interior_shadow_holes(exact, max_hole_px=16)
    assert n == 16 and (out[4:8, 4:8] == 1).all()

    over = _valid_field()
    over[4:8, 4:8] = 0
    over[8, 4] = 0                            # 17 px, one over
    out, n = _fill_interior_shadow_holes(over, max_hole_px=16)
    assert n == 0 and (out[4:8, 4:8] == 0).all()


def test_edge_connected_invalid_region_is_never_filled():
    """Swath geometry — the rotated-rectangle corners and any missing scan that reaches the
    array border — is connected to the edge, so `binary_fill_holes` leaves it untouched.
    This is the property that makes the correction safe to apply blind."""
    valid = _valid_field()
    valid[0:3, 0:3] = 0        # corner, touches two borders
    valid[:, -1] = 0           # a full column at the right edge
    valid[20:22, 0:2] = 0      # notch on the left border
    before = valid.copy()
    out, n_filled = _fill_interior_shadow_holes(valid, max_hole_px=DEFAULT_MAX_INTERIOR_HOLE_PX)
    assert n_filled == 0
    np.testing.assert_array_equal(out, before)


def test_edge_connected_region_stays_out_even_with_an_interior_hole_present():
    """The interesting mixed case: one enclosed puddle and one edge-connected gap. Exactly
    one of them may change."""
    valid = _valid_field()
    valid[0:4, 0:4] = 0        # edge-connected, 16 px — small enough to be filled if the
    valid[12, 12] = 0          # topology test were missing
    out, n_filled = _fill_interior_shadow_holes(valid, max_hole_px=DEFAULT_MAX_INTERIOR_HOLE_PX)
    assert n_filled == 1
    assert out[12, 12] == 1
    assert (out[0:4, 0:4] == 0).all(), "an edge-connected region was filled — geometry lost"


@pytest.mark.parametrize("threshold", [0, -1])
def test_disabled_threshold_is_an_exact_no_op(threshold):
    """`max_interior_hole_px <= 0` must restore the pre-R74 mask bit for bit, so the
    counterfactual comparison in PENDING_REBUILD.md is exact."""
    rng = np.random.default_rng(11)
    valid = (rng.random((40, 40)) > 0.1).astype(np.uint8)
    before = valid.copy()
    out, n_filled = _fill_interior_shadow_holes(valid, max_hole_px=threshold)
    assert n_filled == 0
    np.testing.assert_array_equal(out, before)
    np.testing.assert_array_equal(
        valid, before, err_msg="the input array was mutated in place",
    )


def test_fill_only_ever_adds_coverage():
    """Never removes a valid pixel — the swath border and every real observation are safe
    regardless of threshold. Checked over random fields at several hole densities."""
    rng = np.random.default_rng(5)
    for density in (0.01, 0.05, 0.2):
        valid = (rng.random((48, 48)) > density).astype(np.uint8)
        out, n_filled = _fill_interior_shadow_holes(valid, max_hole_px=32)
        assert ((out == 1) | (valid == 0)).all(), "a valid pixel was un-marked"
        assert (out >= valid).all()
        assert int(out.sum() - valid.sum()) == n_filled, "n_filled disagrees with the mask"


def test_all_valid_and_all_invalid_are_both_no_ops():
    full = np.ones((16, 16), dtype=np.uint8)
    out, n = _fill_interior_shadow_holes(full, max_hole_px=16)
    assert n == 0 and (out == 1).all()

    empty = np.zeros((16, 16), dtype=np.uint8)
    out, n = _fill_interior_shadow_holes(empty, max_hole_px=16)
    assert n == 0 and (out == 0).all(), "an all-nodata window must not be invented into data"


def test_mask_algorithm_identity_is_declared():
    """The version must be bumped whenever the output can change for unchanged inputs;
    Stage 2 persists it so a pre-R74 mask is distinguishable from a post-R74 one."""
    assert COVERAGE_MASK_VERSION >= 2, (
        "version 1 is the pre-R74 algorithm; bump this when the mask output changes"
    )
    assert "shadow" in COVERAGE_MASK_METHOD
    assert DEFAULT_MAX_INTERIOR_HOLE_PX == 16
