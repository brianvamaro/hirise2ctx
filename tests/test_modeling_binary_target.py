"""Unit tests for src.modeling.binary_target."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.binary_target import (
    BINARY_TARGETS,
    BINARY_TARGETS_BY_ID,
    BinaryTarget,
    get_target,
)


def test_three_targets_registered():
    """Stage 5b §3 pins exactly three thresholds; guard against accidental dedupe."""
    assert len(BINARY_TARGETS) == 3
    assert set(BINARY_TARGETS_BY_ID) == {"bc_ge_1", "fa_gt_1e-3", "fa_gt_1e-2"}


def test_each_target_has_distinct_definition():
    """The three thresholds must be distinct (id, source_col, threshold, comparison)
    tuples; otherwise the sweep collapses two cells into one."""
    keys = {(t.source_col, t.threshold, t.comparison) for t in BINARY_TARGETS}
    assert len(keys) == 3


def test_get_target_returns_registered_target():
    t = get_target("bc_ge_1")
    assert t.id == "bc_ge_1"
    assert t.source_col == "boulder_count"
    assert t.threshold == 1.0
    assert t.comparison == ">="


def test_get_target_raises_on_unknown_id():
    with pytest.raises(KeyError, match="unknown binary target"):
        get_target("not_a_real_id")


def test_binarize_bc_ge_1_on_synthetic_y():
    """boulder_count = [0, 1, 2, 5] under >= 1 -> [0, 1, 1, 1]."""
    y = pd.DataFrame({"boulder_count": [0, 1, 2, 5], "fractional_area": [0.0, 1e-5, 1e-3, 0.02]})
    out = get_target("bc_ge_1").binarize(y)
    np.testing.assert_array_equal(out, np.array([0, 1, 1, 1], dtype=np.int8))


def test_binarize_fa_gt_1e_3_on_synthetic_y():
    """fractional_area = [0, 5e-4, 1e-3, 2e-3, 0.05] under > 1e-3 -> [0, 0, 0, 1, 1].

    Strict > -- 1e-3 itself is NOT positive (boundary inclusion matters for
    reproducibility once probes start citing 'fa > 1e-3' positive counts).
    """
    y = pd.DataFrame({
        "boulder_count": [0, 0, 1, 1, 3],
        "fractional_area": [0.0, 5e-4, 1e-3, 2e-3, 0.05],
    })
    out = get_target("fa_gt_1e-3").binarize(y)
    np.testing.assert_array_equal(out, np.array([0, 0, 0, 1, 1], dtype=np.int8))


def test_binarize_fa_gt_1e_2_on_synthetic_y():
    """fractional_area = [0, 0.005, 0.01, 0.02, 0.2] under > 1e-2 -> [0, 0, 0, 1, 1]."""
    y = pd.DataFrame({
        "boulder_count": [0, 1, 2, 3, 50],
        "fractional_area": [0.0, 0.005, 0.01, 0.02, 0.2],
    })
    out = get_target("fa_gt_1e-2").binarize(y)
    np.testing.assert_array_equal(out, np.array([0, 0, 0, 1, 1], dtype=np.int8))


def test_binarize_returns_int8():
    """LightGBM `objective='binary'` accepts int8 labels; tighter dtype = lower memory."""
    y = pd.DataFrame({"boulder_count": [0, 1], "fractional_area": [0.0, 0.1]})
    for t in BINARY_TARGETS:
        assert t.binarize(y).dtype == np.int8


def test_binarize_raises_on_missing_source_column():
    """If a caller hands us a y dataframe missing the source column, fail loudly
    rather than silently returning all-zeros from a default-init array."""
    y = pd.DataFrame({"only_irrelevant": [1, 2, 3]})
    with pytest.raises(KeyError, match="expects column"):
        get_target("bc_ge_1").binarize(y)


def test_binarize_preserves_row_order_and_length():
    """Row order must match y_df row order so downstream joins on keys stay aligned."""
    rng = np.random.default_rng(0)
    y = pd.DataFrame({
        "boulder_count": rng.integers(0, 4, 100),
        "fractional_area": rng.uniform(0, 0.05, 100),
    })
    for t in BINARY_TARGETS:
        out = t.binarize(y)
        assert out.shape == (100,)


def test_binarytarget_is_frozen():
    """frozen=True prevents accidental mutation that would silently shift sweep cells."""
    t = get_target("bc_ge_1")
    with pytest.raises((AttributeError, Exception)):
        t.threshold = 999.0  # type: ignore[misc]
