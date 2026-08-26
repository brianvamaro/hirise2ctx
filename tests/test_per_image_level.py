"""Tests for `src.calibration.per_image_level` — R54's per-image level instrument.

The point of the instrument is that the pooled ratio and the per-image distribution can
disagree completely: on the rebuilt arms the pooled ratio is 1.02 while only ~1 image in 5
sits inside ±20 %. So the tests that matter are the ones showing a perfect pooled ratio
coexisting with terrible per-image agreement — which is exactly the state a caller must not
be able to report as "calibrated".
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import LEVEL_BAND, per_image_level


def test_perfect_pooled_ratio_can_hide_total_per_image_disagreement():
    """The R54 failure mode, planted: one image over by 4x, one under by 4x, pooled == 1."""
    obs = np.array(["A", "A", "B", "B"])
    true = np.array([1.0, 1.0, 4.0, 4.0])
    pred = np.array([4.0, 4.0, 1.0, 1.0])
    per, s = per_image_level(obs, true, pred)
    assert s["pooled_ratio"] == pytest.approx(1.0)        # looks perfectly calibrated
    assert s["n_within_band"] == 0                        # and no image is
    assert sorted(per.ratio.round(3)) == [0.25, 4.0]
    assert "NOT evidence" in s["warning"]


def test_a_genuinely_level_product_scores_every_image_in_band():
    obs = np.repeat(["A", "B", "C"], 4)
    true = np.linspace(0.1, 1.0, 12)
    per, s = per_image_level(obs, true, true * 1.05)
    assert s["n_within_band"] == 3 and s["frac_within_band"] == pytest.approx(1.0)
    assert s["per_image_median"] == pytest.approx(1.05)
    assert s["pooled_ratio"] == pytest.approx(1.05)


def test_zero_truth_image_is_undefined_not_dropped_and_not_clipped():
    """An all-zero-truth image has no ratio. Dropping it would shrink the denominator and
    inflate frac_within_band; clipping it would invent a number."""
    obs = np.array(["A", "A", "Z", "Z"])
    true = np.array([1.0, 1.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 0.5, 0.5])
    per, s = per_image_level(obs, true, pred)
    assert s["n_images"] == 2          # both images are still reported
    assert s["n_undefined"] == 1
    assert np.isnan(per.set_index("obs_id").ratio["Z"])
    # the undefined image still counts against the band share
    assert s["n_within_band"] == 1 and s["frac_within_band"] == pytest.approx(0.5)


def test_band_is_configurable_and_recorded():
    obs = np.array(["A", "A", "B", "B"])
    true = np.array([1.0, 1.0, 1.0, 1.0])
    pred = np.array([1.3, 1.3, 1.0, 1.0])
    _, tight = per_image_level(obs, true, pred, band=(0.9, 1.1))
    _, wide = per_image_level(obs, true, pred, band=(0.5, 1.5))
    assert tight["n_within_band"] == 1 and wide["n_within_band"] == 2
    assert tight["band"] == [0.9, 1.1] and wide["band"] == [0.5, 1.5]


def test_default_band_is_the_plan_calibration_band():
    assert LEVEL_BAND == (0.8, 1.2)


def test_summary_names_what_governs_promotion():
    """The audit's requirement was not just the numbers but an explicit statement of which
    aggregation level governs. A caller must not have to guess."""
    obs = np.array(["A", "A"])
    _, s = per_image_level(obs, np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert s["governs_promotion"] == "per_image_frac_within_band"


def test_per_image_frame_carries_tile_counts_so_a_ratio_can_be_weighted():
    obs = np.array(["A", "A", "A", "B"])
    per, _ = per_image_level(obs, np.ones(4), np.ones(4))
    assert per.set_index("obs_id").n_tiles.to_dict() == {"A": 3, "B": 1}
