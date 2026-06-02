"""Unit tests for `src.stage7d_pooled` -- pooled cross-image colour test.

Synthetic-data tests for the stats primitives + the orchestration. No real parquet
on disk; everything is constructed from numpy in-memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import stage7d_pooled as s7d  # noqa: E402


# ---------------------------------------------------------------------------
# Stats primitives
# ---------------------------------------------------------------------------
def test_cohen_d_known_shift():
    """Two normal samples with mean shift ~1 in unit-std space should give d ~= 1."""
    rng = np.random.default_rng(42)
    x = rng.normal(loc=1.0, scale=1.0, size=2000)
    y = rng.normal(loc=0.0, scale=1.0, size=2000)
    d = s7d.cohen_d(x, y)
    assert 0.85 < d < 1.15


def test_cohen_d_identical_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    d = s7d.cohen_d(x, x.copy())
    assert abs(d) < 1e-12


def test_cohen_d_nan_handling():
    x = np.array([1.0, 2.0, np.nan, 3.0])
    y = np.array([np.nan, np.nan])
    assert np.isnan(s7d.cohen_d(x, y))


def test_mann_whitney_identical_samples():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    r = s7d.mann_whitney_with_effect(x, x.copy())
    assert r["n_rich"] == 200 and r["n_poor"] == 200
    assert r["p_value"] > 0.9
    assert abs(r["effect_size"]) < 1e-9
    assert r["effect_size_type"] == "cohen_d"


def test_mann_whitney_separated_distributions():
    rng = np.random.default_rng(2)
    rich = rng.normal(loc=2.0, scale=1.0, size=500)
    poor = rng.normal(loc=0.0, scale=1.0, size=500)
    r = s7d.mann_whitney_with_effect(rich, poor)
    assert r["p_value"] < 1e-50
    assert r["effect_size"] > 1.5
    assert r["mean_rich"] > r["mean_poor"]


def test_mann_whitney_too_few_samples():
    r = s7d.mann_whitney_with_effect(np.array([1.0]), np.array([2.0, 3.0]))
    assert r["n_rich"] == 1 and r["n_poor"] == 2
    assert np.isnan(r["statistic"])
    assert np.isnan(r["effect_size"])


def test_spearman_monotone():
    x = np.arange(50, dtype=float)
    y = x ** 2  # strictly monotone -> rho == 1
    rho, p, n = s7d.spearman_with_p(x, y)
    assert n == 50
    assert rho == pytest.approx(1.0)
    assert p < 1e-30


def test_spearman_drops_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    rho, p, n = s7d.spearman_with_p(x, y)
    assert n == 3  # only the three paired non-NaN indices
    assert rho == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Per-image standardisation
# ---------------------------------------------------------------------------
def test_per_image_standardise_zero_mean_per_image():
    df = pd.DataFrame({
        "obs_id": ["A"] * 10 + ["B"] * 10,
        "x": np.r_[np.arange(10), np.arange(10) + 100.0],
    })
    out = s7d.per_image_standardise(df, ["x"])
    grouped = out.groupby("obs_id")["x_z"]
    assert np.allclose(grouped.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(grouped.std().to_numpy(), 1.0, atol=1e-12)
    # The two images should overlap in z-space despite the 100-unit offset
    assert out.loc[out.obs_id == "A", "x_z"].min() == pytest.approx(
        out.loc[out.obs_id == "B", "x_z"].min())


def test_per_image_standardise_zero_std_becomes_nan():
    df = pd.DataFrame({"obs_id": ["A", "A", "A"], "x": [1.0, 1.0, 1.0]})
    out = s7d.per_image_standardise(df, ["x"])
    assert out["x_z"].isna().all()


# ---------------------------------------------------------------------------
# Residualisation
# ---------------------------------------------------------------------------
def test_residualise_per_image_removes_linear_dependence():
    rng = np.random.default_rng(7)
    n = 200
    obs = np.repeat(["A", "B"], n // 2)
    x_a = rng.normal(size=n // 2)
    x_b = rng.normal(loc=5.0, size=n // 2)  # different x distribution per image
    x = np.r_[x_a, x_b]
    # Different slopes per image
    y = np.r_[2.0 * x_a + 1.0, -3.0 * x_b + 7.0] + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"obs_id": obs, "x": x, "y": y})
    resid = s7d.residualise_per_image(df, y_col="y", x_col="x")
    # Residuals should be ~uncorrelated with x within each image
    for img in ("A", "B"):
        m = df["obs_id"] == img
        r = np.corrcoef(df.loc[m, "x"], resid[m])[0, 1]
        assert abs(r) < 0.05


def test_residualise_handles_small_or_constant_groups():
    df = pd.DataFrame({
        "obs_id": ["A", "A", "B", "B", "B"],
        "x": [1.0, 2.0, 5.0, 5.0, 5.0],  # B has zero variance in x
        "y": [1.0, 2.0, 9.0, 9.0, 9.0],
    })
    resid = s7d.residualise_per_image(df, y_col="y", x_col="x")
    # A has only 2 rows -> below the min-3 threshold -> NaN
    # B has zero variance in x -> NaN
    assert resid.isna().all()


# ---------------------------------------------------------------------------
# Partition + eligibility + orchestration
# ---------------------------------------------------------------------------
def _make_synth_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """3 images, 100 tiles each. Rich tiles have lower BG, higher IR/BG."""
    rows = []
    for obs_id, n_rich in [("A", 40), ("B", 30), ("C", 5)]:  # C is below min_per_class
        n = 100
        is_rich = np.zeros(n, dtype=bool)
        is_rich[:n_rich] = True
        rng.shuffle(is_rich)
        bg = np.where(is_rich,
                      rng.normal(loc=0.05, scale=0.01, size=n),
                      rng.normal(loc=0.08, scale=0.01, size=n))
        ir = rng.normal(loc=0.17, scale=0.01, size=n)
        red = rng.normal(loc=0.16, scale=0.01, size=n)
        ir_over_bg = ir / bg
        ir_over_red = ir / red
        dust = red / bg
        fa = np.where(is_rich, 0.05, 0.001)
        bc = np.where(is_rich, 100, 5)
        for i in range(n):
            rows.append(dict(obs_id=obs_id, scale_idx=3, ti=i, tj=0,
                             IR_iof=ir[i], RED_iof=red[i], BG_iof=bg[i],
                             IR_over_RED=ir_over_red[i], IR_over_BG=ir_over_bg[i],
                             dust_index_RED_over_BG=dust[i],
                             cos_incidence=0.6,
                             fractional_area=fa[i], boulder_count=bc[i],
                             boulder_area=fa[i] * 320 * 320, tile_area=320 * 320,
                             binary_by_area=fa[i] >= 0.005, binary_by_count=bc[i] >= 1))
    return pd.DataFrame(rows)


def test_add_partitions_columns_present():
    df = _make_synth_dataset(np.random.default_rng(11))
    out = s7d.add_partitions(df)
    assert out["is_rich_P4"].dtype == bool
    assert out["is_rich_P2"].dtype == bool
    assert (out["is_rich_P4"] == (out["fractional_area"] >= 1e-2)).all()
    assert (out["is_rich_P2"] == (out["boulder_count"] > 50)).all()


def test_eligible_images_drops_below_threshold():
    df = s7d.add_partitions(_make_synth_dataset(np.random.default_rng(12)))
    keep = s7d.eligible_images(df, "P4_area", min_per_class=10)
    # A (40 rich, 60 poor) and B (30 rich, 70 poor) qualify; C (5 rich, 95 poor) doesn't
    assert set(keep) == {"A", "B"}


def test_pooled_binary_detects_synthetic_signal():
    df = s7d.add_partitions(_make_synth_dataset(np.random.default_rng(13)))
    out = s7d.run_pooled_binary_tests(df, "P4_area", min_per_class=10)
    bg = out.query(
        "feature == 'BG_iof' and test_type == 'mann_whitney_standardised'").iloc[0]
    # BG was designed lower for rich -> effect_size negative + significant
    assert bg["p_value"] < 1e-10
    assert bg["effect_size"] < -1.0
    # IR was designed null -> effect_size near zero
    ir = out.query(
        "feature == 'IR_iof' and test_type == 'mann_whitney_standardised'").iloc[0]
    assert abs(ir["effect_size"]) < 0.3


def test_pooled_partial_dust_present_and_skips_self():
    df = s7d.add_partitions(_make_synth_dataset(np.random.default_rng(14)))
    out = s7d.run_pooled_binary_tests(df, "P4_area", min_per_class=10)
    types = set(out["test_type"].unique())
    assert {"mann_whitney_raw", "mann_whitney_standardised",
            "mann_whitney_partial_dust"}.issubset(types)
    # dust feature is excluded from partial-dust rows
    dust_partial = out.query(
        "feature == 'dust_index_RED_over_BG' "
        "and test_type == 'mann_whitney_partial_dust'")
    assert len(dust_partial) == 0


def test_per_image_binary_skips_low_count_image():
    df = s7d.add_partitions(_make_synth_dataset(np.random.default_rng(15)))
    out = s7d.run_per_image_binary_tests(df, "P4_area", min_per_class=10)
    assert set(out["obs_id"].unique()) == {"A", "B"}  # C dropped
    assert (out["level"] == "per_image").all()


def test_spearman_runs_pooled_and_per_image():
    df = s7d.add_partitions(_make_synth_dataset(np.random.default_rng(16)))
    out = s7d.run_spearman_tests(df)
    types = set(out["test_type"].unique())
    assert {"spearman_count_standardised", "spearman_count_partial_dust",
            "spearman_count_raw"}.issubset(types)
    # All effect-size types should be spearman_rho
    assert (out["effect_size_type"] == "spearman_rho").all()


def test_run_all_returns_concatenated_results():
    df = s7d.add_partitions(_make_synth_dataset(np.random.default_rng(17)))
    out = s7d.run_all(df, min_per_class=10)
    # Both partition rules + spearman should be present
    assert set(out["partition_rule"].dropna().unique()) == {"P4_area", "P2_count"}
    levels = set(out["level"].unique())
    assert {"pooled", "per_image"}.issubset(levels)
