"""Tests for `src/fgates.py` — the PLAN_FBuild §5 acceptance-gate scorers.

The gate-1 machinery is the load-bearing part: it must be able to tell a planted per-frame BLOCK
offset (an artifact) from a spatially-structured geology field (not an artifact). A scorer that fires
on both — which a bare η² at regional scale does, since 79% of the mosaic map's block-scale η² is
reproduced by rolling the field — is useless as a gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import fgates as fg
from src import leveling as lv


def blocky_field(h=200, w=200, n_frames=4, block_offset=0.0, seed=0):
    """A smooth geology field + per-frame vertical-stripe offsets, in probability space."""
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    geology = gaussian_filter(rng.normal(0, 1, (h, w)), 12.0)
    geology /= geology.std()
    labels = np.zeros((h, w), dtype=np.int32)
    step = w // n_frames
    off = np.linspace(-1, 1, n_frames) * block_offset
    logit = geology.copy()
    for k in range(n_frames):
        c0, c1 = k * step, (k + 1) * step if k < n_frames - 1 else w
        labels[:, c0:c1] = k
        logit[:, c0:c1] += off[k]
    return lv.sigmoid(logit), labels


# --------------------------------------------------------------------------- gate 1
def test_eta2_with_null_flags_a_planted_frame_offset():
    v, lab = blocky_field(block_offset=1.5)
    e, nm, n95, nc, nf = fg.eta2_with_null(v, lab, n_draws=8)
    assert nf == 4 and nc == v.size
    assert e > n95, "a real per-frame offset must exceed its own rotation-null p95"
    assert e - nm > 0.05


def test_eta2_with_null_does_not_flag_pure_geology():
    """No frame offset at all: η² must sit inside the rotation-null spread. This is the check that
    makes the floor-relative reading meaningful."""
    v, lab = blocky_field(block_offset=0.0, seed=5)
    e, nm, n95, _, _ = fg.eta2_with_null(v, lab, n_draws=20, seed=1)
    assert e <= n95 * 1.5, f"eta2 {e} vs null p95 {n95}: geology alone should not clear the floor"


def test_eta2_with_null_returns_nan_when_too_thin():
    v, lab = blocky_field(h=20, w=20, n_frames=2)
    e, nm, n95, nc, nf = fg.eta2_with_null(v, lab, min_cells=10_000)
    assert np.isnan(e) and np.isnan(nm) and nc == 400


def test_eta2_ignores_unlabelled_cells():
    v, lab = blocky_field(block_offset=1.0)
    lab2 = lab.copy()
    lab2[:50] = -1
    e_all, *_ = fg.eta2_with_null(v, lab, n_draws=4)
    e_sub, *_ = fg.eta2_with_null(v, lab2, n_draws=4)
    assert np.isfinite(e_all) and np.isfinite(e_sub) and e_all != e_sub


def test_window_eta2_scores_each_window_against_its_own_null():
    v, lab = blocky_field(h=400, w=400, block_offset=1.2)
    scores = fg.window_eta2(v, lab, "T", win_px=200, n_draws=6)
    assert len(scores) == 4
    assert all(np.isfinite(s.eta2) and np.isfinite(s.null_p95) for s in scores)
    assert all(s.n_cells == 200 * 200 for s in scores)


def test_window_eta2_drops_sliver_windows():
    v, lab = blocky_field(h=250, w=250, block_offset=1.0)
    scores = fg.window_eta2(v, lab, "T", win_px=200, n_draws=4)
    assert len(scores) == 1                       # the 50-px remainders are slivers


def test_summarize_windows_applies_the_bar_to_the_median():
    mk = lambda e: fg.WindowScore("T", 0, 0, 10_000, 5, e, 0.01, 0.02)  # noqa: E731
    s = fg.summarize_windows([mk(0.01), mk(0.02), mk(0.9)])
    assert s["eta2_median"] == pytest.approx(0.02) and s["passes_bar"] is True
    assert s["frac_windows_below_bar"] == pytest.approx(2 / 3)
    s2 = fg.summarize_windows([mk(0.2), mk(0.3), mk(0.01)])
    assert s2["passes_bar"] is False


def test_summarize_windows_handles_no_windows():
    s = fg.summarize_windows([])
    assert s["n_windows"] == 0 and s["passes_bar"] is False and np.isnan(s["eta2_median"])


def test_window_score_excess_and_ratio():
    s = fg.WindowScore("T", 0, 0, 100, 3, 0.20, 0.08, 0.12)
    assert s.excess == pytest.approx(0.12) and s.ratio == pytest.approx(0.20 / 0.12)


# --------------------------------------------------------------------------- gate 2
def _edges(n_frames=8, side=30, overlap=20, bias=None, seed=0):
    rng = np.random.default_rng(seed)
    bias = np.linspace(-0.8, 0.8, n_frames) if bias is None else np.asarray(bias, float)
    rows = np.arange(40)
    truth, pids, keys, logits = {}, [], [], []
    for f in range(n_frames):
        c0 = f * (side - overlap)
        ti, tj = [a.ravel() for a in np.meshgrid(rows, np.arange(c0, c0 + side), indexing="ij")]
        key = lv.pack_key(ti, tj)
        for k in key.tolist():
            truth.setdefault(k, rng.normal(0, 1.2))
        lg = np.array([truth[k] for k in key.tolist()]) + bias[f]
        o = np.argsort(key)
        pids.append(f"F{f}")
        keys.append(key[o])
        logits.append(lg[o].astype(np.float32))
    return lv.build_edges(pids, keys, logits, min_tiles=1), bias


def test_edge_cv_reports_unleveled_insample_and_heldout():
    es, bias = _edges()
    n = len(es.pids)
    o = -(bias - np.median(bias))
    r = fg.edge_cv_for_offsets(es, o, n, 1.0, frac=0.2, repeats=3)
    assert r["insample_dp"] < r["unleveled_dp"]
    assert r["passes"] is True


def test_edge_cv_for_zero_offsets_is_the_baseline():
    es, _ = _edges()
    n = len(es.pids)
    r = fg.edge_cv_for_offsets(es, np.zeros(n), n, 1.0, frac=0.2, repeats=2)
    assert r["insample_dp"] == pytest.approx(r["unleveled_dp"])


# --------------------------------------------------------------------------- gate 3
def test_spearman_rho_over_covalid_cells_only():
    a = np.array([1.0, 2, 3, 4, 5, np.nan] * 20)
    b = np.array([2.0, 4, 6, 8, 10, 1] * 20)
    r, n = fg.spearman_rho(a, b)
    assert r == pytest.approx(1.0) and n == 100


def test_spearman_rho_refuses_thin_overlap():
    r, n = fg.spearman_rho(np.arange(10.0), np.arange(10.0))
    assert np.isnan(r) and n == 10


def test_common_finite_is_the_intersection():
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([1.0, 2.0, np.nan, 4.0])
    c = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(fg.common_finite(a, b, c)) == [True, False, False, True]


# --------------------------------------------------------------------------- gates 5 / 6
def test_cohort_tiles_to_global_uses_the_stage_b_keying():
    """A cohort tile's (ti,tj) in its own window grid must land on the SAME global node Stage B
    would have produced for the same ground position."""
    bounds = pd.DataFrame([{"obs_id": "ESP_1", "minx": 480000.0, "miny": 2000000.0,
                            "maxx": 500000.0, "maxy": 2020000.0, "row0": 0, "col0": 0}])
    labels = pd.DataFrame({"obs_id": ["ESP_1", "ESP_1"], "ti": [0, 3], "tj": [0, 5],
                           "fractional_area": [0.0, 0.05]})
    got = fg.cohort_tiles_to_global(bounds, labels)
    want_tj = np.round((480000.0 + (np.array([0, 5]) + 0.5) * 160.0) / 160.0).astype(int)
    want_ti = np.round((2020000.0 - (np.array([0, 3]) + 0.5) * 160.0) / 160.0).astype(int)
    assert list(got.TJ) == list(want_tj) and list(got.TI) == list(want_ti)


def test_cohort_join_drops_obs_without_bounds():
    bounds = pd.DataFrame([{"obs_id": "ESP_1", "minx": 0.0, "miny": 0.0, "maxx": 1.0,
                            "maxy": 1000.0, "row0": 0, "col0": 0}])
    labels = pd.DataFrame({"obs_id": ["ESP_1", "ESP_2"], "ti": [0, 0], "tj": [0, 0],
                           "fractional_area": [0.0, 0.1]})
    got = fg.cohort_tiles_to_global(bounds, labels)
    assert list(got.obs_id) == ["ESP_1"]


def test_cohort_join_rejects_bad_bounds_schema():
    with pytest.raises(ValueError, match="obs_bounds needs"):
        fg.cohort_tiles_to_global(pd.DataFrame({"obs_id": ["a"]}),
                                  pd.DataFrame({"obs_id": ["a"], "ti": [0], "tj": [0]}))


def test_pooled_skill_matches_the_declared_conventions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 2000), 0, 1)
    r = fg.pooled_skill(y, p)
    assert 0.5 < r["pooled_pr_auc"] <= 1.0
    k = max(1, int(round(0.05 * 2000)))
    assert r["precision@5%"] == pytest.approx(y[np.argsort(-p)[:k]].mean())
    assert r["n"] == 2000


def test_pooled_skill_is_nan_on_single_class():
    r = fg.pooled_skill(np.zeros(100, int), np.linspace(0, 1, 100))
    assert np.isnan(r["pooled_pr_auc"])


def test_pooled_skill_drops_nan_predictions():
    y = np.array([0, 1, 0, 1] * 50)
    p = np.where(np.arange(200) % 4 == 0, np.nan, np.linspace(0, 1, 200))
    assert fg.pooled_skill(y, p)["n"] == 150


def test_abundance_fidelity_reports_both_halves_of_gate6():
    rng = np.random.default_rng(1)
    fa = np.clip(rng.exponential(0.01, 5000), 0, 0.29)
    fa[rng.random(5000) < 0.18] = 0.0
    res = fg.abundance_fidelity(fa, fa * 0.9)
    assert "marginal_l1" in res and "rich_bin_rmse" in res and "spearman" in res
    assert "low_over" not in res, "low_over is degenerate on this dataset and must not be reported"
    assert res["top_ratio"] == pytest.approx(0.9, abs=0.02)
    assert res["passes_top_ratio"] is True


def test_abundance_fidelity_uses_a_common_finite_mask():
    fa = np.concatenate([np.linspace(0, 0.2, 500), [np.nan] * 10])
    pred = np.concatenate([np.linspace(0, 0.2, 500), [0.1] * 10])
    res = fg.abundance_fidelity(fa, pred)
    assert res["n"] == 500 and np.isfinite(res["marginal_l1"])


def test_abundance_fidelity_catches_a_squashed_tail():
    rng = np.random.default_rng(2)
    fa = np.clip(rng.exponential(0.01, 5000), 0, 0.29)
    res = fg.abundance_fidelity(fa, fa * 0.4)
    assert res["top_ratio"] < 0.8 and res["passes_top_ratio"] is False
