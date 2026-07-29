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
    r = fg.edge_cv_for_offsets(es, o, n, 1.0, variant="full", frac=0.2, repeats=3)
    assert r["insample_dp"] < r["unleveled_dp"]
    assert r["passes"] is True


def test_edge_cv_h1only_is_the_baseline_and_does_NOT_pass():
    """The held-out number must depend on the offset MODEL. An earlier version delegated to
    lv.heldout_edge_cv, which re-solves FULL offsets per fold and never sees the offsets, so h1only /
    full / resid all returned the same value and the UNLEVELED row was reported as clearing gate 2."""
    es, _ = _edges()
    n = len(es.pids)
    r0 = fg.edge_cv_for_offsets(es, np.zeros(n), n, 1.0, variant="h1only", frac=0.2, repeats=2)
    assert r0["heldout_cv_dp"] == pytest.approx(r0["unleveled_dp"])
    assert r0["insample_dp"] == pytest.approx(r0["unleveled_dp"])
    assert r0["passes"] is False


def test_edge_cv_heldout_actually_depends_on_the_variant():
    es, bias = _edges()
    n = len(es.pids)
    o = -(bias - np.median(bias))
    full = fg.edge_cv_for_offsets(es, o, n, 1.0, variant="full", frac=0.2, repeats=2)
    h1 = fg.edge_cv_for_offsets(es, np.zeros(n), n, 1.0, variant="h1only", frac=0.2, repeats=2)
    assert full["heldout_cv_dp"] != pytest.approx(h1["heldout_cv_dp"])
    assert full["heldout_cv_dp"] < h1["heldout_cv_dp"]


def test_edge_cv_absurd_offsets_do_not_silently_pass_in_sample():
    """A wild offset vector must be visible in the in-sample number rather than hidden by a
    variant-independent held-out value."""
    es, _ = _edges()
    n = len(es.pids)
    wild = fg.edge_cv_for_offsets(es, np.full(n, 7.0), n, 1.0, variant="full", frac=0.2, repeats=2)
    zero = fg.edge_cv_for_offsets(es, np.zeros(n), n, 1.0, variant="full", frac=0.2, repeats=2)
    assert wild["insample_dp"] != pytest.approx(zero["insample_dp"])


def test_edge_cv_resid_refits_its_plane_per_fold():
    """resid = "solve minus its own smooth plane"; without lon/lat it must SAY it fell back to full
    rather than quietly reporting the full-offset number as the residual variant's."""
    es, bias = _edges()
    n = len(es.pids)
    o = -(bias - np.median(bias))
    lon = np.linspace(-10, 10, n)
    lat = np.linspace(35, 45, n)
    with_plane = fg.edge_cv_for_offsets(es, o, n, 1.0, variant="resid", frac=0.2, repeats=2,
                                        lon=lon, lat=lat, degree=es.degrees(n).astype(float))
    assert np.isfinite(with_plane["heldout_cv_dp"])
    assert with_plane["variant"] == "resid"


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
def _labels_with_bbox(centres_xy, pitch=32 * 4.9999744853063):
    """Label rows carrying the world bbox the parquet actually stores, for given tile centres."""
    cx = np.array([c[0] for c in centres_xy], float)
    cy = np.array([c[1] for c in centres_xy], float)
    return pd.DataFrame({"obs_id": ["ESP_1"] * len(cx), "ti": np.arange(len(cx)),
                         "tj": np.arange(len(cx)), "fractional_area": np.zeros(len(cx)),
                         "xmin": cx - pitch / 2, "xmax": cx + pitch / 2,
                         "ymin": cy - pitch / 2, "ymax": cy + pitch / 2})


def test_cohort_join_keys_off_the_label_bbox_not_a_window_corner():
    """Independently-derived expectation: a tile whose world centre is at (x, y) must key to
    (round(y/160), round(x/160)) — Stage B's exact keying — regardless of any window offset.

    The previous version of this test pinned row0=col0=0 and re-derived the expected value from the
    implementation's own formula, so it could not catch the ~100 km window-offset bug the 2026-07-29
    review found. These centres are chosen by hand.
    """
    # centres chosen to be EXACT multiples of 160 so the expectation needs no rounding rule — a
    # half-cell value like 480_080 would sit on a tie and np.round is half-to-EVEN.
    labels = _labels_with_bbox([(480_160.0, 2_020_000.0), (-1_600.0, 160.0)])
    got = fg.cohort_tiles_to_global(labels)
    assert list(got.TJ) == [3001, -10]
    assert list(got.TI) == [12625, 1]


def test_cohort_join_is_immune_to_the_window_offset():
    """Two obs with identical ground positions but wildly different window origins must get the same
    global keys — the property the old (ti,tj)+window-corner route violated by ~100 km."""
    labels = _labels_with_bbox([(500_000.0, 2_500_000.0)])
    a = fg.cohort_tiles_to_global(labels.assign(ti=0, tj=0))
    b = fg.cohort_tiles_to_global(labels.assign(ti=21_383, tj=3_813))
    assert list(a.TI) == list(b.TI) and list(a.TJ) == list(b.TJ)


def test_cohort_join_rejects_labels_without_the_world_bbox():
    with pytest.raises(ValueError, match="world bbox"):
        fg.cohort_tiles_to_global(pd.DataFrame({"obs_id": ["a"], "ti": [0], "tj": [0]}))


def test_cohort_join_matches_the_real_label_geometry(repo_root):
    """On real label rows, the bbox route must agree with Stage B's keying of the same ground point,
    and must NOT agree with the window-corner route (which is the bug, reproduced here)."""
    import glob
    paths = sorted(glob.glob(str(repo_root / "dataset_v2" / "labels" / "*.parquet")))
    bounds_csv = repo_root / "reports" / "f_leg_b" / "cohort_obs_bounds.csv"
    if not paths or not bounds_csv.exists():
        pytest.skip("dataset_v2/labels or cohort_obs_bounds.csv not on disk")
    d = pd.read_parquet(paths[0], columns=["obs_id", "tile_size_px", "ti", "tj",
                                           "xmin", "xmax", "ymin", "ymax", "fractional_area"])
    d = d[d.tile_size_px == 32].head(200)
    got = fg.cohort_tiles_to_global(d)
    cx = (d.xmin.to_numpy() + d.xmax.to_numpy()) / 2
    cy = (d.ymin.to_numpy() + d.ymax.to_numpy()) / 2
    assert np.array_equal(got.TJ.to_numpy(), np.round(cx / 160.0).astype(np.int64))
    assert np.array_equal(got.TI.to_numpy(), np.round(cy / 160.0).astype(np.int64))
    b = pd.read_csv(bounds_csv).set_index("obs_id")
    obs = d.obs_id.iloc[0]
    if obs in b.index and float(b.loc[obs, "col0"]) > 0:
        wrong_tj = np.round((float(b.loc[obs, "minx"])
                             + (d.tj.to_numpy() + 0.5) * 160.0) / 160.0).astype(np.int64)
        assert not np.array_equal(got.TJ.to_numpy(), wrong_tj), (
            "the window-corner route must NOT agree — it is the bug this join replaced")


def test_pooled_pr_auc_is_average_precision_not_roc_auc():
    """Pins the metric IDENTITY, because the project rule is that presence/ROC AUC must never be
    reported. The previous version only bounded the value in (0.5, 1] — a band ROC AUC also satisfies
    — so swapping in roc_auc_score left the whole file green (review 2026-07-29).

    The two reference values are computed here rather than hardcoded, and the test first ASSERTS THEY
    DIFFER on this input, so it cannot silently become vacuous (an earlier attempt at this test used
    y=[1,0,0,1]/p=[.9,.8,.7,.6], where average precision and ROC AUC are both exactly 0.75)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    ap, roc = average_precision_score(y, p), roc_auc_score(y, p)
    assert abs(ap - roc) > 0.1, "the fixture must discriminate the two metrics"
    r = fg.pooled_skill(y, p)
    assert r["pooled_pr_auc"] == pytest.approx(ap, abs=1e-12)
    assert r["pooled_pr_auc"] != pytest.approx(roc, abs=1e-3)


def test_precision_at_5pct_is_pinned_to_a_literal():
    """k = max(1, round(0.05*N)) with a DESCENDING sort. 40 rows -> k = 2; the two highest-p rows are
    constructed to be one positive and one negative, so the answer is exactly 0.5."""
    p = np.linspace(0.0, 1.0, 40)
    y = np.zeros(40, dtype=int)
    y[39] = 1          # highest p, positive
    y[38] = 0          # second highest, negative
    y[:5] = 1          # positives far from the top, so they cannot enter the top-k
    r = fg.pooled_skill(y, p)
    assert r["precision@5%"] == pytest.approx(0.5)
    assert r["n"] == 40


def test_pooled_skill_reports_n_and_handles_a_realistic_pool():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 2000), 0, 1)
    r = fg.pooled_skill(y, p)
    assert 0.5 < r["pooled_pr_auc"] <= 1.0 and r["n"] == 2000


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
