"""Tests for `src/leveling.py` — the PLAN_FBuild Stage C (H4 build-scale) solver.

Stage C runs ONCE on data that costs a Sherlock day to produce, so the whole solver is exercised
here on synthetic frames with PLANTED per-frame biases (where the right answer is known exactly)
plus a reference check against the frozen pilot normal equations in `scripts/f_h4_level.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import leveling as lv


# --------------------------------------------------------------------------- fixtures
def synth_frames(n_frames=8, side=40, overlap=15, bias=None, seed=0, noise=0.05):
    """Overlapping frames on the global tile grid, each an offset view of one shared logit field.

    Frame f covers a `side`-wide column band stepping by (side − overlap), so consecutive frames
    share `overlap` columns and the graph is a path. Observed logit = truth + bias_f + noise, so the
    correct leveling offsets are −bias (up to the median gauge).
    """
    rng = np.random.default_rng(seed)
    bias = np.linspace(-0.8, 0.8, n_frames) if bias is None else np.asarray(bias, float)
    rows = np.arange(60)
    truth = {}
    pids, keys, logits = [], [], []
    for f in range(n_frames):
        c0 = f * (side - overlap)
        cols = np.arange(c0, c0 + side)
        ti, tj = np.meshgrid(rows, cols, indexing="ij")
        ti, tj = ti.ravel(), tj.ravel()
        key = lv.pack_key(ti, tj)
        for k in key.tolist():
            truth.setdefault(k, rng.normal(0.0, 1.2))
        lg = np.array([truth[k] for k in key.tolist()]) + bias[f] + rng.normal(0, noise, key.size)
        order = np.argsort(key)
        pids.append(f"FRAME_{f:02d}")
        keys.append(key[order])
        logits.append(lg[order].astype(np.float32))
    return pids, keys, logits, bias


def pilot_reference_solve(edges, lam, n):
    """The frozen pilot normal equations, verbatim from scripts/f_h4_level.solve_offsets.

    Kept as an independent copy (importing the pilot script would drag in torch/rasterio) so a
    refactor of the build solver that silently changes the SIGN or weighting convention fails here.
    """
    ata = np.zeros((n, n))
    atb = np.zeros(n)
    for i, j, dbar, w in edges:
        ata[i, i] += w
        ata[j, j] += w
        ata[i, j] -= w
        ata[j, i] -= w
        atb[i] += w * dbar
        atb[j] -= w * dbar
    ata += lam * np.eye(n)
    o = np.linalg.lstsq(ata, atb, rcond=None)[0]
    return o - np.median(o)


# --------------------------------------------------------------------------- keys / geometry
def test_key_roundtrip_handles_negative_tile_indices():
    ti = np.array([-33_000, -1, 0, 1, 33_000])
    tj = np.array([-66_000, -1, 0, 1, 66_000])
    ti2, tj2 = lv.unpack_key(lv.pack_key(ti, tj))
    assert np.array_equal(ti, ti2) and np.array_equal(tj, tj2)


def test_key_is_monotone_so_stage_b_output_is_already_sorted():
    ti = np.array([0, 0, 1, 1])
    tj = np.array([-5, 3, -9, 2])
    assert np.all(np.diff(lv.pack_key(ti, tj)) > 0)


def test_pack_key_rejects_aliasing_column_index():
    with pytest.raises(ValueError):
        lv.pack_key([0], [lv.KEY_SHIFT])


def test_intersect_sorted_matches_intersect1d_both_orders():
    rng = np.random.default_rng(3)
    a = np.unique(rng.integers(0, 500, 200)).astype(np.int64)
    b = np.unique(rng.integers(0, 500, 300)).astype(np.int64)
    ia, ib = lv.intersect_sorted(a, b)
    assert np.array_equal(a[ia], np.intersect1d(a, b))
    assert np.array_equal(a[ia], b[ib])
    ib2, ia2 = lv.intersect_sorted(b, a)          # the size-swap branch
    assert np.array_equal(b[ib2], np.intersect1d(a, b)) and np.array_equal(a[ia2], b[ib2])


# --------------------------------------------------------------------------- edges
def test_build_edges_recovers_exact_shared_tiles_and_delta():
    keys = [lv.pack_key(np.zeros(10, int), np.arange(10)),
            lv.pack_key(np.zeros(10, int), np.arange(6, 16))]
    lg = [np.zeros(10, np.float32), np.full(10, 0.5, np.float32)]
    es = lv.build_edges(["A", "B"], keys, lg, min_tiles=1, dp_sample=100)
    assert es.n_edges == 1
    assert es.w[0] == 4                                        # columns 6..9
    assert es.dbar[0] == pytest.approx(0.5)                    # mean(ℓ_B − ℓ_A)


def test_build_edges_drops_pairs_below_min_tiles():
    pids, keys, logits, _ = synth_frames(n_frames=4, side=30, overlap=5)
    lots = lv.build_edges(pids, keys, logits, min_tiles=1)
    few = lv.build_edges(pids, keys, logits, min_tiles=10_000)
    assert lots.n_edges == 3 and few.n_edges == 0


def test_candidate_pairs_finds_every_real_overlap():
    pids, keys, logits, _ = synth_frames(n_frames=6)
    pairs = lv.candidate_pairs(keys, cell_tiles=8)
    real = {(e_i, e_j) for e_i, e_j in
            zip(*[a.tolist() for a in (lv.build_edges(pids, keys, logits, min_tiles=1).ei,
                                       lv.build_edges(pids, keys, logits, min_tiles=1).ej)])}
    assert real <= set(pairs)


def test_edgeset_filter_keeps_samples_aligned():
    pids, keys, logits, _ = synth_frames(n_frames=5)
    es = lv.build_edges(pids, keys, logits, min_tiles=1, dp_sample=50)
    sub = es.filter(np.array([1, 3]))
    assert sub.n_edges == 2
    for new, old in enumerate([1, 3]):
        s0, s1 = sub.samp_off[new], sub.samp_off[new + 1]
        o0, o1 = es.samp_off[old], es.samp_off[old + 1]
        assert np.array_equal(sub.samp_i[s0:s1], es.samp_i[o0:o1])


# --------------------------------------------------------------------------- the solve
def test_solver_recovers_planted_per_frame_bias():
    pids, keys, logits, bias = synth_frames(n_frames=8, noise=0.02)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    o = lv.solve_offsets(es, 0.0, len(pids))
    expect = -(bias - np.median(bias))                          # gauge: median(o) = 0
    assert np.allclose(o, expect, atol=0.02)


def test_leveling_reduces_colocated_disagreement():
    pids, keys, logits, _ = synth_frames(n_frames=8)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    before = np.median(lv.edge_dp(es, np.zeros(n)))
    after = np.median(lv.edge_dp(es, lv.solve_offsets(es, 0.0, n)))
    assert after < 0.25 * before


def test_solver_matches_the_frozen_pilot_normal_equations():
    pids, keys, logits, _ = synth_frames(n_frames=6)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    ref_edges = list(zip(es.ei.tolist(), es.ej.tolist(), es.dbar.tolist(), es.w.tolist()))
    for lam in (0.0, 10.0, 300.0):
        assert np.allclose(lv.solve_offsets(es, lam, len(pids)),
                           pilot_reference_solve(ref_edges, lam, len(pids)), atol=1e-8)


def test_tikhonov_shrinks_offsets_monotonically():
    pids, keys, logits, _ = synth_frames(n_frames=8)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    sds = [np.std(lv.solve_offsets(es, lam, len(pids)))
           for lam in (0.0, 1e2, 1e4, 1e6)]
    assert all(a >= b - 1e-12 for a, b in zip(sds, sds[1:]))


def test_disconnected_components_get_independent_gauges():
    """Two disjoint clusters: each must be centred on its own median, not blended."""
    pids_a, keys_a, log_a, _ = synth_frames(n_frames=3, bias=[0.0, 0.4, 0.8], seed=1)
    pids_b, keys_b, log_b, _ = synth_frames(n_frames=3, bias=[3.0, 3.4, 3.8], seed=2)
    shift = lv.pack_key(np.array([500]), np.array([500]))[0]     # move cluster B far away
    keys_b = [k + shift for k in keys_b]
    pids = [f"A{i}" for i in range(3)] + [f"B{i}" for i in range(3)]
    es = lv.build_edges(pids, keys_a + keys_b, log_a + log_b, min_tiles=1)
    comp = lv.components(es.ei, es.ej, 6)
    assert np.unique(comp).size == 2
    o = lv.solve_offsets(es, 1.0, 6, comp=comp)
    assert np.median(o[comp == comp[0]]) == pytest.approx(0.0, abs=1e-9)
    assert np.median(o[comp == comp[5]]) == pytest.approx(0.0, abs=1e-9)


def test_heldout_edge_cv_beats_unleveled_on_a_redundant_graph():
    # overlap 28 of 40 => each frame links to f±1, f±2, f±3, so a 20% edge holdout leaves the
    # graph connected (the build graph's median degree is 7 — CV is only meaningful with cycles).
    pids, keys, logits, _ = synth_frames(n_frames=12, side=40, overlap=28)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    unleveled = float(np.median(lv.edge_dp(es, np.zeros(n))))
    cv, skipped = lv.heldout_edge_cv(es, 1.0, n, frac=0.2, repeats=3, seed=0)
    assert cv < unleveled
    assert skipped == 0


def test_heldout_edge_cv_skips_rather_than_scoring_across_a_broken_gauge():
    """On a path graph every dropped edge splits the gauge — report NaN + the skip count, never a
    number computed across two components (that would silently invent a comparison)."""
    pids, keys, logits, _ = synth_frames(n_frames=6, side=40, overlap=20)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    cv, skipped = lv.heldout_edge_cv(es, 1.0, len(pids), frac=0.2, repeats=2, seed=0)
    assert np.isnan(cv) and skipped == 2


def test_lofo_predicts_a_held_out_frames_offset():
    pids, keys, logits, _ = synth_frames(n_frames=10, side=40, overlap=25, noise=0.02)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    o_full = lv.solve_offsets(es, 1.0, n)
    o_hat, err, used = lv.lofo_offsets(es, o_full, 1.0, n)
    interior = np.arange(1, n - 1)                 # ends have a single edge and no cross-check
    assert np.nanmedian(err[interior]) < 0.15
    assert (used[interior] >= 2).all()


# --------------------------------------------------------------------------- graph holes
def test_idw_predicts_from_the_nearest_known_frames():
    lon = np.array([0.0, 1.0, 2.0, 10.0])
    lat = np.zeros(4)
    o = np.array([1.0, 1.0, 1.0, np.nan])
    known = np.array([True, True, True, False])
    pred = lv.idw_predict(lon, lat, o, known, ~known)
    assert pred[3] == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(pred[0])


def test_patch_graph_holes_flags_isolated_and_regauges_side_components():
    lon = np.array([0.0, 0.2, 0.4, 5.0, 5.2, 20.0])
    lat = np.zeros(6)
    comp = np.array([0, 0, 0, 1, 1, 2])
    deg = np.array([2, 2, 2, 1, 1, 0])
    o = np.array([-0.5, 0.0, 0.5, -0.3, 0.3, 0.0])        # component 1 carries its own free gauge
    patched, src = lv.patch_graph_holes(o, comp, deg, lon, lat)
    assert list(src) == ["solved"] * 3 + ["component_gauged"] * 2 + ["interpolated"]
    assert patched[4] - patched[3] == pytest.approx(o[4] - o[3])   # internal structure preserved
    assert np.allclose(patched[:3], o[:3])                         # main component untouched


def test_patch_graph_holes_is_a_noop_on_a_single_component():
    o = np.array([-0.2, 0.0, 0.2])
    patched, src = lv.patch_graph_holes(o, np.zeros(3, int), np.array([1, 2, 1]),
                                        np.array([0.0, 1.0, 2.0]), np.zeros(3))
    assert np.allclose(patched, o) and set(src) == {"solved"}


# --------------------------------------------------------------------------- trend guard
def test_block_permutation_detects_a_planted_gradient():
    rng = np.random.default_rng(0)
    lon = rng.uniform(-12, 12, 200)
    lat = rng.uniform(32, 48, 200)
    trend = lv.trend_significance(0.1 * lon + rng.normal(0, 0.1, 200), lon, lat, n_draws=200, seed=1)
    assert trend["r2"] > 0.8 and trend["p_value"] < 0.05


def test_block_permutation_does_not_fire_on_spatial_noise():
    rng = np.random.default_rng(4)
    lon = rng.uniform(-12, 12, 200)
    lat = rng.uniform(32, 48, 200)
    trend = lv.trend_significance(rng.normal(0, 1, 200), lon, lat, n_draws=200, seed=2)
    assert trend["p_value"] > 0.05


def test_block_permute_preserves_the_value_multiset_within_reason():
    rng = np.random.default_rng(0)
    o = np.arange(20).astype(float)
    blocks = np.repeat(np.arange(4), 5)
    out = lv.block_permute(o, blocks, rng)
    assert out.shape == o.shape and set(out.tolist()) <= set(o.tolist())


def test_weighted_fit_r2_is_invariant_to_coordinate_scaling():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=50), rng.normal(size=50)
    z = 0.3 * x - 0.2 * y + rng.normal(0, 0.1, 50)
    r2a = lv.weighted_fit(lv.design_matrix(x, y), z)[2]
    r2b = lv.weighted_fit(lv.design_matrix(x * 1e5, y * 1e5 + 7), z)[2]
    assert r2a == pytest.approx(r2b, abs=1e-9)


# --------------------------------------------------------------------------- the §4.3 rule table
def _trend(p=0.001, r2=0.4):
    return {"p_value": p, "r2": r2}


def test_verdict_no_trend_when_surface_insignificant():
    v = lv.trend_verdict(_trend(p=0.4), {"r2": 0.9, "p_value": 0.001}, {"r2": 0.9, "p_value": 0.001})
    assert v["verdict"] == "NO_TREND" and v["apply"] == "full" and not v["needs_ruling"]


def test_verdict_geology_dominant_forces_residual_only():
    v = lv.trend_verdict(_trend(), {"r2": 0.10, "p_value": 0.30}, {"r2": 0.70, "p_value": 0.004})
    assert v["verdict"] == "RESIDUAL_ONLY" and v["apply"] == "residual"


def test_verdict_metadata_dominant_applies_full_offsets():
    v = lv.trend_verdict(_trend(), {"r2": 0.70, "p_value": 0.004}, {"r2": 0.10, "p_value": 0.30})
    assert v["verdict"] == "FULL" and v["apply"] == "full"


def test_verdict_ambiguous_never_silently_applies_full_offsets():
    """§0.1 HARD-ABORT guard 1: an ambiguous attribution must escalate, not default."""
    v = lv.trend_verdict(_trend(), {"r2": 0.50, "p_value": 0.01}, {"r2": 0.52, "p_value": 0.01})
    assert v["verdict"] == "AMBIGUOUS" and v["needs_ruling"] and v["apply"] != "full"


def test_verdict_handles_missing_geology_proxies():
    """NaN R² = the proxy rasters are absent, which is 'untestable', not 'geology lost'."""
    v = lv.trend_verdict(_trend(), {"r2": 0.6, "p_value": 0.002},
                         {"r2": float("nan"), "p_value": float("nan")})
    assert v["verdict"] == "FULL"


def test_verdict_missing_metadata_axes_still_honours_guard_1():
    v = lv.trend_verdict(_trend(), {"r2": float("nan"), "p_value": float("nan")},
                         {"r2": 0.6, "p_value": 0.002})
    assert v["verdict"] == "RESIDUAL_ONLY" and v["apply"] == "residual"


def test_verdict_with_no_attribution_axes_at_all_escalates():
    nan = {"r2": float("nan"), "p_value": float("nan")}
    v = lv.trend_verdict(_trend(), dict(nan), dict(nan))
    assert v["verdict"] == "AMBIGUOUS" and v["needs_ruling"] and v["apply"] != "full"


def test_verdict_requires_the_margin_even_when_the_loser_is_insignificant():
    """The 906-frame run, verbatim: metadata cleared α but held the LOWER R².

    Before 2026-07-29 the rule fired on `not g_sig`, returning FULL on metadata R²=0.108 vs geology
    R²=0.142 because geology's permutation p landed at 0.0579 rather than ≤0.05 — 8 draws in 1000,
    and 19-FULL/1-AMBIGUOUS across 20 seeds. PLAN_FBuild §4.3's table always required the winner to
    beat the other by the margin; this pins that reading so the verdict cannot ride on RNG noise.
    """
    v = lv.trend_verdict(_trend(r2=0.957), {"r2": 0.108, "p_value": 0.019},
                         {"r2": 0.142, "p_value": 0.058})
    assert v["verdict"] == "AMBIGUOUS" and v["needs_ruling"] and v["apply"] != "full"


def test_verdict_margin_rule_is_symmetric_for_geology():
    """Mirror image: geology significant, metadata not, but geology does not clear the margin."""
    v = lv.trend_verdict(_trend(), {"r2": 0.142, "p_value": 0.30},
                         {"r2": 0.150, "p_value": 0.004})
    assert v["verdict"] == "AMBIGUOUS" and v["needs_ruling"]


# ------------------------------------------------- the saturation-immune edge metric (2026-07-29)
def test_edge_dlogit_is_immune_to_the_saturation_that_collapses_edge_dp():
    """A large COMMON offset rails every probability: |Δp| → 0 while the real disagreement is intact.

    This is the 906-frame pathology in miniature — it is why |Δp| drove λ* to 0 with |o|max 21.3
    logits, and why λ is selected on |Δlogit| instead.
    """
    pids, keys, logits, _ = synth_frames(n_frames=6, side=40, overlap=25)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    zero, railed = np.zeros(n), np.full(n, 12.0)          # +12 logits saturates p to ~1

    dp0 = float(np.median(lv.edge_dp(es, zero)))
    dp1 = float(np.median(lv.edge_dp(es, railed)))
    dl0 = float(np.median(lv.edge_dlogit(es, zero)))
    dl1 = float(np.median(lv.edge_dlogit(es, railed)))

    assert dp1 < 0.02 * dp0                                # |Δp| collapses...
    assert dl1 == pytest.approx(dl0, rel=1e-9)             # ...while |Δlogit| is unchanged
    assert lv.edge_saturated_frac(es, railed) > 0.95
    assert lv.edge_saturated_frac(es, zero) < 0.10


def test_edge_dlogit_still_rewards_a_correct_solve():
    """Saturation-immunity must not cost sensitivity: the planted bias is still recovered as a win."""
    pids, keys, logits, _ = synth_frames(n_frames=8)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    before = float(np.median(lv.edge_dlogit(es, np.zeros(n))))
    after = float(np.median(lv.edge_dlogit(es, lv.solve_offsets(es, 0.0, n))))
    assert after < 0.5 * before


def test_heldout_edge_cv_accepts_the_dlogit_metric():
    pids, keys, logits, _ = synth_frames(n_frames=12, side=40, overlap=28)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    unleveled = float(np.median(lv.edge_dlogit(es, np.zeros(n))))
    cv, skipped = lv.heldout_edge_cv(es, 1.0, n, frac=0.2, repeats=3, seed=0, metric="dlogit")
    assert skipped == 0 and cv < unleveled
    # and the two metrics are genuinely different instruments on the same folds
    cv_dp, _ = lv.heldout_edge_cv(es, 1.0, n, frac=0.2, repeats=3, seed=0, metric="dp")
    assert not np.isclose(cv, cv_dp)


def test_edge_metric_registry_matches_the_functions():
    assert lv.EDGE_METRICS["dp"] is lv.edge_dp
    assert lv.EDGE_METRICS["dlogit"] is lv.edge_dlogit


# ------------------------------------------- the plane-free (constrained) solve, 2026-07-30
def test_plane_complement_is_orthonormal_and_kills_the_plane():
    rng = np.random.default_rng(0)
    lon, lat = rng.normal(size=40), rng.normal(size=40)
    z = lv.plane_complement(lon, lat)
    assert z.shape == (40, 37)
    assert np.allclose(z.T @ z, np.eye(37), atol=1e-10)          # orthonormal
    for v in (np.ones(40), lon, lat):                            # spans nothing of the plane
        assert np.allclose(z.T @ v, 0.0, atol=1e-9)


def test_plane_complement_is_rank_aware_on_a_degenerate_layout():
    """All frames at one latitude => span{1,lon,lat} is rank 2, so only 2 directions may be removed.

    A plain QR would hand back an arbitrary third basis column here and the constrained solve would
    silently delete a direction carrying real signal.
    """
    lon = np.arange(12, dtype=float)
    z = lv.plane_complement(lon, np.zeros(12))
    assert z.shape == (12, 10)                                   # n - 2, NOT n - 3
    assert np.allclose(z.T @ z, np.eye(10), atol=1e-10)
    for v in (np.ones(12), lon):
        assert np.allclose(z.T @ v, 0.0, atol=1e-9)


def _plane_free_bias(lon, lat, seed=0, scale=0.6):
    """A per-frame bias living ENTIRELY in the plane-free subspace (no constant/lon/lat component).

    Needed because `synth_frames`' default bias is `linspace(-0.8, 0.8)` — a PURE ramp — against
    which the constrained solve is supposed to lose. Separating "local structure" from "region-wide
    plane" by construction is what makes these assertions about the method rather than the fixture.
    """
    z = lv.plane_complement(lon, lat)
    b = z @ np.random.default_rng(seed).normal(size=z.shape[1])
    return scale * b / np.abs(b).max()


def _ssr(es, o):
    r = (o[es.ei] - o[es.ej]) - es.dbar
    return float((es.w * r ** 2).sum())


def test_planefree_recovers_local_structure_when_there_is_no_real_plane():
    """The regime the build is actually in: no constant region-wide gradient exists to estimate
    (measured b = +0.203 / -0.003 / +0.433 per lon tercile). There, constraining costs nothing and
    the local per-frame corrections come back intact."""
    n_frames = 12
    lon, lat = np.arange(n_frames, dtype=float), np.zeros(n_frames)
    local = _plane_free_bias(lon, lat)
    pids, keys, logits, _ = synth_frames(n_frames=n_frames, side=40, overlap=25,
                                         bias=local, noise=0.02)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    pfree = lv.solve_offsets_planefree(es, 0.0, n_frames, lon, lat)
    assert np.corrcoef(-local, pfree)[0, 1] > 0.98


def test_planefree_removes_a_planted_ramp_and_distorts_local_estimates_doing_so():
    """Plant a GENUINE ramp: the constraint kills it, but at a documented cost.

    The constrained solve does not merely delete the plane — it spends its remaining (plane-free)
    freedom flattening the ramp's local differences, so the recovered local pattern degrades
    (corr ~0.6 here vs >0.98 with no ramp planted). This is the real caveat of the method: if a
    region-wide gradient truly exists, this solve both discards it AND biases the local corrections.
    Acceptable for the build only because the measured per-step term is patchy rather than a constant
    gradient (DECISIONS 2026-07-30); it would NOT be acceptable in a region where b is constant.
    """
    n_frames = 12
    lon, lat = np.arange(n_frames, dtype=float), np.zeros(n_frames)
    local = _plane_free_bias(lon, lat)
    pids, keys, logits, _ = synth_frames(n_frames=n_frames, side=40, overlap=25,
                                         bias=local + 0.8 * lon, noise=0.02)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    free = lv.solve_offsets(es, 0.0, n_frames)
    pfree = lv.solve_offsets_planefree(es, 0.0, n_frames, lon, lat)

    A = lv.design_matrix(lon, lat, order=1)
    assert abs(lv.weighted_fit(A, free, None)[0][1]) > 0.5        # free solve carries the ramp
    assert abs(lv.weighted_fit(A, pfree, None)[0][1]) < 1e-6      # constrained one does not
    r = np.corrcoef(-local, pfree)[0, 1]
    assert 0.4 < r < 0.95, f"expected degraded-but-positive local recovery, got {r:.3f}"


def test_planefree_costs_almost_nothing_when_the_truth_has_no_plane():
    """If the real bias has no region-wide component, constraining it away is nearly free."""
    n = 12
    lon, lat = np.arange(n, dtype=float), np.zeros(n)
    pids, keys, logits, _ = synth_frames(n_frames=n, side=40, overlap=28,
                                         bias=_plane_free_bias(lon, lat), noise=0.02)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    free = _ssr(es, lv.solve_offsets(es, 0.0, n))
    pfree = _ssr(es, lv.solve_offsets_planefree(es, 0.0, n, lon, lat))
    assert pfree >= free - 1e-9                                   # never better than unconstrained
    assert pfree < 1.15 * max(free, 1e-12)                        # ...but within 15% here


def test_planefree_is_never_a_better_fit_than_the_free_solve():
    """The general guarantee, even when the planted truth IS a pure ramp (where it loses badly)."""
    pids, keys, logits, _ = synth_frames(n_frames=12, side=40, overlap=28)   # default bias = a ramp
    n = len(pids)
    lon, lat = np.arange(n, dtype=float), np.zeros(n)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    free = _ssr(es, lv.solve_offsets(es, 0.0, n))
    pfree = _ssr(es, lv.solve_offsets_planefree(es, 0.0, n, lon, lat))
    assert pfree >= free - 1e-9
    # and it SHOULD lose here — documents the method's caveat: a genuine region-wide plane is lost
    assert pfree > 10 * free


def test_planefree_beats_post_hoc_detrending_on_its_own_objective():
    """The build's finding: constrain-then-solve dominates solve-then-subtract.

    This is a theorem, not a coincidence — `posthoc` is one member of the plane-free subspace and
    `pfree` is that subspace's minimiser — so it pins the ORDER of the two operations.
    """
    n = 12
    lon, lat = np.arange(n, dtype=float), np.zeros(n)
    pids, keys, logits, _ = synth_frames(n_frames=n, side=40, overlap=28,
                                         bias=_plane_free_bias(lon, lat) + 0.6 * lon, noise=0.05)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    free = lv.solve_offsets(es, 0.0, n)
    A = lv.design_matrix(lon, lat, order=1)
    posthoc = lv.regauge(free - lv.weighted_fit(A, free, None)[1])
    pfree = lv.solve_offsets_planefree(es, 0.0, n, lon, lat)
    assert _ssr(es, pfree) <= _ssr(es, posthoc) + 1e-9


def test_normal_equations_still_reproduce_the_pilot_solver():
    """The refactor that introduced normal_equations() must not change solve_offsets' answer."""
    pids, keys, logits, _ = synth_frames(n_frames=6)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    ref = list(zip(es.ei.tolist(), es.ej.tolist(), es.dbar.tolist(), es.w.tolist()))
    for lam in (0.0, 10.0, 300.0):
        assert np.allclose(lv.solve_offsets(es, lam, len(pids)),
                           pilot_reference_solve(ref, lam, len(pids)), atol=1e-8)


# ------------------------------------------------------- the lean guards (Brian, 2026-07-29)
def test_benefit_concentration_flags_a_gain_bought_from_few_edges():
    pids, keys, logits, _ = synth_frames(n_frames=10, side=40, overlap=25)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    good = lv.solve_offsets(es, 0.0, n)
    rep = lv.benefit_concentration(es, good, np.zeros(n))
    # a genuine, broad improvement: most edges helped, no single edge dominating
    assert rep["frac_edges_worse"] < 0.5
    assert rep["gain_share_top_1pct"] < 0.9
    assert rep["n_edges"] == es.n_edges


def test_benefit_concentration_sign_convention_is_new_vs_reference():
    """gain > 0 must mean `o` fits BETTER than `o_ref` — a flipped sign would invert the guard."""
    pids, keys, logits, _ = synth_frames(n_frames=8)
    es = lv.build_edges(pids, keys, logits, min_tiles=1)
    n = len(pids)
    solved = lv.solve_offsets(es, 0.0, n)
    assert lv.benefit_concentration(es, solved, np.zeros(n))["total_gain"] > 0
    assert lv.benefit_concentration(es, np.zeros(n), solved)["total_gain"] < 0


def test_offset_magnitude_report_uses_the_measured_yardstick():
    spread = lv.frame_level_spread(np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95]))
    assert spread > 0
    small = lv.offset_magnitude_report(np.array([0.1, -0.2, 0.3]), spread)
    assert small["within_frame_spread"] and small["n_over_frame_spread"] == 0
    huge = lv.offset_magnitude_report(np.array([0.1, 21.3, -11.6]), spread)
    assert not huge["within_frame_spread"]
    assert huge["n_over_frame_spread"] == 2
    assert huge["n_over_logit_clip"] == 2            # both exceed the +-9.21 per-tile clip
    assert huge["max_over_frame_spread"] > 1.0


def test_frame_level_spread_is_nan_on_degenerate_input():
    assert np.isnan(lv.frame_level_spread(np.array([0.5])))
    assert np.isnan(lv.frame_level_spread(np.array([np.nan, np.nan])))
