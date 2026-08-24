"""Unit tests for the productized MLP head + deployable ensemble (PLAN_FM §2.6.A).

Synthetic data only (CPU, no checkpoint / dataset). A linearly-separable-ish
embedding cloud lets the tiny MLP learn something so the contracts (shape,
probability range, scaler parity, save/load round-trip, NaN-row imputation) are
exercised on a non-degenerate fit -- without asserting any accuracy gate (that
is the LOIO harness's job, and the recipe is frozen).
"""
import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy/torch

import numpy as np
import pytest

from src.modeling.mlp_head import (
    DeployableHead, FeatureScaler, MLPClassifierHead, build_mlp,
)

EMBED_DIM = 32  # small stand-in for 768 to keep the tests fast


def _synthetic(n=600, d=EMBED_DIM, n_groups=4, seed=0):
    """Embedding cloud + a label correlated with a fixed direction + group codes."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d)
    score = X @ w
    y = (score > np.quantile(score, 0.65)).astype(np.int8)  # ~35% positive
    groups = rng.integers(0, n_groups, size=n).astype(np.int32)
    return X, y, groups


# ----------------------------------------------------------------------------
# FeatureScaler
# ----------------------------------------------------------------------------


def test_scaler_standardizes_and_imputes_nan():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 5)).astype(np.float32)
    X[3, 2] = np.nan  # one NaN cell -> imputed to the column median
    sc = FeatureScaler()
    Xs = sc.fit(X)
    assert np.isfinite(Xs).all()
    # Standardized: per-column mean ~0, std ~1 (on the imputed matrix).
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-5)


def test_scaler_all_nan_column_maps_to_constant():
    X = np.ones((10, 3), dtype=np.float32)
    X[:, 1] = np.nan  # all-NaN column -> median 0, std 0 -> 1, output 0
    sc = FeatureScaler()
    Xs = sc.fit(X)
    assert np.all(Xs[:, 1] == 0.0) and np.isfinite(Xs).all()


def test_scaler_roundtrip_arrays():
    X = np.random.default_rng(2).standard_normal((20, 6)).astype(np.float32)
    sc = FeatureScaler()
    sc.fit(X)
    sc2 = FeatureScaler.from_arrays(sc.to_arrays())
    np.testing.assert_array_equal(sc.apply(X), sc2.apply(X))


# ----------------------------------------------------------------------------
# MLPClassifierHead
# ----------------------------------------------------------------------------


def test_build_mlp_output_is_single_logit():
    import torch

    net = build_mlp(EMBED_DIM)
    out = net(torch.zeros(7, EMBED_DIM))
    assert out.shape == (7, 1)


def test_head_fit_predict_probability_range():
    X, y, groups = _synthetic()
    head = MLPClassifierHead(seed=0, batch=256, epochs=15)
    val = groups == 0
    head.fit(X[~val], y[~val], eval_set=(X[val], y[val]))
    p = head.predict(X)
    assert p.shape == (X.shape[0],)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_head_predict_imputes_nan_rows():
    X, y, groups = _synthetic(n=300)
    head = MLPClassifierHead(seed=0, batch=128, epochs=10)
    head.fit(X, y, eval_set=(X[:64], y[:64]))
    Xn = X.copy()
    Xn[0] = np.nan  # fully-NaN row -> scaler imputes -> finite prediction
    p = head.predict(Xn)
    assert np.isfinite(p).all()


def test_head_save_load_roundtrip(tmp_path):
    X, y, _ = _synthetic(n=300)
    head = MLPClassifierHead(seed=1, batch=128, epochs=10)
    head.fit(X, y, eval_set=(X[:64], y[:64]))
    p = head.predict(X)
    head.save(tmp_path / "h")
    reloaded = MLPClassifierHead()
    reloaded.load(tmp_path / "h")
    np.testing.assert_allclose(reloaded.predict(X), p, rtol=0, atol=1e-6)
    assert reloaded.model_hash() == head.model_hash()


# ----------------------------------------------------------------------------
# DeployableHead
# ----------------------------------------------------------------------------


def test_deployable_fit_predict_and_ensemble_mean():
    X, y, groups = _synthetic(n=800, n_groups=5)
    head = DeployableHead(seeds=(0, 1, 2), batch=256, epochs=12)
    head.fit(X, y, groups=groups, verbose=False)
    assert len(head._members) == 3
    p = head.predict(X)
    assert p.shape == (X.shape[0],)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # Ensemble prediction is exactly the mean of the member predictions.
    members = np.mean([m.predict(X) for m in head._members], axis=0)
    np.testing.assert_allclose(p, members, rtol=0, atol=1e-9)


def test_deployable_requires_multiple_groups():
    X, y, _ = _synthetic(n=100)
    head = DeployableHead(seeds=(0,), batch=64, epochs=3)
    with pytest.raises(ValueError):
        head.fit(X, y, groups=np.zeros(len(y), dtype=np.int32), verbose=False)


def test_deployable_save_load_roundtrip(tmp_path):
    X, y, groups = _synthetic(n=600, n_groups=4)
    obs_to_int = {f"IMG_{i}": i for i in range(4)}
    head = DeployableHead(seeds=(0, 1), batch=256, epochs=10)
    head.fit(X, y, groups=groups, obs_to_int=obs_to_int, verbose=False)
    p = head.predict(X)
    out = tmp_path / "deploy"
    head.save(out)
    assert (out / "recipe.json").exists()

    reloaded = DeployableHead.load(out)
    np.testing.assert_allclose(reloaded.predict(X), p, rtol=0, atol=1e-6)
    assert reloaded.recipe_hash() == head.recipe_hash()
    assert reloaded.model_hash() == head.model_hash()
    assert set(reloaded._train_obs_ids) == set(obs_to_int)


def test_recipe_hash_is_config_only():
    # Same recipe config -> same recipe_hash regardless of trained weights.
    a = DeployableHead(seeds=(0, 1, 2))
    b = DeployableHead(seeds=(0, 1, 2))
    assert a.recipe_hash() == b.recipe_hash()
    c = DeployableHead(seeds=(0, 1, 2), dropout=0.5)
    assert c.recipe_hash() != a.recipe_hash()


# ----------------------------------------------------------------------------
# DeployableHead H2 nuisance-subspace projection
# ----------------------------------------------------------------------------


def _orthonormal_basis(d, k, seed=7):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    return q[:, :k].astype(np.float32)


def test_project_removes_its_own_directions_and_is_idempotent():
    N = _orthonormal_basis(EMBED_DIM, 4)
    head = DeployableHead(seeds=(0,), nuisance_basis=N)
    X = np.random.default_rng(0).standard_normal((50, EMBED_DIM)).astype(np.float32)
    Xp = head._project(X)
    # projected data has zero component along every basis vector
    assert np.allclose(Xp @ N, 0.0, atol=1e-4)
    # idempotent: projecting again changes nothing
    np.testing.assert_allclose(head._project(Xp), Xp, rtol=0, atol=1e-5)


def test_project_passes_nan_rows_through():
    N = _orthonormal_basis(EMBED_DIM, 4)
    head = DeployableHead(seeds=(0,), nuisance_basis=N)
    X = np.random.default_rng(1).standard_normal((10, EMBED_DIM)).astype(np.float32)
    X[0] = np.nan
    Xp = head._project(X)
    assert np.isnan(Xp[0]).all()          # NaN row untouched (no NaN-spread)
    assert np.isfinite(Xp[1:]).all()


def test_project_none_is_identity():
    head = DeployableHead(seeds=(0,))
    X = np.random.default_rng(2).standard_normal((8, EMBED_DIM)).astype(np.float32)
    np.testing.assert_array_equal(head._project(X), X)


def test_nuisance_basis_survives_save_load(tmp_path):
    N = _orthonormal_basis(EMBED_DIM, 4)
    X, y, groups = _synthetic(n=600, n_groups=4)
    head = DeployableHead(seeds=(0, 1), batch=256, epochs=8, nuisance_basis=N)
    head.fit(X, y, groups=groups, verbose=False)
    p = head.predict(X)
    out = tmp_path / "deploy_h2"
    head.save(out)
    assert (out / "nuisance_basis.npy").exists()
    reloaded = DeployableHead.load(out)
    assert reloaded.nuisance_basis is not None
    np.testing.assert_array_equal(reloaded.nuisance_basis, N)
    np.testing.assert_allclose(reloaded.predict(X), p, rtol=0, atol=1e-6)
    assert reloaded.model_hash() == head.model_hash()
    # A head with a basis differs from one without (model_hash folds in the basis).
    plain = DeployableHead(seeds=(0, 1), batch=256, epochs=8)
    plain.fit(X, y, groups=groups, verbose=False)
    assert plain.model_hash() != head.model_hash()


# ----------------------------------------------------------------------------
# DeployableHead H3 consistency-regularized training
# ----------------------------------------------------------------------------


def _consistency_pairs(X, n=200, seed=5):
    """Co-located overlap pairs: same tile + a small frame-nuisance perturbation."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=n)
    ea = X[idx]
    eb = ea + 0.3 * rng.standard_normal(ea.shape).astype(np.float32)  # "other frame"
    return ea, eb


def test_lambda_zero_ignores_pairs_exactly():
    # λ=0 must reproduce the un-regularized fit bit-for-bit even if pairs are passed.
    X, y, groups = _synthetic(n=600, n_groups=4)
    pairs = _consistency_pairs(X)
    base = DeployableHead(seeds=(0, 1), batch=256, epochs=8, lambda_consistency=0.0)
    base.fit(X, y, groups=groups, consistency_pairs=pairs, verbose=False)
    ref = DeployableHead(seeds=(0, 1), batch=256, epochs=8)
    ref.fit(X, y, groups=groups, verbose=False)
    np.testing.assert_allclose(base.predict(X), ref.predict(X), rtol=0, atol=1e-6)


def test_consistency_penalty_changes_weights_and_shrinks_pair_gap():
    # λ>0 should change the fit AND reduce disagreement on the co-located pairs.
    X, y, groups = _synthetic(n=600, n_groups=4)
    pairs = _consistency_pairs(X)
    plain = DeployableHead(seeds=(0, 1), batch=256, epochs=25, lambda_consistency=0.0)
    plain.fit(X, y, groups=groups, verbose=False)
    reg = DeployableHead(seeds=(0, 1), batch=256, epochs=25, lambda_consistency=5.0)
    reg.fit(X, y, groups=groups, consistency_pairs=pairs, verbose=False)
    # Predictions differ (the penalty did something).
    assert not np.allclose(plain.predict(X), reg.predict(X), atol=1e-3)
    # The regularized head disagrees LESS across the co-located pairs.
    ea, eb = pairs
    gap = lambda h: float(np.mean(np.abs(h.predict(ea) - h.predict(eb))))
    assert gap(reg) < gap(plain)


def test_lambda_consistency_survives_save_load(tmp_path):
    X, y, groups = _synthetic(n=500, n_groups=4)
    pairs = _consistency_pairs(X)
    head = DeployableHead(seeds=(0, 1), batch=256, epochs=8, lambda_consistency=3.0)
    head.fit(X, y, groups=groups, consistency_pairs=pairs, verbose=False)
    p = head.predict(X)
    out = tmp_path / "deploy_h3"
    head.save(out)
    reloaded = DeployableHead.load(out)
    assert reloaded.lambda_consistency == 3.0
    np.testing.assert_allclose(reloaded.predict(X), p, rtol=0, atol=1e-6)


def test_the_frozen_recipe_carries_no_measured_metrics():
    """R09's residue. A recipe is a CONFIGURATION; performance is a measurement.

    `FROZEN_RECIPE` is stamped verbatim into every head's `recipe.json`, so a metric
    constant here becomes a claim every head makes about itself. R09 caught the F head
    asserting pooled PR-AUC 0.7832 when its real value was 0.7438; the v2 rebuild caught
    both new heads asserting 0.7832 / 0.7865 when the corrected basis measures
    0.7826 / 0.7778 -- and the A1 head asserting the baseline's numbers outright.

    It must also stay out of the recipe HASH: an unchanged configuration would otherwise
    hash differently the day its LOIO is re-run.
    """
    from src.modeling.mlp_head import FROZEN_RECIPE

    banned = [k for k in FROZEN_RECIPE
              if any(t in k.lower() for t in ("auc", "_pr_", "precision", "recall",
                                              "brier", "rmse", "spearman", "score"))]
    assert not banned, (
        f"FROZEN_RECIPE must not contain measured metrics, found {banned}. "
        f"Metrics belong beside the head (LOIO predictions/summaries), not in the card "
        f"that describes how to build it -- see R09 and DECISIONS 2026-08-23.")

    # and the values that WERE there must not have crept back under any name
    assert 0.7832 not in FROZEN_RECIPE.values()
    assert 0.7865 not in FROZEN_RECIPE.values()


def test_the_arm_a_store_infers_is_the_arm_its_map_driver_demands():
    """The two halves of R07 must agree, or a head trains fine and the map driver refuses it.

    DECISIONS 2026-08-24. `train_deployable_head.py` records
    `args.norm_arm or infer_norm_arm(store_name)`, and `scripts/striping_a1_map.py` calls
    `require_norm_arm(head, src.striping.A1_ARM, strict=True)`. Those are two constants in
    two modules. A head trained with `--norm-arm a1` -- a plausible-looking literal --
    declares 'a1', which is NOT `A1_ARM`, so the driver rejects it. That killed all six
    array tasks after GPUs had been allocated.

    The guard behaved correctly; the invariant it depends on was simply untested.
    """
    from src.modeling.mlp_head import NO_NORM_ARM, infer_norm_arm, require_norm_arm
    from src.striping import A1_ARM

    inferred = infer_norm_arm("fang_embeddings_a1")
    assert inferred == A1_ARM, (
        f"infer_norm_arm gives {inferred!r} but striping.A1_ARM is {A1_ARM!r}; a head trained "
        f"from the a1 store would be refused by scripts/striping_a1_map.py")
    assert infer_norm_arm("fang_embeddings") == NO_NORM_ARM

    # the bare "a1" must NOT satisfy the A1 driver -- that is the whole point of versioning it
    class _Stub:
        norm_arm = "a1"
    with pytest.raises(ValueError, match="different input distributions"):
        require_norm_arm(_Stub(), A1_ARM, where="stub", strict=True)


def test_a_head_built_from_each_store_satisfies_its_own_driver():
    """End-to-end on the constants: whatever a store infers must pass its driver's check."""
    from src.modeling.mlp_head import NO_NORM_ARM, infer_norm_arm, require_norm_arm
    from src.striping import A1_ARM

    for store, expected in (("fang_embeddings_a1", A1_ARM), ("fang_embeddings", NO_NORM_ARM)):
        class _Stub:
            norm_arm = infer_norm_arm(store)
        require_norm_arm(_Stub(), expected, where=store, strict=True)   # must not raise
