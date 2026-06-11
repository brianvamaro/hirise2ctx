"""Unit tests for src.modeling.loaders against synthetic packaged folds.

Integration coverage against the real dataset lives in test_modeling_group_leak.py;
this file uses tmp_path synthetic parquets so it stays fast and runs every CI cycle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling import loaders as L


# ============================================================================
# Synthetic packaged-fold builder
# ============================================================================


def _write_synthetic_package(
    tmp_path: Path,
    *,
    obs_ids: list[str],
    n_folds: int,
    n_tiles_per_image: int = 20,
    scales: tuple[int, ...] = (0, 1),
) -> Path:
    """Create dataset/packaged/test_scheme/{X,y,groups}_*_fold{k}.* mimicking Stage 5."""
    out = tmp_path / "packaged" / "test_scheme"
    out.mkdir(parents=True)

    # Build per-image rows across all scales
    obs_to_int = {obs: i for i, obs in enumerate(obs_ids)}
    rows = []
    for obs in obs_ids:
        for scale_idx in scales:
            tile_size_px = 8 * (2 ** scale_idx)
            for ti in range(n_tiles_per_image):
                rows.append({
                    "obs_id": obs,
                    "scale_idx": int(scale_idx),
                    "tile_size_px": int(tile_size_px),
                    "ti": int(ti),
                    "tj": 0,
                    "valid_pixel_fraction": 1.0,
                    "intensity_mean": float(obs_to_int[obs] * 10 + ti),
                    "shadow_fraction": 0.05,
                    # patch_idx_S* columns -- treated as keys, NOT features
                    "patch_idx_S32": ti % 5 - 1,   # mix in some -1s
                    "patch_idx_S64": ti % 7 - 1,
                    "config_hash_feat": "synthetic",
                    "boulder_area": float(ti) * 0.5,
                    "boulder_count": int(ti),
                    "tile_area": 1600.0 * (4 ** scale_idx),
                    "fractional_area": float(ti) * 0.0001,
                    "binary_by_area": False,
                    "binary_by_count": ti > 0,
                    "count_density": float(ti) * 1e-5,
                    "xmin": 0.0, "ymin": 0.0, "xmax": 40.0, "ymax": 40.0,
                    "tile_size_m": float(tile_size_px * 5),
                })
    full = pd.DataFrame(rows)
    feat_cols_with_keys = [
        "obs_id", "scale_idx", "tile_size_px", "ti", "tj",
        "valid_pixel_fraction", "intensity_mean", "shadow_fraction",
        "patch_idx_S32", "patch_idx_S64", "config_hash_feat",
    ]
    y_cols_with_keys = [
        "obs_id", "scale_idx", "tile_size_px", "ti", "tj",
        "boulder_area", "boulder_count", "tile_area", "fractional_area",
        "binary_by_area", "binary_by_count", "count_density",
        "xmin", "ymin", "xmax", "ymax", "tile_size_m",
    ]

    # LOIO folds: each image is test in exactly one fold
    per_fold = []
    folds = []
    for k in range(n_folds):
        test_obs = [obs_ids[k]]
        train_obs = [o for o in obs_ids if o not in test_obs]
        train_df = full[full["obs_id"].isin(train_obs)].reset_index(drop=True)
        test_df = full[full["obs_id"].isin(test_obs)].reset_index(drop=True)
        train_df[feat_cols_with_keys].to_parquet(out / f"X_train_fold{k}.parquet", index=False)
        train_df[y_cols_with_keys].to_parquet(out / f"y_train_fold{k}.parquet", index=False)
        test_df[feat_cols_with_keys].to_parquet(out / f"X_test_fold{k}.parquet", index=False)
        test_df[y_cols_with_keys].to_parquet(out / f"y_test_fold{k}.parquet", index=False)
        np.save(out / f"groups_train_fold{k}.npy",
                np.asarray([obs_to_int[o] for o in train_df["obs_id"]], dtype=np.int32))
        np.save(out / f"groups_test_fold{k}.npy",
                np.asarray([obs_to_int[o] for o in test_df["obs_id"]], dtype=np.int32))
        folds.append({"fold_idx": k, "test_obs_ids": list(test_obs), "train_obs_ids": list(train_obs)})
        per_fold.append({
            "fold_idx": k, "test_obs_ids": list(test_obs),
            "n_train_tiles": int(len(train_df)),
            "n_test_tiles": int(len(test_df)),
        })

    meta = {
        "name": "test_scheme",
        "obs_to_int": obs_to_int,
        "per_fold": per_fold,
        "folds": folds,
    }
    (out / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


# ============================================================================
# Tests
# ============================================================================


def test_load_fold_returns_separated_X_y_groups(tmp_path):
    obs_ids = [f"OBS_{i}" for i in range(3)]
    dataset_dir = _write_synthetic_package(tmp_path, obs_ids=obs_ids, n_folds=3)
    fold = L.load_fold("test_scheme", 0, scale_idx=0, dataset_dir=dataset_dir)
    # 3 features (valid_pixel_fraction, intensity_mean, shadow_fraction); patch_idx_S*
    # and config_hash_feat are excluded.
    assert fold.feature_names == ["valid_pixel_fraction", "intensity_mean", "shadow_fraction"]
    assert fold.X_train.dtype == np.float32
    assert fold.X_train.shape[1] == 3
    # patch_idx columns end up on keys frame
    assert "patch_idx_S32" in fold.keys_train.columns
    assert "patch_idx_S64" in fold.keys_train.columns
    # groups split correctly
    assert set(np.unique(fold.groups_test).tolist()) == {0}
    assert 0 not in set(np.unique(fold.groups_train).tolist())


def test_scale_filter_subsets_rows(tmp_path):
    obs_ids = [f"OBS_{i}" for i in range(2)]
    dataset_dir = _write_synthetic_package(
        tmp_path, obs_ids=obs_ids, n_folds=2, n_tiles_per_image=10, scales=(0, 1),
    )
    f_all = L.load_fold("test_scheme", 0, scale_idx=None, dataset_dir=dataset_dir)
    f_s0 = L.load_fold("test_scheme", 0, scale_idx=0, dataset_dir=dataset_dir)
    f_s1 = L.load_fold("test_scheme", 0, scale_idx=1, dataset_dir=dataset_dir)
    # 1 train image, 10 rows per scale
    assert f_all.X_train.shape[0] == 20
    assert f_s0.X_train.shape[0] == 10
    assert f_s1.X_train.shape[0] == 10
    # Subsetting preserves column count
    assert f_s0.X_train.shape[1] == f_all.X_train.shape[1]


def test_iter_loio_folds_yields_each_image_as_test_once(tmp_path):
    obs_ids = [f"OBS_{i}" for i in range(4)]
    dataset_dir = _write_synthetic_package(tmp_path, obs_ids=obs_ids, n_folds=4)
    seen_test_obs = set()
    fold_count = 0
    for fold in L.iter_loio_folds("test_scheme", scale_idx=0, dataset_dir=dataset_dir):
        fold_count += 1
        # Each fold's test set is exactly one ObsId (LOIO)
        assert len(fold.held_out_obs_ids) == 1
        seen_test_obs.add(fold.held_out_obs_ids[0])
    assert fold_count == 4
    assert seen_test_obs == set(obs_ids)


def test_gather_patches_drops_margin_rows(tmp_path):
    obs_ids = [f"OBS_{i}" for i in range(2)]
    dataset_dir = _write_synthetic_package(tmp_path, obs_ids=obs_ids, n_folds=2)
    fold = L.load_fold("test_scheme", 0, scale_idx=0, dataset_dir=dataset_dir)
    keys = fold.keys_test.copy()
    # Make a small in-tmp patch stack for the test obs at S=32: shape (max_idx+1, 32, 32)
    obs = keys["obs_id"].iloc[0]
    max_idx = int(keys["patch_idx_S32"].max()) + 1
    if max_idx < 1:
        max_idx = 1
    stack = np.arange(max_idx * 32 * 32, dtype=np.uint8).reshape(max_idx, 32, 32)
    pdir = dataset_dir / "context_patches"
    pdir.mkdir()
    np.save(pdir / f"{obs}_S32.npy", stack)

    patches, valid_rows = L.gather_patches(keys, 32, dataset_dir=dataset_dir)
    # Margin rows (patch_idx_S32 == -1) are dropped from `patches` and not in valid_rows
    n_margin = int((keys["patch_idx_S32"] < 0).sum())
    n_valid = len(keys) - n_margin
    assert patches.shape == (n_valid, 32, 32)
    assert valid_rows.size == n_valid
    assert patches.dtype == np.uint8


def test_gather_patches_assembles_in_original_row_order(tmp_path):
    """gather_patches output must align with valid_rows row-for-row."""
    obs_ids = ["A", "B"]
    dataset_dir = _write_synthetic_package(tmp_path, obs_ids=obs_ids, n_folds=2, n_tiles_per_image=8)
    fold = L.load_fold("test_scheme", 0, scale_idx=0, dataset_dir=dataset_dir)
    keys = fold.keys_train.copy()  # multi-obs train set
    pdir = dataset_dir / "context_patches"
    pdir.mkdir()
    # Each obs gets a stack indexed [0..K-1]; fill image[k] with k repeated so we can
    # verify alignment after gather. Use S=32 to match the synthetic patch_idx column.
    for obs in keys["obs_id"].unique():
        k_needed = int(keys.loc[keys["obs_id"] == obs, "patch_idx_S32"].max()) + 1
        if k_needed < 1:
            k_needed = 1
        stack = np.zeros((k_needed, 32, 32), dtype=np.uint8)
        for k in range(k_needed):
            stack[k, :, :] = k % 256
        np.save(pdir / f"{obs}_S32.npy", stack)

    patches, valid_rows = L.gather_patches(keys, 32, dataset_dir=dataset_dir)
    valid_keys = keys.iloc[valid_rows].reset_index(drop=True)
    for i, (_, row) in enumerate(valid_keys.iterrows()):
        expected = row["patch_idx_S32"] % 256
        assert np.all(patches[i] == expected), (
            f"row {i} ({row['obs_id']}, patch_idx={row['patch_idx_S32']}) misaligned"
        )


# ============================================================================
# Per-image feature standardization (W1 next-bet 1)
# ============================================================================


def _two_group_matrix():
    rng = np.random.default_rng(7)
    groups = np.array([0] * 50 + [1] * 50, dtype=np.int32)
    X = np.empty((100, 3), dtype=np.float32)
    X[:50, 0] = rng.normal(100, 10, 50)   # group 0 bright
    X[50:, 0] = rng.normal(10, 2, 50)     # group 1 dark -- big between-image shift
    X[:, 1] = rng.normal(0, 1, 100)
    X[:, 2] = 5.0                          # constant column
    return X, groups


def test_perimage_zscore_centres_each_group():
    X, groups = _two_group_matrix()
    out = L._standardize_matrix_per_group(X, groups, "zscore")
    for g in (0, 1):
        m = groups == g
        assert abs(out[m, 0].mean()) < 1e-5
        assert out[m, 0].std() == pytest.approx(1.0, abs=1e-4)
    # constant column maps to 0, no NaN/inf anywhere
    assert np.all(out[:, 2] == 0.0)
    assert np.isfinite(out).all()


def test_perimage_rank_is_uniform_and_order_preserving():
    X, groups = _two_group_matrix()
    out = L._standardize_matrix_per_group(X, groups, "rank")
    m = groups == 0
    assert out[m, 0].min() == 0.0 and out[m, 0].max() == 1.0
    # order preserved within group
    order_in = np.argsort(X[m, 0])
    order_out = np.argsort(out[m, 0])
    assert np.array_equal(order_in, order_out)
    # after rank, the two groups' distributions coincide despite the raw shift
    assert abs(out[groups == 0, 0].mean() - out[groups == 1, 0].mean()) < 1e-6


def test_perimage_robust_centres_median():
    X, groups = _two_group_matrix()
    out = L._standardize_matrix_per_group(X, groups, "robust")
    for g in (0, 1):
        m = groups == g
        assert abs(np.median(out[m, 0])) < 1e-5
    assert np.isfinite(out).all()


def test_perimage_groups_are_independent():
    """Perturbing one group's raw values must not change another group's output."""
    X, groups = _two_group_matrix()
    base = L._standardize_matrix_per_group(X, groups, "zscore")
    X2 = X.copy()
    X2[groups == 1] *= 100.0
    out2 = L._standardize_matrix_per_group(X2, groups, "zscore")
    assert np.allclose(base[groups == 0], out2[groups == 0])


def test_perimage_unknown_method_raises():
    X, groups = _two_group_matrix()
    with pytest.raises(ValueError, match="unknown per-image transform"):
        L._standardize_matrix_per_group(X, groups, "minmax")


def test_augment_fold_doubles_width_and_keeps_raw():
    X, groups = _two_group_matrix()
    out = L._standardize_matrix_per_group(X, groups, "zscore")
    cat = np.concatenate([X, out], axis=1)
    # augment helper is exercised end-to-end via the Fold path in the sweep; here we
    # verify the concatenation semantics it relies on: raw block unchanged, std block
    # equals the pure transform.
    assert cat.shape == (100, 6)
    assert np.allclose(cat[:, :3], X)
    assert np.allclose(cat[:, 3:], out)
