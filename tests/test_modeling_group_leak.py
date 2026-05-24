"""Integration tests against the real packaged dataset.

These tests fail if any modeling fold smuggles an ObsId between train and test, or
if iter_loio_folds disagrees with the Stage 5 split definition. Both must hold for
any modeling result to be trustworthy.

Marked `slow` because they read 9 real X/y parquets totalling ~500 MB; rerun whenever
Stage 5 packaging changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.modeling.loaders import iter_loio_folds, load_metadata, n_folds


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_LOIO_9 = REPO_ROOT / "dataset" / "packaged" / "loio_9fold"

pytestmark = pytest.mark.skipif(
    not PACKAGED_LOIO_9.exists(),
    reason="dataset/packaged/loio_9fold not present -- run scripts/run_stage5.py --all first",
)


@pytest.mark.slow
def test_loio_9fold_has_expected_number_of_folds():
    assert n_folds("loio_9fold") == 9


@pytest.mark.slow
def test_no_obs_id_appears_in_both_train_and_test_of_any_fold():
    """Group-leak assertion. Mirrors notebooks/09_splits_qa.ipynb."""
    leaks = []
    for fold in iter_loio_folds("loio_9fold"):
        train_obs = set(fold.keys_train["obs_id"].unique())
        test_obs = set(fold.keys_test["obs_id"].unique())
        overlap = train_obs & test_obs
        if overlap:
            leaks.append((fold.fold_idx, overlap))
    assert not leaks, f"group leak in folds: {leaks}"


@pytest.mark.slow
def test_each_obs_id_appears_as_test_in_exactly_one_fold():
    seen = {}
    for fold in iter_loio_folds("loio_9fold"):
        for obs in fold.keys_test["obs_id"].unique():
            seen.setdefault(obs, []).append(fold.fold_idx)
    dups = {o: folds for o, folds in seen.items() if len(folds) > 1}
    assert not dups, f"ObsId appears in multiple test folds: {dups}"
    # Cross-check against metadata
    meta = load_metadata("loio_9fold")
    assert set(seen) == set(meta["obs_to_int"])


@pytest.mark.slow
def test_scale_subset_preserves_train_test_disjointness():
    """Same group-leak assertion but scale-subset (used by per-scale modeling)."""
    for scale_idx in (0, 3):
        for fold in iter_loio_folds("loio_9fold", scale_idx=scale_idx):
            train_obs = set(fold.keys_train["obs_id"].unique())
            test_obs = set(fold.keys_test["obs_id"].unique())
            assert not (train_obs & test_obs), (
                f"scale_idx={scale_idx} fold={fold.fold_idx}: train/test overlap"
            )


@pytest.mark.slow
def test_x_train_excludes_patch_idx_columns():
    """patch_idx_S* are array indices, not predictive features -- must not be in X."""
    fold = next(iter(iter_loio_folds("loio_9fold", scale_idx=0)))
    assert "patch_idx_S32" not in fold.feature_names
    assert "patch_idx_S64" not in fold.feature_names
    assert "config_hash_feat" not in fold.feature_names
    # But they remain accessible on the keys frame for downstream CNN code
    assert "patch_idx_S32" in fold.keys_train.columns


@pytest.mark.slow
def test_per_fold_test_count_matches_metadata():
    meta = load_metadata("loio_9fold")
    sizes_meta = {f["fold_idx"]: f["n_test_tiles"] for f in meta["per_fold"]}
    for fold in iter_loio_folds("loio_9fold"):
        # Note: meta n_test_tiles is across all scales; with scale_idx=None we should match
        full = next(iter(iter_loio_folds("loio_9fold", scale_idx=None)))
        # Re-iterate explicitly per fold to compare
    # Simpler: load with scale_idx=None and compare
    for fold in iter_loio_folds("loio_9fold", scale_idx=None):
        assert fold.X_test.shape[0] == sizes_meta[fold.fold_idx]
