"""Smoke-test src.modeling.loaders against the packaged dataset."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.loaders import (
    gather_patches,
    iter_loio_folds,
    load_fold,
    load_metadata,
    n_folds,
)


def main() -> int:
    print(f"loio_9fold: {n_folds('loio_9fold')} folds")
    meta = load_metadata("loio_9fold")
    print(f"  obs_to_int has {len(meta['obs_to_int'])} entries")

    # Load fold 0, scale 0
    f = load_fold("loio_9fold", 0, scale_idx=0)
    print(f"\nFold {f.fold_idx} (scale_idx=0), held-out {f.held_out_obs_ids}:")
    print(f"  X_train: {f.X_train.shape}  dtype={f.X_train.dtype}")
    print(f"  X_test : {f.X_test.shape}")
    print(f"  y_train cols: {len(f.y_train.columns)}  groups_train uniq: {len(set(f.groups_train))}")
    print(f"  groups_test uniq: {len(set(f.groups_test))}  (should be 1 for LOIO)")
    print(f"  feature_names ({len(f.feature_names)}): first 5 = {f.feature_names[:5]}")

    # Check group-leak: no overlap between train and test ObsIds
    train_obs = set(f.keys_train["obs_id"].unique())
    test_obs = set(f.keys_test["obs_id"].unique())
    leak = train_obs & test_obs
    print(f"  train ObsIds: {len(train_obs)}, test ObsIds: {len(test_obs)}, leak: {leak or 'none'}")

    # Patch fetch -- use a small slice
    patches, valid_rows = gather_patches(f.keys_test.head(500), 32)
    print(f"\n  gather_patches(first 500 test rows, S=32): shape={patches.shape}, valid_rows={len(valid_rows)}, dtype={patches.dtype}")

    # Iterate all 9 folds, just confirm shapes
    print(f"\nIterating all loio_9fold folds (scale_idx=0):")
    for f in iter_loio_folds("loio_9fold", scale_idx=0):
        print(f"  fold {f.fold_idx}: train={f.X_train.shape[0]:>7d}  test={f.X_test.shape[0]:>6d}  held_out={f.held_out_obs_ids[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
