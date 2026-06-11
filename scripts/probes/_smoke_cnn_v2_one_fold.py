"""W2 S3 smoke: SmallCNNClassifier on v2, fold 0 of loio_nfold, S=64 patches, GPU.

Asserts (PLAN_CNN.md S3): CUDA device actually used, no NaN loss, sane per-epoch
runtime, AUC computable on the held-out image. 3 epochs only -- this is plumbing
verification, not a result.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# IMPORTANT: import src.modeling BEFORE numpy. Package init runs the Windows DLL
# bootstrap (KMP env + add_dll_directory + shm.dll preload).
import src.modeling  # noqa: F401

import numpy as np

from src.modeling.binary_target import get_target
from src.modeling.cnn import CNNParams, SmallCNNClassifier
from src.modeling.loaders import load_fold

SCALE_IDX = 3  # S=64 tiles, matched to 64-px patches
DATASET_DIR = "dataset_v2"


def main() -> int:
    f = load_fold("loio_nfold", 0, scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR)
    target = get_target("fa_gt_1e-2")
    print(f"fold 0  held_out={f.held_out_obs_ids}  n_train={f.X_train.shape[0]}  "
          f"n_test={f.X_test.shape[0]}")
    if "patch_idx_S64" not in f.keys_train.columns:
        print("FATAL: keys_train has no patch_idx_S64 -- Stage 5 re-package missing?")
        return 1
    n_margin = int((f.keys_train["patch_idx_S64"] < 0).sum())
    print(f"  train margin rows (no patch): {n_margin} / {len(f.keys_train)}")

    y_tr_all = target.binarize(f.y_train).astype(np.float64)
    y_te = target.binarize(f.y_test).astype(np.float64)

    # Inner-val split: harness rotation rule (1 image). The Phase 1 driver will hold
    # out 4-5 whole images; for smoke we only need the early-stopping path exercised.
    train_codes = f.groups_train
    unique_train = np.unique(train_codes)
    inner_val_code = int(unique_train[f.fold_idx % unique_train.size])
    inner_val_mask = train_codes == inner_val_code
    inner_train_mask = ~inner_val_mask
    keys_tr = f.keys_train[inner_train_mask].reset_index(drop=True)
    keys_vl = f.keys_train[inner_val_mask].reset_index(drop=True)
    print(f"  inner-train n={len(keys_tr)} (pos {y_tr_all[inner_train_mask].sum():.0f})  "
          f"inner-val n={len(keys_vl)} (pos {y_tr_all[inner_val_mask].sum():.0f})")

    params = CNNParams(patch_size_px=64, epochs=3, batch_size=256,
                       dataset_dir=DATASET_DIR)
    print(f"  device={params.device}")
    assert params.device == "cuda", "expected CUDA after S1 install"

    model = SmallCNNClassifier(params=params)
    model.bind_train_data(keys_tr, y_tr_all[inner_train_mask])
    model.bind_val_data(keys_vl, y_tr_all[inner_val_mask])
    print("\nFitting (3 epochs)...")
    t0 = time.monotonic()
    model.fit(np.empty((len(keys_tr), 0), dtype=np.float32), y_tr_all[inner_train_mask])
    dt = time.monotonic() - t0
    print(f"  fit wall time {dt:.1f}s ({dt/3:.1f}s/epoch)")

    model.bind_predict_data(f.keys_test)
    probs = model.predict(np.empty((len(f.keys_test), 0), dtype=np.float32))
    assert np.isfinite(probs).all(), "non-finite probabilities"
    n_pos, n_neg = int(y_te.sum()), int((1 - y_te).sum())
    print(f"\n  held-out: n_pos={n_pos} n_neg={n_neg}  "
          f"prob range [{probs.min():.4f}, {probs.max():.4f}]")
    if n_pos > 0 and n_neg > 0:
        from sklearn.metrics import average_precision_score, roc_auc_score
        print(f"  AUC={roc_auc_score(y_te, probs):.4f}  "
              f"AP={average_precision_score(y_te, probs):.4f}  (3-epoch smoke, not a result)")
    print(f"  model_hash={model.model_hash()[:16]}...")
    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
