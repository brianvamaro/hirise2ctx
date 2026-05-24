"""Quick CNN smoke test: 3 epochs on fold 0 at S=32, scale_idx=2."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# IMPORTANT: import src.modeling BEFORE numpy. Package init runs the Windows DLL
# bootstrap (KMP env + add_dll_directory + shm.dll preload); numpy loading first
# would pull libiomp5md.dll in ahead of torch's libomp.dll and the order is fragile.
import src.modeling  # noqa: F401

import numpy as np

from src.modeling.cnn import CNNParams, SmallCNNRegressor
from src.modeling.evaluate import per_fold_metrics
from src.modeling.loaders import load_fold


def main() -> int:
    f = load_fold("loio_9fold", 0, scale_idx=2)  # S=32 px tiles, matched to S=32 patches
    print(f"fold 0  held_out={f.held_out_obs_ids}  n_train={f.X_train.shape[0]}  n_test={f.X_test.shape[0]}")

    # Inner-val split (use same rotation rule as harness)
    train_codes = f.groups_train
    unique_train = np.unique(train_codes)
    inner_val_code = int(unique_train[f.fold_idx % unique_train.size])
    inner_val_mask = train_codes == inner_val_code
    inner_train_mask = ~inner_val_mask
    y_tr_all = f.y_train["fractional_area"].to_numpy()

    keys_tr = f.keys_train[inner_train_mask].reset_index(drop=True)
    y_tr = y_tr_all[inner_train_mask]
    keys_vl = f.keys_train[inner_val_mask].reset_index(drop=True)
    y_vl = y_tr_all[inner_val_mask]
    print(f"  inner-train n={len(keys_tr)}  inner-val n={len(keys_vl)}")

    params = CNNParams(patch_size_px=32, epochs=3, batch_size=256, learning_rate=1e-3,
                       early_stopping_patience=10)
    model = SmallCNNRegressor(params=params)
    model.bind_train_data(keys_tr, y_tr)
    model.bind_val_data(keys_vl, y_vl)
    print("\nFitting...")
    model.fit(np.empty((len(keys_tr), 0), dtype=np.float32), y_tr)

    model.bind_predict_data(f.keys_test)
    y_pred = model.predict(np.empty((len(f.keys_test), 0), dtype=np.float32))
    y_test = f.y_test["fractional_area"].to_numpy()
    m = per_fold_metrics(y_test, y_pred, held_out_obs_ids=f.held_out_obs_ids)
    print(f"\n  spearman={m['spearman_rho']:+.4f}  rmse_log1p={m['rmse_log1p']:.4g}  auc={m['presence_auc']:.3f}")
    print(f"  model_hash={model.model_hash()[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
