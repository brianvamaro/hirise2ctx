"""W2 follow-up (free test): AdaBN on the saved cell-A state_dicts.

Li et al. 2016 (arXiv:1603.04779): domain identity lives largely in BatchNorm
statistics. For each LOIO fold, load the trained cell-A classifier, RESET the
BN running stats, re-estimate them with forward passes over the HELD-OUT
image's own patches (no labels, no weight updates -- inference-compatible:
a deployment CTX window supplies thousands of patches), then predict.
The CNN analog of the bet-1 zscore rescue (which fixed all 3
distribution_shift images but cost raw-feature images cohort-wide).

CPU on purpose (GPU is busy with the 3-seed chain); the net is ~35k params.

Usage: python _w2_adabn.py [cell_artifact_dir]
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.modeling.binary_target import get_target
from src.modeling.cnn import CNNParams, SmallCNNClassifier, _PatchDataset
from src.modeling.evaluate import per_fold_metrics_classification
from src.modeling.loaders import gather_patches, iter_loio_folds

CELL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    REPO_ROOT / "models/cnn_bce_S64/40d843617a09e3c7/scale_S64_tfa_gt_1e-2_aug_none")
PATCH_PX = 64
SCALE_IDX = 3
DATASET_DIR = "dataset_v2"


def adabn_reestimate(net: torch.nn.Module, patches: np.ndarray, batch_size: int = 256) -> None:
    """Reset BN running stats and re-estimate them on `patches` (train mode, no grad).

    momentum=None makes BatchNorm accumulate a cumulative (equal-weight) average
    over all batches seen since reset -- one pass over the image suffices.
    """
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    ds = _PatchDataset(patches, np.zeros(patches.shape[0], dtype=np.float32),
                       augment=False, rng_seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    net.train()
    with torch.no_grad():
        for xb, _ in loader:
            net(xb)
    net.eval()


def main() -> int:
    target = get_target("fa_gt_1e-2")
    rows = []
    pooled_y, pooled_base, pooled_ada = [], [], []
    for fold in iter_loio_folds("loio_nfold", scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR):
        held = fold.held_out_obs_ids[0]
        params = CNNParams(patch_size_px=PATCH_PX, dataset_dir=DATASET_DIR,
                           aug_cell="none", device="cpu")
        model = SmallCNNClassifier(params=params)
        model.load(CELL_DIR / f"fold_{held}" / "state_dict.pt")
        y_test = target.binarize(fold.y_test).astype(np.float64)

        model.bind_predict_data(fold.keys_test)
        X_dummy = np.empty((len(fold.keys_test), 0), dtype=np.float32)
        p_base = model.predict(X_dummy)

        patches, _ = gather_patches(fold.keys_test, PATCH_PX, dataset_dir=DATASET_DIR)
        adabn_reestimate(model._net, patches)
        p_ada = model.predict(X_dummy)

        m_base = per_fold_metrics_classification(
            y_test.astype(np.int8), p_base, held_out_obs_ids=fold.held_out_obs_ids)
        m_ada = per_fold_metrics_classification(
            y_test.astype(np.int8), p_ada, held_out_obs_ids=fold.held_out_obs_ids)
        rows.append({"held_out_obs_id": held, "auc_base": m_base["auc"],
                     "auc_adabn": m_ada["auc"], "n_pos": m_base["n_positive"],
                     "n_neg": m_base["n_negative"]})
        pooled_y.append(y_test); pooled_base.append(p_base); pooled_ada.append(p_ada)
        d = (m_ada["auc"] - m_base["auc"]) if np.isfinite(m_ada["auc"]) else float("nan")
        print(f"  {held}: auc {m_base['auc']:.3f} -> {m_ada['auc']:.3f} ({d:+.3f})", flush=True)

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "models/_sweep_cnn/_adabn_cellA.parquet"
    df.to_parquet(out, index=False)
    y = np.concatenate(pooled_y).astype(int)
    pb, pa = np.concatenate(pooled_base), np.concatenate(pooled_ada)
    d = (df.auc_adabn - df.auc_base).dropna()
    from scipy import stats
    p = stats.wilcoxon(d, zero_method="wilcox").pvalue
    print(f"\nper-image AUC: base median {df.auc_base.median():.4f} -> "
          f"adabn {df.auc_adabn.median():.4f}; paired delta mean {d.mean():+.4f} "
          f"median {d.median():+.4f} win {(d > 0).mean():.2f} p={p:.4f}")
    print(f"pooled PR-AUC: base {average_precision_score(y, pb):.4f} -> "
          f"adabn {average_precision_score(y, pa):.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
