"""A1 striping-mitigation SKILL GATE — LOIO per-image AUC, baseline vs A1-normalized embeddings.

Runs the frozen deployable recipe (`fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`, emb-only S=32, the
3-seed MLP ensemble = DeployableHead) under leave-one-image-out CV on two embedding stores:
  * baseline  `dataset_v2/fang_embeddings`      (un-normalized CTX)
  * A1         `dataset_v2/fang_embeddings_a1`   (per-frame robust offset+gain CTX, the mitigation)

Each LOIO fold = one held-out image -> one per-image meaningful AUC (rich/poor at fa>1e-2). The gate:
A1's median per-image AUC must not drop materially vs baseline (the mitigation must not cost skill).
Identical harness for both, so the baseline-vs-A1 delta is the decisive comparison.

Run: conda run -n geospatial python scripts/striping_a1_loio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP/DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.modeling.binary_target import get_target
from src.modeling.loaders import augment_fold_with_fang, iter_loio_folds, load_fang_store
from src.modeling.mlp_head import DeployableHead

SCHEME, SCALE_IDX, PX, POOL = "loio_nfold", 2, 96, "gem"
DATASET_DIR = REPO / "dataset_v2"
TARGET = "fa_gt_1e-2"
FIG = REPO / "reports" / "figures"


def run_store(store_name: str) -> pd.DataFrame:
    target = get_target(TARGET)
    store = load_fang_store(PX, pool=POOL, dataset_dir=DATASET_DIR, store_name=store_name)
    rows = []
    for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR):
        f = augment_fold_with_fang(fold, px=PX, pool=POOL, dataset_dir=DATASET_DIR,
                                   replace=True, store=store)
        ytr = target.binarize(f.y_train).astype(np.float32)
        yte = target.binarize(f.y_test).astype(int)
        head = DeployableHead(recipe=dict(target_id=TARGET))
        head.fit(f.X_train, ytr, groups=f.groups_train, obs_to_int=f.obs_to_int, verbose=False)
        p = head.predict(f.X_test)
        obs = f.keys_test["obs_id"].to_numpy()
        keep = np.isfinite(p) & np.isfinite(yte)
        rows.append(pd.DataFrame({"obs_id": obs[keep], "y": yte[keep], "p": p[keep],
                                  "store": store_name}))
        o0 = obs[0]
        auc = (roc_auc_score(yte[keep], p[keep])
               if len(np.unique(yte[keep])) == 2 else np.nan)
        print(f"  [{store_name}] {o0}: n={keep.sum()} pos={int(yte[keep].sum())} AUC={auc:.3f}",
              flush=True)
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame, label: str) -> dict:
    aucs = []
    for o, g in df.groupby("obs_id"):
        if g["y"].nunique() == 2:
            aucs.append(roc_auc_score(g["y"], g["p"]))
    aucs = np.array(aucs)
    pooled_pr = average_precision_score(df["y"], df["p"])
    out = dict(store=label, n_img=len(aucs), median_auc=float(np.median(aucs)),
               mean_auc=float(np.mean(aucs)), frac_ge_0p7=float(np.mean(aucs >= 0.7)),
               pooled_pr_auc=float(pooled_pr))
    return out, aucs


def main():
    results, auc_by = {}, {}
    allrows = []
    for store in ("fang_embeddings", "fang_embeddings_a1"):
        print(f"=== LOIO over store: {store} ===", flush=True)
        df = run_store(store)
        allrows.append(df)
        s, aucs = summarize(df, store)
        results[store] = s
        auc_by[store] = aucs
    pd.concat(allrows, ignore_index=True).to_csv(FIG / "striping_a1_loio_preds.csv", index=False)
    summ = pd.DataFrame([results["fang_embeddings"], results["fang_embeddings_a1"]])
    summ.to_csv(FIG / "striping_a1_loio_summary.csv", index=False)
    print("\n=== SKILL GATE: baseline vs A1 ===")
    print(summ.to_string(index=False))
    b, a = results["fang_embeddings"], results["fang_embeddings_a1"]
    print(f"\nΔ median per-image AUC (A1 − baseline) = {a['median_auc'] - b['median_auc']:+.4f}")
    print(f"Δ pooled PR-AUC                        = {a['pooled_pr_auc'] - b['pooled_pr_auc']:+.4f}")
    verdict = "PASS (skill preserved)" if a["median_auc"] >= b["median_auc"] - 0.02 else "FAIL (skill dropped)"
    print(f"GATE: {verdict}")


if __name__ == "__main__":
    main()
