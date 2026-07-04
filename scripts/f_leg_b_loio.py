"""F pilot leg B — LOIO skill gate: baseline vs F (calibrated-frame) embeddings.

Runs the frozen deployable recipe (mlp_ens3, S=32, GeM, fa_gt_1e-2) under leave-one-
image-out CV on two embedding stores:
  baseline  dataset_v2/fang_embeddings      (mosaic-trained, un-normalized)
  F         dataset_v2/fang_embeddings_f    (calibrated-frame, perframe-normalized)

Gate: F's median per-image AUC must not drop materially vs baseline (≥ −0.02).
The A1 cycle measured Δ = −0.024 (marginal FAIL at −0.02 threshold).  F removes
the normalization artifact at source, so the expectation is a smaller or zero penalty.

Run (laptop):
  conda run -n geospatial python scripts/f_leg_b_loio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

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
GATE_DELTA = -0.02   # minimum acceptable Δ median AUC (same as A1 cycle)
BASELINE = "fang_embeddings"
F_STORE = "fang_embeddings_f"


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
        head.fit(f.X_train, ytr, groups=f.groups_train, obs_to_int=f.obs_to_int,
                 verbose=False)
        p = head.predict(f.X_test)
        obs = f.keys_test["obs_id"].to_numpy()
        keep = np.isfinite(p) & np.isfinite(yte)
        rows.append(pd.DataFrame({"obs_id": obs[keep], "y": yte[keep], "p": p[keep],
                                  "store": store_name}))
        o0 = obs[0]
        auc = roc_auc_score(yte[keep], p[keep]) if len(np.unique(yte[keep])) == 2 else np.nan
        print(f"  [{store_name}] {o0}: n={keep.sum()} pos={int(yte[keep].sum())} "
              f"AUC={auc:.3f}", flush=True)
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame, label: str) -> tuple[dict, np.ndarray]:
    aucs = []
    for _, g in df.groupby("obs_id"):
        if g["y"].nunique() == 2:
            aucs.append(roc_auc_score(g["y"], g["p"]))
    aucs = np.array(aucs)
    pooled_pr = average_precision_score(df["y"], df["p"])
    s = dict(store=label, n_img=len(aucs), median_auc=float(np.median(aucs)),
             mean_auc=float(np.mean(aucs)), frac_ge_0p7=float(np.mean(aucs >= 0.7)),
             pooled_pr_auc=float(pooled_pr))
    return s, aucs


def main() -> None:
    f_dir = DATASET_DIR / F_STORE
    if not f_dir.exists() or not any(f_dir.glob("*_P96.npz")):
        print(f"ERROR: {f_dir} is empty or missing.\n"
              "Run f_leg_b_embed.py first to generate the F embeddings.")
        sys.exit(1)

    allrows, results = [], {}
    for store in (BASELINE, F_STORE):
        print(f"\n=== LOIO: {store} ===", flush=True)
        df = run_store(store)
        allrows.append(df)
        s, aucs = summarize(df, store)
        results[store] = (s, aucs)

    FIG.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(allrows, ignore_index=True)
    combined.to_csv(FIG / "f_leg_b_loio_preds.csv", index=False)
    summ = pd.DataFrame([results[BASELINE][0], results[F_STORE][0]])
    summ.to_csv(FIG / "f_leg_b_loio_summary.csv", index=False)

    b_s, b_aucs = results[BASELINE]
    f_s, f_aucs = results[F_STORE]

    print("\n=== SKILL GATE: baseline vs F ===")
    print(summ.to_string(index=False))
    delta_med = f_s["median_auc"] - b_s["median_auc"]
    delta_pr = f_s["pooled_pr_auc"] - b_s["pooled_pr_auc"]
    print(f"\nΔ median per-image AUC (F − baseline) = {delta_med:+.4f}  "
          f"(A1 reference: −0.024)")
    print(f"Δ pooled PR-AUC                        = {delta_pr:+.4f}")

    if f_s["median_auc"] >= b_s["median_auc"] + GATE_DELTA:
        print(f"\nGATE: PASS — skill preserved (Δ ≥ {GATE_DELTA:+.2f})")
        print("Next: rebuild deployable head on all F training data + run regional "
              "inference on 907 source frames.")
    else:
        print(f"\nGATE: FAIL — skill dropped by {delta_med:.4f} "
              f"(threshold {GATE_DELTA:+.2f})")
        print("Investigate: check fraction of tiles with valid coverage in fang_embeddings_f/")

    print(f"\nFull results: {FIG}/f_leg_b_loio_summary.csv")


if __name__ == "__main__":
    main()
