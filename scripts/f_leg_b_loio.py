"""F pilot leg B — LOIO skill gate: baseline vs F (calibrated-frame) embeddings.

Runs the frozen deployable recipe (mlp_ens3, S=32, GeM, fa_gt_1e-2) under leave-one-
image-out CV on two embedding stores:
  baseline  dataset_v2/fang_embeddings      (mosaic-trained, un-normalized)
  F         dataset_v2/fang_embeddings_f    (calibrated-frame, perframe-normalized)

Gate: F's median per-image AUC must not drop materially vs baseline (≥ −0.02).
The A1 cycle measured Δ = −0.024 (marginal FAIL at −0.02 threshold).  F removes
the normalization artifact at source, so the expectation is a smaller or zero penalty.

Both stores are restricted to the obs_ids present in BOTH (fair Δ): images whose
CTX frames failed ISIS/extract are dropped from train AND test on both sides
(fang_columns_for_keys asserts on any key missing from a store, so unrestricted
folds would crash — and an asymmetric cohort would bias the medians anyway).

Run (laptop):
  conda run -n geospatial python scripts/f_leg_b_loio.py                                  # perframe
  conda run -n geospatial python scripts/f_leg_b_loio.py --f-store fang_embeddings_f_minnaert
"""
from __future__ import annotations

import argparse
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


def restrict_fold(fold, avail: set[str]):
    """Drop train/test rows whose obs_id is not in `avail`; None if no test rows left."""
    from dataclasses import replace as dc_replace

    mte = fold.keys_test["obs_id"].isin(avail).to_numpy()
    if not mte.any():
        return None
    mtr = fold.keys_train["obs_id"].isin(avail).to_numpy()
    return dc_replace(
        fold,
        X_train=fold.X_train[mtr],
        y_train=fold.y_train[mtr].reset_index(drop=True),
        groups_train=fold.groups_train[mtr],
        keys_train=fold.keys_train[mtr].reset_index(drop=True),
        X_test=fold.X_test[mte],
        y_test=fold.y_test[mte].reset_index(drop=True),
        groups_test=fold.groups_test[mte],
        keys_test=fold.keys_test[mte].reset_index(drop=True),
    )


def run_store(store_name: str, avail: set[str],
              nuisance_basis: np.ndarray | None = None) -> pd.DataFrame:
    target = get_target(TARGET)
    store = load_fang_store(PX, pool=POOL, dataset_dir=DATASET_DIR, store_name=store_name)
    rows = []
    for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR):
        fold = restrict_fold(fold, avail)
        if fold is None:
            continue  # held-out image absent from one of the stores
        f = augment_fold_with_fang(fold, px=PX, pool=POOL, dataset_dir=DATASET_DIR,
                                   replace=True, store=store)
        ytr = target.binarize(f.y_train).astype(np.float32)
        yte = target.binarize(f.y_test).astype(int)
        head = DeployableHead(recipe=dict(target_id=TARGET),
                              nuisance_basis=nuisance_basis)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-store", default="fang_embeddings_f",
                    help="F embedding store to gate (e.g. fang_embeddings_f_minnaert)")
    ap.add_argument("--nuisance-basis", default=None,
                    help="H2: npz with a 'basis' (768, N) array; its first --nuisance-k "
                         "columns are projected out of the F-store head (baseline is left raw)")
    ap.add_argument("--nuisance-k", type=int, default=0,
                    help="H2: number of nuisance directions to remove from the F store")
    ap.add_argument("--tag-suffix", default="",
                    help="append to output CSV names (keep k-sweep runs distinct)")
    args = ap.parse_args()
    f_store = args.f_store
    # output filenames: default store keeps the original names (notebook 27 reads them)
    tag = "" if f_store == "fang_embeddings_f" else f_store.replace("fang_embeddings_f", "")
    tag += args.tag_suffix

    basis = None
    if args.nuisance_basis and args.nuisance_k > 0:
        basis = np.load(args.nuisance_basis)["basis"][:, :args.nuisance_k]
        print(f"H2: F-store head removes top-{args.nuisance_k} nuisance directions "
              f"({Path(args.nuisance_basis).name})")

    f_dir = DATASET_DIR / f_store
    if not f_dir.exists() or not any(f_dir.glob("*_P96.npz")):
        print(f"ERROR: {f_dir} is empty or missing.\n"
              "Run f_leg_b_embed.py first to generate the F embeddings.")
        sys.exit(1)

    def store_obs(name: str) -> set[str]:
        return {p.name[: -len("_P96.npz")]
                for p in (DATASET_DIR / name).glob("*_P96.npz")}

    b_obs, f_obs = store_obs(BASELINE), store_obs(f_store)
    avail = b_obs & f_obs
    print(f"obs_ids: baseline {len(b_obs)}, F {len(f_obs)}, common {len(avail)}")
    if b_obs - f_obs:
        print(f"  missing from F (dropped from BOTH stores): "
              f"{', '.join(sorted(b_obs - f_obs))}")

    allrows, results = [], {}
    for store in (BASELINE, f_store):
        print(f"\n=== LOIO: {store} ===", flush=True)
        # H2 projection applies ONLY to the F store; the mosaic baseline stays raw
        # (its embeddings live in a different space — the nuisance basis is F-derived).
        df = run_store(store, avail, nuisance_basis=basis if store == f_store else None)
        allrows.append(df)
        s, aucs = summarize(df, store)
        results[store] = (s, aucs)

    FIG.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(allrows, ignore_index=True)
    combined.to_csv(FIG / f"f_leg_b_loio_preds{tag}.csv", index=False)
    summ = pd.DataFrame([results[BASELINE][0], results[f_store][0]])
    summ.to_csv(FIG / f"f_leg_b_loio_summary{tag}.csv", index=False)

    b_s, b_aucs = results[BASELINE]
    f_s, f_aucs = results[f_store]

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
        print(f"Investigate: check fraction of tiles with valid coverage in {f_store}/")

    print(f"\nFull results: {FIG}/f_leg_b_loio_summary{tag}.csv")


if __name__ == "__main__":
    main()
