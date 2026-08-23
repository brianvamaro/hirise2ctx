"""A1 striping-mitigation SKILL GATE — LOIO per-image AUC, baseline vs A1-normalized embeddings.

Runs the frozen deployable recipe (`fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`, emb-only S=32, the
3-seed MLP ensemble = DeployableHead) under leave-one-image-out CV on two embedding stores:
  * baseline  `dataset_v2/fang_embeddings`      (un-normalized CTX)
  * A1         `dataset_v2/fang_embeddings_a1`   (per-frame robust offset+gain CTX, the mitigation)

Each LOIO fold = one held-out image -> one per-image meaningful AUC (rich/poor at fa>1e-2). The gate:
A1's median per-image AUC must not drop materially vs baseline (the mitigation must not cost skill).
Identical harness for both, so the baseline-vs-A1 delta is the decisive comparison.

Run: conda run -n geospatial python scripts/striping_a1_loio.py

**PLAN_FBuild §5.1 addition (2026-07-28).** The numbers of record here were produced under 38-image
folds, while every F-build number uses the 36 images common to the mosaic and F embedding stores — so
the A1 skill cell of the §5.1 scorecard was not comparable to the F cells, and post-hoc row filtering
cannot fix it (the folds trained on 37 images rather than 35). Brian ruled: re-run restricted to the
36. Use

    python scripts/striping_a1_loio.py --restrict-store fang_embeddings_f_minnaert_center --tag _36

which writes `striping_a1_loio_{preds,summary}_36.csv` and leaves the 38-image files of record
untouched.
"""
from __future__ import annotations

import argparse
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


def restrict_fold(fold, avail: set[str]):
    """Drop train/test rows whose obs_id is not in `avail`; None if no test rows left.

    Copied verbatim from scripts/f_leg_b_loio.py:48 so the two harnesses restrict IDENTICALLY —
    restricting both sides is what makes the folds train-regime-comparable to the F runs (filtering
    predictions afterwards leaves the folds trained on 37 images instead of 35). Note `y_train`/
    `y_test` are label DataFrames, not arrays, and `groups_test` must be subset too.
    """
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


def store_obs(name: str, dataset_dir: str | Path | None = None) -> set[str]:
    root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    return {p.name[: -len("_P96.npz")] for p in (root / name).glob("*_P96.npz")}


def run_store(store_name: str, avail: set[str] | None = None,
              dataset_dir: str | Path | None = None) -> pd.DataFrame:
    # Isolation criterion 4: the embedding-store root is an argument, so an A1 rebuild can
    # run against a scratch dataset tree instead of the live dataset_v2.
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    target = get_target(TARGET)
    store = load_fang_store(PX, pool=POOL, dataset_dir=dataset_dir, store_name=store_name)
    rows = []
    for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=dataset_dir):
        if avail is not None:
            fold = restrict_fold(fold, avail)
            if fold is None:
                continue
        f = augment_fold_with_fang(fold, px=PX, pool=POOL, dataset_dir=dataset_dir,
                                   replace=True, store=store)
        ytr = target.binarize(f.y_train).astype(np.float32)
        yte = target.binarize(f.y_test).astype(int)
        head = DeployableHead(recipe=dict(target_id=TARGET))
        head.fit(f.X_train, ytr, groups=f.groups_train, obs_to_int=f.obs_to_int, verbose=False)
        p = head.predict(f.X_test)
        kt = f.keys_test
        obs = kt["obs_id"].to_numpy()
        keep = np.isfinite(p) & np.isfinite(yte)
        # ti/tj are CARRIED, not dropped. The audit's rebuild-DAG bullet: "every arm's
        # prediction artifact must retain unique tile keys" -- `bank_calibration.py` joins
        # on (obs_id, ti, tj) and asserts one-to-one, so an artifact without them cannot
        # calibrate the A1 arm at all. DECISIONS 2026-08-21.
        rows.append(pd.DataFrame({"obs_id": obs[keep],
                                  "ti": kt["ti"].to_numpy()[keep],
                                  "tj": kt["tj"].to_numpy()[keep],
                                  "y": yte[keep], "p": p[keep],
                                  "store": store_name}))
        o0 = obs[0]
        auc = (roc_auc_score(yte[keep], p[keep])
               if len(np.unique(yte[keep])) == 2 else np.nan)
        print(f"  [{store_name}] {o0}: n={keep.sum()} pos={int(yte[keep].sum())} AUC={auc:.3f}",
              flush=True)
    return pd.concat(rows, ignore_index=True)



def write_arm_predictions(preds_all, out_dir, tag: str = "") -> dict:
    """Per-arm `predictions.parquet` in the BASELINE schema. Returns {store: path}.

    **DECISIONS 2026-08-21.** This driver used to emit only `obs_id, y, p`. The audit's
    rebuild-DAG bullet says "every arm's prediction artifact must retain unique tile keys",
    and `scripts/bank_calibration.py` joins on `(obs_id, ti, tj)` with `validate="one_to_one"`
    -- so without the keys the A1 arm simply cannot be calibrated, and step 9 is blocked.

    The skill-gate CSVs are left exactly as they were; these parquets are the calibration
    inputs. Duplicate keys are a hard error: a LOIO artifact has one row per tile by
    construction, so a duplicate means folds overlapped or a store was concatenated twice.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    written = {}
    for store, sub in preds_all.groupby("store"):
        arm_dir = out_dir / f"loio_{store}{tag}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        out = (sub[["obs_id", "ti", "tj", "y", "p"]]
               .rename(columns={"y": "y_true", "p": "y_pred"}))
        dup = int(out.duplicated(subset=["obs_id", "ti", "tj"]).sum())
        if dup:
            raise SystemExit(f"{store}: {dup} duplicate (obs_id, ti, tj) keys -- a LOIO "
                             f"prediction artifact needs exactly one row per tile")
        path = arm_dir / "predictions.parquet"
        out.to_parquet(path, index=False)
        written[store] = path
        print(f"  {store}: {len(out)} predictions -> {path}", flush=True)
    return written


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--restrict-store", default=None,
                    help="restrict folds to the obs_ids present in this embedding store "
                         "(e.g. fang_embeddings_f_minnaert_center -> the 36 common images)")
    ap.add_argument("--tag", default="", help="output filename suffix (e.g. _36)")
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR),
                    help="packaged-dataset + embedding-store root (isolation criterion 4)")
    ap.add_argument("--out-dir", default=str(FIG),
                    help="where the LOIO prediction + summary CSVs are written")
    args = ap.parse_args()

    avail = None
    if args.restrict_store:
        avail = (store_obs("fang_embeddings", args.dataset_dir)
                 & store_obs(args.restrict_store, args.dataset_dir))
        print(f"restricting folds to {len(avail)} obs common to fang_embeddings and "
              f"{args.restrict_store}", flush=True)
        if not args.tag:
            raise SystemExit("--restrict-store changes the numbers; pass --tag (e.g. --tag _36) so "
                             "the 38-image files of record are not overwritten")

    results, auc_by = {}, {}
    allrows = []
    for store in ("fang_embeddings", "fang_embeddings_a1"):
        print(f"=== LOIO over store: {store} ===", flush=True)
        df = run_store(store, avail, args.dataset_dir)
        allrows.append(df)
        s, aucs = summarize(df, store)
        results[store] = s
        auc_by[store] = aucs
    tag = args.tag
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_all = pd.concat(allrows, ignore_index=True)
    preds_all.to_csv(out_dir / f"striping_a1_loio_preds{tag}.csv", index=False)
    write_arm_predictions(preds_all, out_dir, tag)
    summ = pd.DataFrame([results["fang_embeddings"], results["fang_embeddings_a1"]])
    summ.to_csv(out_dir / f"striping_a1_loio_summary{tag}.csv", index=False)
    print("\n=== SKILL GATE: baseline vs A1 ===")
    print(summ.to_string(index=False))
    b, a = results["fang_embeddings"], results["fang_embeddings_a1"]
    print(f"\nΔ median per-image AUC (A1 − baseline) = {a['median_auc'] - b['median_auc']:+.4f}")
    print(f"Δ pooled PR-AUC                        = {a['pooled_pr_auc'] - b['pooled_pr_auc']:+.4f}")
    verdict = "PASS (skill preserved)" if a["median_auc"] >= b["median_auc"] - 0.02 else "FAIL (skill dropped)"
    print(f"GATE: {verdict}")


if __name__ == "__main__":
    main()
