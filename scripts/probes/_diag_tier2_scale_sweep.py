"""Tier-2 L2 scale sweep (PLAN_Calibration Stage 2b): does a coarser operating
scale shrink p(y|x) → less compression and a higher *ranking* ceiling?

L2 is the only lever that raises the ranking ceiling (the post-qmatch residual is
ranking). Coarser tiles average over more area → higher SNR, less per-tile label
noise → the conditional p(y|x) narrows → the mean-seeking regressor compresses less.
This quantifies the compression-vs-resolution trade explicitly: the same frozen-
embedding mlp_reg (identity+MSE) at each scale, scored raw and +qmatch with per-
image Spearman.

Scales with a precomputed 3×-context Fang store on disk:
    S=32  scale_idx 2  px96   (the frozen operating scale)
    S=64  scale_idx 3  px192
(S=16 needs a P48 store, S=128 a P384 store + a 128-px label grid — both require a
fresh ViT embedding pass + Stage-4 label regen, out of scope for the cheap sweep.)

CPU/GPU torch. Launch with conda run --no-capture-output ... python -u.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probes"))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; precede numpy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.modeling.loaders import iter_loio_folds, augment_fold_with_fang, load_fang_store
from src.calibration import compression_metrics, quantile_match, loio_calibrate
from _fm_tier2_regression import MLPRegressorEnsemble

DATASET_DIR = REPO / "dataset_v2"
BATCH = 4096
SCALES = [(2, 96, 32), (3, 192, 64)]   # (scale_idx, fang_px, tile_px)


def inner_val(fold, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(len(fold.X_train)) < frac


def run_scale(scale_idx, px):
    store = load_fang_store(px, pool="gem", dataset_dir=DATASET_DIR)
    folds = [augment_fold_with_fang(f, px=px, dataset_dir=DATASET_DIR, replace=True, store=store)
             for f in iter_loio_folds("loio_nfold", scale_idx=scale_idx, dataset_dir=DATASET_DIR)]
    parts = []
    for f in folds:
        vm = inner_val(f)
        mdl = MLPRegressorEnsemble(transform="identity", clip_max=1.0, batch=BATCH)
        yv = f.y_train["fractional_area"].to_numpy()
        mdl.fit(f.X_train[~vm], yv[~vm], eval_set=(f.X_train[vm], yv[vm]))
        d = f.keys_test[["obs_id", "ti", "tj"]].copy()
        d["y_true"] = f.y_test["fractional_area"].to_numpy()
        d["y_pred"] = mdl.predict(f.X_test)
        parts.append(d)
    return pd.concat(parts, ignore_index=True), len(folds)


def main():
    print("Tier-2 L2 scale sweep (mlp_reg identity, emb-only, LOIO)\n", flush=True)
    rows = []
    for scale_idx, px, tile in SCALES:
        t0 = time.monotonic()
        df, nfold = run_scale(scale_idx, px)
        nz_true = float(np.mean(df.y_true.to_numpy() <= 0))
        m = compression_metrics(df.y_true.to_numpy(), df.y_pred.to_numpy())
        cal = loio_calibrate(df, lambda rp, rt, hp: quantile_match(hp, rp, rt))
        mc = compression_metrics(df.y_true.to_numpy(), cal)
        pim = df.groupby("obs_id").apply(
            lambda g: spearmanr(g.y_true, g.y_pred).correlation if g.y_true.nunique() > 1 else np.nan)
        pim = float(np.nanmedian(pim))
        meters = tile * 5
        print(f"S={tile:>3} ({meters} m, {nfold} folds, true near-zero {nz_true:.1%}) "
              f"[{time.monotonic()-t0:.0f}s]", flush=True)
        print(f"   raw  : rho {m['spearman']:.3f} (per-img {pim:.3f}) top {m['top_ratio']:.2f} "
              f"near0 {m['near_zero_pred']:.1%} L1 {m['marginal_l1']:.4f}", flush=True)
        print(f"   +qm  : rho {mc['spearman']:.3f} top {mc['top_ratio']:.2f} "
              f"near0 {mc['near_zero_pred']:.1%} L1 {mc['marginal_l1']:.4f}\n", flush=True)
        rows.append({"tile_px": tile, "meters": meters, "per_img_rho": pim,
                     "raw_top_ratio": m["top_ratio"], "raw_spearman": m["spearman"],
                     "raw_marginal_l1": m["marginal_l1"], "qm_top_ratio": mc["top_ratio"],
                     "qm_marginal_l1": mc["marginal_l1"]})
    out = REPO / "models" / "fang_tier2" / "l1_bakeoff" / "scale_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO)}", flush=True)
    print("read: does coarser S raise per-img rho (ranking ceiling) AND top_ratio "
          "(less raw compression)? That is the L2 trade vs spatial resolution.", flush=True)


if __name__ == "__main__":
    main()
