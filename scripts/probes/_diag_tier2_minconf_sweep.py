"""Tier-2 L2 label-noise sweep (PLAN_Calibration Stage 2b): does filtering BoulderNet
detections by confidence (`score`) shrink the aleatoric floor → less compression /
better ranking?

CLAUDE.md §11 open item: the `.dbf` carries `score` (0.10-0.83, mean 0.41); the
pipeline kept `min_confidence: null`. Cleaner labels (drop low-confidence, likely-FP
detections) should narrow p(y|x). We regenerate the Stage-4 labels at each threshold
(cheap, cached Stage 1/2/3 inputs — no downloads), keep the same `min_size_m=1.4105`,
swap the new `fractional_area` into the frozen emb-only S=32 LOIO folds by
(obs_id, ti, tj) — the tile grid is detection-independent so the cached embeddings
still join — and re-run the same mlp_reg. Scored raw AND +quantile-match, paired
per-image Wilcoxon vs the unfiltered (`none`) labels.

Regen is cached under cache/minconf_sweep/. GPU torch for the LOIO. ~1.5 h cold.
"""
import copy
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probes"))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; precede numpy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src import manifest as M
from src.config import load_config
from src.labeling import stage4_one_image, LABELS_SUBDIR
from src.modeling.loaders import iter_loio_folds, augment_fold_with_fang, load_fang_store
from src.calibration import compression_metrics, quantile_match, loio_calibrate
from _fm_tier2_regression import MLPRegressorEnsemble

DATASET_DIR = REPO / "dataset_v2"
SCRATCH = REPO / "cache_v2" / "minconf_sweep"
OUT = REPO / "models" / "fang_tier2" / "l1_bakeoff"
CONFIG = "config_v2.yaml"   # dataset_v2 was built from this (manifest hirise_40_vclaire.csv)
BATCH = 4096
THRESHOLDS = [(None, "none"), (0.5, "conf050"), (0.7, "conf070")]


def regen_labels(cfg, manifest, obs_ids, min_conf, label):
    """Regenerate S=32 labels for `obs_ids` at `min_confidence=min_conf` into a scratch
    dir; cached (skip if all parquets present). Returns the labels dir."""
    out_dir = SCRATCH / label
    labels_dir = out_dir / LABELS_SUBDIR
    have = {p.stem for p in labels_dir.glob("*.parquet")} if labels_dir.exists() else set()
    todo = [o for o in obs_ids if o not in have]
    if not todo:
        print(f"  [{label}] regen cached ({len(have)} images)", flush=True)
        return labels_dir
    lab_cfg = copy.deepcopy(cfg["labeling"])
    lab_cfg["detection_filters"] = {"min_confidence": min_conf, "min_size_m": 1.4105}
    mdf = manifest.set_index("ObsId")
    print(f"  [{label}] regenerating {len(todo)} images (min_confidence={min_conf}) ...", flush=True)
    t0 = time.monotonic()
    for i, obs in enumerate(todo, 1):
        try:
            stage4_one_image(obs, cache_dir=cfg.cache_dir, output_dir=out_dir,
                             manifest_row=mdf.loc[obs], target_crs=cfg["target_crs"],
                             labeling_cfg=lab_cfg, config_hash=f"minconf_{label}",
                             apply_coreg_shift=True)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"    {obs}: FAILED ({e})", flush=True)
        if i % 10 == 0:
            print(f"    {i}/{len(todo)} [{time.monotonic()-t0:.0f}s]", flush=True)
    print(f"  [{label}] regen done [{time.monotonic()-t0:.0f}s]", flush=True)
    return labels_dir


def lookup(labels_dir):
    parts = []
    for p in labels_dir.glob("*.parquet"):
        d = pd.read_parquet(p)
        parts.append(d[d.tile_size_px == 32][["obs_id", "ti", "tj", "fractional_area"]])
    return pd.concat(parts, ignore_index=True)


def remap(keys_df, lut, fallback):
    """fractional_area for keys_df rows from `lut`; NaN (missing key) → fallback (the
    fold's own fa, so a remap miss can't fabricate a zero)."""
    m = keys_df[["obs_id", "ti", "tj"]].merge(lut, on=["obs_id", "ti", "tj"], how="left")
    fa = m["fractional_area"].to_numpy()
    miss = ~np.isfinite(fa)
    if miss.any():
        fa[miss] = fallback[miss]
    return fa, int(miss.sum())


def inner_val(fold, frac=0.1, seed=0):
    return np.random.default_rng(seed).random(len(fold.X_train)) < frac


def run_threshold(folds, lut):
    parts, misses = [], 0
    for f in folds:
        ytr, m1 = remap(f.keys_train, lut, f.y_train["fractional_area"].to_numpy())
        yte, m2 = remap(f.keys_test, lut, f.y_test["fractional_area"].to_numpy())
        misses += m1 + m2
        vm = inner_val(f)
        mdl = MLPRegressorEnsemble(transform="identity", clip_max=1.0, batch=BATCH)
        mdl.fit(f.X_train[~vm], ytr[~vm], eval_set=(f.X_train[vm], ytr[vm]))
        d = f.keys_test[["obs_id", "ti", "tj"]].copy()
        d["y_true"] = yte
        d["y_pred"] = mdl.predict(f.X_test)
        parts.append(d)
    return pd.concat(parts, ignore_index=True), misses


def per_img(df):
    return {o: spearmanr(g.y_true, g.y_pred).correlation for o, g in df.groupby("obs_id")
            if g.y_true.nunique() > 1}


def main():
    regen_only = "--regen-only" in sys.argv
    cfg = load_config(str(REPO / CONFIG))
    manifest = M.load_manifest(cfg.manifest_path)

    if regen_only:
        # CPU-only label regen (no GPU, no folds) — run concurrently with a GPU job to
        # warm the cache; the full probe then skips regen and goes straight to the LOIO.
        obs_ids = sorted(p.stem for p in (DATASET_DIR / "labels").glob("*.parquet"))
        print(f"--regen-only: {len(obs_ids)} images x {len(THRESHOLDS)} thresholds", flush=True)
        for min_conf, label in THRESHOLDS:
            regen_labels(cfg, manifest, obs_ids, min_conf, label)
        print("regen-only done.", flush=True)
        return

    print("Building emb-only S=32 folds ...", flush=True)
    store = load_fang_store(96, pool="gem", dataset_dir=DATASET_DIR)
    folds = [augment_fold_with_fang(f, px=96, dataset_dir=DATASET_DIR, replace=True, store=store)
             for f in iter_loio_folds("loio_nfold", scale_idx=2, dataset_dir=DATASET_DIR)]
    obs_ids = sorted({o for f in folds for o in f.held_out_obs_ids})
    print(f"  {len(folds)} folds, {len(obs_ids)} images\n", flush=True)

    base_rho, rows = None, []
    for min_conf, label in THRESHOLDS:
        labels_dir = regen_labels(cfg, manifest, obs_ids, min_conf, label)
        lut = lookup(labels_dir)
        # share of true rich tiles retained vs the unfiltered set (a label-noise readout)
        rich = float((lut.fractional_area > 1e-2).mean())
        t0 = time.monotonic()
        df, misses = run_threshold(folds, lut)
        m = compression_metrics(df.y_true.to_numpy(), df.y_pred.to_numpy())
        cal = loio_calibrate(df, lambda rp, rt, hp: quantile_match(hp, rp, rt))
        mc = compression_metrics(df.y_true.to_numpy(), cal)
        rho = per_img(df)
        if label == "none":
            base_rho, ptxt = rho, "(baseline)"
        else:
            keys = [k for k in base_rho if k in rho and np.isfinite(base_rho[k]) and np.isfinite(rho[k])]
            a = np.array([rho[k] for k in keys]); b = np.array([base_rho[k] for k in keys])
            ptxt = f"paired d={np.median(a-b):+.3f} wins {int((a>b).sum())}/{len(keys)} p={wilcoxon(a,b).pvalue:.3f}"
        print(f"{label:>8} (rich {rich:.1%}, miss {misses}) [{time.monotonic()-t0:.0f}s] | "
              f"raw rho {m['spearman']:.3f} (img {np.nanmedian(list(rho.values())):.3f}) top {m['top_ratio']:.2f} "
              f"near0 {m['near_zero_pred']:.1%} | +qm rho {mc['spearman']:.3f} top {mc['top_ratio']:.2f} | {ptxt}", flush=True)
        rows.append({"label": label, "min_conf": min_conf, "rich_share": rich,
                     "raw_top": m["top_ratio"], "raw_perimg_rho": float(np.nanmedian(list(rho.values()))),
                     "raw_pooled_rho": m["spearman"], "qm_top": mc["top_ratio"], "near0_true": m["near_zero_true"]})
        df.to_parquet(OUT / f"preds_minconf_{label}.parquet")
    pd.DataFrame(rows).to_csv(OUT / "minconf_scorecard.csv", index=False)
    print(f"\nwrote minconf_scorecard.csv -> {OUT.relative_to(REPO)}")
    print("read: does cleaner (higher-confidence) labels raise per-img rho / cut compression "
          "WITHOUT a paired ranking loss? Trade: fewer boulders -> sparser, shifted target.")


if __name__ == "__main__":
    main()
