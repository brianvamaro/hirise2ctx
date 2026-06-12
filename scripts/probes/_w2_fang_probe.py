"""W2 Phase 2 lead bet (PLAN_CNN.md 5.1): Fang-ViT frozen GeM embeddings as LightGBM columns.

Standard LOIO harness (loio_nfold, scale_idx 3 = S=64, target fa_gt_1e-2,
lightgbm_classification with the Tier-1 refresh hyperparameters). Variants:

    t1_gem64    Tier-1 52 features + 768 GeM cols from the 64-px input
    t1_gem192   Tier-1 52 features + 768 GeM cols from the 192-px (3x3) input
    emb_only    the 2x768 GeM cols alone (does the FM carry signal by itself?)

Embeddings come from dataset_v2/fang_embeddings/ (written by _w2_fang_embed.py);
192-px-invalid rows carry NaN embedding columns, which LightGBM handles natively
(no row loss vs the Tier-1 reference).

Verdict block per variant, computed exactly as in _w2_seed_ensemble.py: pooled
PR-AUC / prec@5% vs Tier-1 (gate: dPR >= +0.03), paired per-image dAUC vs the
Tier-1 summary on dossier validity-passing images (gate: median >= +0.05,
Wilcoxon p < 0.05), plus per-attributed_cause means. Reference bars only --
this is an exploratory probe, not a promotion claim.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_w2_fang_probe.py
    ... [--variants t1_gem64 ...] [--pool gem]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling.binary_target import get_target
from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, LightGBMClassification, snapshot_params
from src.modeling.loaders import iter_loio_folds

DATASET_DIR = REPO_ROOT / "dataset_v2"
EMB_DIR = DATASET_DIR / "fang_embeddings"
OUT_ROOT = REPO_ROOT / "models" / "fang_probe"
SCHEME = "loio_nfold"
SCALE_IDX = 3  # S=64
TARGET_ID = "fa_gt_1e-2"

T1_PREDS = REPO_ROOT / ("models/lightgbm_classification/99de85c1ad2a72e6/"
                        "scale_S64_tfa_gt_1e-2/predictions.parquet")
T1_SUMMARY = REPO_ROOT / "models/_sweep_binary/20260611T214042Z/summary.parquet"
DOSSIER = DATASET_DIR / "w1_dossier.parquet"

VARIANT_SOURCES = {
    "t1_gem64": ("t1", "64"),
    "t1_gem192": ("t1", "192"),
    "t1_gem64_gem192": ("t1", "64", "192"),
    "emb_only": ("64", "192"),
}


# ============================================================================
# Embedding bank: one (n_total, 768) float32 matrix per input px, plus a
# (obs_id, ti, tj) -> row index for joining onto fold keys.
# ============================================================================


class EmbeddingBank:
    def __init__(self, pool: str, pxs: tuple[int, ...] = (64, 192)):
        self.pool = pool
        self.mats: dict[int, np.ndarray] = {}
        self.index: pd.DataFrame | None = None
        frames = []
        for px in pxs:
            blocks, rows = [], []
            for f in sorted(EMB_DIR.glob(f"*_P{px}.npz")):
                z = np.load(f)
                obs = f.name[: -len(f"_P{px}.npz")]
                emb = z[pool].astype(np.float32)
                valid = z["valid"].astype(bool)
                emb[~valid] = np.nan  # P64 is all-valid; P192 margin rows -> NaN columns
                blocks.append(emb)
                rows.append(pd.DataFrame({"obs_id": obs, "ti": z["ti"], "tj": z["tj"]}))
            self.mats[px] = np.concatenate(blocks, axis=0)
            idx = pd.concat(rows, ignore_index=True)
            idx[f"row{px}"] = np.arange(len(idx))
            frames.append(idx)
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on=["obs_id", "ti", "tj"], validate="one_to_one")
        self.index = merged

    def lookup(self, keys: pd.DataFrame, px: int) -> np.ndarray:
        j = keys[["obs_id", "ti", "tj"]].merge(
            self.index[["obs_id", "ti", "tj", f"row{px}"]],
            on=["obs_id", "ti", "tj"], how="left", validate="one_to_one")
        rows = j[f"row{px}"].to_numpy()
        assert not np.isnan(rows).any(), "embedding bank is missing tiles present in the fold"
        return self.mats[px][rows.astype(np.int64)]


def make_fold_iter(bank: EmbeddingBank, sources: tuple[str, ...]):
    """Wrap the default LOIO iterator, rebuilding X from the requested sources."""

    def _it():
        for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR):
            tr_parts, te_parts, names = [], [], []
            for src in sources:
                if src == "t1":
                    tr_parts.append(fold.X_train)
                    te_parts.append(fold.X_test)
                    names.extend(fold.feature_names)
                else:
                    px = int(src)
                    tr_parts.append(bank.lookup(fold.keys_train, px))
                    te_parts.append(bank.lookup(fold.keys_test, px))
                    names.extend(f"fang_{bank.pool}{px}_{i:03d}" for i in range(768))
            yield replace(
                fold,
                X_train=np.concatenate(tr_parts, axis=1),
                X_test=np.concatenate(te_parts, axis=1),
                feature_names=names,
            )

    return _it


# ============================================================================
# Verdict vs the Tier-1 reference
# ============================================================================


def per_image_auc(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for obs, g in df.groupby("obs_id"):
        y = g["y_true"].to_numpy()
        out[obs] = roc_auc_score(y, g[col].to_numpy()) if 0 < y.sum() < y.size else np.nan
    return pd.Series(out)


def verdict(variant: str, preds: pd.DataFrame) -> dict:
    t1 = pd.read_parquet(T1_PREDS, columns=["obs_id", "ti", "tj", "y_true", "y_pred"])
    t1 = t1.rename(columns={"y_pred": "t1_prob"})
    df = preds[["obs_id", "ti", "tj", "y_true", "y_pred"]].merge(
        t1.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
    assert len(df) == len(preds), f"join loss vs T1: {len(preds)} -> {len(df)}"
    y = df["y_true"].to_numpy().astype(int)
    k = max(1, int(0.05 * y.size))

    dossier = pd.read_parquet(DOSSIER)
    vok = set(dossier[dossier.validity_ok].index)
    t1_auc = pd.read_parquet(T1_SUMMARY).set_index("held_out_obs_id")["auc"]

    out = {}
    for col, label in (("t1_prob", "tier1_ref"), ("y_pred", variant)):
        s = df[col].to_numpy()
        pr = float(average_precision_score(y, s))
        p5 = float(y[np.argsort(-s)[:k]].mean())
        aucs = per_image_auc(df, col)
        row = {"pooled_pr_auc": pr, "prec_at_5": p5, "med_auc": float(aucs.median())}
        if col != "t1_prob":
            d = (aucs - t1_auc).dropna()
            d_v = d[[o in vok for o in d.index]]
            try:
                pval = float(stats.wilcoxon(d_v, zero_method="wilcox").pvalue)
            except ValueError:
                pval = float("nan")
            row.update({
                "dauc_median_v": float(d_v.median()),
                "dauc_win_v": float((d_v > 0).mean()),
                "wilcoxon_p": pval,
                "gate_per_image": bool(d_v.median() >= 0.05 and pval < 0.05),
                "gate_pooled": bool(pr - out["tier1_ref"]["pooled_pr_auc"] >= 0.03),
            })
            cause = dossier["attributed_cause"].reindex(d.index).fillna("unclassified")
            row["dauc_by_cause"] = d.groupby(cause).mean().round(4).to_dict()
            row["per_image_dauc"] = {o: round(float(v), 4) for o, v in d.sort_values().items()}
        out[label] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["t1_gem64", "t1_gem192", "emb_only"],
                    choices=sorted(VARIANT_SOURCES))
    ap.add_argument("--pool", default="gem", choices=["cls", "mean", "gem"])
    args = ap.parse_args()

    target = get_target(TARGET_ID)
    params = LGBMParams(n_estimators=400, learning_rate=0.05, early_stopping_rounds=40)
    bank = EmbeddingBank(args.pool)
    print(f"embedding bank: {len(bank.index)} tiles, pools={args.pool}, "
          f"P192 NaN rows: {int(np.isnan(bank.mats[192][:, 0]).sum())}\n", flush=True)

    for variant in args.variants:
        sources = VARIANT_SOURCES[variant]
        snapshot = {
            "variant": f"fang_probe_{variant}", "task": "classification",
            "target_id": TARGET_ID, "scheme": SCHEME, "dataset_dir": "dataset_v2",
            "scale_idx": SCALE_IDX, "tile_size_px": 64,
            "pool": args.pool, "sources": list(sources),
            "checkpoint": "models/pretrained/mars-mae-dino-vit-base-v1.pth (Zenodo 18180801)",
            "model": snapshot_params("lightgbm_classification", params),
        }
        cfg_hash = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()[:16]
        snapshot["config_hash"] = cfg_hash
        out_dir = OUT_ROOT / variant / cfg_hash

        t0 = time.monotonic()
        print(f"=== {variant} (pool={args.pool}, sources={sources}) ===", flush=True)
        result = run_loio(
            lambda: LightGBMClassification(params=params),
            binarize=target.binarize, task="classification",
            fold_iter=make_fold_iter(bank, sources),
            snapshot=snapshot, verbose=True,
        )
        write_run_artifacts(result, out_dir)
        v = verdict(variant, result.predictions)
        (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
        print(f"\n  [{variant}] {time.monotonic() - t0:.0f} s -> {out_dir.relative_to(REPO_ROOT)}")
        for label, row in v.items():
            slim = {k: (round(val, 4) if isinstance(val, float) else val)
                    for k, val in row.items() if k not in ("per_image_dauc",)}
            print(f"  {label}: {json.dumps(slim, default=str)}")
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
