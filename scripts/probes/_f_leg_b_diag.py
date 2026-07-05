"""F leg B diagnostic: is the bimodal per-image ΔAUC explained by composite mechanics?

For each obs_id: coverage fraction of the composite (finite crop pixels / window),
number of crops, overlap fraction (pixels written by 2+ crops = seam risk), and the
composite's pre-normalization I/F IQR (contrast).  Joined against per-image AUC from
the LOIO gate run (baseline vs F) to see what separates the collapsed images from the improvers.

Run: conda run --no-capture-output -n geospatial python -u scripts/probes/_f_leg_b_diag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score

CROPS_DIR = REPO / "reports" / "f_leg_b" / "obs_crops"
LABELS_DIR = REPO / "dataset_v2" / "labels"
PREDS = REPO / "reports" / "figures" / "f_leg_b_loio_preds.csv"


def obs_stats(obs_id: str) -> dict | None:
    sc = json.loads((LABELS_DIR / f"{obs_id}.json").read_text(encoding="utf-8"))
    ctx_tif = Path(sc["ctx_window_tif"])
    if not ctx_tif.exists():
        return None
    with rasterio.open(ctx_tif) as ds:
        H, W = ds.height, ds.width

    crops = sorted(CROPS_DIR.glob(f"{obs_id}_*_ifcrop.tif"))
    n_written = np.zeros((H, W), dtype=np.uint8)
    vals = []
    for p in crops:
        with rasterio.open(p) as src:
            arr = src.read(1, out_shape=(H, W)).astype(np.float32)  # same grid
        fin = np.isfinite(arr) & (arr > 0)
        n_written[fin] += 1
        vals.append(arr[fin])
    covered = n_written > 0
    v = np.concatenate(vals) if vals else np.array([0.0])
    q75, q25 = np.percentile(v, [75, 25]) if v.size > 1 else (0, 0)
    return dict(
        obs_id=obs_id,
        n_crops=len(crops),
        coverage=float(covered.mean()),
        overlap=float((n_written > 1).mean()),
        if_iqr=float(q75 - q25),
        if_median=float(np.median(v)),
    )


def main() -> None:
    preds = pd.read_csv(PREDS)
    rows = []
    for (obs, store), g in preds.groupby(["obs_id", "store"]):
        if g["y"].nunique() == 2:
            rows.append(dict(obs_id=obs, store=store,
                             auc=roc_auc_score(g["y"], g["p"])))
    auc = pd.DataFrame(rows).pivot(index="obs_id", columns="store", values="auc")
    auc.columns = [c.replace("fang_embeddings_f", "auc_f")
                    .replace("fang_embeddings", "auc_base") for c in auc.columns]
    auc["d_auc"] = auc["auc_f"] - auc["auc_base"]

    stats = pd.DataFrame([s for o in auc.index if (s := obs_stats(o))])
    df = auc.reset_index().merge(stats, on="obs_id").sort_values("d_auc")

    pd.set_option("display.width", 160)
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"))

    print("\nSpearman correlations with d_auc:")
    for c in ("coverage", "overlap", "n_crops", "if_iqr", "if_median"):
        rho = df["d_auc"].corr(df[c], method="spearman")
        print(f"  {c:10s}  rho = {rho:+.3f}")

    out = REPO / "reports" / "f_leg_b" / "diag_per_image.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
