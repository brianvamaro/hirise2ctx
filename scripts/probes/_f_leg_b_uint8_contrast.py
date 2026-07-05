"""Leg B: post-normalization uint8 contrast the embedder ACTUALLY saw, F vs mosaic.

For every embedded obs_id: build the F composite uint8 via the embed code path
(f_leg_b_embed.composite_crops — actual last-write-wins canvas + single perframe
normalization) and read the mosaic ctx_window uint8; compare their IQRs over the
window's valid pixels.  Join per-image ΔAUC.  If craters have F-IQR << mosaic-IQR
while improvers match, the lost-texture-contrast story is confirmed on the real
quantities (the earlier diag if_iqr concatenated crops, double-counting overlaps).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
import rasterio

import f_leg_b_embed as fe

LABELS_DIR = REPO / "dataset_v2" / "labels"
DIAG = REPO / "reports" / "f_leg_b" / "diag_per_image.csv"


def win_iqr(a: np.ndarray) -> float:
    v = a[a > 0].astype(np.float32)
    if v.size < 100:
        return float("nan")
    q75, q25 = np.percentile(v, [75, 25])
    return float(q75 - q25)


def main() -> None:
    diag = pd.read_csv(DIAG)
    rows = []
    for obs in diag["obs_id"]:
        sc = json.loads((LABELS_DIR / f"{obs}.json").read_text(encoding="utf-8"))
        with rasterio.open(sc["ctx_window_tif"]) as ds:
            mosaic = ds.read(1)
            H, W = ds.height, ds.width
        comp = fe.composite_crops(obs, int(sc["mosaic_row_origin"]),
                                  int(sc["mosaic_col_origin"]), H, W)
        rows.append(dict(obs_id=obs, mosaic_iqr=win_iqr(mosaic), f_iqr=win_iqr(comp)))
        print(f"  {obs}: mosaic IQR {rows[-1]['mosaic_iqr']:.1f}  "
              f"F IQR {rows[-1]['f_iqr']:.1f}", flush=True)

    df = diag.merge(pd.DataFrame(rows), on="obs_id")
    df["iqr_ratio"] = df["f_iqr"] / df["mosaic_iqr"]
    df = df.sort_values("d_auc")
    cols = ["obs_id", "d_auc", "auc_base", "auc_f", "mosaic_iqr", "f_iqr", "iqr_ratio"]
    pd.set_option("display.width", 150)
    print("\n" + df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSpearman d_auc vs uint8 iqr_ratio (F/mosaic): "
          f"{df['d_auc'].corr(df['iqr_ratio'], method='spearman'):+.3f}")
    out = REPO / "reports" / "f_leg_b" / "diag_uint8_contrast.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
