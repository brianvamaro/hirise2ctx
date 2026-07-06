"""Quantization-squeeze check: per-image uint8 IQR under the minnaert_w mapping vs ΔAUC.

Hypothesis: a fixed linear stretch shares the 254-DN budget across scenes whose medians
differ ~2.5x, so dim scenes get squeezed into a handful of DN — texture lost to uint8
quantization.  If true, the minnaert_w collapses should be exactly the scenes with tiny
post-mapping DN IQR.
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
from sklearn.metrics import roc_auc_score

import f_leg_b_embed as fe

LABELS_DIR = REPO / "dataset_v2" / "labels"
FIG = REPO / "reports" / "figures"

preds = pd.read_csv(FIG / "f_leg_b_loio_preds_minnaert_w.csv")
rows = []
for (obs, store), g in preds.groupby(["obs_id", "store"]):
    if g["y"].nunique() == 2:
        rows.append(dict(obs_id=obs, store=store, auc=roc_auc_score(g["y"], g["p"])))
auc = pd.DataFrame(rows).pivot(index="obs_id", columns="store", values="auc")
auc.columns = ["base", "f_w"]
auc["d_auc"] = auc["f_w"] - auc["base"]

obs_ids = sorted(auc.index)
ctx = fe.build_mapping_ctx("minnaert", obs_ids, pcts=(0.5, 99.5))

out = []
for obs in obs_ids:
    sc = json.loads((LABELS_DIR / f"{obs}.json").read_text(encoding="utf-8"))
    with rasterio.open(sc["ctx_window_tif"]) as ds:
        H, W = ds.height, ds.width
    w8 = fe.composite_crops(obs, int(sc["mosaic_row_origin"]),
                            int(sc["mosaic_col_origin"]), H, W,
                            mapping="minnaert", ctx=ctx)
    v = w8[w8 > 0].astype(np.float32)
    q75, q25 = np.percentile(v, [75, 25])
    out.append(dict(obs_id=obs, dn_iqr=float(q75 - q25), dn_med=float(np.median(v))))
    print(f"  {obs}: DN IQR {q75 - q25:.0f}  median {np.median(v):.0f}", flush=True)

df = auc.join(pd.DataFrame(out).set_index("obs_id")).sort_values("d_auc")
print("\n" + df.to_string(float_format=lambda x: f"{x:.3f}"))
print(f"\nSpearman d_auc vs DN IQR: "
      f"{df['d_auc'].corr(df['dn_iqr'], method='spearman'):+.3f}")
df.to_csv(REPO / "reports" / "f_leg_b" / "quant_check.csv")
