"""Cross-variant per-image AUC table for all leg-B mappings tested.

Reads every f_leg_b_loio_preds_*.csv, computes per-image AUC vs the shared baseline,
and prints the win/loss split + medians so the PASSING log-stretch variant can be
compared honestly against the failing ones (esp. the ESP_053989 inversion tail).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FIG = REPO / "reports" / "figures"

VARIANTS = {  # label -> preds CSV suffix
    "perframe": "",
    "global": "_global",
    "minn_p2_98": "_minnaert",
    "minn_wide_lin": "_minnaert_w",
    "minn_wide_cubic": "_minnaert_cubic",
    "minn_wide_LOG": "_minnaert_wl",
}


def per_image(suffix: str) -> pd.DataFrame:
    path = FIG / f"f_leg_b_loio_preds{suffix}.csv"
    if not path.exists():
        return None
    preds = pd.read_csv(path)
    rows = []
    for (obs, store), g in preds.groupby(["obs_id", "store"]):
        if g["y"].nunique() == 2:
            rows.append(dict(obs_id=obs, is_base=store == "fang_embeddings",
                             auc=roc_auc_score(g["y"], g["p"])))
    d = pd.DataFrame(rows)
    base = d[d.is_base].set_index("obs_id")["auc"]
    f = d[~d.is_base].set_index("obs_id")["auc"]
    return base, f


table, base_ref = {}, None
for label, suf in VARIANTS.items():
    r = per_image(suf)
    if r is None:
        print(f"(skip {label}: no preds CSV)")
        continue
    base, f = r
    if base_ref is None:
        base_ref = base
    table[label] = f

df = pd.DataFrame({"baseline": base_ref, **table})
print("=== per-image AUC by variant (sorted by log-stretch delta) ===")
df["d_LOG"] = df["minn_wide_LOG"] - df["baseline"]
df = df.sort_values("d_LOG")
pd.set_option("display.width", 200)
print(df.drop(columns="d_LOG").to_string(float_format=lambda x: f"{x:.3f}"))

print("\n=== summary: Δ median AUC vs baseline, win/loss split ===")
for label in table:
    d = df[label] - df["baseline"]
    print(f"  {label:16s} Δmed {df[label].median() - df['baseline'].median():+.4f}  "
          f"mean {df[label].mean():.3f}  win {int((d > 0).sum())}  "
          f"loss {int((d < 0).sum())}  below0.5 {int((df[label] < 0.5).sum())}")

df.to_csv(REPO / "reports" / "f_leg_b" / "variant_summary.csv")
print(f"\nwrote reports/f_leg_b/variant_summary.csv")
