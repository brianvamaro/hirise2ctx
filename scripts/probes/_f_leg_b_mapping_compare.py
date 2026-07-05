"""Leg B mapping iteration: per-image AUC across perframe / global / minnaert vs baseline.

Joins the three LOIO preds CSVs into one per-image table + incidence of each obs's
frames, to see which mapping wins where and what broke the minnaert outliers.
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
LEGB = REPO / "reports" / "f_leg_b"

FILES = {"perframe": "f_leg_b_loio_preds.csv",
         "global": "f_leg_b_loio_preds_global.csv",
         "minnaert": "f_leg_b_loio_preds_minnaert.csv"}


def per_image(path: Path, store_prefix: str) -> pd.DataFrame:
    preds = pd.read_csv(path)
    rows = []
    for (obs, store), g in preds.groupby(["obs_id", "store"]):
        if g["y"].nunique() == 2:
            rows.append(dict(obs_id=obs, store=store,
                             auc=roc_auc_score(g["y"], g["p"])))
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    base = None
    cols = {}
    for name, fn in FILES.items():
        df = per_image(FIG / fn, name)
        b = df[df.store == "fang_embeddings"].set_index("obs_id")["auc"]
        f = df[df.store != "fang_embeddings"].set_index("obs_id")["auc"]
        if base is None:
            base = b
        cols[name] = f
    out = pd.DataFrame({"baseline": base, **cols})
    for name in FILES:
        out[f"d_{name}"] = out[name] - out["baseline"]

    # frame incidence per obs (max over its frames = worst illumination)
    inc = pd.read_csv(LEGB / "frame_incidence.csv")
    om = pd.read_csv(LEGB / "obs_frame_map.csv").merge(inc, on="PRODUCT_ID")
    obs_inc = om.groupby("obs_id")["incidence"].agg(["min", "max"])
    obs_inc.columns = ["inc_min", "inc_max"]
    out = out.join(obs_inc)

    out = out.sort_values("d_minnaert")
    pd.set_option("display.width", 200)
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nmedians:")
    for name in FILES:
        print(f"  {name:9s} Δmedian {out[name].median() - out['baseline'].median():+.4f}  "
              f"n_below_0.5 {(out[name] < 0.5).sum()}  "
              f"n_improve {(out[f'd_{name}'] > 0).sum()}")

    out.to_csv(LEGB / "mapping_compare_per_image.csv")
    print(f"\nwrote {LEGB / 'mapping_compare_per_image.csv'}")


if __name__ == "__main__":
    main()
