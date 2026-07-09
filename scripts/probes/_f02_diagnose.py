"""Is F02 an atmosphere/calibration-level anomaly, or geometry/geology?

For the 7 E8_N44 pilot frames: pull incidence/emission/time (frame_table), measure each
frame's median I/F from the aligned crops, apply the H1 minnaert correction (÷cos^0.580),
and — the key test — fit log(median I/F) = k·log(cos i) + b across the frames and report
each frame's RESIDUAL from that photometric line. A frame whose brightness the incidence
model cannot explain (large residual) is anomalous beyond geometry. Cross-check with its
emission angle and acquisition time (season/atmosphere), and with its frame-mean P(rich).

Run:  conda run -n geospatial python scripts/probes/_f02_diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd

from scripts.f_pilot_crop import aligned_path, crop_pids, preds_path
from src.ctx_edr import frame_table

K = 0.580  # H1 minnaert exponent


def main() -> None:
    pids = crop_pids()
    ft = frame_table("E8_N44").set_index("PRODUCT_ID")
    z = np.load(preds_path("minnaert_center"), allow_pickle=False)
    zpids = [str(p) for p in z["pids"]]
    stack = z["f_wl"]

    rows = []
    for pid in pids:
        a = np.load(aligned_path(pid))
        v = a[np.isfinite(a) & (a > 0)]
        med = float(np.median(v))
        inc = float(ft.loc[pid, "INCIDENCE"])
        emi = float(ft.loc[pid, "EMISSION"]) if "EMISSION" in ft.columns else np.nan
        cos_i = np.cos(np.radians(inc))
        pm = np.nanmean(stack[zpids.index(pid)]) if pid in zpids else np.nan
        rows.append(dict(pid=pid[:3], full=pid, incidence=inc, emission=emi,
                         cos_i=cos_i, med_IF=med, minn_IF=med / cos_i**K,
                         time=str(ft.loc[pid, "IMAGE_TIME"])[:19] if "IMAGE_TIME" in ft.columns else "",
                         frame_mean_Prich=float(pm)))
    df = pd.DataFrame(rows)

    # photometric fit: log(med I/F) = k·log(cos i) + b   -> residual = non-incidence brightness
    x = np.log(df["cos_i"].to_numpy())
    y = np.log(df["med_IF"].to_numpy())
    k_fit, b = np.polyfit(x, y, 1)
    df["logIF_resid"] = y - (k_fit * x + b)          # + = brighter than incidence predicts
    df["resid_z"] = (df["logIF_resid"] - df["logIF_resid"].mean()) / df["logIF_resid"].std(ddof=0)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"photometric fit across 7 frames: k={k_fit:.3f}  (H1 uses k={K})\n")
    show = df[["pid", "incidence", "emission", "med_IF", "minn_IF", "logIF_resid",
               "resid_z", "frame_mean_Prich", "time"]].sort_values("logIF_resid")
    print(show.round(4).to_string(index=False))

    f02 = df[df["pid"] == "F02"].iloc[0]
    others = df[df["pid"] != "F02"]
    print(f"\n--- F02 vs the other 6 ---")
    print(f"incidence:        F02 {f02.incidence:.1f}°   others {others.incidence.min():.1f}–{others.incidence.max():.1f}°")
    print(f"emission:         F02 {f02.emission:.1f}°   others {others.emission.min():.1f}–{others.emission.max():.1f}°")
    print(f"minnaert I/F:     F02 {f02.minn_IF:.4f}   others {others.minn_IF.min():.4f}–{others.minn_IF.max():.4f}")
    print(f"log-IF residual:  F02 {f02.logIF_resid:+.4f} (z={f02.resid_z:+.2f})   "
          f"others |resid| max {others.logIF_resid.abs().max():.4f}")
    print(f"frame-mean P(rich): F02 {f02.frame_mean_Prich:.3f}   "
          f"others {others.frame_mean_Prich.min():.3f}–{others.frame_mean_Prich.max():.3f}")


if __name__ == "__main__":
    main()
