"""PLAN_StripingArtifact mitigation prep — decompose the per-frame CTX radiometry differences,
to decide whether A1 (robust offset+gain) suffices or A2 (full histogram matching) is needed.

For each CTX+SeamMap tile, for each source frame, compute the CTX-DN distribution (median, IQR,
percentiles). Then:
  * how much do frames differ in LEVEL (median spread) vs SCALE (IQR spread)?  -> the offset/gain
    that A1 would remove;
  * after a robust offset+gain normalize (x-median)/IQR per frame, do the normalized percentiles
    COLLAPSE across frames? If yes, A1 captures the difference; residual spread = A2 territory.

Outputs: reports/figures/striping_frame_radiometry.png + striping_frame_radiometry.csv.
Run: conda run -n geospatial python scripts/striping_frame_radiometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.striping import (MAP_DIR, equipped_tiles, frame_label_map, load_frames, read_ctx_on_grid)

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
PCTS = [5, 25, 50, 75, 95]


def main():
    tiles = equipped_tiles()
    print(f"per-frame CTX radiometry over {len(tiles)} tiles")
    rows = []
    lead_raw, lead_norm = [], []          # per-frame DN samples for the lead-tile figure
    LEAD = "E8_N44"
    for t in tiles:
        ctx = read_ctx_on_grid(t, MAP_DIR / f"{t}_abundance.tif")
        fr = load_frames(t)
        L = frame_label_map(t, fr)
        for i in range(len(fr)):
            sel = (L == i) & np.isfinite(ctx)
            if sel.sum() < 200:
                continue
            v = ctx[sel]
            med = np.median(v)
            iqr = np.subtract(*np.percentile(v, [75, 25])) or 1.0
            p = np.percentile(v, PCTS)
            pn = (p - med) / iqr                       # robust offset+gain (A1) normalized percentiles
            rows.append(dict(tile=t, frame=str(fr.iloc[i].get("PRODUCT_ID", i)), n=int(sel.sum()),
                             median=med, iqr=iqr, **{f"p{q}": p[k] for k, q in enumerate(PCTS)},
                             **{f"pn{q}": pn[k] for k, q in enumerate(PCTS)}))
            if t == LEAD:
                samp = np.random.default_rng(0).choice(v, min(4000, v.size), replace=False)
                lead_raw.append(samp)
                lead_norm.append((samp - med) / iqr)
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "striping_frame_radiometry.csv", index=False)

    # ---- between-frame spreads (per tile, then pooled) ----
    print("\nper-tile: between-frame spread of LEVEL (median) and SCALE (iqr), and A1-residual")
    for t, g in df.groupby("tile"):
        lvl = g["median"].std()
        scl = g["iqr"].std() / g["iqr"].mean()
        resid = np.mean([g[f"pn{q}"].std() for q in (5, 95)])   # residual spread after A1 normalize
        print(f"  {t:9s} n_fr={len(g):3d} | level std={lvl:5.1f} DN | scale CV={scl:4.2f} | "
              f"A1-residual(norm p5/p95 std)={resid:.3f}")
        rows_t = dict(tile=t, n_frames=len(g), level_std=lvl, scale_cv=scl, a1_residual=resid)
        rows.append(rows_t)  # not re-saved; just for print
    lvl_all = df.groupby("tile")["median"].std().median()
    scl_all = (df.groupby("tile")["iqr"].std() / df.groupby("tile")["iqr"].mean()).median()
    # A1 residual: spread of normalized p5/p95 across frames, pooled
    resid_raw = np.mean([df.groupby("tile")[f"p{q}"].std().median() for q in (5, 95)])
    resid_a1 = np.mean([df.groupby("tile")[f"pn{q}"].std().median() for q in (5, 95)])
    print(f"\nPOOLED medians: level std={lvl_all:.1f} DN | scale CV={scl_all:.2f}")
    print(f"tail spread (p5/p95 across frames): RAW {resid_raw:.2f} DN  ->  after A1 normalize "
          f"{resid_a1:.3f} (in IQR units)")
    print(f"=> A1 collapses the between-frame tail spread by ~{resid_raw/ (resid_a1*df['iqr'].median()+1e-9):.1f}x"
          if resid_a1 else "")

    # ---- figure: lead-tile per-frame histograms, raw vs A1-normalized ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for s in lead_raw:
        ax[0].hist(s, bins=60, range=(0, 255), histtype="step", lw=0.6, alpha=0.5)
    ax[0].set_title(f"{LEAD}: raw CTX DN per source frame (each line = 1 frame)\n"
                    "spread between frames = the radiometric artifact")
    ax[0].set_xlabel("CTX DN"); ax[0].set_ylabel("count")
    for s in lead_norm:
        ax[1].hist(s, bins=60, range=(-4, 4), histtype="step", lw=0.6, alpha=0.5)
    ax[1].set_title("after A1 (robust offset+gain) normalize per frame\n"
                    "if the lines collapse, A1 captures the difference")
    ax[1].set_xlabel("(DN - median) / IQR"); ax[1].set_ylabel("count")
    fig.tight_layout(); fig.savefig(FIG / "striping_frame_radiometry.png", dpi=120)
    print(f"\nFigure + CSV -> {FIG}")


if __name__ == "__main__":
    main()
