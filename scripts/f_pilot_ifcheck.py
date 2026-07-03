"""F pilot, leg A0 (CPU-only, no GPU): calibrated I/F cross-frame consistency on the crop.

Tests F's foundational premise BEFORE any embedding — Walter 2024's claim that uniformly
`ctxcal`'d CTX frames are mutually consistent to ~±2%. On the 7 aligned E8_N44 crop frames:
  1. pairwise overlap agreement at 160 m (median |ratio-1| per pair — the ±2% check);
  2. per-frame I/F stats vs the mosaic's per-frame spread (level ≈20 DN on 0..255 ≈ 8% of
     range, IQR CV ≈ 0.43 — scripts/striping_frame_radiometry.py), i.e. how much of the
     Dickson per-frame stretch ctxcal removed;
  3. does the residual per-frame level track cos(incidence)? (If yes, `lambert` should beat
     plain `affine` in the embedding pilot; if levels are already flat, `affine` suffices.)
  4. a partition-composite I/F quicklook — do the blocks visually vanish in calibrated
     radiometry?

Run: conda run -n geospatial python scripts/f_pilot_ifcheck.py
Shares the alignment cache with scripts/f_pilot_crop.py (pre-warms the GPU run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap

import numpy as np
from rasterio.transform import Affine
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.f_pilot_crop import (FIG, SIZE, T5, TILE, TILE_PX,  # noqa: E402
                                  align_frame, aligned_path, coarse_if, crop_pids,
                                  frame_labels)
from src.ctx_edr import frame_table  # noqa: E402


def main():
    pids = crop_pids()
    print(f"{len(pids)} frames", flush=True)
    for pid in pids:
        align_frame(pid)

    n = SIZE // TILE_PX
    shape = (n, n)
    cif = {p: coarse_if(p, shape) for p in pids}
    ft = frame_table(TILE).set_index("PRODUCT_ID")

    # ---- per-frame stats + incidence
    print("\n=== per-frame I/F stats (aligned crop) ===")
    rows = []
    for p in pids:
        v = cif[p][np.isfinite(cif[p])]
        q25, q75 = np.percentile(v, [25, 75])
        inc = float(ft.loc[p, "INCIDENCE"])
        rows.append({"pid": p, "median": float(np.median(v)), "iqr": float(q75 - q25),
                     "incidence": inc, "cos_i": float(np.cos(np.radians(inc)))})
    import pandas as pd

    df = pd.DataFrame(rows)
    med_spread = (df["median"].max() - df["median"].min()) / df["median"].mean()
    iqr_cv = df["iqr"].std() / df["iqr"].mean()
    r_cos = float(np.corrcoef(df["cos_i"], df["median"])[0, 1])
    df["median_lam"] = df["median"] / df["cos_i"]
    lam_spread = (df["median_lam"].max() - df["median_lam"].min()) / df["median_lam"].mean()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nlevel spread (max-min)/mean : {med_spread:.1%}   (mosaic per-frame level ≈ 8% of range)")
    print(f"IQR coefficient of variation: {iqr_cv:.2f}    (mosaic scale CV ≈ 0.43)")
    print(f"median vs cos(i) Pearson r  : {r_cos:+.2f}")
    print(f"level spread after Lambert  : {lam_spread:.1%}")

    # ---- pairwise overlap agreement (the ±2% check)
    print("\n=== pairwise overlap agreement at 160 m ===")
    pairs = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = cif[pids[i]], cif[pids[j]]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() < 200:
                continue
            ratio = a[both] / b[both]
            pairs.append({"pair": f"{pids[i][:14]} ~ {pids[j][:14]}", "n": int(both.sum()),
                          "median_ratio": float(np.median(ratio)),
                          "med_absdev_pct": float(np.median(np.abs(ratio - 1)) * 100),
                          "corr": float(np.corrcoef(a[both], b[both])[0, 1])})
    dp = pd.DataFrame(pairs)
    print(dp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nmedian |ratio-1| across pairs: {dp['med_absdev_pct'].median():.1f}%  "
          f"(Walter 2024 flat-field claim: ~±2%)")

    # ---- figure: partition composite + histograms (raw vs lambert)
    t160 = Affine(T5.a * TILE_PX, 0, T5.c, 0, T5.e * TILE_PX, T5.f)
    labels = frame_labels(pids, shape, t160)
    part = np.full(shape, np.nan, dtype=np.float32)
    for i, p in enumerate(pids):
        sel = (labels == i) & np.isfinite(cif[p])
        part[sel] = cif[p][sel]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    im = ax[0].imshow(part, cmap="gray")
    ax[0].set_title("partition composite, calibrated I/F\n(do the frame blocks survive?)")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    for i, p in enumerate(pids):
        v = cif[p][np.isfinite(cif[p])]
        ax[1].hist(v, bins=120, histtype="step", density=True, label=p[:14])
        ax[2].hist(v / float(df.loc[df.pid == p, "cos_i"].iloc[0]), bins=120,
                   histtype="step", density=True)
    ax[1].set_title("per-frame I/F histograms (raw)")
    ax[1].legend(fontsize=7)
    ax[2].set_title("per-frame histograms, Lambert cos(i)-corrected")
    fig.suptitle(f"F pilot A0 — calibrated-frame radiometric consistency on the {TILE} crop "
                 f"(mosaic counterpart: level spread ≈8%, scale CV 0.43)")
    fig.tight_layout()
    out = FIG / "f_pilot_ifcheck.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")
    df.to_csv(FIG / "f_pilot_ifcheck_frames.csv", index=False)
    dp.to_csv(FIG / "f_pilot_ifcheck_pairs.csv", index=False)


if __name__ == "__main__":
    main()
