"""Review fact-check: how much between-frame disagreement SURVIVES the minnaert
correction on the 7 E8_N44 pilot frames?

The 2026-07-05c verdict quoted 10.2% median |ratio-1| — but that was RAW I/F.
Here: per overlap pair, coarse (32 px block-mean) agreement in
  (a) raw I/F              (b) minnaert-corrected I/F (k=0.580)
plus each pair's Δincidence and Δorbit (time proxy), joined with the prediction
|diff| from the eta2 run. Tells us whether the residual floor is
photometric-correctable (corrected disagreement still large, tracks Δi) or
information-level (corrected disagreement small; predictions disagree anyway).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd

from src.ctx_edr import frame_table

WORK = REPO / "reports" / "f_timing" / "pilot_work"
FIG = REPO / "reports" / "figures"
TILE = "E8_N44"
TILE_PX, SIZE = 32, 15008
K = 0.580


def coarse(pid: str) -> np.ndarray:
    a = np.load(WORK / "aligned" / f"{pid}.npy")
    n = SIZE // TILE_PX
    b = a.reshape(n, TILE_PX, n, TILE_PX)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(b, axis=(1, 3)).astype(np.float32)
    frac = np.isfinite(b).mean(axis=(1, 3))
    m[frac < 0.9] = np.nan
    return m


def main() -> None:
    pids = sorted(p.name.replace(".npy", "") for p in (WORK / "aligned").glob("*.npy"))
    ft = frame_table(TILE).set_index("PRODUCT_ID")
    inc = {p: float(ft.loc[p, "INCIDENCE"]) for p in pids}
    cos_i = {p: np.cos(np.radians(inc[p])) for p in pids}
    orbit = {p: int(p.split("_")[1]) for p in pids}

    cif = {p: coarse(p) for p in pids}
    pred_pairs = pd.read_csv(FIG / "f_pilot_overlap_pairs.csv")
    pred_pairs = pred_pairs[(pred_pairs.kind == "pred") &
                            (pred_pairs.mapping == "minnaert_log")]
    pred_by_pair = dict(zip(pred_pairs["pair"], pred_pairs["median_absdiff"]))

    rows = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = cif[pids[i]], cif[pids[j]]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() < 200:
                continue
            raw = np.median(np.abs(a[both] / b[both] - 1))
            ac = a[both] / (cos_i[pids[i]] ** K)
            bc = b[both] / (cos_i[pids[j]] ** K)
            corr = np.median(np.abs(ac / bc - 1))
            pair = f"{pids[i][:3]}~{pids[j][:3]}"
            rows.append(dict(
                pair=pair, n=int(both.sum()),
                d_inc=abs(inc[pids[i]] - inc[pids[j]]),
                d_orbit=abs(orbit[pids[i]] - orbit[pids[j]]),
                raw_absratio=round(float(raw), 4),
                minn_absratio=round(float(corr), 4),
                pred_absdiff=pred_by_pair.get(pair, np.nan),
            ))

    df = pd.DataFrame(rows).sort_values("d_inc")
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    print(f"\nmedian across pairs:  raw {df.raw_absratio.median():.3f}  "
          f"minnaert-corrected {df.minn_absratio.median():.3f}  "
          f"pred |diff| {df.pred_absdiff.median():.3f}")
    for x in ("d_inc", "d_orbit", "raw_absratio", "minn_absratio"):
        rho = df["pred_absdiff"].corr(df[x], method="spearman")
        print(f"  Spearman pred_absdiff vs {x:14s} = {rho:+.3f}")
    out = REPO / "reports" / "f_leg_b" / "review_overlap_residual.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
