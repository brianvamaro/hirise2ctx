"""PLAN_FBuild V1/V5 sizing probe — pick 5 representative frames from the 907-frame list.

The probe (Stage A ISIS + Stage B embed) runs on this small, deliberately-spread subset so V1
(tiles/frame + s/frame → array sizing) and V5 (within-frame incidence ramp → per-frame vs per-row
cos^k(i)) generalize to the full build. Selection is deterministic (farthest-point sampling, no RNG)
over three axes that drive the two questions:

  - n_tiles  : proxy for TRACK LENGTH — the V5 within-frame ramp only bites on long 3-4° frames, so
               the longest frame is force-included and long frames are favored.
  - incidence: the photometric axis V5 corrects (SeamMap incidence is UNTRUSTED — used here only for
               SPREAD; a decimal-shift outlier <10° is de-shifted for selection, not for science).
  - year     : acquisition epoch (F02-class calibration/atmosphere differences live here).

Emits reports/f_build/sizing_frame_list.csv with the columns f_timing_test.sh consumes
(PRODUCT_ID, VOLUME_ID, edr_url) plus the selection features for transparency.

Run (laptop, seconds):
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run --no-capture-output -n geospatial \
      python -u scripts/f_build_sizing_frames.py [--n 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/pandas

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FRAME_LIST = REPO / "reports" / "figures" / "region_frame_list.csv"
OUT_DIR = REPO / "reports" / "f_build"


def _incidence_for_spread(inc: float) -> float:
    """SeamMap incidence, de-shifted for the P20_008839 decimal-shift class (selection only)."""
    if not np.isfinite(inc):
        return np.nan
    if inc < 10.0 and 10.0 <= inc * 10.0 <= 80.0:   # decimal shift (e.g. 4.276 -> 42.76)
        return inc * 10.0
    return inc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="frames to select")
    args = ap.parse_args()

    if not FRAME_LIST.exists():
        raise SystemExit(f"missing {FRAME_LIST} — run scripts/f_build_framelist.py first")
    df = pd.read_csv(FRAME_LIST)
    df["year"] = pd.to_datetime(df["image_time"], errors="coerce").dt.year
    df["inc_sel"] = df["incidence_seammap"].map(_incidence_for_spread)
    # drop rows we cannot place in the feature space (keep them selectable only as a last resort)
    feat = df[["inc_sel", "year", "n_tiles"]].astype(float)
    ok = feat.notna().all(axis=1)
    pool = df[ok].reset_index(drop=True)
    X = feat[ok].to_numpy()
    # z-normalise each axis so spread is comparable
    Xn = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)

    # farthest-point sampling seeded by the LONGEST track (max n_tiles), tie-break lowest index
    seed = int(pool["n_tiles"].to_numpy().argmax())
    chosen = [seed]
    while len(chosen) < min(args.n, len(pool)):
        d = np.min([np.linalg.norm(Xn - Xn[c], axis=1) for c in chosen], axis=0)
        d[chosen] = -1.0
        chosen.append(int(d.argmax()))

    sel = pool.iloc[chosen].copy()
    sel = sel.sort_values(["n_tiles", "PRODUCT_ID"], ascending=[False, True]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["PRODUCT_ID", "VOLUME_ID", "edr_url", "inc_sel", "year", "n_tiles", "tiles"]
    out = OUT_DIR / "sizing_frame_list.csv"
    sel[cols].to_csv(out, index=False)

    print(f"selected {len(sel)} of {len(df)} frames "
          f"(longest track force-included; FPS over incidence/year/n_tiles):\n", flush=True)
    print(sel[["PRODUCT_ID", "VOLUME_ID", "inc_sel", "year", "n_tiles", "tiles"]].to_string(index=False))
    print(f"\n  incidence(sel) span : {sel.inc_sel.min():.1f}–{sel.inc_sel.max():.1f}°")
    print(f"  year span           : {int(sel.year.min())}–{int(sel.year.max())}")
    print(f"  n_tiles span        : {int(sel.n_tiles.min())}–{int(sel.n_tiles.max())} "
          f"(≥3 = long track, the V5 target)")
    print(f"\nwrote {out}")
    print("Next (Sherlock): FRAME_LIST=reports/f_build/sizing_frame_list.csv KEEP_CUBES=1 "
          "sbatch run_f_build_probe.sbatch")


if __name__ == "__main__":
    main()
