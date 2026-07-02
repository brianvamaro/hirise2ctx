"""Build the 10-frame list for the F de-risk timing test (PLAN_StripingArtifact step 2).

Frames = the source frames of the E8_N44 A1-payoff crop (so the timing test doubles as the
before/after comparison site once per-frame inference exists), topped up to N with the tile's
earliest/latest-volume frames for era/size spread. ``--verify`` does a ranged GET on every URL
and records HTTP status + file size (network; ~seconds).

Run: conda run -n geospatial python scripts/f_edr_frame_list.py --verify
Output: reports/f_timing/frame_list.csv  (consumed by f_timing_test.sh on Sherlock)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ctx_edr import frame_table, frames_in_crop

# the A1-payoff crop (scripts/striping_a1_infer_crop.py)
TILE = "E8_N44"
R0, C0, SIZE = 1504, 8992, 15008
N_FRAMES = 10
OUT = REPO / "reports" / "f_timing" / "frame_list.csv"


def ranged_size(url: str, timeout: int = 60):
    import truststore

    truststore.inject_into_ssl()
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "hirise2ctx-research", "Range": "bytes=0-399"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total = (r.headers.get("Content-Range", "") or "/").split("/")[-1]
            pds3 = r.read(400).startswith(b"PDS_VERSION_ID")
            return r.status, (int(total) / 1e6 if total.isdigit() else None), pds3
    except Exception as e:  # noqa: BLE001
        return getattr(e, "code", type(e).__name__), None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="ranged-GET every URL (network)")
    args = ap.parse_args()

    crop = frames_in_crop(TILE, R0, C0, SIZE)
    print(f"{len(crop)} frames intersect the {TILE} A1 crop")
    df = crop.drop(columns="geometry").assign(source=f"{TILE}_a1_crop")

    if len(df) < N_FRAMES:  # top up with era extremes from the whole tile
        allf = frame_table(TILE).drop(columns="geometry").sort_values("VOLUME_ID")
        allf = allf[~allf["PRODUCT_ID"].isin(df["PRODUCT_ID"])]
        k = N_FRAMES - len(df)
        extremes = (allf.iloc[list(range((k + 1) // 2)) + list(range(-(k // 2), 0))]
                    .assign(source="era_extreme", overlap_frac=0.0))
        import pandas as pd

        df = pd.concat([df, extremes], ignore_index=True)

    if args.verify:
        checks = [ranged_size(u) for u in df["edr_url"]]
        df["http"] = [c[0] for c in checks]
        df["size_mb"] = [round(c[1], 1) if c[1] else None for c in checks]
        df["pds3"] = [c[2] for c in checks]
        n_ok = sum(1 for c in checks if c[0] == 206 and c[2])
        print(f"verified: {n_ok}/{len(df)} live PDS3 EDRs, "
              f"total {sum(c[1] or 0 for c in checks):.0f} MB")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} frames)")
    print(df[["PRODUCT_ID", "VOLUME_ID", "source"] +
             (["http", "size_mb", "pds3"] if args.verify else [])].to_string())


if __name__ == "__main__":
    main()
