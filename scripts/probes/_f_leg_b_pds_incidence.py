"""Fetch TRUE incidence angles for all 63 leg-B frames from the PDS volume indexes.

The SeamMap INCIDENCE attribute has at least one gross error (P20_008839: 4.2759 vs
true 42.76, DECISIONS 2026-07-05) and ESP_053989's minnaert-only collapse suggests
subtler ones.  This replaces reports/f_leg_b/frame_incidence.csv wholesale with the
PDS truth (one index.tab per volume, cached under cache/pds_index/), and prints the
SeamMap-vs-PDS deltas so bad SeamMap rows are documented.

Run: conda run --no-capture-output -n geospatial python -u scripts/probes/_f_leg_b_pds_incidence.py
"""
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import truststore

truststore.inject_into_ssl()

import pandas as pd

CACHE = REPO / "cache" / "pds_index"
LEGB = REPO / "reports" / "f_leg_b"
OUT = LEGB / "frame_incidence.csv"


def volume_index(vol: str) -> tuple[list[str], list[str]]:
    """(column names, index rows) for one PDS volume, cached locally."""
    CACHE.mkdir(parents=True, exist_ok=True)
    lbl_p, tab_p = CACHE / f"{vol}_index.lbl", CACHE / f"{vol}_index.tab"
    base = f"https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{vol.lower()}/index"
    for p, name in ((lbl_p, "index.lbl"), (tab_p, "index.tab")):
        if not p.exists():
            with urllib.request.urlopen(f"{base}/{name}", timeout=120) as r:
                p.write_bytes(r.read())
    names = [ln.split("=")[1].strip().strip('"')
             for ln in lbl_p.read_text(encoding="ascii", errors="replace").splitlines()
             if ln.strip().startswith("NAME")]
    rows = tab_p.read_text(encoding="ascii", errors="replace").splitlines()
    return names, rows


def main() -> None:
    fl = pd.read_csv(LEGB / "cohort_frame_list.csv")
    seam = pd.read_csv(OUT).set_index("PRODUCT_ID")  # current (SeamMap+override) values

    out_rows = []
    for vol, g in fl.groupby("VOLUME_ID"):
        names, rows = volume_index(vol)
        i_pid = names.index("PRODUCT_ID")
        i_inc = names.index("INCIDENCE_ANGLE")
        by_pid = {}
        for ln in rows:
            vals = [v.strip().strip('"') for v in ln.split(",")]
            if len(vals) > max(i_pid, i_inc):
                by_pid[vals[i_pid]] = float(vals[i_inc])
        for pid in g["PRODUCT_ID"]:
            if pid not in by_pid:
                print(f"  MISSING from {vol} index: {pid}")
                continue
            pds = by_pid[pid]
            old = float(seam.loc[pid, "incidence"]) if pid in seam.index else float("nan")
            tile = seam.loc[pid, "tile"] if pid in seam.index else ""
            flag = "  <-- SEAMMAP WRONG" if abs(pds - old) > 1.0 else ""
            if flag or abs(pds - old) > 0.2:
                print(f"  {pid}: seammap {old:7.3f}  pds {pds:7.3f}{flag}")
            out_rows.append(dict(tile=tile, PRODUCT_ID=pid, incidence=pds))

    df = pd.DataFrame(out_rows)
    print(f"\n{len(df)} frames; incidence {df.incidence.min():.1f}–{df.incidence.max():.1f} deg")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  (source: PDS volume indexes)")


if __name__ == "__main__":
    main()
