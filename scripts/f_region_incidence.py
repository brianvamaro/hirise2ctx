"""PLAN_FBuild V2 — TRUE per-frame incidence + center latitude for all 907 region frames, from the
PDS volume indexes, for the per-row cos^k(i(lat)) Stage-B mapping.

SeamMap incidence is UNTRUSTED (the P20_008839 decimal-shift class); PDS is truth. This is the
region-scale version of scripts/probes/_f_leg_b_pds_incidence.py — same cached index fetch, but over
reports/figures/region_frame_list.csv and additionally pulling CENTER_LATITUDE. Emits per frame the
center incidence + center latitude; Stage B forms incidence(lat) = inc + slope*(lat - center_lat),
slope = di/dlat (audit ~0.635 deg/deg; the between-frame incidence-vs-lat fit printed here is the
refined candidate). Fails loudly on any frame missing from its volume index (V2 completeness gate).

Run (laptop; network, truststore SSL; indexes cache under cache/pds_index/):
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run --no-capture-output -n geospatial python -u \
      scripts/f_region_incidence.py
"""
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import truststore

truststore.inject_into_ssl()

import numpy as np
import pandas as pd

CACHE = REPO / "cache" / "pds_index"
FL = REPO / "reports" / "figures" / "region_frame_list.csv"
OUT = REPO / "reports" / "figures" / "region_frame_incidence.csv"


def volume_index(vol: str) -> tuple[list[str], list[str]]:
    """(column names, index rows) for one PDS volume, cached locally (reused from the leg-B probe)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    lbl_p, tab_p = CACHE / f"{vol}_index.lbl", CACHE / f"{vol}_index.tab"
    base = f"https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{vol.lower()}/index"
    for p, name in ((lbl_p, "index.lbl"), (tab_p, "index.tab")):
        if not p.exists():
            with urllib.request.urlopen(f"{base}/{name}", timeout=180) as r:
                p.write_bytes(r.read())
    names = [ln.split("=")[1].strip().strip('"')
             for ln in lbl_p.read_text(encoding="ascii", errors="replace").splitlines()
             if ln.strip().startswith("NAME")]
    rows = tab_p.read_text(encoding="ascii", errors="replace").splitlines()
    return names, rows


def _col(names: list[str], *candidates: str) -> int:
    for c in candidates:
        if c in names:
            return names.index(c)
    raise SystemExit(f"none of {candidates} in PDS index columns: {names}")


def main() -> None:
    fl = pd.read_csv(FL)
    seam = fl.set_index("PRODUCT_ID")["incidence_seammap"].to_dict()
    vols = list(fl.groupby("VOLUME_ID"))
    print(f"{len(fl)} frames across {len(vols)} PDS volumes", flush=True)

    rows, missing, flagged = [], [], 0
    for k, (vol, g) in enumerate(vols):
        names, idx = volume_index(vol)
        i_pid = _col(names, "PRODUCT_ID")
        i_inc = _col(names, "INCIDENCE_ANGLE")
        i_lat = _col(names, "CENTER_LATITUDE", "SUB_SPACECRAFT_LATITUDE")
        by = {}
        for ln in idx:
            vals = [v.strip().strip('"') for v in ln.split(",")]
            if len(vals) > max(i_pid, i_inc, i_lat):
                try:
                    by[vals[i_pid]] = (float(vals[i_inc]), float(vals[i_lat]))
                except ValueError:
                    pass
        for pid in g["PRODUCT_ID"]:
            if pid not in by:
                missing.append(pid)
                continue
            inc, lat = by[pid]
            sm = float(seam.get(pid, float("nan")))
            if np.isfinite(sm) and abs(sm - inc) > 1.0:
                flagged += 1
            rows.append(dict(PRODUCT_ID=pid, VOLUME_ID=vol, incidence=round(inc, 4),
                             center_lat=round(lat, 4)))
        if (k + 1) % 25 == 0 or k + 1 == len(vols):
            print(f"  {k+1}/{len(vols)} volumes, {len(rows)} frames resolved", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nresolved {len(df)}/{len(fl)} frames; incidence "
          f"{df.incidence.min():.1f}-{df.incidence.max():.1f} deg, "
          f"center_lat {df.center_lat.min():.1f}-{df.center_lat.max():.1f} deg")
    print(f"SeamMap-vs-PDS incidence disagreements >1 deg: {flagged} (SeamMap untrusted; PDS used)")

    # candidate per-row slope: between-frame incidence-vs-lat fit (proxy for within-frame di/dlat)
    if len(df) > 10:
        slope = float(np.polyfit(df.center_lat, df.incidence, 1)[0])
        print(f"between-frame di/dlat fit = {slope:+.3f} deg/deg "
              f"(audit within-family ~0.635; Stage B slope candidate)")

    if missing:
        print(f"\n⚠ {len(missing)} frames MISSING from their PDS volume index (V2 gate):")
        for pid in missing[:20]:
            print(f"   {pid}")
        raise SystemExit(f"FAIL: {len(missing)} frames unresolved — investigate before Stage B")
    print(f"\nwrote {OUT}  (source: PDS volume indexes)")


if __name__ == "__main__":
    main()
