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
        c = {"pid": _col(names, "PRODUCT_ID"),
             "inc": _col(names, "INCIDENCE_ANGLE"),
             "clat": _col(names, "CENTER_LATITUDE"),
             "clon": _col(names, "CENTER_LONGITUDE"),
             "slat": _col(names, "SUB_SOLAR_LATITUDE", "SUBSOLAR_LATITUDE"),
             "slon": _col(names, "SUB_SOLAR_LONGITUDE", "SUBSOLAR_LONGITUDE")}
        mx = max(c.values())
        by = {}
        for ln in idx:
            vals = [v.strip().strip('"') for v in ln.split(",")]
            if len(vals) > mx:
                try:
                    by[vals[c["pid"]]] = tuple(float(vals[c[k2]])
                                               for k2 in ("inc", "clat", "clon", "slat", "slon"))
                except ValueError:
                    pass
        for pid in g["PRODUCT_ID"]:
            if pid not in by:
                missing.append(pid)
                continue
            inc, clat, clon, slat, slon = by[pid]
            sm = float(seam.get(pid, float("nan")))
            if np.isfinite(sm) and abs(sm - inc) > 1.0:
                flagged += 1
            rows.append(dict(PRODUCT_ID=pid, VOLUME_ID=vol, incidence=round(inc, 4),
                             center_lat=round(clat, 4), center_lon=round(clon, 4),
                             subsolar_lat=round(slat, 4), subsolar_lon=round(slon, 4)))
        if (k + 1) % 25 == 0 or k + 1 == len(vols):
            print(f"  {k+1}/{len(vols)} volumes, {len(rows)} frames resolved", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nresolved {len(df)}/{len(fl)} frames; incidence "
          f"{df.incidence.min():.1f}-{df.incidence.max():.1f} deg, "
          f"center_lat {df.center_lat.min():.1f}-{df.center_lat.max():.1f} deg")
    print(f"SeamMap-vs-PDS incidence disagreements >1 deg: {flagged} (SeamMap untrusted; PDS used)")

    # PHYSICAL-model sanity: cos(i) at the image center from subsolar geometry must reproduce the
    # index INCIDENCE_ANGLE (else the spherical model or a column is wrong). Stage B uses this same
    # geometry per row -> exact per-row incidence, no slope fit (the pooled fit was season-confounded).
    if len(df):
        phi = np.radians(df.center_lat.values)
        phis = np.radians(df.subsolar_lat.values)
        dlam = np.radians((df.center_lon - df.subsolar_lon).values)
        cosi = np.sin(phi) * np.sin(phis) + np.cos(phi) * np.cos(phis) * np.cos(dlam)
        inc_phys = np.degrees(np.arccos(np.clip(cosi, -1.0, 1.0)))
        dinc = np.abs(inc_phys - df.incidence.values)
        print(f"physical incidence(center) vs index: median |Δ| {np.median(dinc):.2f} deg, "
              f"max {dinc.max():.2f} deg, frames >2 deg: {int((dinc > 2).sum())} "
              f"(median <~1 deg ⇒ subsolar geometry OK for the per-row model)")

    if missing:
        print(f"\n⚠ {len(missing)} frames MISSING from their PDS volume index (V2 gate):")
        for pid in missing[:20]:
            print(f"   {pid}")
        raise SystemExit(f"FAIL: {len(missing)} frames unresolved — investigate before Stage B")
    print(f"\nwrote {OUT}  (source: PDS volume indexes)")


if __name__ == "__main__":
    main()
