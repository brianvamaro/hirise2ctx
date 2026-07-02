"""Probe 5: verify the fixed EDR template across 12 frames spanning the mission (4 tiles x 3 volumes)."""
import truststore

truststore.inject_into_ssl()

import urllib.request
from pathlib import Path

import geopandas as gpd

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) hirise2ctx-research"
TEMPLATE = "https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{vol}/data/{pid}.IMG"

def ranged_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-399"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cr = r.headers.get("Content-Range", "")
            body = r.read(400)
            return r.status, cr, body
    except urllib.error.HTTPError as e:
        return e.code, "", b""
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}", "", b""

rows = []
for tile in ["E8_N44", "E-12_N36", "E4_N40", "E16_N44"]:
    d = Path(f"cache/ctx_tiles/_seammap_{tile}")
    g = gpd.read_file(next(d.glob("*SeamMap.shp")))[["PRODUCT_ID", "VOLUME_ID"]].drop_duplicates("PRODUCT_ID")
    g = g.sort_values("VOLUME_ID")
    rows.extend(g.iloc[[0, len(g) // 2, len(g) - 1]].to_dict("records"))

ok = 0
for r in rows:
    url = TEMPLATE.format(vol=r["VOLUME_ID"].lower(), pid=r["PRODUCT_ID"])
    s, cr, body = ranged_get(url)
    is_pds3 = body.startswith(b"PDS_VERSION_ID")
    total = cr.split("/")[-1] if cr else "?"
    mb = f"{int(total)/1e6:7.1f} MB" if total.isdigit() else "      ??"
    ok += 1 if (s == 206 and is_pds3) else 0
    print(f"{r['PRODUCT_ID']:28s} {r['VOLUME_ID']:9s} -> {s} PDS3={is_pds3} {mb}")
print(f"\n{ok}/{len(rows)} resolve to live PDS3 EDRs via the mro/ctx template")
