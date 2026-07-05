"""Fetch the true INCIDENCE_ANGLE for P20_008839_2269_XI_46N046W from the PDS
volume index (SeamMap says 4.28 deg — bogus; CTX ~3PM orbit can't see that)."""
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import truststore

truststore.inject_into_ssl()

VOL = "mrox_0605"
PID = "P20_008839_2269_XI_46N046W"
base = f"https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{VOL}/index"

lbl = urllib.request.urlopen(f"{base}/index.lbl", timeout=60).read().decode("ascii", "replace")
tab = urllib.request.urlopen(f"{base}/index.tab", timeout=120).read().decode("ascii", "replace")

# column order from the label
names = [ln.split("=")[1].strip().strip('"') for ln in lbl.splitlines()
         if ln.strip().startswith("NAME")]
row = next(ln for ln in tab.splitlines() if PID in ln)
vals = [v.strip().strip('"') for v in row.split(",")]
for n, v in zip(names, vals):
    if any(k in n.upper() for k in ("PRODUCT_ID", "INCIDENCE", "EMISSION", "PHASE",
                                    "SOLAR", "IMAGE_TIME", "CENTER_LAT")):
        print(f"{n:28s} {v}")
