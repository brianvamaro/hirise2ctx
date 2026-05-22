"""Dump the SP1 value of every cached decimated HiRISE TIFF, alongside the Stage 1
sidecar's expected value. Used to confirm the JP2-side SP1 fix is being applied."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rasterio

SP1_RE = re.compile(
    r'"(?:standard_parallel_1|Latitude of 1st standard parallel)"\s*,\s*(-?\d+\.?\d*)',
    re.IGNORECASE,
)

for tif in sorted((REPO_ROOT / "cache/hirise_decimated").glob("*_5mpp_full.tif")):
    obs = tif.stem.replace("_5mpp_full", "")
    with rasterio.open(tif) as ds:
        wkt = ds.crs.to_wkt() if ds.crs else ""
    m = SP1_RE.search(wkt)
    cache_sp1 = float(m.group(1)) if m else None

    s1 = REPO_ROOT / "cache/reprojected_detections" / f"{obs}.json"
    expected_sp1 = None
    status = "?"
    if s1.exists():
        info = json.loads(s1.read_text(encoding="utf-8"))
        m2 = SP1_RE.search(info.get("source_crs_wkt", ""))
        expected_sp1 = float(m2.group(1)) if m2 else None
        status = info.get("correction", {}).get("status", "?")

    ok = "OK" if cache_sp1 == expected_sp1 else "MISMATCH"
    print(f"{obs}: cache SP1={cache_sp1}  expected={expected_sp1}  status={status!r}  {ok}")
