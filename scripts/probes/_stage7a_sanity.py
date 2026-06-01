"""Stage 7a sanity check -- verify Stage 1 SP1-corrected sidecars exist for every
colour-covered ObsId. Catches the case where a Stage 7c worker would silently
fall back to the buggy embedded CRS for a subset of images.

Reads `cache_v2/hirise_color/coverage.parquet` and walks the 37 has_color rows,
checking `cache_v2/reprojected_detections/{ObsId}.json` for each.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import pandas as pd

print = functools.partial(print, flush=True)

cov = pd.read_parquet("cache_v2/hirise_color/coverage.parquet")
have = cov[cov["has_color"]].reset_index(drop=True)
print(f"Checking Stage 1 sidecars for {len(have)} colour-covered ObsIds")

missing = []
present = []
for obs_id in have["obs_id"]:
    sidecar = Path(f"cache_v2/reprojected_detections/{obs_id}.json")
    if sidecar.exists():
        present.append(obs_id)
    else:
        missing.append(obs_id)
print(f"  sidecar present: {len(present)} / {len(have)}")
if missing:
    print(f"  MISSING ({len(missing)}):")
    for o in missing:
        print(f"    {o}")
    sys.exit(1)
print("All colour-covered ObsIds have a Stage 1 SP1-corrected sidecar.")
