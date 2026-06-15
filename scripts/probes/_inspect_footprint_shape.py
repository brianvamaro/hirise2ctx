"""Check whether the HiRISE label tiles form the rotated strip (vs the bbox)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OBS = "ESP_045139_2270"
side = json.loads((REPO / "cache_v2" / "ctx_windows" / f"{OBS}.json").read_text())
print("ctx_windows json keys:", list(side.keys()))
bx0, by0, bx1, by1 = side["actual_bounds_target_crs"]
print(f"bbox: {(bx1-bx0)/1000:.1f} x {(by1-by0)/1000:.1f} km")

lab = pd.read_parquet(REPO / "dataset_v2" / "labels" / f"{OBS}.parquet")
s32 = lab[lab["scale_idx"] == 2]
tile_m = float(s32["tile_size_m"].iloc[0])
n = len(s32)
bbox_cells = ((bx1 - bx0) / tile_m) * ((by1 - by0) / tile_m)
print(f"S=32 label tiles: {n}  tile_m={tile_m}")
print(f"bbox would hold ~{bbox_cells:.0f} cells  ->  labels fill {n/bbox_cells:.1%} of bbox")
print(f"  (low fill % => labels are the rotated strip, not the bbox)")

# any per-image footprint geometry in the sidecar?
for k, v in side.items():
    if "bound" in k.lower() or "corner" in k.lower() or "geom" in k.lower() or "poly" in k.lower():
        print(f"  sidecar[{k}] = {v}")
