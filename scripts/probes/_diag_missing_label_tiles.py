"""How many footprint cells lack a label, and why (partial HiRISE coverage)?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.mapping import tiles_to_raster

OBS = "ESP_045139_2270"
lab = pd.read_parquet(REPO / "dataset_v2" / "labels" / f"{OBS}.parquet")
lab = lab[lab["scale_idx"] == 2]
ti, tj = lab["ti"].to_numpy(), lab["tj"].to_numpy()
foot, _, _ = tiles_to_raster(ti, tj, np.ones(len(lab)))
present = np.isfinite(foot)

from scipy.ndimage import binary_fill_holes, binary_closing
interior = binary_fill_holes(binary_closing(present, iterations=2))
holes = interior & ~present   # inside the footprint hull but no label

print(f"{OBS}: label tiles = {len(lab)}")
print(f"  (ti,tj) bbox grid = {foot.size} cells")
print(f"  present (labelled) = {present.sum()}")
print(f"  inside footprint hull = {interior.sum()}")
print(f"  interior holes (covered region, no label) = {holes.sum()} "
      f"({holes.sum()/max(interior.sum(),1):.1%} of the footprint)")
print(f"  bbox cells outside the strip (the rotated-corner nodata) = "
      f"{(~interior).sum()} ({(~interior).sum()/foot.size:.0%} of bbox)")

# any NaN fractional_area among the present labels?
print(f"  present labels with NaN fractional_area = {lab['fractional_area'].isna().sum()}")
