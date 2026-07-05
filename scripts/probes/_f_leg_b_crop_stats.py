"""Per-crop I/F median/IQR for the leg B gallery images — is the crater/improver
split explained by between-frame illumination deltas inside one composite?"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401

import numpy as np
import rasterio

crops_dir = REPO / "reports" / "f_leg_b" / "obs_crops"
for obs in ["ESP_045550_2180", "ESP_046328_2180", "ESP_069763_2235",
            "ESP_055978_2270", "ESP_042964_2160", "ESP_046959_2225"]:
    meds = []
    for p in sorted(crops_dir.glob(obs + "_*_ifcrop.tif")):
        with rasterio.open(p) as src:
            a = src.read(1)
        v = a[np.isfinite(a) & (a > 0)]
        q75, q25 = np.percentile(v, [75, 25])
        meds.append(float(np.median(v)))
        pid = p.name[len(obs) + 1:-11]
        print(f"{obs}  {pid:28s} median {np.median(v):.4f}  IQR {q75 - q25:.4f}  "
              f"cover {v.size / a.size:.0%}", flush=True)
    if len(meds) > 1:
        print(f"  -> frame median ratio {max(meds) / min(meds):.2f}x")
