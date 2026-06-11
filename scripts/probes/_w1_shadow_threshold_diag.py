"""Check 1 mechanism — why are shadow features identically zero on
ESP_046328_2180 and ESP_064510_2260?

The shadow DN cut is `mode - 20` (features.py _compute_dn_thresholds, fixed
offset in DN units). On a low-contrast CTX window, no pixel sits 20 DN below
the mode -> shadow_fraction == 0 everywhere. Report, per image: masked modal
DN, the cut, the fraction of covered pixels below the cut, plus DN p1/p99
contrast width. Flag images where the shadow channel is dead or near-dead.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

CACHE = Path("cache_v2/ctx_windows")
OFFSET = 20

rows = []
for f in sorted(CACHE.glob("*.tif")):
    if f.stem.endswith("_hirise_mask"):
        continue
    obs = f.stem
    mask_f = CACHE / f"{obs}_hirise_mask.tif"
    if not mask_f.exists():
        continue
    with rasterio.open(f) as src:
        arr = src.read(1)
    with rasterio.open(mask_f) as src:
        mask = src.read(1).astype(bool)
    covered = arr[mask & (arr > 0)]
    if covered.size < 1000:
        continue
    counts = np.bincount(covered, minlength=256)
    mode = int(counts.argmax())
    cut = max(0, mode - OFFSET)
    frac_below = float((covered < cut).mean())
    rows.append(dict(obs_id=obs, mode=mode, cut=cut,
                     dn_p1=int(np.percentile(covered, 1)),
                     dn_p99=int(np.percentile(covered, 99)),
                     contrast_p99_p1=int(np.percentile(covered, 99) - np.percentile(covered, 1)),
                     shadow_frac_image=frac_below))

df = pd.DataFrame(rows).sort_values("shadow_frac_image")
df["dead"] = df.shadow_frac_image == 0.0
df["near_dead"] = df.shadow_frac_image < 0.001
print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print(f"\ndead shadow channel: {df.dead.sum()} images; near-dead (<0.1% px): {df.near_dead.sum()}")
df.to_csv("scripts/probes/_w1_shadow_threshold_diag.csv", index=False)
