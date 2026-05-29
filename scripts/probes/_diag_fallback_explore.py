"""Deep-dive on the single Stage-3 fallback image to decide keep / nominal / drop.

Renders: CTX window, HiRISE-on-CTX, block-peak spatial map (where does it correlate?),
and the single-window before/after overlay. Plus the boulder count (the keep/drop tension).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import shift as nd_shift

from src import manifest as M
from src.config import load_config
from src.coregister import block_shift_field, load_shift, warp_hirise_to_ctx_grid
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR

OBS = sys.argv[1] if len(sys.argv) > 1 else "ESP_046803_2325"
cfg = load_config(REPO_ROOT / "config_v2.yaml")
df = M.load_manifest(cfg.manifest_path).set_index("ObsId")
FIG = REPO_ROOT / "reports" / "figures" / f"05_fallback_explore_{OBS}.png"


def norm(a):
    a = a.astype(np.float32); m = a > 0
    lo, hi = np.percentile(a[m], (2, 98)) if m.any() else (0.0, 1.0)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


coreg = load_shift(OBS, cfg.cache_dir)
ctx_tif = cfg.cache_dir / CTX_WINDOWS_SUBDIR / f"{OBS}.tif"
with rasterio.open(ctx_tif) as ds:
    ctx = ds.read(1).astype(np.float32); px_m = abs(ds.transform.a)
with rasterio.open(cfg.cache_dir / CTX_WINDOWS_SUBDIR / f"{OBS}_hirise_mask.tif") as ds:
    mask = ds.read(1)
hi, _, _ = warp_hirise_to_ctx_grid(OBS, jp2_url=str(df.loc[OBS, "JP2_URL"]),
                                   cache_dir=cfg.cache_dir, ctx_window_tif=ctx_tif)
field = pd.DataFrame(block_shift_field(hi, ctx, mask, block_px=128, min_coverage=0.98))

sid = json.loads((cfg.cache_dir / "reprojected_detections" / f"{OBS}.json").read_text())
n_poly = sid.get("n_polygons")

fft = coreg["fft_window"]; size = fft["size_px"]; r0, c0 = fft["row_off"], fft["col_off"]
sw = coreg["single_window"]
ctx_sub = ctx[r0:r0 + size, c0:c0 + size]
hi_sub = hi[r0:r0 + size, c0:c0 + size]
hi_sh = nd_shift(hi_sub, shift=(sw["dy_px"], sw["dx_px"]), order=1, mode="constant", cval=0.0)
n_good = int((field["peak"] >= 0.5).sum()) if len(field) else 0

fig, ax = plt.subplots(2, 2, figsize=(13, 11))
ax[0, 0].imshow(norm(ctx), cmap="gray"); ax[0, 0].set_title("CTX window — is there texture to lock onto?")
ax[0, 1].imshow(norm(hi), cmap="gray"); ax[0, 1].set_title("HiRISE on CTX grid (5 m/px)")
ax[1, 0].imshow(norm(ctx), cmap="gray", alpha=0.5)
if len(field):
    s = ax[1, 0].scatter(field["col_center"], field["row_center"], c=field["peak"],
                         cmap="RdYlGn", vmin=0, vmax=1, s=70, edgecolor="k", lw=0.3)
    fig.colorbar(s, ax=ax[1, 0], fraction=0.046, pad=0.02, label="block peak")
ax[1, 0].set_title(f"where does CTX<->HiRISE correlate?  {n_good}/{len(field)} blocks >= 0.5")
ax[1, 1].imshow(norm(ctx_sub), cmap="Reds", alpha=0.6)
ax[1, 1].imshow(norm(hi_sh), cmap="Blues", alpha=0.6)
ax[1, 1].set_title(f"single-window AFTER shift (CTX red / HiRISE blue, peak={sw['peak']:.2f})")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{OBS}  [{coreg['method']}]  |shift|={coreg['shift_m']['magnitude']:.0f} m  "
             f"global peak={coreg['peak_correlation']:.2f}  |  boulders={n_poly:,}  "
             f"coverage={sid.get('n_polygons')}", fontsize=12)
fig.tight_layout(); fig.savefig(FIG, dpi=110)
print(f"n_polygons={n_poly:,}  blocks={len(field)}  good(>=0.5)={n_good}  "
      f"global_peak={coreg['peak_correlation']:.3f}  method={coreg['method']}")
print(f"wrote {FIG}")
