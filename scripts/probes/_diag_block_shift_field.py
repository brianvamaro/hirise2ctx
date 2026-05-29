"""Smoke-test block_shift_field on a good solve vs the failed one."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import rasterio

from src import manifest as M
from src.config import load_config
from src.coregister import block_shift_field, load_shift, warp_hirise_to_ctx_grid
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR

cfg = load_config(REPO_ROOT / "config_v2.yaml")
df = M.load_manifest(cfg.manifest_path).set_index("ObsId")

for obs in ["ESP_069669_2220", "ESP_049242_2115"]:
    coreg = load_shift(obs, cfg.cache_dir)
    ctx_tif = cfg.cache_dir / CTX_WINDOWS_SUBDIR / f"{obs}.tif"
    mask_tif = cfg.cache_dir / CTX_WINDOWS_SUBDIR / f"{obs}_hirise_mask.tif"
    with rasterio.open(ctx_tif) as ds:
        ctx = ds.read(1).astype(np.float32); px_m = abs(ds.transform.a)
    with rasterio.open(mask_tif) as ds:
        mask = ds.read(1)
    hi, _, _ = warp_hirise_to_ctx_grid(obs, jp2_url=str(df.loc[obs, "JP2_URL"]),
                                       cache_dir=cfg.cache_dir, ctx_window_tif=ctx_tif)
    field = pd.DataFrame(block_shift_field(hi, ctx, mask, block_px=128, min_coverage=0.98))
    g_dx, g_dy = coreg["shift_px"]["dx"], coreg["shift_px"]["dy"]
    good = field["peak"] >= 0.5
    resid_m = np.hypot(field["dx_px"] - g_dx, field["dy_px"] - g_dy) * px_m
    print(f"\n=== {obs}  global peak={coreg['peak_correlation']:.3f}  |shift|={coreg['shift_m']['magnitude']:.0f} m ===")
    print(f"  n_blocks={len(field)}  frac peak>=0.5={good.mean():.2f}")
    print(f"  median |local-global| (good blocks) = {np.median(resid_m[good]):.0f} m" if good.any() else "  (no good blocks)")
    print(f"  local dx/dy std = {(field['dx_px']*px_m).std():.0f}/{(field['dy_px']*px_m).std():.0f} m")
