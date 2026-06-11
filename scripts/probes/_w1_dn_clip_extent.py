"""Size the DN<=1 bottom-clip inside HiRISE coverage for the two affected
windows (plus a healthy control)."""
import numpy as np
import rasterio

for obs in ["ESP_046328_2180", "ESP_064510_2260", "ESP_042964_2160"]:
    with rasterio.open(f"cache_v2/ctx_windows/{obs}.tif") as s:
        a = s.read(1)
    with rasterio.open(f"cache_v2/ctx_windows/{obs}_hirise_mask.tif") as s:
        m = s.read(1).astype(bool)
    c = a[m]
    print(f"{obs}: covered px {c.size:,} | DN==0 {(c == 0).mean():.4f} | "
          f"DN==1 {(c == 1).mean():.4f} | DN<=5 {(c <= 5).mean():.4f} | "
          f"DN<=10 {(c <= 10).mean():.4f}")
