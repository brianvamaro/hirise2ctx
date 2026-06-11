"""W1 Rung 3 — full-res HiRISE visual sampling of BoulderNet detections.

For the 8 post-fix anti-signal images + 2 healthy controls, render a full-res
(native ~0.25-0.5 m/px) JP2 crop centred on the densest detection cluster,
with polygon outlines. Both the JP2 and the BoulderNet shapefile carry the
same source georeferencing (the shapefile was produced from the JP2), so a
same-CRS overlay needs no SP1 reprojection.

Output: reports/figures/w1_rung3_{obs_id}.png — for visual classification of
detections as real boulders vs ripple crests / dune brinks / crater texture.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

DET_ROOT = Path("C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_40_vClaire")
JP2_DIR = Path("cache_v2/hirise_jp2")
FIGDIR = Path("reports/figures")
CROP = 1024  # px

IMAGES = [
    # (obs_id, anti?)
    ("ESP_076499_1160", True), ("ESP_055978_2270", True), ("ESP_054000_2255", True),
    ("ESP_046328_2180", True), ("ESP_064510_2260", True), ("ESP_047976_2020", True),
    ("ESP_049242_2115", True), ("ESP_059686_2235", True),
    ("ESP_042964_2160", False), ("ESP_066634_2210", False),
]

for obs, anti in IMAGES:
    shp = list((DET_ROOT / obs).glob("*mask-nms.shp"))
    jp2 = JP2_DIR / f"{obs}_RED.JP2"
    if not shp or not jp2.exists():
        print(f"{obs}: MISSING {'shp' if not shp else 'jp2'}, skipped")
        continue
    gdf = gpd.read_file(shp[0])
    cx = gdf.geometry.centroid.x.to_numpy()
    cy = gdf.geometry.centroid.y.to_numpy()

    with rasterio.open(jp2) as src:
        # densest cluster in pixel space (200 px bins)
        rows_px, cols_px = rasterio.transform.rowcol(src.transform, cx, cy)
        rows_px = np.asarray(rows_px); cols_px = np.asarray(cols_px)
        rb = rows_px // 200; cb = cols_px // 200
        ids = rb * 1_000_000 + cb
        vals, counts = np.unique(ids, return_counts=True)
        best = vals[np.argmax(counts)]
        r_c = int(best // 1_000_000) * 200 + 100
        c_c = int(best % 1_000_000) * 200 + 100
        r0 = max(0, min(src.height - CROP, r_c - CROP // 2))
        c0 = max(0, min(src.width - CROP, c_c - CROP // 2))
        win = Window(c0, r0, CROP, CROP)
        img = src.read(1, window=win).astype(np.float32)
        wt = src.window_transform(win)
        px_m = abs(src.transform.a)

    # polygons intersecting the crop
    x0, y1 = wt.c, wt.f
    x1 = x0 + CROP * wt.a
    y0 = y1 + CROP * wt.e
    sub = gdf.cx[min(x0, x1):max(x0, x1), min(y0, y1):max(y0, y1)]

    valid = img[img > 0]
    if valid.size == 0:
        print(f"{obs}: crop empty (nodata), skipped")
        continue
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.imshow(img, cmap="gray",
              vmin=np.percentile(valid, 2), vmax=np.percentile(valid, 98),
              extent=(x0, x1, y0, y1))
    sub.boundary.plot(ax=ax, color="red", linewidth=0.6)
    tag = "ANTI-SIGNAL" if anti else "control"
    ax.set_title(f"{obs} ({tag}) — {len(sub)} detections in {CROP*px_m:.0f} m crop "
                 f"@ {px_m:.2f} m/px, densest cluster")
    ax.set_axis_off()
    out = FIGDIR / f"w1_rung3_{obs}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"{obs}: {len(sub)} polys in crop -> {out}")
