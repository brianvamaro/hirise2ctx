"""Probe: world-coordinate bounds of the E8_N44 A1-payoff crop (for f_pilot_extract_crop)."""
import rasterio

R0, C0, SIZE = 1504, 8992, 15008  # native 5 m/px, scripts/striping_a1_infer_crop.py
CF = 32
with rasterio.open("reports/map_region/E8_N44_abundance.tif") as ds:
    t = ds.transform
    print("tile CRS:", ds.crs)
x0, y0 = t * (C0 / CF, R0 / CF)
x1, y1 = t * ((C0 + SIZE) / CF, (R0 + SIZE) / CF)
print(f"crop bounds (minx, miny, maxx, maxy) = ({min(x0,x1):.1f}, {min(y0,y1):.1f}, {max(x0,x1):.1f}, {max(y0,y1):.1f})")
