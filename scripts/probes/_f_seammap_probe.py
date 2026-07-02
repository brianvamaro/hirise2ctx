"""Probe: SeamMap columns + sample PRODUCT_IDs for the F de-risk (EDR URL resolver)."""
import geopandas as gpd

g = gpd.read_file("cache/ctx_tiles/_seammap_E8_N44/MurrayLab_CTX_V01_E008_N44_SeamMap.shp")
print(len(g), "polygons")
print(list(g.columns))
id_cols = [c for c in g.columns if "IMG" in c.upper() or "PRODUCT" in c.upper() or "ID" in c.upper()]
print("id-ish cols:", id_cols)
for c in id_cols:
    print(c, "=>", g[c].iloc[0])
