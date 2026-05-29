import json, re, sys
o = "ESP_069669_2220"
c = json.load(open(f"cache_v2/coregistration/{o}.json"))
s = json.load(open(f"cache_v2/reprojected_detections/{o}.json"))
def name(w):
    m = re.search(r'PROJC[RS]+\["([^"]+)"', w) if w else None
    return m.group(1) if m else None
print("CTX window raster CRS :", name(c["ctx_crs_wkt"]))
print("detection gpkg CRS    :", name(s["target_crs_wkt"]))
print("HiRISE source CRS     :", name(s["source_crs_wkt"]))
