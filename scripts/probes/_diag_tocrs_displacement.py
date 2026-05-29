import sys; from pathlib import Path
REPO=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO))
import numpy as np, rasterio
from pyproj import CRS
from src.config import load_config
from src.detections import load_reprojected
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR
cfg=load_config(REPO/"config_v2.yaml")
for OBS in ["ESP_069669_2220","ESP_017355_2260"]:
    g=load_reprojected(OBS,cfg.cache_dir)
    with rasterio.open(cfg.cache_dir/CTX_WINDOWS_SUBDIR/f"{OBS}.tif") as ds: wcrs=ds.crs
    same=CRS.from_user_input(g.crs).equals(CRS.from_user_input(wcrs))
    c0=g.geometry.centroid
    g2=g.to_crs(wcrs)
    c1=g2.geometry.centroid
    d=np.hypot(c1.x.to_numpy()-c0.x.to_numpy(), c1.y.to_numpy()-c0.y.to_numpy())
    print(f"{OBS}: gdf.crs.equals(window)? {same}  to_crs displacement m: median={np.median(d):.3f} max={d.max():.3f} mean={d.mean():.3f}")
