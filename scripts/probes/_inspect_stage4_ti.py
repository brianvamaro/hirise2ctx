import json, sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

df = pd.read_parquet(REPO_ROOT / "dataset/labels/ESP_069669_2220.parquet")
prov = json.loads((REPO_ROOT / "dataset/labels/ESP_069669_2220.json").read_text())
for s in (8, 16, 32, 64):
    sub = df[df.tile_size_px == s]
    print(f"S={s:>2}  ti range = [{sub.ti.min()}, {sub.ti.max()}]   tj range = [{sub.tj.min()}, {sub.tj.max()}]   n={len(sub)}")
print("mosaic_row_origin:", prov["mosaic_row_origin"], "mosaic_col_origin:", prov["mosaic_col_origin"])
print("finest_grid_cells:", prov["finest_grid_cells"])
