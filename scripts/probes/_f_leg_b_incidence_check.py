"""Check INCIDENCE is available for every leg-B crop frame (needed for minnaert)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import geopandas as gpd
import pandas as pd

SEAM_DIR = REPO / "cache" / "ctx_tiles"

# SeamMap metadata corrections, verified against the PDS volume index
# (_f_leg_b_fetch_true_incidence.py; DECISIONS 2026-07-05). The SeamMap value
# 4.2759 is a decimal-shift of the true 42.76 — it collapsed ESP_053989/068483
# under the minnaert mapping (cos^k division ~1.0 instead of ~0.83).
OVERRIDES = {"P20_008839_2269_XI_46N046W": 42.76}
om = pd.read_csv(REPO / "reports" / "f_leg_b" / "obs_frame_map.csv")

crops_dir = REPO / "reports" / "f_leg_b" / "obs_crops"
have_crop = {p.name for p in crops_dir.glob("*_ifcrop.tif")}
om["has_crop"] = [f"{r.obs_id}_{r.PRODUCT_ID}_ifcrop.tif" in have_crop
                  for r in om.itertuples()]
need = om[om.has_crop]
print(f"{len(need)} (obs, frame) pairs with crops; "
      f"{need.PRODUCT_ID.nunique()} unique frames across {need.tile.nunique()} tiles")

missing = []
inc_rows = []
for tile, g in need.groupby("tile"):
    gpkg = SEAM_DIR / f"_frames_{tile}.gpkg"
    if not gpkg.exists():
        missing.append((tile, "gpkg missing"))
        continue
    fr = gpd.read_file(gpkg)
    cols = [c for c in fr.columns if "INC" in c.upper()]
    if not cols:
        missing.append((tile, f"no INCIDENCE col; cols={list(fr.columns)}"))
        continue
    fr = fr.set_index("PRODUCT_ID")
    for pid in g.PRODUCT_ID.unique():
        if pid in OVERRIDES:
            inc_rows.append(dict(tile=tile, PRODUCT_ID=pid,
                                 incidence=OVERRIDES[pid]))
            print(f"  OVERRIDE {pid}: seammap "
                  f"{float(fr.loc[pid, cols[0]]) if pid in fr.index else float('nan'):.2f} "
                  f"-> {OVERRIDES[pid]} (PDS index)")
        elif pid in fr.index:
            inc_rows.append(dict(tile=tile, PRODUCT_ID=pid,
                                 incidence=float(fr.loc[pid, cols[0]])))
        else:
            missing.append((tile, f"{pid} not in gpkg"))

inc = pd.DataFrame(inc_rows)
print(f"\nincidence found for {len(inc)} frames; "
      f"range {inc.incidence.min():.1f}–{inc.incidence.max():.1f} deg")
if missing:
    print("\nMISSING:")
    for m in missing:
        print(f"  {m}")
else:
    print("all frames covered ✓")
inc.to_csv(REPO / "reports" / "f_leg_b" / "frame_incidence.csv", index=False)
print("wrote reports/f_leg_b/frame_incidence.csv")
