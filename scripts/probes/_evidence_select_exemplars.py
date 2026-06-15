"""Pick exemplar images for the model_evidence gallery + basis figure.

Merges per-image held-out AUC + base rate (from the banked frozen predictions),
terrain_category, and the W1 dossier so we can choose a hybrid regime+geomorphic
set. Also dumps the densest + an empty S=32 tile for ESP_053989_2260 (basis fig).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 50)

pred = pd.read_parquet(REPO / "models" / "fang_probe" /
                       "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2" / "predictions.parquet")
rows = []
for obs, g in pred.groupby("obs_id"):
    base = g["y_true"].mean()
    auc = roc_auc_score(g["y_true"], g["y_pred"]) if g["y_true"].nunique() > 1 else np.nan
    rows.append(dict(obs_id=obs, n_tiles=len(g), base_rate=base, auc=auc))
m = pd.DataFrame(rows).set_index("obs_id")

terr = pd.read_parquet(REPO / "dataset_v2" / "terrain_classification_v2.parquet")[
    ["obs_id", "terrain_category", "BoulderLabel", "CenterLat", "CenterLon_180", "note"]
].set_index("obs_id")
dos = pd.read_parquet(REPO / "dataset_v2" / "w1_dossier.parquet")[
    ["attributed_cause"]]
tbl = m.join(terr).join(dos)
tbl = tbl.sort_values("auc")
print("=== per-image: AUC | base_rate | terrain | cause | label ===")
print(tbl[["auc", "base_rate", "terrain_category", "attributed_cause",
           "BoulderLabel", "CenterLat"]].round(3).to_string())

print("\n=== terrain_category counts ===")
print(terr["terrain_category"].value_counts().to_string())

# --- basis figure tiles for ESP_053989_2260 (S=32 = scale_idx 2) ---
OBS = "ESP_053989_2260"
lab = pd.read_parquet(REPO / "dataset_v2" / "labels" / f"{OBS}.parquet")
s32 = lab[lab["scale_idx"] == 2].copy()
print(f"\n=== {OBS} S=32 tiles: {len(s32)} ; cols={[c for c in s32.columns]}")
rich = s32.sort_values("boulder_count", ascending=False).iloc[0]
poor = s32[s32["boulder_count"] == 0].sample(1, random_state=1).iloc[0]
for tag, t in [("RICH", rich), ("POOR", poor)]:
    cx = (t["xmin"] + t["xmax"]) / 2
    cy = (t["ymin"] + t["ymax"]) / 2
    print(f"{tag}: ti={int(t['ti'])} tj={int(t['tj'])} bc={int(t['boulder_count'])} "
          f"fa={t['fractional_area']:.4f} centre=({cx:.1f},{cy:.1f}) "
          f"bounds=({t['xmin']:.1f},{t['ymin']:.1f},{t['xmax']:.1f},{t['ymax']:.1f})")
