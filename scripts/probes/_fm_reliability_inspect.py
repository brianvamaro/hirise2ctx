"""Inspect the §2.7 per-image novelty vs AUC table: confound diagnosis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
df = pd.read_parquet(REPO / "models" / "fang_probe" / "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2" / "predictions.parquet")
tbl = pd.read_csv(REPO / "reports" / "reliability" / "per_image_novelty.csv")

m = tbl[tbl.method == "mahalanobis"].set_index("obs_id")
print("=== Lowest-AUC images (where the frozen recipe is weakest) ===")
print(m.sort_values("auc")[["auc", "novelty"]].head(8).round(3).to_string())
print("\n=== Highest-novelty images (what the overlay would flag) ===")
print(m.sort_values("novelty", ascending=False)[["auc", "novelty"]].head(8).round(3).to_string())

# the outlier confound
for name in ["mahalanobis", "knn_cos50"]:
    s = tbl[tbl.method == name].set_index("obs_id")
    rho_all, p_all = spearmanr(s.novelty, s.auc)
    s2 = s.drop(index="ESP_076499_1160")
    rho_no, p_no = spearmanr(s2.novelty, s2.auc)
    print(f"\n[{name}] Spearman all n={len(s)}: rho={rho_all:+.3f} p={p_all:.3f}  | "
          f"drop ESP_076499_1160 (n={len(s2)}): rho={rho_no:+.3f} p={p_no:.3f}")
    print(f"   ESP_076499_1160: AUC={s.loc['ESP_076499_1160','auc']:.3f} "
          f"novelty_rank={int((s.novelty > s.loc['ESP_076499_1160','novelty']).sum())+1}/{len(s)}")
