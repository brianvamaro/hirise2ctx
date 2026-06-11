"""Check 3 — Plate Carree latitude distortion of labels and features.

The Murray mosaic is equirectangular with standard parallel 0: at latitude
phi, E-W projected lengths overstate ground truth by 1/cos(phi). Quantify the
three knock-ons per image:

  - effective true min-size floor = 1.4105 * sqrt(cos phi)  (filter applied
    in projected m^2)
  - true tile ground area = (320 m)^2 * cos phi -> bc>=50 true-density
    threshold scales as 1/cos phi
  - GLCM/texture E-W ground scale = 5 m * cos phi per pixel

Then Spearman of |distortion| vs per-image AUC (the geographic cluster makes
this mostly a 076499 outlier story; report it honestly).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MANIFEST = Path("hirise_40_vclaire.csv")
SWEEP = Path("models/_sweep_w0/20260611T013810Z/summary.parquet")

man = pd.read_csv(MANIFEST)
lat_col = [c for c in man.columns if c.lower() == "centerlat"][0]
obs_col = [c for c in man.columns if c.lower() == "obsid"][0]
man = man.set_index(obs_col)

summ = pd.read_parquet(SWEEP)
rec = summ[(summ.variant == "lightgbm_two_stage_balanced") & (summ.target_col == "boulder_count")]
auc = rec.set_index("held_out_obs_id")["meaningful_auc"]

df = pd.DataFrame({"lat": man[lat_col]}).join(auc.rename("auc"), how="inner")
df["cos_lat"] = np.cos(np.radians(df.lat))
df["min_size_true_m"] = 1.4105 * np.sqrt(df.cos_lat)
df["tile_true_area_frac"] = df.cos_lat            # vs nominal 320x320
df["bc50_true_density_x"] = 1 / df.cos_lat        # threshold inflation
df["glcm_ew_ground_m"] = 5.0 * df.cos_lat
df = df.sort_values("cos_lat")
print(df.to_string(float_format=lambda v: f"{v:.3f}"))

rho, p = spearmanr(df.cos_lat, df.auc)
print(f"\ncos(lat) vs AUC: Spearman rho={rho:+.3f} p={p:.4f} (n={len(df)})")
sub = df[df.index != "ESP_076499_1160"]
rho2, p2 = spearmanr(sub.cos_lat, sub.auc)
print(f"excluding ESP_076499_1160: rho={rho2:+.3f} p={p2:.4f} (n={len(sub)})")
print(f"\ncluster spread (excl. 076499): cos_lat {sub.cos_lat.min():.3f}-{sub.cos_lat.max():.3f} "
      f"(min-size floor {sub.min_size_true_m.min():.2f}-{sub.min_size_true_m.max():.2f} m)")
o = df.loc["ESP_076499_1160"]
print(f"ESP_076499_1160: lat {o.lat:.1f}, cos {o.cos_lat:.3f}, true min-size {o.min_size_true_m:.2f} m, "
      f"bc>=50 true-density x{o.bc50_true_density_x:.2f}, GLCM E-W ground {o.glcm_ew_ground_m:.2f} m/px")
df.to_csv("scripts/probes/_w1_latitude_distortion.csv")
