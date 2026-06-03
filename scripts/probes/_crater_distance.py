"""Tier 2 -- per-image crater-distance analysis vs Stage 7d attribution.

For each v2 ObsId, compute the great-circle distance from the HiRISE footprint
center to the nearest catalogued crater of {D >= 1 km, D >= 5 km, D >= 10 km}
using the Robbins & Hynek 2012 catalog (cache_v2/craters/). Cross-tab against
the Stage 7d per-image attribution table.

Hypothesis:
  - If boulders are locally-sourced (crater ejecta), composition_residual
    images should cluster at crater-PROXIMAL locations.
  - If boulders are transported by long-range processes (e.g. megatsunami),
    composition_residual images should NOT correlate with crater proximity
    or might cluster at crater-DISTAL locations.

The image-center distance is a first-order proxy; a tile-level analysis
would refine it. With only 5 composition_residual images, statistical power
is small either way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
CRATERS_PATH = ROOT / "cache_v2" / "craters" / "RobbinsCraters_20121016.tsv"
MARS_RADIUS_KM = 3389.5  # IAU 2009 mean


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance on a sphere (Mars), vectorised, returns km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * MARS_RADIUS_KM * np.arcsin(np.sqrt(a))


# Load + filter the Robbins catalog
print("Loading Robbins 2012 catalog...")
crater_cols = ["CRATER_ID", "LATITUDE_CIRCLE_IMAGE", "LONGITUDE_CIRCLE_IMAGE",
               "DIAM_CIRCLE_IMAGE", "MORPHOLOGY_EJECTA_1",
               "LAYER_1_EJECTARAD_EQUIV"]
craters = pd.read_csv(CRATERS_PATH, sep="\t", usecols=crater_cols,
                      encoding="latin-1", low_memory=False)
craters.columns = ["crater_id", "lat", "lon", "diam_km", "ejecta_morph_1",
                   "ejecta_radius_km_l1"]
# Robbins LONGITUDE_CIRCLE_IMAGE is in -180..180; manifest CenterLon_180 is same.
print(f"  Total craters (D >= 1 km): {len(craters)}")
print(f"  D distribution: min={craters['diam_km'].min():.2f}, "
      f"median={craters['diam_km'].median():.2f}, "
      f"max={craters['diam_km'].max():.2f}")

# Manifest
manifest = pd.read_csv(ROOT / "hirise_40_vclaire.csv")
print(f"\n  v2 manifest images: {len(manifest)}")
print(f"  Cohort lat range: {manifest['CenterLat'].min():.1f} to "
      f"{manifest['CenterLat'].max():.1f}")
print(f"  Cohort lon range: {manifest['CenterLon_180'].min():.1f} to "
      f"{manifest['CenterLon_180'].max():.1f}")

# Pre-filter craters to within +/- 5 degrees of cohort lat/lon for speed
lat_min, lat_max = manifest["CenterLat"].min() - 5, manifest["CenterLat"].max() + 5
nearby = craters[(craters["lat"] >= lat_min) & (craters["lat"] <= lat_max)].copy()
print(f"\nCraters in cohort lat band: {len(nearby)}")

# For each image, compute distance to nearest crater of various diameters
results = []
for _, m in manifest.iterrows():
    obs_lat = m["CenterLat"]
    obs_lon = m["CenterLon_180"]
    row = {"obs_id": m["ObsId"], "lat": obs_lat, "lon": obs_lon}
    for diam_threshold_km in (1.0, 5.0, 10.0, 25.0):
        sub = nearby[nearby["diam_km"] >= diam_threshold_km]
        if len(sub) == 0:
            row[f"nearest_D>={diam_threshold_km}_km_center"] = float("nan")
            row[f"nearest_D>={diam_threshold_km}_km_rim"] = float("nan")
            row[f"nearest_D>={diam_threshold_km}_radii"] = float("nan")
            continue
        d = haversine_km(obs_lat, obs_lon, sub["lat"].to_numpy(),
                         sub["lon"].to_numpy())
        idx = np.argmin(d)
        nearest_d_km_center = float(d[idx])
        nearest_radius = float(sub.iloc[idx]["diam_km"] / 2)
        # rim distance: subtract crater radius, but floor at 0 (inside crater)
        nearest_d_km_rim = max(0.0, nearest_d_km_center - nearest_radius)
        row[f"nearest_D>={diam_threshold_km}_km_center"] = nearest_d_km_center
        row[f"nearest_D>={diam_threshold_km}_km_rim"] = nearest_d_km_rim
        row[f"nearest_D>={diam_threshold_km}_radii"] = nearest_d_km_center / nearest_radius
    results.append(row)

dist = pd.DataFrame(results)
out_path = ROOT / "dataset_v2" / "crater_distance_v2.parquet"
dist.to_parquet(out_path, index=False)
print(f"\nWrote {out_path}")
print("\nPer-image distances (km to nearest crater RIM of D >= threshold):")
disp_cols = ["obs_id", "lat", "lon", "nearest_D>=1.0_km_rim",
             "nearest_D>=5.0_km_rim", "nearest_D>=10.0_km_rim",
             "nearest_D>=25.0_km_rim"]
print(dist[disp_cols].to_string(index=False, float_format="%.1f"))

# ---------- Cross-tab vs Stage 7d attribution ----------
attr = pd.read_parquet(ROOT / "dataset_v2" / "stage7d_attribution_shadow_0.10.parquet")
for rule in ("P4_area", "P2_count"):
    print(f"\n{'='*70}\nPARTITION: {rule}\n{'='*70}")
    sub = attr[attr["partition_rule"] == rule].merge(dist, on="obs_id", how="left")
    print(f"Eligible images: {len(sub)}")

    print("\nMean distance to crater RIM (D>=5 km) by attribution:")
    grouped = sub.groupby("attribution")["nearest_D>=5.0_km_rim"].describe()
    print(grouped[["count", "mean", "std", "min", "50%", "max"]].round(1))

    # Kruskal-Wallis test across attribution groups
    groups = [sub.loc[sub["attribution"] == cat, "nearest_D>=5.0_km_rim"].dropna()
              for cat in ["composition_residual", "dust_attributable", "no_signal"]]
    if all(len(g) >= 2 for g in groups):
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis across 3 categories (D>=5 km rim): "
              f"H={kw_stat:.3f}, p={kw_p:.4f}")

    # Pairwise MW: composition_residual vs (rest) at multiple diameter thresholds
    print()
    for diam in (1.0, 5.0, 10.0, 25.0):
        col = f"nearest_D>={diam}_km_rim"
        comp = sub[sub["attribution"] == "composition_residual"][col].dropna()
        rest = sub[sub["attribution"] != "composition_residual"][col].dropna()
        if len(comp) >= 2 and len(rest) >= 2:
            u, p = stats.mannwhitneyu(comp, rest, alternative="two-sided")
            print(f"MW comp_resid vs rest (D>={diam} km rim): U={u:.1f}, "
                  f"p={p:.4f}, comp_mean={comp.mean():.1f} km, "
                  f"rest_mean={rest.mean():.1f} km")

    # Per-image dump
    print("\nPer-image rim-distance + attribution:")
    disp = sub[["obs_id", "attribution", "nearest_D>=5.0_km_rim",
                "nearest_D>=10.0_km_rim", "nearest_D>=25.0_km_rim"]] \
        .sort_values(["attribution", "nearest_D>=5.0_km_rim"])
    print(disp.to_string(index=False, float_format="%.1f"))
