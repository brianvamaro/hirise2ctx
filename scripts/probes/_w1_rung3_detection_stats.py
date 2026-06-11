"""W1 Rung 3 — BoulderNet detection-quality statistics, anti-signal vs cohort.

Per image: detection count, density per km^2 of labeled footprint, score
distribution (median, fraction below 0.5), equivalent-diameter distribution
(median, fraction within 10% of the min-size filter floor), edge fraction.
Mann-Whitney U anti-signal (8) vs rest (30) on each statistic.

Low scores / floor-hugging sizes / extreme densities on the anti-signal
images would point at label content (rung 3); indistinguishable
distributions push the diagnosis down to rung 4/5.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DET = Path("cache_v2/reprojected_detections")
LABELS = Path("dataset_v2/labels")
ANTI = {
    "ESP_076499_1160", "ESP_055978_2270", "ESP_054000_2255", "ESP_046328_2180",
    "ESP_064510_2260", "ESP_047976_2020", "ESP_049242_2115", "ESP_059686_2235",
}
MIN_SIZE_M = 1.4105  # Stage 4 filter floor (DECISIONS.md 2026-05-26)
OUT_MD = Path("scripts/probes/_w1_rung3_detection_stats.md")

rows = []
for f in sorted(DET.glob("*.gpkg")):
    obs = f.stem
    lab = LABELS / f"{obs}.parquet"
    if not lab.exists():
        continue
    g = gpd.read_file(f)
    n_s64 = int((pd.read_parquet(lab, columns=["scale_idx"]).scale_idx == 3).sum())
    area_km2 = n_s64 * 0.320 ** 2
    diam = 2.0 * np.sqrt(g.geometry.area.to_numpy() / np.pi)
    kept = diam >= MIN_SIZE_M  # mirror the Stage 4 size filter
    rows.append(
        dict(
            obs_id=obs,
            anti=obs in ANTI,
            n_det=len(g),
            n_kept=int(kept.sum()),
            dens_km2=len(g) / area_km2,
            score_med=float(g.score.median()),
            score_lo_frac=float((g.score < 0.5).mean()),
            diam_med=float(np.median(diam)),
            diam_floor_frac=float(((diam >= MIN_SIZE_M) & (diam < MIN_SIZE_M * 1.1)).mean()),
            edge_frac=float(g.is_at_edge.mean()) if "is_at_edge" in g else np.nan,
        )
    )

df = pd.DataFrame(rows).sort_values("anti", ascending=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

stats_cols = ["n_det", "dens_km2", "score_med", "score_lo_frac", "diam_med",
              "diam_floor_frac", "edge_frac"]
lines = []
for c in stats_cols:
    a = df.loc[df.anti, c].dropna()
    h = df.loc[~df.anti, c].dropna()
    u, p = mannwhitneyu(a, h)
    lines.append(f"- `{c}`: anti median {a.median():.3f} vs rest {h.median():.3f} (MWU p={p:.3f})")
    print(lines[-1])

OUT_MD.write_text(
    "# W1 Rung 3 — BoulderNet detection stats, anti-signal (8) vs cohort (30)\n\n"
    "```\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```\n\n"
    "## Anti vs rest (Mann-Whitney U)\n" + "\n".join(lines) + "\n",
    encoding="utf-8",
)
print(f"wrote {OUT_MD}")
