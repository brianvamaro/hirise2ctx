"""W1 Rung 1a follow-up — shift-surface inspection.

(a) Full 5x5 AUC surfaces for the anti-signal images (coherent peak vs noisy
    spike), with n_neg per cell to expose base-rate fragility (ESP_054622_2240
    has only 3 negatives at center).
(b) Cohort-level mean AUC by offset — tests a *global* residual misalignment
    along the row/col axes that sub-tile coreg error would produce.

Writes scripts/probes/_w1_shift_surface.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

GRID = Path("scripts/probes/_w1_shift_rescore.parquet")
OUT_MD = Path("scripts/probes/_w1_shift_surface.md")

ANTI = [
    "ESP_055978_2270", "ESP_076499_1160", "ESP_047976_2020", "ESP_046328_2180",
    "ESP_071699_2260", "ESP_054000_2255", "ESP_055017_2055", "ESP_049242_2115",
    "ESP_054622_2240", "ESP_064510_2260", "ESP_055253_2245",
]

grid = pd.read_parquet(GRID)
grid["n_neg"] = grid.n_overlap - grid.n_pos

lines = ["# W1 — shift-rescore AUC surfaces", ""]

lines.append("## Cohort mean AUC by offset (38 images; global-residual test)")
coh = grid.pivot_table(index="di", columns="dj", values="auc", aggfunc="mean")
lines += ["```", coh.to_string(float_format=lambda v: f"{v:.3f}"), "```", ""]
print(coh.to_string(float_format=lambda v: f"{v:.3f}"))

# delta vs center per axis
center_auc = grid[(grid.di == 0) & (grid.dj == 0)].set_index("obs_id")["auc"]
for di, dj, label in [(1, 0, "(+1,0)"), (-1, 0, "(-1,0)"), (0, 1, "(0,+1)"), (0, -1, "(0,-1)")]:
    off = grid[(grid.di == di) & (grid.dj == dj)].set_index("obs_id")["auc"]
    delta = (off - center_auc).dropna()
    msg = f"- mean AUC delta at {label} vs center: {delta.mean():+.4f} (median {delta.median():+.4f}, n={len(delta)})"
    lines.append(msg)
    print(msg)
lines.append("")

lines.append("## Anti-signal image surfaces (AUC / n_neg)")
for obs in ANTI:
    g = grid[grid.obs_id == obs]
    auc_s = g.pivot(index="di", columns="dj", values="auc")
    neg_s = g.pivot(index="di", columns="dj", values="n_neg")
    lines += [f"### {obs}", "```",
              "AUC:", auc_s.to_string(float_format=lambda v: f"{v:.3f}"),
              "n_neg:", neg_s.to_string(),
              "```", ""]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT_MD}")

# print the smoking-gun surface for the transcript
for obs in ["ESP_054622_2240", "ESP_055978_2270", "ESP_047976_2020"]:
    g = grid[grid.obs_id == obs]
    print(f"\n{obs} AUC surface:")
    print(g.pivot(index="di", columns="dj", values="auc").to_string(float_format=lambda v: f"{v:.3f}"))
    print("n_neg:")
    print(g.pivot(index="di", columns="dj", values="n_neg").to_string())
