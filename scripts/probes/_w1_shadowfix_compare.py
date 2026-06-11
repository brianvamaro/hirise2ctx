"""Compare per-image meaningful AUC before/after the DN-clip shadow fix for
the banked cell (two_stage_balanced x boulder_count @ S=64)."""
import pandas as pd

PRE = "models/_sweep_w0/20260611T013810Z/summary.parquet"
POST = "models/_sweep_w0/20260611T054855Z/summary.parquet"

def cell(p):
    s = pd.read_parquet(p)
    s = s[(s.variant == "lightgbm_two_stage_balanced") & (s.target_col == "boulder_count")]
    return s.set_index("held_out_obs_id")["meaningful_auc"]

pre, post = cell(PRE), cell(POST)
cmp = pd.DataFrame({"pre": pre, "post": post})
cmp["delta"] = cmp.post - cmp.pre
print("Two fixed images:")
print(cmp.loc[["ESP_046328_2180", "ESP_064510_2260"]].to_string(float_format=lambda v: f"{v:.3f}"))
print("\nLargest movers elsewhere (|delta|>0.02):")
others = cmp.drop(["ESP_046328_2180", "ESP_064510_2260"])
print(others[others.delta.abs() > 0.02].sort_values("delta").to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\ncohort: median {pre.median():.3f} -> {post.median():.3f}; "
      f"anti-signal {(pre < 0.5).sum()} -> {(post < 0.5).sum()}")
