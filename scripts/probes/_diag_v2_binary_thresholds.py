"""Quick probe: what does the EXISTING v2 binary sweep say at the higher thresholds?

The user's concern: bc_ge_1 ("any boulder") is too lenient a positive rule.
Have we already got evidence that the model does *better* at fa_gt_1e-2
("boulder-rich")? The v2 binary sweep already ran all three targets x 4 scales.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401
import numpy as np
import pandas as pd

df = pd.read_parquet(REPO / "models/_sweep_binary/20260529T075754Z/aggregate.parquet")
cols = ["variant", "target_id", "scale_idx", "tile_size_px",
        "auc_mean", "auc_std", "n_real_folds", "brier_mean", "ece_mean",
        "lift_at_top_k_mean"]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print()
print("Threshold meaning:")
print("  bc_ge_1    = >=1 boulder (almost any tile that's not pristine flat)")
print("  fa_gt_1e-3 = boulder area > 0.1% of tile")
print("  fa_gt_1e-2 = boulder area > 1.0% of tile  ('boulder-rich')")

# Per-fold to see per-image fold variation
sf = pd.read_parquet(REPO / "models/_sweep_binary/20260529T075754Z/summary.parquet")
print(f"\nper-fold rows: {len(sf):,}  cols: {list(sf.columns)[:8]}...")
