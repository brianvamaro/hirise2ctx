"""Print the full per-bin comparison table for the compression-fix sweep.

Reads the most recent sweep's aggregate.parquet and pivots to a table where each
row is a (variant, scale) and the columns are mean_pred / ratio per truth bin —
the granular view that the in-sweep compression_score scalar collapses.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401
import numpy as np
import pandas as pd

ROOT = REPO_ROOT / "models" / "_sweep_compression_fixes"
runs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
print(f"runs: {[r.name for r in runs]}")
df = pd.read_parquet(runs[-1] / "aggregate.parquet")
print(f"\nLoaded {len(df)} variant-scale rows from {runs[-1].name}")
print(f"columns: {list(df.columns)}\n")

BINS = ("zero", "0_to_1e-4", "1e-4_to_1e-3", "1e-3_to_1e-2", "1e-2_to_max")

# Per-bin mean_pred table
print("\n=== Per-bin MEAN_PRED (linear scale; truth shown as reference row) ===")
pred = df[[f"{b}__mean_pred" for b in BINS]].copy()
pred.columns = list(BINS)
pred.insert(0, "variant", df["variant"])
pred.insert(1, "S", df["tile_size_px"])
pred["spearman"] = df["spearman_rho_mean"]
pred["AUC"] = df["presence_auc_mean"]
print(pred.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# Per-bin ratio table
print("\n=== Per-bin RATIO mean_pred / mean_true (1.0 = perfect; >>1 over, <<1 under) ===")
ratio = df[[f"{b}__ratio" for b in BINS]].copy()
ratio.columns = list(BINS)
ratio.insert(0, "variant", df["variant"])
ratio.insert(1, "S", df["tile_size_px"])
print(ratio.to_string(index=False, float_format=lambda v: "—" if not np.isfinite(v) else f"{v:.3f}"))

# Truth distribution row (constant across variants since truth doesn't change)
truth_row = df[[f"{b}__mean_true" for b in BINS]].iloc[0]
print("\nReference mean_true per bin (constant across variants):")
print({b: f"{truth_row[f'{b}__mean_true']:.5f}" for b in BINS if np.isfinite(truth_row[f'{b}__mean_true'])})

# Compact decision table: variant, scale, headline ranking, headline calibration
print("\n=== Decision table: ranking vs calibration trade-off ===")
hdr = df[["variant", "tile_size_px", "spearman_rho_mean", "presence_auc_mean",
          "compression_score", "zero__mean_pred", "1e-2_to_max__mean_pred",
          "1e-2_to_max__ratio"]].copy()
hdr.columns = ["variant", "S", "Spearman", "AUC", "compression_score",
               "zero_pred", "high_pred", "high_ratio"]
print(hdr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# Persist to a markdown file for the notebook
out_md = Path(__file__).with_suffix(".md")
lines = [
    f"# Compression-fix sweep — {runs[-1].name}",
    "",
    "Composite metric tables from the v2-dev within-image sweep (20 folds).",
    "",
    "## Per-bin mean prediction (linear scale)",
    "",
    pred.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
    "",
    "## Per-bin mean_pred / mean_true ratio",
    "",
    ratio.to_string(index=False, float_format=lambda v: "—" if not np.isfinite(v) else f"{v:.3f}"),
    "",
    "## Decision table",
    "",
    hdr.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
    "",
]
out_md.write_text("\n".join(lines), encoding="utf-8")
print(f"\nMarkdown -> {out_md}")
