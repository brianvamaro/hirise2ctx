"""Why is per-bin RMSE NaN for all bins except 'zero'?

Inspect the metrics.json files: are positive bins genuinely empty (no test
tiles in those bins per fold), or is the value present but my probe missed it?
"""
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

print("variant            scale  fold  bin              n_tiles   rmse           mean_true     mean_pred")
print("-" * 120)
for variant in ("lightgbm_tweedie", "lightgbm_log1p_huber", "lightgbm_two_stage"):
    for scale_dir in sorted((REPO / "models" / variant).glob("*/scale_S*"), key=lambda p: int(p.name[len("scale_S"):])):
        m = json.loads((scale_dir / "metrics.json").read_text())
        scale = scale_dir.name
        for f in m["per_fold"]:
            fold = f["fold_idx"]
            for b in f.get("per_bin_rmse", []):
                rmse_val = "nan" if b["rmse"] is None or (isinstance(b["rmse"], float) and b["rmse"] != b["rmse"]) else f"{b['rmse']:.4e}"
                mt = "-" if b["mean_true"] is None or (isinstance(b["mean_true"], float) and b["mean_true"] != b["mean_true"]) else f"{b['mean_true']:.4e}"
                mp = "-" if b["mean_pred"] is None or (isinstance(b["mean_pred"], float) and b["mean_pred"] != b["mean_pred"]) else f"{b['mean_pred']:.4e}"
                print(f"{variant:<19s} {scale:<6s} {fold:>4d}  {b['bin']:<16s} {b['n_tiles']:>7d}   {rmse_val:<14s} {mt:<13s} {mp}")
        break  # only show first metrics.json per variant for brevity
