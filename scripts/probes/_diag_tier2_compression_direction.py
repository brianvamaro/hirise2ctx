"""Is Tier-2 compression on the low end, the high end, or both?

Per true-abundance bin: mean true vs mean predicted + ratio. Regression-to-the-mean
would over-predict low bins (ratio>1) and under-predict high bins (ratio<1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
df = pd.read_parquet(REPO / "models" / "fang_tier2" /
                     "tier2_mlp_reg_emb_fractional_area_S32" / "1e01ad8b17447599" /
                     "predictions.parquet")
yt = df["y_true"].to_numpy()
yp = np.clip(df["y_pred"].to_numpy(), 0, None)
print(f"n={len(yt)}  true: min={yt.min():.4f} max={yt.max():.4f} mean={yt.mean():.4f} "
      f"exact-zero={np.mean(yt==0):.1%}")
print(f"        pred: min={yp.min():.4f} max={yp.max():.4f} mean={yp.mean():.4f} "
      f"near-zero(<1e-4)={np.mean(yp<1e-4):.1%}")

print("\n--- by TRUE quantile bin (does pred reach the lows / the highs?) ---")
edges = np.quantile(yt, np.linspace(0, 1, 11))
edges[-1] += 1e-9
lab = np.clip(np.digitize(yt, edges) - 1, 0, 9)
print(f"{'bin':>4} {'true_lo':>9} {'true_hi':>9} {'mean_true':>10} {'mean_pred':>10} {'pred/true':>9} {'n':>7}")
for b in range(10):
    m = lab == b
    if not m.any():
        continue
    mt, mp = yt[m].mean(), yp[m].mean()
    ratio = mp / mt if mt > 0 else np.inf
    print(f"{b:>4} {edges[b]:>9.4f} {edges[b+1]:>9.4f} {mt:>10.4f} {mp:>10.4f} {ratio:>9.2f} {m.sum():>7}")

print("\n--- fixed abundance bins (operational reading) ---")
fb = [0, 1e-4, 1e-3, 1e-2, 3e-2, 1.0]
flab = np.clip(np.digitize(yt, fb) - 1, 0, len(fb) - 2)
for b in range(len(fb) - 1):
    m = flab == b
    if not m.any():
        continue
    mt, mp = yt[m].mean(), yp[m].mean()
    print(f"  true in [{fb[b]:.0e},{fb[b+1]:.0e}): n={m.sum():>6}  mean_true={mt:.4f}  "
          f"mean_pred={mp:.4f}  ratio={mp/mt if mt>0 else float('nan'):.2f}")
