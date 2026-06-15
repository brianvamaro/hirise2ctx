"""Compression vs ranking across the banked Tier-2 fractional_area variants."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
T2 = REPO / "models" / "fang_tier2"
VARIANTS = {
    "mlp_reg (emb)": "tier2_mlp_reg_emb_fractional_area_S32",
    "tweedie (emb)": "tier2_lightgbm_tweedie_emb_fractional_area_S32",
    "two_stage_bal (emb)": "tier2_lightgbm_two_stage_balanced_emb_fractional_area_S32",
    "mlp_reg (t1)": "tier2_mlp_reg_t1_fractional_area_S32",
}


def load(name):
    hits = list((T2 / name).glob("*/predictions.parquet"))
    return pd.read_parquet(hits[0]) if hits else None


def stats(df):
    yt = df["y_true"].to_numpy(); yp = np.clip(df["y_pred"].to_numpy(), 0, None)
    rho = spearmanr(yt, yp).correlation
    fb = [0, 1e-4, 1e-3, 1e-2, 3e-2, 1.0]
    lab = np.clip(np.digitize(yt, fb) - 1, 0, len(fb) - 2)
    lowr = yp[lab == 0].mean() / max(yt[lab == 0].mean(), 1e-9)        # over-pred lows
    # fixed top bin 1e-2..max
    topm = yt > 1e-2
    topr = yp[topm].mean() / yt[topm].mean()
    nearzero = float(np.mean(yp < 1e-4))
    # marginal-distribution mismatch: Wasserstein-ish via sorted-quantile L1
    q = np.linspace(0, 1, 101)
    wass = float(np.mean(np.abs(np.quantile(yt, q) - np.quantile(yp, q))))
    return rho, lowr, topr, nearzero, wass, yt.mean(), yp.mean()


print(f"{'variant':>22} {'rho':>6} {'low_ovr':>8} {'top_ratio':>9} "
      f"{'p<1e-4':>7} {'distL1':>8} {'mean_t':>8} {'mean_p':>8}")
print(f"{'(truth near-zero share = 18%)':>22}")
for label, name in VARIANTS.items():
    df = load(name)
    if df is None:
        print(f"{label:>22}  (missing)")
        continue
    rho, lowr, topr, nz, wass, mt, mp = stats(df)
    print(f"{label:>22} {rho:>6.3f} {lowr:>8.0f} {topr:>9.2f} {nz:>7.1%} "
          f"{wass:>8.5f} {mt:>8.4f} {mp:>8.4f}")
print("\nlow_ovr = mean_pred/mean_true in the truly-zero bin (regression-to-mean -> >>1)")
print("top_ratio = mean_pred/mean_true for true fa>1e-2 (compression -> <1)")
print("distL1 = mean |quantile(true)-quantile(pred)| (marginal mismatch; 0 = matched)")
