"""PLAN_FM 2.4 Tier-2: quantify the zero-inflation ceiling on the regression
Spearman + companion metrics (Brian, 2026-06-12). Post-hoc on banked predictions
(no model re-run). Per held-out image then averaged (LOIO-honest):

  zero_frac        fraction of tiles with y_true == 0 (the inflation level)
  rho_overall      Spearman(y_true, y_pred)            -- the headline number
  rho_among_pos    Spearman on y_true > 0 only          -- magnitude skill with the
                   zero mass removed; (rho_among_pos - rho_overall) = the zero drag
  ndcg@5% / ndcg   Normalized DCG (gain = y_true): ranking quality normalized by the
                   IDEAL ordering, so the label-distribution ceiling is built in (1.0
                   = perfect achievable ranking). @5% = top map tiles.
Plus cohort-level corr(zero_frac, rho_overall) (the mechanism) and a pooled
calibration table (mean_true vs mean_pred per true-abundance bin = compression).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FANG_T2 = REPO_ROOT / "models" / "fang_tier2"
FA_EDGES = [0.0, 1e-4, 1e-3, 1e-2, 1.0]
FA_LABELS = ["zero", "0-1e-4", "1e-4-1e-3", "1e-3-1e-2", "1e-2-max"]


def _spearman(a, b):
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    return float(spearmanr(a, b).statistic)


def ndcg_at_k(y_true, y_pred, k):
    k = min(k, y_true.size)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (y_true[np.argsort(-y_pred)][:k] * disc).sum()
    idcg = (np.sort(y_true)[::-1][:k] * disc).sum()
    return float(dcg / idcg) if idcg > 0 else np.nan


def per_image(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for obs, g in df.groupby("obs_id"):
        yt = g["y_true"].to_numpy(float)
        yp = g["y_pred"].to_numpy(float)
        pos = yt > 0
        k5 = max(1, int(0.05 * yt.size))
        rows.append({
            "obs_id": obs, "n": yt.size, "zero_frac": float((yt == 0).mean()),
            "n_pos": int(pos.sum()),
            "rho_overall": _spearman(yt, yp),
            "rho_among_pos": _spearman(yt[pos], yp[pos]) if pos.sum() >= 3 else np.nan,
            "ndcg_at_5pct": ndcg_at_k(yt, yp, k5),
            "ndcg_full": ndcg_at_k(yt, yp, yt.size),
        })
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    yt = df["y_true"].to_numpy(float)
    yp = df["y_pred"].to_numpy(float)
    rows = []
    for i, lab in enumerate(FA_LABELS):
        m = (yt == 0) if i == 0 else (yt > FA_EDGES[i - 1]) & (yt <= FA_EDGES[i])
        if m.sum():
            mt, mp = yt[m].mean(), yp[m].mean()
            rows.append({"bin": lab, "n": int(m.sum()), "mean_true": mt, "mean_pred": mp,
                         "pred/true": (mp / mt) if mt > 0 else np.nan})
    return pd.DataFrame(rows)


def load_cell(label: str) -> pd.DataFrame | None:
    hits = sorted(FANG_T2.glob(f"{label}/*/predictions.parquet"))
    return pd.read_parquet(hits[0]) if hits else None


def main() -> int:
    cells = sys.argv[1:] or [
        "tier2_mlp_reg_emb_fractional_area_S32", "tier2_mlp_reg_t1_fractional_area_S32",
    ]
    for label in cells:
        df = load_cell(label)
        if df is None:
            print(f"!! {label}: no predictions"); continue
        pi = per_image(df)
        zf_rho = _spearman(pi["zero_frac"].to_numpy(), pi["rho_overall"].to_numpy())
        print(f"\n=== {label}  ({pi['n'].sum()} tiles / {len(pi)} images) ===")
        print(f"  mean zero_frac     = {pi['zero_frac'].mean():.3f}")
        print(f"  rho_overall   mean = {pi['rho_overall'].mean():.4f}")
        print(f"  rho_among_pos mean = {pi['rho_among_pos'].mean():.4f}   "
              f"(zero drag = {pi['rho_among_pos'].mean() - pi['rho_overall'].mean():+.4f})")
        print(f"  NDCG@5%       mean = {pi['ndcg_at_5pct'].mean():.4f}")
        print(f"  NDCG (full)   mean = {pi['ndcg_full'].mean():.4f}")
        print(f"  corr(zero_frac, rho_overall) across images = {zf_rho:+.3f}")
        cal = calibration(df)
        print("  calibration (mean_true -> mean_pred; pred/true<1 = compression):")
        for _, r in cal.iterrows():
            print(f"    {r['bin']:>10s} n={int(r['n']):>6d}  true={r['mean_true']:.5f}  "
                  f"pred={r['mean_pred']:.5f}  pred/true={r['pred/true']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
