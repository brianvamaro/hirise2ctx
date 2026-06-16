"""Tier-2 Stage 2c — imbalanced-regression reweighting (PLAN_Calibration L1+L2):
does up-weighting the rare high-abundance tail (so the 18% zeros + bulk don't drown
it) de-compress at the source or lift ranking?

LDS (Label Distribution Smoothing, Yang et al. 2021 https://arxiv.org/abs/2102.09554):
bin the target in log1p space, Gaussian-smooth the empirical label density, weight each
sample by 1/density (or 1/sqrt) so the loss stops being dominated by the low end. Same
emb-only S=32 LOIO mlp_reg as every other Stage-2 probe; each scheme scored raw AND
+quantile-match, paired per-image Wilcoxon vs the unweighted baseline as the guard.

Schemes:
  none        unweighted MSE (the mlp_reg baseline)
  lds_sqrt    w ∝ 1/sqrt(smoothed density)   (mild)
  lds_inv     w ∝ 1/(smoothed density)       (aggressive)

GPU torch; ~15 min/scheme. Writes scorecard to models/fang_tier2/l1_bakeoff/.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probes"))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; precede numpy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from scipy.ndimage import gaussian_filter1d

from src.modeling.loaders import iter_loio_folds, augment_fold_with_fang, load_fang_store
from src.calibration import compression_metrics, quantile_match, loio_calibrate

DATASET_DIR = REPO / "dataset_v2"
OUT = REPO / "models" / "fang_tier2" / "l1_bakeoff"
BATCH = 4096
EPOCHS = 50
SEEDS = (0, 1, 2)


def lds_weights(y, scheme, n_bins=50, sigma=2.0):
    """LDS sample weights: 1/(Gaussian-smoothed log1p-density), normalized to mean 1."""
    if scheme == "none":
        return np.ones(len(y), np.float32)
    t = np.log1p(np.clip(y, 0, None))
    edges = np.linspace(0.0, float(t.max()) + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    dens = gaussian_filter1d(np.bincount(idx, minlength=n_bins).astype(np.float64), sigma)
    dens = dens / dens.sum()
    p = np.clip(dens[idx], 1e-6, None)
    w = 1.0 / np.sqrt(p) if scheme == "lds_sqrt" else 1.0 / p
    w = w / w.mean()                       # mean-1 scale
    return np.clip(w, 0.1, 20.0).astype(np.float32)   # bound final weights (no re-inflate)


class WeightedMLPEnsemble:
    """3-seed 768-256-64-1 MLP, weighted MSE on the standardized identity target.
    Mirrors mlp_reg's preproc; only the per-sample loss weight differs."""
    def __init__(self, scheme="none"):
        self.scheme = scheme
        self._nets, self._mu, self._sd, self._med = [], None, None, None
        self._ty_mu = self._ty_sd = None

    def _fit_feat(self, X):
        self._med = np.nanmedian(X, axis=0); self._med[~np.isfinite(self._med)] = 0.0
        Xi = self._impute(X); self._mu, self._sd = Xi.mean(0), Xi.std(0); self._sd[self._sd == 0] = 1.0
        return ((Xi - self._mu) / self._sd).astype(np.float32)

    def _impute(self, X):
        X = np.array(X, np.float32, copy=True); r, c = np.where(~np.isfinite(X)); X[r, c] = self._med[c]; return X

    def _apply(self, X):
        return ((self._impute(X) - self._mu) / self._sd).astype(np.float32)

    def fit(self, X, y, eval_set=None):
        import torch, torch.nn as nn
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xs = self._fit_feat(np.asarray(X, np.float32))
        y = np.asarray(y, np.float64)
        self._ty_mu, self._ty_sd = float(y.mean()), float(y.std() or 1.0)
        ys = ((y - self._ty_mu) / self._ty_sd).astype(np.float32)
        w = lds_weights(y, self.scheme)
        Xt = torch.from_numpy(Xs).to(dev); yt = torch.from_numpy(ys).to(dev)
        wt = torch.from_numpy(w).to(dev)
        Xv = yv = None
        if eval_set is not None:
            Xv = torch.from_numpy(self._apply(np.asarray(eval_set[0], np.float32))).to(dev)
            yv = torch.from_numpy(((np.asarray(eval_set[1], np.float64) - self._ty_mu) / self._ty_sd).astype(np.float32)).to(dev)
        n = Xt.shape[0]

        def wmse(pred, tgt, wts):
            return (wts * (pred - tgt) ** 2).mean()

        self._nets = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            net = nn.Sequential(nn.Linear(Xs.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_state, bad = float("inf"), None, 0
            for _ in range(EPOCHS):
                net.train(); perm = torch.randperm(n, device=dev)
                for i in range(0, n, BATCH):
                    idx = perm[i:i + BATCH]
                    opt.zero_grad(set_to_none=True)
                    wmse(net(Xt[idx]).squeeze(-1), yt[idx], wt[idx]).backward(); opt.step()
                if Xv is not None:
                    net.eval()
                    with torch.no_grad():       # early-stop on UNWEIGHTED val MSE (fair across schemes)
                        vl = float(((net(Xv).squeeze(-1) - yv) ** 2).mean().item())
                    if vl < best - 1e-7: best, bad, best_state = vl, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= 8: break
            if best_state: net.load_state_dict(best_state)
            self._nets.append(net)

    def predict(self, X):
        import torch
        dev = next(self._nets[0].parameters()).device
        Xs = self._apply(np.asarray(X, np.float32)); acc = np.zeros(len(Xs))
        for net in self._nets:
            net.eval()
            with torch.no_grad():
                for i in range(0, len(Xs), 4096):
                    acc[i:i + 4096] += net(torch.from_numpy(Xs[i:i + 4096]).to(dev)).squeeze(-1).cpu().numpy()
        return np.clip(acc / len(self._nets) * self._ty_sd + self._ty_mu, 0.0, 1.0)


def inner_val(fold, frac=0.1, seed=0):
    return np.random.default_rng(seed).random(len(fold.X_train)) < frac


def run(folds, scheme):
    parts = []
    for f in folds:
        vm = inner_val(f)
        mdl = WeightedMLPEnsemble(scheme=scheme)
        yv = f.y_train["fractional_area"].to_numpy()
        mdl.fit(f.X_train[~vm], yv[~vm], eval_set=(f.X_train[vm], yv[vm]))
        d = f.keys_test[["obs_id", "ti", "tj"]].copy()
        d["y_true"] = f.y_test["fractional_area"].to_numpy()
        d["y_pred"] = mdl.predict(f.X_test)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def per_img(df, col="y_pred"):
    return {o: spearmanr(g.y_true, g[col]).correlation for o, g in df.groupby("obs_id")
            if g.y_true.nunique() > 1}


def main():
    print("Building emb-only S=32 folds ...", flush=True)
    store = load_fang_store(96, pool="gem", dataset_dir=DATASET_DIR)
    folds = [augment_fold_with_fang(f, px=96, dataset_dir=DATASET_DIR, replace=True, store=store)
             for f in iter_loio_folds("loio_nfold", scale_idx=2, dataset_dir=DATASET_DIR)]
    print(f"  {len(folds)} folds\n", flush=True)

    base_rho = None; rows = []
    for scheme in ["none", "lds_sqrt", "lds_inv"]:
        t0 = time.monotonic()
        df = run(folds, scheme)
        m = compression_metrics(df.y_true.to_numpy(), df.y_pred.to_numpy())
        cal = loio_calibrate(df, lambda rp, rt, hp: quantile_match(hp, rp, rt))
        mc = compression_metrics(df.y_true.to_numpy(), cal)
        rho = per_img(df)
        if scheme == "none":
            base_rho = rho
            ptxt = "(baseline)"
        else:
            keys = [k for k in base_rho if k in rho and np.isfinite(base_rho[k]) and np.isfinite(rho[k])]
            a = np.array([rho[k] for k in keys]); b = np.array([base_rho[k] for k in keys])
            ptxt = f"paired d={np.median(a-b):+.3f} wins {int((a>b).sum())}/{len(keys)} p={wilcoxon(a,b).pvalue:.3f}"
        print(f"{scheme:>9} [{time.monotonic()-t0:.0f}s] | raw rho {m['spearman']:.3f} "
              f"(img {np.nanmedian(list(rho.values())):.3f}) top {m['top_ratio']:.2f} near0 {m['near_zero_pred']:.1%}"
              f" | +qm rho {mc['spearman']:.3f} top {mc['top_ratio']:.2f} | {ptxt}", flush=True)
        rows.append({"scheme": scheme, "raw_top": m["top_ratio"], "raw_perimg_rho": float(np.nanmedian(list(rho.values()))),
                     "raw_pooled_rho": m["spearman"], "qm_top": mc["top_ratio"], "near0": m["near_zero_pred"]})
        df.to_parquet(OUT / f"preds_reweight_{scheme}.parquet")
    pd.DataFrame(rows).to_csv(OUT / "reweight_scorecard.csv", index=False)
    print(f"\nwrote reweight_scorecard.csv -> {OUT.relative_to(REPO)}")
    print("read: does up-weighting the tail raise raw top_ratio (less compression) or per-img rho "
          "WITHOUT a paired ranking loss? (qmatch already fixes the marginal.)")


if __name__ == "__main__":
    main()
