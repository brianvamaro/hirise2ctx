"""Tier-2 L1 objective probe (PLAN_Calibration L1): does a less-mean-seeking loss
de-compress at the SOURCE? Same emb-only S=32 LOIO protocol for all three:

  - identity + MSE   (the current mlp_reg baseline)
  - log1p   + MSE    (targets ~the conditional median -> less tail-shy)
  - count   + Poisson NLL (the natural count model; predicted count -> area fraction)

Each is scored raw AND with quantile-matching on top (L3), so we see both the
source de-compression and the full L1->L3 pipeline. Ranking (Spearman) is the
must-not-regress constraint. CPU/GPU torch; ~10-15 min.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probes"))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; precede numpy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.modeling.loaders import iter_loio_folds, augment_fold_with_fang, load_fang_store
from src.calibration import compression_metrics, quantile_match, loio_calibrate
from _fm_tier2_regression import MLPRegressorEnsemble  # reuse for identity/log1p

DATASET_DIR = REPO / "dataset_v2"
BATCH = 4096


# ---------------------------------------------------------------- Poisson head
class PoissonMLPEnsemble:
    """3-seed MLP predicting a Poisson log-rate (loss = exp(z) - y*z); predict=exp(z).
    Same 768-256-64-1 arch + median-impute/z-score features as MLPRegressorEnsemble."""
    name = "poisson_ens3"

    def __init__(self, seeds=(0, 1, 2), epochs=60, batch=BATCH, lr=1e-3, wd=1e-4, patience=8):
        self.seeds, self.epochs, self.batch = seeds, epochs, batch
        self.lr, self.wd, self.patience = lr, wd, patience
        self._nets, self._mu, self._sd, self._med = [], None, None, None

    def _fit_feat(self, X):
        self._med = np.nanmedian(X, axis=0); self._med[~np.isfinite(self._med)] = 0.0
        Xi = self._impute(X); self._mu, self._sd = Xi.mean(0), Xi.std(0); self._sd[self._sd == 0] = 1.0
        return ((Xi - self._mu) / self._sd).astype(np.float32)

    def _impute(self, X):
        X = np.array(X, dtype=np.float32, copy=True); r, c = np.where(~np.isfinite(X)); X[r, c] = self._med[c]; return X

    def _apply(self, X):
        return ((self._impute(X) - self._mu) / self._sd).astype(np.float32)

    def fit(self, X, y, eval_set=None):
        import torch, torch.nn as nn
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xs = self._fit_feat(np.asarray(X, np.float32))
        yt = torch.from_numpy(np.asarray(y, np.float32))
        Xt = torch.from_numpy(Xs).to(dev); yt = yt.to(dev)
        Xv = yv = None
        if eval_set is not None:
            Xv = torch.from_numpy(self._apply(np.asarray(eval_set[0], np.float32))).to(dev)
            yv = torch.from_numpy(np.asarray(eval_set[1], np.float32)).to(dev)

        def ploss(z, y):
            return (torch.exp(z) - y * z).mean()

        self._nets = []
        for seed in self.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            net = nn.Sequential(nn.Linear(Xs.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            best, best_state, bad = float("inf"), None, 0
            for _ in range(self.epochs):
                net.train(); perm = torch.randperm(Xt.shape[0], device=dev)
                for i in range(0, Xt.shape[0], self.batch):
                    idx = perm[i:i + self.batch]
                    opt.zero_grad(set_to_none=True)
                    ploss(net(Xt[idx]).squeeze(-1), yt[idx]).backward(); opt.step()
                if Xv is not None:
                    net.eval()
                    with torch.no_grad():
                        vl = float(ploss(net(Xv).squeeze(-1), yv).item())
                    if vl < best - 1e-7: best, bad, best_state = vl, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= self.patience: break
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
                    acc[i:i + 4096] += np.exp(net(torch.from_numpy(Xs[i:i + 4096]).to(dev)).squeeze(-1).cpu().numpy())
        return acc / len(self._nets)


def inner_val(fold, frac=0.1, seed=0):
    rng = np.random.default_rng(seed); n = len(fold.X_train)
    m = rng.random(n) < frac
    return m


def run(folds, fit_predict):
    """fit_predict(f, val_mask) -> (held_out y_pred in fa units). Collect over folds."""
    rows = []
    for f in folds:
        vm = inner_val(f)
        yp = fit_predict(f, vm)
        d = f.keys_test[["obs_id", "ti", "tj"]].copy()
        d["y_true"] = f.y_test["fractional_area"].to_numpy()
        d["y_pred"] = np.clip(yp, 0, None)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def score(df, tag):
    m = compression_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
    cal = loio_calibrate(df, lambda rp, rt, hp: quantile_match(hp, rp, rt))
    mc = compression_metrics(df["y_true"].to_numpy(), cal)
    pim = df.groupby("obs_id").apply(lambda g: spearmanr(g.y_true, g.y_pred).correlation if g.y_true.nunique() > 1 else np.nan)
    print(f"{tag:>16} | raw: rho {m['spearman']:.3f} (per-img med {np.nanmedian(pim):.3f}) "
          f"top {m['top_ratio']:.2f} near0 {m['near_zero_pred']:.1%} L1 {m['marginal_l1']:.4f}"
          f" | +qmatch: rho {mc['spearman']:.3f} top {mc['top_ratio']:.2f} "
          f"near0 {mc['near_zero_pred']:.1%} L1 {mc['marginal_l1']:.4f}", flush=True)


def main():
    print("Building emb-only S=32 folds ...", flush=True)
    store = load_fang_store(96, pool="gem", dataset_dir=DATASET_DIR)
    folds = [augment_fold_with_fang(f, px=96, dataset_dir=DATASET_DIR, replace=True, store=store)
             for f in iter_loio_folds("loio_nfold", scale_idx=2, dataset_dir=DATASET_DIR)]
    print(f"  {len(folds)} folds; true near-zero share "
          f"{np.mean(np.concatenate([f.y_test['fractional_area'].to_numpy() for f in folds]) <= 0):.1%}", flush=True)

    # count->area conversion factor (global mean individual boulder area / tile area)
    ba = np.concatenate([f.y_train['boulder_area'].to_numpy() for f in folds[:1]])  # any fold's train ~ all
    bc = np.concatenate([f.y_train['boulder_count'].to_numpy() for f in folds[:1]])
    ta = float(folds[0].y_train['tile_area'].iloc[0])
    mean_indiv = ba.sum() / max(bc.sum(), 1)
    print(f"  count->fa: mean indiv boulder area {mean_indiv:.2f} m2 / tile area {ta:.0f} m2", flush=True)

    def fp_mlp(transform):
        def _f(f, vm):
            mdl = MLPRegressorEnsemble(transform=transform, clip_max=1.0, batch=BATCH)
            yv = f.y_train["fractional_area"].to_numpy()
            mdl.fit(f.X_train[~vm], yv[~vm], eval_set=(f.X_train[vm], yv[vm]))
            return mdl.predict(f.X_test)
        return _f

    def fp_poisson(f, vm):
        mdl = PoissonMLPEnsemble(batch=BATCH)
        yc = f.y_train["boulder_count"].to_numpy()
        mdl.fit(f.X_train[~vm], yc[~vm], eval_set=(f.X_train[vm], yc[vm]))
        return mdl.predict(f.X_test) * mean_indiv / ta   # count -> area fraction

    print("\n--- running (same protocol, batch 4096) ---", flush=True)
    score(run(folds, fp_mlp("identity")), "identity+MSE")
    score(run(folds, fp_mlp("log1p")), "log1p+MSE")
    score(run(folds, fp_poisson), "count+Poisson")
    print("\ngoal: top_ratio->1, near0->~truth, L1->0, WITHOUT dropping per-img rho.")


if __name__ == "__main__":
    main()
