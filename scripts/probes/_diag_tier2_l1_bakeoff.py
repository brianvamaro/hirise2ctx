"""Tier-2 L1 bake-off (PLAN_Calibration Stage 2): do *distributional* heads
de-compress at the SOURCE where the cheap target-transform swaps could not?

The cheap L1 swaps (log1p, count-Poisson) were already ruled out
(`_diag_tier2_objectives.py`, 2026-06-15): compression is the intrinsic aleatoric
floor, not a target-scale artefact, so a mean-seeking loss in any monotone scale
stays compressed. This probe tests the heavier L1 lever — losses whose optimum is
*not* the arithmetic mean, each emitting a full per-tile predictive distribution so
we can also read a non-mean summary (a high quantile / the mode) and feed L4:

  - identity + MSE      the current mlp_reg baseline (reference)
  - HL-Gauss            histogram loss over K Gaussian-smoothed bins on a log1p
                        support; soft cross-entropy. Readouts: distribution mean,
                        mode, P90 (Imani&White 2018; Farebrother+ 2024 "Stop
                        Regressing"). The plan's top L1 candidate.
  - quantile/pinball    multi-output P10/P50/P90 (Koenker&Bassett 1978). Median is
                        a robust less-compressed point; [P10,P90] is the L4 interval.
  - ZILN                neural zero-inflated log-normal NLL (pi, mu, sigma): matches
                        the zero-spike + right-tail DGP. Readouts: mixture mean,
                        median, P90.

Same emb-only S=32 LOIO protocol as the objectives probe. Every point readout is
scored RAW and with quantile-matching on top (L3); per-image Spearman is the
must-not-regress guard. Distributional heads also report [P10,P90] coverage (L4).

CPU/GPU torch. Writes a scorecard JSON + per-tile predictions parquet under
models/fang_tier2/l1_bakeoff/ for the §2.5 report. Long-running: launch with
conda run --no-capture-output ... python -u (see [[conda_run_no_capture_output]]).
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probes"))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; precede numpy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

from src.modeling.loaders import iter_loio_folds, augment_fold_with_fang, load_fang_store
from src.calibration import compression_metrics, quantile_match, loio_calibrate
from _fm_tier2_regression import MLPRegressorEnsemble

DATASET_DIR = REPO / "dataset_v2"
OUT_DIR = REPO / "models" / "fang_tier2" / "l1_bakeoff"
BATCH = 4096
EPOCHS = 50
SEEDS = (0, 1, 2)


# ====================================================================== trunk
def _trunk(d_in, d_out):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, d_out))


class _BaseEnsemble:
    """3-seed MLP ensemble with median-impute + z-score features (matches the
    frozen mlp_ens3 preproc). Subclasses set ``head_dim`` and implement
    ``_prep_target`` (fit-time target tensor), ``_loss`` and ``_readout`` (raw net
    output -> dict of named prediction arrays, averaged across seeds)."""
    head_dim = 1

    def __init__(self, epochs=EPOCHS, batch=BATCH, lr=1e-3, wd=1e-4, patience=8):
        self.epochs, self.batch, self.lr, self.wd, self.patience = epochs, batch, lr, wd, patience
        self._nets, self._mu, self._sd, self._med = [], None, None, None

    # ---- feature scaler ----
    def _fit_feat(self, X):
        self._med = np.nanmedian(X, axis=0); self._med[~np.isfinite(self._med)] = 0.0
        Xi = self._impute(X); self._mu, self._sd = Xi.mean(0), Xi.std(0); self._sd[self._sd == 0] = 1.0
        return ((Xi - self._mu) / self._sd).astype(np.float32)

    def _impute(self, X):
        X = np.array(X, dtype=np.float32, copy=True); r, c = np.where(~np.isfinite(X)); X[r, c] = self._med[c]; return X

    def _apply(self, X):
        return ((self._impute(X) - self._mu) / self._sd).astype(np.float32)

    # ---- hooks ----
    def _setup(self, y_train):
        """Per-fold target setup (e.g. bin edges). Default: nothing."""

    def _prep_target(self, y):
        """y (np) -> torch target tensor used by _loss (rows aligned to X)."""
        raise NotImplementedError

    def _loss(self, out, tgt):
        raise NotImplementedError

    def _readout(self, raw):
        """raw: averaged net output [n, head_dim] (np). -> {name: array}."""
        raise NotImplementedError

    def fit(self, X, y, eval_set=None):
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup(np.asarray(y, np.float64))
        Xs = self._fit_feat(np.asarray(X, np.float32))
        Xt = torch.from_numpy(Xs).to(dev)
        Tt = self._prep_target(np.asarray(y, np.float64)).to(dev)
        Xv = Tv = None
        if eval_set is not None:
            Xv = torch.from_numpy(self._apply(np.asarray(eval_set[0], np.float32))).to(dev)
            Tv = self._prep_target(np.asarray(eval_set[1], np.float64)).to(dev)
        n = Xt.shape[0]
        self._nets = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            net = _trunk(Xs.shape[1], self.head_dim).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            best, best_state, bad = float("inf"), None, 0
            for _ in range(self.epochs):
                net.train(); perm = torch.randperm(n, device=dev)
                for i in range(0, n, self.batch):
                    idx = perm[i:i + self.batch]
                    opt.zero_grad(set_to_none=True)
                    self._loss(net(Xt[idx]), Tt[idx]).backward(); opt.step()
                if Xv is not None:
                    net.eval()
                    with torch.no_grad():
                        vl = float(self._loss(net(Xv), Tv).item())
                    if vl < best - 1e-7: best, bad, best_state = vl, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= self.patience: break
            if best_state: net.load_state_dict(best_state)
            self._nets.append(net)

    def _raw(self, X):
        import torch
        dev = next(self._nets[0].parameters()).device
        Xs = self._apply(np.asarray(X, np.float32))
        acc = np.zeros((len(Xs), self.head_dim), np.float64)
        for net in self._nets:
            net.eval()
            with torch.no_grad():
                for i in range(0, len(Xs), 4096):
                    acc[i:i + 4096] += net(torch.from_numpy(Xs[i:i + 4096]).to(dev)).cpu().numpy()
        return acc / len(self._nets)

    def predict(self, X):
        return self._readout(self._raw(X))


# =================================================================== HL-Gauss
class HLGaussEnsemble(_BaseEnsemble):
    """Histogram loss: K bins on a log1p support, Gaussian-smoothed soft labels,
    soft cross-entropy. Readouts: distribution mean E[y], mode, P90 (all y-space)."""
    name = "hlgauss"

    def __init__(self, n_bins=64, smooth=1.0, **kw):
        super().__init__(**kw)
        self.head_dim = n_bins
        self.n_bins, self.smooth = n_bins, smooth
        self._edges = self._centers_t = self._centers_y = None

    @staticmethod
    def _t(y):  # support transform (log1p) — resolution at the low end
        return np.log1p(np.clip(y, 0, None))

    @staticmethod
    def _inv(t):
        return np.expm1(t)

    def _setup(self, y):
        tmax = float(np.quantile(self._t(y), 0.999)) or 1.0
        self._edges = np.linspace(0.0, tmax, self.n_bins + 1)
        self._centers_t = 0.5 * (self._edges[:-1] + self._edges[1:])
        self._centers_y = self._inv(self._centers_t)
        self._sigma = self.smooth * (self._edges[1] - self._edges[0])

    def _prep_target(self, y):
        import torch
        t = np.clip(self._t(y), self._edges[0], self._edges[-1])
        # HL-Gauss: bin mass = Gaussian CDF difference over edges, renormalized
        z = (self._edges[None, :] - t[:, None]) / self._sigma
        cdf = norm.cdf(z)
        soft = np.diff(cdf, axis=1)
        soft = soft / np.clip(soft.sum(1, keepdims=True), 1e-12, None)
        return torch.from_numpy(soft.astype(np.float32))

    def _loss(self, out, tgt):
        import torch.nn.functional as F
        return -(tgt * F.log_softmax(out, dim=1)).sum(1).mean()

    def _readout(self, raw):
        import scipy.special as sp
        q = sp.softmax(raw, axis=1)                       # [n, K]
        cy = self._centers_y
        mean = (q * cy[None, :]).sum(1)
        mode = cy[np.argmax(q, axis=1)]
        cum = np.cumsum(q, axis=1)
        p90 = cy[np.clip(np.argmax(cum >= 0.9, axis=1), 0, self.n_bins - 1)]
        return {"mean": np.clip(mean, 0, 1), "mode": np.clip(mode, 0, 1),
                "p90": np.clip(p90, 0, 1)}


# =============================================================== pinball/quantile
class PinballEnsemble(_BaseEnsemble):
    """Multi-output quantile regression (P10/P50/P90) with the pinball loss, in
    y-space. Readouts: median (point), P90, and the [P10,P90] interval (L4)."""
    name = "pinball"
    head_dim = 3
    TAUS = (0.1, 0.5, 0.9)

    def _prep_target(self, y):
        import torch
        return torch.from_numpy(np.clip(y, 0, None).astype(np.float32))

    def _loss(self, out, tgt):
        import torch
        taus = torch.tensor(self.TAUS, device=out.device, dtype=out.dtype)
        err = tgt[:, None] - out                          # [b, 3]
        return torch.maximum(taus * err, (taus - 1.0) * err).mean()

    def _readout(self, raw):
        q = np.sort(np.clip(raw, 0, 1), axis=1)           # enforce non-crossing
        return {"median": q[:, 1], "p90": q[:, 2], "_p10": q[:, 0], "_p90i": q[:, 2]}


# ====================================================================== ZILN
class ZILNEnsemble(_BaseEnsemble):
    """Neural zero-inflated log-normal NLL: head = (logit_pi, mu, log_sigma).
    p(y)=pi at y=0, else (1-pi)*LogNormal(mu,sigma). Readouts: mixture mean
    (point), median, P90, and the [P10,P90] interval (L4)."""
    name = "ziln"
    head_dim = 3

    def _prep_target(self, y):
        import torch
        return torch.from_numpy(np.clip(y, 0, None).astype(np.float32))

    def _loss(self, out, tgt):
        import torch
        pi = torch.clamp(torch.sigmoid(out[:, 0]), 1e-6, 1 - 1e-6)
        mu = out[:, 1]
        sigma = torch.clamp(torch.exp(out[:, 2]), 1e-3, 10.0)
        zero = tgt <= 0
        ly = torch.log(torch.clamp(tgt, min=1e-6))
        nll_pos = (-torch.log(1 - pi) + torch.log(sigma) + ly
                   + 0.5 * np.log(2 * np.pi) + 0.5 * ((ly - mu) / sigma) ** 2)
        nll = torch.where(zero, -torch.log(pi), nll_pos)
        return nll.mean()

    @staticmethod
    def _mix_quantile(pi, mu, sigma, p):
        """p-quantile of the zero-inflated log-normal mixture (vectorized)."""
        out = np.zeros_like(mu)
        upper = (p - pi) / np.clip(1 - pi, 1e-6, None)
        pos = upper > 0
        out[pos] = np.exp(mu[pos] + sigma[pos] * norm.ppf(np.clip(upper[pos], 1e-6, 1 - 1e-6)))
        return out

    def _readout(self, raw):
        pi = 1.0 / (1.0 + np.exp(-raw[:, 0])); pi = np.clip(pi, 1e-6, 1 - 1e-6)
        mu = raw[:, 1]; sigma = np.clip(np.exp(raw[:, 2]), 1e-3, 10.0)
        mean = (1 - pi) * np.exp(mu + 0.5 * sigma ** 2)
        median = self._mix_quantile(pi, mu, sigma, 0.5)
        p10 = self._mix_quantile(pi, mu, sigma, 0.1)
        p90 = self._mix_quantile(pi, mu, sigma, 0.9)
        return {"mean": np.clip(mean, 0, 1), "median": np.clip(median, 0, 1),
                "p90": np.clip(p90, 0, 1), "_p10": np.clip(p10, 0, 1),
                "_p90i": np.clip(p90, 0, 1)}


# ================================================================== harness
def inner_val(fold, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(len(fold.X_train)) < frac


def run_head(folds, make_model, base_y="fractional_area", point_name="point"):
    """Fit per fold, collect held-out readouts into one df. Heads return a dict of
    named arrays; the array-returning baseline is wrapped as {point_name: array}."""
    parts = []
    for f in folds:
        vm = inner_val(f)
        mdl = make_model()
        yv = f.y_train[base_y].to_numpy()
        mdl.fit(f.X_train[~vm], yv[~vm], eval_set=(f.X_train[vm], yv[vm]))
        pred = mdl.predict(f.X_test)
        if not isinstance(pred, dict):
            pred = {point_name: np.asarray(pred)}
        d = f.keys_test[["obs_id", "ti", "tj"]].copy()
        d["y_true"] = f.y_test["fractional_area"].to_numpy()
        for k, v in pred.items():
            d[k] = np.asarray(v)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def per_image_spearman(df, col):
    pim = df.groupby("obs_id").apply(
        lambda g: spearmanr(g.y_true, g[col]).correlation if g.y_true.nunique() > 1 else np.nan)
    return float(np.nanmedian(pim))


def score_point(df, col, tag, rows):
    yt = df["y_true"].to_numpy(); yp = df[col].to_numpy()
    m = compression_metrics(yt, yp)
    cal = loio_calibrate(df.assign(y_pred=yp), lambda rp, rt, hp: quantile_match(hp, rp, rt))
    mc = compression_metrics(yt, cal)
    pim = per_image_spearman(df, col)
    print(f"{tag:>22} | raw rho {m['spearman']:.3f} (img {pim:.3f}) top {m['top_ratio']:.2f} "
          f"near0 {m['near_zero_pred']:.1%} L1 {m['marginal_l1']:.4f}"
          f" | +qm rho {mc['spearman']:.3f} top {mc['top_ratio']:.2f} "
          f"near0 {mc['near_zero_pred']:.1%} L1 {mc['marginal_l1']:.4f}", flush=True)
    rows.append({"readout": tag, "per_img_rho": pim, **{f"raw_{k}": v for k, v in m.items()},
                 **{f"qm_{k}": v for k, v in mc.items()}})


def coverage(df, tag, rows):
    inside = ((df["y_true"] >= df["_p10"]) & (df["y_true"] <= df["_p90i"])).mean()
    width = (df["_p90i"] - df["_p10"]).mean()
    print(f"{tag:>22} | [P10,P90] coverage {inside:.1%} (nominal 80%)  mean width {width:.4f}", flush=True)
    rows.append({"readout": tag, "coverage_p10_p90": float(inside), "mean_width": float(width)})


def main():
    print("Building emb-only S=32 folds ...", flush=True)
    store = load_fang_store(96, pool="gem", dataset_dir=DATASET_DIR)
    folds = [augment_fold_with_fang(f, px=96, dataset_dir=DATASET_DIR, replace=True, store=store)
             for f in iter_loio_folds("loio_nfold", scale_idx=2, dataset_dir=DATASET_DIR)]
    nz = np.mean(np.concatenate([f.y_test["fractional_area"].to_numpy() for f in folds]) <= 0)
    print(f"  {len(folds)} folds; true near-zero share {nz:.1%}\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorecard, cover_rows = [], []

    print("=== baseline: identity + MSE (mlp_reg) ===", flush=True)
    t0 = time.monotonic()
    base = run_head(folds, lambda: MLPRegressorEnsemble(transform="identity", clip_max=1.0, batch=BATCH))
    print(f"  ({time.monotonic()-t0:.0f}s)", flush=True)

    heads = {
        "hlgauss": lambda: HLGaussEnsemble(),
        "pinball": lambda: PinballEnsemble(),
        "ziln": lambda: ZILNEnsemble(),
    }
    head_dfs = {}
    for name, mk in heads.items():
        print(f"\n=== head: {name} ===", flush=True)
        t0 = time.monotonic()
        head_dfs[name] = run_head(folds, mk)
        print(f"  ({time.monotonic()-t0:.0f}s)", flush=True)

    print("\n----------------------------- SCORECARD (point readouts) -----------------------------", flush=True)
    score_point(base, "point", "mlp_reg(mean)", scorecard)
    for name, df in head_dfs.items():
        for col in [c for c in df.columns if c not in ("obs_id", "ti", "tj", "y_true") and not c.startswith("_")]:
            score_point(df, col, f"{name}.{col}", scorecard)

    print("\n----------------------------- INTERVAL COVERAGE (L4) ---------------------------------", flush=True)
    for name, df in head_dfs.items():
        if "_p10" in df.columns:
            coverage(df, f"{name}[P10,P90]", cover_rows)

    # persist
    pd.DataFrame(scorecard).to_csv(OUT_DIR / "scorecard.csv", index=False)
    (OUT_DIR / "coverage.json").write_text(json.dumps(cover_rows, indent=2), encoding="utf-8")
    base[["obs_id", "ti", "tj", "y_true", "point"]].to_parquet(OUT_DIR / "preds_mlp_reg.parquet")
    for name, df in head_dfs.items():
        df.to_parquet(OUT_DIR / f"preds_{name}.parquet")
    print(f"\nwrote scorecard.csv + per-tile parquets -> {OUT_DIR.relative_to(REPO)}", flush=True)
    print("\ngoal: a readout with top_ratio->1 and near0->~truth WITHOUT dropping per-img rho "
          f"(baseline mlp_reg per-img rho above); coverage near 80%.", flush=True)


if __name__ == "__main__":
    main()
