"""PLAN_FM §2.4 Tier-2: calibrated-abundance REGRESSION on the frozen emb-only
features. Sibling of the frozen binary Tier-1 recipe (mlp_ens3 / GeM / emb-only /
S=32) — predict *how much* abundance per tile, not just rich/poor.

Central question (PLAN_FM 4 + [[modeling_single_stage_future]]): does the
two-stage hurdle still earn its complexity now that the features are much
stronger, or is a single-stage regressor finally sufficient? The known trap is
dynamic-range COMPRESSION (notebook 12, 2026-05-29): regression on a
zero-inflated/right-skewed target hedges to the mean and flattens the
high-abundance tail. The per-bin RMSE + calibration tables surface it.

Heads (all on emb-only S=32, LOIO over the 38 v2 images):
    lightgbm_tweedie               single-stage Tweedie (zero-inflation-aware)
    lightgbm_two_stage_balanced    the incumbent hurdle (presence -> magnitude)
    mlp_reg                        3-seed MLP regressor (the frozen mlp_ens3's
                                   regression analog) — single-stage, NEW here

Targets (run BOTH when launched — Brian, 2026-06-12): `fractional_area` (physical
abundance, continuity with the frozen Tier-1) and `boulder_count` (log1p — the
count sibling; 1b showed the FM advantage transfers to count). Each emb cell is
reported against the matching Tier-1 handcrafted-feature baseline (same head,
same harness) so the lift is the FM contribution to *regression*, not just
classification.

STATUS 2026-06-12: built, NOT yet run (Brian: design now, compute later).
Run commands at the bottom of this docstring.

Usage:
    conda run --no-capture-output -n geospatial python -u \
        scripts/probes/_fm_tier2_regression.py --variant mlp_reg \
        --target fractional_area --features emb
    # baseline: --features t1   (handcrafted, no embeddings)
    # both targets: loop --target over {fractional_area, boulder_count}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np

from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, make_factory, snapshot_params
from src.modeling.loaders import augment_fold_with_fang, iter_loio_folds, load_fang_store

DATASET_DIR = REPO_ROOT / "dataset_v2"
OUT_ROOT = REPO_ROOT / "models" / "fang_tier2"
SCHEME = "loio_nfold"
TILE_PX = 32           # frozen operating scale
SCALE_IDX = 2          # S=32
CTX_PX = 3 * TILE_PX   # 96-px 3x3-context input = the frozen embedding

# target name -> (y dataframe column, MLP target transform, rich/poor meaningful
# threshold for the ranking metrics). The count threshold is a REAL rich/poor cut
# (bc_ge_50, the 1b read) -- NOT presence: run_loio's default 1e-2 applied to raw
# counts means count > 0.01 == count >= 1 == presence (degenerate; the bc_ge_1
# trap). per_bin_rmse still uses fractional_area bin edges, so for the count target
# read Spearman + meaningful_auc(@50), not the per-bin table.
TARGETS = {
    "fractional_area": ("fractional_area", "identity", 1e-2),
    "boulder_count": ("boulder_count", "log1p", 50.0),
}


# ============================================================================
# MLP regressor (the frozen mlp_ens3's regression analog) — Model protocol
# ============================================================================


def _transform(y: np.ndarray, kind: str) -> np.ndarray:
    return np.log1p(np.clip(y, 0.0, None)) if kind == "log1p" else y


def _inverse(y: np.ndarray, kind: str) -> np.ndarray:
    return np.expm1(y) if kind == "log1p" else y


class MLPRegressorEnsemble:
    """3-seed MLP regressor on the frozen embedding matrix.

    768-256-64-1, dropout 0.2, linear output, MSE on the (transformed,
    standardized) target, AdamW + early stop on the rotated inner-val image —
    the regression mirror of the bake-off classifier. The feature scaler and the
    target (transform + standardize) are fit on the training fold only; `predict`
    back-transforms to the original target scale (Model protocol contract) and
    clips to the valid range. Predictions are the mean across seeds in original
    space (the deterministic-promotable form, per the 3-seed rule).

    PERF NOTE (2026-06-12): at S=32 (~147k train rows/fold x 3 seeds x 38 folds)
    these cells take ~15 min each and run the GPU at only ~15% util -- the tiny
    net is OVERHEAD-bound (per-batch CPU->GPU copy + kernel launch dominate the
    trivial matmul), not compute-bound. NEXT TIME, ~3-5x faster with no material
    effect on the numbers: raise `batch` to 4096 (8x fewer steps) and move the
    full Xt/yt to the device ONCE before the epoch loop (147k x 768 f32 ~= 450 MB,
    fits the 8 GB card) instead of copying each minibatch. LightGBM avoids this
    entirely (multithreaded C++ over the whole matrix) -- hence its cells fly.
    """

    name = "mlp_reg_ens3"

    def __init__(self, transform: str = "identity", clip_max: float | None = None,
                 seeds: tuple[int, ...] = (0, 1, 2), epochs: int = 60,
                 batch: int = 512, lr: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 8) -> None:
        self.transform = transform
        self.clip_max = clip_max
        self.seeds = seeds
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.weight_decay, self.patience = weight_decay, patience
        self._nets: list = []
        self._mu = self._sd = self._med = None          # feature scaler
        self._ty_mu = self._ty_sd = None                # target scaler (transformed space)

    # ---- feature standardization (median-impute then z-score; fit on train) ----
    def _fit_feat(self, X):
        self._med = np.nanmedian(X, axis=0)
        self._med[~np.isfinite(self._med)] = 0.0
        Xi = self._impute(X)
        self._mu, self._sd = Xi.mean(0), Xi.std(0)
        self._sd[self._sd == 0] = 1.0
        return (Xi - self._mu) / self._sd

    def _impute(self, X):
        X = np.array(X, dtype=np.float32, copy=True)
        r, c = np.where(~np.isfinite(X))
        X[r, c] = self._med[c]
        return X

    def _apply_feat(self, X):
        return ((self._impute(X) - self._mu) / self._sd).astype(np.float32)

    def _build(self, d_in: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def fit(self, X, y, *, groups=None, eval_set=None):
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xs = self._fit_feat(np.asarray(X, dtype=np.float32))
        ty = _transform(np.asarray(y, dtype=np.float64), self.transform)
        self._ty_mu, self._ty_sd = float(ty.mean()), float(ty.std() or 1.0)
        ys = ((ty - self._ty_mu) / self._ty_sd).astype(np.float32)

        Xv = yv = None
        if eval_set is not None:
            Xv_raw, yv_raw = eval_set
            Xv = torch.from_numpy(self._apply_feat(np.asarray(Xv_raw, dtype=np.float32))).to(device)
            tyv = _transform(np.asarray(yv_raw, dtype=np.float64), self.transform)
            yv = torch.from_numpy(((tyv - self._ty_mu) / self._ty_sd).astype(np.float32)).to(device)

        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(ys)
        n = Xt.shape[0]
        loss_fn = nn.MSELoss()
        self._nets = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = self._build(Xs.shape[1]).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            best, best_state, bad = float("inf"), None, 0
            for _epoch in range(self.epochs):
                net.train()
                perm = torch.randperm(n)
                for i in range(0, n, self.batch):
                    idx = perm[i: i + self.batch]
                    xb = Xt[idx].to(device, non_blocking=True)
                    yb = yt[idx].to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(net(xb).squeeze(-1), yb)
                    loss.backward()
                    opt.step()
                if Xv is not None:
                    net.eval()
                    with torch.no_grad():
                        vl = float(loss_fn(net(Xv).squeeze(-1), yv).item())
                    if vl < best - 1e-7:
                        best, bad = vl, 0
                        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= self.patience:
                            break
            if best_state is not None:
                net.load_state_dict(best_state)
            self._nets.append(net)

    def predict(self, X):
        import torch

        device = next(self._nets[0].parameters()).device
        Xs = self._apply_feat(np.asarray(X, dtype=np.float32))
        preds = np.zeros(Xs.shape[0], dtype=np.float64)
        for net in self._nets:
            net.eval()
            out = np.empty(Xs.shape[0], dtype=np.float64)
            with torch.no_grad():
                for i in range(0, Xs.shape[0], 4096):
                    xb = torch.from_numpy(Xs[i: i + 4096]).to(device)
                    out[i: i + 4096] = net(xb).squeeze(-1).cpu().numpy()
            # un-standardize -> inverse-transform -> original target space
            preds += _inverse(out * self._ty_sd + self._ty_mu, self.transform)
        preds /= len(self._nets)
        preds = np.clip(preds, 0.0, self.clip_max)
        return preds

    def predict_presence_prob(self, X):
        return None  # single-stage

    def save(self, path):  # probe-tier: no persistence
        pass

    def load(self, path):
        pass

    def model_hash(self) -> str:
        blob = (self._mu.tobytes() if self._mu is not None else b"") + self.name.encode()
        return hashlib.sha256(blob).hexdigest()


# ============================================================================
# Fold iterator: emb-only (S=32 P96) or Tier-1 handcrafted baseline
# ============================================================================


def make_fold_iter(features: str, store):
    """emb -> replace X with the 768 frozen embedding cols; t1 -> raw handcrafted X."""
    def _it():
        for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR):
            if features == "emb":
                yield augment_fold_with_fang(fold, px=CTX_PX, dataset_dir=DATASET_DIR,
                                             replace=True, store=store)
            else:
                yield fold
    return _it


def model_factory(variant: str, target_transform: str, clip_max: float | None):
    if variant == "mlp_reg":
        return lambda: MLPRegressorEnsemble(transform=target_transform, clip_max=clip_max)
    params = LGBMParams(n_estimators=400, learning_rate=0.05, early_stopping_rounds=40)
    return make_factory(variant, params)


def _print_summary(agg: dict, meaningful_threshold: float) -> None:
    """Report ranking + the OPERATIONALLY MEANINGFUL (rich/poor) metrics.

    The rich/poor cut is `y_true > meaningful_threshold` -- per target
    (fractional_area > 1e-2, boulder_count >= 50). Deliberately NOT presence
    (y_true > 0): detecting a single boulder is near-degenerate at S=32 and
    scientifically uninteresting (Brian, 2026-06-12) -- the same trap as the
    bc_ge_1 binary target.
    """
    def g(k):
        return agg.get(k, float("nan"))

    print(f"  spearman_rho   = {g('spearman_rho_mean'):.4f} +/- {g('spearman_rho_std'):.4f} "
          f"(n={agg.get('spearman_n')})", flush=True)
    print(f"  meaningful_auc = {g('meaningful_auc_mean'):.4f}   "
          f"pr_auc(@{meaningful_threshold:g}) = {g('pr_auc_mean'):.4f}")
    print(f"  precision@5%   = {g('precision_at_top_5pct_mean'):.4f}   "
          f"rmse_log1p = {g('rmse_log1p_mean'):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True,
                    choices=["mlp_reg", "lightgbm_tweedie", "lightgbm_two_stage_balanced"])
    ap.add_argument("--target", default="fractional_area", choices=sorted(TARGETS))
    ap.add_argument("--features", default="emb", choices=["emb", "t1"])
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    target_col, transform, meaningful_threshold = TARGETS[args.target]
    clip_max = 1.0 if args.target == "fractional_area" else None
    store = load_fang_store(CTX_PX, pool="gem", dataset_dir=DATASET_DIR) if args.features == "emb" else None

    label = f"tier2_{args.variant}_{args.features}_{args.target}_S{TILE_PX}"
    snapshot = {
        "variant": label, "task": "regression", "target_col": target_col,
        "transform": transform, "scheme": SCHEME, "dataset_dir": "dataset_v2",
        "scale_idx": SCALE_IDX, "tile_size_px": TILE_PX, "features": args.features,
        "pool": "gem" if args.features == "emb" else None,
    }
    cfg_hash = hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()[:16]
    snapshot["config_hash"] = cfg_hash
    out_dir = OUT_ROOT / label / cfg_hash

    if (out_dir / "metrics.json").exists() and not args.force:
        agg = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["aggregate"]
        print(f"=== {label}: cached (--force to rerun) ===")
        _print_summary(agg, meaningful_threshold)
        return 0

    t0 = time.monotonic()
    print(f"=== {label} (target_col={target_col}, transform={transform}, "
          f"meaningful>={meaningful_threshold:g}) ===", flush=True)
    result = run_loio(
        model_factory(args.variant, transform, clip_max),
        target_col=target_col, task="regression",
        fold_iter=make_fold_iter(args.features, store),
        snapshot=snapshot, verbose=True,
        meaningful_threshold=meaningful_threshold,
    )
    write_run_artifacts(result, out_dir)
    print(f"\n  [{label}] {time.monotonic() - t0:.0f} s -> {out_dir.relative_to(REPO_ROOT)}")
    _print_summary(result.aggregate, meaningful_threshold)
    print(f"  (per-bin RMSE + calibration in {out_dir.relative_to(REPO_ROOT)}/metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
