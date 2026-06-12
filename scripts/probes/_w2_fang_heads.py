"""PLAN_FM.md queue item 1: head bake-off on frozen Fang-ViT embeddings.

Is LightGBM the right reader of a dense 768-dim embedding? Trees split on
single axes; the FM-evaluation standard is linear/MLP/kNN probes. Four heads
on the IDENTICAL gem192 feature matrix (768 cols, S=64, no handcrafted
features so the comparison is purely head-class) in the identical LOIO
harness:

    lgbm    LightGBMClassification, Tier-1 refresh hyperparameters (ref class)
    logreg  sklearn LogisticRegression (lbfgs, C=1.0, class_weight=balanced),
            per-fold standardization inside the wrapper
    knn     cosine kNN (L2-normalize rows -> euclidean), k=50, distance-weighted
    mlp     torch 768-256-64-1, dropout 0.2, BCE(pos_weight), AdamW, early
            stopping on the rotated inner-val image; 3 seeds (stochastic cell
            -> the 3-seed rule applies) + their mean-prob ensemble

Fixed hyperparameters throughout -- this is a head-CLASS read, not a tuning
exercise (PLAN_FM.md 3: recipe shopping ends before the confirmation
declaration). Verdicts vs the Tier-1 reference exactly as in _w2_fang_probe.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_w2_fang_heads.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd

from scripts.probes._w2_fang_probe import (  # noqa: E402  -- probe-tier reuse
    DATASET_DIR, OUT_ROOT, SCALE_CONFIG, SCHEME, TARGET_ID,
    EmbeddingBank, make_fold_iter, verdict,
)
from src.modeling.binary_target import get_target
from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, LightGBMClassification

TILE_PX = 64
SCALE_IDX, T1_PREDS_REL, T1_SUMMARY_REL = SCALE_CONFIG[TILE_PX]


# ============================================================================
# Head wrappers (Model protocol: fit / predict / predict_presence_prob / ...)
# ============================================================================


@dataclass
class _StandardizedHead:
    """Shared per-fold standardization: fit mean/std on the training fold only."""

    name: str = "std_head"
    _mu: np.ndarray | None = field(default=None, init=False, repr=False)
    _sd: np.ndarray | None = field(default=None, init=False, repr=False)

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        return self._apply(X)

    def _apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sd

    def predict_presence_prob(self, X):
        return self.predict(X)

    def save(self, path):  # probe-tier: no persistence needed
        pass

    def model_hash(self) -> str:
        blob = (self._mu.tobytes() if self._mu is not None else b"") + self.name.encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass
class LogRegHead(_StandardizedHead):
    name: str = "logreg"
    _clf: object = field(default=None, init=False, repr=False)

    def fit(self, X, y, *, groups=None, eval_set=None):
        from sklearn.linear_model import LogisticRegression

        Xs = self._fit_scaler(np.asarray(X, dtype=np.float64))
        self._clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                       solver="lbfgs")
        self._clf.fit(Xs, np.asarray(y).astype(int))

    def predict(self, X):
        return self._clf.predict_proba(self._apply(np.asarray(X, dtype=np.float64)))[:, 1]


@dataclass
class KNNHead(_StandardizedHead):
    """Cosine kNN: L2-normalize rows, euclidean distance, distance-weighted vote."""

    name: str = "knn50"
    k: int = 50
    _clf: object = field(default=None, init=False, repr=False)

    @staticmethod
    def _l2(X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    def fit(self, X, y, *, groups=None, eval_set=None):
        from sklearn.neighbors import KNeighborsClassifier

        self._clf = KNeighborsClassifier(n_neighbors=self.k, weights="distance", n_jobs=-1)
        self._clf.fit(self._l2(np.asarray(X, dtype=np.float64)), np.asarray(y).astype(int))

    def predict(self, X):
        return self._clf.predict_proba(self._l2(np.asarray(X, dtype=np.float64)))[:, 1]


@dataclass
class MLPHead(_StandardizedHead):
    """768-256-64-1 torch MLP, BCE(pos_weight), AdamW, ES on the harness eval_set."""

    name: str = "mlp"
    seed: int = 0
    epochs: int = 60
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    _net: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.name = f"mlp_seed{self.seed}"

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

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Xs = self._fit_scaler(np.asarray(X, dtype=np.float32)).astype(np.float32)
        yv = np.asarray(y, dtype=np.float32)
        n_pos = float((yv == 1).sum())
        pos_weight = torch.tensor([(yv.size - n_pos) / n_pos if n_pos else 1.0],
                                  device=device)
        self._net = self._build(Xs.shape[1]).to(device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Xv_t = yv_t = None
        if eval_set is not None:
            Xv, yv_arr = eval_set
            Xv_t = torch.from_numpy(self._apply(np.asarray(Xv, dtype=np.float32)).astype(np.float32)).to(device)
            yv_t = torch.from_numpy(np.asarray(yv_arr, dtype=np.float32)).to(device)

        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(yv)
        n = Xt.shape[0]
        best, best_state, bad = float("inf"), None, 0
        for _epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i: i + self.batch]
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self._net(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
            if Xv_t is not None:
                self._net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(self._net(Xv_t).squeeze(-1), yv_t).item())
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self._net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self._net.load_state_dict(best_state)

    def predict(self, X):
        import torch

        device = next(self._net.parameters()).device
        Xs = self._apply(np.asarray(X, dtype=np.float32)).astype(np.float32)
        self._net.eval()
        out = np.empty(Xs.shape[0], dtype=np.float64)
        with torch.no_grad():
            for i in range(0, Xs.shape[0], 4096):
                xb = torch.from_numpy(Xs[i: i + 4096]).to(device)
                out[i: i + 4096] = torch.sigmoid(self._net(xb).squeeze(-1)).cpu().numpy()
        return out


# ============================================================================
# Bake-off driver
# ============================================================================


def run_head(label: str, factory, bank: EmbeddingBank) -> pd.DataFrame:
    target = get_target(TARGET_ID)
    snapshot = {
        "variant": f"fang_heads_{label}", "task": "classification",
        "target_id": TARGET_ID, "scheme": SCHEME, "dataset_dir": "dataset_v2",
        "scale_idx": SCALE_IDX, "tile_size_px": TILE_PX,
        "pool": "gem", "sources": ["ctx"], "head": label,
    }
    cfg_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()[:16]
    snapshot["config_hash"] = cfg_hash
    out_dir = OUT_ROOT / f"heads_{label}" / cfg_hash

    t0 = time.monotonic()
    print(f"=== head: {label} (gem192-only matrix) ===", flush=True)
    result = run_loio(
        factory, binarize=target.binarize, task="classification",
        fold_iter=make_fold_iter(bank, ("ctx",), SCALE_IDX, {"own": 64, "ctx": 192}),
        snapshot=snapshot, verbose=True,
    )
    write_run_artifacts(result, out_dir)
    v = verdict(label, result.predictions,
                REPO_ROOT / T1_PREDS_REL, REPO_ROOT / T1_SUMMARY_REL)
    (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print(f"\n  [{label}] {time.monotonic() - t0:.0f} s -> {out_dir.relative_to(REPO_ROOT)}")
    for lbl, row in v.items():
        slim = {k: (round(val, 4) if isinstance(val, float) else val)
                for k, val in row.items() if k not in ("per_image_dauc", "dauc_by_cause")}
        print(f"  {lbl}: {json.dumps(slim, default=str)}", flush=True)
    print()
    return result.predictions


def main() -> int:
    bank = EmbeddingBank("gem", pxs=(64, 192))
    params = LGBMParams(n_estimators=400, learning_rate=0.05, early_stopping_rounds=40)

    heads = [
        ("lgbm", lambda: LightGBMClassification(params=params)),
        ("logreg", LogRegHead),
        ("knn50", KNNHead),
        ("mlp_seed0", lambda: MLPHead(seed=0)),
        ("mlp_seed1", lambda: MLPHead(seed=1)),
        ("mlp_seed2", lambda: MLPHead(seed=2)),
    ]
    preds_by_head: dict[str, pd.DataFrame] = {}
    for label, factory in heads:
        preds_by_head[label] = run_head(label, factory, bank)

    # MLP 3-seed mean-prob ensemble (the stochastic cell's promotable form).
    base = None
    for s in (0, 1, 2):
        p = preds_by_head[f"mlp_seed{s}"][["obs_id", "ti", "tj", "y_true", "y_pred"]]
        p = p.rename(columns={"y_pred": f"p{s}"})
        base = p if base is None else base.merge(
            p.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
    base["y_pred"] = base[["p0", "p1", "p2"]].mean(axis=1)
    v = verdict("mlp_ens3", base, REPO_ROOT / T1_PREDS_REL, REPO_ROOT / T1_SUMMARY_REL)
    out_dir = OUT_ROOT / "heads_mlp_ens3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print("=== mlp_ens3 (mean of 3 seeds) ===")
    for lbl, row in v.items():
        slim = {k: (round(val, 4) if isinstance(val, float) else val)
                for k, val in row.items() if k not in ("per_image_dauc", "dauc_by_cause")}
        print(f"  {lbl}: {json.dumps(slim, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
