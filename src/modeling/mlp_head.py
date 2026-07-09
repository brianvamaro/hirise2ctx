"""Productized MLP probe head + deployable ensemble (PLAN_FM §2.6.A).

The frozen recipe's head (`mlp_ens3`, DECISIONS.md 2026-06-12 "Freeze window
CLOSED") lived only inside the LOIO probe harness (`scripts/probes/
_w2_fang_heads.py::MLPHead`, re-trained per fold). A *map* needs ONE model
trained on ALL images, persisted, and re-loadable. This module is that head,
factored into `src/`:

  * `MLPClassifierHead`  -- one 768-256-64-1 BCE MLP, Model-protocol compliant
    (fit / predict / save / load / model_hash), so it can also be dropped into
    the LOIO harness. Faithful to the frozen recipe: median-impute + z-score
    feature scaler fit on train, dropout 0.2, AdamW lr1e-3 wd1e-4, BCE with
    `pos_weight = n_neg/n_pos`, early stop (patience 8) on a held-out inner-val
    image's BCE loss, sigmoid-probability output.
  * `DeployableHead`     -- the 3-seed ensemble trained on all data. `fit`
    rotates one inner-val image per seed (so every image is in-training for at
    least n_seeds-1 seeds), `predict` is the mean of the seed sigmoid
    probabilities (the deterministic-promotable form of the stochastic cell,
    exactly the W2 SmallCNN lesson). `save`/`load` persists the scalers, the
    seed state-dicts, and a recipe card.

PERF (PLAN_FM §2.6.A / `_fm_tier2_regression.py` PERF NOTE): the frozen probe
trained at `batch=512`, which left the tiny net GPU-overhead-bound (~15% util).
The default here is `batch=4096` with the full standardized train tensor pinned
to the device ONCE before the epoch loop (147k x 768 f32 ~= 450 MB, fits the
8 GB card), ~3-5x faster with no material effect on the numbers. Batch size is
not part of the frozen recipe card (which names arch/dropout/optimizer/target),
so this is an implementation choice, not a recipe change; the LOIO-validated
0.7832 stands as the recipe's generalization estimate, and this is the same
recipe trained on all data.

Torch is imported lazily inside the methods (matching `src.fm_embeddings` /
`src.modeling.gbm`) so importing this module never requires torch.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.modeling.base import hash_bytes

EMBED_DIM = 768
DEFAULT_HIDDEN = (256, 64)
DEFAULT_DROPOUT = 0.2


# ============================================================================
# Feature scaler (median-impute then z-score) -- frozen-recipe parity
# ============================================================================


@dataclass
class FeatureScaler:
    """Median-impute NaN columns then standardize. Fit on the training fold only.

    The frozen emb-only matrix has NaN rows where a tile's 3x3-context box spilled
    past the window margin (`load_fang_store` marks them); the recipe imputes the
    per-column median (`nanmedian`) and z-scores. An all-NaN column maps to the
    constant 0 (median 0, the column carries no information). Mirrors the probe
    `_StandardizedHead` so the persisted head matches what was validated.
    """

    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    med: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        import warnings

        X = np.asarray(X, dtype=np.float64)
        with warnings.catch_warnings():  # all-NaN columns -> median NaN -> 0 (intended)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.med = np.nanmedian(X, axis=0)
        self.med[~np.isfinite(self.med)] = 0.0
        Xi = self._impute(X)
        self.mu = Xi.mean(axis=0)
        self.sd = Xi.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return ((Xi - self.mu) / self.sd).astype(np.float32)

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        r, c = np.where(~np.isfinite(X))
        X[r, c] = self.med[c]
        return X

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None:
            raise RuntimeError("FeatureScaler.apply before fit")
        return ((self._impute(np.asarray(X, dtype=np.float64)) - self.mu) / self.sd).astype(np.float32)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"mu": self.mu, "sd": self.sd, "med": self.med}

    @classmethod
    def from_arrays(cls, d) -> "FeatureScaler":
        return cls(mu=np.asarray(d["mu"]), sd=np.asarray(d["sd"]), med=np.asarray(d["med"]))


def build_mlp(d_in: int, hidden=DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT):
    """Frozen-recipe MLP: Linear/ReLU/Dropout stack ending in a 1-logit head."""
    import torch.nn as nn

    layers: list = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ============================================================================
# Single-seed MLP classifier head (Model protocol)
# ============================================================================


@dataclass
class MLPClassifierHead:
    """One 768-256-64-1 BCE MLP returning sigmoid probabilities.

    Implements the `src.modeling.base.Model` protocol, so it can be a drop-in
    factory in the LOIO harness as well as a member of `DeployableHead`. The
    weights and feature scaler are deterministic given `seed`; `predict` returns
    P(positive) in [0, 1] (NaN feature rows are median-imputed by the scaler, so
    predictions are always finite -- the caller drops invalid tiles upstream).
    """

    name: str = "mlp_clf"
    seed: int = 0
    hidden: tuple[int, ...] = DEFAULT_HIDDEN
    dropout: float = DEFAULT_DROPOUT
    epochs: int = 60
    batch: int = 4096           # perf fix; not in the frozen recipe card
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    scaler: FeatureScaler = field(default_factory=FeatureScaler, repr=False)
    _net: object = field(default=None, init=False, repr=False)
    _d_in: int | None = field(default=None, init=False, repr=False)

    def fit(self, X, y, *, groups=None, eval_set=None) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Xs = self.scaler.fit(np.asarray(X, dtype=np.float32))
        yv = np.asarray(y, dtype=np.float32)
        self._d_in = Xs.shape[1]
        n_pos = float((yv == 1).sum())
        pos_weight = torch.tensor([(yv.size - n_pos) / n_pos if n_pos else 1.0], device=device)

        self._net = build_mlp(self._d_in, self.hidden, self.dropout).to(device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # PERF: pin the full standardized train tensor to the device once.
        Xt = torch.from_numpy(Xs).to(device)
        yt = torch.from_numpy(yv).to(device)
        n = Xt.shape[0]

        Xv_t = yv_t = None
        if eval_set is not None:
            Xv_raw, yv_raw = eval_set
            Xv_t = torch.from_numpy(self.scaler.apply(np.asarray(Xv_raw, dtype=np.float32))).to(device)
            yv_t = torch.from_numpy(np.asarray(yv_raw, dtype=np.float32)).to(device)

        best, best_state, bad = float("inf"), None, 0
        for _epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch):
                idx = perm[i: i + self.batch]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self._net(Xt[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
            if Xv_t is not None:
                self._net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(self._net(Xv_t).squeeze(-1), yv_t).item())
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self._net.load_state_dict(best_state)

    def predict(self, X) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("MLPClassifierHead.predict before fit/load")
        device = next(self._net.parameters()).device
        Xs = self.scaler.apply(np.asarray(X, dtype=np.float32))
        self._net.eval()
        out = np.empty(Xs.shape[0], dtype=np.float64)
        with torch.no_grad():
            for i in range(0, Xs.shape[0], self.batch):
                xb = torch.from_numpy(Xs[i: i + self.batch]).to(device)
                out[i: i + self.batch] = torch.sigmoid(self._net(xb).squeeze(-1)).cpu().numpy()
        return out

    def predict_presence_prob(self, X) -> np.ndarray:
        return self.predict(X)

    # ---- persistence ----
    def save(self, path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "state.pt")
        np.savez(path / "scaler.npz", **self.scaler.to_arrays())
        (path / "meta.json").write_text(json.dumps({
            "name": self.name, "seed": self.seed, "hidden": list(self.hidden),
            "dropout": self.dropout, "d_in": self._d_in, "batch": self.batch,
        }, indent=2), encoding="utf-8")

    def load(self, path) -> None:
        import torch

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.hidden = tuple(meta["hidden"])
        self.dropout = meta["dropout"]
        self._d_in = meta["d_in"]
        self.batch = meta.get("batch", self.batch)
        self.scaler = FeatureScaler.from_arrays(np.load(path / "scaler.npz"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = build_mlp(self._d_in, self.hidden, self.dropout).to(device)
        self._net.load_state_dict(torch.load(path / "state.pt", map_location=device))
        self._net.eval()

    def model_hash(self) -> str:
        import torch
        import io

        if self._net is None:
            return ""
        buf = io.BytesIO()
        torch.save(self._net.state_dict(), buf)
        blob = buf.getvalue()
        if self.scaler.mu is not None:
            blob += self.scaler.mu.tobytes() + self.scaler.sd.tobytes()
        return hash_bytes(blob)


# ============================================================================
# Deployable 3-seed ensemble (train-on-all)
# ============================================================================

# The frozen recipe these defaults reproduce. Recorded on the recipe card so a
# loaded head self-describes which validated cell it deploys.
FROZEN_RECIPE = {
    "cell": "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2",
    "target_id": "fa_gt_1e-2",
    "scale_idx": 2,
    "tile_size_px": 32,
    "input_px": 96,          # 3x3-context box side fed to the ViT
    "pool": "gem",
    "embedding": "fang_vit_b16_gem_p3",
    "loio_pooled_pr_auc": 0.7832,
    "loio_med_per_image_auc": 0.7865,
}


class DeployableHead:
    """Frozen `mlp_ens3` recipe trained on ALL images -> one persisted model.

    `fit(X, y, groups)` trains `len(seeds)` `MLPClassifierHead`s; seed s holds out
    inner-val image `sorted(unique(groups))[s % n_images]` for early stopping and
    trains on the rest. Across the default 3 seeds every image is in-training for
    >= 2 seeds, and no image is permanently excluded from the ensemble. `predict`
    returns the mean of the seed sigmoid probabilities.

    This mirrors each LOIO fold's procedure (train on N-1 images, one rotated
    inner-val monitor) minus the held-out test fold -- so it is the same recipe,
    applied to all data, that the LOIO numbers on FROZEN_RECIPE validated.
    """

    name = "deployable_mlp_ens3"

    def __init__(self, seeds: tuple[int, ...] = (0, 1, 2), hidden=DEFAULT_HIDDEN,
                 dropout: float = DEFAULT_DROPOUT, epochs: int = 60, batch: int = 4096,
                 lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 8,
                 recipe: dict | None = None, nuisance_basis: np.ndarray | None = None) -> None:
        self.seeds = tuple(seeds)
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.weight_decay, self.patience = weight_decay, patience
        self.recipe = dict(recipe or FROZEN_RECIPE)
        # H2 (PLAN_StripingArtifact PHASE 2): optional frame-nuisance subspace removed
        # from the embeddings BEFORE the scaler. N is (EMBED_DIM, k) with orthonormal
        # columns; projection X <- X - (X @ N) @ N.T is applied identically in fit and
        # predict, so it travels to deploy via load() -> train/deploy parity for free.
        self.nuisance_basis = (None if nuisance_basis is None
                               else np.asarray(nuisance_basis, dtype=np.float32))
        self._members: list[MLPClassifierHead] = []
        self._train_obs_ids: list[str] = []
        self._trained_at: str | None = None

    def _project(self, X: np.ndarray) -> np.ndarray:
        """Remove the frame-nuisance subspace (H2). No-op if no basis is set.

        Only finite rows are projected; all-NaN invalid-tile rows pass through
        unchanged for the FeatureScaler to median-impute downstream.
        """
        if self.nuisance_basis is None:
            return X
        N = self.nuisance_basis
        out = np.array(X, dtype=np.float32, copy=True)
        fin = np.isfinite(out).all(axis=1)
        if fin.any():
            Xf = out[fin]
            out[fin] = Xf - (Xf @ N) @ N.T
        return out

    def _member(self, seed: int) -> MLPClassifierHead:
        return MLPClassifierHead(
            name=f"{self.name}_seed{seed}", seed=seed, hidden=self.hidden,
            dropout=self.dropout, epochs=self.epochs, batch=self.batch, lr=self.lr,
            weight_decay=self.weight_decay, patience=self.patience)

    def fit(self, X, y, *, groups, obs_to_int: dict[str, int] | None = None,
            verbose: bool = True) -> "DeployableHead":
        X = self._project(np.asarray(X, dtype=np.float32))
        y = np.asarray(y).astype(np.float32)
        groups = np.asarray(groups)
        unique = np.unique(groups)
        if unique.size < 2:
            raise ValueError("DeployableHead.fit needs >= 2 distinct image groups for inner-val rotation")

        int_to_obs = {v: k for k, v in (obs_to_int or {}).items()}
        self._members = []
        for s in self.seeds:
            val_code = unique[s % unique.size]
            val_mask = groups == val_code
            tr_mask = ~val_mask
            head = self._member(s)
            if verbose:
                vname = int_to_obs.get(int(val_code), str(val_code))
                print(f"  seed {s}: train n={int(tr_mask.sum())} ({unique.size - 1} imgs), "
                      f"inner-val={vname} n={int(val_mask.sum())}", flush=True)
            head.fit(X[tr_mask], y[tr_mask], groups=groups[tr_mask],
                     eval_set=(X[val_mask], y[val_mask]))
            self._members.append(head)
        self._train_obs_ids = sorted(int_to_obs.get(int(c), str(c)) for c in unique)
        self._trained_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        return self

    def predict(self, X) -> np.ndarray:
        if not self._members:
            raise RuntimeError("DeployableHead.predict before fit/load")
        X = self._project(np.asarray(X, dtype=np.float32))
        acc = np.zeros(X.shape[0], dtype=np.float64)
        for head in self._members:
            acc += head.predict(X)
        return acc / len(self._members)

    def predict_presence_prob(self, X) -> np.ndarray:
        return self.predict(X)

    def recipe_hash(self) -> str:
        """Stable hash of the recipe config (independent of trained weights)."""
        blob = json.dumps({
            "name": self.name, "seeds": list(self.seeds), "hidden": list(self.hidden),
            "dropout": self.dropout, "epochs": self.epochs, "lr": self.lr,
            "weight_decay": self.weight_decay, "patience": self.patience,
            "recipe": self.recipe,
        }, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def model_hash(self) -> str:
        blob = "|".join(h.model_hash() for h in self._members).encode()
        if self.nuisance_basis is not None:
            blob += self.nuisance_basis.tobytes()
        return hash_bytes(blob)

    # ---- persistence ----
    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for s, head in zip(self.seeds, self._members):
            head.save(path / f"seed{s}")
        if self.nuisance_basis is not None:
            np.save(path / "nuisance_basis.npy", self.nuisance_basis)
        card = {
            "name": self.name, "seeds": list(self.seeds), "hidden": list(self.hidden),
            "dropout": self.dropout, "epochs": self.epochs, "batch": self.batch,
            "lr": self.lr, "weight_decay": self.weight_decay, "patience": self.patience,
            "recipe": self.recipe, "recipe_hash": self.recipe_hash(),
            "model_hash": self.model_hash(), "n_train_images": len(self._train_obs_ids),
            "train_obs_ids": self._train_obs_ids, "trained_at_iso": self._trained_at,
            "nuisance_k": (None if self.nuisance_basis is None
                           else int(self.nuisance_basis.shape[1])),
        }
        (path / "recipe.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "DeployableHead":
        path = Path(path)
        card = json.loads((path / "recipe.json").read_text(encoding="utf-8"))
        basis_path = path / "nuisance_basis.npy"
        basis = np.load(basis_path) if basis_path.exists() else None
        head = cls(seeds=tuple(card["seeds"]), hidden=tuple(card["hidden"]),
                   dropout=card["dropout"], epochs=card["epochs"], batch=card["batch"],
                   lr=card["lr"], weight_decay=card["weight_decay"],
                   patience=card["patience"], recipe=card.get("recipe"),
                   nuisance_basis=basis)
        head._members = []
        for s in head.seeds:
            m = head._member(s)
            m.load(path / f"seed{s}")
            head._members.append(m)
        head._train_obs_ids = card.get("train_obs_ids", [])
        head._trained_at = card.get("trained_at_iso")
        return head
