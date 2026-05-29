"""Small CNN baseline on Stage 4b context patches.

Architecture from PLAN_modeling.md §4: three conv blocks (16/32/32 channels),
BN-before-ReLU, MaxPool x2 between blocks 1 and 2, GlobalAvgPool before the
2-FC head with Dropout(0.3). ~30k params at S=32, ~35k at S=64. Loss:
`log1p(fractional_area)` + Huber. PyTorch has no native Tweedie; the
variance-stabilising log1p+Huber form is the practical analogue and matches
the GBM `LightGBMLog1pHuber` shadow.

Augmentations (per PLAN_modeling.md §4, applied per-batch on uint8 inputs before
the /255.0 cast):
  - 50% horizontal flip, 50% vertical flip,
  - random 90 deg rotation 0/1/2/3,
  - brightness jitter +-15% of the per-tile intensity range,
  - contrast jitter in [0.85, 1.15],
  - additive Gaussian noise sigma=2 on the 0-255 scale.

These target per-image confounds (illumination geometry, atmospheric haze,
sensor drift) -- the LOIO-CV failure mode per PLAN_modeling.md §11.5.

Patches come from `dataset/context_patches/{ObsId}_S{size}.npy` via
`src.modeling.loaders.gather_patches`. Rows with `patch_idx_S{size} == -1`
(window-edge margin) are dropped during training and predicted as zero (the
field-floor baseline) during inference.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.modeling.base import Model
from src.modeling.loaders import gather_patches


# ============================================================================
# Hyperparameters
# ============================================================================


@dataclass
class CNNParams:
    patch_size_px: int = 32              # 32 or 64; matches dataset/context_patches/{}_S{}.npy
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    huber_beta: float = 1.0              # PyTorch SmoothL1Loss `beta`
    early_stopping_patience: int = 8     # epochs without inner-val improvement
    seed: int = 0
    n_workers: int = 0                   # Windows CPU; keep at 0 to avoid multi-process spawn cost
    device: str = "cpu"                  # explicit -- the geospatial env doesn't ship CUDA
    dataset_dir: str | None = None       # None = ./dataset (v1); set to dataset_v2[_dev] for the A/B


# ============================================================================
# Architecture
# ============================================================================


class SmallCNN(nn.Module):
    """3 conv blocks -> GAP -> 2 FC. See PLAN_modeling.md Section 4 for the rationale."""

    def __init__(self, patch_size_px: int = 32, dropout: float = 0.3):
        super().__init__()
        # Input: (B, 1, S, S)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # S -> S/2
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # S/2 -> S/4
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),        # -> (B, 32, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(-1)


# ============================================================================
# Dataset + augmentation
# ============================================================================


class _PatchDataset(Dataset):
    """Patches as uint8 tensors + the regression label.

    `augment=True` applies the PLAN §4 augmentation pipeline on the uint8 array
    before the /255 float cast, so augmentation cost is paid in uint8 space
    (small fast ops); the float conversion happens once at the end.
    """

    def __init__(self, patches: np.ndarray, y_log: np.ndarray, augment: bool, rng_seed: int = 0):
        assert patches.ndim == 3, "expected (N, S, S) uint8"
        assert patches.shape[0] == y_log.shape[0]
        self.patches = patches
        self.y_log = y_log.astype(np.float32, copy=False)
        self.augment = augment
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return self.patches.shape[0]

    def _augment_one(self, img: np.ndarray) -> np.ndarray:
        # img: (S, S) uint8
        # 50% horizontal flip
        if self.rng.random() < 0.5:
            img = img[:, ::-1]
        # 50% vertical flip
        if self.rng.random() < 0.5:
            img = img[::-1, :]
        # random 90 deg rotation (0/1/2/3)
        k = int(self.rng.integers(0, 4))
        if k:
            img = np.rot90(img, k)
        # brightness jitter +-15% of the tile's range
        img = img.astype(np.int16)
        rng_brightness = self.rng.uniform(-0.15, 0.15) * 255
        img = img + int(rng_brightness)
        # contrast jitter in [0.85, 1.15], around the per-tile mean
        c = self.rng.uniform(0.85, 1.15)
        mean = img.mean()
        img = ((img - mean) * c + mean).astype(np.int16)
        # additive Gaussian noise sigma=2 on the 0-255 scale
        noise = self.rng.normal(0, 2.0, size=img.shape)
        img = img + noise.astype(np.int16)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.patches[idx]
        if self.augment:
            img = self._augment_one(img)
        x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0) / 255.0  # (1, S, S)
        y = torch.tensor(self.y_log[idx], dtype=torch.float32)
        return x, y


# ============================================================================
# Model wrapper
# ============================================================================


@dataclass
class SmallCNNRegressor:
    """Implements the Model Protocol over `SmallCNN` with the PLAN §4 training loop."""

    params: CNNParams = field(default_factory=CNNParams)
    name: str = "cnn_log1p_huber"

    _net: SmallCNN | None = field(default=None, init=False, repr=False)
    _train_keys: object = field(default=None, init=False, repr=False)
    _train_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _val_keys: object = field(default=None, init=False, repr=False)
    _val_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _state_blob: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.name = f"cnn_log1p_huber_S{self.params.patch_size_px}"

    # ------------------------------------------------------------------
    # Data binding -- the CNN needs the original `keys` dataframe + dataset_dir
    # (not just the X feature matrix), so callers must pre-bind them via
    # `bind_data` before invoking the Protocol's `fit`.
    # ------------------------------------------------------------------

    def bind_train_data(self, keys, y):
        self._train_keys = keys
        self._train_y = np.asarray(y, dtype=np.float64)

    def bind_val_data(self, keys, y):
        self._val_keys = keys
        self._val_y = np.asarray(y, dtype=np.float64)

    def bind_predict_data(self, keys):
        self._predict_keys = keys  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Model Protocol
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,                  # ignored: CNN consumes patches, not the tabular X
        y: np.ndarray,
        *,
        groups=None,
        eval_set=None,
    ) -> None:
        assert self._train_keys is not None and self._train_y is not None, (
            "bind_train_data must be called before fit"
        )
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        # Materialise patches once per epoch -- much faster than per-item __getitem__
        # I/O for 9-image datasets. At Stage-4b size (~10-100k patches per fold) this
        # fits in RAM (uint8: 32x32 ~1 KB, 64x64 ~4 KB -> 100k tiles ~100/400 MB).
        train_patches, train_rows = gather_patches(
            self._train_keys, self.params.patch_size_px, dataset_dir=self.params.dataset_dir)
        y_train_valid = self._train_y[train_rows]
        y_train_log = np.log1p(np.clip(y_train_valid, 0.0, None))

        ds_train = _PatchDataset(train_patches, y_train_log, augment=True, rng_seed=self.params.seed)
        loader_train = DataLoader(
            ds_train, batch_size=self.params.batch_size, shuffle=True,
            num_workers=self.params.n_workers, drop_last=False,
        )

        # Inner-validation set is whatever the caller bound via bind_val_data.
        # Fall back to no early stopping if not bound.
        val_loader: DataLoader | None = None
        if self._val_keys is not None and self._val_y is not None and len(self._val_keys) > 0:
            val_patches, val_rows = gather_patches(
                self._val_keys, self.params.patch_size_px, dataset_dir=self.params.dataset_dir)
            y_val_log = np.log1p(np.clip(self._val_y[val_rows], 0.0, None))
            ds_val = _PatchDataset(val_patches, y_val_log, augment=False, rng_seed=self.params.seed)
            val_loader = DataLoader(
                ds_val, batch_size=self.params.batch_size, shuffle=False,
                num_workers=self.params.n_workers,
            )

        device = torch.device(self.params.device)
        self._net = SmallCNN(self.params.patch_size_px, dropout=self.params.dropout).to(device)
        opt = optim.AdamW(self._net.parameters(), lr=self.params.learning_rate,
                          weight_decay=self.params.weight_decay)
        loss_fn = nn.HuberLoss(reduction="mean", delta=self.params.huber_beta)

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        for epoch in range(1, self.params.epochs + 1):
            self._net.train()
            train_loss_sum = 0.0
            train_n = 0
            for xb, yb in loader_train:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                train_loss_sum += float(loss.item()) * xb.size(0)
                train_n += xb.size(0)
            train_loss = train_loss_sum / max(train_n, 1)

            val_loss = float("nan")
            if val_loader is not None:
                self._net.eval()
                val_loss_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device); yb = yb.to(device)
                        pred = self._net(xb)
                        val_loss_sum += float(loss_fn(pred, yb).item()) * xb.size(0)
                        val_n += xb.size(0)
                val_loss = val_loss_sum / max(val_n, 1)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    bad_epochs = 0
                    # Capture state_dict bytes so we can restore "best" at the end.
                    buf = io.BytesIO()
                    torch.save({k: v.cpu().clone() for k, v in self._net.state_dict().items()}, buf)
                    best_state = buf.getvalue()
                else:
                    bad_epochs += 1

            print(f"    epoch {epoch:>3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"bad={bad_epochs}", flush=True)

            if val_loader is not None and bad_epochs >= self.params.early_stopping_patience:
                print(f"    early stop at epoch {epoch}")
                break

        # Restore best-validation state.
        if best_state is not None:
            self._net.load_state_dict(torch.load(io.BytesIO(best_state)))
            self._state_blob = best_state
        else:
            buf = io.BytesIO()
            torch.save({k: v.cpu().clone() for k, v in self._net.state_dict().items()}, buf)
            self._state_blob = buf.getvalue()

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "fit() before predict()"
        keys = getattr(self, "_predict_keys", None)
        assert keys is not None, "bind_predict_data must be called before predict()"
        # n_total = number of rows in X (which equals len(keys)); CNN can only predict
        # for rows with a valid patch_idx_S{S} >= 0. Margin rows get the field-floor
        # baseline (0.0).
        n_total = X.shape[0]
        patches, valid_rows = gather_patches(keys, self.params.patch_size_px, dataset_dir=self.params.dataset_dir)
        out = np.zeros(n_total, dtype=np.float64)
        if valid_rows.size == 0:
            return out

        device = torch.device(self.params.device)
        self._net.eval()
        ds = _PatchDataset(patches, np.zeros(patches.shape[0], dtype=np.float32),
                           augment=False, rng_seed=0)
        loader = DataLoader(ds, batch_size=self.params.batch_size, shuffle=False,
                            num_workers=self.params.n_workers)
        preds_log = np.empty(patches.shape[0], dtype=np.float32)
        cursor = 0
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                p = self._net(xb).cpu().numpy()
                preds_log[cursor : cursor + p.shape[0]] = p
                cursor += p.shape[0]
        preds = np.clip(np.expm1(preds_log.astype(np.float64)), 0.0, None)
        out[valid_rows] = preds
        return out

    def predict_presence_prob(self, X: np.ndarray) -> np.ndarray | None:
        return None  # single-stage regression

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self._state_blob is not None
        path.write_bytes(self._state_blob)

    def load(self, path: str | Path) -> None:
        self._state_blob = Path(path).read_bytes()
        self._net = SmallCNN(self.params.patch_size_px, dropout=self.params.dropout)
        self._net.load_state_dict(torch.load(io.BytesIO(self._state_blob)))

    def model_hash(self) -> str:
        if self._state_blob is None:
            return ""
        return hashlib.sha256(self._state_blob).hexdigest()


@dataclass
class SmallCNNClassifier:
    """Binary presence CNN: the `SmallCNN` backbone + `BCEWithLogitsLoss(pos_weight)`.

    The documented fix for the v1 below-chance CNN (modeling_results.md §3.3): rather than
    a log1p+Huber regression that collapses to ~0 on a zero-inflated target, train a
    dedicated presence classifier with class-balanced `pos_weight = n_neg / n_pos`.

    Mirrors `SmallCNNRegressor`'s bind/fit/predict API so the same driver loop serves both.
    `fit` receives a **binary 0/1** `y` (the caller binarises via
    `src.modeling.binary_target`); `predict` returns P(positive) in [0, 1];
    `predict_presence_prob` returns the same. Margin rows (no patch) predict P=0.
    """

    params: CNNParams = field(default_factory=CNNParams)
    name: str = "cnn_bce"

    _net: SmallCNN | None = field(default=None, init=False, repr=False)
    _train_keys: object = field(default=None, init=False, repr=False)
    _train_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _val_keys: object = field(default=None, init=False, repr=False)
    _val_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _state_blob: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.name = f"cnn_bce_S{self.params.patch_size_px}"

    def bind_train_data(self, keys, y):
        self._train_keys = keys
        self._train_y = np.asarray(y, dtype=np.float64)

    def bind_val_data(self, keys, y):
        self._val_keys = keys
        self._val_y = np.asarray(y, dtype=np.float64)

    def bind_predict_data(self, keys):
        self._predict_keys = keys  # type: ignore[attr-defined]

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups=None, eval_set=None) -> None:
        assert self._train_keys is not None and self._train_y is not None, (
            "bind_train_data must be called before fit"
        )
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        train_patches, train_rows = gather_patches(
            self._train_keys, self.params.patch_size_px, dataset_dir=self.params.dataset_dir)
        y_train = (self._train_y[train_rows] > 0.5).astype(np.float32)
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32)

        ds_train = _PatchDataset(train_patches, y_train, augment=True, rng_seed=self.params.seed)
        loader_train = DataLoader(ds_train, batch_size=self.params.batch_size, shuffle=True,
                                  num_workers=self.params.n_workers, drop_last=False)

        val_loader: DataLoader | None = None
        if self._val_keys is not None and self._val_y is not None and len(self._val_keys) > 0:
            val_patches, val_rows = gather_patches(
                self._val_keys, self.params.patch_size_px, dataset_dir=self.params.dataset_dir)
            y_val = (self._val_y[val_rows] > 0.5).astype(np.float32)
            ds_val = _PatchDataset(val_patches, y_val, augment=False, rng_seed=self.params.seed)
            val_loader = DataLoader(ds_val, batch_size=self.params.batch_size, shuffle=False,
                                    num_workers=self.params.n_workers)

        device = torch.device(self.params.device)
        self._net = SmallCNN(self.params.patch_size_px, dropout=self.params.dropout).to(device)
        opt = optim.AdamW(self._net.parameters(), lr=self.params.learning_rate,
                          weight_decay=self.params.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        for epoch in range(1, self.params.epochs + 1):
            self._net.train()
            for xb, yb in loader_train:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self._net(xb), yb)
                loss.backward(); opt.step()

            val_loss = float("nan")
            if val_loader is not None:
                self._net.eval()
                vs = 0.0; vn = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device); yb = yb.to(device)
                        vs += float(loss_fn(self._net(xb), yb).item()) * xb.size(0); vn += xb.size(0)
                val_loss = vs / max(vn, 1)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss; bad_epochs = 0
                    buf = io.BytesIO()
                    torch.save({k: v.cpu().clone() for k, v in self._net.state_dict().items()}, buf)
                    best_state = buf.getvalue()
                else:
                    bad_epochs += 1
            print(f"    epoch {epoch:>3d}  val_bce={val_loss:.4f}  bad={bad_epochs}", flush=True)
            if val_loader is not None and bad_epochs >= self.params.early_stopping_patience:
                print(f"    early stop at epoch {epoch}"); break

        if best_state is not None:
            self._net.load_state_dict(torch.load(io.BytesIO(best_state)))
            self._state_blob = best_state
        else:
            buf = io.BytesIO()
            torch.save({k: v.cpu().clone() for k, v in self._net.state_dict().items()}, buf)
            self._state_blob = buf.getvalue()

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "fit() before predict()"
        keys = getattr(self, "_predict_keys", None)
        assert keys is not None, "bind_predict_data must be called before predict()"
        n_total = X.shape[0]
        patches, valid_rows = gather_patches(keys, self.params.patch_size_px, dataset_dir=self.params.dataset_dir)
        out = np.zeros(n_total, dtype=np.float64)
        if valid_rows.size == 0:
            return out
        device = torch.device(self.params.device)
        self._net.eval()
        ds = _PatchDataset(patches, np.zeros(patches.shape[0], dtype=np.float32), augment=False, rng_seed=0)
        loader = DataLoader(ds, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.n_workers)
        probs = np.empty(patches.shape[0], dtype=np.float32)
        cursor = 0
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                p = torch.sigmoid(self._net(xb)).cpu().numpy()
                probs[cursor: cursor + p.shape[0]] = p
                cursor += p.shape[0]
        out[valid_rows] = probs.astype(np.float64)
        return out

    def predict_presence_prob(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        assert self._state_blob is not None
        path.write_bytes(self._state_blob)

    def load(self, path: str | Path) -> None:
        self._state_blob = Path(path).read_bytes()
        self._net = SmallCNN(self.params.patch_size_px, dropout=self.params.dropout)
        self._net.load_state_dict(torch.load(io.BytesIO(self._state_blob)))

    def model_hash(self) -> str:
        if self._state_blob is None:
            return ""
        return hashlib.sha256(self._state_blob).hexdigest()
