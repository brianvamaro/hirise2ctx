"""Load packaged Stage 5 X/y/groups into model-ready arrays.

Thin wrapper over `dataset/packaged/{scheme}/`. Knows how to:

- discover available schemes and folds,
- load one fold as numpy arrays,
- iterate all LOIO folds in order,
- subset by `scale_idx` (per-scale modeling per PLAN_modeling.md §6 Option A),
- expose feature-name lists and the ObsId<->int code map.

Does NOT re-derive splits. The split definitions live in `dataset/splits/` (written
by Stage 5 via `src.dataset.build_split`); modeling reads from the materialised
parquets so split definition is single-sourced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
PACKAGED_SUBDIR = "packaged"

# Mirrors src.dataset.TILE_KEY_COLUMNS; duplicated to avoid the import cycle (this module
# is consumed by training code; src.dataset is consumed by the pipeline build).
TILE_KEY_COLUMNS = ["obs_id", "scale_idx", "tile_size_px", "ti", "tj"]


# ============================================================================
# Fold loading
# ============================================================================


@dataclass(frozen=True)
class Fold:
    """One fold of model-ready data, optionally filtered to a single scale.

    `X_train` / `X_test` are float32 ndarrays shaped (n, n_features); feature names
    in `feature_names`. `y_train` / `y_test` are the full label dataframes (caller
    picks the column). `groups_train` / `groups_test` are int32 ObsId codes
    (decode via `obs_to_int`). `keys_train` / `keys_test` carry the tile keys
    needed to join back to context patches and label columns.

    `held_out_obs_ids`: the test set's ObsIds, in the order Stage 5 wrote the fold.
    For loio_9fold this is exactly one ObsId.
    """

    fold_idx: int
    scheme: str
    scale_idx: int | None  # None = all scales concatenated
    X_train: np.ndarray
    y_train: pd.DataFrame
    groups_train: np.ndarray
    keys_train: pd.DataFrame
    X_test: np.ndarray
    y_test: pd.DataFrame
    groups_test: np.ndarray
    keys_test: pd.DataFrame
    feature_names: list[str]
    obs_to_int: dict[str, int]
    held_out_obs_ids: list[str]


def package_dir(scheme: str, dataset_dir: Path | str | None = None) -> Path:
    base = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    return base / PACKAGED_SUBDIR / scheme


def load_metadata(scheme: str, dataset_dir: Path | str | None = None) -> dict:
    """Read `dataset/packaged/{scheme}/metadata.json`."""
    return json.loads((package_dir(scheme, dataset_dir) / "metadata.json").read_text(encoding="utf-8"))


def n_folds(scheme: str, dataset_dir: Path | str | None = None) -> int:
    return len(load_metadata(scheme, dataset_dir)["per_fold"])


def _feature_columns(x_df: pd.DataFrame) -> list[str]:
    """Everything that's not a tile-key column, a context-patch index, or provenance is a feature.

    `patch_idx_S*` columns are integer positions into the per-ObsId context-patch stacks
    (`dataset/context_patches/{obs}_S{size}.npy`); they're not predictive features
    themselves, so we keep them out of the model's X matrix and surface them on the
    keys frame instead (see `_key_columns`).
    """
    drop = set(TILE_KEY_COLUMNS) | {"config_hash_feat"}
    return [
        c for c in x_df.columns
        if c not in drop and not c.startswith("patch_idx_S")
    ]


def _key_columns(x_df: pd.DataFrame) -> list[str]:
    """Tile-key + patch-index columns the caller may need to join other tables / patches."""
    return [
        c for c in x_df.columns
        if c in TILE_KEY_COLUMNS or c.startswith("patch_idx_S")
    ]


def _subset_to_scale(df: pd.DataFrame, scale_idx: int | None) -> pd.DataFrame:
    if scale_idx is None:
        return df
    return df[df["scale_idx"] == scale_idx].reset_index(drop=True)


def load_fold(
    scheme: str,
    fold_idx: int,
    *,
    scale_idx: int | None = None,
    dataset_dir: Path | str | None = None,
) -> Fold:
    """Load one fold of the packaged dataset, optionally filtered to a single scale."""
    meta = load_metadata(scheme, dataset_dir)
    pdir = package_dir(scheme, dataset_dir)

    x_train_df = pd.read_parquet(pdir / f"X_train_fold{fold_idx}.parquet")
    y_train_df = pd.read_parquet(pdir / f"y_train_fold{fold_idx}.parquet")
    x_test_df = pd.read_parquet(pdir / f"X_test_fold{fold_idx}.parquet")
    y_test_df = pd.read_parquet(pdir / f"y_test_fold{fold_idx}.parquet")
    groups_train = np.load(pdir / f"groups_train_fold{fold_idx}.npy")
    groups_test = np.load(pdir / f"groups_test_fold{fold_idx}.npy")

    # Filter to one scale before computing feature columns -- columns are unaffected,
    # but row counts drop.
    if scale_idx is not None:
        train_mask = (x_train_df["scale_idx"] == scale_idx).to_numpy()
        test_mask = (x_test_df["scale_idx"] == scale_idx).to_numpy()
        x_train_df = x_train_df[train_mask].reset_index(drop=True)
        y_train_df = y_train_df[train_mask].reset_index(drop=True)
        groups_train = groups_train[train_mask]
        x_test_df = x_test_df[test_mask].reset_index(drop=True)
        y_test_df = y_test_df[test_mask].reset_index(drop=True)
        groups_test = groups_test[test_mask]

    feat_cols = _feature_columns(x_train_df)
    key_cols = _key_columns(x_train_df)

    X_train = x_train_df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    X_test = x_test_df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    keys_train = x_train_df[key_cols].reset_index(drop=True)
    keys_test = x_test_df[key_cols].reset_index(drop=True)

    fold_meta = meta["per_fold"][fold_idx]
    held_out = list(fold_meta["test_obs_ids"])

    return Fold(
        fold_idx=fold_idx,
        scheme=scheme,
        scale_idx=scale_idx,
        X_train=X_train,
        y_train=y_train_df.reset_index(drop=True),
        groups_train=groups_train.astype(np.int32, copy=False),
        keys_train=keys_train,
        X_test=X_test,
        y_test=y_test_df.reset_index(drop=True),
        groups_test=groups_test.astype(np.int32, copy=False),
        keys_test=keys_test,
        feature_names=feat_cols,
        obs_to_int=dict(meta["obs_to_int"]),
        held_out_obs_ids=held_out,
    )


def iter_loio_folds(
    scheme: str = "loio_9fold",
    *,
    scale_idx: int | None = None,
    dataset_dir: Path | str | None = None,
) -> Iterator[Fold]:
    """Yield every fold of a scheme in order. Matches PLAN_modeling.md §5."""
    for k in range(n_folds(scheme, dataset_dir)):
        yield load_fold(scheme, k, scale_idx=scale_idx, dataset_dir=dataset_dir)


# ============================================================================
# Context patches (CNN input)
# ============================================================================


def _patches_dir(dataset_dir: Path | str | None = None) -> Path:
    base = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    return base / "context_patches"


def load_context_patch_stack(
    obs_id: str,
    patch_size_px: int,
    *,
    dataset_dir: Path | str | None = None,
) -> np.ndarray:
    """Load a single (n_tiles, S, S) uint8 stack written by Stage 4b."""
    p = _patches_dir(dataset_dir) / f"{obs_id}_S{patch_size_px}.npy"
    return np.load(p)


def gather_patches(
    keys: pd.DataFrame,
    patch_size_px: int,
    *,
    dataset_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a (n_valid, S, S) uint8 batch + the row indices it covers.

    `keys` must carry columns `obs_id` and `patch_idx_S{S}`. Rows with
    `patch_idx_S{S} == -1` (window-edge margin) are dropped and not returned;
    the second return value is the surviving row indices into `keys`.
    """
    col = f"patch_idx_S{patch_size_px}"
    if col not in keys.columns:
        raise KeyError(f"keys is missing required column {col!r}")
    patch_idx_all = keys[col].to_numpy()
    valid_mask = patch_idx_all >= 0
    valid_rows = np.where(valid_mask)[0]
    out = np.empty((valid_rows.size, patch_size_px, patch_size_px), dtype=np.uint8)
    # Index into `out` at the original row order of `valid_keys`; group by obs_id
    # only to amortise the per-stack mmap open (cheap, but still nicer once per file).
    obs_arr = keys["obs_id"].to_numpy()[valid_mask]
    patch_idx_valid = patch_idx_all[valid_mask]
    for obs in pd.unique(obs_arr):
        obs_mask = obs_arr == obs
        stack = np.load(_patches_dir(dataset_dir) / f"{obs}_S{patch_size_px}.npy", mmap_mode="r")
        out[obs_mask] = stack[patch_idx_valid[obs_mask]]
    return out, valid_rows
