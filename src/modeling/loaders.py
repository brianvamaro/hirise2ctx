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
# Per-image feature standardization (W1 next-bet 1)
# ============================================================================

PER_IMAGE_TRANSFORMS = ("rank", "zscore", "robust")


def _standardize_matrix_per_group(X: np.ndarray, groups: np.ndarray, method: str) -> np.ndarray:
    """Standardize each feature column WITHIN each group (image) independently.

    Targets the W1 `distribution_shift` failure class (DECISIONS.md 2026-06-10):
    images whose within-image texture->label relationship is strong and
    cohort-consistent but whose absolute feature values sit outside the training
    distribution (photometric/source/latitude shift). Inference-compatible by
    construction: the test image's own tile population supplies the statistics,
    exactly as a CTX inference window would.

      rank   -> fractional rank in [0, 1] within the image (average ties)
      zscore -> (x - mean) / std within the image
      robust -> (x - median) / IQR within the image

    Constant columns within a group map to 0.0 (rank: 0.5 from tie-averaging is
    re-centred to 0.5 only for rank -- a constant column carries no ordering
    information either way).
    """
    if method not in PER_IMAGE_TRANSFORMS:
        raise ValueError(f"unknown per-image transform {method!r}; pick from {PER_IMAGE_TRANSFORMS}")
    from scipy.stats import rankdata

    out = np.empty_like(X, dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        block = X[m].astype(np.float64)
        if method == "rank":
            n = block.shape[0]
            if n == 1:
                out[m] = 0.5
                continue
            ranks = np.apply_along_axis(rankdata, 0, block)
            out[m] = ((ranks - 1) / (n - 1)).astype(np.float32)
        elif method == "zscore":
            mu = block.mean(axis=0)
            sd = block.std(axis=0)
            sd[sd == 0] = 1.0
            out[m] = ((block - mu) / sd).astype(np.float32)
        else:  # robust
            med = np.median(block, axis=0)
            iqr = np.percentile(block, 75, axis=0) - np.percentile(block, 25, axis=0)
            iqr[iqr == 0] = 1.0
            out[m] = ((block - med) / iqr).astype(np.float32)
    return out


def standardize_fold_per_image(fold: Fold, method: str) -> Fold:
    """Return a copy of `fold` with X_train and X_test standardized per image.

    Train images use their own training rows; the held-out image uses its own
    test rows -- no statistics cross the split boundary.
    """
    from dataclasses import replace

    return replace(
        fold,
        X_train=_standardize_matrix_per_group(fold.X_train, fold.groups_train, method),
        X_test=_standardize_matrix_per_group(fold.X_test, fold.groups_test, method),
    )


def augment_fold_with_per_image(fold: Fold, method: str) -> Fold:
    """Return a copy of `fold` with per-image-standardized copies of every feature
    CONCATENATED to the raw features (width doubles).

    Rationale (bet-1 sweep, DECISIONS.md 2026-06-11): pure per-image transforms
    rescue the distribution_shift images but hurt images where absolute feature
    values carry signal -- giving the GBM both views lets it choose per split.
    """
    from dataclasses import replace

    std_train = _standardize_matrix_per_group(fold.X_train, fold.groups_train, method)
    std_test = _standardize_matrix_per_group(fold.X_test, fold.groups_test, method)
    return replace(
        fold,
        X_train=np.concatenate([fold.X_train, std_train], axis=1),
        X_test=np.concatenate([fold.X_test, std_test], axis=1),
        feature_names=list(fold.feature_names) + [f"pistd_{method}_{n}" for n in fold.feature_names],
    )


# ============================================================================
# Fang-ViT embeddings as an optional feature source (PLAN_FM §2.2)
# ============================================================================
#
# The frozen recipe (DECISIONS.md 2026-06-12) is emb-only: the LOIO harness
# rebuilds X from the cached embedding store rather than from the packaged X
# matrix. This is the numpy-only join half (the torch extraction that *writes*
# the store lives in `src.fm_embeddings`). Store layout, written per image:
#   {dataset_dir}/fang_embeddings/{obs_id}_P{px}.npz  with arrays
#   ti, tj (int32), valid (bool), cls/mean/gem (n, 768) float32.
# px encodes the INPUT size: P96 = the S=32 3×3-context input (the frozen one),
# P192 = S=64 3×3-context, P32/P64 = the own-tile inputs.

EMBED_DIM = 768


def _fang_dir(dataset_dir: Path | str | None = None) -> Path:
    base = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    return base / "fang_embeddings"


def load_fang_store(
    px: int,
    *,
    pool: str = "gem",
    dataset_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the cached embedding store for one input size into a keyed matrix.

    Returns ``(index, matrix)``: `index` is a DataFrame with columns
    `obs_id, ti, tj, row` (row = position into `matrix`), `matrix` is
    (n_total, 768) float32 with NaN rows where `valid` is False (window-margin
    tiles whose context box was incomplete). Concatenates every
    `*_P{px}.npz` under the store, sorted by filename for a stable row order.
    """
    if pool not in ("cls", "mean", "gem"):
        raise ValueError(f"unknown pool {pool!r}; pick from cls/mean/gem")
    fdir = _fang_dir(dataset_dir)
    files = sorted(fdir.glob(f"*_P{px}.npz"))
    if not files:
        raise FileNotFoundError(f"no Fang embedding store *_P{px}.npz under {fdir}")
    blocks, rows = [], []
    for f in files:
        z = np.load(f)
        obs = f.name[: -len(f"_P{px}.npz")]
        emb = z[pool].astype(np.float32, copy=True)
        emb[~z["valid"].astype(bool)] = np.nan
        blocks.append(emb)
        rows.append(pd.DataFrame({"obs_id": obs, "ti": z["ti"], "tj": z["tj"]}))
    matrix = np.concatenate(blocks, axis=0)
    index = pd.concat(rows, ignore_index=True)
    index["row"] = np.arange(len(index), dtype=np.int64)
    return index, matrix


def fang_columns_for_keys(
    keys: pd.DataFrame,
    px: int,
    *,
    pool: str = "gem",
    dataset_dir: Path | str | None = None,
    store: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Look up the (n_keys, 768) embedding block + column names for `keys`.

    `keys` must carry `obs_id, ti, tj`; the join is one-to-one and asserts no
    miss (every requested tile must exist in the store). Pass a prebuilt `store`
    (from `load_fang_store`) to avoid re-reading the npz across many folds.
    """
    index, matrix = store if store is not None else load_fang_store(px, pool=pool, dataset_dir=dataset_dir)
    j = keys[["obs_id", "ti", "tj"]].merge(index, on=["obs_id", "ti", "tj"],
                                           how="left", validate="one_to_one")
    r = j["row"].to_numpy()
    assert not np.isnan(r).any(), "Fang store is missing tiles present in the keys"
    names = [f"fang_{pool}{px}_{i:03d}" for i in range(EMBED_DIM)]
    return matrix[r.astype(np.int64)], names


def augment_fold_with_fang(
    fold: Fold,
    *,
    px: int,
    pool: str = "gem",
    dataset_dir: Path | str | None = None,
    replace: bool = False,
    store: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> Fold:
    """Return a copy of `fold` with Fang embedding columns joined onto X.

    `replace=False` concatenates the 768 embedding columns after the existing
    features (the t1+emb matrix); `replace=True` swaps X for the embeddings alone
    (the frozen emb-only recipe). Train/test are looked up from their own keys, so
    no statistics cross the split boundary. NaN margin rows are preserved for the
    head to impute (the bake-off MLP/LightGBM both handle NaN).
    """
    from dataclasses import replace as dc_replace

    if store is None:
        store = load_fang_store(px, pool=pool, dataset_dir=dataset_dir)
    emb_train, names = fang_columns_for_keys(fold.keys_train, px, pool=pool,
                                             dataset_dir=dataset_dir, store=store)
    emb_test, _ = fang_columns_for_keys(fold.keys_test, px, pool=pool,
                                        dataset_dir=dataset_dir, store=store)
    if replace:
        return dc_replace(fold, X_train=emb_train, X_test=emb_test, feature_names=names)
    return dc_replace(
        fold,
        X_train=np.concatenate([fold.X_train, emb_train], axis=1),
        X_test=np.concatenate([fold.X_test, emb_test], axis=1),
        feature_names=list(fold.feature_names) + names,
    )


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
