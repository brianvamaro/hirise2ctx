"""Stage 5 -- leave-image-out splits + dataset packaging.

Per CLAUDE.md Section 4 acceptance #5: splits are by image (group-aware), never random
per-tile. Tiles within an image share illumination, surface composition, and BoulderNet
detector behaviour, so a random-tile split leaks the per-image background into the test
fold and inflates every downstream metric.

Decisions locked in via AskUserQuestion 2026-05-24 (see DECISIONS.md):

- **Schemes shipped:** `loio_9fold` (true leave-one-image-out, one ObsId per test fold)
  primary + `loio_3fold_balanced` (3 image-balanced test folds) secondary.
- **ESP_065711_1545** (empty-shapefile, 25k true-zero tiles) is included in the splits
  with `BoulderLabel='unknown'`. Its zero-frac tiles are genuine "boulder absent"
  signal (HiRISE-covered, no detections), not measurement noise.
- **Consolidated `all.parquet`** emitted per scheme alongside the per-fold parquets --
  saves repeated joins for ad-hoc analysis.
- **Two loading paths:** `package_split` materialises per-fold parquets (default path
  for the current 9-image manifest); `iter_train_batches`/`iter_test_batches` yield
  per-ObsId batches without materialising the whole dataset (forward-looking pattern
  for the 50-200+ image case per PLAN_Stage5.md §11b).

ESP_057469_2215 is excluded from the priority10 sweep upstream (Stage 4, see
DECISIONS.md 2026-05-22 tile-straddle entry); the splitter automatically ignores any
ObsId without a `dataset/labels/{ObsId}.parquet` file, so it never appears here either.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

SPLITS_SUBDIR = "splits"
PACKAGED_SUBDIR = "packaged"
LABELS_SUBDIR = "labels"     # matches src.labeling.LABELS_SUBDIR; duplicated to avoid the import cycle
FEATURES_SUBDIR = "features"  # matches src.features.FEATURES_SUBDIR

# Tile-key columns identify a single tile uniquely. All other columns split into the
# X (features) or y (labels) side.
TILE_KEY_COLUMNS = ["obs_id", "scale_idx", "tile_size_px", "ti", "tj"]

# Label-side columns -- everything we'd want to predict, plus per-tile context useful
# for analysis. Anything not in this list and not a tile-key column is treated as X.
LABEL_COLUMNS = [
    "boulder_area", "boulder_count", "tile_area",
    "fractional_area", "binary_by_area", "binary_by_count", "count_density",
    # `categorical` is emitted only when `labeling.categorical_bins` is non-empty; we
    # tolerate its absence by intersecting against the actual dataframe columns at
    # package time.
    "categorical",
]

# Per-tile bound columns we keep on the labels side for downstream analysis (plotting
# heatmaps requires xmin/ymin/xmax/ymax, but they're not features).
LABEL_CONTEXT_COLUMNS = ["xmin", "ymin", "xmax", "ymax", "tile_size_m"]


# ============================================================================
# Per-image inventory (cheap to build; drives both splitting and packaging)
# ============================================================================

def discover_obs_ids(labels_dir: Path) -> list[str]:
    """Return all ObsIds that have a Stage 4 labels parquet on disk, sorted by name."""
    labels_dir = Path(labels_dir)
    paths = sorted(labels_dir.glob("*.parquet"))
    return [p.stem for p in paths]


def build_image_inventory(
    obs_ids: list[str],
    manifest: pd.DataFrame,
    labels_dir: Path,
) -> pd.DataFrame:
    """Build a per-image inventory dataframe for stratification.

    Columns: ObsId, BoulderLabel (from manifest), n_tiles_total, n_tiles_per_scale (dict),
    frac_mean_S8 (float, finest-scale mean of fractional_area), n_polys_after_filter
    (from Stage 4 sidecar).
    """
    labels_dir = Path(labels_dir)
    rows = []
    manifest = manifest.set_index("ObsId")
    for obs in obs_ids:
        parquet = labels_dir / f"{obs}.parquet"
        sidecar = labels_dir / f"{obs}.json"
        df = pd.read_parquet(parquet, columns=["tile_size_px", "fractional_area"])
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        finest = int(df["tile_size_px"].min())
        finest_rows = df[df["tile_size_px"] == finest]
        rows.append({
            "ObsId": obs,
            "BoulderLabel": str(manifest.loc[obs, "BoulderLabel"]),
            "n_tiles_total": int(len(df)),
            "n_tiles_finest": int(len(finest_rows)),
            "frac_mean_finest": float(finest_rows["fractional_area"].mean()) if len(finest_rows) else 0.0,
            "n_polys_after_filter": int(prov.get("n_polygons_after_filter", 0)),
        })
    return pd.DataFrame(rows).set_index("ObsId")


# ============================================================================
# Split construction
# ============================================================================

def _assign_loio_9fold(inventory: pd.DataFrame, seed: int) -> list[list[str]]:
    """True leave-one-image-out: each ObsId is the test set in exactly one fold.

    Order is deterministic (sorted by ObsId) so the same inventory always produces the
    same fold structure. `seed` is recorded in metadata but unused here -- LOIO has no
    randomness.
    """
    return [[obs] for obs in sorted(inventory.index)]


def _assign_size_balanced_kfold(
    inventory: pd.DataFrame, n_folds: int, seed: int,
) -> list[list[str]]:
    """Greedy size-balanced k-fold:
    1. Group ObsIds by `BoulderLabel`.
    2. Within each label group, shuffle deterministically with `seed`.
    3. Assign each ObsId to the test fold that currently has the fewest images in its
       own label group, breaking ties by smallest overall fold size, then by fold_idx.

    This produces folds whose sizes differ by at most 1 (when the total isn't divisible
    by n_folds), with the *label composition* in each fold close to the overall ratios
    given the integer constraint. With our 9-image manifest (5 rich, 2 poor, 2 unknown)
    and k=3, the per-fold sizes come out to 3/3/3.
    """
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    # Count per-label assignments per fold for the tiebreaker.
    label_counts: dict[str, list[int]] = {}

    # Iterate labels by descending group size so the biggest group gets the cleanest
    # distribution. Within each group, deterministic shuffle.
    groups = inventory.groupby("BoulderLabel").groups  # dict of label -> Index of ObsIds
    label_order = sorted(groups.keys(), key=lambda lab: (-len(groups[lab]), lab))
    for label in label_order:
        label_counts[label] = [0] * n_folds
        members = sorted(groups[label])  # deterministic initial order
        order = rng.permutation(len(members))
        for idx in order:
            obs = members[int(idx)]
            # Choose the fold with min label_counts[label]; tiebreaker = overall fold size;
            # final tiebreaker = fold_idx.
            best = min(
                range(n_folds),
                key=lambda k: (label_counts[label][k], len(folds[k]), k),
            )
            folds[best].append(obs)
            label_counts[label][best] += 1
    # Sort each fold's ObsIds for stable output.
    return [sorted(f) for f in folds]


def _fold_summary(
    test_obs_ids: list[str],
    inventory: pd.DataFrame,
) -> dict[str, Any]:
    """Summary stats for one fold's test set."""
    sub = inventory.loc[test_obs_ids]
    return {
        "n_images": int(len(sub)),
        "n_tiles_total": int(sub["n_tiles_total"].sum()),
        "n_tiles_finest": int(sub["n_tiles_finest"].sum()),
        "boulder_labels": {
            label: int((sub["BoulderLabel"] == label).sum())
            for label in sorted(sub["BoulderLabel"].unique())
        },
        "frac_mean_finest_avg": float(sub["frac_mean_finest"].mean()) if len(sub) else 0.0,
    }


def _split_metadata_hash(meta: dict) -> str:
    """Stable hash of the split-defining fields. Used as provenance; excludes timestamp."""
    keys = ("name", "kind", "n_folds", "stratification", "manifest_obs_ids", "folds")
    canonical = json.dumps(
        {k: meta[k] for k in keys if k in meta},
        sort_keys=True, default=str, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_split(
    *,
    name: str,
    n_folds: int,
    stratification: str,
    seed: int = 0,
    inventory: pd.DataFrame,
    config_hash: str,
) -> dict:
    """Return the split-metadata dict; doesn't write anything to disk.

    Caller is responsible for writing the JSON via `write_split_metadata`. Splitting
    this in two so tests can build a split in-memory without touching the filesystem.
    """
    obs_ids = sorted(inventory.index)
    if stratification == "none":
        if n_folds != len(obs_ids):
            # n_folds != n_images with no stratification means we'd be doing arbitrary
            # round-robin -- caller probably wanted "boulder_label_size_balanced".
            raise ValueError(
                f"Scheme {name!r}: stratification='none' requires n_folds == n_images "
                f"({len(obs_ids)}); got n_folds={n_folds}."
            )
        test_folds = _assign_loio_9fold(inventory, seed=seed)
    elif stratification == "boulder_label_size_balanced":
        test_folds = _assign_size_balanced_kfold(inventory, n_folds=n_folds, seed=seed)
    else:
        raise ValueError(
            f"Scheme {name!r}: unknown stratification {stratification!r}. "
            f"Supported: 'none', 'boulder_label_size_balanced'."
        )

    folds_meta = []
    all_obs_set = set(obs_ids)
    for fold_idx, test in enumerate(test_folds):
        train = sorted(all_obs_set - set(test))
        folds_meta.append({
            "fold_idx": fold_idx,
            "test_obs_ids": sorted(test),
            "train_obs_ids": train,
            "test_summary": _fold_summary(test, inventory),
            "train_summary": _fold_summary(train, inventory),
        })

    metadata = {
        "name": name,
        "kind": "leave-image-out",
        "n_folds": n_folds,
        "stratification": stratification,
        "seed": int(seed),
        "manifest_obs_ids": obs_ids,
        "folds": folds_meta,
        "config_hash": config_hash,
    }
    metadata["split_hash"] = _split_metadata_hash(metadata)
    metadata["written_at_iso"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return metadata


def write_split_metadata(metadata: dict, output_dir: Path) -> Path:
    """Write the split JSON to `output_dir/splits/{name}.json`. Returns the path."""
    output_dir = Path(output_dir)
    splits_dir = output_dir / SPLITS_SUBDIR
    splits_dir.mkdir(parents=True, exist_ok=True)
    path = splits_dir / f"{metadata['name']}.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_split_metadata(name: str, output_dir: Path) -> dict:
    """Read back a split-metadata JSON."""
    path = Path(output_dir) / SPLITS_SUBDIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================================
# Streaming iterator path (per-ObsId batches; doesn't materialise the dataset)
# ============================================================================

def _join_one_image(
    obs_id: str, *, labels_dir: Path, features_dir: Path | None, scale_filter: list[int] | None,
) -> pd.DataFrame:
    """Load + join labels (always) and features (if present) for one ObsId.

    Filters by `scale_filter` (list of `tile_size_px` to keep) if given.
    """
    labels = pd.read_parquet(Path(labels_dir) / f"{obs_id}.parquet")
    if features_dir is not None:
        feat_path = Path(features_dir) / f"{obs_id}.parquet"
        if feat_path.exists():
            features = pd.read_parquet(feat_path)
            labels = labels.merge(features, on=TILE_KEY_COLUMNS, suffixes=("", "_feat"))
    if scale_filter is not None:
        labels = labels[labels["tile_size_px"].isin(scale_filter)].reset_index(drop=True)
    return labels


def iter_train_batches(
    metadata: dict,
    fold_idx: int,
    *,
    labels_dir: Path,
    features_dir: Path | None = None,
    scale_filter: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per training ObsId for the given fold.

    Use this when the full materialised dataset would be too large to hold in memory --
    PLAN_Stage5.md §11b's pattern for the 50-200+ image case. At our current 9-image
    manifest the materialised dataset fits in ~500 MB and the streaming path adds I/O
    overhead, so prefer `package_split` for the current sweep. Documented as the
    forward-looking alternative.
    """
    for obs in metadata["folds"][fold_idx]["train_obs_ids"]:
        yield _join_one_image(
            obs, labels_dir=labels_dir, features_dir=features_dir,
            scale_filter=scale_filter,
        )


def iter_test_batches(
    metadata: dict,
    fold_idx: int,
    *,
    labels_dir: Path,
    features_dir: Path | None = None,
    scale_filter: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    """Counterpart of `iter_train_batches`, for the test split."""
    for obs in metadata["folds"][fold_idx]["test_obs_ids"]:
        yield _join_one_image(
            obs, labels_dir=labels_dir, features_dir=features_dir,
            scale_filter=scale_filter,
        )


# ============================================================================
# In-memory packaging path (default for 9-image manifest)
# ============================================================================

def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (X_cols, y_cols) given a joined dataframe.

    X columns = features (everything not in TILE_KEY_COLUMNS, LABEL_COLUMNS,
    LABEL_CONTEXT_COLUMNS, or special provenance columns).
    y columns = LABEL_COLUMNS that exist in the frame, plus tile-context.
    """
    label_cols = [c for c in LABEL_COLUMNS if c in df.columns]
    context_cols = [c for c in LABEL_CONTEXT_COLUMNS if c in df.columns]
    excluded = set(TILE_KEY_COLUMNS) | set(label_cols) | set(context_cols) | {"config_hash"}
    x_cols = [c for c in df.columns if c not in excluded]
    return x_cols, label_cols + context_cols


def package_split(
    metadata: dict,
    *,
    labels_dir: Path,
    features_dir: Path | None,
    output_dir: Path,
    scale_filter: list[int] | None = None,
    emit_all_parquet: bool = True,
    config_hash: str,
) -> dict:
    """Materialise per-fold X/y parquets + (optional) consolidated `all.parquet`.

    Output layout:
        dataset/packaged/{name}/
          X_train_fold{k}.parquet, y_train_fold{k}.parquet,
          X_test_fold{k}.parquet,  y_test_fold{k}.parquet,
          groups_train_fold{k}.npy, groups_test_fold{k}.npy,
          all.parquet  (when emit_all_parquet=True)
          metadata.json  (provenance, scale_filter, feature/label config hashes)

    The in-memory concat is fine at 9 images (~500 MB total joined dataframe). At 50+
    images, switch to the streaming iterator pattern (see PLAN_Stage5.md §11b).
    """
    name = metadata["name"]
    out_dir = Path(output_dir) / PACKAGED_SUBDIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + join everything once, attach obs_id-based fold/split membership later.
    per_image: dict[str, pd.DataFrame] = {}
    for obs in metadata["manifest_obs_ids"]:
        per_image[obs] = _join_one_image(
            obs, labels_dir=labels_dir, features_dir=features_dir,
            scale_filter=scale_filter,
        )

    # ObsId -> integer code, for the groups_{}.npy files.
    obs_to_int = {obs: i for i, obs in enumerate(metadata["manifest_obs_ids"])}

    # Per-fold materialisation.
    per_fold_counts: list[dict] = []
    for fold in metadata["folds"]:
        k = fold["fold_idx"]
        test_obs = fold["test_obs_ids"]
        train_obs = fold["train_obs_ids"]
        train_df = pd.concat([per_image[o] for o in train_obs], ignore_index=True) if train_obs else pd.DataFrame()
        test_df = pd.concat([per_image[o] for o in test_obs], ignore_index=True) if test_obs else pd.DataFrame()

        x_cols, y_cols = _split_columns(train_df)
        # Always include the tile-key columns on both sides for downstream joins.
        x_keep = TILE_KEY_COLUMNS + x_cols
        y_keep = TILE_KEY_COLUMNS + y_cols

        train_df[x_keep].to_parquet(out_dir / f"X_train_fold{k}.parquet", index=False)
        train_df[y_keep].to_parquet(out_dir / f"y_train_fold{k}.parquet", index=False)
        test_df[x_keep].to_parquet(out_dir / f"X_test_fold{k}.parquet", index=False)
        test_df[y_keep].to_parquet(out_dir / f"y_test_fold{k}.parquet", index=False)

        train_groups = np.asarray([obs_to_int[o] for o in train_df["obs_id"]], dtype=np.int32)
        test_groups = np.asarray([obs_to_int[o] for o in test_df["obs_id"]], dtype=np.int32)
        np.save(out_dir / f"groups_train_fold{k}.npy", train_groups)
        np.save(out_dir / f"groups_test_fold{k}.npy", test_groups)

        per_fold_counts.append({
            "fold_idx": k,
            "n_train_tiles": int(len(train_df)),
            "n_test_tiles": int(len(test_df)),
            "n_train_x_cols": len(x_cols),
            "n_y_cols": len(y_cols),
            "test_obs_ids": list(test_obs),
        })

    # Optional consolidated all.parquet: every row tagged with fold_idx + split.
    all_path: str | None = None
    if emit_all_parquet:
        # Each tile appears in exactly one test fold (LOIO/balanced k-fold), and in
        # `n_folds - 1` train folds. To avoid duplicating rows, the `all.parquet` view
        # uses a single fold_idx that's the test fold the tile belongs to, and the
        # split column is always 'test' (so 'all' here means "the union of test folds
        # across the scheme"). Reconstructing per-fold train sets from this is an
        # `obs_id != test_obs_ids[fold_idx]` filter -- which is what the per-fold
        # parquets already give you. The consolidated parquet is for ad-hoc analysis
        # like "show me every tile + its fold-idx + label", not for training loops.
        all_rows = []
        for fold in metadata["folds"]:
            for obs in fold["test_obs_ids"]:
                df = per_image[obs].copy()
                df["fold_idx"] = int(fold["fold_idx"])
                all_rows.append(df)
        all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        all_path = str(out_dir / "all.parquet")
        all_df.to_parquet(all_path, index=False)

    # Provenance metadata next to the parquets.
    package_meta = {
        "name": name,
        "split_hash": metadata["split_hash"],
        "config_hash": config_hash,
        "scale_filter": list(scale_filter) if scale_filter is not None else None,
        "emit_all_parquet": bool(emit_all_parquet),
        "obs_to_int": obs_to_int,
        "per_fold": per_fold_counts,
        "all_parquet_path": all_path,
        "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(package_meta, indent=2), encoding="utf-8")
    return package_meta


def load_package_metadata(name: str, output_dir: Path) -> dict:
    """Read the packaging-side metadata.json for a scheme."""
    return json.loads(
        (Path(output_dir) / PACKAGED_SUBDIR / name / "metadata.json").read_text(encoding="utf-8")
    )
