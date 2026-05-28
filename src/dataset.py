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

# Stage 4 special-case ObsId: the empty-truth image (no detections survive Stage 4 filters).
# Excluded from within-image CV by default because the within-image experiment needs a
# non-empty test quadrant to be diagnostic.
EMPTY_TRUTH_OBS_ID = "ESP_065711_1545"

# Maps tile_size_px -> integer factor relative to the finest scale. Hard-coded here
# instead of re-derived per call because the within-image partitioner needs the same
# factor convention at every step (per-scale ti_mid is the finest-scale ti_mid // factor;
# strictly coherent quadrant assignment requires the finest ti_mid to be a multiple of
# the coarsest factor). Extend this dict if new scales are added upstream.
SCALE_TO_FACTOR_FROM_FINEST = {8: 1, 16: 2, 32: 4, 64: 8}

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

def _compute_quadrant_definitions(
    obs_id: str,
    labels_dir: Path,
    *,
    finest_scale_px: int = 8,
    scale_to_factor: dict[int, int] | None = None,
) -> dict[str, dict[str, int]]:
    """Compute per-image, per-scale (ti_mid, tj_mid) cuts for the 2x2 quadrant partition.

    Approach (resolves the inconsistency in PLAN_Stage5c.md §3): the finest-scale median
    is *snapped to a multiple of the coarsest factor* and then divided down to each coarser
    scale. This produces a single shared physical cut across all scales -- every S=8 tile
    lands in the same quadrant as its S=16 / S=32 / S=64 parent, exactly. Equivalent to
    "compute the median in S=8 units, snap to multiple of 8, divide by (Sk/S8) to get the
    cut at scale Sk."

    Returns a dict keyed by stringified tile_size_px (matches the metadata JSON layout
    in PLAN_Stage5c.md §4):

        {"8":  {"ti_mid": 1352, "tj_mid": 5184},
         "16": {"ti_mid": 676,  "tj_mid": 2592},
         "32": {"ti_mid": 338,  "tj_mid": 1296},
         "64": {"ti_mid": 169,  "tj_mid": 648}}

    Only scales actually present in this image's labels parquet appear in the dict.
    """
    if scale_to_factor is None:
        scale_to_factor = SCALE_TO_FACTOR_FROM_FINEST
    df = pd.read_parquet(Path(labels_dir) / f"{obs_id}.parquet", columns=["tile_size_px", "ti", "tj"])
    finest = df[df["tile_size_px"] == finest_scale_px]
    if len(finest) == 0:
        raise ValueError(
            f"{obs_id}: no rows at tile_size_px={finest_scale_px}; cannot compute quadrant cuts."
        )
    # Use the integer-valued median directly (each tile's ti/tj is already an integer
    # index into the CTX pixel grid). For an even-sized sample, np.median returns a
    # midpoint which we then snap.
    raw_ti_mid = float(np.median(finest["ti"].to_numpy()))
    raw_tj_mid = float(np.median(finest["tj"].to_numpy()))
    coarsest_factor = max(scale_to_factor.values())
    # Floor-snap to a multiple of the coarsest factor so the cut is integer-exact at the
    # coarsest scale (and therefore exact at every finer scale by division).
    ti_mid_finest = (int(raw_ti_mid) // coarsest_factor) * coarsest_factor
    tj_mid_finest = (int(raw_tj_mid) // coarsest_factor) * coarsest_factor
    present_scales = sorted(set(int(s) for s in df["tile_size_px"].unique()))
    defs: dict[str, dict[str, int]] = {}
    for tile_px in present_scales:
        if tile_px not in scale_to_factor:
            # Unknown scale (e.g. someone added S=128 upstream without updating the
            # factor map) -- skip rather than guess.
            continue
        factor = scale_to_factor[tile_px]
        defs[str(tile_px)] = {
            "ti_mid": int(ti_mid_finest // factor),
            "tj_mid": int(tj_mid_finest // factor),
        }
    return defs


def _quadrant_array_for_image(
    df: pd.DataFrame,
    quadrant_definitions: dict[str, dict[str, int]],
    *,
    buffer_tiles: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row in `df` to a quadrant (0..3) and a keep/drop mask.

    Quadrant predicate: quadrant = 2 * (ti >= ti_mid) + (tj >= tj_mid), where ti_mid/tj_mid
    are looked up per-scale from `quadrant_definitions`. Buffer drops tiles whose
    |ti - ti_mid| < buffer_tiles OR |tj - tj_mid| < buffer_tiles at their scale; buffer=0
    therefore drops nothing, buffer=1 drops exactly the cut-line row/column tiles.

    Returns (quadrant_array, keep_mask) -- both shape (len(df),).
    Rows with `tile_size_px` not in `quadrant_definitions` get quadrant=-1 and keep=False.
    """
    q_arr = np.full(len(df), -1, dtype=np.int32)
    keep = np.zeros(len(df), dtype=bool)
    tile_px_arr = df["tile_size_px"].to_numpy()
    ti_arr = df["ti"].to_numpy()
    tj_arr = df["tj"].to_numpy()
    for tile_px_str, qd in quadrant_definitions.items():
        tile_px = int(tile_px_str)
        sel = tile_px_arr == tile_px
        if not sel.any():
            continue
        ti_mid = int(qd["ti_mid"])
        tj_mid = int(qd["tj_mid"])
        ti_sub = ti_arr[sel]
        tj_sub = tj_arr[sel]
        q_sub = (2 * (ti_sub >= ti_mid).astype(np.int32)) + (tj_sub >= tj_mid).astype(np.int32)
        q_arr[sel] = q_sub
        if buffer_tiles > 0:
            in_buf = (np.abs(ti_sub - ti_mid) < buffer_tiles) | (np.abs(tj_sub - tj_mid) < buffer_tiles)
            keep_sub = ~in_buf
        else:
            keep_sub = np.ones(int(sel.sum()), dtype=bool)
        keep[sel] = keep_sub
    return q_arr, keep


def _within_image_fold_summary(
    obs_id: str,
    quadrant_idx: int,
    quadrant_definitions: dict[str, dict[str, int]],
    inventory: pd.DataFrame,
    labels_dir: Path,
    *,
    buffer_tiles: int,
) -> tuple[dict, dict[str, int], dict[str, int]]:
    """Compute (test_summary, n_test_tiles_per_scale, n_train_tiles_per_scale) for one fold.

    test_summary mirrors the LOIO `_fold_summary` shape but represents one quadrant of
    one image. We retain the per-scale tile counts as a separate dict so analyses can
    inspect per-scale balance directly.
    """
    df = pd.read_parquet(
        Path(labels_dir) / f"{obs_id}.parquet",
        columns=["tile_size_px", "ti", "tj", "fractional_area"],
    )
    q_arr, keep = _quadrant_array_for_image(df, quadrant_definitions, buffer_tiles=buffer_tiles)
    test_mask = (q_arr == quadrant_idx) & keep
    train_mask = (q_arr != quadrant_idx) & (q_arr >= 0) & keep

    finest_px = min(int(s) for s in quadrant_definitions.keys())
    finest_test_mask = test_mask & (df["tile_size_px"].to_numpy() == finest_px)
    frac_mean_finest = (
        float(df.loc[finest_test_mask, "fractional_area"].mean())
        if int(finest_test_mask.sum()) > 0 else 0.0
    )

    label = str(inventory.loc[obs_id, "BoulderLabel"]) if obs_id in inventory.index else ""
    test_summary = {
        "n_images": 1,
        "n_tiles_total": int(test_mask.sum()),
        "n_tiles_finest": int(finest_test_mask.sum()),
        "boulder_labels": {label: 1} if label else {},
        "frac_mean_finest_avg": frac_mean_finest,
    }
    n_test = {s: int(((q_arr == quadrant_idx) & keep & (df["tile_size_px"].to_numpy() == int(s))).sum())
              for s in quadrant_definitions.keys()}
    n_train = {s: int(((q_arr != quadrant_idx) & (q_arr >= 0) & keep & (df["tile_size_px"].to_numpy() == int(s))).sum())
               for s in quadrant_definitions.keys()}
    return test_summary, n_test, n_train


def _assign_within_image_kfold(
    inventory: pd.DataFrame,
    *,
    labels_dir: Path,
    n_folds_per_image: int = 4,
    buffer_tiles: int = 0,
    excluded_obs_ids: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Generate per-(image, quadrant) folds. PLAN_Stage5c.md §3.

    Returns (folds_meta, used_obs_ids):
      - folds_meta: flat list of fold dicts; each carries test_obs_id, test_quadrant,
        quadrant_definitions, plus the LOIO-compatible test_obs_ids/train_obs_ids singleton
        lists so downstream loaders (which key off test_obs_ids) keep working unchanged.
      - used_obs_ids: the manifest list after excluding `excluded_obs_ids`. Used as the
        canonical `manifest_obs_ids` in the split-metadata JSON.

    Only `n_folds_per_image == 4` (2x2) is supported. PLAN_Stage5c.md §11 q1 -- 3x3 is
    a follow-up if reviewers ask for finer-grained per-image variance estimates.
    """
    if n_folds_per_image != 4:
        raise NotImplementedError(
            f"Only n_folds_per_image=4 (2x2 spatial partition) is supported; got {n_folds_per_image}."
        )
    excluded = set(excluded_obs_ids or [])
    used_obs_ids = [o for o in sorted(inventory.index) if o not in excluded]
    folds: list[dict] = []
    fold_idx = 0
    for obs_id in used_obs_ids:
        defs = _compute_quadrant_definitions(obs_id, labels_dir)
        for q in range(n_folds_per_image):
            test_summary, n_test, n_train = _within_image_fold_summary(
                obs_id, q, defs, inventory, labels_dir, buffer_tiles=buffer_tiles,
            )
            folds.append({
                "fold_idx": fold_idx,
                "test_obs_id": obs_id,
                "test_quadrant": int(q),
                "quadrant_definitions": defs,
                "n_test_tiles_per_scale": n_test,
                "n_train_tiles_per_scale": n_train,
                # LOIO-compatible shape: a fold's "test_obs_ids" is the singleton list of
                # the image being tested; "train_obs_ids" is the same singleton (training
                # data is the OTHER three quadrants of the SAME image). Downstream code
                # that reads test_obs_ids[0] (e.g. loaders.load_fold -> held_out_obs_ids)
                # continues to work.
                "test_obs_ids": [obs_id],
                "train_obs_ids": [obs_id],
                "test_summary": test_summary,
                "train_summary": test_summary,  # whole-image stats; per-quadrant detail in n_*_tiles_per_scale
            })
            fold_idx += 1
    return folds, used_obs_ids


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
    keys = (
        "name", "kind", "n_folds", "stratification", "manifest_obs_ids", "folds",
        # within-image specific:
        "n_folds_per_image", "buffer_tiles", "excluded_obs_ids",
    )
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
    # within-image-only kwargs (ignored by LOIO / balanced k-fold paths):
    labels_dir: Path | None = None,
    n_folds_per_image: int = 4,
    buffer_tiles: int = 0,
    excluded_obs_ids: list[str] | None = None,
) -> dict:
    """Return the split-metadata dict; doesn't write anything to disk.

    Caller is responsible for writing the JSON via `write_split_metadata`. Splitting
    this in two so tests can build a split in-memory without touching the filesystem.

    Stratification dispatch:
      - 'none' -> true LOIO, one image per fold (kind='leave-image-out').
      - 'boulder_label_size_balanced' -> greedy size-balanced k-fold (kind='leave-image-out').
      - 'within_image' -> 2x2 spatial quadrant per image, n_folds = n_images * n_folds_per_image
        (kind='within-image'). Requires `labels_dir`. PLAN_Stage5c.md.
    """
    obs_ids = sorted(inventory.index)
    if stratification == "within_image":
        if labels_dir is None:
            raise ValueError(
                f"Scheme {name!r}: stratification='within_image' requires the labels_dir kwarg."
            )
        folds_meta, used_obs_ids = _assign_within_image_kfold(
            inventory, labels_dir=labels_dir,
            n_folds_per_image=n_folds_per_image,
            buffer_tiles=buffer_tiles,
            excluded_obs_ids=excluded_obs_ids,
        )
        expected_n_folds = len(used_obs_ids) * n_folds_per_image
        if n_folds != expected_n_folds:
            raise ValueError(
                f"Scheme {name!r}: stratification='within_image' with {len(used_obs_ids)} "
                f"non-excluded images and n_folds_per_image={n_folds_per_image} expects "
                f"n_folds={expected_n_folds}; got {n_folds}."
            )
        metadata = {
            "name": name,
            "kind": "within-image",
            "n_folds": len(folds_meta),
            "stratification": stratification,
            "seed": int(seed),
            "n_folds_per_image": int(n_folds_per_image),
            "buffer_tiles": int(buffer_tiles),
            "manifest_obs_ids": used_obs_ids,
            "excluded_obs_ids": sorted(excluded_obs_ids or []),
            "folds": folds_meta,
            "config_hash": config_hash,
        }
        metadata["split_hash"] = _split_metadata_hash(metadata)
        metadata["written_at_iso"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        return metadata

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
            f"Supported: 'none', 'boulder_label_size_balanced', 'within_image'."
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

    Dispatches on `metadata['kind']`: 'leave-image-out' uses the standard per-ObsId
    packaging; 'within-image' partitions each image's tiles into per-quadrant folds
    (PLAN_Stage5c.md). Both produce the same per-fold parquet layout, so downstream
    loaders work unchanged.
    """
    if metadata.get("kind") == "within-image":
        return _package_within_image_split(
            metadata,
            labels_dir=labels_dir, features_dir=features_dir,
            output_dir=output_dir, scale_filter=scale_filter,
            emit_all_parquet=emit_all_parquet, config_hash=config_hash,
        )
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


# ============================================================================
# Within-image packaging (Stage 5c -- spatial quadrant CV)
# ============================================================================

def _package_within_image_split(
    metadata: dict,
    *,
    labels_dir: Path,
    features_dir: Path | None,
    output_dir: Path,
    scale_filter: list[int] | None = None,
    emit_all_parquet: bool = True,
    config_hash: str,
) -> dict:
    """Per-(image, quadrant) packaging counterpart of `package_split`.

    Train rows for fold k are the OTHER three quadrants of the SAME image as fold k's
    test quadrant. Groups arrays store the per-row quadrant index (0..3) -- this lets the
    LOIO inner-validation rotation in `src.modeling.evaluate.run_loio` cycle through
    the 3 training quadrants for early-stopping without collision with the held-out one
    (the inner-val code is `unique_train[fold_idx % n_unique]`, which here picks one of
    the 3 non-test quadrants).
    """
    name = metadata["name"]
    out_dir = Path(output_dir) / PACKAGED_SUBDIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image: dict[str, pd.DataFrame] = {}
    for obs in metadata["manifest_obs_ids"]:
        per_image[obs] = _join_one_image(
            obs, labels_dir=labels_dir, features_dir=features_dir,
            scale_filter=scale_filter,
        )

    obs_to_int = {obs: i for i, obs in enumerate(metadata["manifest_obs_ids"])}
    buffer_tiles = int(metadata.get("buffer_tiles", 0))

    per_fold_counts: list[dict] = []
    for fold in metadata["folds"]:
        k = int(fold["fold_idx"])
        obs = fold["test_obs_id"]
        test_quadrant = int(fold["test_quadrant"])
        defs = fold["quadrant_definitions"]
        df = per_image[obs]
        q_arr, keep = _quadrant_array_for_image(df, defs, buffer_tiles=buffer_tiles)
        test_mask_arr = (q_arr == test_quadrant) & keep
        train_mask_arr = (q_arr != test_quadrant) & (q_arr >= 0) & keep
        train_df = df[train_mask_arr].reset_index(drop=True)
        test_df = df[test_mask_arr].reset_index(drop=True)
        train_q = q_arr[train_mask_arr].astype(np.int32, copy=False)
        test_q = q_arr[test_mask_arr].astype(np.int32, copy=False)

        x_cols, y_cols = _split_columns(train_df)
        x_keep = TILE_KEY_COLUMNS + x_cols
        y_keep = TILE_KEY_COLUMNS + y_cols

        train_df[x_keep].to_parquet(out_dir / f"X_train_fold{k}.parquet", index=False)
        train_df[y_keep].to_parquet(out_dir / f"y_train_fold{k}.parquet", index=False)
        test_df[x_keep].to_parquet(out_dir / f"X_test_fold{k}.parquet", index=False)
        test_df[y_keep].to_parquet(out_dir / f"y_test_fold{k}.parquet", index=False)
        np.save(out_dir / f"groups_train_fold{k}.npy", train_q)
        np.save(out_dir / f"groups_test_fold{k}.npy", test_q)

        per_fold_counts.append({
            "fold_idx": k,
            "n_train_tiles": int(len(train_df)),
            "n_test_tiles": int(len(test_df)),
            "n_train_x_cols": len(x_cols),
            "n_y_cols": len(y_cols),
            "test_obs_ids": [obs],
            "test_obs_id": obs,
            "test_quadrant": test_quadrant,
        })

    all_path: str | None = None
    if emit_all_parquet:
        # Every (image, quadrant) tile appears exactly once -- tagged with its test fold_idx
        # and its quadrant index. Reconstruct per-fold sets by `fold_idx == k` for test or
        # `test_obs_id == fold.test_obs_id & quadrant != fold.test_quadrant` for train.
        all_rows = []
        for fold in metadata["folds"]:
            k = int(fold["fold_idx"])
            obs = fold["test_obs_id"]
            test_quadrant = int(fold["test_quadrant"])
            df = per_image[obs]
            q_arr, keep = _quadrant_array_for_image(df, fold["quadrant_definitions"], buffer_tiles=buffer_tiles)
            sel = (q_arr == test_quadrant) & keep
            sub = df[sel].copy()
            sub["fold_idx"] = k
            sub["test_quadrant"] = test_quadrant
            all_rows.append(sub)
        all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        all_path = str(out_dir / "all.parquet")
        all_df.to_parquet(all_path, index=False)

    package_meta = {
        "name": name,
        "kind": "within-image",
        "split_hash": metadata["split_hash"],
        "config_hash": config_hash,
        "scale_filter": list(scale_filter) if scale_filter is not None else None,
        "emit_all_parquet": bool(emit_all_parquet),
        "obs_to_int": obs_to_int,
        "n_folds_per_image": int(metadata.get("n_folds_per_image", 4)),
        "buffer_tiles": buffer_tiles,
        "per_fold": per_fold_counts,
        "all_parquet_path": all_path,
        "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(package_meta, indent=2), encoding="utf-8")
    return package_meta
