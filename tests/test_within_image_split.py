"""Stage 5c (within-image k-fold spatial split) tests.

Unit tests use synthetic per-image label parquets in tmp_path -- no real Stage 4
artifacts required. One slow integration test against the real `dataset/labels/`
parquets confirms the priority10 manifest produces 8x4 = 32 folds.

The within-image scheme guarantees an exact multi-scale quadrant invariant: every S=8
tile lands in the same quadrant as its S=16 / S=32 / S=64 parent (the shared cut is
snapped to a multiple of the coarsest scale factor; see
src.dataset._compute_quadrant_definitions).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    EMPTY_TRUTH_OBS_ID,
    LABELS_SUBDIR,
    PACKAGED_SUBDIR,
    SPLITS_SUBDIR,
    TILE_KEY_COLUMNS,
    _compute_quadrant_definitions,
    _quadrant_array_for_image,
    build_image_inventory,
    build_split,
    discover_obs_ids,
    load_package_metadata,
    package_split,
    write_split_metadata,
)


# ============================================================================
# Synthetic image helpers
# ============================================================================

def _write_multiscale_image(
    labels_dir: Path,
    obs_id: str,
    *,
    ti_lo_s8: int = 0,
    ti_hi_s8: int = 64,   # exclusive; gives 64 S=8 tiles in ti -> 8 S=64 tiles
    tj_lo_s8: int = 0,
    tj_hi_s8: int = 64,
    rich: bool = True,
    n_polys: int = 100,
    features_dir: Path | None = None,
) -> None:
    """Write a synthetic multi-scale labels parquet aligned to a CTX-mosaic grid.

    S=8 spans the full range; coarser scales are nested derivations (ti_S16 = ti_S8 // 2 etc.).
    A "rich" image gets non-zero fractional_area on the upper-right quadrant (quadrant 3),
    leaving the lower-left empty -- enough variance to verify quadrant-aware metrics.
    """
    rows: list[dict] = []
    # S=8 tiles
    for ti in range(ti_lo_s8, ti_hi_s8):
        for tj in range(tj_lo_s8, tj_hi_s8):
            mid_ti = (ti_lo_s8 + ti_hi_s8) // 2
            mid_tj = (tj_lo_s8 + tj_hi_s8) // 2
            in_quad3 = (ti >= mid_ti) and (tj >= mid_tj)
            frac = 0.01 if (rich and in_quad3) else 0.0
            rows.append({
                "obs_id": obs_id, "scale_idx": 0, "tile_size_px": 8,
                "tile_size_m": 40.0, "ti": ti, "tj": tj,
                "xmin": float(ti * 40.0), "ymin": float(tj * 40.0),
                "xmax": float((ti + 1) * 40.0), "ymax": float((tj + 1) * 40.0),
                "boulder_area": frac * 1600.0, "boulder_count": 1 if frac > 0 else 0,
                "tile_area": 1600.0, "fractional_area": frac,
                "binary_by_area": frac > 0.005, "binary_by_count": frac > 0,
                "count_density": (1 / 1600.0) if frac > 0 else 0.0,
                "config_hash": "test",
            })
    # Coarser scales: S=16 = S=8 // 2, S=32 = S=8 // 4, S=64 = S=8 // 8
    for tile_px, scale_idx, factor in [(16, 1, 2), (32, 2, 4), (64, 3, 8)]:
        ti_lo = ti_lo_s8 // factor
        ti_hi = ti_hi_s8 // factor
        tj_lo = tj_lo_s8 // factor
        tj_hi = tj_hi_s8 // factor
        for ti in range(ti_lo, ti_hi):
            for tj in range(tj_lo, tj_hi):
                mid_ti_s8 = (ti_lo_s8 + ti_hi_s8) // 2
                mid_tj_s8 = (tj_lo_s8 + tj_hi_s8) // 2
                in_quad3 = (ti * factor >= mid_ti_s8) and (tj * factor >= mid_tj_s8)
                frac = 0.01 if (rich and in_quad3) else 0.0
                rows.append({
                    "obs_id": obs_id, "scale_idx": scale_idx, "tile_size_px": tile_px,
                    "tile_size_m": float(tile_px) * 5.0, "ti": ti, "tj": tj,
                    "xmin": float(ti * tile_px * 5.0), "ymin": float(tj * tile_px * 5.0),
                    "xmax": float((ti + 1) * tile_px * 5.0), "ymax": float((tj + 1) * tile_px * 5.0),
                    "boulder_area": frac * (tile_px * 5.0) ** 2, "boulder_count": 1 if frac > 0 else 0,
                    "tile_area": (tile_px * 5.0) ** 2, "fractional_area": frac,
                    "binary_by_area": frac > 0.005, "binary_by_count": frac > 0,
                    "count_density": (1.0 / (tile_px * 5.0) ** 2) if frac > 0 else 0.0,
                    "config_hash": "test",
                })
    pd.DataFrame(rows).to_parquet(labels_dir / f"{obs_id}.parquet", index=False)
    (labels_dir / f"{obs_id}.json").write_text(
        json.dumps({"obs_id": obs_id, "n_polygons_after_filter": n_polys}), encoding="utf-8",
    )
    if features_dir is not None:
        # Match TILE_KEY_COLUMNS (obs_id, scale_idx, tile_size_px, ti, tj) so _join_one_image
        # can merge features on the four key columns.
        feat_rows: list[dict] = []
        for tile_px, scale_idx, factor in [(8, 0, 1), (16, 1, 2), (32, 2, 4), (64, 3, 8)]:
            for ti in range(ti_lo_s8 // factor, ti_hi_s8 // factor):
                for tj in range(tj_lo_s8 // factor, tj_hi_s8 // factor):
                    feat_rows.append({
                        "obs_id": obs_id, "scale_idx": scale_idx, "tile_size_px": tile_px,
                        "ti": ti, "tj": tj,
                        "intensity_mean": 100.0,
                        "shadow_fraction": 0.05,
                        "config_hash": "test",
                    })
        pd.DataFrame(feat_rows).to_parquet(features_dir / f"{obs_id}.parquet", index=False)


def _synthetic_manifest(obs_labels: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"ObsId": list(obs_labels), "BoulderLabel": list(obs_labels.values())})


# ============================================================================
# Quadrant cut computation
# ============================================================================

def test_quadrant_cuts_are_strictly_coherent_across_scales(tmp_path):
    """An S=8 tile's quadrant must equal its parent S=16/S=32/S=64 tile's quadrant.

    The shared-cut design (finest median snapped to a multiple of the coarsest factor)
    is what makes this strictly true. PLAN_Stage5c.md §8 invariant.
    """
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_multiscale_image(labels_dir, "OBS_001", ti_hi_s8=64, tj_hi_s8=64)
    defs = _compute_quadrant_definitions("OBS_001", labels_dir)
    assert set(defs.keys()) == {"8", "16", "32", "64"}, defs
    # All medians divisible by the cross-scale factor: ti_mid_S8 == 2*ti_mid_S16 == 4*ti_mid_S32 == 8*ti_mid_S64.
    assert defs["8"]["ti_mid"] == 2 * defs["16"]["ti_mid"] == 4 * defs["32"]["ti_mid"] == 8 * defs["64"]["ti_mid"]
    assert defs["8"]["tj_mid"] == 2 * defs["16"]["tj_mid"] == 4 * defs["32"]["tj_mid"] == 8 * defs["64"]["tj_mid"]

    # Per-tile coherence: each S=8 tile's quadrant == its S=64 parent's quadrant.
    df = pd.read_parquet(labels_dir / "OBS_001.parquet")
    q_arr, _ = _quadrant_array_for_image(df, defs)
    df = df.assign(quadrant=q_arr)
    s8 = df[df["tile_size_px"] == 8].copy()
    s64 = df[df["tile_size_px"] == 64][["ti", "tj", "quadrant"]].rename(
        columns={"ti": "parent_ti", "tj": "parent_tj", "quadrant": "parent_q"}
    )
    s8["parent_ti"] = s8["ti"] // 8
    s8["parent_tj"] = s8["tj"] // 8
    merged = s8.merge(s64, on=["parent_ti", "parent_tj"], how="inner")
    assert (merged["quadrant"] == merged["parent_q"]).all(), (
        f"{(merged['quadrant'] != merged['parent_q']).sum()} of {len(merged)} S=8 tiles "
        f"disagree with their S=64 parent quadrant"
    )


def test_quadrant_cut_is_a_multiple_of_coarsest_factor_at_finest_scale(tmp_path):
    """Snap behaviour: with raw median ~31.5 on a 64-tile dim, the snapped S=8 ti_mid is 24 or 32."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_multiscale_image(labels_dir, "OBS_001", ti_hi_s8=64, tj_hi_s8=64)
    defs = _compute_quadrant_definitions("OBS_001", labels_dir)
    assert defs["8"]["ti_mid"] % 8 == 0
    assert defs["8"]["tj_mid"] % 8 == 0


# ============================================================================
# Splitter
# ============================================================================

def _build_within_image_meta(tmp_path: Path, n_images: int, *, include_empty: bool = False, buffer_tiles: int = 0):
    labels_dir = tmp_path / "labels"
    features_dir = tmp_path / "features"
    labels_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    obs_labels: dict[str, str] = {}
    for i in range(n_images):
        obs = f"OBS_{i:03d}"
        obs_labels[obs] = "Boulder rich"
        _write_multiscale_image(labels_dir, obs, features_dir=features_dir)
    if include_empty:
        _write_multiscale_image(labels_dir, EMPTY_TRUTH_OBS_ID, features_dir=features_dir, rich=False)
        obs_labels[EMPTY_TRUTH_OBS_ID] = "unknown"
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    n_non_excluded = n_images  # excluded_obs_ids = [EMPTY_TRUTH_OBS_ID]
    meta = build_split(
        name="within_image_4fold", n_folds=n_non_excluded * 4,
        stratification="within_image", seed=0,
        inventory=inv, config_hash="test",
        labels_dir=labels_dir, n_folds_per_image=4,
        buffer_tiles=buffer_tiles,
        excluded_obs_ids=[EMPTY_TRUTH_OBS_ID],
    )
    return meta, inv, labels_dir, features_dir


def test_within_image_4fold_partitions_each_image_into_4_quadrants(tmp_path):
    meta, _, labels_dir, _ = _build_within_image_meta(tmp_path, n_images=2)
    assert meta["kind"] == "within-image"
    assert meta["n_folds"] == 8
    # Each image appears as test_obs_id in exactly 4 folds, one per quadrant.
    by_obs: dict[str, list[int]] = {}
    for fold in meta["folds"]:
        by_obs.setdefault(fold["test_obs_id"], []).append(fold["test_quadrant"])
    for obs, quadrants in by_obs.items():
        assert sorted(quadrants) == [0, 1, 2, 3], f"{obs}: {quadrants}"


def test_within_image_quadrants_dont_overlap_and_cover_image(tmp_path):
    """For one image, the 4 fold test sets are disjoint and their union == the image."""
    meta, _, labels_dir, _ = _build_within_image_meta(tmp_path, n_images=1)
    obs = "OBS_000"
    df = pd.read_parquet(labels_dir / f"{obs}.parquet")
    folds = [f for f in meta["folds"] if f["test_obs_id"] == obs]
    union_tile_keys: set[tuple[int, int, int]] = set()
    seen_in_multiple = set()
    for fold in folds:
        defs = fold["quadrant_definitions"]
        q_arr, keep = _quadrant_array_for_image(df, defs)
        test_mask = (q_arr == fold["test_quadrant"]) & keep
        tile_keys = set(zip(df.loc[test_mask, "tile_size_px"], df.loc[test_mask, "ti"], df.loc[test_mask, "tj"]))
        # Disjoint with the union so far.
        overlap = tile_keys & union_tile_keys
        if overlap:
            seen_in_multiple.update(overlap)
        union_tile_keys |= tile_keys
    assert not seen_in_multiple, f"{len(seen_in_multiple)} tiles assigned to multiple quadrants"
    expected_total = len(df)
    assert len(union_tile_keys) == expected_total


def test_within_image_train_is_same_image_only(tmp_path):
    meta, _, _, _ = _build_within_image_meta(tmp_path, n_images=3)
    for fold in meta["folds"]:
        assert fold["train_obs_ids"] == [fold["test_obs_id"]], (
            f"fold {fold['fold_idx']} train_obs_ids={fold['train_obs_ids']} test={fold['test_obs_id']}"
        )


def test_within_image_excludes_empty_truth_image(tmp_path):
    meta, _, _, _ = _build_within_image_meta(tmp_path, n_images=2, include_empty=True)
    assert EMPTY_TRUTH_OBS_ID not in meta["manifest_obs_ids"]
    assert meta["excluded_obs_ids"] == [EMPTY_TRUTH_OBS_ID]
    for fold in meta["folds"]:
        assert fold["test_obs_id"] != EMPTY_TRUTH_OBS_ID
    # n_folds = 2 non-excluded images x 4 quadrants = 8
    assert meta["n_folds"] == 8


def test_within_image_split_reproducibility_with_seed(tmp_path):
    m1, _, _, _ = _build_within_image_meta(tmp_path, n_images=2)
    m2, _, _, _ = _build_within_image_meta(tmp_path, n_images=2)
    # Same inventory -> identical folds and identical split_hash.
    assert [f["fold_idx"] for f in m1["folds"]] == [f["fold_idx"] for f in m2["folds"]]
    assert [f["test_obs_id"] for f in m1["folds"]] == [f["test_obs_id"] for f in m2["folds"]]
    assert [f["test_quadrant"] for f in m1["folds"]] == [f["test_quadrant"] for f in m2["folds"]]
    assert m1["split_hash"] == m2["split_hash"]


def test_within_image_metadata_records_quadrant_definitions(tmp_path):
    meta, _, _, _ = _build_within_image_meta(tmp_path, n_images=1)
    fold = meta["folds"][0]
    defs = fold["quadrant_definitions"]
    # All four scales present, with integer ti_mid/tj_mid values.
    assert set(defs.keys()) == {"8", "16", "32", "64"}
    for scale_str, qd in defs.items():
        assert isinstance(qd["ti_mid"], int) and isinstance(qd["tj_mid"], int)
        # No bizarre values: cuts inside the per-scale tile range.
        scale = int(scale_str)
        factor = scale // 8 or 1
        assert 0 <= qd["ti_mid"] <= 64 // factor
        assert 0 <= qd["tj_mid"] <= 64 // factor


def test_within_image_buffer_drops_boundary_tiles(tmp_path):
    """With buffer_tiles=1, no test or train row has ti == ti_mid or tj == tj_mid at any scale."""
    meta, _, labels_dir, _ = _build_within_image_meta(tmp_path, n_images=1, buffer_tiles=1)
    obs = "OBS_000"
    df = pd.read_parquet(labels_dir / f"{obs}.parquet")
    for fold in meta["folds"]:
        defs = fold["quadrant_definitions"]
        q_arr, keep = _quadrant_array_for_image(df, defs, buffer_tiles=1)
        kept = df[keep].copy()
        for scale_str, qd in defs.items():
            scale = int(scale_str)
            sub = kept[kept["tile_size_px"] == scale]
            assert (sub["ti"] != qd["ti_mid"]).all(), (
                f"fold {fold['fold_idx']} scale={scale}: cut-line ti found in kept rows"
            )
            assert (sub["tj"] != qd["tj_mid"]).all()


def test_within_image_n_folds_must_match_expected_count(tmp_path):
    """Passing n_folds != n_images * n_folds_per_image raises a clear error."""
    labels_dir = tmp_path / "labels"
    features_dir = tmp_path / "features"
    labels_dir.mkdir(); features_dir.mkdir()
    obs_labels: dict[str, str] = {}
    for i in range(3):
        obs = f"OBS_{i:03d}"
        obs_labels[obs] = "Boulder rich"
        _write_multiscale_image(labels_dir, obs, features_dir=features_dir)
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    with pytest.raises(ValueError, match="expects n_folds=12"):
        build_split(
            name="bad", n_folds=8,  # wrong: 3 * 4 = 12
            stratification="within_image", seed=0,
            inventory=inv, config_hash="test",
            labels_dir=labels_dir, n_folds_per_image=4,
        )


def test_within_image_requires_labels_dir(tmp_path):
    obs_labels = {"OBS_000": "Boulder rich"}
    labels_dir = tmp_path / "labels"
    features_dir = tmp_path / "features"
    labels_dir.mkdir(); features_dir.mkdir()
    _write_multiscale_image(labels_dir, "OBS_000", features_dir=features_dir)
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(["OBS_000"], manifest, labels_dir)
    with pytest.raises(ValueError, match="requires the labels_dir kwarg"):
        build_split(
            name="bad", n_folds=4, stratification="within_image", seed=0,
            inventory=inv, config_hash="test",
            # labels_dir omitted
        )


# ============================================================================
# Packaging round-trip
# ============================================================================

def test_within_image_packaged_round_trip(tmp_path):
    meta, _, labels_dir, features_dir = _build_within_image_meta(tmp_path, n_images=2)
    out_dir = tmp_path / "out"
    write_split_metadata(meta, out_dir)
    pkg = package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=out_dir, emit_all_parquet=True, config_hash="test",
    )
    assert pkg["kind"] == "within-image"
    assert pkg["n_folds_per_image"] == 4
    # Each fold's per-fold count summary records test/train tile counts.
    for fold_pkg in pkg["per_fold"]:
        assert "test_obs_id" in fold_pkg
        assert "test_quadrant" in fold_pkg
        # 4 quadrants, one image -> 4 partitions of ~equal size. Across all 4 folds for one
        # image, sum of test tiles should equal sum of finest+S16+S32+S64 = total tiles.
    # Round-trip read of metadata.
    loaded = load_package_metadata("within_image_4fold", out_dir)
    assert loaded["kind"] == "within-image"
    assert loaded["split_hash"] == meta["split_hash"]


def test_within_image_packaged_per_fold_parquets_have_expected_columns(tmp_path):
    meta, _, labels_dir, features_dir = _build_within_image_meta(tmp_path, n_images=1)
    out_dir = tmp_path / "out"
    package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=out_dir, emit_all_parquet=False, config_hash="test",
    )
    pdir = out_dir / PACKAGED_SUBDIR / "within_image_4fold"
    for k in range(meta["n_folds"]):
        x = pd.read_parquet(pdir / f"X_train_fold{k}.parquet")
        y = pd.read_parquet(pdir / f"y_train_fold{k}.parquet")
        # Tile keys present on both.
        for col in TILE_KEY_COLUMNS:
            assert col in x.columns and col in y.columns
        # Groups arrays exist with same length as X.
        train_groups = np.load(pdir / f"groups_train_fold{k}.npy")
        test_groups = np.load(pdir / f"groups_test_fold{k}.npy")
        assert len(train_groups) == len(x)
        x_test = pd.read_parquet(pdir / f"X_test_fold{k}.parquet")
        assert len(test_groups) == len(x_test)


def test_within_image_groups_have_3_unique_train_codes_per_fold(tmp_path):
    """Groups arrays carry per-row quadrant indices. Training has 3 distinct quadrants
    (the test quadrant is excluded), so unique_train should always be 3 codes.

    This is the invariant `src.modeling.evaluate.run_loio` depends on: its inner-validation
    rotation `unique_train[fold_idx % n_unique]` must produce a code that is NOT in the
    held-out set. With 3 train quadrants and the test quadrant separate, the rotation is
    always safe.
    """
    meta, _, labels_dir, features_dir = _build_within_image_meta(tmp_path, n_images=1)
    out_dir = tmp_path / "out"
    package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=out_dir, emit_all_parquet=False, config_hash="test",
    )
    pdir = out_dir / PACKAGED_SUBDIR / "within_image_4fold"
    for k in range(meta["n_folds"]):
        train_groups = np.load(pdir / f"groups_train_fold{k}.npy")
        test_groups = np.load(pdir / f"groups_test_fold{k}.npy")
        unique_train = set(np.unique(train_groups).tolist())
        unique_test = set(np.unique(test_groups).tolist())
        assert len(unique_train) == 3, f"fold {k}: unique_train={unique_train}"
        assert len(unique_test) == 1, f"fold {k}: unique_test={unique_test}"
        assert not (unique_train & unique_test), f"fold {k}: train/test code collision"


def test_within_image_packaged_test_tile_counts_match_metadata(tmp_path):
    """Per-fold n_test_tiles in package metadata equals the packaged X_test rows."""
    meta, _, labels_dir, features_dir = _build_within_image_meta(tmp_path, n_images=2)
    out_dir = tmp_path / "out"
    pkg = package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=out_dir, emit_all_parquet=False, config_hash="test",
    )
    pdir = out_dir / PACKAGED_SUBDIR / "within_image_4fold"
    for fold_pkg in pkg["per_fold"]:
        k = fold_pkg["fold_idx"]
        x_test = pd.read_parquet(pdir / f"X_test_fold{k}.parquet")
        assert len(x_test) == fold_pkg["n_test_tiles"]


# ============================================================================
# Integration tests against the real priority10 dataset (slow-marked)
# ============================================================================

@pytest.mark.slow
def test_within_image_4fold_on_priority10_yields_32_folds():
    """8 non-empty images x 4 quadrants = 32 folds against the real Stage 4 labels."""
    repo_root = Path(__file__).resolve().parents[1]
    labels_dir = repo_root / "dataset" / LABELS_SUBDIR
    splits_path = repo_root / "dataset" / SPLITS_SUBDIR / "within_image_4fold.json"
    if not splits_path.exists():
        pytest.skip("Run scripts/run_stage5.py --all (after config.yaml update) first.")
    meta = json.loads(splits_path.read_text(encoding="utf-8"))
    assert meta["kind"] == "within-image"
    assert meta["n_folds"] == 32, f"expected 32 folds, got {meta['n_folds']}"
    assert EMPTY_TRUTH_OBS_ID in meta["excluded_obs_ids"]
    obs_on_disk = set(discover_obs_ids(labels_dir))
    used = set(meta["manifest_obs_ids"])
    assert EMPTY_TRUTH_OBS_ID not in used
    assert used == obs_on_disk - {EMPTY_TRUTH_OBS_ID}
