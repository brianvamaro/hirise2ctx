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
    LABEL_COLUMNS,
    LABEL_CONTEXT_COLUMNS,
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


# R91 (review 2026-07-31, tests-deep-within-image-1): the multi-image fixtures below are
# DELIBERATELY ragged -- each image has a different, non-square (ti != tj) extent at a
# different origin, because real HiRISE footprints do (the docstring example in
# src.dataset._compute_quadrant_definitions is ti_mid 1352 / tj_mid 5184, a 3.8x asymmetry).
# With every image an identical 64x64 square, four independent defects in the cut
# computation were literal no-ops -- most importantly "compute the quadrant cut once and
# reuse it for every image", which on real footprints collapses 8 of 9 images into a single
# quadrant. Do not make these extents uniform or symmetric again.
#
# Constraints if you edit this table:
#   * every bound is a multiple of 8 in S=8 units, so the S=16/32/64 nesting stays exact;
#   * the resulting cut must land STRICTLY inside the extent at every scale (all four
#     quadrants non-empty at S=64 too), otherwise the partition tests degenerate;
#   * ti_mid != tj_mid for every image, so a ti/tj transposition is not a no-op.
# The cut is the S=8 median floor-snapped to a multiple of the coarsest factor among the
# scales PRESENT here, i.e. 8 for this S=8..S=64 fixture (R97 — it used to be 16, taken from
# the whole factor map including the dev-only S=128 entry). These four extents happen to
# snap identically either way, which is precisely why the dedicated R97 tests below use a
# fixture whose median distinguishes step 8 from step 16.
_MULTI_IMAGE_EXTENTS_S8: list[tuple[int, int, int, int]] = [
    # (ti_lo, ti_hi, tj_lo, tj_hi)             shape    -> snapped S=8 cut (ti_mid, tj_mid)
    (0, 48, 32, 128),      # at (0, 32)        48 x 96  -> ( 16,  64)
    (32, 112, 160, 224),   # at (32, 160)      80 x 64  -> ( 64, 176)
    (128, 224, 8, 40),     # at (128, 8)       96 x 32  -> (160,  16)
    (56, 88, 40, 136),     # at (56, 40)       32 x 96  -> ( 64,  80)
]


def _extent_s8(index: int) -> tuple[int, int, int, int]:
    """Per-image (ti_lo, ti_hi, tj_lo, tj_hi) in S=8 units. See _MULTI_IMAGE_EXTENTS_S8."""
    return _MULTI_IMAGE_EXTENTS_S8[index % len(_MULTI_IMAGE_EXTENTS_S8)]


# ============================================================================
# Quadrant cut computation
# ============================================================================

def _write_s8_only(labels_dir: Path, obs_id: str, scales: dict[int, int], n_s8: int) -> None:
    """Labels containing exactly `scales` {tile_size_px: factor}, nested from S=8.

    Deliberately minimal — this exercises `_compute_quadrant_definitions`' scale bookkeeping,
    not the partitioner, so the quadrants need not be non-empty.
    """
    rows = []
    for tile_px, factor in sorted(scales.items()):
        for ti in range(n_s8 // factor):
            for tj in range(n_s8 // factor):
                rows.append({
                    "obs_id": obs_id, "tile_size_px": tile_px, "ti": ti, "tj": tj,
                    "fractional_area": 0.0, "n_polygons_after_filter": 0,
                })
    pd.DataFrame(rows).to_parquet(labels_dir / f"{obs_id}.parquet", index=False)
    (labels_dir / f"{obs_id}.json").write_text(
        json.dumps({"obs_id": obs_id, "n_polygons_after_filter": 0}), encoding="utf-8"
    )


# R97 fixture: 24 S=8 tiles per axis -> median ti = 11.5 -> int() = 11.
#   floor-snap to a multiple of 8  -> 8   (correct: the production ladder stops at S=64)
#   floor-snap to a multiple of 16 -> 0   (what the dev-only S=128 entry used to force)
# The two answers must differ, or the regression below cannot see anything.
_R97_N_S8 = 24
_PRODUCTION_LADDER = {8: 1, 16: 2, 32: 4, 64: 8}


def test_quadrant_snap_step_comes_from_the_scales_present_not_the_global_table(tmp_path):
    """R97. The snap step must be the coarsest factor among the scales this image actually
    has, not `max(SCALE_TO_FACTOR_FROM_FINEST.values())`.

    Commit 29b0adb ("CNN + S128 HELD as dev-only") added `128: 16` to the global table and
    thereby doubled the production snap step from 8 to 16 — for a scale no shipped config
    emits. Measured read-only on 2026-08-06: that moves the quadrant cut for **29 of 38**
    v2 images, and it is why the v1 within-image split looked drifted (v1 matches a step-8
    recompute 8/8; the v2 split, built after 29b0adb, matches step 16 38/38).
    """
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_s8_only(labels_dir, "OBS_R97", _PRODUCTION_LADDER, _R97_N_S8)

    # The global table, S=128 entry and all, must give the same answer as a table that
    # contains only the production ladder. That equality IS the finding.
    with_dev_scale = _compute_quadrant_definitions("OBS_R97", labels_dir)
    ladder_only = _compute_quadrant_definitions(
        "OBS_R97", labels_dir, scale_to_factor=_PRODUCTION_LADDER,
    )
    assert with_dev_scale == ladder_only, (
        "a factor-map entry for a scale this image does not contain changed its cut: "
        f"{with_dev_scale} vs {ladder_only}"
    )
    assert with_dev_scale["8"]["ti_mid"] == 8, with_dev_scale
    assert with_dev_scale["8"]["tj_mid"] == 8, with_dev_scale

    # Guard the guard: with the old `max(table.values())` behaviour the answer really was
    # different, so this fixture can distinguish them.
    inflated = _compute_quadrant_definitions(
        "OBS_R97", labels_dir, scale_to_factor={**_PRODUCTION_LADDER, 128: 16},
    )
    assert inflated == ladder_only, "the fix should ignore the absent 128 entry"
    n = _R97_N_S8
    assert (int(np.median(np.arange(n))) // 8) * 8 != (int(np.median(np.arange(n))) // 16) * 16, (
        "fixture degenerate: step 8 and step 16 snap to the same cut here"
    )


def test_quadrant_snap_step_grows_when_a_coarser_scale_is_actually_present(tmp_path):
    """The mixed set: an image that really does contain S=128 must snap to 16, or its S=128
    tiles would not nest coherently."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    mixed = {**_PRODUCTION_LADDER, 128: 16}
    _write_s8_only(labels_dir, "OBS_MIX", mixed, 128)

    defs = _compute_quadrant_definitions("OBS_MIX", labels_dir)
    assert set(defs) == {"8", "16", "32", "64", "128"}
    ti8 = defs["8"]["ti_mid"]
    assert ti8 % 16 == 0, f"S=128 present, so the S=8 cut must be a multiple of 16: {ti8}"
    for tile_px, factor in mixed.items():
        assert defs[str(tile_px)]["ti_mid"] == ti8 // factor
        assert defs[str(tile_px)]["tj_mid"] == defs["8"]["tj_mid"] // factor


def test_quadrant_definitions_reject_labels_with_no_known_scale(tmp_path):
    """A scale absent from the factor map is skipped rather than guessed — but skipping
    *everything* used to return an empty dict, which the partitioner would read as "no
    tile belongs to any quadrant"."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_s8_only(labels_dir, "OBS_ODD", {8: 1}, 16)
    with pytest.raises(ValueError, match="scale_to_factor"):
        _compute_quadrant_definitions("OBS_ODD", labels_dir, scale_to_factor={16: 2, 32: 4})


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
        ti_lo, ti_hi, tj_lo, tj_hi = _extent_s8(i)
        _write_multiscale_image(
            labels_dir, obs, features_dir=features_dir,
            ti_lo_s8=ti_lo, ti_hi_s8=ti_hi, tj_lo_s8=tj_lo, tj_hi_s8=tj_hi,
        )
    if include_empty:
        ti_lo, ti_hi, tj_lo, tj_hi = _extent_s8(n_images)
        _write_multiscale_image(
            labels_dir, EMPTY_TRUTH_OBS_ID, features_dir=features_dir, rich=False,
            ti_lo_s8=ti_lo, ti_hi_s8=ti_hi, tj_lo_s8=tj_lo, tj_hi_s8=tj_hi,
        )
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
    ti_lo, ti_hi, tj_lo, tj_hi = _extent_s8(0)  # OBS_000's ragged extent (R91)
    # All four scales present, with integer ti_mid/tj_mid values.
    assert set(defs.keys()) == {"8", "16", "32", "64"}
    for scale_str, qd in defs.items():
        assert isinstance(qd["ti_mid"], int) and isinstance(qd["tj_mid"], int)
        # No bizarre values: the cut lies STRICTLY inside this image's own per-scale tile
        # range on each axis, which is what makes all four quadrants non-empty. The ti and
        # tj ranges differ (R91), so this is two independent bounds, not one restated.
        scale = int(scale_str)
        factor = scale // 8 or 1
        assert ti_lo // factor < qd["ti_mid"] < ti_hi // factor, (scale_str, qd)
        assert tj_lo // factor < qd["tj_mid"] < tj_hi // factor, (scale_str, qd)


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

    R91: run over THREE images with different ragged extents, not one. This is the
    assertion that catches a cut computed from the wrong image (or the wrong axis), and it
    can only see that if the images are geometrically distinguishable.
    """
    meta, _, labels_dir, features_dir = _build_within_image_meta(tmp_path, n_images=3)
    out_dir = tmp_path / "out"
    package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=out_dir, emit_all_parquet=False, config_hash="test",
    )
    pdir = out_dir / PACKAGED_SUBDIR / "within_image_4fold"
    obs_of_fold = {int(f["fold_idx"]): f["test_obs_id"] for f in meta["folds"]}
    for k in range(meta["n_folds"]):
        obs = obs_of_fold[k]
        train_groups = np.load(pdir / f"groups_train_fold{k}.npy")
        test_groups = np.load(pdir / f"groups_test_fold{k}.npy")
        unique_train = set(np.unique(train_groups).tolist())
        unique_test = set(np.unique(test_groups).tolist())
        assert len(unique_train) == 3, f"fold {k} ({obs}): unique_train={unique_train}"
        assert len(unique_test) == 1, f"fold {k} ({obs}): unique_test={unique_test}"
        assert not (unique_train & unique_test), f"fold {k} ({obs}): train/test code collision"


def test_within_image_packaged_folds_contain_exactly_the_expected_tiles(tmp_path):
    """R87, within-image arm. `test_within_image_groups_*` pins the quadrant *codes*, and
    everything else here is a row count — so nothing checked which tiles actually landed
    in each packaged parquet. A fallback to a random per-tile split would keep every count
    intact and stay green.

    For this scheme the fold identity is (image, quadrant): every row of fold k must come
    from that fold's own image, test rows are exactly its test quadrant, train rows are
    exactly the other three, and the two tile-key sets must be disjoint.
    """
    meta, _, labels_dir, features_dir = _build_within_image_meta(tmp_path, n_images=3)
    out_dir = tmp_path / "out"
    package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=out_dir, emit_all_parquet=False, config_hash="test",
    )
    pdir = out_dir / PACKAGED_SUBDIR / "within_image_4fold"

    def keyset(df):
        return set(map(tuple, df[TILE_KEY_COLUMNS].itertuples(index=False, name=None)))

    for fold in meta["folds"]:
        k = int(fold["fold_idx"])
        obs = fold["test_obs_id"]
        test_q = int(fold["test_quadrant"])

        # Recompute the expected partition straight from the persisted definitions.
        labels = pd.read_parquet(labels_dir / f"{obs}.parquet")
        feats = pd.read_parquet(features_dir / f"{obs}.parquet")
        joined = labels.merge(feats, on=TILE_KEY_COLUMNS, suffixes=("", "_feat"))
        q_arr, keep = _quadrant_array_for_image(joined, fold["quadrant_definitions"], buffer_tiles=0)
        expect_test = keyset(joined[(q_arr == test_q) & keep])
        expect_train = keyset(joined[(q_arr != test_q) & (q_arr >= 0) & keep])
        assert expect_test and expect_train, "fixture must exercise a non-degenerate fold"

        x_test = pd.read_parquet(pdir / f"X_test_fold{k}.parquet")
        x_train = pd.read_parquet(pdir / f"X_train_fold{k}.parquet")
        assert keyset(x_test) == expect_test, f"fold {k} ({obs} q{test_q}): test tiles differ"
        assert keyset(x_train) == expect_train, f"fold {k} ({obs} q{test_q}): train tiles differ"
        assert not (keyset(x_train) & keyset(x_test)), (
            f"fold {k}: a tile is in BOTH packaged train and test"
        )
        # Every row belongs to this fold's image -- a cross-image mix would still count right.
        assert set(x_train["obs_id"]) == {obs} and set(x_test["obs_id"]) == {obs}
        # y must carry the same tiles, in the same order, as X.
        for side, x in (("train", x_train), ("test", x_test)):
            y = pd.read_parquet(pdir / f"y_{side}_fold{k}.parquet")
            pd.testing.assert_frame_equal(x[TILE_KEY_COLUMNS], y[TILE_KEY_COLUMNS])


def test_within_image_packaged_x_never_carries_a_label_column(tmp_path):
    """R88, within-image arm: `_package_within_image_split` shares `_split_columns`."""
    meta, _, labels_dir, features_dir = _build_within_image_meta(tmp_path, n_images=1)
    out_dir = tmp_path / "out"
    package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=out_dir, emit_all_parquet=False, config_hash="test",
    )
    pdir = out_dir / PACKAGED_SUBDIR / "within_image_4fold"
    forbidden = (set(LABEL_COLUMNS) | set(LABEL_CONTEXT_COLUMNS)) - set(TILE_KEY_COLUMNS)
    for k in range(meta["n_folds"]):
        for side in ("train", "test"):
            cols = set(pd.read_parquet(pdir / f"X_{side}_fold{k}.parquet").columns)
            leaked = cols & forbidden
            assert not leaked, f"X_{side}_fold{k} carries target column(s) {sorted(leaked)}"


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
