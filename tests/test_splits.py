"""Stage 5 (split + packaging) tests.

Unit tests use synthetic inventories + per-image parquets in tmp_path -- no real caches
or downloads. Two slow integration tests against the real Stage 4/4b outputs verify the
priority10 manifest reproduces the expected fold structure and that the consolidated
all.parquet matches the per-fold rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    LABELS_SUBDIR,
    PACKAGED_SUBDIR,
    SPLITS_SUBDIR,
    TILE_KEY_COLUMNS,
    _assign_size_balanced_kfold,
    build_image_inventory,
    build_split,
    discover_obs_ids,
    iter_test_batches,
    iter_train_batches,
    load_package_metadata,
    load_split_metadata,
    package_split,
    write_split_metadata,
)


# ============================================================================
# Synthetic helpers
# ============================================================================

def _synthetic_inventory(obs_labels: dict[str, str], n_tiles_per_image: int = 100) -> pd.DataFrame:
    """Build a per-image inventory dataframe from {ObsId: BoulderLabel}."""
    rows = []
    for i, (obs, label) in enumerate(sorted(obs_labels.items())):
        rows.append({
            "ObsId": obs,
            "BoulderLabel": label,
            "n_tiles_total": n_tiles_per_image,
            "n_tiles_finest": n_tiles_per_image * 8 // 10,
            "frac_mean_finest": 0.001 * (i + 1),
            "n_polys_after_filter": (i + 1) * 100,
        })
    return pd.DataFrame(rows).set_index("ObsId")


def _write_synthetic_image_parquets(
    tmp_path: Path, obs_ids: list[str], *, n_tiles_per_image: int = 10,
) -> tuple[Path, Path]:
    """Create minimal labels + features parquets for each ObsId in tmp_path.

    Returns (labels_dir, features_dir).
    """
    labels_dir = tmp_path / "labels"
    features_dir = tmp_path / "features"
    labels_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    for i, obs in enumerate(obs_ids):
        rows = []
        for ti in range(n_tiles_per_image):
            rows.append({
                "obs_id": obs,
                "scale_idx": 0,
                "tile_size_px": 8,
                "tile_size_m": 40.0,
                "ti": int(ti),
                "tj": 0,
                "xmin": 0.0, "ymin": 0.0, "xmax": 40.0, "ymax": 40.0,
                "boulder_area": float(ti) * 0.5,
                "boulder_count": int(ti),
                "tile_area": 1600.0,
                "fractional_area": float(ti) * 0.0001,
                "binary_by_area": False,
                "binary_by_count": False,
                "count_density": float(ti) * 1e-4,
                "config_hash": "test",
            })
        labels = pd.DataFrame(rows)
        labels.to_parquet(labels_dir / f"{obs}.parquet", index=False)
        # Stage 4 sidecar too -- build_image_inventory reads n_polygons_after_filter.
        (labels_dir / f"{obs}.json").write_text(
            json.dumps({"obs_id": obs, "n_polygons_after_filter": (i + 1) * 10}),
            encoding="utf-8",
        )
        feats = pd.DataFrame({
            "obs_id": obs,
            "scale_idx": 0,
            "tile_size_px": 8,
            "ti": list(range(n_tiles_per_image)),
            "tj": 0,
            "intensity_mean": [100.0 + i * 5] * n_tiles_per_image,
            "shadow_fraction": [0.05] * n_tiles_per_image,
            "config_hash": "test",
        })
        feats.to_parquet(features_dir / f"{obs}.parquet", index=False)
    return labels_dir, features_dir


def _synthetic_manifest(obs_labels: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "ObsId": list(obs_labels),
        "BoulderLabel": list(obs_labels.values()),
    })


# ============================================================================
# Inventory
# ============================================================================

def test_build_image_inventory_round_trip(tmp_path):
    """Inventory built from synthetic parquets has the expected columns and counts."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(3)}
    labels_dir, _ = _write_synthetic_image_parquets(tmp_path, sorted(obs_labels))
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    assert set(inv.index) == set(obs_labels)
    assert (inv["BoulderLabel"] == "Boulder rich").all()
    assert (inv["n_tiles_total"] == 10).all()


def test_discover_obs_ids_finds_parquets(tmp_path):
    obs_ids = [f"OBS_{i:03d}" for i in range(4)]
    labels_dir, _ = _write_synthetic_image_parquets(tmp_path, obs_ids)
    found = discover_obs_ids(labels_dir)
    assert found == sorted(obs_ids)


# ============================================================================
# Split construction
# ============================================================================

def test_loio_9fold_uses_each_image_exactly_once_in_test():
    obs_labels = {f"OBS_{i:03d}": ["Boulder rich", "Boulder poor", "unknown"][i % 3]
                   for i in range(9)}
    inv = _synthetic_inventory(obs_labels)
    meta = build_split(name="loio_9fold", n_folds=9, stratification="none",
                        seed=0, inventory=inv, config_hash="test")
    assert len(meta["folds"]) == 9
    test_obs_lists = [tuple(f["test_obs_ids"]) for f in meta["folds"]]
    flat = [obs for tup in test_obs_lists for obs in tup]
    assert len(flat) == 9
    assert set(flat) == set(obs_labels)
    # Each image appears in exactly one test fold.
    assert len(set(flat)) == 9


def test_stratified_3fold_balances_image_count():
    """3-fold size-balanced on 9 images (5 rich + 2 poor + 2 unknown) gives 3/3/3 sizes."""
    obs_labels = {}
    for i in range(5):
        obs_labels[f"R_{i:03d}"] = "Boulder rich"
    for i in range(2):
        obs_labels[f"P_{i:03d}"] = "Boulder poor"
    for i in range(2):
        obs_labels[f"U_{i:03d}"] = "unknown"
    inv = _synthetic_inventory(obs_labels)
    meta = build_split(name="loio_3fold_balanced", n_folds=3,
                        stratification="boulder_label_size_balanced",
                        seed=0, inventory=inv, config_hash="test")
    sizes = [len(f["test_obs_ids"]) for f in meta["folds"]]
    assert sizes == [3, 3, 3], f"Expected 3/3/3 sizes; got {sizes}"
    # Every image appears in exactly one test fold.
    flat = [obs for f in meta["folds"] for obs in f["test_obs_ids"]]
    assert sorted(flat) == sorted(obs_labels)
    # The 5-3 split of the 5 rich images means at most two folds have 2 rich.
    rich_per_fold = [
        sum(1 for o in f["test_obs_ids"] if obs_labels[o] == "Boulder rich")
        for f in meta["folds"]
    ]
    assert sorted(rich_per_fold) == [1, 2, 2]


def test_split_reproducibility_with_seed():
    """Same seed + same inventory -> identical split."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" if i < 5 else "unknown"
                   for i in range(9)}
    inv = _synthetic_inventory(obs_labels)
    m1 = build_split(name="loio_3fold_balanced", n_folds=3,
                      stratification="boulder_label_size_balanced",
                      seed=42, inventory=inv, config_hash="test")
    m2 = build_split(name="loio_3fold_balanced", n_folds=3,
                      stratification="boulder_label_size_balanced",
                      seed=42, inventory=inv, config_hash="test")
    assert [f["test_obs_ids"] for f in m1["folds"]] == [f["test_obs_ids"] for f in m2["folds"]]
    assert m1["split_hash"] == m2["split_hash"]


def test_split_different_seed_can_change_assignment():
    """Different seed -> potentially different fold assignment (regression-test that
    `seed` actually feeds the random choice)."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(9)}
    inv = _synthetic_inventory(obs_labels)
    a = _assign_size_balanced_kfold(inv, n_folds=3, seed=0)
    b = _assign_size_balanced_kfold(inv, n_folds=3, seed=999)
    # With deterministic-permutation behaviour, at least one of the two seeds should
    # produce a different per-fold assignment. With all images in the same label group
    # the seed is the only randomness in play.
    assert a != b


def test_no_obs_id_in_both_train_and_test_in_any_fold():
    """The single most important correctness property -- group-leak check."""
    obs_labels = {f"OBS_{i:03d}": ["Boulder rich", "Boulder poor", "unknown"][i % 3]
                   for i in range(7)}
    inv = _synthetic_inventory(obs_labels)
    meta = build_split(name="lo", n_folds=7, stratification="none", seed=0,
                       inventory=inv, config_hash="test")
    for fold in meta["folds"]:
        leak = set(fold["test_obs_ids"]) & set(fold["train_obs_ids"])
        assert not leak, f"fold {fold['fold_idx']}: leak {leak}"


def test_split_grows_with_manifest_to_12_images():
    """Nothing in the splitter is hardcoded to 9 images."""
    obs_labels = {f"OBS_{i:03d}": ["Boulder rich", "Boulder poor", "unknown"][i % 3]
                   for i in range(12)}
    inv = _synthetic_inventory(obs_labels)
    meta = build_split(name="x", n_folds=12, stratification="none", seed=0,
                       inventory=inv, config_hash="test")
    assert len(meta["folds"]) == 12
    assert set(meta["manifest_obs_ids"]) == set(obs_labels)


def test_split_none_stratification_requires_n_folds_equals_n_images():
    """stratification='none' with n_folds != n_images is a config error -- the caller
    presumably meant 'boulder_label_size_balanced'."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(9)}
    inv = _synthetic_inventory(obs_labels)
    with pytest.raises(ValueError, match="stratification='none' requires"):
        build_split(name="x", n_folds=3, stratification="none", seed=0,
                    inventory=inv, config_hash="test")


# ============================================================================
# Packaging
# ============================================================================

def test_package_split_round_trip(tmp_path):
    """Build -> package -> reload -> row counts match expectation."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(4)}
    labels_dir, features_dir = _write_synthetic_image_parquets(
        tmp_path, sorted(obs_labels), n_tiles_per_image=10,
    )
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    meta = build_split(name="loio_4fold", n_folds=4, stratification="none",
                       seed=0, inventory=inv, config_hash="test")
    out_dir = tmp_path / "out"
    write_split_metadata(meta, out_dir)
    pkg = package_split(meta, labels_dir=labels_dir, features_dir=features_dir,
                        output_dir=out_dir, emit_all_parquet=True, config_hash="test")
    # Each fold's test should be 10 rows (one image); train 30 rows (other three).
    for fold in pkg["per_fold"]:
        assert fold["n_test_tiles"] == 10
        assert fold["n_train_tiles"] == 30
    # Round-trip read: reload metadata and confirm match.
    loaded = load_package_metadata("loio_4fold", out_dir)
    assert loaded["name"] == "loio_4fold"
    assert loaded["split_hash"] == meta["split_hash"]


def test_package_emits_all_parquet_when_enabled(tmp_path):
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(3)}
    labels_dir, features_dir = _write_synthetic_image_parquets(
        tmp_path, sorted(obs_labels), n_tiles_per_image=5,
    )
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    meta = build_split(name="loio_3", n_folds=3, stratification="none",
                       seed=0, inventory=inv, config_hash="test")
    out_dir = tmp_path / "out"
    pkg = package_split(meta, labels_dir=labels_dir, features_dir=features_dir,
                        output_dir=out_dir, emit_all_parquet=True, config_hash="test")
    assert pkg["all_parquet_path"] is not None
    all_df = pd.read_parquet(pkg["all_parquet_path"])
    # Each tile appears once in `all.parquet` (tagged with its test fold_idx).
    assert len(all_df) == 3 * 5
    assert set(all_df["fold_idx"]) == {0, 1, 2}
    assert "obs_id" in all_df.columns


def test_package_no_all_parquet_when_disabled(tmp_path):
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(3)}
    labels_dir, features_dir = _write_synthetic_image_parquets(
        tmp_path, sorted(obs_labels), n_tiles_per_image=5,
    )
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    meta = build_split(name="loio_3", n_folds=3, stratification="none",
                       seed=0, inventory=inv, config_hash="test")
    out_dir = tmp_path / "out"
    pkg = package_split(meta, labels_dir=labels_dir, features_dir=features_dir,
                        output_dir=out_dir, emit_all_parquet=False, config_hash="test")
    assert pkg["all_parquet_path"] is None


def test_package_groups_npy_aligns_with_x_rows(tmp_path):
    """groups_{train,test}_fold{k}.npy must have one entry per row in the matching X parquet."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(3)}
    labels_dir, features_dir = _write_synthetic_image_parquets(
        tmp_path, sorted(obs_labels), n_tiles_per_image=7,
    )
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    meta = build_split(name="loio_3", n_folds=3, stratification="none",
                       seed=0, inventory=inv, config_hash="test")
    out_dir = tmp_path / "out"
    package_split(meta, labels_dir=labels_dir, features_dir=features_dir,
                  output_dir=out_dir, emit_all_parquet=True, config_hash="test")
    for k in range(3):
        x_train = pd.read_parquet(out_dir / PACKAGED_SUBDIR / "loio_3" / f"X_train_fold{k}.parquet")
        groups = np.load(out_dir / PACKAGED_SUBDIR / "loio_3" / f"groups_train_fold{k}.npy")
        assert len(x_train) == len(groups)


def test_scale_filter_restricts_emitted_rows(tmp_path):
    """scale_filter=[8] on synthetic parquets that only have S=8 leaves rows unchanged."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(2)}
    labels_dir, features_dir = _write_synthetic_image_parquets(
        tmp_path, sorted(obs_labels), n_tiles_per_image=4,
    )
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    meta = build_split(name="loio_2", n_folds=2, stratification="none",
                       seed=0, inventory=inv, config_hash="test")
    out_dir = tmp_path / "out"
    pkg = package_split(meta, labels_dir=labels_dir, features_dir=features_dir,
                        output_dir=out_dir, scale_filter=[8], emit_all_parquet=False,
                        config_hash="test")
    assert pkg["per_fold"][0]["n_test_tiles"] == 4


# ============================================================================
# Streaming iterator
# ============================================================================

def test_streaming_iterator_yields_one_dataframe_per_obs(tmp_path):
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(3)}
    labels_dir, features_dir = _write_synthetic_image_parquets(
        tmp_path, sorted(obs_labels), n_tiles_per_image=4,
    )
    manifest = _synthetic_manifest(obs_labels)
    inv = build_image_inventory(sorted(obs_labels), manifest, labels_dir)
    meta = build_split(name="loio_3", n_folds=3, stratification="none",
                       seed=0, inventory=inv, config_hash="test")
    fold = meta["folds"][0]
    train_batches = list(iter_train_batches(
        meta, fold_idx=0, labels_dir=labels_dir, features_dir=features_dir,
    ))
    test_batches = list(iter_test_batches(
        meta, fold_idx=0, labels_dir=labels_dir, features_dir=features_dir,
    ))
    assert len(train_batches) == len(fold["train_obs_ids"])
    assert len(test_batches) == len(fold["test_obs_ids"])
    # Every train batch belongs to a train ObsId; no leak.
    for batch in train_batches:
        assert set(batch["obs_id"]) <= set(fold["train_obs_ids"])
    for batch in test_batches:
        assert set(batch["obs_id"]) <= set(fold["test_obs_ids"])


# ============================================================================
# Slow integration tests on the real Stage 4/4b sweep
# ============================================================================

@pytest.mark.slow
def test_priority10_loio_9fold_matches_sweep():
    repo_root = Path(__file__).resolve().parents[1]
    labels_dir = repo_root / "dataset" / LABELS_SUBDIR
    splits_path = repo_root / "dataset" / SPLITS_SUBDIR / "loio_9fold.json"
    if not splits_path.exists():
        pytest.skip("Run scripts/run_stage5.py --all first to produce real outputs.")
    meta = load_split_metadata("loio_9fold", repo_root / "dataset")
    # Every ObsId on disk should appear exactly once across the test sets.
    obs_on_disk = set(discover_obs_ids(labels_dir))
    flat = [obs for f in meta["folds"] for obs in f["test_obs_ids"]]
    assert sorted(flat) == sorted(obs_on_disk)
    # No group leakage in any fold.
    for fold in meta["folds"]:
        assert not (set(fold["train_obs_ids"]) & set(fold["test_obs_ids"]))


@pytest.mark.slow
def test_priority10_all_parquet_row_count_matches_sum_of_test_folds():
    repo_root = Path(__file__).resolve().parents[1]
    pkg_path = repo_root / "dataset" / PACKAGED_SUBDIR / "loio_9fold" / "all.parquet"
    if not pkg_path.exists():
        pytest.skip("Run scripts/run_stage5.py --all first.")
    pkg = load_package_metadata("loio_9fold", repo_root / "dataset")
    all_df = pd.read_parquet(pkg_path)
    expected = sum(f["n_test_tiles"] for f in pkg["per_fold"])
    assert len(all_df) == expected
