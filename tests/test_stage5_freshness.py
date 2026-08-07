"""R04 — a Stage-5 failure must be loud, and a stale package must be undetectable no longer.

`_run_one` wrapped `build_split` in `try/except ValueError`, printed "FAILED to build", and
returned `None`; `main` discarded the return and `return 0` unconditionally. Both guards
that can raise are live-data driven, so the realistic path is: a cohort expansion lands,
the hand-edited `n_folds` is stale, Stage 5 prints a failure and exits 0, the previous
cohort's `packaged/{scheme}/` stays on disk, and the next sweep trains and reports on the
old folds with no warning. Nothing downstream compared the package against the split
metadata, the label cohort, or the label *contents*.

Synthetic parquets in `tmp_path` throughout — no live root, no producer against one.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.dataset import (
    PACKAGED_SUBDIR,
    SPLITS_SUBDIR,
    build_image_inventory,
    build_split,
    package_split,
    write_split_metadata,
)
from src.modeling import loaders as L

from .test_splits import _synthetic_manifest, _write_synthetic_image_parquets

SCHEME = "loio_3"


def _make_package(tmp_path: Path, n_images: int = 3) -> Path:
    """Build a complete synthetic dataset root: labels, features, split, package."""
    obs_labels = {f"OBS_{i:03d}": "Boulder rich" for i in range(n_images)}
    labels_dir, features_dir = _write_synthetic_image_parquets(
        tmp_path, sorted(obs_labels), n_tiles_per_image=6,
    )
    # `verify_package_freshness` looks for `<root>/labels` and `<root>/features`.
    assert labels_dir == tmp_path / "labels" and features_dir == tmp_path / "features"
    inv = build_image_inventory(sorted(obs_labels), _synthetic_manifest(obs_labels), labels_dir)
    meta = build_split(name=SCHEME, n_folds=n_images, stratification="none", seed=0,
                       inventory=inv, config_hash="test")
    write_split_metadata(meta, tmp_path)
    package_split(meta, labels_dir=labels_dir, features_dir=features_dir,
                  output_dir=tmp_path, emit_all_parquet=False, config_hash="test")
    L._VERIFIED.clear()
    return tmp_path


def _meta_path(root: Path) -> Path:
    return root / PACKAGED_SUBDIR / SCHEME / "metadata.json"


# ============================================================================
# The driver's exit code
# ============================================================================

def _load_driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_stage5.py"
    spec = importlib.util.spec_from_file_location("run_stage5_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage5_main_exits_nonzero_when_a_scheme_fails(tmp_path, monkeypatch, capsys):
    """The exit code, which used to be 0 no matter what."""
    root = _make_package(tmp_path)
    driver = _load_driver()

    cfg = type("Cfg", (), {})()
    cfg.raw = {"splits": {"schemes": {SCHEME: {"n_folds": 3, "stratification": "none", "seed": 0}},
                          "emit_all_parquet": False}}
    cfg.output_dir = root
    cfg.manifest_path = root / "manifest.csv"
    cfg.hash = "test"
    cfg.__class__.__getitem__ = lambda self, k: self.raw[k]

    monkeypatch.setattr(driver, "load_config", lambda _p: cfg)
    monkeypatch.setattr(driver.M, "load_manifest", lambda _p: _synthetic_manifest(
        {f"OBS_{i:03d}": "Boulder rich" for i in range(3)}))
    monkeypatch.setattr(driver, "build_split", lambda **kw: (_ for _ in ()).throw(
        ValueError("stratification='none' requires n_folds == n_images")))
    monkeypatch.setattr(sys, "argv", ["run_stage5.py", SCHEME])

    assert driver.main() == 1, "a scheme that failed to build must not exit 0"
    out = capsys.readouterr().out
    assert "FAILED" in out and "STALE" in out, out


def test_stage5_main_exits_zero_when_every_scheme_builds(tmp_path, monkeypatch):
    """The guard must not turn a healthy run into a failure."""
    root = _make_package(tmp_path)
    driver = _load_driver()

    cfg = type("Cfg", (), {})()
    cfg.raw = {"splits": {"schemes": {SCHEME: {"n_folds": 3, "stratification": "none", "seed": 0}},
                          "emit_all_parquet": False}}
    cfg.output_dir = root
    cfg.manifest_path = root / "manifest.csv"
    cfg.hash = "test"
    cfg.__class__.__getitem__ = lambda self, k: self.raw[k]

    monkeypatch.setattr(driver, "load_config", lambda _p: cfg)
    monkeypatch.setattr(driver.M, "load_manifest", lambda _p: _synthetic_manifest(
        {f"OBS_{i:03d}": "Boulder rich" for i in range(3)}))
    monkeypatch.setattr(sys, "argv", ["run_stage5.py", SCHEME, "--no-package"])
    assert driver.main() == 0


# ============================================================================
# Package freshness
# ============================================================================

def test_a_freshly_packaged_split_verifies_clean(tmp_path):
    root = _make_package(tmp_path)
    L.verify_package_freshness(SCHEME, root, force=True)  # must not raise or warn
    meta = json.loads(_meta_path(root).read_text(encoding="utf-8"))
    assert meta["source_digests"]["digest"]
    assert set(meta["source_digests"]["per_obs"]) == set(meta["obs_to_int"])


def test_split_hash_drift_is_detected(tmp_path):
    root = _make_package(tmp_path)
    split_path = root / SPLITS_SUBDIR / f"{SCHEME}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["split_hash"] = "0" * 64
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(L.StalePackageError, match="split_hash"):
        L.verify_package_freshness(SCHEME, root, force=True)


def test_cohort_expansion_leaves_the_old_package_detectably_stale(tmp_path):
    """The exact R04 failure scenario: new images land, Stage 5 fails, the previous
    cohort's package stays on disk."""
    root = _make_package(tmp_path, n_images=3)
    newcomer = {"OBS_099": "Boulder rich"}
    _write_synthetic_image_parquets(root, ["OBS_099"], n_tiles_per_image=6)
    assert (root / "labels" / "OBS_099.parquet").exists() and newcomer
    with pytest.raises(L.StalePackageError, match="Never packaged.*OBS_099"):
        L.verify_package_freshness(SCHEME, root, force=True)


def test_label_content_change_at_a_fixed_cohort_is_detected(tmp_path):
    """The pre-R74 case: same images, same config, different label rows. Cohort, split and
    YAML hashes are all identical — only a content digest can see it."""
    root = _make_package(tmp_path)
    victim = root / "labels" / "OBS_000.parquet"
    df = pd.read_parquet(victim)
    df = df.iloc[:-1]  # one fewer eligible tile, exactly like a coverage-mask change
    df.to_parquet(victim, index=False)

    meta = json.loads(_meta_path(root).read_text(encoding="utf-8"))
    with pytest.raises(L.StalePackageError, match="label/feature content"):
        L.verify_package_freshness(SCHEME, root, force=True)
    # ...and none of the cheap identifiers moved, which is the point.
    split = json.loads((root / SPLITS_SUBDIR / f"{SCHEME}.json").read_text(encoding="utf-8"))
    assert split["split_hash"] == meta["split_hash"]
    assert meta["config_hash"] == "test"


def test_feature_content_change_is_detected(tmp_path):
    root = _make_package(tmp_path)
    victim = root / "features" / "OBS_001.parquet"
    df = pd.read_parquet(victim)
    df["intensity_mean"] = df["intensity_mean"] + 1.0
    df.to_parquet(victim, index=False)
    with pytest.raises(L.StalePackageError, match="OBS_001"):
        L.verify_package_freshness(SCHEME, root, force=True)


def test_a_package_without_source_digests_warns_rather_than_failing(tmp_path):
    """Every package on disk today predates the field; bricking them would be worse than
    the defect. They must still be *named* as unverifiable."""
    root = _make_package(tmp_path)
    meta = json.loads(_meta_path(root).read_text(encoding="utf-8"))
    del meta["source_digests"]
    _meta_path(root).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.warns(UserWarning, match="predates source-digest provenance"):
        L.verify_package_freshness(SCHEME, root, force=True)


def test_load_fold_runs_the_freshness_check(tmp_path):
    """The verification has to sit on the path modelling actually uses, not beside it."""
    root = _make_package(tmp_path)
    L.load_fold(SCHEME, 0, dataset_dir=root)  # clean

    victim = root / "labels" / "OBS_002.parquet"
    pd.read_parquet(victim).iloc[:-1].to_parquet(victim, index=False)
    L._VERIFIED.clear()
    with pytest.raises(L.StalePackageError):
        L.load_fold(SCHEME, 0, dataset_dir=root)


def test_verification_runs_once_per_process_unless_forced(tmp_path):
    """Re-hashing the sources on every fold load would make a 38-fold sweep pay for it 38
    times."""
    root = _make_package(tmp_path)
    L.verify_package_freshness(SCHEME, root)
    victim = root / "labels" / "OBS_000.parquet"
    pd.read_parquet(victim).iloc[:-1].to_parquet(victim, index=False)
    L.verify_package_freshness(SCHEME, root)  # cached, does not re-hash
    with pytest.raises(L.StalePackageError):
        L.verify_package_freshness(SCHEME, root, force=True)
