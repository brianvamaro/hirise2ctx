"""The step-6 embedding cache must not be reused when the label key set has moved.

DECISIONS 2026-08-20k. The resume was `if npz.exists(): skip`, which made the whole of
step 6 a 4-second silent no-op on the v2 rebuild: the cached stores held the pre-rebuild
pool (161,005 tiles) against 164,644 fresh labels, all 38 images had a different (ti, tj)
set, and 7,390 new tiles had no embedding at all. Nothing downstream could have caught it —
the npz records no provenance of any kind, so a stale store and a fresh one look identical.

Read-only: everything here is built in tmp_path.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _mod():
    """Import the probe by path; it is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "_w2_fang_embed", REPO / "scripts" / "probes" / "_w2_fang_embed.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _write_store(out_dir: Path, obs: str, ti, tj, *, ctx_px: int, tile_px: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ti)
    payload = dict(ti=np.asarray(ti, np.int32), tj=np.asarray(tj, np.int32),
                   valid=np.ones(n, bool), cls=np.zeros((n, 768), np.float32),
                   mean=np.zeros((n, 768), np.float32), gem=np.zeros((n, 768), np.float32))
    np.savez_compressed(out_dir / f"{obs}_P{ctx_px}.npz", **payload)
    np.savez_compressed(out_dir / f"{obs}_P{tile_px}.npz", **payload)


@pytest.fixture
def probe(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "OUT_DIR", tmp_path / "fang_embeddings")
    monkeypatch.setattr(m, "TILE_PX", 32)
    monkeypatch.setattr(m, "CONTEXT_PX", 96)
    return m


def test_absent_store_is_recomputed(probe):
    keys = pd.DataFrame({"ti": [1, 2], "tj": [3, 4]})
    assert probe._cache_is_stale("ESP_X", keys) == "absent"


def test_matching_key_set_is_reused(probe):
    keys = pd.DataFrame({"ti": [1, 2, 3], "tj": [10, 11, 12]})
    _write_store(probe.OUT_DIR, "ESP_X", keys.ti, keys.tj, ctx_px=96, tile_px=32)
    assert probe._cache_is_stale("ESP_X", keys) is None, "an exact match must still skip"


def test_a_grown_pool_forces_recompute(probe):
    """The real v2 case: the label pool gained tiles the cached store never saw."""
    cached = pd.DataFrame({"ti": [1, 2, 3], "tj": [10, 11, 12]})
    _write_store(probe.OUT_DIR, "ESP_X", cached.ti, cached.tj, ctx_px=96, tile_px=32)
    grown = pd.DataFrame({"ti": [1, 2, 3, 4], "tj": [10, 11, 12, 13]})
    why = probe._cache_is_stale("ESP_X", grown)
    assert why is not None and "1 new tiles missing" in why


def test_tiles_that_moved_away_also_force_recompute(probe):
    """Tiles moved BOTH ways on the real rebuild -- a pure size check would miss this."""
    cached = pd.DataFrame({"ti": [1, 2, 3], "tj": [10, 11, 12]})
    _write_store(probe.OUT_DIR, "ESP_X", cached.ti, cached.tj, ctx_px=96, tile_px=32)
    shifted = pd.DataFrame({"ti": [1, 2, 9], "tj": [10, 11, 99]})   # same COUNT, different set
    why = probe._cache_is_stale("ESP_X", shifted)
    assert why is not None, "same-size but different key set must not be reused"
    assert "1 cached tiles gone" in why


def test_half_written_store_is_recomputed(probe):
    """Only the context npz present -- the own-tile one missing."""
    keys = pd.DataFrame({"ti": [1], "tj": [2]})
    probe.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probe.OUT_DIR / "ESP_X_P96.npz",
                        ti=np.array([1], np.int32), tj=np.array([2], np.int32))
    assert probe._cache_is_stale("ESP_X", keys) == "absent"
