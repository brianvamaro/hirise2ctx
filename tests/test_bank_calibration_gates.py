"""The calibration banker must take explicit paths and refuse to write a failing layer.

Three defects, all in `scripts/bank_calibration.py` and all named by the 2026-08-06 audit:

- every path was hard-coded, so a scratch rebuild could not run without writing live
  `models/`;
- `layer.save(OUT)` ran **before** the promotion gates were evaluated, and `main` returned
  0 regardless — a run that printed `FAIL` still overwrote the banked calibrator and still
  reported success;
- the predictions↔labels merge was a bare `how="inner"`, so any key that failed to join
  silently left the calibration pool.

Synthetic parquets in `tmp_path`; `--out` never points at a repository path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_banker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bank_calibration.py"
    spec = importlib.util.spec_from_file_location("bank_calibration_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_pool(tmp_path: Path, n_images: int = 6, n_tiles: int = 120):
    """Predictions + per-image label parquets that join 1:1 and carry real signal."""
    rng = np.random.default_rng(0)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    pred_rows = []
    for i in range(n_images):
        obs = f"OBS_{i:03d}"
        y = (rng.random(n_tiles) < 0.35).astype(int)
        p = np.clip(0.25 * rng.normal(size=n_tiles) + 0.3 + 0.4 * y, 0.001, 0.999)
        fa = np.where(y == 1, np.abs(rng.normal(0.03, 0.01, n_tiles)), 0.0)
        keys = dict(obs_id=obs, ti=np.arange(n_tiles), tj=0)
        pred_rows.append(pd.DataFrame({**keys, "y_true": y, "y_pred": p}))
        pd.DataFrame({
            **keys, "tile_size_px": 32, "fractional_area": fa,
        }).to_parquet(labels_dir / f"{obs}.parquet", index=False)
    preds = tmp_path / "predictions.parquet"
    pd.concat(pred_rows, ignore_index=True).to_parquet(preds, index=False)
    return preds, labels_dir


def _force_gates(monkeypatch, mod, *, ece: float, top_ratio: float):
    monkeypatch.setattr(mod, "expected_calibration_error", lambda *a, **k: ece)
    monkeypatch.setattr(mod, "compression_metrics", lambda *a, **k: {
        "top_ratio": top_ratio, "near_zero_pred": 0.5, "near_zero_true": 0.5,
        "marginal_l1": 0.01, "spearman": 0.5,
    })


def test_banker_writes_to_an_explicit_out_path_when_gates_pass(tmp_path, monkeypatch):
    """Isolation criterion 4: the rebuild must be able to bank into a scratch tree."""
    mod = _load_banker()
    preds, labels_dir = _synthetic_pool(tmp_path)
    out = tmp_path / "scratch" / "calibration.npz"
    _force_gates(monkeypatch, mod, ece=0.01, top_ratio=1.0)

    rc = mod.main(["--predictions", str(preds), "--labels-dir", str(labels_dir),
                   "--out", str(out)])
    assert rc == 0
    assert out.exists(), "the layer was not written to the requested path"


@pytest.mark.parametrize("ece,top_ratio,which", [
    (0.20, 1.00, "ECE"),
    (0.01, 1.90, "top_ratio"),
    (0.20, 0.10, "both"),
])
def test_banker_writes_nothing_and_exits_nonzero_when_a_gate_fails(
    tmp_path, monkeypatch, capsys, ece, top_ratio, which,
):
    """The ordering defect: `save` used to run before the gates were even computed."""
    mod = _load_banker()
    preds, labels_dir = _synthetic_pool(tmp_path)
    out = tmp_path / "scratch" / "calibration.npz"
    _force_gates(monkeypatch, mod, ece=ece, top_ratio=top_ratio)

    rc = mod.main(["--predictions", str(preds), "--labels-dir", str(labels_dir),
                   "--out", str(out)])
    assert rc == 1, f"{which} gate failed but the banker reported success"
    assert not out.exists(), f"{which} gate failed and the layer was written anyway"
    assert "GATE FAILURE" in capsys.readouterr().out


def test_an_existing_banked_layer_survives_a_failing_run(tmp_path, monkeypatch):
    """The consequential half: a failing re-fit must not clobber the layer in production."""
    mod = _load_banker()
    preds, labels_dir = _synthetic_pool(tmp_path)
    out = tmp_path / "scratch" / "calibration.npz"
    out.parent.mkdir()
    out.write_bytes(b"previously banked layer")

    _force_gates(monkeypatch, mod, ece=0.9, top_ratio=5.0)
    assert mod.main(["--predictions", str(preds), "--labels-dir", str(labels_dir),
                     "--out", str(out)]) == 1
    assert out.read_bytes() == b"previously banked layer"


def test_force_banks_a_failing_layer_and_records_that_it_did(tmp_path, monkeypatch):
    mod = _load_banker()
    preds, labels_dir = _synthetic_pool(tmp_path)
    out = tmp_path / "scratch" / "calibration.npz"
    _force_gates(monkeypatch, mod, ece=0.9, top_ratio=5.0)

    assert mod.main(["--predictions", str(preds), "--labels-dir", str(labels_dir),
                     "--out", str(out), "--force"]) == 0
    from src.calibration import CalibrationLayer
    meta = CalibrationLayer.load(out).meta
    assert meta["gates_passed"] is False and meta["forced"] is True, meta


# ---------------------------------------------------------------- join completeness

def test_incomplete_join_is_refused_rather_than_silently_shrinking_the_pool(tmp_path):
    """A prediction row with no label used to vanish into `how="inner"`. That is exactly
    how a recovered or dropped tile leaves the calibration pool unnoticed."""
    mod = _load_banker()
    preds, labels_dir = _synthetic_pool(tmp_path, n_images=3, n_tiles=40)
    victim = labels_dir / "OBS_001.parquet"
    pd.read_parquet(victim).iloc[:-5].to_parquet(victim, index=False)

    with pytest.raises(SystemExit, match="incomplete join"):
        mod.main(["--predictions", str(preds), "--labels-dir", str(labels_dir),
                  "--out", str(tmp_path / "scratch" / "c.npz")])


def test_duplicate_join_keys_are_refused(tmp_path):
    mod = _load_banker()
    preds, labels_dir = _synthetic_pool(tmp_path, n_images=3, n_tiles=40)
    victim = labels_dir / "OBS_002.parquet"
    df = pd.read_parquet(victim)
    pd.concat([df, df.iloc[:1]], ignore_index=True).to_parquet(victim, index=False)

    with pytest.raises(SystemExit, match="duplicate"):
        mod.main(["--predictions", str(preds), "--labels-dir", str(labels_dir),
                  "--out", str(tmp_path / "scratch" / "c.npz")])
