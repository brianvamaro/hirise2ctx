"""End-to-end smoke test for `scripts/f_region_stagec.py` (PLAN_FBuild Stage C).

Stage C consumes a Sherlock-day's worth of Stage-B npzs, so it has to be known-good BEFORE the real
inputs land. This drives the actual `main()` over synthetic per-frame npzs with planted per-frame
biases and checks (a) every declared artifact is written, (b) the solved offsets cancel the planted
biases, (c) the residual-only column and the trend verdict are populated.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import leveling as lv


def _write_stage_b_outputs(d, n_frames=10, side=40, overlap=28, seed=0):
    """Synthetic Stage-B npz/json pairs: one shared P(rich) field seen through per-frame biases."""
    rng = np.random.default_rng(seed)
    bias = np.linspace(-0.6, 0.6, n_frames)
    rows = np.arange(17_000, 17_060)
    truth = {}
    for f in range(n_frames):
        cols = np.arange(-400 + f * (side - overlap), -400 + f * (side - overlap) + side)
        ti, tj = [a.ravel() for a in np.meshgrid(rows, cols, indexing="ij")]
        key = lv.pack_key(ti, tj)
        for k in key.tolist():
            truth.setdefault(k, rng.normal(-1.0, 1.2))
        lg = np.array([truth[k] for k in key.tolist()]) + bias[f]
        pid = f"P{f:02d}_000000_2000_XN_20N000W"
        np.savez_compressed(d / f"{pid}.npz", TI=ti, TJ=tj,
                            prob=lv.sigmoid(lg).astype(np.float32))
        (d / f"{pid}.json").write_text(json.dumps({
            "PRODUCT_ID": pid, "index_incidence": 45.0 + f, "subsolar_lat": 20.0,
            "frame_median": float(np.exp(0.05 * f)), "n_tiles": int(ti.size),
            "prob_mean": float(lv.sigmoid(lg).mean()),
        }), encoding="utf-8")
    return bias


def _write_frame_csvs(fig, n_frames=10):
    pids = [f"P{f:02d}_000000_2000_XN_20N000W" for f in range(n_frames)]
    rng = np.random.default_rng(1)
    pd.DataFrame({"PRODUCT_ID": pids, "VOLUME_ID": "MROX_0000",
                  "image_time": [f"20{8 + f % 9:02d}-03-0{1 + f % 8}T06:00:00.000" for f in range(n_frames)],
                  }).to_csv(fig / "region_frame_list.csv", index=False)
    pd.DataFrame({"PRODUCT_ID": pids, "incidence": 45.0 + np.arange(n_frames),
                  "center_lat": 40.0 + rng.uniform(-3, 3, n_frames),
                  "center_lon": (355.0 + np.arange(n_frames)) % 360.0,
                  "subsolar_lat": 20.0, "subsolar_lon": 60.0,
                  }).to_csv(fig / "region_frame_incidence.csv", index=False)


@pytest.fixture()
def stagec(tmp_path, monkeypatch):
    import scripts.f_region_stagec as sc

    fig, work, logits = tmp_path / "figures", tmp_path / "work", tmp_path / "logits"
    for p in (fig, work, logits):
        p.mkdir()
    monkeypatch.setattr(sc, "FIG", fig)
    monkeypatch.setattr(sc, "WORK", work)
    monkeypatch.setattr(sc, "FRAME_LIST", fig / "region_frame_list.csv")
    monkeypatch.setattr(sc, "INC_CSV", fig / "region_frame_incidence.csv")
    return sc, fig, work, logits


def _run(sc, logits, monkeypatch, *extra):
    argv = ["f_region_stagec.py", "--logits-dir", str(logits), "--min-tiles", "10",
            "--cache-min-tiles", "5", "--perm-draws", "50", "--cv-repeats", "2",
            "--cv-frac", "0.2", "--mola", "/nope/mola.tif", "--themis", "/nope/themis.tif",
            *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert sc.main() == 0


def test_stage_c_end_to_end_recovers_planted_offsets(stagec, monkeypatch):
    sc, fig, work, logits = stagec
    bias = _write_stage_b_outputs(logits)
    _write_frame_csvs(fig)
    _run(sc, logits, monkeypatch)

    off = pd.read_csv(fig / "fbuild_stagec_offsets.csv")
    assert len(off) == len(bias)
    expect = -(bias - np.median(bias))
    assert np.allclose(off["offset_logit"], expect, atol=0.05)
    # offsets decompose into the smooth surface + residual, and residual-only is gauged separately
    assert np.allclose(off["trend_fitted"] + off["trend_resid"], off["offset_logit"], atol=1e-3)
    assert np.median(off["offset_residual_only"]) == pytest.approx(0.0, abs=1e-3)
    assert (off["component"] == 0).all() and (off["degree"] > 0).all()


def test_stage_c_writes_every_declared_artifact(stagec, monkeypatch):
    sc, fig, work, logits = stagec
    _write_stage_b_outputs(logits)
    _write_frame_csvs(fig)
    _run(sc, logits, monkeypatch)

    for name in ("fbuild_stagec_offsets.csv", "fbuild_stagec_lambda.csv",
                 "fbuild_stagec_graph.csv", "fbuild_trend_guard.csv",
                 "fbuild_stagec_attribution.csv", "fbuild_stagec_watchlist.csv",
                 "fbuild_trend_guard.png", "fbuild_stagec_offsets.png"):
        assert (fig / name).exists(), name
    assert (work / "stagec_edges_min5.npz").exists()

    guard = pd.read_csv(fig / "fbuild_trend_guard.csv").iloc[0]
    assert guard["verdict"] in {"NO_TREND", "FULL", "RESIDUAL_ONLY", "AMBIGUOUS"}
    assert guard["full_heldout_cv_dp"] < guard["baseline_dp"]      # gate 2, in miniature
    assert guard["n_components"] == 1


def test_stage_c_censuses_missing_frames(stagec, monkeypatch):
    """Stage B is resumable and may be incomplete — Stage C must say so, not quietly solve a subset."""
    sc, fig, work, logits = stagec
    _write_stage_b_outputs(logits)
    _write_frame_csvs(fig, n_frames=13)                 # 3 frames planned but never produced
    _run(sc, logits, monkeypatch)

    miss = pd.read_csv(fig / "fbuild_stagec_missing_frames.csv")
    assert len(miss) == 3 and (miss["status"] == "no_logits").all()


def test_stage_c_reuses_the_edge_cache(stagec, monkeypatch):
    sc, fig, work, logits = stagec
    _write_stage_b_outputs(logits)
    _write_frame_csvs(fig)
    _run(sc, logits, monkeypatch)
    first = pd.read_csv(fig / "fbuild_stagec_offsets.csv")["offset_logit"].to_numpy()
    _run(sc, logits, monkeypatch)                       # second pass hits the cache
    assert np.allclose(first, pd.read_csv(fig / "fbuild_stagec_offsets.csv")["offset_logit"])
