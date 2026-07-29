"""End-to-end tests for `scripts/f_region_staged.py` (PLAN_FBuild Stage D composite driver).

Driven over synthetic Stage-B npzs + a synthetic reference map tile, because the real Stage-B logits
are still being produced on Sherlock. Covers the composite arithmetic, the three offset variants, the
H6 provenance layers, and — most importantly — the §0.1 guard-1 rule that an AMBIGUOUS trend-guard
verdict must NOT silently ship a headline map.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import fcompose as fc
from src import leveling as lv

PITCH = 159.9991835298017
ORIGIN = (-711136.371096145, 2133729.111655494)     # E-12_N32's real origin
TILE = "T00_N00"
SIDE = 24


def _write_ref(map_dir, tile=TILE, side=SIDE):
    """A stand-in for reports/map_region/{tile}_prob_raw.tif — Stage D only reads its grid."""
    import rasterio
    from rasterio.transform import Affine

    map_dir.mkdir(parents=True, exist_ok=True)
    tf = Affine(PITCH, 0.0, ORIGIN[0], 0.0, -PITCH, ORIGIN[1])
    crs = ('PROJCS["Mars_2015_Ocentric_Equirectangular_clon_0",GEOGCS["GCS_Mars_2015_Ocentric",'
           'DATUM["Mars_2015",SPHEROID["Mars_2015",3396190,169.894447223612]],'
           'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
           'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
           'PARAMETER["central_meridian",0],UNIT["metre",1]]')
    with rasterio.open(map_dir / f"{tile}_prob_raw.tif", "w", driver="GTiff", height=side,
                       width=side, count=1, dtype="float32", crs=crs, transform=tf,
                       nodata=np.nan) as ds:
        ds.write(np.zeros((side, side), np.float32), 1)
    return fc.tile_grid_from_raster(map_dir / f"{tile}_prob_raw.tif", tile)


def _write_stage_b(logits_dir, grid, biases, seed=0):
    """n frames, each a biased view of one shared truth field over the whole tile."""
    rng = np.random.default_rng(seed)
    logits_dir.mkdir(parents=True, exist_ok=True)
    ti_lo, ti_hi = grid.TI_range()
    tj_lo, tj_hi = grid.TJ_range()
    TI, TJ = [a.ravel() for a in np.meshgrid(np.arange(ti_lo, ti_hi + 1),
                                             np.arange(tj_lo, tj_hi + 1), indexing="ij")]
    truth = rng.normal(-1.0, 1.0, TI.size)
    pids = []
    for k, b in enumerate(biases):
        pid = f"P{k:02d}_000000_2000_XN_20N000W"
        pids.append(pid)
        np.savez_compressed(logits_dir / f"{pid}.npz", TI=TI, TJ=TJ,
                            prob=lv.sigmoid(truth + b).astype(np.float32))
        (logits_dir / f"{pid}.json").write_text(json.dumps({
            "PRODUCT_ID": pid, "n_tiles": int(TI.size), "frame_median": 1.0,
            "prob_mean": float(lv.sigmoid(truth + b).mean())}), encoding="utf-8")
    return pids, TI, TJ, truth


def _write_stagec(fig, pids, biases, verdict="FULL", apply_="full", needs_ruling=False,
                  sources=None, resid_scale=0.5):
    fig.mkdir(parents=True, exist_ok=True)
    off = -(np.asarray(biases, float) - np.median(biases))
    pd.DataFrame({"PRODUCT_ID": pids, "offset_logit": off,
                  "offset_residual_only": off * resid_scale,
                  "offset_source": sources or ["solved"] * len(pids),
                  "incidence": 40.0 + np.arange(len(pids)) * 5.0,
                  "component": 0, "degree": 3}).to_csv(fig / "fbuild_stagec_offsets.csv", index=False)
    pd.DataFrame([{"verdict": verdict, "apply": apply_, "needs_ruling": needs_ruling,
                   "why": "synthetic", "lambda_star": 12.3}]).to_csv(fig / "fbuild_trend_guard.csv",
                                                                     index=False)
    return off


def _synth_calibration(path):
    """Identity Tier-1, abundance = 0.3 * p Tier-2 — enough to prove the wiring and the ceiling."""
    from src.calibration import CalibrationLayer

    x = np.linspace(0.0, 1.0, 64)
    path.parent.mkdir(parents=True, exist_ok=True)
    CalibrationLayer((x, x), (x, 0.3 * x), meta={"synthetic": True}).save(path)
    return path


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    import scripts.f_region_staged as sd

    fig = tmp_path / "figures"
    fig.mkdir()
    monkeypatch.setattr(sd, "FIG", fig)
    return sd, tmp_path, fig


def _run(sd, tmp_path, fig, monkeypatch, *extra):
    argv = ["f_region_staged.py",
            "--logits-dir", str(tmp_path / "logits"),
            "--offsets", str(fig / "fbuild_stagec_offsets.csv"),
            "--guard", str(fig / "fbuild_trend_guard.csv"),
            "--map-dir", str(tmp_path / "map_region"),
            "--out-dir", str(tmp_path / "out"),
            "--tiles", TILE, *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert sd.main() == 0


def _read(path):
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1)


# --------------------------------------------------------------------------- composite correctness
def test_full_variant_cancels_the_planted_biases(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    biases = [0.8, -0.3, 0.1]
    pids, TI, TJ, truth = _write_stage_b(tmp / "logits", grid, biases)
    off = _write_stagec(fig, pids, biases)
    _run(sd, tmp, fig, monkeypatch, "--raw")

    p = _read(tmp / "out" / f"{TILE}_full_prob_raw.tif")
    rows, cols = fc.frame_rows_cols(grid, TI, TJ)
    # offsets are exactly -(bias - median(bias)), so the leveled mean logit is truth + median(bias)
    want = lv.sigmoid(truth + np.median(biases))
    assert np.allclose(p[rows, cols], want, atol=1e-5)


def test_h1only_variant_is_the_unleveled_composite(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    biases = [0.8, -0.3, 0.1]
    pids, TI, TJ, truth = _write_stage_b(tmp / "logits", grid, biases)
    _write_stagec(fig, pids, biases)
    _run(sd, tmp, fig, monkeypatch, "--raw")

    p = _read(tmp / "out" / f"{TILE}_h1only_prob_raw.tif")
    rows, cols = fc.frame_rows_cols(grid, TI, TJ)
    assert np.allclose(p[rows, cols], lv.sigmoid(truth + np.mean(biases)), atol=1e-5)


def test_resid_variant_uses_the_residual_only_column(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    biases = [0.8, -0.3, 0.1]
    pids, TI, TJ, truth = _write_stage_b(tmp / "logits", grid, biases)
    _write_stagec(fig, pids, biases, resid_scale=0.5)
    _run(sd, tmp, fig, monkeypatch, "--raw")

    rows, cols = fc.frame_rows_cols(grid, TI, TJ)
    p = _read(tmp / "out" / f"{TILE}_resid_prob_raw.tif")[rows, cols]
    off = -(np.asarray(biases) - np.median(biases))
    want = lv.sigmoid(truth + np.mean(np.asarray(biases) + 0.5 * off))
    assert np.allclose(p, want, atol=1e-5)
    # and it differs from the full-offset composite (else the test proves nothing)
    assert not np.allclose(p, _read(tmp / "out" / f"{TILE}_full_prob_raw.tif")[rows, cols], atol=1e-3)


def test_the_three_variants_come_from_one_stage_b_run(staged, monkeypatch):
    """PLAN §1 deliverable 5: H1-only / full / residual-only must all fall out of one pass."""
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.5, -0.5])
    _write_stagec(fig, pids, [0.5, -0.5])
    _run(sd, tmp, fig, monkeypatch, "--raw")
    for v in ("h1only", "full", "resid"):
        assert (tmp / "out" / f"{TILE}_{v}_prob_raw.tif").exists(), v


# --------------------------------------------------------------------------- H6 provenance
def test_h6_layers_are_written_and_variant_independent(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, TI, TJ, _ = _write_stage_b(tmp / "logits", grid, [0.4, -0.2, 0.0])
    _write_stagec(fig, pids, [0.4, -0.2, 0.0],
                  sources=["solved", "interpolated", "component_gauged"])
    _run(sd, tmp, fig, monkeypatch, "--raw")

    n = _read(tmp / "out" / f"{TILE}_n_frames.tif")
    assert np.nanmax(n) == 3
    src = _read(tmp / "out" / f"{TILE}_offset_source.tif")
    assert np.nanmax(src) == fc.OFFSET_SOURCE_CODE["interpolated"]      # worst contributor wins
    inc = _read(tmp / "out" / f"{TILE}_incidence.tif")
    assert np.nanmin(inc) == pytest.approx(40.0)                        # lowest-incidence frame
    prim = _read(tmp / "out" / f"{TILE}_primary_frame.tif")
    assert np.nanmax(prim) >= 0
    for v in ("h1only", "full", "resid"):
        assert (tmp / "out" / f"{TILE}_{v}_overlap_dp.tif").exists()


def test_overlap_dp_shrinks_when_the_offsets_are_applied(staged, monkeypatch):
    """The whole point of H4: co-located disagreement must fall after leveling."""
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [1.0, -1.0])
    _write_stagec(fig, pids, [1.0, -1.0])
    _run(sd, tmp, fig, monkeypatch, "--raw")

    before = np.nanmedian(_read(tmp / "out" / f"{TILE}_h1only_overlap_dp.tif"))
    after = np.nanmedian(_read(tmp / "out" / f"{TILE}_full_overlap_dp.tif"))
    assert after < 0.02 * before or after < 1e-6


def test_frames_without_an_offset_row_are_flagged_not_dropped(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.3, -0.3, 0.9])
    _write_stagec(fig, pids[:2], [0.3, -0.3])          # third frame has logits but no offset row
    _run(sd, tmp, fig, monkeypatch, "--raw")

    n = _read(tmp / "out" / f"{TILE}_n_frames.tif")
    assert np.nanmax(n) == 3                          # still composited
    src = _read(tmp / "out" / f"{TILE}_offset_source.tif")
    assert np.nanmax(src) == fc.OFFSET_SOURCE_CODE["none"]
    side = json.loads((tmp / "out" / f"{TILE}.json").read_text(encoding="utf-8"))
    assert pids[2] in side["frames_without_offset"]


def test_partition_layer_takes_each_pixels_owner_frame(staged, monkeypatch):
    """Gate 1 scores the PARTITION composite (one owner frame per pixel), not the mean one — it is
    the only labelling that is row-comparable to the mosaic map, which has one value per pixel."""
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    biases = [0.6, -0.6]
    pids, TI, TJ, truth = _write_stage_b(tmp / "logits", grid, biases)
    _write_stagec(fig, pids, biases)
    # left half owned by frame 0, right half by frame 1 (global lut order == sorted pids)
    labels = np.zeros(grid.shape, np.int32)
    labels[:, grid.width // 2:] = 1
    monkeypatch.setattr(sd, "seam_labels", lambda tile, g, lut: labels)
    _run(sd, tmp, fig, monkeypatch, "--raw")

    part = _read(tmp / "out" / f"{TILE}_full_prob_partition.tif")
    off = -(np.asarray(biases) - np.median(biases))
    rows, cols = fc.frame_rows_cols(grid, TI, TJ)
    field = np.full(grid.shape, np.nan)
    field[rows, cols] = truth
    left = lv.sigmoid(field[:, :grid.width // 2] + biases[0] + off[0])
    assert np.allclose(part[:, :grid.width // 2], left, atol=1e-5, equal_nan=True)
    # Partition (single owner) vs mean (all covering frames) can only differ where the frames
    # DISAGREE — under the full offsets they agree exactly by construction, so the discriminating
    # comparison is on the UN-leveled variant.
    part0 = _read(tmp / "out" / f"{TILE}_h1only_prob_partition.tif")
    mean0 = _read(tmp / "out" / f"{TILE}_h1only_prob_raw.tif")
    assert not np.allclose(part0, mean0, atol=1e-4, equal_nan=True)
    assert np.allclose(part[:, :grid.width // 2],
                       _read(tmp / "out" / f"{TILE}_full_prob_raw.tif")[:, :grid.width // 2],
                       atol=1e-5, equal_nan=True)      # leveled frames agree -> both composites match
    side = json.loads((tmp / "out" / f"{TILE}.json").read_text(encoding="utf-8"))
    assert side["per_variant"]["full"]["partition_coverage"] > 0.9


def test_partition_layer_is_skipped_when_seammap_is_unavailable(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.2, -0.2])
    _write_stagec(fig, pids, [0.2, -0.2])
    _run(sd, tmp, fig, monkeypatch, "--raw", "--no-partition")
    assert not (tmp / "out" / f"{TILE}_full_prob_partition.tif").exists()
    assert (tmp / "out" / f"{TILE}_full_prob_raw.tif").exists()


# --------------------------------------------------------------------------- the guard-1 rule
def test_ambiguous_verdict_writes_no_headline_map(staged, monkeypatch):
    """§0.1 guard 1 / §7 Q3: an AMBIGUOUS attribution must escalate, not ship full offsets."""
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.5, -0.5])
    _write_stagec(fig, pids, [0.5, -0.5], verdict="AMBIGUOUS",
                  apply_="full_pending_ruling", needs_ruling=True)
    _run(sd, tmp, fig, monkeypatch, "--raw")

    assert not (tmp / "out" / f"{TILE}_prob_raw.tif").exists()
    assert (tmp / "out" / f"{TILE}_full_prob_raw.tif").exists()       # variants still written
    assert (tmp / "out" / f"{TILE}_resid_prob_raw.tif").exists()
    side = json.loads((tmp / "out" / f"{TILE}.json").read_text(encoding="utf-8"))
    assert side["headline_variant"] is None and side["trend_guard"]["needs_ruling"] is True


def test_headline_override_ships_an_explicitly_named_variant(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.5, -0.5])
    _write_stagec(fig, pids, [0.5, -0.5], verdict="AMBIGUOUS",
                  apply_="full_pending_ruling", needs_ruling=True)
    _run(sd, tmp, fig, monkeypatch, "--raw", "--headline", "resid")

    assert (tmp / "out" / f"{TILE}_prob_raw.tif").exists()
    assert np.allclose(_read(tmp / "out" / f"{TILE}_prob_raw.tif"),
                       _read(tmp / "out" / f"{TILE}_resid_prob_raw.tif"), equal_nan=True)


def test_residual_verdict_ships_the_residual_variant(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.5, -0.5])
    _write_stagec(fig, pids, [0.5, -0.5], verdict="RESIDUAL_ONLY", apply_="residual")
    _run(sd, tmp, fig, monkeypatch, "--raw")
    assert np.allclose(_read(tmp / "out" / f"{TILE}_prob_raw.tif"),
                       _read(tmp / "out" / f"{TILE}_resid_prob_raw.tif"), equal_nan=True)


# --------------------------------------------------------------------------- calibration wiring
def test_calibration_layers_produce_prob_and_abundance(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.4, -0.4])
    _write_stagec(fig, pids, [0.4, -0.4])
    cal = _synth_calibration(tmp / "cal" / "calibration.npz")
    _run(sd, tmp, fig, monkeypatch, "--calibration-f", str(cal),
         "--calibration-mosaic", str(cal))

    p = _read(tmp / "out" / f"{TILE}_full_prob_raw.tif")
    ab = _read(tmp / "out" / f"{TILE}_full_abundance.tif")
    fin = np.isfinite(p)
    assert np.allclose(ab[fin], 0.3 * p[fin], atol=1e-5)          # the synthetic Tier-2 map
    assert (tmp / "out" / f"{TILE}_full_abundance_moscal.tif").exists()
    assert (tmp / "out" / f"{TILE}_prob.tif").exists()            # headline plain names
    assert (tmp / "out" / f"{TILE}_abundance.tif").exists()


def test_calibration_is_applied_once_to_the_composite_not_per_frame(staged, monkeypatch):
    """calibrate_abundance is nonlinear, so mean-then-calibrate != calibrate-then-mean. Pin the
    declared order (calibrate the composited P) with a deliberately convex Tier-2 map."""
    from src.calibration import CalibrationLayer

    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    biases = [1.2, -1.2]
    pids, TI, TJ, truth = _write_stage_b(tmp / "logits", grid, biases)
    _write_stagec(fig, pids, biases)
    x = np.linspace(0.0, 1.0, 256)
    cal = tmp / "cal" / "convex.npz"
    cal.parent.mkdir(parents=True, exist_ok=True)
    CalibrationLayer((x, x), (x, x ** 3), meta={"synthetic": "convex"}).save(cal)
    _run(sd, tmp, fig, monkeypatch, "--calibration-f", str(cal))

    rows, cols = fc.frame_rows_cols(grid, TI, TJ)
    p = _read(tmp / "out" / f"{TILE}_full_prob_raw.tif")[rows, cols]
    ab = _read(tmp / "out" / f"{TILE}_full_abundance.tif")[rows, cols]
    assert np.allclose(ab, p ** 3, atol=2e-5)
    per_frame_then_mean = np.mean([lv.sigmoid(truth + b) ** 3 for b in biases], axis=0)
    assert not np.allclose(ab, per_frame_then_mean, atol=1e-3)


# --------------------------------------------------------------------------- bookkeeping
def test_registration_report_records_the_subpixel_translation(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.2, -0.2])
    _write_stagec(fig, pids, [0.2, -0.2])
    _run(sd, tmp, fig, monkeypatch, "--raw")

    reg = pd.read_csv(fig / "fbuild_staged_registration.csv")
    assert list(reg.tile) == [TILE]
    assert reg.Kj.iloc[0] == grid.Kj and reg.Ki.iloc[0] == grid.Ki
    assert 0.0 <= reg.dx_m.iloc[0] <= 80.0 and 0.0 <= reg.dy_m.iloc[0] <= 80.0
    tiles = pd.read_csv(fig / "fbuild_staged_tiles.csv")
    assert set(tiles.variant) == {"h1only", "full", "resid"}
    assert (tiles.coverage > 0.9).all()


def test_rerun_skips_completed_tiles_unless_overwritten(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.3, -0.3])
    _write_stagec(fig, pids, [0.3, -0.3])
    _run(sd, tmp, fig, monkeypatch, "--raw")
    first = _read(tmp / "out" / f"{TILE}_full_prob_raw.tif").copy()
    (fig / "fbuild_staged_registration.csv").unlink()
    _run(sd, tmp, fig, monkeypatch, "--raw")                       # skips -> writes no report
    assert not (fig / "fbuild_staged_registration.csv").exists()
    _run(sd, tmp, fig, monkeypatch, "--raw", "--overwrite")
    assert np.allclose(first, _read(tmp / "out" / f"{TILE}_full_prob_raw.tif"), equal_nan=True)


def test_frame_index_and_lut_are_cached(staged, monkeypatch):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.1, -0.1])
    _write_stagec(fig, pids, [0.1, -0.1])
    _run(sd, tmp, fig, monkeypatch, "--raw")

    idx = pd.read_csv(tmp / "out" / "frame_index.csv")
    assert set(idx.PRODUCT_ID) == set(pids)
    assert (idx.n_tiles > 0).all() and (idx.TI_max >= idx.TI_min).all()
    lut = pd.read_csv(tmp / "out" / "frame_lut.csv")
    assert list(lut.frame_idx) == sorted(lut.frame_idx)


def test_missing_reference_grid_is_skipped_loudly(staged, monkeypatch, capsys):
    sd, tmp, fig = staged
    grid = _write_ref(tmp / "map_region")
    pids, *_ = _write_stage_b(tmp / "logits", grid, [0.1, -0.1])
    _write_stagec(fig, pids, [0.1, -0.1])
    monkeypatch.setattr(sys, "argv", [
        "f_region_staged.py", "--logits-dir", str(tmp / "logits"),
        "--offsets", str(fig / "fbuild_stagec_offsets.csv"),
        "--guard", str(fig / "fbuild_trend_guard.csv"),
        "--map-dir", str(tmp / "map_region"), "--out-dir", str(tmp / "out"),
        "--tiles", "NOT_A_TILE", "--raw"])
    assert sd.main() == 0
    assert "no reference raster" in capsys.readouterr().out
