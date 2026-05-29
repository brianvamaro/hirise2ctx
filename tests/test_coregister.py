"""Stage 3 (co-registration) tests.

Unit tests use synthetic arrays so they run in milliseconds without touching caches.
The integration test on ESP_069669_2220 is marked `slow` and auto-skips when its
Stage 2 caches aren't on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import shift as nd_shift

from src.coregister import (
    COREGISTRATION_SUBDIR,
    block_shift_field,
    find_tracking_features,
    phase_correlate_translation,
    select_fft_window,
    stage3_one_image,
)
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR


# -------------------------------------------------------------------------
# Synthetic FFT window selection
# -------------------------------------------------------------------------

def test_select_fft_window_returns_power_of_two_inside_mask():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[64:448, 64:448] = 1  # 384x384 fully-covered interior
    size, r, c = select_fft_window(mask, max_px=256)
    assert size == 256, f"largest power-of-2 ≤ 256 that fits in a 384x384 interior is 256, got {size}"
    block = mask[r : r + size, c : c + size]
    assert block.sum() == size * size


def test_select_fft_window_steps_down_when_max_doesnt_fit():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[150:330, 150:330] = 1  # 180x180 covered region — 256 won't fit; 128 will
    size, r, c = select_fft_window(mask, max_px=256)
    assert size == 128
    block = mask[r : r + size, c : c + size]
    assert block.sum() == size * size


def test_select_fft_window_raises_on_empty_mask():
    mask = np.zeros((256, 256), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="all zero"):
        select_fft_window(mask, max_px=128)


def test_select_fft_window_raises_when_min_size_doesnt_fit():
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:130, 100:130] = 1  # 30x30 — too small for min_px=64
    with pytest.raises(RuntimeError, match="no power-of-2"):
        select_fft_window(mask, max_px=128, min_px=64)


# -------------------------------------------------------------------------
# Phase correlation on synthetic data
# -------------------------------------------------------------------------

def _synthetic_texture(size: int, seed: int = 0) -> np.ndarray:
    """Bandpass-filtered noise — rich phase spectrum, no DC dominance."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((size, size)).astype(np.float32)
    # Smooth slightly so the gradients are realistic (pure white noise has too much
    # high-frequency content; our 5 m/px imagery has structure on tens-of-pixels scale).
    from scipy.ndimage import gaussian_filter
    arr = gaussian_filter(arr, sigma=2.0)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


@pytest.mark.parametrize("true_dy,true_dx", [(0.0, 0.0), (1.5, -2.7), (-3.2, 4.4), (0.25, 0.0)])
def test_phase_correlation_recovers_known_shift(true_dy: float, true_dx: float):
    size = 256
    ref = _synthetic_texture(size, seed=42)
    # Shift the reference by (true_dy, true_dx); phase_correlate_translation should
    # recover the shift that aligns `moving` back to `ref` — i.e. the negative.
    moving = nd_shift(ref, shift=(true_dy, true_dx), order=3, mode="reflect")
    dy, dx, peak = phase_correlate_translation(ref, moving, upsample_factor=20)
    assert dy == pytest.approx(-true_dy, abs=0.15), f"dy {dy} vs expected {-true_dy}"
    assert dx == pytest.approx(-true_dx, abs=0.15), f"dx {dx} vs expected {-true_dx}"
    if abs(true_dy) + abs(true_dx) > 0:
        assert peak > 0.7, f"peak correlation {peak} too low for clean synthetic recovery"


def test_phase_correlation_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        phase_correlate_translation(np.zeros((64, 64)), np.zeros((64, 32)))


# -------------------------------------------------------------------------
# Whole-image block shift field (Stage 3 QA — coregister.block_shift_field)
# -------------------------------------------------------------------------

def test_block_shift_field_recovers_uniform_shift():
    """A globally-shifted texture should yield a coherent per-block field that recovers
    the known shift at every fully-covered block with high peak."""
    size = 512
    true_dy, true_dx = 3.0, -2.0
    ctx = _synthetic_texture(size, seed=7)
    # `hi` is ctx shifted by (true_dy, true_dx); block_shift_field solves the shift that
    # brings hi back onto ctx -> expect (-true_dy, -true_dx) in every block.
    hi = nd_shift(ctx, shift=(true_dy, true_dx), order=3, mode="reflect")
    mask = np.ones((size, size), dtype=np.uint8)
    field = block_shift_field(hi, ctx, mask, block_px=128, min_coverage=0.98)
    assert len(field) == 16, f"4x4 non-overlapping 128px blocks expected, got {len(field)}"
    dys = np.array([b["dy_px"] for b in field])
    dxs = np.array([b["dx_px"] for b in field])
    peaks = np.array([b["peak"] for b in field])
    assert np.median(dys) == pytest.approx(-true_dy, abs=0.3)
    assert np.median(dxs) == pytest.approx(-true_dx, abs=0.3)
    # Coherent field: tight spread + high confidence everywhere.
    assert dys.std() < 0.5 and dxs.std() < 0.5
    assert (peaks > 0.7).mean() > 0.9


def test_block_shift_field_skips_undercovered_blocks():
    """Blocks below `min_coverage` (mask or CTX-nodata) are not evaluated."""
    size = 256
    ctx = _synthetic_texture(size, seed=3)
    hi = nd_shift(ctx, shift=(1.0, 1.0), order=3, mode="reflect")
    mask = np.ones((size, size), dtype=np.uint8)
    mask[:128, :] = 0  # top half uncovered -> top row of 128px blocks must be dropped
    field = block_shift_field(hi, ctx, mask, block_px=128, min_coverage=0.98)
    assert len(field) == 2, f"only the bottom row of 2 blocks is fully covered, got {len(field)}"
    assert all(b["row_off"] == 128 for b in field)


# -------------------------------------------------------------------------
# Tracking feature selection (used by the QA notebook's marker overlay)
# -------------------------------------------------------------------------

def test_find_tracking_features_returns_distinct_peaks():
    """Synthetic image with 4 isolated bright spots — should find all 4 within tolerance."""
    img = np.zeros((128, 128), dtype=np.float32)
    truth = [(20, 30), (40, 90), (80, 25), (100, 100)]
    for r, c in truth:
        img[r, c] = 1.0
    coords = find_tracking_features(img, n_features=8, min_distance=10, edge_margin=5)
    assert len(coords) == 4
    # Every truth point should be within 3 px of one returned coord (Gaussian smoothing
    # can nudge the peak by a couple of pixels).
    for r, c in truth:
        d2 = ((coords[:, 0] - r) ** 2 + (coords[:, 1] - c) ** 2)
        assert d2.min() <= 9, f"truth ({r}, {c}) not within 3 px of any returned coord"


def test_find_tracking_features_respects_edge_margin():
    """Peaks within edge_margin must not be returned."""
    img = np.zeros((100, 100), dtype=np.float32)
    img[2, 2] = 1.0   # inside margin — should be rejected
    img[50, 50] = 1.0  # interior — should be kept
    coords = find_tracking_features(img, n_features=8, min_distance=10, edge_margin=12)
    assert len(coords) == 1
    assert tuple(coords[0]) == (50, 50) or abs(coords[0][0] - 50) <= 2  # Gaussian nudge


def test_find_tracking_features_returns_empty_on_uniform_image():
    """A flat image has no peaks — the function should return an empty array, not raise."""
    img = np.full((64, 64), 0.5, dtype=np.float32)
    coords = find_tracking_features(img, n_features=8)
    assert coords.shape[0] == 0


# -------------------------------------------------------------------------
# Stage 3 integration on ESP_069669_2220 (slow, gated on Stage 2 caches)
# -------------------------------------------------------------------------

OBS_ID = "ESP_069669_2220"


def _stage2_outputs_exist(cache_dir: Path) -> bool:
    return (
        (cache_dir / CTX_WINDOWS_SUBDIR / f"{OBS_ID}.tif").exists()
        and (cache_dir / CTX_WINDOWS_SUBDIR / f"{OBS_ID}_hirise_mask.tif").exists()
        and (cache_dir / "hirise_jp2" / f"{OBS_ID}_RED.JP2").exists()
    )


@pytest.mark.slow
def test_stage3_runs_on_ESP_069669_2220(cfg):
    """End-to-end Stage 3 on the canonical trusted_prj image.

    Asserts the writeable provenance JSON appears and the solved shift sits inside the
    O(200 m) acceptance band from CLAUDE.md §3.3. No peak-correlation threshold yet —
    the 2026-05-21 decision is to collect the distribution across all 10 images first.
    """
    cache_dir = cfg.cache_dir
    if not _stage2_outputs_exist(cache_dir):
        pytest.skip(
            f"{OBS_ID}: Stage 2 caches missing; run scripts/run_stage2.py {OBS_ID} first."
        )

    from src import manifest as M

    df = M.load_manifest(cfg.manifest_path)
    row = df.set_index("ObsId").loc[OBS_ID]
    cfg_coreg = cfg["coregistration"]

    prov = stage3_one_image(
        OBS_ID,
        cache_dir=cache_dir,
        manifest_row=row,
        fft_window_px=int(cfg_coreg["fft_window_px"]),
        upsample_factor=20,
        config_hash=cfg.hash,
    )

    out_json = cache_dir / COREGISTRATION_SUBDIR / f"{OBS_ID}.json"
    assert out_json.exists(), "Stage 3 did not write provenance JSON"

    # Sanity: the FFT window must be a power of 2.
    size = prov["fft_window"]["size_px"]
    assert size & (size - 1) == 0 and size >= 64, f"FFT window size {size} not a power of 2 ≥ 64"

    # CLAUDE.md §3.3 acceptance: residual HiRISE↔CTX offset should be O(200 m), not km.
    # CTX mosaic registration error is ~200 m, so we expect shifts in the 0-400 m range
    # for a trusted_prj image. Anything in km territory means the CRS handling is wrong.
    mag = prov["shift_m"]["magnitude"]
    assert mag < 1000.0, (
        f"{OBS_ID}: solved shift |{mag:.1f}| m is in km territory — CRS handling is likely "
        "wrong (re-check the per-image local-radius reprojection from Stage 1)."
    )

    # Peak correlation should be a finite number on a real image with texture.
    peak = prov["peak_correlation"]
    assert np.isfinite(peak), f"peak correlation is not finite: {peak}"


@pytest.mark.slow
def test_stage3_is_idempotent(cfg):
    """Re-running Stage 3 produces an identical shift (deterministic FFT on cached input)."""
    cache_dir = cfg.cache_dir
    if not _stage2_outputs_exist(cache_dir):
        pytest.skip(f"{OBS_ID}: Stage 2 caches missing")

    from src import manifest as M

    df = M.load_manifest(cfg.manifest_path)
    row = df.set_index("ObsId").loc[OBS_ID]
    cfg_coreg = cfg["coregistration"]

    p1 = stage3_one_image(
        OBS_ID,
        cache_dir=cache_dir,
        manifest_row=row,
        fft_window_px=int(cfg_coreg["fft_window_px"]),
        upsample_factor=20,
        config_hash=cfg.hash,
    )
    p2 = stage3_one_image(
        OBS_ID,
        cache_dir=cache_dir,
        manifest_row=row,
        fft_window_px=int(cfg_coreg["fft_window_px"]),
        upsample_factor=20,
        config_hash=cfg.hash,
    )
    assert p1["shift_px"] == p2["shift_px"]
    assert p1["fft_window"] == p2["fft_window"]
