"""Stage 4b (feature extraction) tests.

Fast unit tests use synthetic CTX windows + labels parquets -- no caches, no downloads.
The slow integration test on ESP_069669_2220 (`@pytest.mark.slow`) requires the existing
Stage 2 + Stage 4 caches and asserts a 1:1 row join with the labels parquet, plus the
"don't crash on real data" basics.

The unit-test fixtures intentionally mirror tests/test_labeling.py's `_make_window` so
the two test files stay structurally parallel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

from src.features import (
    DEFAULT_FEATURES_CFG,
    FEATURES_SUBDIR,
    _compute_dn_thresholds,
    _glcm_per_tile,
    _gradient_stats_per_tile,
    _intensity_stats_per_tile,
    _lacunarity_per_tile,
    _lbp_hist_per_tile,
    _quantize_for_glcm,
    _shadow_bright_per_tile,
    _stack_tiles,
    _subtile_variance_per_tile,
    load_features,
    load_features_provenance,
    stage4b_one_image,
)
from src.labeling import LABELS_SUBDIR


# ============================================================================
# Synthetic helpers
# ============================================================================

# --- R78: the fixture must never pin the mosaic grid phase to (0, 0) ----------------
# Real geometry, read off disk: dataset_v2/labels/ESP_042964_2160.json carries
# mosaic_row_origin = 894 / mosaic_col_origin = 12645, and its parent Murray tile
# cache_v2/ctx_tiles/E-8_N32.json carries inner_transform origin
# (-474197.58018644986, 2133889.110839024).  0 of the 52 production label sidecars has
# either origin equal to 0, yet this fixture used to write 0/0 -- which is precisely why
# the origin-sign-flip and the tiles-inside-window bounds-guard mutants both survived
# this suite: with both origins zero, `ti*S - origin`, `ti*S + origin` and `ti*S` are the
# same expression.  Same fixture defect as the ~100 km fgates mis-key
# (src/fgates.py:211-231).
MOSAIC_ORIGIN_XY = (-474197.58018644986, 2133889.110839024)   # E-8_N32 inner_transform
MOSAIC_ROW_ORIGIN = 894                                        # ESP_042964_2160 sidecar
MOSAIC_COL_ORIGIN = 12645                                      # ESP_042964_2160 sidecar


def _write_synthetic_stage4_cache(
    tmp_path: Path,
    *,
    height: int = 128,
    width: int = 128,
    pixel_m: float = 5.0,
    arr: np.ndarray | None = None,
    mask_fill: int = 1,
    tile_size_px: int = 16,
    n_tiles_axis: int = 4,
    row_origin: int = MOSAIC_ROW_ORIGIN,
    col_origin: int = MOSAIC_COL_ORIGIN,
) -> tuple[Path, Path, str, np.ndarray]:
    """Write minimal Stage 2 (CTX window + mask) + Stage 4 (labels parquet + sidecar)
    caches that Stage 4b can consume. Returns (cache_dir, output_dir, obs_id, ctx_arr).

    R78: the window's upper-left sits at mosaic pixel ``(row_origin, col_origin)`` of a
    mosaic whose CRS origin is `MOSAIC_ORIGIN_XY`, so the emitted tile indices are the
    absolute ones Stage 4 emits and Stage 4b's `ti*S - mosaic_row_origin` arithmetic is
    genuinely exercised.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    (cache_dir / "ctx_windows").mkdir(parents=True, exist_ok=True)
    (output_dir / LABELS_SUBDIR).mkdir(parents=True, exist_ok=True)

    obs_id = "SYN_FEAT_000"
    mx_origin_x, mx_origin_y = MOSAIC_ORIGIN_XY
    transform = Affine(
        pixel_m, 0, mx_origin_x + col_origin * pixel_m,
        0, -pixel_m, mx_origin_y - row_origin * pixel_m,
    )
    if arr is None:
        # Mid-gray with a sprinkle of dark "shadow" pixels and bright "sunlit" pixels so
        # the DN-mode threshold has a real shadow + bright tail to find.
        rng = np.random.default_rng(42)
        arr = rng.integers(low=100, high=160, size=(height, width), dtype=np.uint8)
        arr[5:8, 5:8] = 20      # shadow patch
        arr[5:8, 30:33] = 240   # bright patch
    ctx_tif = cache_dir / "ctx_windows" / f"{obs_id}.tif"
    mask_tif = cache_dir / "ctx_windows" / f"{obs_id}_hirise_mask.tif"

    with rasterio.open(
        ctx_tif, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", transform=transform,
    ) as dst:
        dst.write(arr, 1)
    mask = np.full((height, width), mask_fill, dtype=np.uint8)
    with rasterio.open(
        mask_tif, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", transform=transform,
    ) as dst:
        dst.write(mask, 1)

    # Hand-craft a Stage 4 labels parquet with a small grid of tiles at one scale.
    # R78: absolute (ti, tj) starting at the first tile fully inside the offset window,
    # with mosaic-anchored world bounds -- not a 0-based grid at the CRS origin.
    ti0 = math.ceil(row_origin / tile_size_px)
    tj0 = math.ceil(col_origin / tile_size_px)
    rows = []
    for ti in range(ti0, ti0 + n_tiles_axis):
        for tj in range(tj0, tj0 + n_tiles_axis):
            rows.append({
                "obs_id": obs_id,
                "scale_idx": 0,
                "tile_size_px": tile_size_px,
                "tile_size_m": tile_size_px * pixel_m,
                "ti": ti,
                "tj": tj,
                "xmin": mx_origin_x + tj * tile_size_px * pixel_m,
                "ymin": mx_origin_y - (ti + 1) * tile_size_px * pixel_m,
                "xmax": mx_origin_x + (tj + 1) * tile_size_px * pixel_m,
                "ymax": mx_origin_y - ti * tile_size_px * pixel_m,
                "boulder_area": 0.0,
                "boulder_count": 0,
                "tile_area": (tile_size_px * pixel_m) ** 2,
                "fractional_area": 0.0,
                "binary_by_area": False,
                "binary_by_count": False,
                "count_density": 0.0,
                "config_hash": "synthetic",
            })
    labels_df = pd.DataFrame(rows)
    labels_df.to_parquet(output_dir / LABELS_SUBDIR / f"{obs_id}.parquet", index=False)
    sidecar = {
        "obs_id": obs_id,
        "tile_sizes_px": [tile_size_px],
        # R78: real phase, not (0, 0) -- see MOSAIC_ROW_ORIGIN / MOSAIC_COL_ORIGIN above.
        "mosaic_row_origin": row_origin,
        "mosaic_col_origin": col_origin,
        "ctx_window_tif": str(ctx_tif),
        "hirise_mask_tif": str(mask_tif),
    }
    (output_dir / LABELS_SUBDIR / f"{obs_id}.json").write_text(
        json.dumps(sidecar), encoding="utf-8",
    )
    return cache_dir, output_dir, obs_id, arr


# ============================================================================
# Intensity stats
# ============================================================================

def test_intensity_stats_constant_tile():
    """A constant-intensity tile has std=0, skew=0, kurt=0, IQR=0; mean=p50=value."""
    tile = np.full((1, 8, 8), 100, dtype=np.uint8)
    out = _intensity_stats_per_tile(tile)
    assert out["intensity_mean"][0] == 100.0
    assert out["intensity_std"][0] == 0.0
    assert out["intensity_min"][0] == 100.0
    assert out["intensity_max"][0] == 100.0
    assert out["intensity_p50"][0] == 100.0
    assert out["intensity_iqr"][0] == 0.0
    assert out["intensity_skewness"][0] == 0.0
    assert out["intensity_kurtosis"][0] == 0.0


def test_intensity_stats_ramp_tile_p10_p90():
    """A linear ramp 0..63 has p10=6.3, p90=56.7, p50=31.5 -- numpy percentile defaults."""
    tile = np.arange(64, dtype=np.uint8).reshape(1, 8, 8)
    out = _intensity_stats_per_tile(tile)
    assert out["intensity_mean"][0] == pytest.approx(31.5)
    # Per numpy default ('linear'): p10 of 0..63 = 6.3, p90 = 56.7
    assert out["intensity_p10"][0] == pytest.approx(6.3)
    assert out["intensity_p90"][0] == pytest.approx(56.7)


# ============================================================================
# GLCM
# ============================================================================

def test_glcm_quantize_levels():
    """Quantization should bin uint8 values into `levels` distinct buckets."""
    arr = np.arange(256, dtype=np.uint8)
    q = _quantize_for_glcm(arr, levels=8)
    # Bucket width = 32; values 0..31 -> 0, 32..63 -> 1, ..., 224..255 -> 7
    assert q[0] == 0
    assert q[31] == 0
    assert q[32] == 1
    assert q[255] == 7
    assert q.max() == 7
    assert q.min() == 0


def test_glcm_uniform_image_has_zero_contrast(tmp_path):
    """A constant-intensity tile has zero GLCM contrast (no co-occurring pairs differ)."""
    arr = np.full((32, 32), 100, dtype=np.uint8)
    quantized = _quantize_for_glcm(arr, levels=8)
    r_win = np.array([0], dtype=np.int64)
    c_win = np.array([0], dtype=np.int64)
    out = _glcm_per_tile(
        quantized, r_win, c_win, S=32,
        levels=8, distances=[1, 2, 3], angles=[0.0, math.pi / 4],
        properties=["contrast", "homogeneity"],
        angle_average=True, max_distances=3,
    )
    # Constant image -> no diagonal co-occurrence; contrast = 0, homogeneity = 1.
    assert out["glcm_contrast_d1"][0] == 0.0
    assert out["glcm_contrast_d2"][0] == 0.0
    assert out["glcm_contrast_d3"][0] == 0.0
    assert out["glcm_homogeneity_d1"][0] == pytest.approx(1.0)


def test_glcm_padding_with_nan_for_missing_distances():
    """When `distances` is a subset of `max_distances`, missing columns are NaN."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    quantized = _quantize_for_glcm(arr, levels=8)
    r_win = np.array([0], dtype=np.int64)
    c_win = np.array([0], dtype=np.int64)
    out = _glcm_per_tile(
        quantized, r_win, c_win, S=16,
        levels=8, distances=[1], angles=[0.0],
        properties=["contrast"], angle_average=True, max_distances=3,
    )
    # d=1 has a real value, d=2/d=3 padded with NaN.
    assert out["glcm_contrast_d1"][0] == 0.0
    assert np.isnan(out["glcm_contrast_d2"][0])
    assert np.isnan(out["glcm_contrast_d3"][0])


# ============================================================================
# Gradient
# ============================================================================

def test_gradient_on_step_function():
    """A horizontal step has nonzero gradient magnitude along the step row.

    Synthetic 32x32 image: top half = 0, bottom half = 255. Sobel gradient magnitude
    should be large along row 15-16 boundary, near-zero elsewhere.
    """
    img = np.zeros((32, 32), dtype=np.float32)
    img[16:, :] = 255.0
    from scipy.ndimage import sobel
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    mag = np.hypot(gx, gy).astype(np.float32)
    direction = np.arctan2(gy, gx).astype(np.float32)
    grad = {"magnitude": mag, "direction": direction, "gx": gx, "gy": gy}
    # One tile covering the whole image.
    out = _gradient_stats_per_tile(
        grad, r_win=np.array([0]), c_win=np.array([0]), S=32,
    )
    # The mean should be small (only ~2 rows of strong gradient out of 32), but p99 should
    # capture the boundary row.
    assert out["grad_mag_p99"][0] > out["grad_mag_mean"][0]
    assert out["grad_mag_p99"][0] > 0


# ============================================================================
# Shadow / bright
# ============================================================================

def test_dn_mode_threshold_finds_modal_peak():
    """Mode of a unimodal histogram should equal the underlying mean DN."""
    rng = np.random.default_rng(0)
    # Centered at DN=120 with std=10.
    arr = np.clip(rng.normal(120, 10, size=(200, 200)), 0, 255).astype(np.uint8)
    mask = np.ones_like(arr)
    cfg = {"shadow_offset_dn": 20, "strict_offset_dn": 35, "bright_offset_dn": 30}
    out = _compute_dn_thresholds(arr, mask, cfg)
    # Mode should land within 2 DN of 120.
    assert abs(out["mode"] - 120) <= 2
    # Thresholds derived as offsets from mode.
    assert out["shadow"] == max(0, out["mode"] - 20)
    assert out["shadow_strict"] == max(0, out["mode"] - 35)
    assert out["bright"] == min(255, out["mode"] + 30)


def test_dn_threshold_survives_clip_spike():
    """A DN=1 bottom-clip spike must not hijack the modal threshold.

    Regression for the W1 round-2 finding (DECISIONS.md 2026-06-10): windows with
    a few percent of bottom-clipped DN=1 pixels made mode=1, shadow cut 0, and
    every shadow feature identically zero image-wide (ESP_046328_2180,
    ESP_064510_2260 — both anti-signal)."""
    rng = np.random.default_rng(1)
    arr = np.clip(rng.normal(120, 10, size=(200, 200)), 0, 255).astype(np.uint8)
    arr[:30, :] = 1  # 15% bottom-clipped band — a bigger spike than any single DN bin
    mask = np.ones_like(arr)
    cfg = {"shadow_offset_dn": 20, "strict_offset_dn": 35, "bright_offset_dn": 30}
    out = _compute_dn_thresholds(arr, mask, cfg)
    assert abs(out["mode"] - 120) <= 2, "clip spike must be excluded from the histogram"
    assert out["shadow"] >= 90, "shadow cut must stay near mode-20, not collapse to 0"
    # And the cut must actually be able to fire on real (unclipped) dark pixels.
    covered = arr[(mask == 1) & (arr > 1)]
    assert (covered < out["shadow"]).any()


def test_dn_threshold_percentile_fallback_when_mode_is_dark():
    """If the (unclipped) mode is itself within shadow_offset of the clip floor,
    fall back to percentile cuts instead of a dead shadow==0 threshold."""
    rng = np.random.default_rng(2)
    arr = np.clip(rng.normal(15, 5, size=(200, 200)), 0, 255).astype(np.uint8)
    mask = np.ones_like(arr)
    cfg = {"shadow_offset_dn": 20, "strict_offset_dn": 35, "bright_offset_dn": 30}
    out = _compute_dn_thresholds(arr, mask, cfg)
    assert out["method"] == "percentile_fallback_low_mode"
    assert out["shadow"] > 1
    covered = arr[(mask == 1) & (arr > 1)]
    frac = (covered < out["shadow"]).mean()
    assert 0.0 < frac < 0.25, f"fallback cut should mark a small dark tail, got {frac:.3f}"


def test_shadow_fraction_on_synthetic_bimodal_image():
    """Shadow fraction should reflect the fraction of pixels below threshold."""
    # 8x8 tile: 16 dark pixels, 48 bright pixels.
    tile = np.full((8, 8), 200, dtype=np.uint8)
    tile[:2, :8] = 30  # 16 dark
    thresholds = {"mode": 200, "shadow": 100, "shadow_strict": 50, "bright": 230}
    out = _shadow_bright_per_tile(
        tile, thresholds, r_win=np.array([0]), c_win=np.array([0]), S=8,
    )
    assert out["shadow_fraction"][0] == pytest.approx(16 / 64)
    # No pixels above 230 -> bright_cap = 0; strict (<50) -> 16 pixels qualify.
    assert out["bright_cap_fraction"][0] == 0.0
    assert out["shadow_fraction_strict"][0] == pytest.approx(16 / 64)


# ============================================================================
# LBP
# ============================================================================

def test_lbp_hist_sums_to_one():
    """LBP histogram per tile should sum to 1.0 (normalized by tile_area)."""
    # Build a synthetic LBP label map with values in {0..9}.
    rng = np.random.default_rng(0)
    lbp = rng.integers(0, 10, size=(16, 16), dtype=np.int8)
    out = _lbp_hist_per_tile(
        lbp, r_win=np.array([0]), c_win=np.array([0]), S=16, n_bins=10,
    )
    total = sum(out[f"lbp_hist_{k}"][0] for k in range(10))
    assert total == pytest.approx(1.0)


# ============================================================================
# Subtile variance
# ============================================================================

def test_subtile_variance_zero_on_uniform_tile():
    """Constant intensity -> zero subtile variance."""
    tile = np.full((1, 16, 16), 100, dtype=np.uint8)
    var = _subtile_variance_per_tile(tile, S=16)
    assert var[0] == 0.0


def test_subtile_variance_positive_on_split_tile():
    """A tile half-bright half-dark has positive subtile variance."""
    tile = np.zeros((1, 16, 16), dtype=np.uint8)
    tile[0, :8, :] = 0
    tile[0, 8:, :] = 255
    var = _subtile_variance_per_tile(tile, S=16)
    # The 4 sub-block means alternate between 0 and 255 -> variance = ((127.5)^2 * 4) / 4
    assert var[0] > 0


# ============================================================================
# Lacunarity
# ============================================================================

def test_lacunarity_on_uniform_shadow_mask_equals_one():
    """A perfectly uniform shadow mask (all-1) should give L(b) = 1.0 for any box size."""
    mask = np.ones((32, 32), dtype=np.uint8)
    out = _lacunarity_per_tile(
        mask, r_win=np.array([0]), c_win=np.array([0]), S=32, box_sizes=[2, 4],
    )
    assert out["lacunarity_shadow_b2"][0] == pytest.approx(1.0)
    assert out["lacunarity_shadow_b4"][0] == pytest.approx(1.0)


def test_lacunarity_on_clumped_shadow_mask_above_one():
    """Clustered shadows should give lacunarity > 1."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:8, :8] = 1  # one cluster
    out = _lacunarity_per_tile(
        mask, r_win=np.array([0]), c_win=np.array([0]), S=32, box_sizes=[2, 4],
    )
    assert out["lacunarity_shadow_b2"][0] > 1.0
    assert out["lacunarity_shadow_b4"][0] > 1.0


# ============================================================================
# Stack tiles
# ============================================================================

def test_stack_tiles_preserves_pixel_values():
    """`_stack_tiles` slices the right pixels for each (r_win, c_win)."""
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    r_win = np.array([0, 8], dtype=np.int64)
    c_win = np.array([0, 8], dtype=np.int64)
    out = _stack_tiles(arr, r_win, c_win, S=8)
    assert out.shape == (2, 8, 8)
    assert np.array_equal(out[0], arr[0:8, 0:8])
    assert np.array_equal(out[1], arr[8:16, 8:16])


# ============================================================================
# Stage 4b end-to-end on synthetic cache
# ============================================================================

def test_stage4b_synthetic_emits_one_row_per_label(tmp_path):
    cache_dir, output_dir, obs_id, arr = _write_synthetic_stage4_cache(tmp_path)
    prov = stage4b_one_image(
        obs_id, cache_dir=cache_dir, output_dir=output_dir,
        features_cfg=DEFAULT_FEATURES_CFG, config_hash="test",
    )
    df = load_features(obs_id, output_dir)
    # 4x4 grid of S=16 tiles -> 16 rows
    assert len(df) == 16
    assert prov["n_tiles_total"] == 16
    # Required column families present.
    for col in [
        "obs_id", "scale_idx", "tile_size_px", "ti", "tj",
        "intensity_mean", "intensity_std", "intensity_skewness", "intensity_kurtosis",
        "grad_mag_mean", "grad_mag_p99", "grad_dir_circvar",
        "shadow_fraction", "shadow_fraction_strict", "bright_cap_fraction",
        "edge_density", "edge_orientation_entropy",
        "intensity_subtile_var",
        "glcm_contrast_d1",
        "lbp_hist_0", "lbp_hist_9",
        "valid_pixel_fraction",
        "patch_idx_S32", "patch_idx_S64",
        "config_hash",
    ]:
        assert col in df.columns, f"missing column {col!r}"
    # valid_pixel_fraction must be 1.0 by construction.
    assert (df["valid_pixel_fraction"] == 1.0).all()

    # R78: pin the labels->window registration against pixels, not just row counts.
    # At the real phase (row 894, col 12645) the first S=16 tile is
    #   ti = ceil(894/16)  = 56 -> r_win = 56*16 - 894 = 2
    #   tj = ceil(12645/16) = 791 -> c_win = 791*16 - 12645 = 11
    # so it must read window rows [2, 18) and cols [11, 27). r_win != c_win, so this also
    # discriminates a row/col swap; the origin was previously 0 and all three of
    # `ti*S - origin`, `ti*S + origin`, `ti*S` agreed.
    assert (df["ti"].min(), df["tj"].min()) == (56, 791)
    first = df[(df["ti"] == 56) & (df["tj"] == 791)]
    assert len(first) == 1
    assert first["intensity_mean"].iloc[0] == pytest.approx(
        float(arr[2:18, 11:27].mean())
    )
    # The next tile along each axis, to pin the stride as well as the offset.
    assert df[(df["ti"] == 56) & (df["tj"] == 792)]["intensity_mean"].iloc[0] == (
        pytest.approx(float(arr[2:18, 27:43].mean()))
    )
    assert df[(df["ti"] == 57) & (df["tj"] == 791)]["intensity_mean"].iloc[0] == (
        pytest.approx(float(arr[18:34, 11:27].mean()))
    )


def test_stage4b_is_idempotent(tmp_path):
    """Two runs with the same inputs must produce identical parquets."""
    cache_dir, output_dir, obs_id, _ = _write_synthetic_stage4_cache(tmp_path)
    stage4b_one_image(
        obs_id, cache_dir=cache_dir, output_dir=output_dir,
        features_cfg=DEFAULT_FEATURES_CFG, config_hash="test",
    )
    df1 = load_features(obs_id, output_dir).copy()
    stage4b_one_image(
        obs_id, cache_dir=cache_dir, output_dir=output_dir,
        features_cfg=DEFAULT_FEATURES_CFG, config_hash="test",
    )
    df2 = load_features(obs_id, output_dir)
    # The dataframes should be element-wise equal (NaNs in matching positions are fine).
    pd.testing.assert_frame_equal(df1, df2)


def test_stage4b_context_patches_disabled(tmp_path):
    """When `context_patch.enabled = false`, no patch files and no patch_idx columns."""
    cache_dir, output_dir, obs_id, _ = _write_synthetic_stage4_cache(tmp_path)
    cfg = json.loads(json.dumps(DEFAULT_FEATURES_CFG))  # deep-ish copy
    cfg["context_patch"] = {"enabled": False, "sizes_px": []}
    prov = stage4b_one_image(
        obs_id, cache_dir=cache_dir, output_dir=output_dir,
        features_cfg=cfg, config_hash="test",
    )
    df = load_features(obs_id, output_dir)
    assert "patch_idx_S32" not in df.columns
    assert "patch_idx_S64" not in df.columns
    assert prov["context_patch"]["enabled"] is False


def test_stage4b_context_patches_bundle_indices(tmp_path):
    """When enabled, every tile with sufficient margin gets a patch_idx into the bundle."""
    cache_dir, output_dir, obs_id, _ = _write_synthetic_stage4_cache(
        tmp_path, height=192, width=192, n_tiles_axis=8,
    )
    cfg = json.loads(json.dumps(DEFAULT_FEATURES_CFG))
    cfg["context_patch"] = {"enabled": True, "sizes_px": [32]}
    prov = stage4b_one_image(
        obs_id, cache_dir=cache_dir, output_dir=output_dir,
        features_cfg=cfg, config_hash="test",
    )
    df = load_features(obs_id, output_dir)
    assert "patch_idx_S32" in df.columns
    # Tiles near the window edge can't fit a centered 32-px patch; the others must.
    n_valid = int((df["patch_idx_S32"] >= 0).sum())
    assert n_valid == prov["context_patch"]["patch_counts"][32]
    # Patch file exists on disk and has the expected shape.
    patches_path = output_dir / "context_patches" / f"{obs_id}_S32.npy"
    assert patches_path.exists()
    patches = np.load(patches_path)
    assert patches.shape == (n_valid, 32, 32)
    assert patches.dtype == np.uint8


# ============================================================================
# Slow integration test (uses real Stage 2 + Stage 4 caches if present)
# ============================================================================

@pytest.mark.slow
def test_features_align_with_labels_row_for_row(tmp_path):
    """Stage 4b on ESP_069669_2220 must emit exactly the same (scale, ti, tj) set as
    Stage 4's labels parquet. Skips if Stage 4 hasn't been run.

    R77: Stage 4b is a PRODUCER -- it writes dataset/features/{obs}.parquet, its sidecar
    and both context-patch .npy stacks. It must never be pointed at the live `dataset/`
    tree (those paths are gitignored and git cannot restore them). The real labels are
    copied read-only into an isolated tree, and every write lands there.
    """
    import shutil

    repo_root = Path(__file__).resolve().parents[1]
    obs = "ESP_069669_2220"
    cache_dir = repo_root / "cache"
    src_labels = repo_root / "dataset" / LABELS_SUBDIR / f"{obs}.parquet"
    src_sidecar = src_labels.with_suffix(".json")
    if not src_labels.exists() or not src_sidecar.exists():
        pytest.skip(f"Stage 4 cache for {obs} not present")

    # Isolated output tree: Stage 4b reads labels from output_dir/labels and writes
    # features next to them, so the labels must be staged in rather than referenced.
    output_dir = tmp_path / "dataset"
    (output_dir / LABELS_SUBDIR).mkdir(parents=True)
    shutil.copy2(src_labels, output_dir / LABELS_SUBDIR / src_labels.name)
    shutil.copy2(src_sidecar, output_dir / LABELS_SUBDIR / src_sidecar.name)
    labels = output_dir / LABELS_SUBDIR / f"{obs}.parquet"

    from src.config import load_config
    cfg = load_config(repo_root / "config.yaml")
    stage4b_one_image(
        obs, cache_dir=cache_dir, output_dir=output_dir,
        features_cfg=cfg["features"], config_hash=cfg.hash,
    )
    labels_df = pd.read_parquet(labels)
    features_df = load_features(obs, output_dir)
    # Same row count.
    assert len(features_df) == len(labels_df)
    # Same (scale_idx, tile_size_px, ti, tj) tuple set.
    label_keys = set(zip(
        labels_df["scale_idx"], labels_df["tile_size_px"], labels_df["ti"], labels_df["tj"],
    ))
    feature_keys = set(zip(
        features_df["scale_idx"], features_df["tile_size_px"], features_df["ti"], features_df["tj"],
    ))
    assert label_keys == feature_keys


@pytest.mark.slow
def test_features_sanity_on_real_data():
    """Run-on-real basic checks: fractional values in range, intensity stats finite."""
    repo_root = Path(__file__).resolve().parents[1]
    obs = "ESP_069669_2220"
    features = repo_root / "dataset" / FEATURES_SUBDIR / f"{obs}.parquet"
    if not features.exists():
        # R77: this is a pure READ of the committed artifact. It no longer free-rides on
        # test_features_align_..., which now writes to tmp_path; run scripts/run_stage4b.py.
        pytest.skip(f"Stage 4b cache for {obs} not present (run Stage 4b first)")
    df = pd.read_parquet(features)
    # Intensity stats must be finite for all rows; CTX is uint8 so values in [0, 255].
    assert df["intensity_mean"].between(0, 255).all()
    assert df["intensity_std"].between(0, 255).all()
    # shadow_fraction/strict/bright_cap in [0, 1].
    for col in ["shadow_fraction", "shadow_fraction_strict", "bright_cap_fraction",
                "valid_pixel_fraction", "edge_density"]:
        finite = df[col].dropna()
        assert finite.between(0, 1).all(), f"{col} out of [0, 1]"
    # LBP histograms must sum to ~1.0 per row.
    lbp_cols = [f"lbp_hist_{k}" for k in range(10)]
    lbp_sum = df[lbp_cols].sum(axis=1)
    assert (lbp_sum.between(0.999, 1.001)).all()
    # GLCM contrast >= 0 always.
    finite = df["glcm_contrast_d1"].dropna()
    assert (finite >= 0).all()
