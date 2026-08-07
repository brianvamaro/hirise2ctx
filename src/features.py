"""Stage 4b -- per-tile CTX texture features.

Reads the cached CTX window + HiRISE coverage mask from Stage 2 and the per-tile
eligibility set from Stage 4's `dataset/labels/{ObsId}.parquet`; emits feature columns to
`dataset/features/{ObsId}.parquet` joinable 1:1 on `(scale_idx, ti, tj)`. Optionally also
writes per-(ObsId, patch_size) `.npy` stacks of raw CTX context patches.

Why a separate stage: CLAUDE.md acceptance #4 -- adding or changing features must not
require re-running Stages 1-3 or Stage 4. Stage 4b reads existing caches only.

Design (PLAN_Stage4b.md + AskUserQuestion decisions 2026-05-23):

- **Resolution-preservation**: CTX pixels are NEVER spatially downsampled. The only
  information-discarding step is GLCM intensity quantization, and even that is
  scale-aware -- finer scales get coarser quantization because tiny tiles can't fill
  large co-occurrence matrices meaningfully (Clausi 2002).
- **Window-once, tile-many**: per-image artifacts (Sobel gradient magnitude/direction,
  LBP map, Canny edge map, shadow/bright/strict-shadow binary masks, per-quantization
  level integer arrays) are computed once over the full CTX window. Per-tile features
  are then reshape-and-reduce operations on rectangular blocks -- O(n_pixels) per family
  per image, regardless of how many tiles fall inside.
- **GLCM and lacunarity loop per tile**: graycomatrix can't be vectorised over tiles in
  skimage. For each emitted tile, we slice the quantized array and call graycomatrix
  once. Average over the 4 angles for rotation-invariance (AskUserQuestion 2026-05-23 =
  rotation-averaged single-value-per-property).
- **Shadow detector** = DN-mode-derived absolute threshold (AskUserQuestion 2026-05-23):
  one bincount per image finds the modal DN of HiRISE-covered pixels; thresholds are
  `mode - shadow_offset_dn` (normal), `mode - strict_offset_dn` (stricter), and
  `mode + bright_offset_dn` (sunlit boulder tops). Stable across tiles within an image;
  pairs naturally with `bright_cap_fraction` for the shadow/sunlit asymmetry that's a
  stronger boulder signal than either alone (PLAN_Stage4b.md §3.4, Kirk et al. 2008).
- **LBP** = rotation-invariant uniform, P=8 R=1 (skimage method='uniform'); produces a
  10-bin histogram per tile (`lbp_hist_0` .. `lbp_hist_9`).
- **Context patches** = bundled per (obs_id, patch_size) into a single `.npy` stack
  rather than ~1.3M individual {ti}_{tj}.npy files (per PLAN_Stage4b.md §6's literal
  layout). The features parquet stores integer row indices into the stack; -1 means
  insufficient window margin for a centered patch. Deviation from the plan documented
  in DECISIONS.md.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES_SUBDIR = "features"
CONTEXT_PATCHES_SUBDIR = "context_patches"

# Default config -- mirrored from config.yaml's `features:` block, but every helper accepts
# overrides so the synthetic-data unit tests can exercise it without a full config load.
DEFAULT_FEATURES_CFG: dict[str, Any] = {
    "enabled": [
        "intensity_stats", "glcm", "gradient", "shadow_fraction",
        "lbp", "lacunarity", "subtile_variance", "canny_edges",
    ],
    "glcm": {
        "levels_per_scale": {8: 8, 16: 16, 32: 16, 64: 32},
        "distances_per_scale": {8: [1], 16: [1, 2, 3], 32: [1, 2, 3], 64: [1, 2, 3]},
        "angles": [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4],
        "angle_average": True,
        "properties": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
    },
    "gradient": {"sigma": 1.0},
    "shadow_fraction": {
        "method": "dn_mode_offset",
        "shadow_offset_dn": 20,
        "strict_offset_dn": 35,
        "bright_offset_dn": 30,
    },
    "lbp": {"method": "uniform", "P": 8, "R": 1},
    "lacunarity": {"box_sizes_px": [2, 4], "min_tile_size_px": 32},
    "subtile_variance": {"min_tile_size_px": 16},
    "canny_edges": {
        # R28 (Brian, 2026-08-06): thresholds are PERCENTILES of this frame's own gradient
        # magnitude, so `edge_density` no longer tracks how much contrast the CTX frame
        # happens to have. 0.80/0.90 = top 20 % of gradients are edge candidates, top 10 %
        # are seeds; that lands mid-range of the pre-fix 0.025-0.307 cohort spread, so it
        # keeps roughly the current amount of signal. See `_compute_canny_window` for the
        # measurements and for why 0.1/0.2 must NOT be reused as quantiles.
        "sigma": 1.0, "use_quantiles": True,
        "low_threshold": 0.80, "high_threshold": 0.90,
        "n_orientation_bins": 8, "min_tile_size_px": 16,
    },
    "context_patch": {"enabled": True, "sizes_px": [32, 64]},
}

# Stage 4b deliberately recognises the same "drop this ObsId from --all sweeps" set that
# Stage 4 already uses. Kept in sync with scripts/run_stage4.py EXCLUDED_FROM_SWEEP.
#   ESP_057469_2215 — v1: tile-straddle, 0.1% HiRISE coverage (DECISIONS.md 2026-05-22).
#   ESP_046803_2325 — v2 vClaire: featureless CTX, 0/210 co-registration blocks correlate
#                     (DECISIONS.md 2026-05-28; notebook 05 fallback deep-dive).
EXCLUDED_FROM_SWEEP = {"ESP_057469_2215", "ESP_046803_2325"}

# How wide a histogram bin we use when finding the modal DN. CTX uint8 is 0..255; one bin
# per integer is fine, no smoothing needed since real CTX scenes have a well-defined
# unimodal terrain peak.
_DN_HISTOGRAM_BINS = 256

# Skimage's GLCM correlation property divides by stddev and emits NaN on uniform-intensity
# tiles. We replace NaN with 0 in the parquet so the schema stays cleanly numeric; the
# loss is intentional and reported in the per-image provenance.
_GLCM_NAN_FILL = 0.0


# ============================================================================
# Per-image cached artifacts
# ============================================================================

def _load_window_and_mask(ctx_tif: Path, mask_tif: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the Stage 2 CTX window and its companion HiRISE coverage mask as uint8 arrays."""
    import rasterio
    with rasterio.open(ctx_tif) as src:
        arr = src.read(1)
    with rasterio.open(mask_tif) as src:
        mask = src.read(1)
    if arr.shape != mask.shape:
        raise RuntimeError(
            f"Window {arr.shape} and mask {mask.shape} shapes differ; "
            "Stage 2 caches are inconsistent."
        )
    return arr.astype(np.uint8, copy=False), mask.astype(np.uint8, copy=False)


_DN_CLIP_FLOOR = 1  # CTX mosaic bottom-clips very dark terrain to DN=1 (DN=0 is nodata)


def _compute_dn_thresholds(arr: np.ndarray, mask: np.ndarray, cfg: dict) -> dict[str, int]:
    """Find the modal DN of HiRISE-mask-covered pixels and derive shadow/bright thresholds.

    Returns the four absolute DN cuts plus the mode for provenance. Falls back to image
    percentiles if the masked region is too small to be meaningful (e.g. nominal-footprint
    fallback windows).

    Pixels at or below `_DN_CLIP_FLOOR` are excluded from the histogram: windows
    containing bottom-clipped dark terrain concentrate it into a DN=1 spike that
    becomes the modal DN even at a few percent areal fraction, driving the shadow
    cut to `max(0, 1-20) = 0` and killing every shadow feature image-wide (the
    W1 round-2 finding on ESP_046328_2180 / ESP_064510_2260, DECISIONS.md
    2026-06-10). If the cut still lands at or below the clip floor, fall back to
    the 10th/5th percentiles of the unclipped covered pixels.
    """
    covered = arr[mask == 1]
    covered = covered[covered > _DN_CLIP_FLOOR]
    if covered.size < 1000:
        # Degenerate window -- empty/near-empty HiRISE coverage; fall back to global pcts.
        # ESP_057469_2215 (the tile-straddle case) hits this; Stage 4b would skip it
        # anyway by the EXCLUDED_FROM_SWEEP rule.
        all_vals = arr.ravel()
        mode = int(np.median(all_vals))
        shadow = int(max(0, np.percentile(all_vals, 10)))
        shadow_strict = int(max(0, np.percentile(all_vals, 5)))
        bright = int(min(255, np.percentile(all_vals, 95)))
        return {"mode": mode, "shadow": shadow, "shadow_strict": shadow_strict,
                "bright": bright, "method": "image_percentile_fallback"}
    counts = np.bincount(covered, minlength=_DN_HISTOGRAM_BINS)
    mode = int(counts.argmax())
    shadow = max(0, mode - int(cfg["shadow_offset_dn"]))
    shadow_strict = max(0, mode - int(cfg["strict_offset_dn"]))
    bright = min(255, mode + int(cfg["bright_offset_dn"]))
    if shadow <= _DN_CLIP_FLOOR:
        # Mode sits so low that the offset cut can never fire on real pixels --
        # use percentile cuts of the unclipped covered DNs instead.
        shadow = int(max(_DN_CLIP_FLOOR + 1, np.percentile(covered, 10)))
        shadow_strict = int(max(_DN_CLIP_FLOOR + 1, np.percentile(covered, 5)))
        return {"mode": mode, "shadow": shadow, "shadow_strict": shadow_strict,
                "bright": bright, "method": "percentile_fallback_low_mode"}
    return {"mode": mode, "shadow": shadow, "shadow_strict": shadow_strict,
            "bright": bright, "method": cfg.get("method", "dn_mode_offset")}


def _compute_gradient_window(arr: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
    """Sobel gradient (magnitude + direction) over the full CTX window.

    Returns dict with 'magnitude' (float32) and 'direction' (float32 in radians).
    Optional Gaussian smoothing with `sigma` is applied first as a low-pass against CTX
    sensor noise.
    """
    from scipy.ndimage import gaussian_filter, sobel
    arr_f = arr.astype(np.float32)
    if sigma > 0:
        arr_f = gaussian_filter(arr_f, sigma=sigma)
    gx = sobel(arr_f, axis=1, mode="reflect").astype(np.float32)
    gy = sobel(arr_f, axis=0, mode="reflect").astype(np.float32)
    magnitude = np.hypot(gx, gy).astype(np.float32)
    direction = np.arctan2(gy, gx).astype(np.float32)
    return {"magnitude": magnitude, "direction": direction, "gx": gx, "gy": gy}


def _compute_lbp_window(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """Uniform LBP over the full CTX window (P=8, R=1, method='uniform').

    Output is an int8 array with values in {0..P+1} for P=8 -> 10 distinct labels. Stored
    int8 to keep the array small (50 MB for a typical 7000x5000-px window).
    """
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(arr, P=int(cfg["P"]), R=int(cfg["R"]), method=cfg["method"])
    return lbp.astype(np.int8)


def _compute_canny_window(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """Canny edge map over the full CTX window. Returns uint8 binary array (0/1).

    **R28 — the thresholds are absolute unless you ask for quantiles.** With
    `use_quantiles=False` (the default), `low_threshold=None` makes skimage use the
    constants 0.1 / 0.2 as *absolute* gradient magnitudes on the `img_as_float` image, not
    values derived from this image's gradient distribution — the opposite of what
    `config.yaml` used to claim. Measured on a synthetic scene: reducing the DN spread ~3x
    collapses edge density from 0.345 to 0.0026 (x0.01), and across the 38-image cohort
    per-image `edge_density` tracks per-image `intensity_std` at Spearman rho = 0.965 with
    a 12.2x spread. That is per-frame radiometry leaking into a texture feature — the same
    failure the project already found and fixed for `shadow_fraction` (DECISIONS
    2026-06-10). With `use_quantiles=True` the same contrast change leaves edge density
    unmoved (x1.00).

    Two traps, both measured, both worth not rediscovering:

    - `low_threshold=None` is **not** the same as `low_threshold=0.1`. skimage maps None to
      0.1 directly, but an explicit value goes through `low_threshold /= dtype_max`; on a
      uint8 window that is 0.1/255, which passes almost every gradient (density 0.345 ->
      0.384 and, being far below any real gradient, is contrast-invariant for the wrong
      reason). So the current behaviour cannot be written into the config as `0.1`.
    - with `use_quantiles=True` the thresholds are *percentiles of gradient magnitude*, so
      they must be high (e.g. 0.8/0.9 -> density 0.130; 0.9/0.95 -> 0.062). Reusing 0.1/0.2
      as quantiles would mark ~80-90 % of pixels as edges.
    """
    from skimage.feature import canny

    use_quantiles = bool(cfg.get("use_quantiles", False))
    low = cfg.get("low_threshold")
    high = cfg.get("high_threshold")
    if use_quantiles and (low is None or high is None):
        raise ValueError(
            "canny_edges.use_quantiles=true requires explicit low_threshold and "
            "high_threshold in [0, 1] (percentiles of gradient magnitude). Leaving them "
            "null falls back to skimage's absolute 0.1/0.2 constants, which is the R28 "
            "defect this option exists to avoid."
        )
    edges = canny(
        arr,
        sigma=float(cfg["sigma"]),
        low_threshold=low,
        high_threshold=high,
        use_quantiles=use_quantiles,
    )
    return edges.astype(np.uint8)


# ============================================================================
# Per-tile reductions (vectorized via reshape-and-reduce)
# ============================================================================

def _stack_tiles(arr: np.ndarray, r_win: np.ndarray, c_win: np.ndarray, S: int) -> np.ndarray:
    """Stack the per-tile windows of `arr` into a (n_tiles, S, S) array.

    `r_win[i]` and `c_win[i]` are the window-pixel top-left of tile i. Asserts all tiles
    lie fully inside the window (Stage 4 guarantees this). View-only when contiguous; a
    copy otherwise -- mostly cheap.
    """
    n = r_win.size
    out = np.empty((n, S, S), dtype=arr.dtype)
    # Loop -- numpy fancy-indexing into a (n, S, S) view isn't directly expressible.
    # The loop is ~microseconds per tile and dwarfed by anything that touches each pixel.
    for i in range(n):
        r, c = int(r_win[i]), int(c_win[i])
        out[i] = arr[r:r + S, c:c + S]
    return out


def _intensity_stats_per_tile(tiles_u8: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized per-tile intensity reductions on a (n_tiles, S, S) uint8 stack.

    Returns 10 columns including p10/p50/p90/IQR (np.percentile axis-fold), skewness,
    and kurtosis. Mean/std use float64 for numerical stability across image scales.
    """
    tiles = tiles_u8.astype(np.float64)
    n = tiles.shape[0]
    flat = tiles.reshape(n, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1, ddof=0)
    pcts = np.percentile(flat, [10, 25, 50, 75, 90], axis=1)
    p10, p25, p50, p75, p90 = pcts[0], pcts[1], pcts[2], pcts[3], pcts[4]
    # Skewness + excess kurtosis via centered moments. Where std==0 (uniform tile),
    # the centered moments are 0 / 0 -> we set to 0 deliberately.
    centered = flat - mean[:, None]
    var = (centered ** 2).mean(axis=1)
    m3 = (centered ** 3).mean(axis=1)
    m4 = (centered ** 4).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(var > 0, m3 / (var ** 1.5), 0.0)
        kurt = np.where(var > 0, m4 / (var ** 2) - 3.0, 0.0)
    return {
        "intensity_mean": mean,
        "intensity_std": std,
        "intensity_min": flat.min(axis=1),
        "intensity_max": flat.max(axis=1),
        "intensity_p10": p10,
        "intensity_p50": p50,
        "intensity_p90": p90,
        "intensity_iqr": p75 - p25,
        "intensity_skewness": skew,
        "intensity_kurtosis": kurt,
    }


def _gradient_stats_per_tile(
    grad: dict[str, np.ndarray], r_win: np.ndarray, c_win: np.ndarray, S: int,
) -> dict[str, np.ndarray]:
    """Per-tile gradient reductions from the pre-computed full-window magnitude/direction.

    `grad_mag_p99` was added 2026-05-23: boulder edges are rare bright outliers; the p90
    saturates in busy tiles so the p99 catches the very-rare strongest edges.
    """
    mag_tiles = _stack_tiles(grad["magnitude"], r_win, c_win, S)  # (n, S, S) float32
    dir_tiles = _stack_tiles(grad["direction"], r_win, c_win, S)
    n = mag_tiles.shape[0]
    flat_m = mag_tiles.reshape(n, -1)
    flat_d = dir_tiles.reshape(n, -1)
    pcts = np.percentile(flat_m, [90, 99], axis=1)
    # Circular variance of direction weighted by magnitude (low-magnitude directions are
    # noise). Define w = mag / mag.sum(); circvar = 1 - |sum(w * exp(i*2*theta))|.
    # The factor of 2 in the angle handles 180-deg ambiguity (a horizontal edge has
    # direction pi or 0 depending on which side is brighter -- both are the same edge).
    weights = flat_m / (flat_m.sum(axis=1, keepdims=True) + 1e-12)
    mean_cos = (weights * np.cos(2.0 * flat_d)).sum(axis=1)
    mean_sin = (weights * np.sin(2.0 * flat_d)).sum(axis=1)
    R = np.hypot(mean_cos, mean_sin)
    circvar = 1.0 - R
    return {
        "grad_mag_mean": flat_m.mean(axis=1).astype(np.float64),
        "grad_mag_std": flat_m.std(axis=1, ddof=0).astype(np.float64),
        "grad_mag_p90": pcts[0].astype(np.float64),
        "grad_mag_p99": pcts[1].astype(np.float64),
        "grad_dir_circvar": circvar.astype(np.float64),
    }


def _shadow_bright_per_tile(
    arr: np.ndarray, thresholds: dict[str, int],
    r_win: np.ndarray, c_win: np.ndarray, S: int,
) -> dict[str, np.ndarray]:
    """Per-tile shadow / strict-shadow / bright-cap fractions, all from per-image DN cuts."""
    shadow_mask = (arr < thresholds["shadow"]).astype(np.uint8)
    strict_mask = (arr < thresholds["shadow_strict"]).astype(np.uint8)
    bright_mask = (arr > thresholds["bright"]).astype(np.uint8)
    shadow_tiles = _stack_tiles(shadow_mask, r_win, c_win, S)
    strict_tiles = _stack_tiles(strict_mask, r_win, c_win, S)
    bright_tiles = _stack_tiles(bright_mask, r_win, c_win, S)
    n = shadow_tiles.shape[0]
    denom = float(S * S)
    return {
        "shadow_fraction": shadow_tiles.reshape(n, -1).sum(axis=1, dtype=np.float64) / denom,
        "shadow_fraction_strict": strict_tiles.reshape(n, -1).sum(axis=1, dtype=np.float64) / denom,
        "bright_cap_fraction": bright_tiles.reshape(n, -1).sum(axis=1, dtype=np.float64) / denom,
    }


def _lbp_hist_per_tile(
    lbp: np.ndarray, r_win: np.ndarray, c_win: np.ndarray, S: int, n_bins: int,
) -> dict[str, np.ndarray]:
    """Per-tile uniform-LBP histogram. 10 bins for P=8 method='uniform'."""
    tiles = _stack_tiles(lbp, r_win, c_win, S)  # int8, values in {0..n_bins-1}
    n = tiles.shape[0]
    flat = tiles.reshape(n, -1)
    out = {}
    denom = float(flat.shape[1])
    for k in range(n_bins):
        out[f"lbp_hist_{k}"] = (flat == k).sum(axis=1, dtype=np.int64).astype(np.float64) / denom
    return out


def _canny_per_tile(
    edges: np.ndarray, direction: np.ndarray,
    r_win: np.ndarray, c_win: np.ndarray, S: int, n_bins: int,
) -> dict[str, np.ndarray]:
    """Per-tile edge density and orientation entropy from pre-computed Canny + Sobel direction."""
    edge_tiles = _stack_tiles(edges, r_win, c_win, S)
    dir_tiles = _stack_tiles(direction, r_win, c_win, S)
    n = edge_tiles.shape[0]
    flat_e = edge_tiles.reshape(n, -1).astype(np.bool_)
    flat_d = dir_tiles.reshape(n, -1)
    denom = float(flat_e.shape[1])
    density = flat_e.sum(axis=1, dtype=np.int64).astype(np.float64) / denom
    # Orientation entropy: bin edge-pixel directions into n_bins over [0, pi) (180-deg
    # ambiguous), compute Shannon entropy of the per-tile histogram. Tiles with no edges
    # get entropy = 0 by convention (no information).
    # Map direction in [-pi, pi] -> [0, pi) by taking modulo pi.
    bin_edges = np.linspace(0.0, math.pi, n_bins + 1)
    entropies = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d = flat_d[i][flat_e[i]]
        if d.size == 0:
            entropies[i] = 0.0
            continue
        d_mod = np.mod(d, math.pi)
        hist, _ = np.histogram(d_mod, bins=bin_edges)
        p = hist / hist.sum()
        # 0 * log(0) = 0
        nz = p > 0
        entropies[i] = float(-(p[nz] * np.log(p[nz])).sum())
    return {"edge_density": density, "edge_orientation_entropy": entropies}


def _subtile_variance_per_tile(tiles_u8: np.ndarray, S: int) -> np.ndarray:
    """Variance of (S/2)-block means within each tile. PLAN_Stage4b.md §3.5.4.

    Captures internal heterogeneity that single-tile std misses; cheap because it reuses
    the already-stacked tiles.
    """
    sub = S // 2
    n = tiles_u8.shape[0]
    # Reshape each (S, S) tile into 4 sub-blocks of (sub, sub), take their means, then
    # variance across the 4 means. Result is a single float per tile.
    blocks = tiles_u8.astype(np.float64).reshape(n, 2, sub, 2, sub)
    sub_means = blocks.mean(axis=(2, 4))  # (n, 2, 2)
    return sub_means.reshape(n, -1).var(axis=1, ddof=0)


def _lacunarity_per_tile(
    shadow_mask: np.ndarray, r_win: np.ndarray, c_win: np.ndarray, S: int, box_sizes: list[int],
) -> dict[str, np.ndarray]:
    """Gliding-box lacunarity on the per-image shadow mask, per tile, per box size.

    Lacunarity L(b) = E[M^2] / E[M]^2 where M is the sum of pixels inside a sliding b*b
    box (Allain & Cloitre 1991; Plesko et al. 2009 for lunar precedent). 1.0 means perfectly
    uniform; > 1 means clustered/gappy. Degenerate at tile sizes below box_size+1 in either
    axis, so the caller restricts this to S >= min_tile_size_px (default 32).
    """
    n = r_win.size
    out = {f"lacunarity_shadow_b{b}": np.full(n, np.nan, dtype=np.float64) for b in box_sizes}
    # Pre-compute the integral image of the shadow mask once for cheap arbitrary-box sums.
    integral = shadow_mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    # Pad with a zero row/col on top/left so the box-sum formula simplifies.
    H, W = shadow_mask.shape
    pad = np.zeros((H + 1, W + 1), dtype=np.int64)
    pad[1:, 1:] = integral
    integral = pad

    for b in box_sizes:
        col = f"lacunarity_shadow_b{b}"
        if b > S:
            continue  # left as NaN
        # For each tile, build the box-sum image over its (S x S) window then compute
        # E[M^2] / E[M]^2.
        for i in range(n):
            r, c = int(r_win[i]), int(c_win[i])
            # Sum over box of size b*b for top-left at (r+dr, c+dc), dr in [0, S-b], dc in [0, S-b].
            # Using the integral image: box_sum(r0, c0) = I[r0+b, c0+b] - I[r0, c0+b] - I[r0+b, c0] + I[r0, c0]
            r0s = np.arange(r, r + S - b + 1)
            c0s = np.arange(c, c + S - b + 1)
            I = integral
            box_sums = (
                I[r0s[:, None] + b, c0s[None, :] + b]
                - I[r0s[:, None], c0s[None, :] + b]
                - I[r0s[:, None] + b, c0s[None, :]]
                + I[r0s[:, None], c0s[None, :]]
            ).astype(np.float64)
            M1 = box_sums.mean()
            M2 = (box_sums ** 2).mean()
            # R27: a tile with no shadow pixels has no gliding-box statistic at all, so it
            # stays NaN -- the same "not computable" convention every other Stage 4b column
            # uses. It used to emit 0.0, which is out of range (lacunarity is >= 1 by
            # Cauchy-Schwarz) and therefore a sentinel: measured over dataset_v2, 42,015 of
            # 198,320 S>=32 rows were exactly 0.0, every one with shadow_fraction == 0, the
            # smallest non-zero value was exactly 1.0, and nothing fell in (0, 1). Stage 6a's
            # neighbour aggregation is NaN-aware (`np.isfinite`) but not sentinel-aware, so
            # it averaged the sentinel in with real measurements and produced
            # `nbr_mean_lacunarity_*` values inside the impossible interval (0, 1).
            if M1 > 0:
                out[col][i] = M2 / (M1 ** 2)
    return out


# ============================================================================
# GLCM (per-tile, loops -- skimage doesn't vectorize over tiles)
# ============================================================================

def _quantize_for_glcm(arr: np.ndarray, levels: int) -> np.ndarray:
    """Linearly quantize uint8 [0, 255] -> [0, levels-1] for GLCM."""
    # Use integer division. levels==256 is a no-op; levels==8 produces values 0..7.
    bin_width = 256 // levels
    out = (arr // bin_width).astype(np.uint8)
    np.clip(out, 0, levels - 1, out=out)
    return out


def _glcm_per_tile(
    arr_quantized: np.ndarray, r_win: np.ndarray, c_win: np.ndarray, S: int,
    *, levels: int, distances: list[int], angles: list[float], properties: list[str],
    angle_average: bool, max_distances: int,
) -> dict[str, np.ndarray]:
    """Per-tile rotation-averaged GLCM properties.

    Schema is stable across scales: emits `glcm_{property}_d{k}` columns for k =
    1..max_distances, padding with NaN where `distances` doesn't include k. This keeps
    a single columnar schema across the per-scale parquets at concat time.
    """
    from skimage.feature import graycomatrix, graycoprops

    n = r_win.size
    # Allocate output: for each property, max_distances columns, all NaN initially.
    out: dict[str, np.ndarray] = {}
    for prop in properties:
        for k in range(1, max_distances + 1):
            out[f"glcm_{prop}_d{k}"] = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return out

    # Inner loop. Each tile is small (8x8 .. 64x64); graycomatrix is the bottleneck.
    distances_arr = np.asarray(distances, dtype=np.intp)
    angles_arr = np.asarray(angles, dtype=np.float64)
    for i in range(n):
        r, c = int(r_win[i]), int(c_win[i])
        tile = arr_quantized[r:r + S, c:c + S]
        glcm = graycomatrix(
            tile, distances=distances_arr, angles=angles_arr,
            levels=levels, symmetric=True, normed=True,
        )
        for prop in properties:
            vals = graycoprops(glcm, prop)  # (n_distances, n_angles)
            if angle_average:
                vals = vals.mean(axis=1)  # (n_distances,)
            else:
                # Future-proofing: per-angle would emit different columns. Not implemented.
                vals = vals.mean(axis=1)
            for k_idx, d in enumerate(distances):
                col = f"glcm_{prop}_d{d}"
                v = float(vals[k_idx])
                out[col][i] = _GLCM_NAN_FILL if not math.isfinite(v) else v
    return out


# ============================================================================
# Context patches
# ============================================================================

def _build_context_patches(
    arr: np.ndarray,
    *,
    tiles_by_scale: dict[int, dict[str, np.ndarray]],
    patch_sizes_px: list[int],
    window_h: int, window_w: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[int, np.ndarray]]]:
    """Build per-(patch_size) uint8 stacks of CTX patches centered on each emitted tile.

    Returns (patches_per_size, indices_per_scale_per_size):
      - `patches_per_size[P]` is a (N_P, P, P) uint8 array; N_P = number of tiles (across
        all scales) for which a centered P-px patch fits inside the window.
      - `indices_per_scale_per_size[S][P]` is an int32 array, one entry per emitted tile
        at scale S, giving the row index into patches_per_size[P] (or -1 if no fit).

    The same patch is *not* deduplicated across scales -- different tiles may share the
    same center (e.g. a coarse tile center coincides with one sub-tile center) but we
    store separate rows for join simplicity. Net disk cost is bounded by ~4x the
    finest-only count and well under 10 GB across the priority10 manifest.
    """
    patches_per_size: dict[int, list[np.ndarray]] = {P: [] for P in patch_sizes_px}
    indices_per_scale_per_size: dict[int, dict[int, np.ndarray]] = {
        S: {P: np.full(grids["ti"].size, -1, dtype=np.int32) for P in patch_sizes_px}
        for S, grids in tiles_by_scale.items()
    }
    for S, grids in tiles_by_scale.items():
        r_win = grids["r_win"]
        c_win = grids["c_win"]
        for P in patch_sizes_px:
            stack = patches_per_size[P]
            half = P // 2
            S_half = S // 2
            for i in range(r_win.size):
                # Patch is centered on tile center: tile (r_win, c_win) has center at
                # (r_win + S/2, c_win + S/2); patch top-left = center - P/2.
                rc = int(r_win[i]) + S_half
                cc = int(c_win[i]) + S_half
                r0 = rc - half
                c0 = cc - half
                if r0 < 0 or c0 < 0 or r0 + P > window_h or c0 + P > window_w:
                    continue
                indices_per_scale_per_size[S][P][i] = len(stack)
                stack.append(arr[r0:r0 + P, c0:c0 + P].copy())
    bundled = {
        P: (np.stack(patches_per_size[P], axis=0) if patches_per_size[P]
            else np.zeros((0, P, P), dtype=np.uint8))
        for P in patch_sizes_px
    }
    return bundled, indices_per_scale_per_size


# ============================================================================
# Top-level entry point
# ============================================================================

def stage4b_one_image(
    obs_id: str,
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    features_cfg: dict | None = None,
    config_hash: str,
) -> dict:
    """Compute per-tile features for one ObsId and cache them next to the labels parquet.

    Iterates the eligible-tile rows already in `dataset/labels/{obs_id}.parquet` -- no
    re-derivation of the grid. Writes `dataset/features/{obs_id}.parquet` + sidecar +
    optional `dataset/context_patches/{obs_id}_S{P}.npy` stacks.
    """
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    if features_cfg is None:
        features_cfg = DEFAULT_FEATURES_CFG

    # Merge incoming features_cfg with defaults so partial overrides still work for tests.
    cfg = _deep_merge_defaults(features_cfg, DEFAULT_FEATURES_CFG)

    enabled = set(cfg["enabled"])
    labels_parquet = output_dir / "labels" / f"{obs_id}.parquet"
    labels_sidecar = output_dir / "labels" / f"{obs_id}.json"
    if not labels_parquet.exists() or not labels_sidecar.exists():
        raise FileNotFoundError(
            f"Stage 4b requires Stage 4 outputs {labels_parquet} + {labels_sidecar}. "
            f"Run scripts/run_stage4.py {obs_id} first."
        )

    labels_df = pd.read_parquet(labels_parquet)
    labels_prov = json.loads(labels_sidecar.read_text(encoding="utf-8"))
    mosaic_row_origin = int(labels_prov["mosaic_row_origin"])
    mosaic_col_origin = int(labels_prov["mosaic_col_origin"])
    ctx_window_tif = Path(labels_prov["ctx_window_tif"])
    mask_tif = Path(labels_prov["hirise_mask_tif"])

    arr, mask = _load_window_and_mask(ctx_window_tif, mask_tif)
    H, W = arr.shape

    # ---- per-image artifacts (compute once) ----
    timings: dict[str, float] = {}

    t0 = time.monotonic()
    dn_thresholds = _compute_dn_thresholds(arr, mask, cfg["shadow_fraction"])
    timings["dn_thresholds"] = time.monotonic() - t0

    if "gradient" in enabled or "canny_edges" in enabled:
        t0 = time.monotonic()
        grad = _compute_gradient_window(arr, sigma=float(cfg["gradient"]["sigma"]))
        timings["gradient_window"] = time.monotonic() - t0
    else:
        grad = None

    if "lbp" in enabled:
        t0 = time.monotonic()
        lbp_map = _compute_lbp_window(arr, cfg["lbp"])
        timings["lbp_window"] = time.monotonic() - t0
        n_lbp_bins = int(cfg["lbp"]["P"]) + 2  # method='uniform' -> P+2 distinct labels
    else:
        lbp_map = None
        n_lbp_bins = 0

    if "canny_edges" in enabled:
        t0 = time.monotonic()
        canny_map = _compute_canny_window(arr, cfg["canny_edges"])
        timings["canny_window"] = time.monotonic() - t0
    else:
        canny_map = None

    if "shadow_fraction" in enabled or "lacunarity" in enabled:
        shadow_full_mask = (arr < dn_thresholds["shadow"]).astype(np.uint8)
    else:
        shadow_full_mask = None

    # Pre-quantize the CTX window per (levels) used at any scale, so GLCM doesn't redo it
    # per tile. PLAN_Stage4b.md §3.2: levels vary by scale (8 / 16 / 32 -> 3 quantizations).
    if "glcm" in enabled:
        glcm_cfg = cfg["glcm"]
        unique_levels = sorted(set(int(v) for v in glcm_cfg["levels_per_scale"].values()))
        t0 = time.monotonic()
        quantized = {L: _quantize_for_glcm(arr, L) for L in unique_levels}
        timings["glcm_quantize"] = time.monotonic() - t0
    else:
        quantized = {}

    # Schema-stability max for GLCM distance columns. The schema across all scales
    # accepts up to max_distances columns per property; per-scale entries fill the
    # subset that's actually computed at that scale, others stay NaN.
    if "glcm" in enabled:
        max_distances = max(
            (max(d) for d in glcm_cfg["distances_per_scale"].values()), default=0,
        )
    else:
        max_distances = 0

    # ---- iterate scales emitted by Stage 4 ----
    tiles_by_scale: dict[int, dict[str, Any]] = {}
    per_scale_feature_frames: list[pd.DataFrame] = []
    per_scale_timings: dict[str, dict[str, float]] = {}

    for tile_size_px, scale_group in labels_df.groupby("tile_size_px", sort=True):
        S = int(tile_size_px)
        scale_idx = int(scale_group["scale_idx"].iloc[0])
        ti = scale_group["ti"].to_numpy(dtype=np.int64)
        tj = scale_group["tj"].to_numpy(dtype=np.int64)
        # Window-pixel slices per tile.
        r_win = (ti * S - mosaic_row_origin).astype(np.int64)
        c_win = (tj * S - mosaic_col_origin).astype(np.int64)
        # Sanity: Stage 4 guarantees tiles fit entirely inside the window.
        if not (
            (r_win >= 0).all() and (c_win >= 0).all()
            and (r_win + S <= H).all() and (c_win + S <= W).all()
        ):
            raise RuntimeError(
                f"{obs_id}: scale {S}: some Stage 4 tiles fall outside the cached CTX "
                "window -- Stage 2/4 cache mismatch."
            )

        tiles_u8 = _stack_tiles(arr, r_win, c_win, S)
        tiles_by_scale[S] = {
            "scale_idx": scale_idx, "ti": ti, "tj": tj, "r_win": r_win, "c_win": c_win,
        }
        sc_timings: dict[str, float] = {}
        rows: dict[str, Any] = {
            "obs_id": obs_id,
            "scale_idx": scale_idx,
            "tile_size_px": S,
            "ti": ti, "tj": tj,
        }

        # Valid pixel fraction is computed from the HiRISE mask (always 1.0 for Stage 4
        # eligible tiles by construction, but recorded as an explicit column so a future
        # relaxed-eligibility config can filter downstream without re-running).
        mask_tiles = _stack_tiles(mask, r_win, c_win, S)
        rows["valid_pixel_fraction"] = (
            mask_tiles.reshape(mask_tiles.shape[0], -1).mean(axis=1).astype(np.float64)
        )

        if "intensity_stats" in enabled:
            t0 = time.monotonic()
            rows.update(_intensity_stats_per_tile(tiles_u8))
            sc_timings["intensity_stats"] = time.monotonic() - t0

        if "gradient" in enabled and grad is not None:
            t0 = time.monotonic()
            rows.update(_gradient_stats_per_tile(grad, r_win, c_win, S))
            sc_timings["gradient"] = time.monotonic() - t0

        if "shadow_fraction" in enabled:
            t0 = time.monotonic()
            rows.update(_shadow_bright_per_tile(arr, dn_thresholds, r_win, c_win, S))
            sc_timings["shadow_fraction"] = time.monotonic() - t0

        if "lbp" in enabled and lbp_map is not None:
            t0 = time.monotonic()
            rows.update(_lbp_hist_per_tile(lbp_map, r_win, c_win, S, n_bins=n_lbp_bins))
            sc_timings["lbp"] = time.monotonic() - t0

        if "subtile_variance" in enabled and S >= int(cfg["subtile_variance"]["min_tile_size_px"]):
            t0 = time.monotonic()
            rows["intensity_subtile_var"] = _subtile_variance_per_tile(tiles_u8, S)
            sc_timings["subtile_variance"] = time.monotonic() - t0

        if "canny_edges" in enabled and S >= int(cfg["canny_edges"]["min_tile_size_px"]) \
                and canny_map is not None and grad is not None:
            t0 = time.monotonic()
            rows.update(_canny_per_tile(
                canny_map, grad["direction"], r_win, c_win, S,
                n_bins=int(cfg["canny_edges"]["n_orientation_bins"]),
            ))
            sc_timings["canny_edges"] = time.monotonic() - t0

        if "lacunarity" in enabled and S >= int(cfg["lacunarity"]["min_tile_size_px"]) \
                and shadow_full_mask is not None:
            t0 = time.monotonic()
            rows.update(_lacunarity_per_tile(
                shadow_full_mask, r_win, c_win, S,
                box_sizes=[int(b) for b in cfg["lacunarity"]["box_sizes_px"]],
            ))
            sc_timings["lacunarity"] = time.monotonic() - t0

        if "glcm" in enabled:
            t0 = time.monotonic()
            L = int(glcm_cfg["levels_per_scale"][S])
            ds = [int(d) for d in glcm_cfg["distances_per_scale"][S]]
            rows.update(_glcm_per_tile(
                quantized[L], r_win, c_win, S,
                levels=L, distances=ds, angles=glcm_cfg["angles"],
                properties=glcm_cfg["properties"],
                angle_average=bool(glcm_cfg["angle_average"]),
                max_distances=max_distances,
            ))
            sc_timings["glcm"] = time.monotonic() - t0

        per_scale_feature_frames.append(pd.DataFrame(rows))
        per_scale_timings[str(S)] = sc_timings

    features_df = pd.concat(per_scale_feature_frames, ignore_index=True) if per_scale_feature_frames else pd.DataFrame()

    # ---- context patches (per (obs_id, patch_size) bundle) ----
    patch_indices_columns: dict[int, np.ndarray] = {}
    patch_provenance: dict[str, Any] = {"enabled": False}
    if cfg["context_patch"]["enabled"] and tiles_by_scale:
        sizes = [int(P) for P in cfg["context_patch"]["sizes_px"]]
        bundled, indices_per_scale_per_size = _build_context_patches(
            arr, tiles_by_scale=tiles_by_scale, patch_sizes_px=sizes,
            window_h=H, window_w=W,
        )
        patches_dir = output_dir / CONTEXT_PATCHES_SUBDIR
        patches_dir.mkdir(parents=True, exist_ok=True)
        patch_files: dict[int, str] = {}
        for P, stack in bundled.items():
            out_path = patches_dir / f"{obs_id}_S{P}.npy"
            np.save(out_path, stack)
            patch_files[P] = str(out_path)

        # Attach patch_idx_S{P} columns to features_df. The rows are ordered per-scale
        # block; we rebuild the column by walking the features_df rows in the same
        # tile-by-scale order and concatenating per-scale index arrays.
        for P in sizes:
            col_parts: list[np.ndarray] = []
            # The scales iterate in the same order as features_df was built (sorted by
            # tile_size_px); use the labels_df groupby's group order to match.
            scale_order = sorted(tiles_by_scale.keys())
            for S in scale_order:
                col_parts.append(indices_per_scale_per_size[S][P])
            patch_indices_columns[P] = np.concatenate(col_parts) if col_parts else np.zeros(0, dtype=np.int32)
        for P in sizes:
            features_df[f"patch_idx_S{P}"] = patch_indices_columns[P]
        patch_provenance = {
            "enabled": True,
            "sizes_px": sizes,
            "patch_files": patch_files,
            "patch_counts": {int(P): int(bundled[P].shape[0]) for P in sizes},
            "patch_bytes_estimate": {
                int(P): int(bundled[P].nbytes) for P in sizes
            },
        }

    features_df["config_hash"] = config_hash

    # ---- write parquet + sidecar ----
    features_dir = output_dir / FEATURES_SUBDIR
    features_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = features_dir / f"{obs_id}.parquet"
    sidecar_path = features_dir / f"{obs_id}.json"
    features_df.to_parquet(parquet_path, index=False)

    provenance = {
        "obs_id": obs_id,
        "n_tiles_total": int(len(features_df)),
        "per_scale_tile_counts": {
            int(S): int((features_df["tile_size_px"] == S).sum())
            for S in sorted({int(s) for s in features_df["tile_size_px"]})
        } if len(features_df) else {},
        "enabled_features": sorted(enabled),
        "ctx_window_tif": str(ctx_window_tif),
        "hirise_mask_tif": str(mask_tif),
        "labels_parquet": str(labels_parquet),
        "mosaic_row_origin": mosaic_row_origin,
        "mosaic_col_origin": mosaic_col_origin,
        "dn_thresholds": dn_thresholds,
        "glcm": {
            "levels_per_scale": {int(k): int(v) for k, v in cfg["glcm"]["levels_per_scale"].items()},
            "distances_per_scale": {
                int(k): [int(d) for d in v]
                for k, v in cfg["glcm"]["distances_per_scale"].items()
            },
            "angle_average": bool(cfg["glcm"]["angle_average"]),
            "properties": list(cfg["glcm"]["properties"]),
            "max_distances_in_schema": max_distances,
            "nan_fill": _GLCM_NAN_FILL,
        } if "glcm" in enabled else None,
        "lbp": {"method": cfg["lbp"]["method"], "P": int(cfg["lbp"]["P"]), "R": int(cfg["lbp"]["R"]),
                "n_bins": n_lbp_bins} if "lbp" in enabled else None,
        "lacunarity": {"box_sizes_px": [int(b) for b in cfg["lacunarity"]["box_sizes_px"]],
                       "min_tile_size_px": int(cfg["lacunarity"]["min_tile_size_px"])} if "lacunarity" in enabled else None,
        "context_patch": patch_provenance,
        "timings_per_image_seconds": timings,
        "timings_per_scale_seconds": per_scale_timings,
        "parquet_path": str(parquet_path),
        "config_hash": config_hash,
        "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    sidecar_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _deep_merge_defaults(override: dict, default: dict) -> dict:
    """Shallow-deep merge: copy default, override leaves from `override`.

    Only descends into dicts; lists and scalars are taken from `override` as-is. Sufficient
    for our nested-config shape (each leaf is a list/scalar, not a structured dict).
    """
    out = {}
    for k, v_default in default.items():
        if k in override:
            v_override = override[k]
            if isinstance(v_default, dict) and isinstance(v_override, dict):
                out[k] = _deep_merge_defaults(v_override, v_default)
            else:
                out[k] = v_override
        else:
            out[k] = v_default
    # Carry over any extra keys override has that aren't in default.
    for k in override:
        if k not in default:
            out[k] = override[k]
    return out


def load_features(obs_id: str, output_dir: str | Path) -> pd.DataFrame:
    """Load a Stage 4b features parquet."""
    return pd.read_parquet(Path(output_dir) / FEATURES_SUBDIR / f"{obs_id}.parquet")


def load_features_provenance(obs_id: str, output_dir: str | Path) -> dict:
    """Load a Stage 4b features provenance sidecar."""
    p = Path(output_dir) / FEATURES_SUBDIR / f"{obs_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_context_patches(obs_id: str, patch_size: int, output_dir: str | Path) -> np.ndarray:
    """Load a Stage 4b context patch stack as a memory-mapped uint8 array."""
    p = Path(output_dir) / CONTEXT_PATCHES_SUBDIR / f"{obs_id}_S{patch_size}.npy"
    return np.load(p, mmap_mode="r")
