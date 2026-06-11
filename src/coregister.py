"""Stage 3 — HiRISE↔CTX co-registration via sub-pixel phase correlation.

Per CLAUDE.md, Stage 3 is an *optional* refinement that only shifts the per-image grid
anchor. The pipeline must still work on nominal geolocation; this module adds a
`(dx, dy)` correction cached per ObsId so Stage 4 can use it (or ignore it).

Pipeline for one ObsId:

1. Read the cached decimated HiRISE (5 m/px in the HiRISE source CRS) and the cached
   CTX window (5 m/px in the CTX mosaic CRS, written by Stage 2).
2. Reproject the HiRISE imagery onto the CTX window's exact transform + shape using
   **bilinear** resampling. Result: two co-located 5 m/px arrays in the same CRS and on
   the same pixel grid. (Contrast: Stage 2's coverage mask uses `nearest` to keep the
   binary boundary crisp; here we want intensity, so bilinear.)
3. **Single-window solve** (kept for provenance + fallback): pick a power-of-2 sub-window
   from the central region of HiRISE coverage, Hann-window it, and phase-correlate against
   the co-located CTX sub-window with sub-pixel upsampling.
4. **Robust block-median solve** (primary; DECISIONS.md 2026-05-28): tile the *whole*
   window into `block_px` blocks, phase-correlate each fully-covered block, and take the
   median `(dy, dx)` over blocks whose local Pearson peak ≥ `block_peak_min`. This is
   robust to a single central window landing on a featureless/artifact patch and returning
   junk (the ESP_049242_2115 failure mode) even though the rest of the image registers
   cleanly. When fewer than `min_confident_blocks` clear the floor (a genuinely bland
   scene), fall back to the single-window shift. The chosen `(dy, dx)` is converted to
   metres via the CTX transform.
5. `peak_correlation` is the median confident-block peak (block-median path) or the
   single-window post-shift Pearson correlation (fallback) — a confidence proxy easier to
   threshold than the raw phase-correlation peak height.
6. Write `cache/coregistration/{obs_id}.json` with the chosen shift, its `method`, the
   preserved single-window result, the block-field statistics, peak correlation, FFT
   window placement, and provenance. **No hard flag/fail thresholds applied** — the
   notebook 05 whole-image validation is where accept/flag thresholds are eyeballed.

The shift's sign convention: `dx_m`, `dy_m` are the corrections to *add* to a HiRISE
pixel's projected coordinate so it lines up with CTX. Equivalently, if you re-warp the
HiRISE imagery using `(target_x + dx_m, target_y + dy_m)` as the destination origin,
the result aligns with CTX. Stage 4 will only use the magnitudes for now.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

COREGISTRATION_SUBDIR = "coregistration"


def _warp_hirise_to_ctx_grid(
    obs_id: str,
    *,
    jp2_url: str,
    cache_dir: Path,
    ctx_window_tif: Path,
) -> tuple[np.ndarray, Any, Any]:
    """Reproject the decimated HiRISE imagery onto the CTX window's grid.

    Returns `(warped_array, ctx_transform, ctx_crs)`. `warped_array` is float32 in the
    HiRISE source dtype's value range; pixels outside the HiRISE footprint are 0.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from . import hirise_imagery

    hirise_imagery.ensure_jp2_local(obs_id, jp2_url, cache_dir)
    hi_arr, hi_transform, hi_crs = hirise_imagery.read_full_footprint_decimated(
        obs_id, jp2_url, cache_dir, target_mpp=5.0,
    )

    with rasterio.open(ctx_window_tif) as ctx_src:
        ctx_transform = ctx_src.transform
        ctx_crs = ctx_src.crs
        ctx_shape = (ctx_src.height, ctx_src.width)

    warped = np.zeros(ctx_shape, dtype=np.float32)
    reproject(
        source=hi_arr.astype(np.float32),
        destination=warped,
        src_transform=hi_transform,
        src_crs=hi_crs,
        dst_transform=ctx_transform,
        dst_crs=ctx_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.bilinear,
    )
    return warped, ctx_transform, ctx_crs


def _largest_power_of_two(n: int) -> int:
    """Largest power of 2 ≤ n. Returns 0 if n < 1."""
    if n < 1:
        return 0
    return 1 << (int(math.log2(n)))


def select_fft_window(
    coverage_mask: np.ndarray,
    max_px: int,
    min_px: int = 64,
) -> tuple[int, int, int]:
    """Pick a power-of-2 sub-window fully inside the coverage mask.

    Returns `(size_px, row_off, col_off)`. `size_px` is the largest power of 2 ≤ `max_px`
    such that a `size_px × size_px` block fits entirely inside `coverage_mask == 1`,
    centered as close as possible to the coverage centroid.

    Raises `RuntimeError` if no power-of-2 ≥ `min_px` fits.

    Algorithm: compute the mask's interior centroid, then for each candidate size (largest
    first), search a small neighborhood around the centroid for a placement whose block
    is fully covered. Falls back to scanning all valid placements at the size if the
    centroid-anchored search fails (rare; happens for highly non-convex coverage).
    """
    if coverage_mask.ndim != 2:
        raise ValueError(f"coverage_mask must be 2D, got shape {coverage_mask.shape}")
    if coverage_mask.dtype != np.uint8 and coverage_mask.dtype != np.bool_:
        coverage_mask = (coverage_mask > 0).astype(np.uint8)
    h, w = coverage_mask.shape

    ys, xs = np.nonzero(coverage_mask)
    if len(ys) == 0:
        raise RuntimeError("select_fft_window: coverage_mask is all zero — no HiRISE coverage")
    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))

    # Integral image: O(1) sum query for any rectangle.
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(coverage_mask.astype(np.int64), axis=0), axis=1)

    def block_sum(r0: int, c0: int, size: int) -> int:
        r1, c1 = r0 + size, c0 + size
        return int(
            integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
        )

    size = _largest_power_of_two(min(max_px, min(h, w)))
    while size >= min_px:
        target_area = size * size

        # Try centered placement first.
        r0 = max(0, min(h - size, cy - size // 2))
        c0 = max(0, min(w - size, cx - size // 2))
        if block_sum(r0, c0, size) == target_area:
            return size, r0, c0

        # Otherwise scan a coarse grid of placements; step = size//4 keeps this cheap.
        step = max(1, size // 4)
        best: tuple[int, int, int] | None = None
        best_d2 = None
        for rr in range(0, h - size + 1, step):
            for cc in range(0, w - size + 1, step):
                if block_sum(rr, cc, size) != target_area:
                    continue
                d2 = (rr + size // 2 - cy) ** 2 + (cc + size // 2 - cx) ** 2
                if best is None or d2 < best_d2:
                    best = (size, rr, cc)
                    best_d2 = d2
        if best is not None:
            return best

        size //= 2

    raise RuntimeError(
        f"select_fft_window: no power-of-2 ≥ {min_px} px fits inside the coverage mask "
        f"(mask shape {coverage_mask.shape}, total covered pixels {int(coverage_mask.sum())})"
    )


def _hann2d(size: int) -> np.ndarray:
    """Separable 2-D Hann window of side `size`."""
    from scipy.signal import windows

    w = windows.hann(size, sym=False).astype(np.float32)
    return np.outer(w, w)


def phase_correlate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    upsample_factor: int = 20,
) -> tuple[float, float, float]:
    """Sub-pixel phase correlation between two equal-shape 2D arrays.

    Returns `(dy_px, dx_px, peak_correlation)` where the shift is the translation that,
    applied to `moving`, aligns it with `reference`. `peak_correlation` is the Pearson
    correlation of the shifted `moving` against `reference` over their mutually-valid
    interior — a confidence proxy.
    """
    from scipy.ndimage import shift as nd_shift
    from skimage.registration import phase_cross_correlation

    if reference.shape != moving.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {moving.shape}")
    if reference.ndim != 2:
        raise ValueError(f"phase_correlate_translation expects 2D arrays, got {reference.ndim}D")

    ref = reference.astype(np.float32)
    mov = moving.astype(np.float32)

    # Mean-subtract within the FFT window so the DC bin doesn't dominate when one image
    # is dimmer than the other, then apply a Hann window to reduce spectral leakage from
    # the array edges (matters at modest window sizes like 256).
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    w = _hann2d(ref.shape[0])
    ref_w = ref * w
    mov_w = mov * w

    shift_rc, _err, _phasediff = phase_cross_correlation(
        ref_w, mov_w,
        upsample_factor=upsample_factor,
        normalization="phase",
    )
    dy, dx = float(shift_rc[0]), float(shift_rc[1])

    # Pearson correlation after applying the solved shift. We crop a margin around the
    # array so newly-introduced edge pixels (filled with 0 by nd_shift) don't bias the
    # correlation downward.
    margin = max(2, int(math.ceil(max(abs(dy), abs(dx))))) + 2
    if margin * 2 >= min(ref.shape):
        peak = float("nan")
    else:
        mov_shifted = nd_shift(moving.astype(np.float32), shift=(dy, dx), order=1, mode="constant", cval=0.0)
        a = reference.astype(np.float32)[margin:-margin, margin:-margin]
        b = mov_shifted[margin:-margin, margin:-margin]
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        peak = float((a * b).sum() / denom) if denom > 0 else float("nan")

    return dy, dx, peak


def _robust_shift_from_field(
    field: list[dict],
    *,
    block_peak_min: float,
    min_confident_blocks: int,
) -> tuple[float, float, dict] | None:
    """Robust per-image shift = median of the confident blocks' local shifts.

    A single central FFT window can land on a featureless / artifact-ridden patch and
    return a junk translation even when the rest of the image registers cleanly (the
    ESP_049242_2115 failure mode; DECISIONS.md 2026-05-28). Taking the median over all
    blocks whose local `peak >= block_peak_min` is robust to those outliers.

    Returns `(dy_px, dx_px, stats)` or None when fewer than `min_confident_blocks` blocks
    clear the peak floor (the genuinely-bland case → caller falls back to single-window).
    """
    if not field:
        return None
    peaks = np.array([b["peak"] for b in field], dtype=np.float64)
    conf = peaks >= block_peak_min
    n_conf = int(conf.sum())
    if n_conf < min_confident_blocks:
        return None
    dys = np.array([b["dy_px"] for b in field], dtype=np.float64)[conf]
    dxs = np.array([b["dx_px"] for b in field], dtype=np.float64)[conf]
    dy = float(np.median(dys))
    dx = float(np.median(dxs))
    stats = {
        "n_blocks": int(len(field)),
        "n_confident_blocks": n_conf,
        "median_block_peak": float(np.median(peaks[conf])),
        # Median absolute deviation: a robust spread of the confident-block shifts (px).
        "block_mad_px": {
            "dy": float(np.median(np.abs(dys - dy))),
            "dx": float(np.median(np.abs(dxs - dx))),
        },
    }
    return dy, dx, stats


def shift_px_to_world_m(
    dy_px: float,
    dx_px: float,
    *,
    px_x: float,
    px_y: float,
) -> tuple[float, float]:
    """Convert an array-space (row, col) shift to a world-space (dx_m, dy_m) correction.

    `(dy_px, dx_px)` follows the `phase_correlate_translation` convention: the
    translation to apply to the moving array (HiRISE) to align it with the
    reference (CTX). Columns increase with world x, but **rows increase as
    world y decreases** (north-up grid, `transform.e < 0`), so the row
    component must flip sign on the way to world coordinates. Getting this
    wrong inverts the y-correction and doubles the misalignment instead of
    removing it — the W1 rung-1 bug (DECISIONS.md 2026-06-10, W1 entry).
    """
    return float(dx_px * px_x), float(-dy_px * px_y)


def stage3_one_image(
    obs_id: str,
    *,
    cache_dir: str | Path,
    manifest_row,
    fft_window_px: int,
    upsample_factor: int,
    config_hash: str,
    block_px: int = 256,
    block_peak_min: float = 0.5,
    min_confident_blocks: int = 6,
) -> dict:
    """Solve a per-image (dx, dy) translation to register HiRISE onto CTX.

    Requires Stage 2 caches to exist for `obs_id`:
      cache/ctx_windows/{obs_id}.tif          (CTX window in CTX CRS)
      cache/ctx_windows/{obs_id}_hirise_mask.tif  (HiRISE coverage)
      cache/hirise_jp2/{obs_id}_RED.JP2 (downloaded on demand by ensure_jp2_local)
      cache/hirise_decimated/{obs_id}_5mpp_full.tif (built on demand)

    Writes:
      cache/coregistration/{obs_id}.json  — shift, peak correlation, FFT window, provenance.

    Returns the provenance dict.
    """
    import rasterio

    from . import ctx_retrieve  # for the CTX_WINDOWS_SUBDIR constant

    cache_dir = Path(cache_dir)
    ctx_window_tif = cache_dir / ctx_retrieve.CTX_WINDOWS_SUBDIR / f"{obs_id}.tif"
    mask_tif = cache_dir / ctx_retrieve.CTX_WINDOWS_SUBDIR / f"{obs_id}_hirise_mask.tif"
    if not ctx_window_tif.exists():
        raise FileNotFoundError(
            f"Stage 3 requires Stage 2 output {ctx_window_tif}. Run scripts/run_stage2.py {obs_id} first."
        )
    if not mask_tif.exists():
        raise FileNotFoundError(
            f"Stage 3 requires HiRISE coverage mask {mask_tif}. Re-run Stage 2 for {obs_id}."
        )

    # 1. Warp HiRISE onto the CTX grid.
    hi_warped, ctx_transform, ctx_crs = _warp_hirise_to_ctx_grid(
        obs_id,
        jp2_url=str(manifest_row["JP2_URL"]),
        cache_dir=cache_dir,
        ctx_window_tif=ctx_window_tif,
    )

    with rasterio.open(ctx_window_tif) as src:
        ctx_arr = src.read(1).astype(np.float32)
    with rasterio.open(mask_tif) as src:
        coverage_mask = src.read(1).astype(np.uint8)

    # 2. Pick an FFT window that lies entirely inside HiRISE coverage. We also intersect
    # with `(ctx_arr > 0)` so CTX nodata regions (rare but possible at tile seams) can't
    # poison the correlation.
    ctx_valid = (ctx_arr > 0).astype(np.uint8)
    combined = (coverage_mask & ctx_valid).astype(np.uint8)
    size_px, row_off, col_off = select_fft_window(combined, max_px=int(fft_window_px))

    hi_sub = hi_warped[row_off : row_off + size_px, col_off : col_off + size_px]
    ctx_sub = ctx_arr[row_off : row_off + size_px, col_off : col_off + size_px]

    # 3a. Single-window solve (kept for provenance + as the fallback). Convention:
    # `phase_cross_correlation(ref, mov)` returns the shift to apply to `mov` to match
    # `ref`; CTX is the fixed reference, HiRISE is moving, so the result is the correction
    # to apply to HiRISE to bring it onto CTX.
    sw_dy, sw_dx, sw_peak = phase_correlate_translation(
        ctx_sub, hi_sub, upsample_factor=int(upsample_factor),
    )

    # 3b. Robust whole-image solve: median of the per-block shift field (DECISIONS.md
    # 2026-05-28). A single central window can land on a bad patch and return junk even
    # when the rest of the image registers cleanly; the block-median is robust to that.
    # Falls back to the single-window solve when too few blocks correlate (genuinely bland).
    field = block_shift_field(
        hi_warped, ctx_arr, combined,
        block_px=int(block_px), min_coverage=0.98, upsample_factor=int(upsample_factor),
    )
    robust = _robust_shift_from_field(
        field, block_peak_min=float(block_peak_min), min_confident_blocks=int(min_confident_blocks),
    )
    if robust is not None:
        dy_px, dx_px, block_stats = robust
        peak = block_stats["median_block_peak"]
        method = "block_median"
    else:
        dy_px, dx_px, peak = sw_dy, sw_dx, sw_peak
        method = "single_window_fallback"
        n_conf = int(sum(1 for b in field if b["peak"] >= float(block_peak_min)))
        block_stats = {
            "n_blocks": int(len(field)),
            "n_confident_blocks": n_conf,
            "median_block_peak": None,
            "block_mad_px": None,
        }

    # 4. Convert pixel shift to metres on the CTX grid.
    px_x = abs(ctx_transform.a)
    px_y = abs(ctx_transform.e)
    dx_m, dy_m = shift_px_to_world_m(dy_px, dx_px, px_x=px_x, px_y=px_y)
    shift_m = float(math.hypot(dx_m, dy_m))

    provenance = {
        "obs_id": obs_id,
        "ctx_window_tif": str(ctx_window_tif),
        "ctx_transform": list(ctx_transform)[:6],
        "ctx_crs_wkt": ctx_crs.to_wkt() if ctx_crs else None,
        "method": method,
        "fft_window": {
            "size_px": int(size_px),
            "row_off": int(row_off),
            "col_off": int(col_off),
            "config_max_px": int(fft_window_px),
        },
        "shift_px": {"dy": dy_px, "dx": dx_px},
        "shift_m": {"dy": dy_m, "dx": dx_m, "magnitude": shift_m},
        "peak_correlation": peak,
        # The single-window result is preserved so the block-median can always be compared
        # against (and reverted to) the original central-FFT solve.
        "single_window": {
            "dy_px": sw_dy, "dx_px": sw_dx, "peak": sw_peak,
            "dy_m": -sw_dy * px_y, "dx_m": sw_dx * px_x,
            "magnitude_m": float(math.hypot(sw_dx * px_x, sw_dy * px_y)),
        },
        "block_field": {"block_px": int(block_px), "block_peak_min": float(block_peak_min),
                        "min_confident_blocks": int(min_confident_blocks), **block_stats},
        "upsample_factor": int(upsample_factor),
        "config_hash": config_hash,
        "solved_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }

    out_dir = cache_dir / COREGISTRATION_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{obs_id}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def warp_hirise_to_ctx_grid(
    obs_id: str,
    *,
    jp2_url: str,
    cache_dir: str | Path,
    ctx_window_tif: str | Path,
) -> tuple[np.ndarray, Any, Any]:
    """Public wrapper over `_warp_hirise_to_ctx_grid` for QA / validation callers.

    Returns `(hirise_warped_on_ctx_grid, ctx_transform, ctx_crs)`.
    """
    return _warp_hirise_to_ctx_grid(
        obs_id, jp2_url=jp2_url, cache_dir=Path(cache_dir), ctx_window_tif=Path(ctx_window_tif),
    )


def block_shift_field(
    hi_warped: np.ndarray,
    ctx_arr: np.ndarray,
    coverage_mask: np.ndarray,
    *,
    block_px: int = 128,
    step_px: int | None = None,
    min_coverage: float = 0.98,
    upsample_factor: int = 20,
) -> list[dict]:
    """Per-block phase-correlation shift field across the WHOLE CTX window.

    Stage 3 solves a single rigid `(dy, dx)` from one central FFT sub-window. This
    function tests whether that single translation actually holds everywhere: it tiles the
    window into `block_px` blocks (stride `step_px`, default = `block_px` for
    non-overlapping), and for each block whose HiRISE-and-CTX coverage is ≥ `min_coverage`
    runs `phase_correlate_translation(ctx_block, hi_block)`.

    A spatially-coherent field whose local shifts cluster tightly around the global Stage-3
    shift confirms a rigid translation is adequate. A fanned-out field (systematic spatial
    gradient → residual rotation/scale) or one dominated by low `peak` blocks indicates the
    single shift does not describe the whole image — or that the global solve itself failed
    (e.g. a bland-plains scene with no correlatable structure).

    Returns one dict per evaluated block:
        {row_off, col_off, row_center, col_center, dy_px, dx_px, peak, coverage}
    (empty list if no block meets `min_coverage`).
    """
    if hi_warped.shape != ctx_arr.shape:
        raise ValueError(f"shape mismatch: hi_warped {hi_warped.shape} vs ctx {ctx_arr.shape}")
    step = int(step_px) if step_px is not None else int(block_px)
    h, w = ctx_arr.shape
    # A pixel is usable only where BOTH HiRISE coverage and CTX have real data.
    combined = ((coverage_mask > 0) & (ctx_arr > 0)).astype(np.float32)
    ctx_f = ctx_arr.astype(np.float32)
    hi_f = hi_warped.astype(np.float32)

    out: list[dict] = []
    for r0 in range(0, h - block_px + 1, step):
        for c0 in range(0, w - block_px + 1, step):
            cov = float(combined[r0 : r0 + block_px, c0 : c0 + block_px].mean())
            if cov < min_coverage:
                continue
            ctx_b = ctx_f[r0 : r0 + block_px, c0 : c0 + block_px]
            hi_b = hi_f[r0 : r0 + block_px, c0 : c0 + block_px]
            dy, dx, peak = phase_correlate_translation(ctx_b, hi_b, upsample_factor=upsample_factor)
            out.append({
                "row_off": int(r0), "col_off": int(c0),
                "row_center": int(r0 + block_px // 2), "col_center": int(c0 + block_px // 2),
                "dy_px": float(dy), "dx_px": float(dx),
                "peak": float(peak), "coverage": cov,
            })
    return out


def load_shift(obs_id: str, cache_dir: str | Path) -> dict | None:
    """Return the cached Stage 3 provenance for `obs_id`, or None if not solved yet."""
    p = Path(cache_dir) / COREGISTRATION_SUBDIR / f"{obs_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def find_tracking_features(
    image: np.ndarray,
    *,
    n_features: int = 8,
    min_distance: int = 24,
    edge_margin: int = 12,
    sigma: float = 1.5,
) -> np.ndarray:
    """Pick `n_features` distinctive intensity peaks for the marker overlay.

    Returns an `(N, 2)` array of `(row, col)` coordinates, with `N <= n_features`
    (may return fewer for very bland scenes — that's fine, the caller plots whatever
    is found).

    Smoothing with a small Gaussian first suppresses single-pixel noise so the picked
    peaks are real features, not speckle. `edge_margin` keeps markers off the panel
    edges where they'd be partially clipped after a shift. `min_distance` spreads
    markers across the field of view rather than letting them cluster on one bright
    region.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max

    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")
    smooth = gaussian_filter(image.astype(np.float32), sigma=sigma)

    h, w = smooth.shape
    if edge_margin * 2 >= h or edge_margin * 2 >= w:
        edge_margin = 0
    interior = smooth[edge_margin : h - edge_margin, edge_margin : w - edge_margin]

    coords = peak_local_max(
        interior,
        min_distance=min_distance,
        num_peaks=n_features,
        exclude_border=False,
    )
    # peak_local_max returns (row, col) relative to `interior`; offset back to `image`.
    if len(coords) > 0:
        coords = coords + np.array([edge_margin, edge_margin])
    return coords
