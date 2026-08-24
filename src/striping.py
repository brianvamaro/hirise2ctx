"""Shared logic for the regional-map striping/rectangular-artifact investigation.

Single source of truth (CLAUDE.md §7: real logic lives in src/) for
``scripts/striping_characterize.py``, ``scripts/striping_seam_test.py`` and
``notebooks/25_striping_artifact.ipynb``. See ``PLAN_StripingArtifact.md`` for the plan and
DECISIONS 2026-06-18c for the verdict.

Everything here is read-only over the already-written per-tile GeoTIFFs in
``reports/map_region/`` (abundance = calibrated; prob_raw = raw model output before qmatch),
the cached Murray Lab CTX mosaic zips, and the cached per-frame SeamMap shapefiles. No
re-inference.
"""
from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter, sobel
from shapely.geometry import LineString, MultiLineString

REPO = Path(__file__).resolve().parents[1]
MAP_DIR = REPO / "reports" / "map_region"
CTX_ZIP_DIR = REPO / "cache_v2" / "ctx_tiles"
# SeamMaps live beside the tile zips they come out of. **This used to be `cache/ctx_tiles`
# while `CTX_ZIP_DIR` was `cache_v2/ctx_tiles`** — two different roots that happen to be the
# SAME directory on the dev laptop, where `cache_v2/ctx_tiles` is an NTFS junction into
# `cache/`. On Linux there is no junction: a fresh Sherlock clone has no `cache/` at all
# (it is gitignored), so `load_frames` fetched the SeamMap over vsicurl and then died writing
# its GeoPackage cache into a directory that did not exist. Caught in pre-flight before the
# step-11 A1 array was submitted (DECISIONS 2026-08-23e). One root, and it is created on use.
SEAM_DIR = REPO / "cache_v2" / "ctx_tiles"
PX_M = 160.0  # tile_px=32 * 5 m/px = 160 m per coarse (abundance) pixel


# --------------------------------------------------------------------------- I/O
def load_raster(path: str | Path) -> np.ndarray:
    """Read a single-band GeoTIFF as float64 with nodata -> NaN."""
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        nd = ds.nodata
    if nd is not None:
        a[a == nd] = np.nan
    return a


def equipped_tiles() -> list[str]:
    """Map tiles that have an abundance raster + cached CTX zip + cached SeamMap."""
    out = []
    for p in sorted(MAP_DIR.glob("*_abundance.tif")):
        t = p.name.replace("_abundance.tif", "")
        if (CTX_ZIP_DIR / f"{t}.zip").exists() and find_seam_shp(t) is not None:
            out.append(t)
    return out


def all_map_tiles() -> list[str]:
    return sorted(p.name.replace("_abundance.tif", "")
                  for p in MAP_DIR.glob("*_abundance.tif"))


def _inner_tif_name(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as z:
        tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
    if not tifs:
        raise FileNotFoundError(f"no .tif in {zip_path}")
    return sorted(tifs)[0]


def read_ctx_to(tile: str, dst_transform, dst_shape, dst_crs) -> np.ndarray:
    """Read the Murray CTX mosaic tile area-averaged onto an arbitrary destination grid.

    Both rasters are in the same Mars equirectangular CRS, so this is a pure resampling
    (no warp); mosaic nodata (0) -> NaN. Pixels outside this tile stay NaN.
    """
    zip_path = CTX_ZIP_DIR / f"{tile}.zip"
    vsizip = f"/vsizip/{zip_path.as_posix()}/{_inner_tif_name(zip_path)}"
    dst = np.zeros(dst_shape, dtype=np.float32)
    with rasterio.open(vsizip) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform,
                  src_crs=src.crs if src.crs else dst_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.average)
    ctx = dst.astype(np.float64)
    ctx[ctx == 0] = np.nan
    return ctx


def read_ctx_on_grid(tile: str, ref_path: str | Path) -> np.ndarray:
    """Read the CTX tile onto the exact grid of the GeoTIFF at ``ref_path``."""
    with rasterio.open(ref_path) as ref:
        return read_ctx_to(tile, ref.transform, (ref.height, ref.width), ref.crs)


def mosaic_tiles(tiles: list[str], kind: str = "abundance", with_ctx: bool = True):
    """Merge several per-tile GeoTIFFs into one (array, ctx_or_None, transform, crs). nodata->NaN.

    With ``with_ctx`` (default) also returns a co-registered CTX-brightness mosaic on the same
    grid (so tile-boundary 'rectangle edges' can be inspected across the seam). Set
    ``with_ctx=False`` for tiles that have no cached CTX zip (e.g. just visualising abundance)."""
    from rasterio.merge import merge

    from src.mapping import assert_shared_lattice

    srcs = [rasterio.open(MAP_DIR / f"{t}_{kind}.tif") for t in tiles]
    # R01: this is a second merge path over the same per-tile products, and guarding only
    # `mapping.mosaic_geotiffs` would leave the striping/A1 analysis silently misregistered
    # -- exactly the failure the guard exists to make loud. Warn rather than raise here:
    # this function is the notebook-24/25 *analysis* path over already-shipped tiles, whose
    # whole subject is the artifact as it exists. See DECISIONS 2026-08-06x.
    try:
        assert_shared_lattice([s.transform for s in srcs])
    except ValueError as exc:
        warnings.warn(
            f"mosaic_tiles: {exc} -- merge() floors each tile's fractional offset, so these "
            "tiles are placed with a whole-cell displacement (measured on the shipped "
            "product: 25 of 26 tiles, median 140 m). Fine for inspecting the existing "
            "artifact; do not read positions off it.",
            RuntimeWarning, stacklevel=2,
        )
    arr, transform = merge(srcs)
    crs = srcs[0].crs
    nd = srcs[0].nodata
    for s in srcs:
        s.close()
    arr = arr[0].astype(np.float64)
    if nd is not None:
        arr[arr == nd] = np.nan
    ctx = None
    if with_ctx:
        ctx = np.full(arr.shape, np.nan)
        for t in tiles:
            c = read_ctx_to(t, transform, arr.shape, crs)
            ctx = np.where(np.isfinite(c), c, ctx)
    return arr, ctx, transform, crs


def lonlat_to_rc(transform, lon: float, lat: float, radius: float = 3396190.0):
    """Mosaic (row, col) for a lon/lat in the Mars equirectangular clon_0 CRS."""
    x = np.deg2rad(lon) * radius
    y = np.deg2rad(lat) * radius
    return (int(round((y - transform.f) / transform.e)),
            int(round((x - transform.c) / transform.a)))


def find_seam_shp(tile: str) -> Path | None:
    d = SEAM_DIR / f"_seammap_{tile}"
    if not d.exists():
        return None
    shps = list(d.glob("*SeamMap.shp"))
    return shps[0] if shps else None


def _padded_tile(tile: str) -> str:
    """Murray Lab zero-padded URL form: E8_N36 -> E008_N36, E-12_N32 -> E-012_N32."""
    import re

    m = re.fullmatch(r"E(-?\d+)_N(-?\d+)", tile)
    lon_i, lat_i = int(m.group(1)), int(m.group(2))
    lon = f"E{abs(lon_i):03d}" if lon_i >= 0 else f"E-{abs(lon_i):03d}"
    lat = f"N{abs(lat_i):02d}" if lat_i >= 0 else f"N-{abs(lat_i):02d}"
    return f"{lon}_{lat}"


def _tile_crs(tile: str):
    """CRS of a Murray tile, from the tile itself.

    **R07.** This used to be read off ``reports/map_region/{tile}_abundance.tif``, which made
    `load_frames` — and therefore the whole A1 statistic — fail outright for any tile with no
    rendered abundance product. That is not an edge case: the 39 Stage-2 training windows span
    **20** Murray tiles while only the 26 map-footprint tiles have an abundance raster, so the
    R07 training fix could not even run. Source frames are a property of the CTX tile, not of
    our product; take the CRS from the tile.
    """
    zp = CTX_ZIP_DIR / f"{tile}.zip"
    if zp.exists():
        with rasterio.open(f"/vsizip/{zp.as_posix()}/{_inner_tif_name(zp)}") as ds:
            return ds.crs
    ab = MAP_DIR / f"{tile}_abundance.tif"
    if ab.exists():
        with rasterio.open(ab) as ds:
            return ds.crs
    return None


def load_frames(tile: str, dissolve: bool = True):
    """Per-source-frame CTX footprints for ``tile``, in the Murray tile's CRS.

    The Murray Lab SeamMap is a *partition* (one source frame per pixel); its polygons are
    fragments, so by default we **dissolve by PRODUCT_ID** to recover the ~dozens of actual
    source CTX images. Reads the local cached SeamMap if present, else pulls just the shapefile
    out of the remote tile zip via range requests (``/vsizip/vsicurl/``) and caches the result
    as a GeoPackage. CRS comes from `_tile_crs` (the SeamMap .prj is the same Mars clon_0 CRS
    but is sometimes not fetched by vsicurl) — from the **tile**, not from our map product, so
    this works for tiles outside the 26-tile map footprint.
    """
    import os
    import geopandas as gpd

    cache_gpkg = SEAM_DIR / f"_frames_{tile}.gpkg"
    if cache_gpkg.exists():
        g = gpd.read_file(cache_gpkg)
    else:
        ab_crs = _tile_crs(tile)
        local = find_seam_shp(tile)
        if local is not None:
            g = gpd.read_file(local)
        else:
            os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
            try:
                import truststore
                truststore.inject_into_ssl()
            except Exception:
                pass
            pad = _padded_tile(tile)
            url = ("https://murray-lab.caltech.edu/CTX/V01/tiles/"
                   f"MurrayLab_GlobalCTXMosaic_V01_{pad}.zip")
            inner = (f"MurrayLab_GlobalCTXMosaic_V01_{pad}/"
                     f"MurrayLab_CTX_V01_{pad}_SeamMap.shp")
            g = gpd.read_file(f"/vsizip/vsicurl/{url}/{inner}")
        if g.crs is None:
            g = g.set_crs(ab_crs)
        elif ab_crs is not None and g.crs.to_string() != ab_crs.to_string():
            g = g.to_crs(ab_crs)
        if dissolve:
            g = g.dissolve(by="PRODUCT_ID", as_index=False)
        # the cache root may not exist yet on a fresh checkout -- see the SEAM_DIR note
        cache_gpkg.parent.mkdir(parents=True, exist_ok=True)
        g.to_file(cache_gpkg, driver="GPKG")
    return g


# ----------------------------------------------------------------- field maths
def detrend(a: np.ndarray, sig: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Remove the large-scale (geology) trend via a NaN-aware Gaussian (~5 km at sig=30 px).

    Returns (residual field with NaN outside coverage, finite mask).
    """
    finite = np.isfinite(a)
    f0 = np.where(finite, a, 0.0)
    m = finite.astype(float)
    trend = gaussian_filter(f0, sig) / np.maximum(gaussian_filter(m, sig), 1e-6)
    return np.where(finite, a - trend, np.nan), finite


def banding_indices(field: np.ndarray, finite: np.ndarray) -> tuple[float, float]:
    """Fraction of variance organised into vertical bands (col structure = N-S tracks)
    vs horizontal bands (row structure). High vertical >> horizontal = N-S striping."""
    f = np.where(finite, field, np.nan)
    tot = np.nanvar(f)
    if not np.isfinite(tot) or tot <= 0:
        return np.nan, np.nan
    vi = np.nanvar(np.nanmean(f, axis=0)) / tot
    hi = np.nanvar(np.nanmean(f, axis=1)) / tot
    return float(vi), float(hi)


def grad_mag(a: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude; NaN preserved outside coverage."""
    f = np.where(np.isfinite(a), a, 0.0)
    g = np.hypot(sobel(f, axis=1), sobel(f, axis=0))
    g[~np.isfinite(a)] = np.nan
    return g


# --------------------------------------------------------------- A1 mitigation
# Reference robust center/scale for per-frame CTX normalization, measured over the cohort+region
# (global median DN and median IQR across 380 source frames; scripts/striping_frame_radiometry.py).
A1_REF_MEDIAN = 125.0
A1_REF_IQR = 27.7


# **R38.** The floor for a VALID pixel, so that DN 0 in an A1 array means exactly and only
# "mosaic nodata". It used to be 0, which made a legitimately dark pixel indistinguishable from a
# data gap: `src.mapping` infers nodata from `arr == 0`, so such a pixel was counted as missing
# coverage, could push a whole tile past `max_zero_fraction`, and was excluded from `a1_stats`.
# Measured on the real native patch stacks: 0.041 % of valid pixels on the training path and
# 0.04-0.41 % at deploy; 0.64 % / 6.7 % of tiles carried at least one false-black pixel while
# still passing the mask. Three sibling implementations of this same stretch already floor at 1
# (`f_leg_b_embed.py`, `f_pilot_crop.py` x2) and say why.
#
# **This fixes the SENTINEL problem, not the INFORMATION problem, and the two must not be
# conflated** (R13, 2026-08-10): DN 0 and DN 1 damage the frozen embedding *identically to three
# decimals*, because the harm is blackness, not the sentinel value. So flooring at 1 does not
# rescue the dark tail — it only stops the tail being miscounted as absent data, and in doing so
# makes it invisible to the nodata gate. That is why `a1_clip_counts` exists and why the drivers
# record the clipped fraction separately: it is a RADIOMETRIC quality signal, not a coverage one.
# Brian's call 2026-08-10: record it, do not change the transfer function (R07 already cut the
# damage ~10x, leaving ~0.04 % of native pixels — too small to justify re-tuning A1_REF).
A1_VALID_FLOOR = 1


def a1_stats(arr: np.ndarray) -> tuple[float, float]:
    """Robust (median, IQR) of the valid (DN>0; mosaic nodata=0) pixels of a CTX array.

    A degenerate (zero) IQR returns NaN, **not** 1.0. The old `or 1.0` looked like a harmless
    guard but actively defeated the caller's: `a1_stats_native_tile` admits a frame only when
    `iqr > 0`, and substituting 1.0 sailed through that check and handed the frame a gain of
    `s0/1 = 27.7x`. NaN routes it to the fallback statistic instead, which is what the guard
    was for.
    """
    v = arr[arr > 0].astype(np.float64)
    if v.size < A1_MIN_FRAME_PX:
        return np.nan, np.nan
    med = float(np.median(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    return (med, iqr) if iqr > 0 else (np.nan, np.nan)


def a1_apply(arr: np.ndarray, med: float, iqr: float,
             m0: float = A1_REF_MEDIAN, s0: float = A1_REF_IQR,
             floor: int = A1_VALID_FLOOR) -> np.ndarray:
    """A1 normalization: remap CTX DN by robust offset+gain to the (m0, s0) reference,
    `(x - med)/iqr * s0 + m0`, clipped to `[floor, 255]` uint8. nodata (DN==0) stays 0.

    **R38: `floor` is 1, not 0.** See `A1_VALID_FLOOR`. Pass `floor=0` only to reproduce a
    pre-R38 artifact knowingly.

    Pass the SAME (med, iqr) for every patch of one source frame (single-frame training window
    or a deploy frame) so within-frame texture is preserved and only the between-frame level/scale
    is removed."""
    if not (np.isfinite(med) and np.isfinite(iqr) and iqr > 0):
        return arr.astype(np.uint8)
    a = arr.astype(np.float64)
    out = np.clip((a - med) / iqr * s0 + m0, floor, 255)
    out[arr == 0] = 0
    return out.astype(np.uint8)


def a1_clip_counts(arr: np.ndarray, med: float, iqr: float,
                   m0: float = A1_REF_MEDIAN, s0: float = A1_REF_IQR,
                   floor: int = A1_VALID_FLOOR) -> dict:
    """How many VALID pixels `a1_apply` would clip, split by end. **R38's measurable guard.**

    Flooring at 1 stops a clipped pixel being miscounted as nodata, but it does not un-destroy
    its texture — and after the fix nothing else can see it, because it no longer reads as 0.
    Counting it here is what keeps the information loss observable, and it is what surfaces the
    low-IQR frames where the clip actually bites (the worst frame in the committed 380-frame
    table has a threshold of +138.7 DN against a 160 m IQR of 6.4).
    """
    valid = arr > 0
    n_valid = int(valid.sum())
    if not n_valid or not (np.isfinite(med) and np.isfinite(iqr) and iqr > 0):
        return {"n_valid": n_valid, "n_floored": 0, "n_ceiled": 0}
    v = (arr[valid].astype(np.float64) - med) / iqr * s0 + m0
    return {"n_valid": n_valid,
            "n_floored": int((v < floor).sum()),
            "n_ceiled": int((v > 255).sum())}


def a1_normalize_window(arr: np.ndarray, **ref) -> np.ndarray:
    """A1 for a single-frame training window: derive (median, IQR) from the window itself."""
    med, iqr = a1_stats(arr)
    return a1_apply(arr, med, iqr, **ref)


def a1_normalize_per_frame(arr: np.ndarray, labels: np.ndarray, **ref) -> np.ndarray:
    """A1 at deploy: normalize each source frame (by its `labels` id) with its own robust stats.

    **R08 hazard, kept deliberately:** pixels in no frame, or in a frame with <50 valid pixels,
    are returned at **raw DN** — a mixture of normalized and unnormalized values in one array.
    Use `a1_normalize_native` instead for anything that feeds the embedder; this function is
    retained for the diagnostics that already reference it.
    """
    out = arr.copy()
    for f in np.unique(labels[labels >= 0]):
        sel = (labels == f) & (arr > 0)
        if sel.sum() < 50:
            continue
        med, iqr = a1_stats(np.where(sel, arr, 0))
        out[sel] = a1_apply(arr, med, iqr, **ref)[sel]
    return out


# ============================================================================
# R07 — the ONE A1 statistic, shared by training and deployment
# ============================================================================
#
# R07 measured: training normalised each Stage-2 window by ONE native-resolution statistic,
# while both deploy paths derive a per-SeamMap-frame statistic from CTX area-averaged to
# 160 m. Two independent mismatches, both quantified over all 39 Stage-2 windows:
#
#   resolution  IQR_native / IQR_160m = 1.35x median, 1.83x p95, 2.15x max. Training pins the
#               input IQR to exactly 27.7; deploy delivered a median of 37.3 (max 59.6) and
#               clipped ~10x more pixels.
#   unit        The training comment claimed "each training window is ~one CTX source frame".
#               Measured against the cached SeamMaps: only 10 of 38 windows lie in one frame;
#               22 span two, 3 span three, max four; dominant-frame share median 81%, min 48%.
#
# So training removed between-WINDOW scale and deployment removes between-FRAME scale. The
# functions below are the single definition both sides now call:
#   unit       = one dissolved SeamMap source frame
#   resolution = native 5 m/px DN
#   support    = the frame's extent within the parent Murray tile (see A1_ARM)
#   fallback   = pixels in no qualifying frame take the enclosing array's own native statistic,
#                never raw DN (that mixture is R08)

A1_MIN_FRAME_PX = 50            # matches the pre-existing threshold in the 160 m paths
# The canonical name of this arm. `src.modeling.mlp_head.A1_NORM_ARM` repeats the literal
# rather than importing it (that module would drag torch's OpenMP bootstrap into every
# notebook that touches striping); a test pins the two equal.
A1_ARM = "a1_native_perframe_tilesupport_v2"


def a1_stats_from_hist(hist) -> tuple[float, float]:
    """Exact robust (median, IQR) of uint8 DN from a 256-bin count histogram.

    Streaming a 2.2-Gpx Murray tile cannot hold its values, but uint8 has only 256 of them, so
    a histogram gives the *exact* percentiles rather than an approximation. DN 0 (mosaic
    nodata) is excluded, matching `a1_stats`.
    """
    h = np.asarray(hist, dtype=np.int64).copy()
    h[0] = 0                                       # nodata sentinel
    n = int(h.sum())
    if n < A1_MIN_FRAME_PX:
        return np.nan, np.nan
    c = np.cumsum(h)

    def pct(p):
        # numpy's linear interpolation on the sorted values, evaluated from the CDF
        x = p / 100.0 * (n - 1)
        lo, hi = int(np.floor(x)), int(np.ceil(x))
        v_lo = int(np.searchsorted(c, lo + 1))
        v_hi = int(np.searchsorted(c, hi + 1))
        return v_lo + (v_hi - v_lo) * (x - lo)

    med = float(pct(50))
    iqr = float(pct(75) - pct(25))
    # R38: NaN, not 1.0 — see `a1_stats`. `a1_stats_native_tile` admits a frame only when
    # `iqr > 0`, and the old `or 1.0` walked straight through that guard with a fabricated
    # IQR, giving the frame a gain of s0/1 = 27.7x instead of routing it to the fallback.
    return (med, iqr) if iqr > 0 else (np.nan, np.nan)


def a1_clip_counts_from_hist(hist, med: float, iqr: float,
                             m0: float = A1_REF_MEDIAN, s0: float = A1_REF_IQR,
                             floor: int = A1_VALID_FLOOR) -> dict:
    """**R38, exactly.** Valid pixels a frame's A1 remap would clip, from its DN histogram.

    uint8 has 256 possible values, so "how many pixels clip" is a dot product against the
    histogram rather than an estimate — and it costs nothing, because `frame_hist_native`
    already builds the histogram to derive (median, IQR) in the first place.

    Computing it here rather than per read window matters: windows overlap by 96 px, so summing
    per-window pixel counts double-counts the seams, and a resumed run would only see the
    windows it recomputed. Once per tile, from the histogram, is both exact and resume-proof.
    """
    h = np.asarray(hist, dtype=np.int64).copy()
    h[0] = 0                                       # nodata sentinel is not a valid pixel
    n_valid = int(h.sum())
    if not n_valid or not (np.isfinite(med) and np.isfinite(iqr) and iqr > 0):
        return {"n_valid": n_valid, "n_floored": 0, "n_ceiled": 0}
    dn = np.arange(256, dtype=np.float64)
    v = (dn - med) / iqr * s0 + m0
    return {"n_valid": n_valid,
            "n_floored": int(h[v < floor].sum()),
            "n_ceiled": int(h[v > 255].sum())}


def frame_labels_on(transform, shape, frames, *, dtype: str = "int32") -> np.ndarray:
    """Rasterize dissolved source frames onto an arbitrary grid; -1 where no frame covers."""
    return rasterize([(g, i) for i, g in enumerate(frames.geometry)], out_shape=tuple(shape),
                     transform=transform, fill=-1, dtype=dtype, all_touched=False)


def frame_hist_native(src_path: str | Path, frames, *, block: int = 4096,
                      n_frames: int | None = None, progress=None) -> np.ndarray:
    """Stream a native-resolution raster once, accumulating a per-frame DN histogram.

    Returns `(n_frames + 1, 256)` counts; row `n_frames` is the no-frame residue. Blocked so a
    GB-scale Murray tile never materialises, and the frame labels are rasterized per block for
    the same reason (a native-resolution label map for a whole tile would be ~4.5 GB).
    """
    n = len(frames) if n_frames is None else n_frames
    hist = np.zeros((n + 1, 256), dtype=np.int64)
    # A Murray tile carries ~30-90 dissolved frames but any one block touches a handful, and
    # rasterizing all of them per block dominated the runtime (measured: ~45 min/tile, which
    # would have made the per-frame native statistic impractical for both training and deploy).
    # Pre-filtering by bounding box is what makes R07's fix affordable.
    bounds = np.array([g.bounds for g in frames.geometry], dtype=np.float64)
    geoms = list(frames.geometry)
    with rasterio.open(str(src_path)) as ds:
        H, W = ds.height, ds.width
        for r0 in range(0, H, block):
            h = min(block, H - r0)
            for c0 in range(0, W, block):
                w = min(block, W - c0)
                win = rasterio.windows.Window(c0, r0, w, h)
                arr = ds.read(1, window=win)
                if not arr.any():
                    continue
                tr = ds.window_transform(win)
                bx0, by1 = tr * (0, 0)
                bx1, by0 = tr * (w, h)
                hit = np.where((bounds[:, 0] < bx1) & (bounds[:, 2] > bx0)
                               & (bounds[:, 1] < by1) & (bounds[:, 3] > by0))[0]
                if hit.size:
                    lab = rasterize([(geoms[i], int(i) + 1) for i in hit],
                                    out_shape=arr.shape, transform=tr, fill=0,
                                    dtype="int32", all_touched=False) - 1
                else:
                    lab = np.full(arr.shape, -1, dtype=np.int32)
                lab[lab < 0] = n                    # residue row
                # bincount over the flattened (frame, DN) index; `np.add.at` on 16 M elements
                # is an order of magnitude slower and this runs once per block per tile
                idx = (lab.ravel().astype(np.int32) << 8) | arr.ravel()
                hist += np.bincount(idx, minlength=(n + 1) * 256).reshape(n + 1, 256)
            if progress:
                progress(min(r0 + block, H), H)
    return hist


def a1_stats_native_tile(tile: str, frames, *, zip_dir: Path | None = None,
                         block: int = 4096, progress=None) -> tuple[dict, tuple, dict]:
    """Per-frame native (median, IQR) over each frame's extent in the whole Murray tile.

    Returns `(stats, fallback_stats, provenance)`. `stats` maps frame index ->
    `(median, IQR)` for frames with at least `A1_MIN_FRAME_PX` valid pixels;
    `fallback_stats` is the tile-wide native statistic used for everything else.
    """
    zd = Path(zip_dir) if zip_dir is not None else CTX_ZIP_DIR
    zip_path = zd / f"{tile}.zip"
    vsizip = f"/vsizip/{zip_path.as_posix()}/{_inner_tif_name(zip_path)}"
    hist = frame_hist_native(vsizip, frames, block=block, progress=progress)
    n = len(frames)
    stats, small = {}, []
    for i in range(n):
        med, iqr = a1_stats_from_hist(hist[i])
        if np.isfinite(med) and np.isfinite(iqr) and iqr > 0:
            stats[i] = (med, iqr)
        elif hist[i].sum():
            small.append(i)
    fallback = a1_stats_from_hist(hist.sum(axis=0))
    counts = hist.sum(axis=1)
    total = int(counts.sum() - hist[:, 0].sum())
    covered = int(sum(counts[i] - hist[i, 0] for i in stats))

    # R38: what the clip destroys, measured exactly and once. Every valid pixel is normalized by
    # either its own frame's statistic or the fallback, so summing those two populations covers
    # the tile with no overlap. `worst_frames` is what surfaces the low-IQR frames where the clip
    # actually bites -- the whole point of recording this rather than leaving it invisible.
    clip = {"n_valid": 0, "n_floored": 0, "n_ceiled": 0}
    per_frame = {}
    for i in range(n):
        c = (a1_clip_counts_from_hist(hist[i], *stats[i]) if i in stats
             else a1_clip_counts_from_hist(hist[i], *fallback))
        for k in clip:
            clip[k] += c[k]
        if c["n_valid"] and (c["n_floored"] or c["n_ceiled"]):
            per_frame[i] = {**c, "clipped_fraction":
                            (c["n_floored"] + c["n_ceiled"]) / c["n_valid"]}
    residue = a1_clip_counts_from_hist(hist[n], *fallback)     # pixels in no frame at all
    for k in clip:
        clip[k] += residue[k]
    clipped = clip["n_floored"] + clip["n_ceiled"]
    worst = sorted(per_frame.items(), key=lambda kv: -kv[1]["clipped_fraction"])[:5]

    prov = {
        "a1_arm": A1_ARM, "statistic": "median_iqr", "resolution": "native_5m",
        "unit": "seammap_source_frame", "support": "frame_extent_in_murray_tile",
        "min_frame_px": A1_MIN_FRAME_PX, "n_frames": n, "n_frames_with_stats": len(stats),
        "n_frames_too_small": len(small), "frames_too_small": small,
        "fallback_median": fallback[0], "fallback_iqr": fallback[1],
        "fallback_pixel_fraction": (1.0 - covered / total) if total else None,
        # R38 -- a RADIOMETRIC loss statistic, deliberately kept apart from the nodata counts.
        # A clipped pixel is not a data gap and (since the floor moved to 1) no longer looks
        # like one; this is the only place its texture loss is visible.
        "clip_floor": A1_VALID_FLOOR,
        "clip_n_valid_px": clip["n_valid"],
        "clip_n_floored_px": clip["n_floored"],
        "clip_n_ceiled_px": clip["n_ceiled"],
        "clip_fraction": (clipped / clip["n_valid"]) if clip["n_valid"] else None,
        "clip_n_frames_affected": len(per_frame),
        "clip_worst_frames": [{"frame": int(i), **v} for i, v in worst],
    }
    return stats, fallback, prov


def a1_normalize_native(arr: np.ndarray, labels: np.ndarray, stats: dict,
                        fallback: tuple[float, float] | None = None, **ref) -> np.ndarray:
    """Apply the R07 A1 statistic to a native-DN array. **No pixel is left at raw DN.**

    `stats` is `{frame_index: (median, IQR)}`; every valid pixel whose frame is absent from it
    — unlabelled, or a frame below `A1_MIN_FRAME_PX` — is normalized by `fallback` instead.
    Leaving those at raw DN is R08: it puts two different radiometric scales into one array and
    hands the mixture to a frozen embedder that cannot tell them apart.

    **R08's contract is RATIFIED (Brian, 2026-08-10): normalize them, never drop them.** The
    open alternative was to mask the fallback population as nodata, which is exact but removes
    real ground. Measured on three whole Murray tiles, it is the wrong trade by three orders of
    magnitude: the population is *isolated single pixels* (horizontal run length median 1, p90 2)
    at 0.0058–0.0108 % of valid pixels, but dropping them makes each one nodata and R13's
    zero-tolerance context gate then sterilises every cell whose 96-px box touches one —
    **3.11 % / 3.37 % / 4.38 % of the tile** against 0.00 / 0.00 / 0.072 % today. `A1_MIN_FRAME_PX`
    is likewise a tripwire rather than a tuning knob: exactly **1 frame of 214** across four real
    tiles fell below it. Both halves are pinned in `tests/test_a1_statistic.py`.

    **R38.** Output DN 0 now means *only* nodata — valid pixels floor at `A1_VALID_FLOOR`. How
    much the clip destroys is not counted here: `a1_stats_native_tile` derives it exactly, once
    per tile, from the DN histogram it already builds (`a1_clip_counts_from_hist`).
    """
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = arr > 0
    done = np.zeros(arr.shape, dtype=bool)
    for f, (med, iqr) in stats.items():
        sel = (labels == f) & valid
        if sel.any():
            out[sel] = a1_apply(arr, med, iqr, **ref)[sel]
            done |= sel
    rest = valid & ~done
    if rest.any():
        if fallback is None or not np.isfinite(fallback[0]):
            raise ValueError(
                f"{int(rest.sum())} valid pixels are in no qualifying frame and no fallback "
                f"statistic was supplied; returning them as raw DN is the R08 defect.")
        out[rest] = a1_apply(arr, fallback[0], fallback[1], **ref)[rest]
    return out


def stripe_enhance(a: np.ndarray, along: float = 20.0, trend: float = 30.0) -> np.ndarray:
    """Highlight VERTICAL (N-S) striping: detrend, then smooth ALONG the stripe (down rows)
    so column-coherent bands survive while isotropic speckle averages out. Returns the
    enhanced residual (NaN outside coverage)."""
    finite = np.isfinite(a)
    det, _ = detrend(a, trend)
    enh = gaussian_filter(np.nan_to_num(det), (along, 0.0))
    return np.where(finite, enh, np.nan)


def angular_radial_power(field: np.ndarray):
    """2-D FFT log-power + power binned by wavevector orientation and radial wavenumber.

    Orientation = angle of the spatial wavevector from +x (East). Vertical stripes ->
    horizontal wavevector -> ~0 deg; horizontal stripes -> ~90 deg.
    """
    H, W = field.shape
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    F = np.fft.fftshift(np.fft.fft2(field * win))
    power = np.abs(F) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=PX_M))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(W, d=PX_M))[None, :]
    kr = np.sqrt(kx**2 + ky**2)
    ang = np.mod(np.degrees(np.arctan2(ky * np.ones_like(kx),
                                       kx * np.ones_like(ky))), 180.0)
    kr_max = kr.max()
    sel = (kr > 0.02 * kr_max) & (kr < 0.9 * kr_max)
    nbin = 36
    edges = np.linspace(0, 180, nbin + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    apow = np.array([power[sel & (ang >= edges[i]) & (ang < edges[i + 1])].sum()
                     for i in range(nbin)])
    apow = apow / (apow.sum() + 1e-30)
    return power, centers, apow


def frame_label_map(tile: str, frames) -> np.ndarray:
    """Rasterize dissolved CTX source frames -> int label per pixel on the abundance grid
    (-1 = no frame). Row index in ``frames`` is the label."""
    with rasterio.open(MAP_DIR / f"{tile}_abundance.tif") as ds:
        shape, transform = (ds.height, ds.width), ds.transform
    shapes = [(geom, i) for i, geom in enumerate(frames.geometry)]
    return rasterize(shapes, out_shape=shape, transform=transform, fill=-1,
                     dtype="int32", all_touched=False)


def eta2(values: np.ndarray, labels: np.ndarray, finite: np.ndarray) -> float:
    """Fraction of variance of ``values`` explained by group ``labels`` (between/total).

    The spatial analogue: how much of the abundance variance is organised *between* CTX
    source frames. NaN-safe (a rolled null can drag NaNs into the coverage mask)."""
    v = values[finite]
    lab = labels[finite]
    keep = (lab >= 0) & np.isfinite(v)
    v, lab = v[keep], lab[keep]
    if v.size < 2:
        return np.nan
    gm = v.mean()
    ss_tot = np.sum((v - gm) ** 2)
    if ss_tot <= 0:
        return np.nan
    ss_bet = sum(sel.sum() * (v[sel].mean() - gm) ** 2 for sel in (lab == f for f in np.unique(lab)))
    return float(ss_bet / ss_tot)


def eta2_rotation_null(values, labels, finite, n: int = 40, seed: int = 0):
    """Null for :func:`eta2`: roll the value field under the fixed frame mask (keeps block
    geometry, breaks frame/geology alignment). Returns (mean, 95th-percentile)."""
    rng = np.random.default_rng(seed)
    H, W = values.shape
    out = [eta2(np.roll(values, (rng.integers(H // 8, H), rng.integers(W // 8, W)), (0, 1)),
                labels, finite) for _ in range(n)]
    out = [o for o in out if np.isfinite(o)]
    return (float(np.mean(out)), float(np.percentile(out, 95))) if out else (np.nan, np.nan)


def per_frame_stats(tile: str, frames, resid, ctx, ab, finite, labels, min_px: int = 50):
    """Per-frame table: mean detrended-abundance, mean CTX DN, mean raw abundance."""
    import pandas as pd

    rows = []
    for i in range(len(frames)):
        sel = finite & (labels == i)
        if sel.sum() < min_px:
            continue
        rows.append(dict(tile=tile, frame=str(frames.iloc[i].get("PRODUCT_ID", i)),
                         n_px=int(sel.sum()), mean_resid=float(resid[sel].mean()),
                         mean_ctx=float(ctx[sel].mean()), mean_ab=float(ab[sel].mean())))
    return pd.DataFrame(rows)


def boundary_steps(resid, ctx, labels, min_pairpx: int = 30):
    """Adjacent-frame near-boundary step table: per frame pair, the mean resid/ctx on each
    side of their shared seam -> (dResid, dCtx). Geology is continuous across a seam, so a
    correlation of dResid with dCtx isolates radiometry from geology."""
    import pandas as pd

    parts = []
    for ax in (0, 1):
        a = labels.take(range(0, labels.shape[ax] - 1), axis=ax)
        b = labels.take(range(1, labels.shape[ax]), axis=ax)
        diff = (a != b) & (a >= 0) & (b >= 0)
        if not diff.any():
            continue
        ra = resid.take(range(0, resid.shape[ax] - 1), axis=ax)[diff]
        rb = resid.take(range(1, resid.shape[ax]), axis=ax)[diff]
        ca = ctx.take(range(0, ctx.shape[ax] - 1), axis=ax)[diff]
        cb = ctx.take(range(1, ctx.shape[ax]), axis=ax)[diff]
        ai, bi = a[diff], b[diff]
        lo = np.minimum(ai, bi)
        a_is_lo = ai == lo
        parts.append(pd.DataFrame({
            "lo": lo, "hi": np.maximum(ai, bi),
            "r_lo": np.where(a_is_lo, ra, rb), "r_hi": np.where(a_is_lo, rb, ra),
            "c_lo": np.where(a_is_lo, ca, cb), "c_hi": np.where(a_is_lo, cb, ca)}))
    if not parts:
        return pd.DataFrame(columns=["dResid", "dCtx", "n"])
    df = pd.concat(parts, ignore_index=True).dropna()
    g = df.groupby(["lo", "hi"]).agg(r_lo=("r_lo", "mean"), r_hi=("r_hi", "mean"),
                                     c_lo=("c_lo", "mean"), c_hi=("c_hi", "mean"),
                                     n=("r_lo", "size")).reset_index()
    g = g[g["n"] >= min_pairpx]
    return pd.DataFrame({"dResid": g["r_hi"] - g["r_lo"], "dCtx": g["c_hi"] - g["c_lo"], "n": g["n"]})


def seam_line_mask(tile: str, ref_path: str | Path):
    """Rasterize the per-frame footprint *boundaries* (candidate seams) onto the grid,
    and return (mask, geodataframe-in-ref-CRS) for plotting/analysis."""
    import geopandas as gpd

    g = gpd.read_file(find_seam_shp(tile))
    with rasterio.open(ref_path) as ref:
        out_shape = (ref.height, ref.width)
        transform, crs = ref.transform, ref.crs
    if g.crs is not None and crs is not None and g.crs.to_string() != crs.to_string():
        g = g.to_crs(crs)
    lines = []
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        b = geom.boundary
        if isinstance(b, (LineString, MultiLineString)):
            lines.append(b)
    mask = rasterize(((ln, 1) for ln in lines), out_shape=out_shape,
                     transform=transform, fill=0, dtype="uint8", all_touched=True)
    return mask.astype(bool), g
