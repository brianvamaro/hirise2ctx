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
SEAM_DIR = REPO / "cache" / "ctx_tiles"
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


def load_frames(tile: str, dissolve: bool = True):
    """Per-source-frame CTX footprints for ``tile``, in the abundance CRS.

    The Murray Lab SeamMap is a *partition* (one source frame per pixel); its polygons are
    fragments, so by default we **dissolve by PRODUCT_ID** to recover the ~dozens of actual
    source CTX images. Reads the local cached SeamMap if present, else pulls just the shapefile
    out of the remote tile zip via range requests (``/vsizip/vsicurl/``) and caches the result
    as a GeoPackage. CRS is taken from the tile's abundance raster (the SeamMap .prj is the same
    Mars clon_0 CRS but is sometimes not fetched by vsicurl).
    """
    import os
    import geopandas as gpd

    cache_gpkg = SEAM_DIR / f"_frames_{tile}.gpkg"
    with rasterio.open(MAP_DIR / f"{tile}_abundance.tif") as ds:
        ab_crs = ds.crs
    if cache_gpkg.exists():
        g = gpd.read_file(cache_gpkg)
    else:
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


def a1_stats(arr: np.ndarray) -> tuple[float, float]:
    """Robust (median, IQR) of the valid (DN>0; mosaic nodata=0) pixels of a CTX array."""
    v = arr[arr > 0].astype(np.float64)
    if v.size < 50:
        return np.nan, np.nan
    med = float(np.median(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25]))) or 1.0
    return med, iqr


def a1_apply(arr: np.ndarray, med: float, iqr: float,
             m0: float = A1_REF_MEDIAN, s0: float = A1_REF_IQR) -> np.ndarray:
    """A1 normalization: remap CTX DN by robust offset+gain to the (m0, s0) reference,
    `(x - med)/iqr * s0 + m0`, clipped to [0,255] uint8. nodata (DN==0) stays 0.

    Pass the SAME (med, iqr) for every patch of one source frame (single-frame training window
    or a deploy frame) so within-frame texture is preserved and only the between-frame level/scale
    is removed."""
    if not np.isfinite(med):
        return arr.astype(np.uint8)
    a = arr.astype(np.float64)
    out = np.clip((a - med) / iqr * s0 + m0, 0, 255)
    out[arr == 0] = 0
    return out.astype(np.uint8)


def a1_normalize_window(arr: np.ndarray, **ref) -> np.ndarray:
    """A1 for a single-frame training window: derive (median, IQR) from the window itself."""
    med, iqr = a1_stats(arr)
    return a1_apply(arr, med, iqr, **ref)


def a1_normalize_per_frame(arr: np.ndarray, labels: np.ndarray, **ref) -> np.ndarray:
    """A1 at deploy: normalize each source frame (by its `labels` id) with its own robust stats."""
    out = arr.copy()
    for f in np.unique(labels[labels >= 0]):
        sel = (labels == f) & (arr > 0)
        if sel.sum() < 50:
            continue
        med, iqr = a1_stats(np.where(sel, arr, 0))
        out[sel] = a1_apply(arr, med, iqr, **ref)[sel]
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
