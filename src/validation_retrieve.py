"""Independent validation-raster retrieval for the regional map (PLAN_RegionalMap §3, phase 1).

Net-new for the circum-Chryse regional map: fetch + reproject the external reference layers
(one **topographic**, two **thermal**) onto the **CTX mosaic CRS** so they co-register with
the predicted abundance/probability GeoTIFFs (`reports/map_region/`). Three products, all
GDAL-readable global mosaics (URLs + facts verified 2026-06-17, recorded in DECISIONS.md):

- **MOLA MEGDM 463 m/px** topography (USGS planetarymaps) — shaded-relief context underlay
  (PLAN §5 fig-1) + the −3795 / −4100 m paleoshoreline contours (leg 3).
- **THEMIS night-IR 100 m/px** (USGS, 60N60S) — the thermal-bright deposit proxy for the
  spatial-co-location money panel (legs 1–2). **15 GB global → windowed `/vsicurl/` only.**
- **TES thermal inertia ~3 km/px** (ASU Putzig & Mellon nightside 2005) — the calibrated
  rockiness measure for the rank-correlation leg. (`nmap2003.tif` is small and global.)

Design mirrors `ctx_retrieve`: the source unit/CRS is **read from the file, never assumed**
(USGS "simple cylindrical" mosaics may be tagged in metres or degrees), our geographic
region bounds are projected into the source CRS via pyproj, only the covering window is read,
and the result is warped onto a target grid in the CTX clon_0 CRS. Default read mode is
windowed `/vsicurl/` (THEMIS is too big to download); a download-then-read fallback exists.
TLS: set `HIRISE2CTX_INSECURE_TLS=1` to skip GDAL cert verification (same opt-in used on
Sherlock for the USGS/Caltech incomplete-chain hosts).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

VALIDATION_SUBDIR = "validation"
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


# =====================================================================================
# Source access (windowed /vsicurl read, or download-then-read fallback)
# =====================================================================================


def _gdal_http_env() -> dict:
    """GDAL/curl env for robust `/vsicurl/` reads of the USGS / ASU mosaics.

    Points curl at certifi's CA bundle (this conda env's stdlib trust store is incomplete,
    same issue `ctx_retrieve` works around with truststore). Honours the project-wide
    `HIRISE2CTX_INSECURE_TLS=1` opt-in (the USGS/Caltech hosts send incomplete cert chains
    that Linux OpenSSL won't AIA-complete — see SHERLOCK_RUN.md / DECISIONS.md). Also tunes
    vsicurl to not list the directory and to cache fetched ranges.
    """
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
        "VSI_CACHE": "TRUE",
        "GDAL_HTTP_USERAGENT": "hirise2ctx/0.1 (research; brianvamaro@gmail.com)",
    }
    if os.environ.get("HIRISE2CTX_INSECURE_TLS") == "1":
        env["GDAL_HTTP_UNSAFESSL"] = "YES"
    else:
        try:
            import certifi
            env["CURL_CA_BUNDLE"] = certifi.where()
        except ImportError:
            pass
    return env


def _download_raster(url: str, dest_path: Path, *, timeout: float = 120.0) -> Path:
    """Stream `url` to `dest_path` (temp-then-rename). No size floor (TES is ~tens of MB).

    Reuses `ctx_retrieve`'s truststore side-effect for OS-level SSL trust on Windows.
    """
    import urllib.request

    from . import pds_labels  # noqa: F401  (triggers truststore.inject_into_ssl())

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(
        url, headers={"User-Agent": "hirise2ctx/0.1 (research; brianvamaro@gmail.com)"}
    )
    from . import net

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
            declared = net.content_length_of(resp)
            while True:
                chunk = resp.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
        # R66: `resp.read(amt)` returns b"" on a premature EOF rather than raising, so this
        # loop cannot tell a finished download from a dropped connection. This path had no
        # size floor at all, so a short file was committed silently. Measured before the
        # fix: 4,883,003 bytes committed against a declared 8,878,189.
        net.verify_download(tmp, url=url, declared_length=declared)
        tmp.replace(dest_path)
    finally:
        tmp.unlink(missing_ok=True)
    return dest_path


@contextmanager
def open_source(url: str, *, read_mode: str = "vsicurl", cache_dir: str | Path | None = None):
    """Open a remote raster as a rasterio dataset, windowed-read ready.

    - `read_mode="vsicurl"` (default): open `/vsicurl/{url}` under the GDAL HTTP env so only
      the requested window's byte ranges transfer. Required for the 15 GB THEMIS mosaic.
    - `read_mode="download"`: fetch the whole file into `<cache_dir>/thermal/_src/` once, then
      open locally. Fine for the small/medium MOLA + TES mosaics; never for THEMIS.
    """
    import rasterio

    if read_mode == "download":
        if cache_dir is None:
            raise ValueError("read_mode='download' requires cache_dir")
        src_dir = Path(cache_dir) / VALIDATION_SUBDIR / "_src"
        local = src_dir / url.rsplit("/", 1)[-1]
        if not local.exists():
            _download_raster(url, local)
        with rasterio.open(local) as ds:
            yield ds
        return

    if read_mode != "vsicurl":
        raise ValueError(f"unknown read_mode {read_mode!r} (expected 'vsicurl' or 'download')")
    with rasterio.Env(**_gdal_http_env()):
        with rasterio.open(f"/vsicurl/{url}") as ds:
            yield ds


# =====================================================================================
# Geometry — region bounds <-> source/target CRS, target grid construction
# =====================================================================================


def seam_lon(crs_wkt: str) -> float:
    """Longitude (deg, -180..180) of the source CRS's left/right edge = its 'seam'.

    For an equirectangular raster the seam sits at `central_meridian + 180`. A region that
    crosses it projects to a *wrapped* (wrong, antipodal) axis-aligned bbox, so we split the
    read there. `central_meridian` 0 -> seam ±180 (the usual -180/180 mosaics: MOLA, TES);
    `central_meridian` 180 -> seam 0 (THEMIS night-IR, which our circum-Chryse region crosses).
    """
    import pyproj

    cm = pyproj.CRS.from_user_input(crs_wkt).to_dict().get("lon_0", 0.0)
    return ((cm + 360.0) % 360.0) - 180.0


def split_bounds_at_seam(bounds_lonlat, seam: float, eps: float = 1e-4):
    """Split a (W, S, E, N) bbox into 1 or 2 sub-bboxes that don't straddle `seam`.

    If `W < seam < E` the region wraps the raster edge -> return the two halves (nudged
    `eps` off the seam to avoid the ±180 projection ambiguity at the exact edge). Each half
    then projects to a correct contiguous bbox; the caller reads + reprojects both and merges.
    """
    w, s, e, n = (float(v) for v in bounds_lonlat)
    if w < seam < e:
        return [(w, s, seam - eps, n), (seam + eps, s, e, n)]
    return [(w, s, e, n)]


def bounds_lonlat_to_crs(bounds_lonlat, crs_wkt: str):
    """Project a geographic (W, S, E, N) bbox into `crs_wkt`'s coordinates.

    Transforms the four corners (geodetic deg, planetocentric) into the target CRS via
    pyproj and returns the enclosing axis-aligned bbox. Works whether the target CRS is a
    geographic CRS tagged in degrees or a projected equirectangular CRS in metres. The caller
    must pre-split any seam-crossing region (`split_bounds_at_seam`) — a wrapped bbox here
    would otherwise enclose the antipode.
    """
    import pyproj

    crs = pyproj.CRS.from_user_input(crs_wkt)
    geo = crs.geodetic_crs
    w, s, e, n = (float(v) for v in bounds_lonlat)
    tf = pyproj.Transformer.from_crs(geo, crs, always_xy=True)
    xs, ys = tf.transform([w, e, w, e], [s, s, n, n])
    return (min(xs), min(ys), max(xs), max(ys))


def build_target_grid(bounds_lonlat, dst_crs_wkt: str, res_m: float):
    """Return `(transform, width, height)` for a north-up grid in `dst_crs_wkt`.

    The grid covers `bounds_lonlat` (projected into the CTX CRS) at `res_m` per pixel,
    expanded outward to whole pixels. `res_m` is interpreted in the CTX CRS's linear unit
    (metres for the clon_0 equirectangular mosaic CRS).
    """
    from rasterio.transform import Affine

    left, bottom, right, top = bounds_lonlat_to_crs(bounds_lonlat, dst_crs_wkt)
    width = int(math.ceil((right - left) / res_m))
    height = int(math.ceil((top - bottom) / res_m))
    transform = Affine(res_m, 0.0, left, 0.0, -res_m, top)
    return transform, width, height


def windowed_read(src, bounds_lonlat, *, buffer_deg: float = 0.5):
    """Read only the window of `src` covering `bounds_lonlat` (+ `buffer_deg`).

    Returns `(array2d, window_transform)`. The buffer guards against edge resampling and
    the ~200 m CTX/THEMIS co-registration slack at the boundary. `src` is an open rasterio
    dataset; its own CRS is used to place the window (units read from the file). The region
    must not straddle the source seam (`split_bounds_at_seam` first) — `fetch_region_raster`
    handles that.
    """
    from rasterio.windows import from_bounds

    w, s, e, n = bounds_lonlat
    buffered = (w - buffer_deg, s - buffer_deg, e + buffer_deg, n + buffer_deg)
    left, bottom, right, top = bounds_lonlat_to_crs(buffered, src.crs.to_wkt())
    win = from_bounds(left, bottom, right, top, transform=src.transform)
    # clamp to the raster and snap to whole pixels
    col_off = max(0, int(math.floor(win.col_off)))
    row_off = max(0, int(math.floor(win.row_off)))
    col_end = min(src.width, int(math.ceil(win.col_off + win.width)))
    row_end = min(src.height, int(math.ceil(win.row_off + win.height)))
    if col_end <= col_off or row_end <= row_off:
        raise ValueError(
            f"requested window {bounds_lonlat} does not intersect source raster "
            f"(src bounds {src.bounds}); check the longitude domain."
        )
    win = win.__class__(col_off=col_off, row_off=row_off,
                        width=col_end - col_off, height=row_end - row_off)
    data = src.read(1, window=win)
    return data, src.window_transform(win)


def reproject_to_grid(src_array, src_transform, src_crs_wkt, *, dst_crs_wkt, dst_transform,
                      dst_shape, resampling: str = "bilinear", src_nodata=None):
    """Warp `src_array` onto the explicit `(dst_crs, dst_transform, dst_shape)` grid.

    Returns a float32 array with NaN where the destination has no source coverage. Used to
    land each thermal/topo product on the same grid as (or co-registered with) the CTX
    abundance mosaic. `resampling` is any `rasterio.enums.Resampling` name (`nearest` for
    categorical/index rasters, `bilinear`/`cubic` for continuous fields).
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_array.astype(np.float32, copy=False),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs_wkt,
        dst_transform=dst_transform,
        dst_crs=dst_crs_wkt,
        src_nodata=src_nodata,
        dst_nodata=np.nan,
        resampling=getattr(Resampling, resampling),
    )
    return dst


# =====================================================================================
# Hillshade (for the MOLA shaded-relief context underlay; pure numpy, testable)
# =====================================================================================


def hillshade(dem, *, res_m: float = 463.0, azimuth_deg: float = 315.0,
              altitude_deg: float = 45.0, z_factor: float = 1.0):
    """Standard Horn-gradient hillshade of a DEM, in [0, 1].

    `dem` is elevation (m) on a regular grid of `res_m` spacing; NaNs propagate. Used only
    for the regional-context figure underlay (PLAN §5 fig-1) — not a quantitative product.
    """
    dem = np.asarray(dem, dtype=np.float64)
    dy, dx = np.gradient(dem * z_factor, res_m)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)  # azimuth measured clockwise from north
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    shaded = (np.sin(alt) * np.cos(slope)
              + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    shaded = np.clip(shaded, 0.0, 1.0)
    shaded[~np.isfinite(dem)] = np.nan  # carry the DEM's nodata through (gradient hides it)
    return shaded


# =====================================================================================
# Orchestration — fetch one product onto the CTX-CRS regional grid (cached)
# =====================================================================================


def reference_grid(reference_tif: str | Path):
    """Read `(crs_wkt, transform, (height, width))` from an existing GeoTIFF.

    Pass the regional abundance/probability mosaic to land a thermal product on the
    **exact same grid** (for per-pixel rank-correlation legs); or omit and use
    `build_target_grid` for a custom-resolution context underlay.
    """
    import rasterio

    with rasterio.open(reference_tif) as src:
        return src.crs.to_wkt(), src.transform, (int(src.height), int(src.width))


def fetch_region_raster(
    product: str,
    *,
    source_url: str,
    bounds_lonlat,
    dst_crs_wkt: str,
    out_path: str | Path,
    cache_dir: str | Path,
    dst_transform=None,
    dst_shape=None,
    dst_res_m: float | None = None,
    read_mode: str = "vsicurl",
    resampling: str = "bilinear",
    src_nodata=None,
    buffer_deg: float = 0.5,
    overwrite: bool = False,
) -> dict:
    """Fetch one product, reproject onto the CTX-CRS regional grid, cache the GeoTIFF.

    Target grid is either explicit (`dst_transform` + `dst_shape`, e.g. from
    `reference_grid` for co-registration) or built from `dst_res_m` via `build_target_grid`.
    Writes `out_path` (float32, NaN nodata, `dst_crs_wkt`) + an `<out_path>.json` provenance
    sidecar, and returns the provenance dict. Idempotent: skips the network if `out_path`
    exists and `overwrite` is False.
    """
    from . import mapping

    out_path = Path(out_path)
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    if out_path.exists() and sidecar.exists() and not overwrite:
        return json.loads(sidecar.read_text(encoding="utf-8"))

    if dst_transform is None or dst_shape is None:
        if dst_res_m is None:
            raise ValueError("provide either (dst_transform, dst_shape) or dst_res_m")
        dst_transform, width, height = build_target_grid(bounds_lonlat, dst_crs_wkt, dst_res_m)
        dst_shape = (height, width)

    with open_source(source_url, read_mode=read_mode, cache_dir=cache_dir) as src:
        src_crs_wkt = src.crs.to_wkt()
        src_nd = src.nodata if src_nodata is None else src_nodata
        # Pre-buffer, then split at the source seam so a region crossing the raster edge
        # (e.g. circum-Chryse over THEMIS's central_meridian=180 mosaic) reads as two halves.
        bw, bs, be, bn = bounds_lonlat
        buffered = (bw - buffer_deg, bs - buffer_deg, be + buffer_deg, bn + buffer_deg)
        parts = split_bounds_at_seam(buffered, seam_lon(src_crs_wkt))
        out_arr = None
        for part in parts:
            data, win_transform = windowed_read(src, part, buffer_deg=0.0)
            a = reproject_to_grid(
                data, win_transform, src_crs_wkt,
                dst_crs_wkt=dst_crs_wkt, dst_transform=dst_transform, dst_shape=dst_shape,
                resampling=resampling, src_nodata=src_nd,
            )
            out_arr = a if out_arr is None else np.where(np.isfinite(out_arr), out_arr, a)

    mapping.write_geotiff(out_path, out_arr, dst_transform, dst_crs_wkt)

    provenance = {
        "product": product,
        "source_url": source_url,
        "read_mode": read_mode,
        "bounds_lonlat": [float(v) for v in bounds_lonlat],
        "n_seam_parts": len(parts),
        "src_crs_wkt": src_crs_wkt,
        "src_nodata": None if src_nd is None else float(src_nd),
        "dst_crs_wkt": dst_crs_wkt,
        "dst_transform": [dst_transform[i] for i in range(6)],
        "dst_shape": [int(dst_shape[0]), int(dst_shape[1])],
        "resampling": resampling,
        "buffer_deg": float(buffer_deg),
        "valid_fraction": float(np.isfinite(out_arr).mean()),
        "fetched_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
