"""HiRISE imagery for QA overlays. Two-tier caching to minimize streaming:

1. **JP2 download cache** (`cache/hirise_jp2/{ObsId}_RED.JP2`). Optional but encouraged
   for repeat visual analyses on the same image — once you've downloaded the JP2 once
   (~200-500 MB over plain HTTP, single connection, much faster than many small
   /vsicurl/ range requests), every subsequent decimated read or native window read is
   a local file operation. Trade ~hundreds of MB of disk for instant zooms anywhere.

2. **Derived caches** (`cache/hirise_decimated/{ObsId}_*.tif`). GeoTIFFs at specific
   resolutions and bounds, derived from whichever source is available (local JP2 if
   present, else /vsicurl/). Idempotent — same parameters reuse the same file.

Access patterns:

- `ensure_jp2_local(obs_id, jp2_url, cache_dir)`: download the JP2 once. Call at
  notebook start for images you'll zoom on multiple times.

- `read_full_footprint_decimated(obs_id, jp2_url, cache_dir, target_mpp)`: full image
  at a coarse resolution (e.g. ~5 m/px).

- `read_native_window(obs_id, jp2_url, window_bounds_src_crs, cache_dir, window_tag)`:
  a small geographic window at native resolution. Crops from the local JP2 if cached;
  otherwise streams via /vsicurl/.

**HiRISE PDS SP1 bug — JP2 metadata side:** the same upstream HiRISE PDS bug that
poisons 4 of 10 BoulderNet `.prj` files (DECISIONS.md 2026-05-20) also poisons the
corresponding JP2s. The JP2 reports `Standard_Parallel_1=0` even though the pixel
coordinates were generated with the PDS-correct projection latitude. If we just trust
`rasterio.open(jp2).crs`, every downstream reprojection (Stage 2 mask, Stage 3 phase
correlation) mis-locates the HiRISE imagery on the CTX grid for those 4 images.

Fix: when a Stage 1 sidecar exists for the ObsId with a corrected CRS, override the
JP2's embedded CRS at read time. The affine *transform* (origin + pixel scale) is
correct — it was computed under the right projection — so only the CRS label needs
replacement. Caches built before this fix are detected by CRS mismatch and rebuilt.
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import truststore
from pyproj import CRS as _PyprojCRS
from rasterio.transform import Affine
from rasterio.windows import from_bounds

# Match the standard parallel under either name pyproj might emit:
#   ESRI WKT1:  PARAMETER["Standard_Parallel_1",20]
#   EPSG/WKT2:  PARAMETER["Latitude of 1st standard parallel",20,...]
# We accept either, case-insensitive. Used by `_crs_equal` because pyproj's `.equals()`
# canonicalizes spherical Equirectangular in a way that compares two SP1 values as
# equal even though they project longitudes to different metric x values. The literal
# parse is what catches the cache-staleness case for SP1-corrected ObsIds.
_SP1_LITERAL_PATTERN = re.compile(
    r'"(?:standard_parallel_1|Latitude of 1st standard parallel)"\s*,\s*(-?\d+\.?\d*)',
    re.IGNORECASE,
)

truststore.inject_into_ssl()

CACHE_SUBDIR = "hirise_decimated"
JP2_CACHE_SUBDIR = "hirise_jp2"
_STAGE1_CACHE_SUBDIR = "reprojected_detections"


def _corrected_source_crs(obs_id: str, cache_dir: str | Path) -> rasterio.crs.CRS | None:
    """Return the Stage 1 corrected HiRISE source CRS for `obs_id`, or None if Stage 1
    hasn't been run.

    For SP1-corrected ObsIds, this CRS replaces the buggy CRS embedded in the JP2.
    For trusted-prj ObsIds, it equals the JP2's embedded CRS — a no-op override that
    keeps the read path uniform across both regimes.
    """
    sidecar = Path(cache_dir) / _STAGE1_CACHE_SUBDIR / f"{obs_id}.json"
    if not sidecar.exists():
        return None
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    wkt = info.get("source_crs_wkt")
    if not wkt:
        return None
    return rasterio.crs.CRS.from_wkt(_PyprojCRS.from_user_input(wkt).to_wkt())


def _sp1_literal(crs) -> float | None:
    """Return the literal `Standard_Parallel_1` value in `crs`'s WKT, or None if absent."""
    if crs is None:
        return None
    wkt = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
    m = _SP1_LITERAL_PATTERN.search(wkt)
    return float(m.group(1)) if m else None


def _crs_equal(a, b) -> bool:
    """True iff two CRS objects represent the same projection AND agree on SP1.

    Compares via pyproj's `equals` (which canonicalizes WKT — robust to whitespace and
    parameter ordering) AND on the literal `Standard_Parallel_1` value parsed from each
    WKT. The literal check is required because pyproj normalizes spherical Equirectangular
    via EPSG method 1029 which drops SP1 (longitudes still scale by 1/cos(SP1) at apply
    time, but `equals()` doesn't see the difference). For our SP1-bug-affected images this
    means a buggy SP1=0 cached CRS would otherwise compare equal to the SP1=20 corrected
    CRS, and the staleness check would silently accept a stale cache.
    """
    if a is None or b is None:
        return False
    if not _PyprojCRS.from_user_input(a).equals(_PyprojCRS.from_user_input(b)):
        return False
    return _sp1_literal(a) == _sp1_literal(b)


def _vsicurl(url: str) -> str:
    return f"/vsicurl/{url}"


def _cache_path(cache_dir: str | Path, obs_id: str, suffix: str) -> Path:
    out = Path(cache_dir) / CACHE_SUBDIR / f"{obs_id}_{suffix}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _jp2_cache_path(cache_dir: str | Path, obs_id: str) -> Path:
    out = Path(cache_dir) / JP2_CACHE_SUBDIR / f"{obs_id}_RED.JP2"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def ensure_jp2_local(obs_id: str, jp2_url: str, cache_dir: str | Path) -> Path:
    """Ensure the full HiRISE JP2 is cached locally at `cache/hirise_jp2/{obs_id}_RED.JP2`.

    Downloads once via plain HTTP (single connection, no /vsicurl/ range overhead).
    Subsequent calls are no-ops. After this, all reads of the same image happen against
    a local file — fast crops, no network.

    Returns the local path.
    """
    out_path = _jp2_cache_path(cache_dir, obs_id)
    if out_path.exists() and out_path.stat().st_size > 1_000_000:  # > 1 MB sanity
        return out_path
    tmp = out_path.with_suffix(".JP2.partial")
    req = urllib.request.Request(jp2_url, headers={"User-Agent": "hirise2ctx/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)  # 1 MB chunks
    tmp.replace(out_path)
    return out_path


def _open_source(obs_id: str, jp2_url: str, cache_dir: str | Path):
    """Return a `rasterio.open` target preferring a local cached JP2 over /vsicurl/."""
    local = _jp2_cache_path(cache_dir, obs_id)
    if local.exists() and local.stat().st_size > 1_000_000:
        return rasterio.open(local)
    return rasterio.open(_vsicurl(jp2_url))


def read_full_footprint_decimated(
    obs_id: str,
    jp2_url: str,
    cache_dir: str | Path,
    target_mpp: float = 5.0,
) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
    """Return the entire HiRISE footprint as a 2D array decimated to `target_mpp` m/px.

    On first call, streams from `/vsicurl/{jp2_url}` (slow — typically tens of seconds)
    and writes a GeoTIFF cache. On subsequent calls, reads the cache (fast).

    The CRS embedded in the JP2 is overridden by the Stage 1 corrected CRS when a Stage 1
    sidecar exists for `obs_id` — this carries the SP1 fix through to JP2-derived rasters.
    A cached file whose embedded CRS disagrees with the corrected CRS is regenerated.

    Returns `(array, transform, crs)` where `array` is uint16 (HiRISE is 10-bit packed
    into 16-bit) and `transform` is in metres on the HiRISE source sphere.
    """
    cache = _cache_path(cache_dir, obs_id, f"{int(target_mpp)}mpp_full")
    corrected = _corrected_source_crs(obs_id, cache_dir)

    if cache.exists():
        with rasterio.open(cache) as ds:
            cached_crs = ds.crs
            if corrected is None or _crs_equal(cached_crs, corrected):
                return ds.read(1), ds.transform, cached_crs
        # Cache predates the JP2-side SP1 fix; rebuild below.

    with _open_source(obs_id, jp2_url, cache_dir) as ds:
        scale = target_mpp / float(ds.res[0])
        out_w = max(1, int(ds.width / scale))
        out_h = max(1, int(ds.height / scale))
        arr = ds.read(1, out_shape=(out_h, out_w))
        # Adjust transform to match the decimated grid
        x_scale = ds.transform.a * (ds.width / out_w)
        y_scale = ds.transform.e * (ds.height / out_h)
        new_transform = Affine(
            x_scale, ds.transform.b, ds.transform.c,
            ds.transform.d, y_scale, ds.transform.f,
        )
        crs = corrected if corrected is not None else ds.crs

    with rasterio.open(
        cache, "w",
        driver="GTiff",
        height=out_h, width=out_w, count=1, dtype=arr.dtype,
        crs=crs, transform=new_transform,
        compress="deflate", predictor=2,
    ) as out:
        out.write(arr, 1)
    return arr, new_transform, crs


def read_native_window(
    obs_id: str,
    jp2_url: str,
    bounds_src_crs: tuple[float, float, float, float],
    cache_dir: str | Path,
    window_tag: str,
) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
    """Read a window of the JP2 at native resolution.

    `bounds_src_crs` is `(left, bottom, right, top)` in the HiRISE source CRS's metres.
    `window_tag` is a short identifier used in the cache filename (e.g. "zoom_center").

    Caches the result as a GeoTIFF; reuses on subsequent calls. Applies the same Stage 1
    corrected-CRS override as `read_full_footprint_decimated` (see module docstring).
    """
    cache = _cache_path(cache_dir, obs_id, f"native_{window_tag}")
    corrected = _corrected_source_crs(obs_id, cache_dir)
    if cache.exists():
        with rasterio.open(cache) as ds:
            cached_crs = ds.crs
            if corrected is None or _crs_equal(cached_crs, corrected):
                return ds.read(1), ds.transform, cached_crs

    with _open_source(obs_id, jp2_url, cache_dir) as ds:
        window = from_bounds(*bounds_src_crs, transform=ds.transform)
        # Intersect with the dataset extent so out-of-range windows fail loudly with a
        # helpful message instead of writing a 0x0 GeoTIFF deep in rasterio's C layer.
        clipped = window.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        if clipped.width <= 0 or clipped.height <= 0:
            raise ValueError(
                f"requested window {bounds_src_crs} is entirely outside the JP2's spatial "
                f"extent for {obs_id}. JP2 bounds in its own CRS: "
                f"({ds.transform.c:.1f}, {ds.transform.f + ds.height * ds.transform.e:.1f}, "
                f"{ds.transform.c + ds.width * ds.transform.a:.1f}, {ds.transform.f:.1f}). "
                f"Likely cause: bounds were computed after a reprojection round-trip while "
                f"the JP2's own embedded CRS metadata is wrong (HiRISE PDS bug); read the "
                f"shapefile directly in its native CRS instead of reprojecting through CTX."
            )
        arr = ds.read(1, window=clipped)
        new_transform = rasterio.windows.transform(clipped, ds.transform)
        crs = corrected if corrected is not None else ds.crs

    with rasterio.open(
        cache, "w",
        driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
        crs=crs, transform=new_transform,
        compress="deflate", predictor=2,
    ) as out:
        out.write(arr, 1)
    return arr, new_transform, crs
