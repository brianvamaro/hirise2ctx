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
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import truststore
from rasterio.transform import Affine
from rasterio.windows import from_bounds

truststore.inject_into_ssl()

CACHE_SUBDIR = "hirise_decimated"
JP2_CACHE_SUBDIR = "hirise_jp2"


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

    Returns `(array, transform, crs)` where `array` is uint16 (HiRISE is 10-bit packed
    into 16-bit) and `transform` is in metres on the HiRISE source sphere.
    """
    cache = _cache_path(cache_dir, obs_id, f"{int(target_mpp)}mpp_full")
    if cache.exists():
        with rasterio.open(cache) as ds:
            return ds.read(1), ds.transform, ds.crs

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
        crs = ds.crs

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

    Caches the result as a GeoTIFF; reuses on subsequent calls.
    """
    cache = _cache_path(cache_dir, obs_id, f"native_{window_tag}")
    if cache.exists():
        with rasterio.open(cache) as ds:
            return ds.read(1), ds.transform, ds.crs

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
        crs = ds.crs

    with rasterio.open(
        cache, "w",
        driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
        crs=crs, transform=new_transform,
        compress="deflate", predictor=2,
    ) as out:
        out.write(arr, 1)
    return arr, new_transform, crs
