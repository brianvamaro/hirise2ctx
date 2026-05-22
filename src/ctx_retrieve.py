"""CTX mosaic retrieval.

- **Stage 0.5 helpers** (`resolve_target_crs`, `discover_murray_lab_url_template`,
  `read_ctx_tile_crs`): originally meant to runtime-probe the Murray Lab CTX mosaic
  CRS via a header-only `/vsicurl/` open. Superseded once `config.yaml::target_crs`
  was hardcoded to the canonical IAU-2000 Mars equirectangular WKT (see DECISIONS.md
  2026-05-20). Code kept for reference but not on the hot path.
- **Stage 2 helpers** (`ensure_tile_cached`, `compute_window_bounds`,
  `nominal_footprint_bounds`, `extract_ctx_window`, `stage2_one_image`):
  download-then-window mode per user choice. Each unique Murray Lab tile is fetched
  once into `cache/ctx_tiles/`, and per-ObsId CTX windows (polygon-bbox + buffer,
  snapped to the tile's native pixel grid) are written to `cache/ctx_windows/`.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def _ssl_context() -> ssl.SSLContext:
    """SSL context that finds CA certs even when urllib's defaults can't.

    On Windows + conda, the stdlib `urllib` doesn't read the system trust store, so
    plain `urlopen()` against any HTTPS site raises CERTIFICATE_VERIFY_FAILED. `certifi`
    is a transitive dep of `pyproj` in this env, so its CA bundle is available — use it.
    """
    try:
        import certifi  # transitive dep via pyproj
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

URL_TEMPLATE_CACHE = "ctx_url_template.txt"
CRS_CACHE = "ctx_crs.wkt"

# Murray Lab tile names look like E000_N40, W040_N20, E152_S08, etc.
# Two captured groups so we can rebuild the filename with `{tile_name}` substitution.
_TILE_RE = re.compile(r"\b(?P<tile>(?:E|W)\d{3}_(?:N|S)\d{2})(?P<ext>\.(?:tif|TIF|tiff|TIFF))\b")
_HREF_RE = re.compile(r"""href\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE)


def discover_murray_lab_url_template(catalog_url: str, cache_dir: str | Path) -> str:
    """Return a URL template containing `{tile_name}` as a substitution token.

    Caches the resolved template in `<cache_dir>/ctx_url_template.txt`. Re-running is a
    no-op if the cache exists.

    The catalog page is fetched exactly once per cache lifetime. We scan for any anchor
    href that contains a tile-shaped filename (e.g. `E000_N40.tif`); the first such match
    is generalized by replacing the literal tile substring with `{tile_name}`.
    """
    cache_path = Path(cache_dir) / URL_TEMPLATE_CACHE
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()

    req = urllib.request.Request(
        catalog_url,
        headers={"User-Agent": "hirise2ctx/0.1 (research; brianvamaro@gmail.com)"},
    )
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    candidates: list[str] = []
    for href in _HREF_RE.findall(html):
        m = _TILE_RE.search(href)
        if m:
            absolute = urllib.parse.urljoin(catalog_url, href)
            template = absolute.replace(m.group("tile"), "{tile_name}")
            candidates.append(template)
    # Fall back: search the raw HTML body for tile-shaped substrings even without an
    # explicit href attribute (some autoindex pages wrap them in plain text).
    if not candidates:
        for m in _TILE_RE.finditer(html):
            template = urllib.parse.urljoin(catalog_url, m.group("tile") + m.group("ext"))
            template = template.replace(m.group("tile"), "{tile_name}")
            candidates.append(template)

    if not candidates:
        raise RuntimeError(
            f"could not find any Murray Lab tile filename in {catalog_url!r}. "
            f"Set `ctx_mosaic.url_template` explicitly in config.yaml as the manual override "
            f"(e.g. 'https://.../{{tile_name}}.tif')."
        )

    # Pick the shortest unique candidate — long auto-generated links usually contain
    # query strings or thumbnails; the canonical tile URL is the short one.
    template = sorted(set(candidates), key=len)[0]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(template, encoding="utf-8")
    log_path = Path(cache_dir) / "ctx_discovery.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{_dt.datetime.now(_dt.timezone.utc).isoformat()}\t"
            f"catalog={catalog_url}\tn_candidates={len(set(candidates))}\ttemplate={template}\n"
        )
    return template


def build_tile_url(template: str, tile_name: str) -> str:
    if "{tile_name}" not in template:
        raise ValueError(f"url template missing '{{tile_name}}' token: {template!r}")
    return template.format(tile_name=tile_name)


def read_ctx_tile_crs(tile_name: str, url_template: str, cache_dir: str | Path) -> str:
    """Open a single Murray Lab CTX tile via `/vsicurl/` and return its CRS WKT.

    Reads only the GeoTIFF header (via HTTP Range requests under the hood — no raster
    pixels are transferred). Caches the WKT to `<cache_dir>/ctx_crs.wkt`.
    """
    cache_path = Path(cache_dir) / CRS_CACHE
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()

    import rasterio  # imported lazily; not needed for unit tests

    url = build_tile_url(url_template, tile_name)
    vsicurl_url = f"/vsicurl/{url}"
    with rasterio.open(vsicurl_url) as src:
        if src.crs is None:
            raise RuntimeError(f"tile has no CRS: {url}")
        wkt = src.crs.to_wkt()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(wkt, encoding="utf-8")
    log_path = Path(cache_dir) / "ctx_discovery.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{_dt.datetime.now(_dt.timezone.utc).isoformat()}\t"
            f"crs_probe tile={tile_name}\turl={url}\twkt_chars={len(wkt)}\n"
        )
    return wkt


def resolve_target_crs(cfg) -> str:
    """Return the target CRS WKT, honoring the `from_ctx_tile` sentinel.

    - If `cfg['target_crs']` is the string `'from_ctx_tile'`, run Stage 0.5: discover the
      URL template (cached) and probe the configured `probe_tile`'s CRS (cached).
    - Otherwise return `cfg['target_crs']` as-is (already a WKT or a CRS string).
    """
    target = cfg["target_crs"]
    if target != "from_ctx_tile":
        return target
    mosaic = cfg["ctx_mosaic"]
    cache_dir = cfg.cache_dir if hasattr(cfg, "cache_dir") else Path(cfg["cache_dir"])
    template = mosaic.get("url_template") or discover_murray_lab_url_template(
        mosaic["catalog_url"], cache_dir
    )
    return read_ctx_tile_crs(mosaic["probe_tile"], template, cache_dir)


# =====================================================================================
# Stage 2 — download-then-window CTX retrieval
# =====================================================================================

CTX_TILES_SUBDIR = "ctx_tiles"
CTX_WINDOWS_SUBDIR = "ctx_windows"
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB
_TILE_MIN_BYTES = 50 * 1024 * 1024  # 50 MB sanity floor (real tiles are ~1-2 GB)


def _download_to(url: str, dest_path: Path, *, on_progress=None, timeout: float = 60.0) -> Path:
    """Stream `url` to `dest_path.tmp` then atomically rename to `dest_path`.

    `on_progress` (if given) is called with `(bytes_so_far, total_bytes_or_zero)` after each
    chunk write. The temp-then-rename pattern means an interrupted download leaves a `.tmp`
    file that a re-run can safely overwrite, but never a half-written `dest_path`.

    Uses `truststore` (imported by `src.pds_labels`) for OS-level SSL trust, which is what
    fixed the Caltech HTTPS issue we hit during Stage 0.5.
    """
    # Trigger truststore.inject_into_ssl() side-effect from pds_labels (idempotent)
    from . import pds_labels  # noqa: F401

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "hirise2ctx/0.1 (research; brianvamaro@gmail.com)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        try:
            total = int(resp.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            total = 0
        downloaded = 0
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(downloaded, total)
    if tmp.stat().st_size < _TILE_MIN_BYTES:
        size = tmp.stat().st_size
        tmp.unlink()
        raise RuntimeError(
            f"download of {url} returned only {size} bytes (< {_TILE_MIN_BYTES}); "
            "treating as malformed and refusing to commit to cache."
        )
    tmp.replace(dest_path)
    return dest_path


def _padded_manifest_form(murray_tile: str) -> str | None:
    """If `murray_tile` differs from a zero-padded manifest-style form, return the padded
    form (`E0_N40` -> `E000_N40`); else return None.

    Heuristic fallback used only if Murray Lab's published filename for a small/zero
    coordinate uses the manifest-style padding instead of the bare signed-int form we
    saw on `E160_N-20`. The retriever tries the bare form first and falls back to this.
    """
    m = re.fullmatch(r"E(-?\d+)_N(-?\d+)", murray_tile)
    if m is None:
        return None
    lon_i = int(m.group(1))
    lat_i = int(m.group(2))
    lon_padded = f"E{abs(lon_i):03d}" if lon_i >= 0 else f"W{abs(lon_i):03d}"
    lat_padded = f"N{abs(lat_i):02d}" if lat_i >= 0 else f"S{abs(lat_i):02d}"
    return f"{lon_padded}_{lat_padded}"


def ensure_tile_cached(
    murray_tile: str,
    *,
    url_template: str,
    cache_dir: str | Path,
    on_progress=None,
) -> tuple[Path, str]:
    """Ensure `cache/ctx_tiles/{murray_tile}.zip` is present; return (zip_path, inner_tif_name).

    On first cache miss, fetches `url_template.format(tile_name=murray_tile)`. If that
    URL returns 404 (Murray Lab uses a zero-padded variant for this tile), retries once
    with the manifest-style padded form and re-records the actual filename in the sidecar.

    Caches a JSON sidecar with the source URL, download time, inner-tif name, the tile's
    CRS WKT, affine transform, shape, and dtype — so subsequent stages don't have to
    re-open the 1-2 GB zip just to read the header.
    """
    from . import ctx_tiles
    import rasterio

    cache_dir = Path(cache_dir)
    tiles_dir = cache_dir / CTX_TILES_SUBDIR
    tiles_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tiles_dir / f"{murray_tile}.zip"
    sidecar = tiles_dir / f"{murray_tile}.json"

    if not zip_path.exists():
        url = ctx_tiles.build_tile_url(url_template, murray_tile)
        used_url = url
        used_tile_name = murray_tile
        try:
            _download_to(url, zip_path, on_progress=on_progress)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            padded = _padded_manifest_form(murray_tile)
            if padded is None or padded == murray_tile:
                raise
            alt_url = ctx_tiles.build_tile_url(url_template, padded)
            _download_to(alt_url, zip_path, on_progress=on_progress)
            used_url = alt_url
            used_tile_name = padded

    if not sidecar.exists():
        with zipfile.ZipFile(zip_path) as zf:
            tif_members = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if len(tif_members) != 1:
            raise RuntimeError(
                f"{zip_path}: expected exactly one .tif inside, found {tif_members!r}. "
                "Add explicit member selection if Murray Lab changed the zip layout."
            )
        inner_tif = tif_members[0]
        vsizip = f"/vsizip/{zip_path.as_posix()}/{inner_tif}"
        with rasterio.open(vsizip) as src:
            inner_crs_wkt = src.crs.to_wkt() if src.crs else None
            inner_transform = list(src.transform)[:6]
            inner_shape = [int(src.height), int(src.width)]
            inner_dtype = str(src.dtypes[0])

        # Best-effort retrieval of which URL we actually used (may not be in scope if
        # the zip already existed from a previous run — leave None then).
        try:
            recorded_url: str | None = used_url  # type: ignore[name-defined]
            recorded_tile: str = used_tile_name  # type: ignore[name-defined]
        except NameError:
            recorded_url = None
            recorded_tile = murray_tile

        sidecar.write_text(
            json.dumps(
                {
                    "murray_tile": murray_tile,
                    "resolved_tile_name": recorded_tile,
                    "source_url": recorded_url,
                    "downloaded_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "zip_bytes": zip_path.stat().st_size,
                    "inner_tif": inner_tif,
                    "inner_crs_wkt": inner_crs_wkt,
                    "inner_transform": inner_transform,
                    "inner_shape": inner_shape,
                    "inner_dtype": inner_dtype,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    info = json.loads(sidecar.read_text(encoding="utf-8"))
    return zip_path, info["inner_tif"]


def _snap_bounds_to_pixel_grid(
    bounds: tuple[float, float, float, float],
    transform,
) -> tuple[float, float, float, float]:
    """Snap `(xmin, ymin, xmax, ymax)` outward to integer pixel offsets of `transform`.

    `transform` may be a `rasterio.Affine` or a 6-tuple `(a, b, c, d, e, f)`. Standard
    north-up rasters have `a > 0, e < 0`, and the tile origin (upper-left) is `(c, f)`.
    Snapping outward means the resulting bbox always *contains* the input bbox.
    """
    a, b, c, d, e, f = (transform[i] for i in range(6))
    px_x = abs(a)
    px_y = abs(e)
    xmin, ymin, xmax, ymax = bounds
    xmin_snapped = c + math.floor((xmin - c) / px_x) * px_x
    xmax_snapped = c + math.ceil((xmax - c) / px_x) * px_x
    ymax_snapped = f - math.floor((f - ymax) / px_y) * px_y
    ymin_snapped = f - math.ceil((f - ymin) / px_y) * px_y
    return (xmin_snapped, ymin_snapped, xmax_snapped, ymax_snapped)


def compute_window_bounds(gdf, buffer_m: float, ctx_transform) -> tuple[float, float, float, float]:
    """Return the polygon-footprint bbox + `buffer_m`, snapped to `ctx_transform`'s pixel grid.

    Raises `ValueError` if `gdf` is empty (use `nominal_footprint_bounds` instead).
    """
    if len(gdf) == 0:
        raise ValueError(
            "compute_window_bounds: gdf is empty; "
            "use nominal_footprint_bounds for shapefiles with zero detections."
        )
    xmin, ymin, xmax, ymax = (float(v) for v in gdf.total_bounds)
    expanded = (xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)
    return _snap_bounds_to_pixel_grid(expanded, ctx_transform)


def nominal_footprint_bounds(
    manifest_row,
    target_crs: str,
    width_m: float,
    length_m: float,
    ctx_transform,
) -> tuple[float, float, float, float]:
    """Center-on-manifest fallback for empty shapefiles.

    Projects `manifest_row.CenterLat / CenterLon_180` from the target CRS's geodetic base
    into `target_crs` (both share the IAU-2000 Mars sphere), builds a rectangle of size
    `width_m x length_m` around it, and snaps to the CTX pixel grid. The `width_m` axis
    is east-west; `length_m` is north-south, matching the typical HiRISE swath geometry.
    """
    import pyproj

    target = pyproj.CRS.from_user_input(target_crs)
    geographic = target.geodetic_crs
    transformer = pyproj.Transformer.from_crs(geographic, target, always_xy=True)
    lon = float(manifest_row["CenterLon_180"])
    lat = float(manifest_row["CenterLat"])
    cx, cy = transformer.transform(lon, lat)
    half_w = width_m / 2.0
    half_l = length_m / 2.0
    bounds = (cx - half_w, cy - half_l, cx + half_w, cy + half_l)
    return _snap_bounds_to_pixel_grid(bounds, ctx_transform)


def extract_ctx_window(
    zip_path: str | Path,
    inner_tif: str,
    bounds: tuple[float, float, float, float],
    out_path: str | Path,
) -> Path:
    """Window-read `/vsizip/{zip_path}/{inner_tif}` to `bounds` and write a small GeoTIFF.

    Output CRS + per-pixel size + dtype are preserved from the source; only the affine
    origin and shape change. The output is integer-pixel-aligned because `bounds` should
    already have been snapped via `_snap_bounds_to_pixel_grid` — this function additionally
    rounds the rasterio window object as a defense in depth.
    """
    import rasterio
    from rasterio.windows import from_bounds

    zip_path = Path(zip_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vsizip = f"/vsizip/{zip_path.as_posix()}/{inner_tif}"
    with rasterio.open(vsizip) as src:
        window = from_bounds(*bounds, transform=src.transform)
        # Round window to integer offsets + integer pixel count. We don't use
        # `Window.round_shape()` (deprecated in rasterio 2.0) — manual via int().
        col_off = int(round(window.col_off))
        row_off = int(round(window.row_off))
        width = int(round(window.width))
        height = int(round(window.height))
        window = window.__class__(col_off=col_off, row_off=row_off, width=width, height=height)
        data = src.read(window=window)
        new_transform = src.window_transform(window)
        profile = src.profile.copy()
        # The source CTX mosaic tiles are stored as a single ~47420-px-wide internal TIFF
        # block; copying that block size into our small (~1500 px) output is invalid
        # ("TileWidth/TileHeight must be multiples of 16 and not larger than the image").
        # Drop the source block geometry and let rasterio pick reasonable defaults.
        for k in ("blockxsize", "blockysize", "tiled"):
            profile.pop(k, None)
        profile.update(
            {
                "driver": "GTiff",
                "height": int(data.shape[1]),
                "width": int(data.shape[2]),
                "transform": new_transform,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
    return out_path


def build_hirise_coverage_mask(
    obs_id: str,
    *,
    jp2_url: str,
    cache_dir: str | Path,
    ctx_window_tif: str | Path,
    out_path: str | Path,
) -> tuple[Path, float]:
    """Write a uint8 HiRISE-coverage mask aligned to the CTX window grid.

    The mask is 1 where the decimated HiRISE (5 m/px) has a valid (non-zero) pixel after
    reprojection onto the CTX window's CRS + transform + shape, and 0 elsewhere. Used by
    Stage 4 to suppress label generation outside the HiRISE swath AND inside the swath
    where HiRISE itself has NaN/0 pixels (rotated-rectangle corners, missing scans).

    `nearest` resampling preserves the binary 0-vs-valid distinction at swath edges
    (bilinear would interpolate between 0 and valid, producing spurious partial-coverage
    pixels along the boundary).

    Triggers a JP2 download via `hirise_imagery.ensure_jp2_local` if not already cached —
    a one-time ~200-500 MB hit per ObsId. Subsequent calls reuse the local JP2 and the
    cached decimated GeoTIFF in `cache/hirise_decimated/{ObsId}_5mpp_full.tif`.

    Returns `(mask_path, coverage_fraction)` where `coverage_fraction` is the share of
    CTX-window pixels with HiRISE coverage.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from . import hirise_imagery

    cache_dir = Path(cache_dir)
    ctx_window_tif = Path(ctx_window_tif)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hirise_imagery.ensure_jp2_local(obs_id, jp2_url, cache_dir)
    hi_arr, hi_transform, hi_crs = hirise_imagery.read_full_footprint_decimated(
        obs_id, jp2_url, cache_dir, target_mpp=5.0,
    )

    with rasterio.open(ctx_window_tif) as ctx_src:
        ctx_transform = ctx_src.transform
        ctx_crs = ctx_src.crs
        ctx_shape = (ctx_src.height, ctx_src.width)

    valid_src = (hi_arr > 0).astype(np.uint8)
    mask = np.zeros(ctx_shape, dtype=np.uint8)
    reproject(
        source=valid_src,
        destination=mask,
        src_transform=hi_transform,
        src_crs=hi_crs,
        dst_transform=ctx_transform,
        dst_crs=ctx_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=ctx_shape[0], width=ctx_shape[1], count=1,
        dtype="uint8", crs=ctx_crs, transform=ctx_transform,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256, nodata=0,
    ) as dst:
        dst.write(mask, 1)

    coverage_fraction = float(mask.mean())
    return out_path, coverage_fraction


def stage2_one_image(
    obs_id: str,
    *,
    cache_dir: str | Path,
    manifest_row,
    target_crs: str,
    url_template: str,
    buffer_m: float,
    nominal_width_m: float,
    nominal_length_m: float,
    config_hash: str,
    on_progress=None,
) -> dict:
    """End-to-end Stage 2 for one ObsId. Mirrors `detections.stage1_one_image`.

    Reads the cached Stage-1 GeoDataFrame, picks polygon-bbox or nominal-fallback bounds,
    ensures the matching Murray Lab tile zip is cached, extracts a windowed GeoTIFF to
    `cache/ctx_windows/{obs_id}.tif`, builds the HiRISE coverage mask
    `{obs_id}_hirise_mask.tif`, and writes a provenance JSON sidecar. Returns the
    provenance dict.
    """
    from . import ctx_tiles
    from . import detections as det

    cache_dir = Path(cache_dir)
    gdf = det.load_reprojected(obs_id, cache_dir)
    murray_tile = ctx_tiles.murray_tile_for_manifest_row(manifest_row)

    zip_path, inner_tif = ensure_tile_cached(
        murray_tile,
        url_template=url_template,
        cache_dir=cache_dir,
        on_progress=on_progress,
    )
    tile_sidecar = json.loads(
        (cache_dir / CTX_TILES_SUBDIR / f"{murray_tile}.json").read_text(encoding="utf-8")
    )
    tile_transform = tile_sidecar["inner_transform"]

    if len(gdf) > 0:
        bounds = compute_window_bounds(gdf, buffer_m, tile_transform)
        footprint_source = "polygon_bbox"
    else:
        bounds = nominal_footprint_bounds(
            manifest_row, target_crs, nominal_width_m, nominal_length_m, tile_transform
        )
        footprint_source = "nominal_from_manifest"

    out_dir = cache_dir / CTX_WINDOWS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{obs_id}.tif"
    extract_ctx_window(zip_path, inner_tif, bounds, out_tif)

    # Read back the actually-written window so the sidecar records ground truth
    # (rasterio's window rounding may have nudged things by ≤ 1 pixel).
    import rasterio
    with rasterio.open(out_tif) as src:
        actual_transform = list(src.transform)[:6]
        actual_shape = [int(src.height), int(src.width)]
        actual_bounds = list(src.bounds)

    # HiRISE coverage mask, aligned to the CTX window. Stage 4 must consume this to
    # avoid emitting "boulder absence" labels outside the HiRISE swath (or on the
    # NaN/0 corners inside the swath's rotated-rectangle outline).
    mask_path = out_dir / f"{obs_id}_hirise_mask.tif"
    _, coverage_fraction = build_hirise_coverage_mask(
        obs_id,
        jp2_url=str(manifest_row["JP2_URL"]),
        cache_dir=cache_dir,
        ctx_window_tif=out_tif,
        out_path=mask_path,
    )

    provenance = {
        "obs_id": obs_id,
        "source_murray_tile": murray_tile,
        "source_zip": str(zip_path),
        "source_inner_tif": inner_tif,
        "requested_bounds_target_crs": list(bounds),
        "actual_bounds_target_crs": actual_bounds,
        "actual_transform": actual_transform,
        "actual_shape": actual_shape,
        "buffer_m": float(buffer_m),
        "footprint_source": footprint_source,
        "n_polygons_anchor": int(len(gdf)),
        "hirise_mask_path": str(mask_path),
        "hirise_coverage_fraction": coverage_fraction,
        "config_hash": config_hash,
        "extracted_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (out_dir / f"{obs_id}.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance
