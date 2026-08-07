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

from . import net

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


_JP2_SIGNATURE = bytes.fromhex("0000000C6A5020200D0A870A")
# Codestream markers that stand alone -- no 2-byte length segment follows them.
_J2K_NO_SEGMENT = {0xFF4F, 0xFFD9, 0xFF93} | {0xFF30 + i for i in range(16)}


def inspect_jp2_integrity(path: str | Path) -> dict:
    """Is this JP2 a byte-complete JPEG2000 file? Pure bytes -- never raises.

    R66, mirroring `detections.inspect_shapefile_integrity`. Motivation: a truncated
    download is *not* detectable by opening the file. GDAL reports the full declared
    dimensions and `read()` returns a full-shape array with the missing region silently
    **zero-filled** -- which Stage 2 then converts into "no HiRISE coverage". So the file
    has to be asked about itself, at the byte level.

    Deliberately does NOT use rasterio/GDAL: opening a JP2 through GDAL can write an
    `.aux.xml` PAM sidecar next to it, and these files live in artifact roots.

    The trap: a JP2 box header may carry `Lbox == 0`, meaning "this box extends to EOF",
    and **all 46 PDS JP2s in this project use exactly that for their `jp2c` box**. So the
    box walk alone can never detect truncation. The real test is inside the codestream:
    walk the SOT tile-part chain via each `Psot` and require it to land on the `EOC`
    marker (`FFD9`) at exactly `size - 2`.

    Returns a dict whose `status` is one of:
      "complete"   -- the tile-part chain lands exactly on EOC at EOF
      "truncated"  -- it does not, including every mid-marker ran-off-the-end case
      "not_jp2"    -- no JP2 signature box (e.g. the GeoTIFF the isolation suite writes
                      under a `.JP2` name); a caller may still choose to accept it
      "unreadable" -- the file could not be read at all
    """
    path = Path(path)
    out: dict = {"status": "unreadable", "path": str(path)}
    try:
        size = path.stat().st_size
        out["actual_bytes"] = int(size)
        with open(path, "rb") as fh:
            head = fh.read(12)
            if head[:12] != _JP2_SIGNATURE:
                out["status"] = "not_jp2"
                out["note"] = "no JPEG2000 signature box"
                return out

            # --- walk top-level boxes to find the contiguous codestream box -------------
            pos, cs_start, cs_end = 12, None, None
            while pos + 8 <= size:
                fh.seek(pos)
                hdr = fh.read(8)
                if len(hdr) < 8:
                    break
                lbox = int.from_bytes(hdr[0:4], "big")
                tbox = hdr[4:8]
                body = pos + 8
                if lbox == 1:                      # 64-bit XLBox follows the type
                    xl = fh.read(8)
                    if len(xl) < 8:
                        out.update(status="truncated", note="ran off the end in an XLBox")
                        return out
                    lbox = int.from_bytes(xl, "big")
                    body = pos + 16
                if lbox == 0:                      # "extends to EOF" -- the PDS case
                    end = size
                elif lbox < 8:
                    out.update(status="truncated", note=f"nonsense box length {lbox}")
                    return out
                else:
                    end = pos + lbox
                if tbox == b"jp2c":
                    cs_start, cs_end = body, min(end, size)
                    break
                if end <= pos:
                    break
                pos = end
            if cs_start is None:
                out.update(status="truncated", note="no jp2c codestream box found")
                return out
            out["codestream_start"] = int(cs_start)

            # --- main header: SOC, SIZ, then marker segments up to the first SOT --------
            fh.seek(cs_start)
            if fh.read(2) != b"\xff\x4f":
                out.update(status="truncated", note="codestream does not start with SOC")
                return out
            if fh.read(2) != b"\xff\x51":
                out.update(status="truncated", note="SIZ does not follow SOC")
                return out
            lsiz = int.from_bytes(fh.read(2), "big")
            p = cs_start + 4 + lsiz
            while True:
                if p + 2 > size:
                    out.update(status="truncated",
                               note="ran off the end walking the main header")
                    return out
                fh.seek(p)
                marker = int.from_bytes(fh.read(2), "big")
                if marker == 0xFF90:               # SOT -- main header is done
                    break
                if marker >> 8 != 0xFF:
                    out.update(status="truncated",
                               note=f"not a marker at offset {p} (got {marker:#06x})")
                    return out
                if marker in _J2K_NO_SEGMENT:
                    p += 2
                    continue
                seg = fh.read(2)
                if len(seg) < 2:
                    out.update(status="truncated", note="ran off the end in a segment length")
                    return out
                p += 2 + int.from_bytes(seg, "big")

            # --- tile-part chain: each SOT's Psot must chain to the next, then EOC ------
            n_parts = 0
            while True:
                if p + 12 > size:
                    out.update(status="truncated", n_tile_parts=n_parts,
                               note="ran off the end inside an SOT segment")
                    return out
                fh.seek(p)
                if fh.read(2) != b"\xff\x90":
                    out.update(status="truncated", n_tile_parts=n_parts,
                               note=f"expected SOT at offset {p}")
                    return out
                fh.read(2)                          # Lsot (always 10)
                fh.read(2)                          # Isot
                psot = int.from_bytes(fh.read(4), "big")
                n_parts += 1
                if psot == 0:
                    # "runs to EOC" -- legal for the last tile-part.
                    p = size - 2
                    break
                p += psot
                if p > size:
                    out.update(status="truncated", n_tile_parts=n_parts,
                               note=f"tile-part {n_parts} claims {psot} bytes, past EOF")
                    return out
                if p + 2 <= size:
                    fh.seek(p)
                    if fh.read(2) == b"\xff\xd9":
                        break

            out["n_tile_parts"] = n_parts
            fh.seek(max(0, size - 2))
            if fh.read(2) != b"\xff\xd9":
                out.update(status="truncated", note="file does not end with the EOC marker")
                return out
            if p != size - 2:
                out.update(status="truncated",
                           note=f"tile-part chain ends at {p}, EOC is at {size - 2}")
                return out
            out["status"] = "complete"
    except OSError as exc:
        out["note"] = f"{type(exc).__name__}: {exc}"
    return out


def _reject_if_truncated(path: Path, *, context: str) -> None:
    """Raise if a cached JP2 is positively truncated. Lenient by design.

    Only a **positive** `"truncated"` verdict is fatal. `"not_jp2"` is let through: the
    artifact-isolation suite deliberately stages a GeoTIFF under a `{OBS}_RED.JP2` name,
    and more importantly a structurally unusual but legitimate JP2 must not be rejected by
    a walker that merely failed to parse it. Strict at commit time, lenient at reuse time
    -- the same split `detections.describe_null_geometry_drop` uses.
    """
    verdict = inspect_jp2_integrity(path)
    if verdict.get("status") == "truncated":
        raise RuntimeError(
            f"{context}: cached JP2 {path} is TRUNCATED "
            f"({verdict.get('note')}; {verdict.get('actual_bytes', -1):,} bytes on disk). "
            "GDAL would not complain -- it zero-fills the missing region, which Stage 2 "
            "reads as 'no HiRISE coverage'. Delete the file and re-run to re-download it. "
            "See DECISIONS 2026-08-06t."
        )


def ensure_jp2_local(obs_id: str, jp2_url: str, cache_dir: str | Path) -> Path:
    """Ensure the full HiRISE JP2 is cached locally at `cache/hirise_jp2/{obs_id}_RED.JP2`.

    Downloads once via plain HTTP (single connection, no /vsicurl/ range overhead).
    Subsequent calls are no-ops. After this, all reads of the same image happen against
    a local file — fast crops, no network.

    R66: the download is verified against `Content-Length` **and** structurally before it
    is published, and a already-cached file that is positively truncated raises rather
    than being silently preferred over `/vsicurl/` forever.

    Returns the local path.
    """
    out_path = _jp2_cache_path(cache_dir, obs_id)
    if out_path.exists() and out_path.stat().st_size > 1_000_000:  # > 1 MB sanity
        _reject_if_truncated(out_path, context=f"{obs_id}")
        return out_path
    tmp = out_path.with_suffix(".JP2.partial")
    req = urllib.request.Request(jp2_url, headers={"User-Agent": "hirise2ctx/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
            declared = net.content_length_of(resp)
            shutil.copyfileobj(resp, f, length=1 << 20)  # 1 MB chunks
        # R66: `copyfileobj` reads until EOF and a premature EOF is not an error, so
        # without this the partial file would be published and thereafter trusted.
        net.verify_download(
            tmp, url=jp2_url, declared_length=declared, min_bytes=1_000_000,
            validate=lambda p: (
                None if inspect_jp2_integrity(p)["status"] != "truncated"
                else f"incomplete JPEG2000 codestream ({inspect_jp2_integrity(p).get('note')})"
            ),
        )
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


def _open_source(obs_id: str, jp2_url: str, cache_dir: str | Path):
    """Return a `rasterio.open` target preferring a local cached JP2 over /vsicurl/."""
    local = _jp2_cache_path(cache_dir, obs_id)
    if local.exists() and local.stat().st_size > 1_000_000:
        _reject_if_truncated(local, context=f"{obs_id}")
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
