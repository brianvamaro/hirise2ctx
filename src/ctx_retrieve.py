"""Stage 0.5 — discover the Murray Lab CTX mosaic tile URL pattern and read one tile's
CRS WKT via a header-only `/vsicurl/` open.

Network usage is bounded: at most one HTML GET of the catalog page (to discover the URL
pattern) and one GeoTIFF header read (via HTTP Range, no pixel payload).
"""
from __future__ import annotations

import datetime as _dt
import re
import ssl
import urllib.parse
import urllib.request
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
