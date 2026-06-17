"""Fetch + cache + parse HiRISE PDS3 `.LBL` text labels.

The labels carry the authoritative HiRISE-side metadata we need for two purposes:

1. **Stage 1 .prj correction.** 4 of 10 BoulderNet shapefiles ship with `.prj` files that
   incorrectly set `Standard_Parallel_1=0` (datum `D_unnamed`) even though the geometry
   was generated with the PDS-declared projection latitude. The `.LBL`'s `CENTER_LATITUDE`
   is the truth; we override the bad `.prj` value with it at read time.
2. **Stage 2+ feature backfill.** Incidence and emission angles for the 8 non-diversity
   manifest rows are blank — they live in the `.LBL` (CLAUDE.md §11).

Labels are small (~10–20 KB each). The module caches the raw text under
`{cache_dir}/pds_labels/{ObsId}.LBL` and parses keywords on demand.

TLS is platform-specific (both symptoms are the same `CERTIFICATE_VERIFY_FAILED: unable to
get local issuer certificate`, but the fixes differ):
- **Windows + conda:** the stdlib CA bundle misses issuers; `truststore.inject_into_ssl()`
  delegates verification to the OS (Windows) trust store, like browsers do.
- **Linux/macOS (e.g. Sherlock):** the interpreter's OpenSSL may have no working default CA
  path, and truststore on Linux defers to that same broken store, so instead point urllib
  at `certifi`'s CA bundle via `SSL_CERT_FILE` (OpenSSL reads it at context creation).
"""
from __future__ import annotations

import os as _os
import re
import ssl as _ssl  # noqa: F401  (kept for future tweaks)
import urllib.request
from pathlib import Path
from typing import Any

# Configure TLS trust once at import (see module docstring).
if _os.environ.get("HIRISE2CTX_INSECURE_TLS") == "1":
    # OPT-IN escape hatch (off by default). Some hosts (e.g. murray-lab.caltech.edu, zenodo)
    # send an incomplete cert chain and OpenSSL on Linux won't AIA-fetch the missing
    # intermediate (Windows schannel does, which is why it works on the laptop but not on
    # Sherlock). For these PUBLIC, fixed-URL downloads only, skip verification; integrity is
    # still bounded by post-download size/zip-validity checks. Never enable for anything
    # sensitive. Applies process-wide to urllib downloads.
    import ssl as _ssl_insecure

    _ssl_insecure._create_default_https_context = _ssl_insecure._create_unverified_context
elif _os.name == "nt":
    import truststore

    truststore.inject_into_ssl()  # idempotent
else:
    try:
        import certifi as _certifi

        # setdefault: an explicit SSL_CERT_FILE in the environment still wins.
        _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    except Exception:  # pragma: no cover - certifi missing -> fall back to system default
        pass


CACHE_SUBDIR = "pds_labels"


def fetch_label(obs_id: str, label_url: str, cache_dir: str | Path) -> Path:
    """Download `label_url` to `<cache_dir>/pds_labels/{obs_id}.LBL` if not already cached.

    Returns the local path. Idempotent.
    """
    out_dir = Path(cache_dir) / CACHE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{obs_id}.LBL"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    req = urllib.request.Request(
        label_url,
        headers={"User-Agent": "hirise2ctx/0.1 (research; brianvamaro@gmail.com)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
    if len(body) < 200:
        raise RuntimeError(f"{obs_id}: PDS label fetch returned only {len(body)} bytes from {label_url}")
    out_path.write_bytes(body)
    return out_path


_KW_RE = re.compile(r"^\s*([A-Z][A-Z0-9_:]+)\s*=\s*(.+?)\s*$", re.MULTILINE)


def _strip_units(value: str) -> float:
    """Pull the leading numeric value from a PDS field like `21.6398 <DEG>` or `3393.83 <KM>`."""
    m = re.match(r"\s*([-+\d.eE]+)", value)
    if m is None:
        raise ValueError(f"could not parse numeric value from {value!r}")
    return float(m.group(1))


def read_label(obs_id: str, cache_dir: str | Path) -> dict[str, str]:
    """Return parsed PDS3 keywords as a flat string-to-string mapping (first occurrence wins).

    For nested objects (`OBJECT = IMAGE`, etc.) this only captures the top-level entry per
    keyword — sufficient for projection metadata, which lives near the file start. Re-parse
    the file directly if a deeper field is needed later.
    """
    path = Path(cache_dir) / CACHE_SUBDIR / f"{obs_id}.LBL"
    text = path.read_text(encoding="latin-1", errors="replace")
    out: dict[str, str] = {}
    for m in _KW_RE.finditer(text):
        k = m.group(1)
        v = m.group(2).strip().strip('"')
        out.setdefault(k, v)
    return out


def projection_origin(obs_id: str, cache_dir: str | Path) -> dict[str, float]:
    """Return the PDS-declared projection origin and sphere radius for `obs_id`.

    Keys: `center_lat_deg`, `center_lon_deg` (in [0, 360)), `a_axis_km`.

    Assumes the label has already been fetched (`fetch_label`).
    """
    kw = read_label(obs_id, cache_dir)
    return {
        "center_lat_deg": _strip_units(kw["CENTER_LATITUDE"]),
        "center_lon_deg": _strip_units(kw["CENTER_LONGITUDE"]),
        "a_axis_km": _strip_units(kw["A_AXIS_RADIUS"]),
    }


def image_footprint(obs_id: str, cache_dir: str | Path) -> dict[str, float]:
    """Return the PDS-declared image footprint extent in lat/lon degrees."""
    kw = read_label(obs_id, cache_dir)
    return {
        "max_lat_deg": _strip_units(kw["MAXIMUM_LATITUDE"]),
        "min_lat_deg": _strip_units(kw["MINIMUM_LATITUDE"]),
        "east_lon_deg": _strip_units(kw["EASTERNMOST_LONGITUDE"]),
        "west_lon_deg": _strip_units(kw["WESTERNMOST_LONGITUDE"]),
    }


def ensure_all_labels(manifest_df, cache_dir: str | Path) -> dict[str, Path]:
    """Pre-fetch labels for every manifest row. Returns ObsId -> cached path."""
    out: dict[str, Path] = {}
    for _, row in manifest_df.iterrows():
        out[row["ObsId"]] = fetch_label(row["ObsId"], row["LabelURL"], cache_dir)
    return out
