"""Manifest <-> Murray Lab CTX tile-name conversion + URL building.

The manifest uses the W/S-prefixed form (`W040_N20`, `E152_S08`, `E000_N40`); the Murray
Lab mosaic v01 publishes zipped GeoTIFFs whose filename uses a signed-int form,
`E{lon_int}_N{lat_int}` (e.g. `E-40_N20`, `E152_N-8`, `E160_N-20`). See `DECISIONS.md`
2026-05-20 Stage 0.5 entry for how the two conventions were discovered.

This module is intentionally tiny and dependency-free so the sign-handling can be tested
exhaustively without any network.
"""
from __future__ import annotations

import re

_MANIFEST_TILE_RE = re.compile(r"^([EW])(\d+)_([NS])(\d+)$")


def manifest_to_murray(tile_name: str) -> str:
    """Convert a manifest CTX tile name (`W040_N20`) to Murray Lab form (`E-40_N20`).

    Murray Lab prints both axes as signed Python ints without zero padding, so `W040`
    becomes `E-40`, `S08` becomes `N-8`, and `E000` becomes `E0`. The exact zero-padding
    convention for small/zero values is unverified by us at planning time — `DECISIONS.md`
    has the verified examples, and the caller (Stage 2 `ensure_tile_cached`) handles a
    404-then-retry-with-zero-padding fallback.
    """
    m = _MANIFEST_TILE_RE.fullmatch(tile_name)
    if m is None:
        raise ValueError(
            f"unrecognized manifest tile name {tile_name!r}; expected "
            "'<E|W><deg>_<N|S><deg>' (e.g. 'W040_N20', 'E152_S08')"
        )
    ew, lon_abs, ns, lat_abs = m.groups()
    lon = int(lon_abs) * (-1 if ew == "W" else 1)
    lat = int(lat_abs) * (-1 if ns == "S" else 1)
    return f"E{lon}_N{lat}"


def build_tile_url(url_template: str, murray_tile: str) -> str:
    """Substitute `{tile_name}` in `url_template` with `murray_tile`."""
    if "{tile_name}" not in url_template:
        raise ValueError(
            f"url_template missing required '{{tile_name}}' substitution token: "
            f"{url_template!r}"
        )
    return url_template.format(tile_name=murray_tile)


def murray_tile_for_manifest_row(row) -> str:
    """Convenience wrapper: pull `CTX_TileName` from a manifest row and translate it."""
    return manifest_to_murray(str(row["CTX_TileName"]))
