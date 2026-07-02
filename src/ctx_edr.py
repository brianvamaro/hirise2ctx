"""CTX EDR resolution — SeamMap (VOLUME_ID, PRODUCT_ID) -> live PDS archive URL.

The F mitigation (per-source-frame inference, PLAN_StripingArtifact) needs the raw CTX EDRs
beneath the Murray mosaic. The SeamMap's cached ``PDS_IMG`` URLs are stale (all 404: the PDS
Imaging Node renamed the path segment ``mars_reconnaissance_orbiter/ctx`` -> ``mro/ctx`` after
the 2024 mosaic release). The frames never moved: the SeamMap's own ``VOLUME_ID`` + ``PRODUCT_ID``
fields fully determine the current URL — no resolver library needed. Verified 12/12 across
mission-spanning volumes (mrox_0009..mrox_3355), DECISIONS 2026-07-02; re-runnable check =
``scripts/probes/_f_edr_url_verify.py``. The ODE REST API returns the identical URL and is the
documented fallback if JPL reorganizes again.
"""
from __future__ import annotations

from pathlib import Path

import rasterio
from shapely.geometry import box

from src.striping import MAP_DIR, load_frames

EDR_URL_TEMPLATE = "https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{volume}/data/{product_id}.IMG"
#: fallback resolver (returns the same Product URL under "Product_files"):
ODE_REST_TEMPLATE = ("https://oderest.rsl.wustl.edu/live2/?query=product&results=fmp&output=JSON"
                     "&pt=EDR&iid=CTX&ihid=MRO&productid={product_id}")
COARSE_FACTOR = 32  # abundance grid is 32x coarser than the native 5 m/px CTX grid


def edr_url(volume_id: str, product_id: str) -> str:
    """Live PDS archive URL for a CTX EDR, from the two SeamMap fields that define it."""
    return EDR_URL_TEMPLATE.format(volume=volume_id.lower(), product_id=product_id)


def frame_table(tile: str):
    """Unique source frames for ``tile`` with their EDR URLs (one row per PRODUCT_ID).

    Columns: PRODUCT_ID, VOLUME_ID, EMISSION, INCIDENCE, IMAGE_TIME, edr_url, geometry.
    """
    g = load_frames(tile)
    keep = [c for c in ("PRODUCT_ID", "VOLUME_ID", "EMISSION", "INCIDENCE", "IMAGE_TIME") if c in g.columns]
    g = g[keep + ["geometry"]].copy()
    g["edr_url"] = [edr_url(v, p) for v, p in zip(g["VOLUME_ID"], g["PRODUCT_ID"])]
    return g


def frames_in_crop(tile: str, r0: int, c0: int, size: int, min_frac: float = 0.01):
    """Frames intersecting a native-pixel crop of ``tile``, largest overlap first.

    ``r0/c0/size`` are native 5 m/px coordinates (the convention of the striping crop scripts).
    Frames covering less than ``min_frac`` of the crop are dropped (slivers). Adds an
    ``overlap_frac`` column.
    """
    with rasterio.open(MAP_DIR / f"{tile}_abundance.tif") as ds:
        t = ds.transform
    cf = COARSE_FACTOR
    x0, y0 = t * (c0 / cf, r0 / cf)
    x1, y1 = t * ((c0 + size) / cf, (r0 + size) / cf)
    crop = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    g = frame_table(tile)
    inter = g.geometry.intersection(crop).area / crop.area
    g = g.assign(overlap_frac=inter)
    g = g[g["overlap_frac"] >= min_frac].sort_values("overlap_frac", ascending=False)
    return g.reset_index(drop=True)
