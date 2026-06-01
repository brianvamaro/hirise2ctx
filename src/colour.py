"""HiRISE colour (BG/RED/IR) data access for Stage 7 compositional analysis.

The PDS RDR product `COLOR.JP2` is a single 3-band JP2 with bands in this order:

  Band 1 = NEAR-INFRARED (~900 nm)
  Band 2 = RED (~700 nm)
  Band 3 = BLUE-GREEN (~500 nm)

at 0.25 m/px in I/F (intensity / flux ratio) units, covering the central ~1-3 km swath
of the HiRISE observation (RED is ~6 km wide). The PLAN_Compositional.md §2.1 sentence
about separate IRB+RGB PDS products is incorrect — runtime PDS inspection (2026-05-31)
finds a single COLOR.JP2 alongside the panchromatic RED.JP2.

Lambertian photometric correction (first-order):
  I/F_corrected = I/F_observed / cos(incidence_angle)
The COLOR.JP2 is uncorrected I/F. cos(i) is constant per image since the colour swath is
small enough (~3 km) that the incidence-angle gradient is negligible at this stage.

Native DN -> I/F mapping (when reading raw DN instead of letting rasterio scale):
  I/F = DN * SCALING_FACTOR + OFFSET    (both in COLOR.LBL)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

CACHE_SUBDIR = "hirise_color"
_STAGE1_CACHE_SUBDIR = "reprojected_detections"

BAND_IR = 1
BAND_RED = 2
BAND_BG = 3
BAND_NAMES = ("IR", "RED", "BG")


@dataclass(frozen=True)
class ColorLBL:
    """Subset of COLOR.LBL metadata used in Stage 7."""

    obs_id: str
    incidence_deg: float
    emission_deg: float
    phase_deg: float
    solar_longitude_deg: float
    north_azimuth_deg: float
    scaling_factor: float
    offset: float
    map_scale_mpp: float
    lines: int
    line_samples: int
    bands: int

    @property
    def cos_incidence(self) -> float:
        return float(np.cos(np.deg2rad(self.incidence_deg)))


def color_jp2_path(cache_dir, obs_id: str) -> Path:
    return Path(cache_dir) / CACHE_SUBDIR / f"{obs_id}_COLOR.JP2"


def color_lbl_path(cache_dir, obs_id: str) -> Path:
    return Path(cache_dir) / CACHE_SUBDIR / f"{obs_id}_COLOR.LBL"


def corrected_source_crs(obs_id: str, cache_dir) -> rasterio.crs.CRS | None:
    """Return the SP1-corrected HiRISE source CRS for `obs_id`, loaded from the Stage 1
    sidecar. Both the COLOR.JP2 and the original detection shapefile suffer the same
    upstream SP1 bug (CRS metadata reports SP1=0 but pixel coords are under SP1=lat).
    The corrected CRS in the sidecar is the truth for both.
    Returns None if Stage 1 hasn't been run for this image.
    """
    sidecar = Path(cache_dir) / _STAGE1_CACHE_SUBDIR / f"{obs_id}.json"
    if not sidecar.exists():
        return None
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    wkt = info.get("source_crs_wkt")
    if not wkt:
        return None
    return rasterio.crs.CRS.from_wkt(wkt)


_KEY_RE_TMPL = r"^\s*{key}\s*=\s*([^\n]+)$"


def parse_color_lbl(lbl_path) -> ColorLBL:
    """Parse a HiRISE COLOR.LBL into the metadata we need for photometric correction."""
    text = Path(lbl_path).read_text(encoding="latin-1", errors="replace")

    def _scalar(key: str) -> str | None:
        m = re.search(_KEY_RE_TMPL.format(key=re.escape(key)), text, re.MULTILINE)
        return m.group(1).strip() if m else None

    def _num(key: str) -> float:
        raw = _scalar(key)
        if raw is None:
            raise KeyError(f"{key!r} missing in {lbl_path}")
        return float(re.split(r"\s+", raw)[0])

    return ColorLBL(
        obs_id=Path(lbl_path).stem.replace("_COLOR", ""),
        incidence_deg=_num("INCIDENCE_ANGLE"),
        emission_deg=_num("EMISSION_ANGLE"),
        phase_deg=_num("PHASE_ANGLE"),
        solar_longitude_deg=_num("SOLAR_LONGITUDE"),
        north_azimuth_deg=_num("NORTH_AZIMUTH"),
        scaling_factor=_num("SCALING_FACTOR"),
        offset=_num("OFFSET"),
        map_scale_mpp=_num("MAP_SCALE"),
        lines=int(_num("LINES")),
        line_samples=int(_num("LINE_SAMPLES")),
        bands=int(_num("BANDS")),
    )


def lambertian_correct(iof: np.ndarray, incidence_deg: float) -> np.ndarray:
    """Lambertian correction `I/F / cos(i)`. First-order — assumes uniform incidence
    across the ~3 km colour swath, which is fine at this stage."""
    cos_i = float(np.cos(np.deg2rad(incidence_deg)))
    if cos_i <= 0:
        raise ValueError(f"non-illuminated geometry: incidence={incidence_deg} deg")
    return iof / cos_i


def read_color_window(ds, bounds_jp2_crs):
    """Read a 3-band window from an already-open COLOR.JP2 rasterio dataset.

    `ds` is an open `rasterio.DatasetReader` for the COLOR.JP2. Keep it open across
    many polygon/tile reads -- opening a JP2 is expensive (codec init) and would
    otherwise dominate the runtime.

    `bounds_jp2_crs` = (left, bottom, right, top) in the SOURCE CRS the dataset's
    pixel-coordinate transform was actually computed under (the SP1-corrected CRS for
    HiRISE images affected by the upstream PDS SP1 bug). `ds.crs` itself is buggy in
    those cases; do not pass bounds that have been transformed using `ds.crs`.

    Returns `(arr, transform)` with `arr` shape (3, H, W) or `(None, None)` if the
    window is entirely outside the JP2 extent.
    """
    win = from_bounds(*bounds_jp2_crs, transform=ds.transform)
    clipped = win.intersection(Window(0, 0, ds.width, ds.height))
    if clipped.width < 1 or clipped.height < 1:
        return None, None
    arr = ds.read(window=clipped)
    win_transform = rasterio.windows.transform(clipped, ds.transform)
    return arr, win_transform


def color_jp2_crs(jp2_path) -> rasterio.crs.CRS:
    """Return the CRS embedded in the COLOR.JP2."""
    with rasterio.open(jp2_path) as ds:
        return ds.crs


def color_jp2_bounds(jp2_path) -> tuple[float, float, float, float]:
    """Return (left, bottom, right, top) of the COLOR.JP2 in its own CRS."""
    with rasterio.open(jp2_path) as ds:
        return tuple(ds.bounds)


def region_means(
    arr: np.ndarray,
    mask: np.ndarray,
    *,
    valid_min: float = 1e-9,
    min_pixels: int = 8,
) -> dict[str, float | int] | None:
    """Compute per-band mean over the pixels selected by `mask` (True = keep).

    `arr` shape is (3, H, W) in band order IR, RED, BG. Pixels exactly == 0 in ANY band
    are treated as nodata (HiRISE COLOR pads off-swath area with 0). Returns None if
    fewer than `min_pixels` valid pixels remain after masking.
    """
    if arr is None or mask is None or not mask.any():
        return None
    # Off-swath / nodata: 0 in all 3 bands typically. Use a strict per-band > valid_min.
    valid = mask & (arr[0] > valid_min) & (arr[1] > valid_min) & (arr[2] > valid_min)
    n = int(valid.sum())
    if n < min_pixels:
        return None
    return {
        "n_pixels": n,
        "IR": float(arr[0][valid].mean()),
        "RED": float(arr[1][valid].mean()),
        "BG": float(arr[2][valid].mean()),
    }


def polygon_masks(
    polygon_geom,
    buffer_inner_m: float,
    buffer_outer_m: float,
    *,
    window_transform,
    window_shape: tuple[int, int],
):
    """Build (interior_mask, ring_mask) for a single polygon in JP2 pixel coords.

    The interior mask is the polygon itself; the ring mask is the annulus from
    `buffer_inner_m` to `buffer_outer_m` outside the polygon, with the polygon itself
    excluded. Both are rasterised into the given window grid.
    """
    interior = ~geometry_mask(
        [polygon_geom], out_shape=window_shape, transform=window_transform, invert=False
    )
    outer = polygon_geom.buffer(buffer_outer_m)
    inner = polygon_geom.buffer(buffer_inner_m)
    ring_geom = outer.difference(inner)
    if ring_geom.is_empty:
        return interior, np.zeros(window_shape, dtype=bool)
    ring = ~geometry_mask(
        [ring_geom], out_shape=window_shape, transform=window_transform, invert=False
    )
    return interior, ring
