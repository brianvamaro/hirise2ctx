"""The deployed abundance layer's **size-floor basis** — R03 / R83 / R84, PLAN_RegionalMap leg 4.

`fractional_area` is not "rock abundance". It is *the area share of boulders large enough for
BoulderNet to have detected them in that particular HiRISE image*, and that qualifier varies across
the training cohort by a factor of ~3.6 in area. The deployed raster is quantile-matched onto a pool
that **mixes** two such conventions, and until now the product recorded none of it: `write_geotiff`
wrote no tags, the label sidecars carry only a global `detection_filters` block identical across all
38 images, and `DATA_DICTIONARY.md` describes `fractional_area` with no size caveat at all.

Why a module rather than a constant. A provenance field that *asserts* rather than *measures* has
been caught on this project four times, so everything here is derived from the artifacts and carries
the inputs it was derived from. `SizeFloorBasis.measure()` re-derives it; `product_tags()` renders it
into GeoTIFF metadata; a banked JSON lets a map driver stamp a raster without re-reading 7 M polygons.

The three measured facts, on the v2 cohort (2026-08-11, read-only over all 38
`cache_v2/reprojected_detections/*.gpkg` + `cache_v2/pds_labels/*.LBL` + the S=32 label pool):

* **The effective floor is not the raw detection floor.** Stage 4 applies one global
  `min_size_m = 1.4105 m` equivalent-circle diameter (1.5626 m²) *after* Stage 1, so
  `effective = max(global filter, the image's own natural floor)`. For the 0.25 m/px cohort the
  filter *is* the floor — 1.5626 m² for all 12, uniformly. For the 0.50 m/px cohort the filter
  removes nothing and each image keeps its own: 2.9652–5.5719 m² (diam 1.943–2.664 m), 26 distinct
  values. **The coarse cohort is the internally heterogeneous one** (R83's correction to R03, which
  had it the other way round because it read the Stage-1 minima as if they were post-filter).
* **The pool is a 78.4 / 21.6 tile-share mixture** — 126,214 coarse and 34,791 fine of 161,005
  S=32 tiles. Independently re-derived here; the audit had flagged the figure as unverified.
* **Tile share is not image share.** By image the split is 68.4 / 31.6. Quoting one for the other
  is wrong by ten points, and R84's number is the tile one.

**What this is not.** A regional output pixel does not inherit a HiRISE detection floor — CTX at
5 m/px cannot resolve any of these boulders individually. The floor describes *the training target's
definition*, i.e. which boulders the abundance number is counting, and therefore what an external
product must match to be comparable. Keep it distinct from detector completeness.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# The Stage-4 global filter, in the units `src.labeling._apply_detection_filters` uses: an
# equivalent-circle DIAMETER in metres. Mirrored from `config_v2.yaml detection_filters.min_size_m`
# rather than imported, because this module must be able to describe a *banked* product whose
# config may since have moved; `measure()` records the value it actually used.
DEFAULT_MIN_SIZE_M = 1.4105
SIZE_FLOOR_BASIS_VERSION = "v2_mixed_floor_1"


def diameter_to_area(diam_m: float) -> float:
    """Equivalent-circle area for a diameter, the convention `min_size_m` is expressed in."""
    return math.pi * (float(diam_m) / 2.0) ** 2


def area_to_diameter(area_m2: float) -> float:
    return 2.0 * math.sqrt(float(area_m2) / math.pi)


def effective_floor_m2(natural_floor_m2: float, min_size_m: float = DEFAULT_MIN_SIZE_M) -> float:
    """The floor a labelled image actually has: the global filter, or its own, whichever is higher.

    This is the whole of R83's correction in one line. Reading the Stage-1 polygon minimum as "the
    floor" understates the fine cohort's by ~2x, because the Stage-4 filter has not been applied to
    those files yet — and it is *only* the fine cohort the filter touches.
    """
    return max(float(natural_floor_m2), diameter_to_area(min_size_m))


@dataclass(frozen=True)
class SizeFloorBasis:
    """The size convention of the pool a deployed abundance layer was calibrated on.

    Every field is measured. `per_image` is the audit trail; the scalars are what a product tag
    quotes. Construct with `measure()` or `load()`, never by hand — the point is that it cannot
    claim a mixture it did not count.
    """

    version: str
    min_size_m: float                     # the global Stage-4 filter in force when measured
    n_images: int
    n_tiles: int
    tile_px: int
    floor_min_m2: float
    floor_max_m2: float
    floor_tile_weighted_mean_m2: float
    n_distinct_floors: int
    tile_share_by_scale: dict             # {map_scale_mpp: share of POOL TILES}
    image_share_by_scale: dict            # {map_scale_mpp: share of IMAGES} -- a different number
    per_image: list = field(default_factory=list)

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_records(cls, per_image: list, tile_counts: dict, *, tile_px: int = 32,
                     min_size_m: float = DEFAULT_MIN_SIZE_M) -> "SizeFloorBasis":
        """Assemble from per-image records + `{obs_id: n_pool_tiles}`. Pure; the tested seam."""
        recs = []
        for r in per_image:
            eff = effective_floor_m2(r["natural_floor_m2"], min_size_m)
            recs.append({**r, "effective_floor_m2": eff,
                         "effective_floor_diam_m": area_to_diameter(eff),
                         "n_pool_tiles": int(tile_counts.get(r["obs_id"], 0)),
                         "floor_is_the_global_filter":
                             bool(r["natural_floor_m2"] < diameter_to_area(min_size_m))})
        n_tiles = sum(r["n_pool_tiles"] for r in recs)
        if not n_tiles:
            raise ValueError("the pool is empty; a size-floor basis over 0 tiles states nothing")

        def _share(key, denom, weights):
            out = {}
            for r, w in zip(recs, weights):
                out[r[key]] = out.get(r[key], 0.0) + w / denom
            return {str(k): float(v) for k, v in sorted(out.items())}

        floors = np.array([r["effective_floor_m2"] for r in recs], dtype=float)
        w = np.array([r["n_pool_tiles"] for r in recs], dtype=float)
        return cls(
            version=SIZE_FLOOR_BASIS_VERSION,
            min_size_m=float(min_size_m),
            n_images=len(recs), n_tiles=int(n_tiles), tile_px=int(tile_px),
            floor_min_m2=float(floors.min()), floor_max_m2=float(floors.max()),
            floor_tile_weighted_mean_m2=float((floors * w).sum() / w.sum()),
            # distinct floors are counted only where they carry tiles -- an image with no pool
            # tiles contributes no floor to the product, however distinct its own happens to be
            n_distinct_floors=int(np.unique(floors[w > 0]).size),
            tile_share_by_scale=_share("map_scale_mpp", n_tiles, w),
            image_share_by_scale=_share("map_scale_mpp", len(recs), np.ones(len(recs))),
            per_image=sorted(recs, key=lambda r: r["obs_id"]),
        )

    # ---- persistence --------------------------------------------------------------------
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SizeFloorBasis":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("version") != SIZE_FLOOR_BASIS_VERSION:
            raise ValueError(
                f"size-floor basis is {d.get('version')!r}, this code writes "
                f"{SIZE_FLOOR_BASIS_VERSION!r}; re-measure rather than mixing conventions")
        return cls(**d)

    # ---- what a product carries ----------------------------------------------------------
    def summary(self) -> str:
        """One human-readable sentence, for a tag and for figure captions."""
        by = ", ".join(f"{float(v):.1%} at {k} m/px" for k, v in
                       sorted(self.tile_share_by_scale.items(), key=lambda kv: -float(kv[1])))
        return (f"Target = area share of boulders above a per-image detection floor of "
                f"{self.floor_min_m2:.3f}-{self.floor_max_m2:.3f} m2 equivalent-circle area "
                f"({area_to_diameter(self.floor_min_m2):.2f}-"
                f"{area_to_diameter(self.floor_max_m2):.2f} m diameter); the calibration pool is a "
                f"mixture of {self.n_distinct_floors} floors over {self.n_images} HiRISE images "
                f"({by}, by pool tile). NOT size-independent rock abundance.")

    def product_tags(self) -> dict:
        """GeoTIFF tags. **R84's fix** — `write_geotiff` wrote none, so a shipped raster could not
        state what its abundance number counts.

        Flat strings on purpose: GDAL metadata is string-valued, and a reader that has to parse
        nested JSON out of a tag will not bother. `SIZE_FLOOR_PER_IMAGE_JSON` is deliberately
        omitted — 38 records do not belong in every raster header; the banked basis file is the
        audit trail and `SIZE_FLOOR_BASIS_VERSION` names it.
        """
        return {
            "SIZE_FLOOR_BASIS_VERSION": self.version,
            "SIZE_FLOOR_MIN_M2": f"{self.floor_min_m2:.4f}",
            "SIZE_FLOOR_MAX_M2": f"{self.floor_max_m2:.4f}",
            "SIZE_FLOOR_MEAN_M2_TILE_WEIGHTED": f"{self.floor_tile_weighted_mean_m2:.4f}",
            "SIZE_FLOOR_N_DISTINCT": str(self.n_distinct_floors),
            "SIZE_FLOOR_N_IMAGES": str(self.n_images),
            "SIZE_FLOOR_N_POOL_TILES": str(self.n_tiles),
            "SIZE_FLOOR_TILE_SHARE_BY_MPP": json.dumps(self.tile_share_by_scale),
            "SIZE_FLOOR_IMAGE_SHARE_BY_MPP": json.dumps(self.image_share_by_scale),
            "SIZE_FLOOR_GLOBAL_MIN_SIZE_M": f"{self.min_size_m}",
            "SIZE_FLOOR_SUMMARY": self.summary(),
        }


# ============================================================================
# Measurement (reads artifacts; kept out of the dataclass so the dataclass stays testable)
# ============================================================================


def map_scale_from_pds_label(lbl_path: str | Path) -> float | None:
    """`MAP_SCALE` in m/px from a HiRISE PDS `.LBL`.

    Source of truth for the pixel scale. `scripts/build_vclaire_manifest.py` takes `MapPixel_mpp`
    from the label *spreadsheet* instead, which is why two cohort rows are blank
    (`LabelSource: none`) — both are 0.5 m/px and always were, readable from the cached `.LBL`.
    """
    p = Path(lbl_path)
    if not p.exists():
        return None
    for line in p.read_text(errors="ignore").splitlines():
        if line.strip().startswith("MAP_SCALE"):
            try:
                return float(line.split("=")[1].strip().split()[0].strip("<>"))
            except (IndexError, ValueError):
                return None
    return None


def natural_floor_from_detections(gpkg_path: str | Path) -> dict | None:
    """Smallest reprojected detection polygon in a Stage-1 `.gpkg`, plus the area distribution.

    This is the image's *natural* floor — what BoulderNet actually produced — before Stage 4's
    global filter. `effective_floor_m2` combines the two.
    """
    import pyogrio

    p = Path(gpkg_path)
    if not p.exists():
        return None
    a = pyogrio.read_dataframe(p, columns=[]).geometry.area.to_numpy()
    a = a[np.isfinite(a) & (a > 0)]
    if not a.size:
        return None
    return {"n_polygons": int(a.size),
            "natural_floor_m2": float(a.min()),
            "median_area_m2": float(np.median(a)),
            "area_below_coarse_floor_share": float(a[a < 6.25].sum() / a.sum())}
