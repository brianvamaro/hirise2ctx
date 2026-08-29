"""Shared logic for PLAN_MapValidation -- the five notebooks (30-34) that test the shipped
boulder-abundance map against independent data.

Single source of truth (CLAUDE.md: real logic lives in importable ``src/`` modules, notebooks
*call* it) for notebooks 30 geology, 31 craters, 32 illumination, 33 thermal and 34 Rodriguez.

**Everything here is read-only.** ``scripts/map_union.py`` is the sole producer of
``reports/map_union``; this module reads it and never writes a map artifact. The per-arm
products ``reports/map_region`` / ``reports/map_a1`` / ``reports/map_extended`` stay frozen.

Three project rules are baked into the API rather than left to each notebook:

* **One read surface.** ``load_union`` refuses anything that is not a union mosaic, because
  the two shipped arms overlap in 8 tiles and pooling them naively double-counts 15% of the
  footprint (see ``scripts/map_union.py``).
* **Three targets, one cell set** (ruling 3). ``three_targets`` masks ``abundance``,
  ``prob_raw`` and the ``prob >= 0.5`` rich flag to a *shared* finite mask, so a contrast can
  never be an artifact of the three layers describing different cells.
* **Significance never comes from the pixel count** (ruling 5). The 160 m cells are massively
  spatially autocorrelated, so the inferential helpers here bootstrap over **groups** --
  polygons, craters, CTX source frames -- and ``cluster_bootstrap_ci`` reports ``n_groups``
  alongside ``n_cells`` so a write-up cannot quietly quote the wrong n.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: The one read surface for every validation notebook. Built by `scripts/map_union.py`.
UNION_DIR = REPO / "reports" / "map_union"

#: Calibrated-probability cutoff defining a "rich" cell (ruling 4 -- notebook 24's existing
#: binary convention, kept so the two analyses mean the same thing by "rich").
RICH_PROB = 0.5

#: Native coarse-cell pitch: tile_px=32 x 5 m/px CTX. Mirrors `striping.PX_M`.
PX_M = 160.0

#: The three targets of ruling 3, in reporting order.
TARGET_NAMES = ("abundance", "prob_raw", "rich")

#: The single standing caveat block. Every notebook 30-34 quotes **this string**, so the
#: caveats cannot drift apart between notebooks or soften over time.
CAVEAT_MD = """
> ### Standing caveats — these apply to every result in this notebook
>
> **1. The CTX source-frame striping artifact is present and UNCORRECTED.** The map inherits
> rectangular, frame-shaped structure from per-frame CTX radiometry (cause solved; no
> mitigation survives — A1 was demoted to a sensitivity arm on 2026-08-25). Per
> PLAN_MapValidation ruling 2 this is **not controlled for here**: no rotation nulls, no A1
> arm. Quantifying and correcting it is a **separate investigation**, with notebook 32 as its
> entry point. **Consequence: every contrast below is an UPPER BOUND on the geologic signal.**
>
> **2. `abundance` is size-floor-referenced, not absolute rock abundance.** It is the area
> share of boulders above a *per-image* detection floor (1.563–5.572 m², i.e. 1.41–2.66 m
> diameter) mixed over 20 floors / 38 images (`v2_mixed_floor_2`, `models/deployable_g2`). It
> is **not** size-independent rock abundance — which matters most in notebook 33, where
> TES/IRTM rock abundance is a genuinely different physical quantity.
>
> **3. Truth coverage thins fast outside circum-Chryse.** 23 of the 39-image training cohort
> sit inside the shipped 26-tile block; only 1 sits in the new southern block. Any claim about
> `map_extended` terrain is **extrapolation**, and captions must say so.
>
> **4. Map cells ≠ label cells.** The map grid is globally anchored (R01); the Stage-4 label
> grid stays tile-anchored. A map↔label comparison must resample, never index-match.
>
> **5. Never presence AUC.** Skill-like numbers use the rich/poor threshold `fa > 1e-2`
> family (`meaningful_auc` / `pr_auc@1e-2` / `precision@5%`), plus Spearman ρ and per-bin RMSE.
""".strip()


# --------------------------------------------------------------------------- reading
def _union_path(union_dir, layer: str) -> Path:
    return Path(union_dir) / f"regional_{layer}_mosaic.tif"


def load_union(layer: str = "abundance", *, union_dir=None, dtype: str = "float32",
               require_union_tags: bool = True):
    """Read the union mosaic for one layer. **Read-only, and never builds.**

    Returns ``(arr, transform, crs_wkt, meta)`` exactly like
    ``src.mapping.load_regional_mosaic``, with ``meta["tags"]`` carrying the ``SIZE_FLOOR_*``
    basis and the ``UNION_*`` provenance, plus ``meta["union_tiles"]`` /
    ``meta["tile_origin"]`` parsed out of the tags.

    ``require_union_tags`` (default on) is the guard that makes "one read surface" real: it
    refuses a mosaic with no ``UNION_N_TILES`` tag. Being handed an *arm* mosaic instead of
    the union is a silent 50%-coverage bug — the arrays load, every statistic computes, and
    the answer is about a third of Mars. Turn it off only to read a deliberately non-union
    mosaic (an arm, for a like-for-like comparison), and say so in the notebook.

    ⚠ Memory: the union is ~16.3k x 10.4k cells, so **~680 MB per layer at float32** and
    double that at float64. Default is float32; ask for float64 only if you need it.
    """
    from src.mapping import COARSE_GRID_ID, load_regional_mosaic

    union_dir = UNION_DIR if union_dir is None else Path(union_dir)
    path = _union_path(union_dir, layer)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build the union first (it is not built on demand, and this "
            "loader will not write it):\n"
            "    conda run --no-capture-output -n geospatial python -u scripts/map_union.py")
    arr, transform, crs_wkt, meta = load_regional_mosaic(
        union_dir, layer, allow_build=False, dtype=dtype)
    tags = meta.get("tags") or {}
    if require_union_tags and "UNION_N_TILES" not in tags:
        raise ValueError(
            f"{path} carries no UNION_N_TILES tag, so it is not a union mosaic — most likely "
            "this is a single-arm product (map_region is 26 of the 54 mapped tiles). Reading "
            "it here would silently make every statistic below a statement about half the "
            "footprint. Point at reports/map_union, or pass require_union_tags=False "
            "deliberately.")
    grid = tags.get("MOSAIC_GRID_ID")
    if grid is not None and grid != COARSE_GRID_ID:
        raise ValueError(
            f"{path} is on lattice {grid!r}, not {COARSE_GRID_ID!r}. R01 exists to stop a "
            "sub-cell phase becoming a whole-cell displacement; a cross-lattice comparison "
            "is not co-registered.")
    meta = dict(meta)
    meta["layer"] = layer
    meta["union_tiles"] = [t for t in (tags.get("UNION_TILES", "") or "").split(",") if t]
    meta["n_union_tiles"] = int(tags.get("UNION_N_TILES", 0) or 0)
    meta["tile_origin"] = _parse_json_tag(tags.get("UNION_TILE_ORIGIN"))
    meta["adopted_tiles"] = [t for t in (tags.get("UNION_ADOPTED_TILES", "") or "").split(",")
                             if t]
    meta["size_floor"] = {k: v for k, v in tags.items() if k.startswith("SIZE_FLOOR_")}
    return arr, transform, crs_wkt, meta


def _parse_json_tag(value):
    """A GeoTIFF tag is a string; a missing tag is an ABSENCE, never an empty result.

    Returns ``None`` when the tag is absent, so a caller can tell "no measurement" from "no
    tiles" -- the same distinction the sidecar QA had to learn the hard way (CLAUDE.md: a
    missing `overlap` key is an absence of measurement, not a zero).
    """
    import json

    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def union_tiles(*, union_dir=None, layer: str = "abundance") -> list[str]:
    """The union's tile list, from the mosaic's own tags (not from a hardcoded list)."""
    _, _, _, meta = load_union(layer, union_dir=union_dir, dtype="float32")
    return list(meta["union_tiles"])


@dataclass
class Targets:
    """The ruling-3 triple on **one shared finite mask**, plus the grid it lives on.

    ``abundance`` and ``prob_raw`` are float arrays, NaN outside the shared mask. ``rich`` is
    a **bool** array (``prob >= RICH_PROB``) and is meaningless where ``finite`` is False --
    always index it with ``finite``, which is why the mask is returned rather than assumed.
    """
    abundance: np.ndarray
    prob_raw: np.ndarray
    rich: np.ndarray
    finite: np.ndarray
    transform: object
    crs_wkt: str
    meta: dict

    def as_dict(self) -> dict:
        """The three targets keyed by ``TARGET_NAMES``, ready for a per-target loop.

        ``rich`` comes through as float (1.0/0.0/NaN) so that the same summary code -- a
        median, a mean, a bootstrap -- means "rich fraction" for it and "level" for the other
        two, without a special case at every call site.
        """
        rich = np.where(self.finite, self.rich.astype(np.float32), np.nan)
        return {"abundance": self.abundance, "prob_raw": self.prob_raw, "rich": rich}

    @property
    def n_finite(self) -> int:
        return int(self.finite.sum())


def three_targets(*, union_dir=None, dtype: str = "float32",
                  rich_prob: float = RICH_PROB) -> Targets:
    """Load the ruling-3 triple: ``abundance``, ``prob_raw``, and rich = ``prob >= 0.5``.

    All three are masked to the **intersection** of their finite cells. That intersection is
    the point: a result that holds on all three is robust, one that appears only in
    ``abundance`` is likely a calibration-curve artifact -- and neither statement means
    anything if the three are describing different cells.

    A result on ``rich`` alone is a statement about the *tail crossing 0.5*, not about level.

    ⚠ Memory: peak is ~4 layer-loads (~2.7 GB at float32) while ``prob`` is still live; it
    settles to ~1.7 GB. The three layers are read one at a time and ``prob`` is dropped as
    soon as the rich flag is computed.
    """
    ab, transform, crs_wkt, meta = load_union("abundance", union_dir=union_dir, dtype=dtype)
    praw, tr2, _, meta_raw = load_union("prob_raw", union_dir=union_dir, dtype=dtype)
    prob, tr3, _, meta_prob = load_union("prob", union_dir=union_dir, dtype=dtype)
    for name, tr in (("prob_raw", tr2), ("prob", tr3)):
        if tuple(tr)[:6] != tuple(transform)[:6]:
            raise ValueError(
                f"the union's abundance and {name} mosaics do not share a transform, so the "
                "three targets cannot describe the same cells. Rebuild the union.")
    if not (ab.shape == praw.shape == prob.shape):
        raise ValueError(f"union layer shapes disagree: abundance {ab.shape}, "
                         f"prob_raw {praw.shape}, prob {prob.shape}")

    finite = np.isfinite(ab) & np.isfinite(praw) & np.isfinite(prob)
    rich = prob >= rich_prob
    del prob
    ab[~finite] = np.nan
    praw[~finite] = np.nan
    n_tiles = {"abundance": meta["n_union_tiles"], "prob_raw": meta_raw["n_union_tiles"],
               "prob": meta_prob["n_union_tiles"]}
    if len(set(n_tiles.values())) != 1:
        raise ValueError(
            f"the union's layers cover different tile counts {n_tiles} -- the three targets "
            "must describe the same footprint. Rebuild the union.")
    meta = dict(meta)
    meta.pop("layer", None)
    meta["rich_prob"] = rich_prob
    meta["n_finite_shared"] = int(finite.sum())
    meta["n_tiles_per_layer"] = n_tiles
    return Targets(abundance=ab, prob_raw=praw, rich=rich, finite=finite,
                   transform=transform, crs_wkt=crs_wkt, meta=meta)


# ------------------------------------------------------------------------- zonal reads
def _is_mapping(x) -> bool:
    from collections.abc import Mapping

    return isinstance(x, Mapping)


def _window_slices(bounds, transform, shape, *, pad: int = 1):
    """Row/col slices of the array block covering ``bounds`` (xyxy in the raster CRS).

    Windowing is not an optimisation detail here -- it is what makes a 67-polygon zonal pass
    over a 169-million-cell mosaic finish. Returns ``None`` when the bounds miss the raster.
    """
    from rasterio.transform import guard_transform, rowcol

    tr = guard_transform(transform)
    x0, y0, x1, y1 = bounds
    rows, cols = rowcol(tr, [x0, x1, x0, x1], [y0, y0, y1, y1], op=float)
    r0 = max(0, int(np.floor(min(rows))) - pad)
    r1 = min(shape[0], int(np.ceil(max(rows))) + pad)
    c0 = max(0, int(np.floor(min(cols))) - pad)
    c1 = min(shape[1], int(np.ceil(max(cols))) + pad)
    if r0 >= r1 or c0 >= c1:
        return None
    return slice(r0, r1), slice(c0, c1)


def zonal_cells(geom, arr, transform, *, all_touched: bool = False, finite_only: bool = True):
    """The cell values inside ``geom`` -- the **distribution**, not a summary.

    Returns a flat 1-D array (finite values only, by default). ``arr`` may instead be a
    mapping of name -> array (e.g. ``Targets.as_dict()``), in which case one geometry mask is
    computed and a dict of 1-D arrays is returned, all drawn from the **same cells** so the
    three targets stay comparable per polygon.

    Deliberately returns the distribution: the target is heavily zero-inflated and
    right-skewed (CLAUDE.md), so a mean over it is close to meaningless and a function that
    returned one would invite exactly that mistake. ECDFs, quantiles and cluster bootstraps
    all need the values.

    ``all_touched=False`` (pixel centres) matches the project's rasterize convention.
    Returns an empty array (or dict of empty arrays) when the geometry covers no finite cell.
    """
    from rasterio.features import geometry_mask
    from rasterio.transform import guard_transform
    from rasterio.windows import Window, transform as window_transform

    arrays = arr if _is_mapping(arr) else {"_": arr}
    ref = next(iter(arrays.values()))
    shape = ref.shape
    empty = {k: np.empty(0, dtype=np.float64) for k in arrays}
    sl = _window_slices(geom.bounds, transform, shape)
    if sl is None:
        return empty if _is_mapping(arr) else empty["_"]
    rs, cs = sl
    win = Window(cs.start, rs.start, cs.stop - cs.start, rs.stop - rs.start)
    wtr = window_transform(win, guard_transform(transform))
    inside = ~geometry_mask([geom], out_shape=(win.height, win.width), transform=wtr,
                            all_touched=all_touched, invert=False)
    out = {}
    for name, a in arrays.items():
        block = a[rs, cs]
        vals = np.asarray(block[inside], dtype=np.float64).ravel()
        out[name] = vals[np.isfinite(vals)] if finite_only else vals
    return out if _is_mapping(arr) else out["_"]


def radial_annuli(cx: float, cy: float, radius_m: float, arr, transform, *,
                  edges_R=(0.0, 1.0, 1.5, 2.0, 3.0, 5.0), finite_only: bool = True):
    """Cell values per annulus, with annulus edges in **crater radii** (§3 of the plan).

    ``(cx, cy)`` is the crater centre and ``radius_m`` its radius, both in the raster's CRS
    units (metres). Returns a list of 1-D arrays aligned with
    ``zip(edges_R[:-1], edges_R[1:])``; if ``arr`` is a mapping, returns a dict of such lists,
    every target drawn from the same annulus cells.

    Radii, not metres, is the whole point: an ejecta signature scales with crater size, so
    profiles in R are stackable across the size range while profiles in km are not.

    ⚠ The union's cells are 160 m. A crater with ``radius_m`` of a few cells has almost no
    cells in its inner annuli, so the caller must apply a minimum-count rule (deliberately
    left open -- PLAN_MapValidation §9 open question 1) rather than plotting noise.
    """
    edges = [float(e) for e in edges_R]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"edges_R must be strictly increasing with >=2 entries, got {edges_R}")
    if not (radius_m > 0):
        raise ValueError(f"radius_m must be positive, got {radius_m}")

    from rasterio.transform import guard_transform, xy

    arrays = arr if _is_mapping(arr) else {"_": arr}
    ref = next(iter(arrays.values()))
    shape = ref.shape
    reach = edges[-1] * radius_m
    bounds = (cx - reach, cy - reach, cx + reach, cy + reach)
    empty = {k: [np.empty(0, dtype=np.float64) for _ in edges[:-1]] for k in arrays}
    sl = _window_slices(bounds, transform, shape)
    if sl is None:
        return empty if _is_mapping(arr) else empty["_"]
    rs, cs = sl
    tr = guard_transform(transform)
    rr = np.arange(rs.start, rs.stop)
    cc = np.arange(cs.start, cs.stop)
    # cell-centre coordinates of the block, via the transform rather than by hand
    xs, _ = xy(tr, np.zeros_like(cc), cc)
    _, ys = xy(tr, rr, np.zeros_like(rr))
    dx = (np.asarray(xs, dtype=np.float64) - cx)[None, :]
    dy = (np.asarray(ys, dtype=np.float64) - cy)[:, None]
    r_over_R = np.sqrt(dx * dx + dy * dy) / float(radius_m)

    out = {}
    for name, a in arrays.items():
        block = np.asarray(a[rs, cs], dtype=np.float64)
        per_annulus = []
        for lo, hi in zip(edges, edges[1:]):
            # [lo, hi): half-open, so a cell lands in exactly one annulus
            sel = (r_over_R >= lo) & (r_over_R < hi)
            vals = block[sel].ravel()
            per_annulus.append(vals[np.isfinite(vals)] if finite_only else vals)
        out[name] = per_annulus
    return out if _is_mapping(arr) else out["_"]


# ------------------------------------------------------------------- effective n (ruling 5)
def frame_effective_n(tiles=None, *, union_dir=None, on_missing: str = "raise") -> dict:
    """Distinct CTX **source frames** contributing to the union footprint.

    This is the coarsest honest unit of independence in the map: the striping artifact is
    per-source-frame, so two cells in one frame are not two samples. Returned counts are what
    a write-up should quote next to the cell count -- ruling 5 forbids taking significance
    from the pixel count, and this is the denominator that replaces it.

    Frames are counted by ``PRODUCT_ID`` and **deduplicated across tiles**, because a CTX
    frame straddles Murray tile boundaries; summing per-tile counts overcounts.

    Uses ``src.striping.load_frames``, which reads the cached SeamMap if present and otherwise
    pulls just the shapefile out of the remote tile zip over ``/vsizip/vsicurl/`` range
    requests -- **no 1.8 GB tile download** -- and caches a GeoPackage. First call over 54
    uncached tiles therefore needs network.

    ``on_missing="raise"`` (default) fails on a tile whose SeamMap cannot be read;
    ``"skip"`` records it under ``failed`` and carries on, which keeps the count **a lower
    bound** -- say so if you use it.
    """
    from src import striping

    if on_missing not in ("raise", "skip"):
        raise ValueError(f"on_missing must be 'raise' or 'skip', got {on_missing!r}")
    tiles = union_tiles(union_dir=union_dir) if tiles is None else list(tiles)
    per_tile, failed = {}, {}
    seen: set[str] = set()
    for t in tiles:
        try:
            g = striping.load_frames(t, dissolve=True)
        except Exception as exc:                       # noqa: BLE001 -- reported, not hidden
            if on_missing == "raise":
                raise
            failed[t] = f"{type(exc).__name__}: {exc}"
            continue
        ids = {str(v) for v in g["PRODUCT_ID"]} if "PRODUCT_ID" in g else set()
        per_tile[t] = len(ids)
        seen |= ids
    return {"n_frames": len(seen), "n_tiles": len(tiles), "per_tile": per_tile,
            "product_ids": sorted(seen), "failed": failed,
            "sum_per_tile": sum(per_tile.values()),
            "note": "n_frames is deduplicated across tiles; sum_per_tile is not and "
                    "overcounts, because CTX frames straddle Murray tile boundaries."
                    + (" LOWER BOUND: some tiles failed." if failed else "")}


def cluster_bootstrap_ci(group_values, *, stat=np.median, n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> dict:
    """Percentile CI for ``stat`` over pooled cells, resampling **groups** with replacement.

    ``group_values`` is a sequence of 1-D arrays -- one per polygon, crater or source frame.
    Each bootstrap replicate draws ``n_groups`` groups with replacement and pools *all* their
    cells, so the uncertainty reflects how few independent units there are (67 polygons, not
    57 million cells). This is the one inferential guard PLAN_MapValidation keeps after
    ruling 2 removed the rotation nulls, and it is honest error-bar accounting rather than
    artifact correction -- it does **not** make a contrast free of the striping artifact.

    Returns ``point``/``lo``/``hi`` plus ``n_groups`` and ``n_cells``. Report **both** n's:
    quoting the cell count is the mistake this function exists to prevent.
    """
    groups = [np.asarray(g, dtype=np.float64).ravel() for g in group_values]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size]
    n_groups = len(groups)
    n_cells = int(sum(g.size for g in groups))
    out = {"n_groups": n_groups, "n_cells": n_cells, "n_boot": int(n_boot),
           "alpha": float(alpha), "stat": getattr(stat, "__name__", str(stat))}
    if n_groups == 0:
        return {**out, "point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = float(stat(np.concatenate(groups)))
    if n_groups == 1:
        # one group is one sample: there is no between-group spread to resample. Returning a
        # zero-width interval would read as certainty, so the CI is explicitly undefined.
        return {**out, "point": point, "lo": float("nan"), "hi": float("nan"),
                "note": "CI undefined: a single group is a single independent sample."}
    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n_groups, size=n_groups)
        reps[i] = stat(np.concatenate([groups[j] for j in idx]))
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {**out, "point": point, "lo": float(lo), "hi": float(hi)}


def cluster_bootstrap_ratio_ci(counts, sums, *, n_boot: int = 2000, seed: int = 0,
                               alpha: float = 0.05) -> dict:
    """Cluster bootstrap for a **ratio** statistic, from per-group `(count, sum)` pairs only.

    For any statistic that is a ratio of sums over cells -- **mean abundance**
    (`sum(abundance) / n_cells`) and **rich fraction** (`count(prob>=0.5) / n_cells`) are both
    exactly that -- the pooled value over a resampled set of groups is
    `sum(sums[idx]) / sum(counts[idx])`. So this is **exact**, not an approximation, and it
    needs O(n_groups) memory instead of the cell arrays.

    That matters concretely: the 122-tile union has 265.8 M finite cells, so holding each
    polygon's values to feed `cluster_bootstrap_ci` would be ~2 GB per target. Subsampling
    instead would silently change the estimator (it would weight small polygons like large
    ones); this keeps the exact statistic and drops only the ability to bootstrap a
    *quantile*, which is not a ratio. Use `cluster_bootstrap_ci` when the statistic is a
    median or a quantile and the cell arrays are small enough to hold (crater annuli).

    Groups with `count == 0` are dropped -- an empty polygon is not an observation. Returns
    `point`/`lo`/`hi` plus `n_groups` and `n_cells`; report both n's.
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    sums = np.asarray(sums, dtype=np.float64).ravel()
    if counts.shape != sums.shape:
        raise ValueError(f"counts {counts.shape} and sums {sums.shape} must be the same length")
    keep = np.isfinite(counts) & np.isfinite(sums) & (counts > 0)
    counts, sums = counts[keep], sums[keep]
    n_groups = int(counts.size)
    n_cells = int(counts.sum())
    out = {"n_groups": n_groups, "n_cells": n_cells, "n_boot": int(n_boot),
           "alpha": float(alpha), "stat": "ratio_of_sums"}
    if n_groups == 0:
        return {**out, "point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = float(sums.sum() / counts.sum())
    if n_groups == 1:
        return {**out, "point": point, "lo": float("nan"), "hi": float("nan"),
                "note": "CI undefined: a single group is a single independent sample."}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(int(n_boot), n_groups))
    reps = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {**out, "point": point, "lo": float(lo), "hi": float(hi)}


def variance_decomposition(group_values) -> dict:
    """Between-group vs within-group variance over a set of per-group cell distributions.

    Notebook 30 §3 turns on this number: if within-unit-between-polygon variance dominates
    between-unit variance, "geologic unit" is not a useful predictor of boulder abundance at
    this scale -- a publishable negative that PLAN_MapValidation §10 named in advance.

    ``eta2`` is the between-group share of total variance, computed the same way as
    ``striping.eta2`` so the two are comparable numbers.
    """
    groups = [np.asarray(g, dtype=np.float64).ravel() for g in group_values]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size]
    if len(groups) < 2:
        return {"n_groups": len(groups), "n_cells": int(sum(g.size for g in groups)),
                "eta2": float("nan"), "between": float("nan"), "within": float("nan"),
                "total": float("nan"),
                "note": "eta2 undefined with fewer than two non-empty groups."}
    allv = np.concatenate(groups)
    grand = float(allv.mean())
    total = float(((allv - grand) ** 2).sum())
    between = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups))
    within = total - between
    return {"n_groups": len(groups), "n_cells": int(allv.size),
            "eta2": (between / total) if total > 0 else float("nan"),
            "between": between, "within": within, "total": total}


def nested_variance_decomposition(labels, counts, sums, sumsqs) -> dict:
    """Three-level variance split from per-polygon moments: **between-unit**,
    **within-unit-between-polygon**, **within-polygon**.

    This is the number notebook 30 turns on, and PLAN_MapValidation §10 named the answer in
    advance: **if within-unit-between-polygon variance dominates between-unit variance, then
    "geologic unit" is not a useful predictor of boulder abundance at this scale** — a
    publishable negative rather than a disappointing plot.

    Takes moments, not values, so it is exact over all 265.8 M union cells without holding
    them: `labels[i]` is polygon i's unit, and `counts`/`sums`/`sumsqs` are its cell count,
    sum and sum of squares. The three components add to the total sum of squares by
    construction (the law of total variance applied twice), which the returned
    `closure_residual` checks — a non-zero residual means the moments are inconsistent.

    Note the shares are of the **sum of squares about the grand mean**, i.e. eta-squared-like
    fractions directly comparable to `striping.eta2` and to `variance_decomposition`.
    """
    labels = np.asarray(labels)
    n = np.asarray(counts, dtype=np.float64).ravel()
    s = np.asarray(sums, dtype=np.float64).ravel()
    q = np.asarray(sumsqs, dtype=np.float64).ravel()
    if not (labels.shape[0] == n.size == s.size == q.size):
        raise ValueError("labels, counts, sums and sumsqs must be the same length")
    keep = (n > 0) & np.isfinite(n) & np.isfinite(s) & np.isfinite(q)
    labels, n, s, q = labels[keep], n[keep], s[keep], q[keep]
    if labels.size == 0:
        return {"n_units": 0, "n_polygons": 0, "n_cells": 0, "eta2_between_unit": float("nan"),
                "note": "no non-empty polygons."}
    N = float(n.sum())
    grand = float(s.sum() / N)
    total = float(q.sum() - N * grand ** 2)          # SS about the grand mean
    poly_mean = s / n
    within_poly = float((q - n * poly_mean ** 2).sum())
    between_unit = 0.0
    within_unit = 0.0
    units = np.unique(labels)
    for u in units:
        m = labels == u
        n_u = float(n[m].sum())
        mean_u = float(s[m].sum() / n_u)
        between_unit += n_u * (mean_u - grand) ** 2
        within_unit += float((n[m] * (poly_mean[m] - mean_u) ** 2).sum())
    resid = total - (between_unit + within_unit + within_poly)
    return {"n_units": int(units.size), "n_polygons": int(labels.size), "n_cells": int(N),
            "grand_mean": grand, "total_ss": total,
            "ss_between_unit": between_unit,
            "ss_within_unit_between_polygon": within_unit,
            "ss_within_polygon": within_poly,
            "eta2_between_unit": (between_unit / total) if total > 0 else float("nan"),
            "eta2_within_unit_between_polygon": (within_unit / total) if total > 0
            else float("nan"),
            "eta2_within_polygon": (within_poly / total) if total > 0 else float("nan"),
            "closure_residual": resid,
            "closure_residual_relative": (abs(resid) / total) if total > 0 else float("nan")}


# ------------------------------------------------------- notebook 30: geologic units
#: Tanaka et al. 2014 global geologic map of Mars, SIM3292 (DOI 10.3133/sim3292), as
#: downloaded. The `/vsizip/` path needs FORWARD slashes and a SINGLE leading slash -- the
#: backslash and `//` forms both fail.
SIM3292_ZIP = "C:/Users/brian/Downloads/sim3292_database.zip"
SIM3292_GDB = (f"/vsizip/{SIM3292_ZIP}/SIM3292_MarsGlobalGeologicGIS_20M/"
               "SIM3292_geodatabase.gdb")
SIM3292_LAYER = "SIM3292_Global_Geology"

#: Reportability floor for a geologic unit / crater stratum, ruled 2026-08-29 with the real
#: distributions in hand: **50,000 mapped cells** = 1,280 km2 at 160 m. A unit below it is
#: **flagged and excluded from the headline ranking, never silently dropped** -- 12 of the 75
#: polygons in the union get 0 cells (they fall in the union's ~20% nodata), and `ANa` has
#: 63,262 km2 of bbox area with 0 mapped cells, which is a coverage fact worth reporting.
MIN_CELLS_UNIT = 50_000

#: Mars epoch ordering for the stratigraphic-age axis. Keys are the epoch part of a Tanaka
#: unit code; values are `(rank, label)` with rank increasing = YOUNGER. Units spanning two
#: epochs (`HN`, `AH`, `AN`) get the midpoint AND `spans=True`, because placing a two-epoch
#: unit at a single age is an interpretation, not a measurement.
_EPOCH_RANKS = {
    "eN": (1.0, "Early Noachian"), "mN": (2.0, "Middle Noachian"),
    "lN": (3.0, "Late Noachian"), "N": (2.0, "Noachian (undivided)"),
    "HN": (3.5, "Hesperian-Noachian"),
    "eH": (4.0, "Early Hesperian"), "lH": (5.0, "Late Hesperian"),
    "H": (4.5, "Hesperian (undivided)"),
    "AH": (5.5, "Amazonian-Hesperian"),
    "AN": (3.5, "Amazonian-Noachian"),
    "eA": (6.0, "Early Amazonian"), "mA": (7.0, "Middle Amazonian"),
    "lA": (8.0, "Late Amazonian"), "A": (7.0, "Amazonian (undivided)"),
}
_SPANNING_EPOCHS = ("HN", "AH", "AN")


def stratigraphic_rank(unit: str) -> dict:
    """Parse a Tanaka SIM3292 unit code into a stratigraphic age rank.

    Codes are `[e|m|l]<epoch letters><terrain letter>`, e.g. `mNh` = Middle Noachian highland,
    `AHi` = Amazonian-and-Hesperian impact, `Hto` = Hesperian transition outflow. Returns
    `{"unit", "epoch", "rank", "label", "spans", "terrain"}`; `rank` increases with
    **younger**, and is `nan` for a code this function does not recognise -- reported, not
    guessed, because a mis-parsed code would silently move a unit along the very age axis the
    conclusion rests on.

    `spans=True` marks a two-epoch unit (`HNt`, `AHi`, `AHv`, `ANa`). Those carry a midpoint
    rank so they can be plotted, but a claim about abundance-vs-age must say whether it
    survives excluding them: an `AHi` unit is not "5.5 old", it is *undated within a ~3 Gyr
    window*, and `ANa` spans nearly all of Mars history.
    """
    u = str(unit)
    for prefix in ("e", "m", "l"):
        if u.startswith(prefix) and len(u) > 1 and u[1].isupper():
            for epoch_len in (2, 1):
                key = prefix + u[1:1 + epoch_len]
                if key in _EPOCH_RANKS:
                    rank, label = _EPOCH_RANKS[key]
                    return {"unit": u, "epoch": key, "rank": rank, "label": label,
                            "spans": key[1:] in _SPANNING_EPOCHS,
                            "terrain": u[1 + epoch_len:]}
    for epoch_len in (2, 1):                     # no e/m/l prefix: HNt, AHi, Hto, Nhu
        key = u[:epoch_len]
        if key in _EPOCH_RANKS:
            rank, label = _EPOCH_RANKS[key]
            return {"unit": u, "epoch": key, "rank": rank, "label": label,
                    "spans": key in _SPANNING_EPOCHS, "terrain": u[epoch_len:]}
    return {"unit": u, "epoch": None, "rank": float("nan"), "label": "unparsed",
            "spans": False, "terrain": ""}


def bounds_lonlat(bounds, *, radius_m: float = 3396190.0):
    """Projected `(x0, y0, x1, y1)` metres -> `(lon0, lat0, lon1, lat1)` degrees.

    The map CRS is equirectangular with `standard_parallel_1=0` and `central_meridian=0` on the
    clon_0 sphere, so degrees are metres / (pi*R/180) exactly -- no pyproj round trip, and no
    chance of the answer depending on which CRS string a caller happened to pass.
    """
    x0, y0, x1, y1 = (float(v) for v in bounds)
    deg = np.pi * radius_m / 180.0
    return x0 / deg, y0 / deg, x1 / deg, y1 / deg


def load_geology(bounds, crs_wkt: str, *, gdb: str | None = None, layer: str | None = None,
                 densify_deg: float = 0.25, buffer_m: float = 20_000.0):
    """SIM3292 geologic polygons clipped to `bounds`, in the map's CRS. Returns `(gdf, report)`.

    `bounds` is `(x0, y0, x1, y1)` in the **map's projected metres** (the union's own
    `rasterio.transform.array_bounds`). The returned frame carries `Unit`, `UnitDesc`,
    `SphArea_km` and a reprojected `geometry`, plus `area_km2` computed **after** reprojection.

    **The trap this function exists for: reprojecting SIM3292 out of Robinson produces
    non-finite coordinates, and `make_valid` then CRASHES** with
    `CGAlgorithmsDD::orientationIndex encountered NaN/Inf numbers`. Measured 2026-08-29 on the
    real layer: all 1311 polygons are valid in the source `Robinson_clon0_Mars_2000_Sphere`,
    but the *inverse* Robinson overflows for **62** of them (vertices at |lon| -> 180 go to
    `inf`), and a non-finite geometry makes `.intersects()` return garbage behind nothing but a
    `RuntimeWarning: invalid value encountered in intersects`. So a naive
    `to_crs(...).clip(...)` yields a *plausible, wrong* polygon set. PLAN_MapValidation's
    "67 polygons / 16 units" was measured that way and is not trustworthy.

    The order of operations is therefore the method, not an optimisation:

    1. Build the selection window in **lon/lat** and **densify** it (`densify_deg`) -- Robinson
       curves straight edges, so an undensified rectangle under-covers.
    2. Project it **forward** into Robinson (the safe direction) and **buffer outward**
       (`buffer_m`), so the intermediate cut provably removes nothing inside the final bounds.
    3. Select **and clip in Robinson**, where every geometry is finite and valid. This is what
       tames the one globe-spanning polygon (`lHl`, 13.4 M km2, 143,524 vertices) that
       overflows even after selection.
    4. *Then* reproject the clipped subset, repair, and clip exactly to `bounds`.

    Robinson is neither equal-area nor conformal, so **any area computed in the source CRS is
    wrong**. `area_km2` is computed after reprojection; `SphArea_km` is the authors' own
    spherical area for the *whole, unclipped* polygon and is kept for reference only.

    `report` records the counts at each step and the repair tally, so a notebook can state what
    was fixed rather than implying nothing was.
    """
    import geopandas as gpd
    import shapely
    from shapely.geometry import box

    gdb = SIM3292_GDB if gdb is None else gdb
    layer = SIM3292_LAYER if layer is None else layer
    src = gpd.read_file(gdb, layer=layer, engine="pyogrio")   # fiona is NOT installed here

    def _nonfinite(geoms) -> int:
        return int(sum(not np.isfinite(np.asarray(shapely.get_coordinates(g))).all()
                       for g in geoms))

    rep = {"source_polygons": len(src), "source_units": int(src["Unit"].nunique()),
           "source_crs": src.crs.name,
           "source_invalid": int((~src.geometry.is_valid).sum()),
           "source_nonfinite": _nonfinite(src.geometry)}

    x0, y0, x1, y1 = (float(v) for v in bounds)
    lon0, lat0, lon1, lat1 = bounds_lonlat((x0, y0, x1, y1))
    rep["window_lonlat"] = [lon0, lat0, lon1, lat1]
    win_ll = shapely.segmentize(box(lon0, lat0, lon1, lat1), densify_deg)
    win_rob = gpd.GeoSeries([win_ll], crs=src.crs.geodetic_crs).to_crs(src.crs).iloc[0]
    win_rob = win_rob.buffer(buffer_m)

    hit = src[src.geometry.intersects(win_rob)].copy()
    rep["selected_in_source_crs"] = len(hit)
    cut = gpd.clip(hit, win_rob)
    cut = cut[~cut.geometry.is_empty].copy()
    rep["nonfinite_after_source_clip"] = _nonfinite(cut.geometry)

    proj = cut.to_crs(crs_wkt)
    rep["nonfinite_after_reprojection"] = _nonfinite(proj.geometry)
    n_invalid = int((~proj.geometry.is_valid).sum())
    if n_invalid:
        proj["geometry"] = proj.geometry.make_valid()
    rep["repaired"] = n_invalid
    rep["invalid_after_repair"] = int((~proj.geometry.is_valid).sum())

    out = gpd.clip(proj, box(x0, y0, x1, y1))
    out = out[~out.geometry.is_empty].reset_index(drop=True)
    out["area_km2"] = out.geometry.area / 1e6
    rep["polygons"] = len(out)
    rep["units"] = int(out["Unit"].nunique())
    rep["area_km2_total"] = float(out["area_km2"].sum())
    rep["bbox_area_km2"] = (x1 - x0) * (y1 - y0) / 1e6
    # SIM3292 is a complete PARTITION of Mars, so the clipped polygons must tile the window
    # exactly. This ratio is a real validity gate, not a summary: it came out at 1.000000 on
    # the 122-tile union, and a shortfall would mean the Robinson cut ate area that belongs.
    rep["partition_closure"] = (rep["area_km2_total"] / rep["bbox_area_km2"]
                                if rep["bbox_area_km2"] else float("nan"))
    return out, rep
