"""Step-12 QA over the shipped regional map arms (PLAN_Rebuild §3 step 12).

Read-only over ``reports/map_region/`` (baseline) and ``reports/map_a1/`` (A1), which step 11
shipped at 26 tiles each and step 12 promoted to these canonical names. Nothing here
re-infers or re-renders; the only writes are the regional mosaics, which
``scripts/map_mosaics.py`` puts through ``src.mapping.write_geotiff`` (atomic + verified).

**The load-bearing reason this module exists: the sidecars come in three schema generations,
and a QA table that conflates them reports a fiction.** The overlap-agreement gate moved
twice mid-flight (DECISIONS 2026-08-24d, 2026-08-25):

``g1_scalar_only``
    ``overlap_disagreements`` only -- a scalar counted on the CALIBRATED ``prob`` layer,
    where isotonic collapses raw fp16 disagreements onto shared knots. It reads 0 on every
    g1 tile, and that 0 means *not measured on the gate quantity*, **not** *no disagreement*.
``g2_raw_fraction``
    ``overlap.prob_raw.fraction`` -- the fraction of duplicated cells disagreeing at ANY
    magnitude. Right layer, but with no per-cell significance floor, so it is an upper
    bound on the gate quantity rather than the gate quantity.
``g3_floored``
    adds ``n_significant`` / ``fraction_raw`` / ``significant_abs`` -- the fraction after a
    1e-6 per-cell floor plus a 16-cell absolute floor. **No shipped tile is g3**: the floor
    landed after the last render.

:func:`overlap_status` therefore returns an explicit ``unknown_on_gate_layer`` for g1 rather
than a number, and every aggregate here carries its own denominator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

GEN_G1 = "g1_scalar_only"
GEN_G2 = "g2_raw_fraction"
GEN_G3 = "g3_floored"
SIDECAR_GENERATIONS = (GEN_G1, GEN_G2, GEN_G3)

LAYERS = ("abundance", "prob", "prob_raw")
OVERLAP_FRACTION_GATE = 0.01      # 1 % of duplicated cells (DECISIONS 2026-08-24d)
OVERLAP_ABS_FLOOR = 16            # cells; the fraction is a noisy estimator below this


# --------------------------------------------------------------------------- sidecars
def sidecar_paths(map_dir: str | Path) -> dict[str, Path]:
    """Per-tile sidecars in an arm directory, keyed by Murray tile. Manifests excluded."""
    out = {}
    for p in sorted(Path(map_dir).glob("*.json")):
        if p.stem.endswith("manifest"):
            continue
        out[p.stem] = p
    return out


def load_arm(map_dir: str | Path) -> dict[str, dict]:
    """Load every per-tile sidecar in an arm directory."""
    return {t: json.loads(p.read_text(encoding="utf-8"))
            for t, p in sidecar_paths(map_dir).items()}


def sidecar_generation(sc: dict) -> str:
    """Which of the three schema generations wrote this sidecar.

    Keyed off the *presence* of structure, never off a value: a missing key means the
    producer predated the field, and a producer that predates a field cannot vouch for it.
    """
    ov = sc.get("overlap")
    if not isinstance(ov, dict):
        return GEN_G1
    gate = ov.get("gate_layer", "prob_raw")
    if "n_significant" in (ov.get(gate) or {}):
        return GEN_G3
    return GEN_G2


def overlap_status(sc: dict) -> dict:
    """Generation-aware overlap-agreement row. Never reads a missing key as zero.

    ``verdict`` is one of ``pass`` / ``fail`` / ``unknown_on_gate_layer``, and only the
    numbers the sidecar's own generation supports are populated. ``n_dup`` counts coarse
    cells written by more than one window of the 144-window sweep -- R14's premise that the
    sweep partitions the cells was measured on a 36-window sweep and is false for the
    shipped one (62,559-80,570 duplicated cells per tile).
    """
    gen = sidecar_generation(sc)
    row = {"generation": gen, "verdict": "unknown_on_gate_layer", "gate_layer": None,
           "n_dup": None, "n_disagree": None, "fraction": None, "n_significant": None,
           "max_abs": None, "scalar_overlap_disagreements": sc.get("overlap_disagreements"),
           "note": ""}
    if gen == GEN_G1:
        row["note"] = ("scalar counted on the CALIBRATED prob layer; isotonic collapses raw "
                       "fp16 disagreements onto shared knots, so its 0 is not evidence")
        return row

    ov = sc["overlap"]
    layer = ov.get("gate_layer", "prob_raw")
    blk = ov.get(layer) or {}
    row.update(gate_layer=layer, n_dup=blk.get("n_dup"), n_disagree=blk.get("n_disagree"),
               max_abs=blk.get("max_abs"), fraction=blk.get("fraction"))
    if gen == GEN_G3:
        row["n_significant"] = blk.get("n_significant")
        row["note"] = "post-1e-6-floor fraction: the gate quantity as finally defined"
        count = row["n_significant"]
    else:
        row["note"] = ("RAW fraction, any magnitude, pre-1e-6-floor: an UPPER BOUND on the "
                       "gate quantity, not the gate quantity itself")
        count = row["n_disagree"]
    if row["fraction"] is not None and count is not None:
        row["verdict"] = ("pass" if (row["fraction"] <= OVERLAP_FRACTION_GATE
                                     or count < OVERLAP_ABS_FLOOR) else "fail")
    return row


def device_status(sc: dict, *, arm: str) -> dict:
    """Which GPU rendered this tile, and how strongly that is known.

    ⚠ **Absence of ``device`` is arm-conditional and NOT self-identifying.** The field landed
    mid-step-11, so it is missing on the 21 oldest baseline tiles *and* the 7 oldest A1
    tiles -- which are different hardware. The baseline array ran entirely on an RTX 2080 Ti
    (established from the per-window rates in its own logs), while the 7 A1 tiles are the
    Pascal renders (P100 / TITAN Xp) Brian ruled to keep with the mixed provenance recorded.
    So the inference comes from the run logs, not from the sidecar, and this says which.
    See DECISIONS 2026-08-24e.
    """
    dev = (sc.get("run") or {}).get("device")
    if dev:
        return {"device": dev, "device_evidence": "recorded in sidecar",
                "device_inferred": False}
    if arm == "a1":
        return {"device": "Pascal (P100 / TITAN Xp)", "device_inferred": True,
                "device_evidence": "run logs; the field postdates these renders "
                                   "(DECISIONS 2026-08-24e)"}
    return {"device": "RTX 2080 Ti", "device_inferred": True,
            "device_evidence": "run logs; the field postdates these renders "
                               "(DECISIONS 2026-08-24e)"}


def raster_records(sc: dict) -> dict[str, dict]:
    """The sidecar's own ``rasters[]`` record, keyed by layer kind."""
    return {r["kind"]: r for r in (sc.get("rasters") or []) if "kind" in r}


# --------------------------------------------------------------------------- mosaics
def mosaic_footprint(arr: np.ndarray) -> dict:
    """Coverage accounting for a merged regional mosaic.

    Two kinds of nodata are expected and must be told apart from a real hole: the 26 tiles
    form an **L** (the N44 row runs two tiles further east), and the tile *pitch* on the
    global lattice is ~1481.9 cells while each tile raster is 1479, so thin NaN **seams**
    run between adjacent tiles where the straddling coarse cells belong to neither. This
    quantifies both so a genuine hole cannot hide inside "some nodata is normal".
    """
    finite = np.isfinite(arr)
    n = int(arr.size)
    n_fin = int(finite.sum())
    out = {"shape": list(arr.shape), "n_cells": n, "n_finite": n_fin,
           "finite_fraction": (n_fin / n) if n else float("nan"),
           "n_nodata": n - n_fin,
           "rows_all_nodata": int((~finite.any(axis=1)).sum()),
           "cols_all_nodata": int((~finite.any(axis=0)).sum())}
    if n_fin:
        out.update(value_min=float(np.nanmin(arr)), value_max=float(np.nanmax(arr)),
                   value_mean=float(np.nanmean(arr)))
    return out


def seam_widths(arr: np.ndarray, *, max_width: int = 8) -> dict:
    """Census of INTERIOR NaN run lengths along rows -- the inter-tile seam width check.

    A run counts only if finite cells bound it on both sides within the same row, so the
    L-shaped corner and the outside margin are excluded by construction. Anything wider
    than ``max_width`` is bucketed into ``gt_max``: that is the bucket a real hole lands in.
    """
    finite = np.isfinite(arr)
    hist: dict[int, int] = {}
    for r in range(arr.shape[0]):
        f = finite[r]
        if not f.any():
            continue
        lo = int(np.argmax(f))
        hi = int(len(f) - np.argmax(f[::-1]))
        run = 0
        for v in f[lo:hi]:
            if v:
                if run:
                    hist[run] = hist.get(run, 0) + 1
                run = 0
            else:
                run += 1
    out: dict = {k: v for k, v in sorted(hist.items()) if k <= max_width}
    over = sum(v for k, v in hist.items() if k > max_width)
    if over:
        out["gt_max"] = over
        out["widest"] = max(hist)
    return out


def difference_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """``b - a`` (A1 minus baseline) on the cells both arms cover.

    Cell-for-cell differencing is legitimate only because ``scripts/verify_arm_parity.py``
    established one lattice and cell-for-cell co-registration across the arms; the shape
    assertion here keeps a silently mis-shaped input from being broadcast into a result.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}: the arms are not "
                         "differenceable -- re-run scripts/verify_arm_parity.py")
    both = np.isfinite(a) & np.isfinite(b)
    n = int(both.sum())
    out = {"n_common": n,
           "only_a": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
           "only_b": int((np.isfinite(b) & ~np.isfinite(a)).sum())}
    if not n:
        return out
    d = (b[both].astype(np.float64) - a[both].astype(np.float64))
    out.update(mean=float(d.mean()),
               sd=float(d.std(ddof=1)) if n > 1 else float("nan"),
               median=float(np.median(d)),
               p01=float(np.percentile(d, 1)), p99=float(np.percentile(d, 99)),
               min=float(d.min()), max=float(d.max()), max_abs=float(np.abs(d).max()),
               frac_nonzero=float((d != 0).mean()))
    return out
