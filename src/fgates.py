"""PLAN_FBuild §5 acceptance-gate scorers (Stage D), plus the §5.1 comparison metrics.

Gate 1 is the one that needed a methodology ruling. As written ("partition η² over frame footprints
≤ ~0.05 on the full block") it is not interpretable at 907-frame scale, because η² has no
group-count correction and grows mechanically with frame count/footprint. Measured on the EXISTING
mosaic-path map, 2026-07-28:

    scope                     partition η²    its own rotation-null p95
    per 4° tile (median)          0.185              0.127
    merged 26-tile block          0.3575             0.3189   (79% of it is reproduced by rolling)
    detrended (σ=30 px)           0.0123             0.0023

So the literal bar sits *below* the geological floor (nothing can pass), while the detrended version
is already passed by the un-mitigated map (nothing can fail). The 0.05 bar was calibrated on a ~75 km
/ 7-frame crop, where the mosaic scores 0.1948 against a null of 0.083-0.117.

**Brian ruled 2026-07-28:** headline = partition η² on pilot-scale WINDOWS, each with its own
rotation null, reported as a distribution across windows with the bar applied to the median window;
alongside it the full-block number reported floor-relative (η² − null_mean, η²/null_p95). Both are
computed here, for every map row, on one grid and one quantity (raw P(rich)).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import fcompose as fc
from src import leveling as lv
from src import striping as st

REPO = Path(__file__).resolve().parents[1]
WINDOW_PX = 469          # ~75 km at 160 m/px — the E8_N44 pilot crop's coarse size (gate 1's scale)
ETA2_BAR = 0.05          # the pre-declared reopening bar, applied to the MEDIAN window
NULL_DRAWS = 20          # rotation-null draws per window (~0.5 s each at tile scale)
THEMIS_TOL = 0.02        # gate 3 "not degraded" (mirrors scripts/f_h4_themis.GATE_TOL)
SKILL_TOL = -0.02        # gate 5 delta tolerance
TOP_RATIO_BAND = (0.8, 1.2)     # gate 6, inherited from PLAN_Calibration §6


# --------------------------------------------------------------------------- gate 1
@dataclass
class WindowScore:
    tile: str
    r0: int
    c0: int
    n_cells: int
    n_frames: int
    eta2: float
    null_mean: float
    null_p95: float

    @property
    def excess(self) -> float:
        return self.eta2 - self.null_mean

    @property
    def ratio(self) -> float:
        return self.eta2 / self.null_p95 if self.null_p95 > 0 else np.nan


def eta2_with_null(values: np.ndarray, labels: np.ndarray, *, n_draws: int = NULL_DRAWS,
                   seed: int = 0, min_cells: int = 2000, min_frames: int = 2):
    """(η², null_mean, null_p95, n_cells, n_frames) over one array, or NaNs if too thin.

    `finite` is passed as isfinite(values) & (labels >= 0) — the convention every caller in this repo
    uses — and the rotation null is `src.striping.eta2_rotation_null`, which rolls the VALUE field
    under the FIXED frame mask, so it preserves both the block geometry and the field's own spatial
    autocorrelation while breaking the frame-geology alignment. That is what makes it a geological
    floor rather than a white-noise reference.
    """
    fin = np.isfinite(values) & (labels >= 0)
    n_cells = int(fin.sum())
    n_frames = int(np.unique(labels[fin]).size) if n_cells else 0
    if n_cells < min_cells or n_frames < min_frames:
        return np.nan, np.nan, np.nan, n_cells, n_frames
    e = st.eta2(values, labels, fin)
    nm, n95 = st.eta2_rotation_null(values, labels, fin, n=n_draws, seed=seed)
    return float(e), float(nm), float(n95), n_cells, n_frames


def window_eta2(values: np.ndarray, labels: np.ndarray, tile: str, *, win_px: int = WINDOW_PX,
                n_draws: int = NULL_DRAWS, seed: int = 0) -> list[WindowScore]:
    """Gate 1 headline: partition η² per pilot-scale window, each against its OWN rotation null."""
    out = []
    h, w = values.shape
    for r0 in range(0, h, win_px):
        for c0 in range(0, w, win_px):
            r1, c1 = min(r0 + win_px, h), min(c0 + win_px, w)
            if (r1 - r0) * (c1 - c0) < 0.5 * win_px * win_px:
                continue
            v = values[r0:r1, c0:c1]
            lab = labels[r0:r1, c0:c1]
            e, nm, n95, nc, nf = eta2_with_null(v, lab, n_draws=n_draws, seed=seed)
            if not np.isfinite(e):
                continue
            out.append(WindowScore(tile, r0, c0, nc, nf, e, nm, n95))
    return out


def summarize_windows(scores: list[WindowScore]) -> dict:
    if not scores:
        return {"n_windows": 0, "eta2_median": np.nan, "eta2_p90": np.nan,
                "null_mean_median": np.nan, "null_p95_median": np.nan,
                "excess_median": np.nan, "ratio_median": np.nan, "passes_bar": False}
    e = np.array([s.eta2 for s in scores])
    nm = np.array([s.null_mean for s in scores])
    n95 = np.array([s.null_p95 for s in scores])
    ex = np.array([s.excess for s in scores])
    ra = np.array([s.ratio for s in scores])
    med = float(np.median(e))
    return {"n_windows": len(scores), "eta2_median": med, "eta2_p90": float(np.percentile(e, 90)),
            "null_mean_median": float(np.median(nm)), "null_p95_median": float(np.median(n95)),
            "excess_median": float(np.median(ex)), "ratio_median": float(np.median(ra)),
            "frac_windows_below_bar": float((e <= ETA2_BAR).mean()),
            "passes_bar": bool(med <= ETA2_BAR)}


# --------------------------------------------------------------------------- gate 2
def edge_cv_for_offsets(edges, offsets: np.ndarray, n: int, lam: float, *, variant: str = "full",
                        frac: float = 0.05, repeats: int = 4, seed: int = 0, metric: str = "dp",
                        lon=None, lat=None, degree=None) -> dict:
    """Gate 2: held-out-edge disagreement for the offset MODEL the map actually uses.

    The held-out number must be **variant-aware**. An earlier version delegated to
    `lv.heldout_edge_cv`, which re-solves FULL offsets on each fold and never sees `offsets` — so the
    headline was byte-identical for h1only / full / resid, and the H1-only row (which applies no
    offsets at all, and whose held-out disagreement IS the unleveled baseline by construction) was
    reported as clearing the gate. Caught by the 2026-07-29 adversarial review; that is exactly the
    failure this docstring used to claim it prevented.

    So the fold loop lives here and rebuilds the variant's own offsets per fold:
      * `h1only` — no offsets: the held-out value IS the baseline, and `passes` is False by
        construction (there is nothing to generalise).
      * `full` / `lcv` — solve on the retained edges at this variant's λ, score the held-out ones.
      * `resid` — solve on the retained edges, then subtract the degree-weighted lon/lat plane refit
        **on that fold's own solve** before scoring (the residual-only model is "solve minus its
        smooth component", so the plane has to be refit inside the fold or the CV leaks).

    `metric` is "dp" (probability space, pre-registered) or "dlogit" (saturation-immune; see
    `lv.edge_dlogit` — in probability space the statistic is minimised by railing the sigmoid).
    """
    metric_fn = lv.EDGE_METRICS[metric] if hasattr(lv, "EDGE_METRICS") else lv.edge_dp
    base = float(np.median(metric_fn(edges, np.zeros(n))))
    insample = float(np.median(metric_fn(edges, np.asarray(offsets, dtype=float))))
    out = {"variant": variant, "metric": metric, "unleveled_dp": base, "insample_dp": insample}
    if variant == "h1only":
        return {**out, "heldout_cv_dp": base, "cv_edges_skipped": 0, "passes": False,
                "note": "unleveled by construction — the held-out value IS the baseline"}
    if variant == "resid" and (lon is None or lat is None):
        print("  ⚠ gate 2 resid: no lon/lat given, cannot refit the fold plane -> scoring the FULL "
              "offset model instead (the number is NOT the residual-only variant's)", flush=True)
        variant = "full"

    rng = np.random.default_rng(seed)
    m = edges.n_edges
    k = max(1, int(round(frac * m)))
    dps, skipped = [], 0
    for _ in range(repeats):
        hold = rng.choice(m, size=min(k, m), replace=False)
        mask = np.ones(m, dtype=bool)
        mask[hold] = False
        comp = lv.components(edges.ei[mask], edges.ej[mask], n)
        o = lv.solve_offsets(edges, lam, n, comp=comp, edge_mask=mask)
        if variant == "resid":
            w = None if degree is None else np.asarray(degree, dtype=float)
            fitted = lv.weighted_fit(lv.design_matrix(lon, lat, order=1), o, w)[1]
            o = lv.regauge(o - fitted, comp)
        ok = comp[edges.ei[hold]] == comp[edges.ej[hold]]
        skipped += int((~ok).sum())
        if ok.any():
            dps.append(metric_fn(edges, o, hold[ok]))
    cv = float(np.median(np.concatenate(dps))) if dps else float("nan")
    return {**out, "heldout_cv_dp": cv, "cv_edges_skipped": skipped,
            "passes": bool(np.isfinite(cv) and cv < base)}


# --------------------------------------------------------------------------- gate 3
def themis_on_grid(grid: fc.TileGrid, themis_path: Path) -> np.ndarray:
    """Reproject the cached THEMIS night-IR raster onto a tile grid (gate 3's co-registration)."""
    import rasterio
    from rasterio.transform import Affine

    from src import validation_retrieve as vr

    with rasterio.open(themis_path) as ds:
        arr = ds.read(1).astype(np.float32)
        src_tf, src_crs, src_nd = ds.transform, ds.crs.to_wkt(), ds.nodata
    return vr.reproject_to_grid(arr, src_tf, src_crs, dst_crs_wkt=grid.crs_wkt,
                                dst_transform=Affine(*grid.transform), dst_shape=grid.shape,
                                resampling="bilinear", src_nodata=src_nd)


def spearman_rho(a: np.ndarray, b: np.ndarray, min_n: int = 50) -> tuple[float, int]:
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < min_n:
        return np.nan, n
    return float(spearmanr(a[m], b[m]).statistic), n


# --------------------------------------------------------------------------- gates 5 & 6 support
def cohort_tiles_to_global(labels: pd.DataFrame) -> pd.DataFrame:
    """Join the labelled HiRISE cohort tiles to Stage B's GLOBAL (TI, TJ) keys.

    Keyed off the **world bbox the label rows already carry** (`xmin/xmax/ymin/ymax`), which is the
    only self-consistent source: `TI = round(y_centre/160)`, `TJ = round(x_centre/160)` — Stage B's
    exact keying (`f_region_stageb.py:167-168`).

    An earlier version reconstructed the world position from `(ti, tj)` and
    `reports/f_leg_b/cohort_obs_bounds.csv`'s window corner, on the premise that `ti/tj` are indices
    in each observation's own window grid. **That premise is false** — `src/labeling.py:363-370`
    emits them on the lattice anchored at the PARENT Murray tile's `inner_transform` origin, i.e. the
    window offset is already baked in (`dataset/DATA_DICTIONARY.md:184`; every other consumer, e.g.
    `src/features.py:653`, subtracts `mosaic_row_origin` to get back to window coords). Anchoring
    already-absolute indices at the window corner added `(col0, row0)·4.99997 m` on top: measured
    displacement over all 38 obs was a **median 94.7 km in x / 108.8 km in y** (ESP_017355_2260:
    +19,065 m / −106,915 m). It failed silently — ~79k mis-keyed rows still landed inside a block tile
    on finite pixels — so gates 5 and 6 were pairing labels with predictions ~100 km apart and
    publishing plausible numbers (on E16_N44: pooled pr_auc 0.544 / Spearman −0.180 mis-keyed vs
    0.939 / +0.791 correct). Found by the 2026-07-29 adversarial review; the old unit test could not
    catch it because it pinned `row0=col0=0` and re-derived the expectation from the same formula.

    Using the bbox also drops the `32 * 5.0 = 160.0` pitch approximation — label tiles are actually
    32 × 4.9999744853063 = 159.9992 m wide.
    """
    need = {"xmin", "xmax", "ymin", "ymax"}
    if not need <= set(labels.columns):
        raise ValueError(f"labels need the world bbox columns {sorted(need)} (read them from "
                         f"dataset_v2/labels/*.parquet); has {sorted(labels.columns)}")
    cx = (labels["xmin"].to_numpy(float) + labels["xmax"].to_numpy(float)) / 2.0
    cy = (labels["ymin"].to_numpy(float) + labels["ymax"].to_numpy(float)) / 2.0
    return labels.assign(TI=np.round(cy / lv.TILE_M).astype(np.int64),
                         TJ=np.round(cx / lv.TILE_M).astype(np.int64))


def pooled_skill(y: np.ndarray, p: np.ndarray) -> dict:
    """Pooled pr_auc@1e-2 + precision@5% exactly as `scripts/f_h4_legb.pooled_metrics` does.

    Reimplemented here (rather than imported) only so `src/` carries no dependency on a script, but
    the conventions are copied verbatim: ONE average_precision_score over the pooled vector, and
    k = max(1, round(0.05*N)) with np.argsort (not argpartition — the tie order differs and every
    number of record used argsort). NO presence AUC anywhere (project rule).
    """
    from sklearn.metrics import average_precision_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0 or np.unique(y).size < 2:
        return {"pooled_pr_auc": np.nan, "precision@5%": np.nan, "n": int(y.size)}
    k = max(1, int(round(0.05 * p.size)))
    top = np.argsort(-p)[:k]
    return {"pooled_pr_auc": float(average_precision_score(y, p)),
            "precision@5%": float(y[top].mean()), "n": int(y.size)}


def abundance_fidelity(fa_true: np.ndarray, ab_pred: np.ndarray) -> dict:
    """Gate 6: marginal-L1 + top_ratio (compression_metrics) AND per-bin RMSE, on paired data.

    `compression_metrics` is NOT NaN-safe (np.quantile/spearmanr propagate) and its `low_over` key is
    degenerate on this dataset (the truly-zero bin's mean_true is exactly 0, so the denominator hits
    the 1e-9 floor and it reads ~2e6) — so a common finite mask is applied here and low_over is
    dropped. Per-bin RMSE is a separate call: it is not one of compression_metrics' six keys.
    """
    from src.calibration import compression_metrics
    from src.modeling.evaluate import per_bin_rmse

    keys = ("spearman", "top_ratio", "near_zero_pred", "near_zero_true", "marginal_l1",
            "rich_bin_rmse")
    m = np.isfinite(fa_true) & np.isfinite(ab_pred)
    if m.sum() < 100:
        # Return the FULL key set (NaN-filled) — gate 6 selects these columns for its table, so a
        # short dict here raised KeyError after the CSV had already been written (review 2026-07-29).
        return {"n": int(m.sum()), **{k: np.nan for k in keys}, "passes_top_ratio": False,
                "per_bin": pd.DataFrame(), "note": "too few paired finite cells to score"}
    yt, yp = np.asarray(fa_true, float)[m], np.asarray(ab_pred, float)[m]
    cm = {k: v for k, v in compression_metrics(yt, yp).items() if k != "low_over"}
    bins = per_bin_rmse(yt, yp)
    rich = bins[bins["bin"] == "1e-2_to_max"]
    return {"n": int(m.sum()), **cm,
            "rich_bin_rmse": float(rich["rmse"].iloc[0]) if len(rich) else np.nan,
            "per_bin": bins,
            "passes_top_ratio": bool(TOP_RATIO_BAND[0] <= cm.get("top_ratio", np.nan)
                                     <= TOP_RATIO_BAND[1])}


# --------------------------------------------------------------------------- shared raster helpers
def read_layer(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        nd = ds.nodata
    if nd is not None and np.isfinite(nd):
        a[a == nd] = np.nan
    return a


def common_finite(*arrays: np.ndarray) -> np.ndarray:
    """One mask over every row of a comparison, so a coverage difference can never masquerade as a
    metric difference (§5.1 requires all rows on the same footprint)."""
    m = np.isfinite(arrays[0])
    for a in arrays[1:]:
        m &= np.isfinite(a)
    return m
