"""PLAN_FBuild Stage C — the H4 solve on the full 907-frame overlap graph + the trend guard.

Consumes Stage B's per-frame `{PRODUCT_ID}.npz` ({TI, TJ, prob} on the global 160 m tile grid) and
emits the per-frame additive **logit offsets** o_f that Stage D applies at composite time. Stage C
writes no rasters: the offset TABLE is the deliverable, which is what makes the H1-only (o=0),
full-offset and residual-only composites all reproducible from one Stage-B run (PLAN_FBuild §1.5).

Steps (all pre-declared in PLAN_FBuild §4 before any build offset was seen):
  1. load npzs + census against `region_frame_list.csv` (V4: components with failed frames removed);
  2. overlap edges — exact (δ̄_ij, W_ij) over co-located global tiles, cached to
     `reports/f_stagec/stagec_edges.npz` so the solve can be re-run/re-tuned in seconds;
  3. λ sweep, λ* picked by **held-out-edge CV** (§4: random 5% edge sample — the non-circular
     instrument; post-H4 η² alone would be circular, which is what killed option D);
  4. solve 907 offsets (weighted LS + λ·Σo², gauge median(o)=0 per component);
  5. leave-one-FRAME-out generalization + the §0.1 guard-3 "under-pinned large offset" watchlist;
  6. **trend guard** — weighted lon/lat surface fit, block-permutation significance, and attribution
     against metadata (epoch / incidence / residual radiometric level) vs geology proxies
     (MOLA elevation, THEMIS night-IR) → the §4.3 rule-table verdict (full vs residual-only);
  7. §0.1 guard-4 mean-flattening diagnostic: corr(offset, frame-mean P(rich)).

Run (laptop after `tar`-ing the npzs home, or on a Sherlock login node next to them — pure
numpy/scipy, no GPU, ~minutes):
  conda run --no-capture-output -n geospatial python -u scripts/f_region_stagec.py \
      --logits-dir reports/f_region_logits
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP/DLL bootstrap; must precede numpy/pandas

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import leveling as lv

FIG = REPO / "reports" / "figures"
WORK = REPO / "reports" / "f_stagec"
FRAME_LIST = FIG / "region_frame_list.csv"
INC_CSV = FIG / "region_frame_incidence.csv"
MOLA = REPO / "cache_v2" / "validation" / "mola_dem_region.tif"
THEMIS = REPO / "cache_v2" / "validation" / "themis_night_ir_region.tif"

# λ is scaled by the median edge weight (= co-located tile count): the pilot's absolute λ*=300 sat
# at ≈0.1·median(W) there, and build-scale edges carry 10-1000x more tiles, so an absolute grid
# would be meaningless. Fractions bracket the pilot's operating point by 3 decades either side.
LAMBDA_FRACS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
MIN_TILES_GRID = [50, 100, 200, 500, 1000]     # sensitivity table around the pilot's 200
BIG_OFFSET = 1.0                                # §0.1 guard 3/4 "large offset" threshold (logit)


# --------------------------------------------------------------------------- 1. load
def load_logits(logits_dir: Path, max_frames: int = 0):
    """Per-frame sorted (global tile key, logit) + the Stage-B json metadata."""
    npzs = sorted(logits_dir.glob("*.npz"))
    if max_frames:
        npzs = npzs[:max_frames]
    if not npzs:
        raise SystemExit(f"no *.npz in {logits_dir} — Stage B must run first")
    pids, keys, logits, meta = [], [], [], []
    n_bytes = 0
    for p in npzs:
        z = np.load(p)
        ti, tj, prob = z["TI"], z["TJ"], z["prob"]
        if ti.size == 0:
            print(f"  ⚠ {p.stem}: 0 tiles -> excluded (Stage-A/B hole; H6-flag)", flush=True)
            continue
        key = lv.pack_key(ti, tj)
        order = np.argsort(key, kind="stable")
        key, prob = key[order], prob[order]
        uniq, first = np.unique(key, return_index=True)     # Stage B dedups, but never assume
        if uniq.size != key.size:
            key, prob = uniq, prob[first]
        pids.append(p.stem)
        keys.append(key)
        logits.append(lv.logit(prob).astype(np.float32))
        n_bytes += key.nbytes + logits[-1].nbytes
        jf = p.with_suffix(".json")
        meta.append(json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else {"PRODUCT_ID": p.stem})
    print(f"loaded {len(pids)} frames, {sum(k.size for k in keys):,} tiles "
          f"({n_bytes / 1e9:.2f} GB resident)", flush=True)
    return pids, keys, logits, pd.DataFrame(meta).set_index("PRODUCT_ID")


def frame_table(pids, meta) -> pd.DataFrame:
    """Per-frame covariates: geometry (centers), epoch, incidence, H1 centering statistic."""
    fl = pd.read_csv(FRAME_LIST).set_index("PRODUCT_ID") if FRAME_LIST.exists() else pd.DataFrame()
    inc = pd.read_csv(INC_CSV).set_index("PRODUCT_ID") if INC_CSV.exists() else pd.DataFrame()
    rows = []
    for pid in pids:
        r = {"PRODUCT_ID": pid}
        for src, cols in ((inc, ("incidence", "center_lat", "center_lon", "subsolar_lat")),
                          (fl, ("image_time",))):
            for c in cols:
                r[c] = src.loc[pid, c] if (len(src) and pid in src.index and c in src.columns) else np.nan
        m = meta.loc[pid] if pid in meta.index else {}
        r["frame_median"] = float(m.get("frame_median", np.nan)) if len(m) else np.nan
        r["n_tiles"] = int(m.get("n_tiles", 0)) if len(m) else 0
        r["prob_mean"] = float(m.get("prob_mean", np.nan)) if len(m) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows).set_index("PRODUCT_ID")
    # lon in [-180, 180): the circum-Chryse block straddles the prime meridian (348°..11°)
    df["lon"] = ((df["center_lon"].astype(float) + 180.0) % 360.0) - 180.0
    df["lat"] = df["center_lat"].astype(float)
    t = pd.to_datetime(df["image_time"], errors="coerce", utc=True)
    df["epoch_year"] = t.dt.year + (t.dt.dayofyear - 1) / 365.25
    df["ln_frame_median"] = np.log(df["frame_median"].astype(float).where(lambda s: s > 0))
    return df


def census(pids) -> pd.DataFrame:
    """V4 — Stage-A/B failure census: which of the 907 planned frames actually produced logits."""
    if not FRAME_LIST.exists():
        return pd.DataFrame()
    planned = list(pd.read_csv(FRAME_LIST)["PRODUCT_ID"])
    have = set(pids)
    missing = [p for p in planned if p not in have]
    extra = [p for p in pids if p not in set(planned)]
    print(f"census: {len(have)}/{len(planned)} planned frames have logits; "
          f"{len(missing)} missing, {len(extra)} unplanned", flush=True)
    if missing:
        print(f"  missing (first 12): {missing[:12]}", flush=True)
        print(f"  ⚠ Stage B incomplete or Stage-A holes — offsets solve on the frames present; "
              f"re-run Stage C after `sbatch run_f_region_stageb.sbatch` finishes.", flush=True)
    return pd.DataFrame({"PRODUCT_ID": missing, "status": "no_logits"})


# --------------------------------------------------------------------------- 2. edges
def get_edges(pids, keys, logits, args) -> lv.EdgeSet:
    cache = WORK / f"stagec_edges_min{args.cache_min_tiles}.npz"
    if cache.exists() and not args.rebuild_edges:
        es = lv.EdgeSet.load(cache)
        if es.pids == list(pids):
            print(f"edges: loaded cache {cache} ({es.n_edges} edges >= {args.cache_min_tiles} tiles)",
                  flush=True)
            return es
        print("edges: cache frame list differs from the loaded npzs -> rebuilding", flush=True)
    t0 = time.monotonic()
    pairs = lv.candidate_pairs(keys, cell_tiles=args.cell_tiles)
    print(f"edges: {len(pairs):,} candidate pairs from coarse-cell co-occurrence "
          f"(of {len(pids) * (len(pids) - 1) // 2:,} possible)", flush=True)
    es = lv.build_edges(pids, keys, logits, min_tiles=args.cache_min_tiles,
                        dp_sample=args.dp_sample, seed=args.seed, pairs=pairs, progress=2000)
    WORK.mkdir(parents=True, exist_ok=True)
    es.save(cache)
    print(f"edges: {es.n_edges:,} with >= {args.cache_min_tiles} co-located tiles "
          f"({time.monotonic() - t0:.0f}s) -> {cache}", flush=True)
    return es


def graph_report(es: lv.EdgeSet, n: int, min_tiles: int) -> pd.DataFrame:
    """V4 census + how sensitive the graph is to where the min-shared-tiles cut is drawn.

    Works on (ei, ej, w) directly rather than `EdgeSet.filter` — the sensitivity sweep does not need
    the per-edge |Δp| sample store, and copying it six times would move ~100 MB per row for nothing.
    """
    rows = []
    for mt in sorted(set(MIN_TILES_GRID) | {min_tiles}):
        keep = es.w >= mt
        ei, ej, w = es.ei[keep], es.ej[keep], es.w[keep]
        comp = lv.components(ei, ej, n)
        deg = np.zeros(n, dtype=np.int64)
        np.add.at(deg, ei, 1)
        np.add.at(deg, ej, 1)
        sizes = np.bincount(comp)
        rows.append({"min_shared_tiles": mt, "n_edges": int(keep.sum()),
                     "n_components": int(sizes.size), "largest_comp": int(sizes.max()),
                     "isolated_frames": int((deg == 0).sum()),
                     "median_degree": float(np.median(deg)), "max_degree": int(deg.max()),
                     "median_shared_tiles": float(np.median(w)) if w.size else np.nan,
                     "is_operating_point": mt == min_tiles})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3-4. λ and solve
def lambda_sweep(es: lv.EdgeSet, n: int, comp, args) -> tuple[pd.DataFrame, float, np.ndarray, float]:
    med_w = float(np.median(es.w))
    base = np.zeros(n)
    base_dp = float(np.median(lv.edge_dp(es, base)))
    print(f"unleveled in-sample median |Δp| over {es.n_edges} edges: {base_dp:.4f} "
          f"(pilot unleveled 0.0738)", flush=True)
    rows, solved = [], {}
    for frac in LAMBDA_FRACS:
        lam = frac * med_w
        o = lv.solve_offsets(es, lam, n, comp=comp)
        solved[frac] = o
        insamp = float(np.median(lv.edge_dp(es, o)))
        cv, skipped = lv.heldout_edge_cv(es, lam, n, frac=args.cv_frac, repeats=args.cv_repeats,
                                         seed=args.seed)
        rows.append({"lambda_frac_medW": frac, "lambda": round(lam, 3),
                     "insample_dp": round(insamp, 4), "heldout_cv_dp": round(cv, 4),
                     "cv_edges_skipped": skipped, "max_abs_offset": round(float(np.abs(o).max()), 3),
                     "sd_offset": round(float(np.std(o)), 4)})
        print(f"  λ={lam:>10.1f} ({frac:g}·medW): in|Δp| {insamp:.4f}  heldout|Δp| {cv:.4f}  "
              f"|o|max {np.abs(o).max():.3f}", flush=True)
    df = pd.DataFrame(rows)
    if not np.isfinite(df["heldout_cv_dp"]).any():
        # every held-out edge spanned two components of the fit graph -> the CV instrument is
        # undefined on this graph. Refuse to pick λ silently (gate 2 rests on this number).
        raise SystemExit("held-out edge CV is undefined (all held-out edges disconnected the "
                         "gauge). The graph is too sparse — lower --min-tiles or check Stage B "
                         "coverage before trusting any λ.")
    # pick by held-out CV; tie -> larger λ (more regularized / conservative), as the pilot did
    best = df.sort_values(["heldout_cv_dp", "lambda"], ascending=[True, False]).iloc[0]
    return df, float(best["lambda"]), solved[float(best["lambda_frac_medW"])], base_dp


# --------------------------------------------------------------------------- 6. geology proxies
def sample_raster_per_frame(path: Path, keys, max_pts: int = 2000, seed: int = 0) -> np.ndarray:
    """Mean raster value over each frame's tile footprint (both rasters are already on the CTX CRS).

    Global tile (TI,TJ) -> CTX world (x, y) = (TJ, TI)·160 m is the exact inverse of Stage B's
    keying, so this needs no reprojection — only the raster's own inverse affine.
    """
    import rasterio

    out = np.full(len(keys), np.nan)
    rng = np.random.default_rng(seed)
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        nd, inv = ds.nodata, ~ds.transform
        H, W = arr.shape
    if nd is not None:
        arr[arr == nd] = np.nan
    for f, k in enumerate(keys):
        if k.size == 0:
            continue
        sel = rng.choice(k.size, size=min(max_pts, k.size), replace=False)
        ti, tj = lv.unpack_key(k[sel])
        col, row = inv * (tj * lv.TILE_M, ti * lv.TILE_M)
        col = np.round(np.asarray(col)).astype(int)
        row = np.round(np.asarray(row)).astype(int)
        ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        if ok.any():
            v = arr[row[ok], col[ok]]
            if np.isfinite(v).any():
                out[f] = float(np.nanmean(v))
    return out


def geology_axes(keys, args) -> dict[str, np.ndarray]:
    axes = {}
    for name, path, flag in (("mola_elev", Path(args.mola), "--mola"),
                             ("themis_night_ir", Path(args.themis), "--themis")):
        if path.exists():
            t0 = time.monotonic()
            axes[name] = sample_raster_per_frame(path, keys, seed=args.seed)
            print(f"  {name}: sampled per frame ({np.isfinite(axes[name]).sum()} valid, "
                  f"{time.monotonic() - t0:.0f}s)", flush=True)
        else:
            print(f"  ⚠ {path} missing -> geology axis '{name}' unavailable "
                  f"(fetch with scripts/fetch_validation_data.py, or pass {flag})", flush=True)
    return axes


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", default=str(REPO / "reports" / "f_region_logits"))
    ap.add_argument("--min-tiles", type=int, default=200, help="edges used in the solve (pilot: 200)")
    ap.add_argument("--cache-min-tiles", type=int, default=50, help="edges kept in the cache (for the sensitivity table)")
    ap.add_argument("--dp-sample", type=int, default=1000, help="co-located logit pairs kept per edge for |Δp|")
    ap.add_argument("--cell-tiles", type=int, default=64, help="coarse cell for the candidate-pair prescreen")
    ap.add_argument("--cv-frac", type=float, default=0.05, help="held-out edge fraction (§4)")
    ap.add_argument("--cv-repeats", type=int, default=4)
    ap.add_argument("--perm-draws", type=int, default=1000, help="block-permutation draws (§4.2)")
    ap.add_argument("--cell-deg", type=float, default=4.0, help="permutation block size (§4.2)")
    ap.add_argument("--lofo-n", type=int, default=0, help="frames for leave-one-frame-out (0 = all)")
    ap.add_argument("--mola", default=str(MOLA))
    ap.add_argument("--themis", default=str(THEMIS))
    ap.add_argument("--max-frames", type=int, default=0, help="debug: cap frames loaded")
    ap.add_argument("--rebuild-edges", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    logits_dir = Path(args.logits_dir)

    # 1 -------------------------------------------------------------- load + census
    pids, keys, logits, meta = load_logits(logits_dir, args.max_frames)
    n = len(pids)
    miss = census(pids)
    if len(miss):
        miss.to_csv(FIG / "fbuild_stagec_missing_frames.csv", index=False)
    ft = frame_table(pids, meta)

    # 2 -------------------------------------------------------------- overlap graph
    es_all = get_edges(pids, keys, logits, args)
    graph = graph_report(es_all, n, args.min_tiles)
    graph.to_csv(FIG / "fbuild_stagec_graph.csv", index=False)
    print("\n=== overlap graph (V4 component census; sensitivity to the min-shared-tiles cut) ===")
    print(graph.to_string(index=False), flush=True)

    es = es_all.filter(es_all.w >= args.min_tiles)
    if es.n_edges == 0:
        raise SystemExit(f"no edges with >= {args.min_tiles} co-located tiles — check Stage B keying "
                         f"(scripts/f_region_colocheck.py)")
    comp = lv.components(es.ei, es.ej, n)
    deg = es.degrees(n)
    n_comp = int(np.unique(comp).size)
    if n_comp > 1:
        print(f"⚠ V4: {n_comp} components ({int((deg == 0).sum())} isolated frames) -> each solved on "
              f"its own gauge, then reconciled onto the main one by IDW (`offset_source` column)",
              flush=True)

    # 3-4 ------------------------------------------------------------ λ sweep + solve
    print(f"\n=== λ sweep ({es.n_edges} edges >= {args.min_tiles} tiles, {n} frames) ===", flush=True)
    lam_df, lam_star, o_star, base_dp = lambda_sweep(es, n, comp, args)
    lam_df.to_csv(FIG / "fbuild_stagec_lambda.csv", index=False)
    star = lam_df.loc[np.isclose(lam_df["lambda"], lam_star)].iloc[0]
    print(f"\nPICK λ*={lam_star:.1f}: held-out |Δp| {star['heldout_cv_dp']:.4f} vs unleveled "
          f"{base_dp:.4f}  (gate 2: materially below)", flush=True)

    # §4 end — make every frame's offset comparable to the main component's gauge (or flag it).
    lon, lat = ft["lon"].to_numpy(float), ft["lat"].to_numpy(float)
    o_star, off_src = lv.patch_graph_holes(o_star, comp, deg, lon, lat)
    n_patched = int((off_src != "solved").sum())
    if n_patched:
        print(f"⚠ {n_patched} frames not solved on the main gauge: "
              f"{dict(zip(*np.unique(off_src[off_src != 'solved'], return_counts=True)))} "
              f"-> H6 offset-provenance flag (PLAN_FBuild §1 deliverable 2)", flush=True)
    dp_star = lv.edge_dp(es, o_star)

    # 5 -------------------------------------------------------------- LOFO + watchlist
    lofo_frames = None
    if args.lofo_n:
        order = np.argsort(-np.abs(o_star))
        lofo_frames = np.unique(np.concatenate([order[:args.lofo_n // 2],
                                                np.random.default_rng(args.seed).choice(
                                                    n, min(args.lofo_n // 2, n), replace=False)]))
    t0 = time.monotonic()
    o_hat, pred_err, n_used = lv.lofo_offsets(es, o_star, lam_star, n, frames=lofo_frames)
    print(f"\nLOFO: {int(np.isfinite(o_hat).sum())} frames, median |predicted−full| "
          f"{np.nanmedian(pred_err):.3f} ({time.monotonic() - t0:.0f}s)", flush=True)

    # 6 -------------------------------------------------------------- trend guard (§4.2/§4.3)
    wdeg = deg.astype(float)                                  # §4.1: weight the surface fit by degree
    print(f"\n=== trend guard ({args.perm_draws} block-permutation draws, {args.cell_deg}° cells) ===",
          flush=True)
    tr1 = lv.trend_significance(o_star, lon, lat, w=wdeg, order=1, cell_deg=args.cell_deg,
                                n_draws=args.perm_draws, seed=args.seed)
    tr2 = lv.trend_significance(o_star, lon, lat, w=wdeg, order=2, cell_deg=args.cell_deg,
                                n_draws=args.perm_draws, seed=args.seed)
    for nm, tr in (("linear", tr1), ("quadratic", tr2)):
        print(f"  {nm:>9} surface: R² {tr['r2']:.3f}  p {tr['p_value']:.4f}  "
              f"(null median {tr['null_median_r2']:.3f}, p95 {tr['null_p95_r2']:.3f}, "
              f"{tr['n_blocks']} blocks)", flush=True)
    smooth = tr1["fitted"]
    resid = o_star - smooth
    o_resid = lv.regauge(resid, comp)

    print("  attribution axes:", flush=True)
    axes = {"incidence": ft["incidence"].to_numpy(float),
            "epoch_year": ft["epoch_year"].to_numpy(float),
            "subsolar_lat": ft["subsolar_lat"].to_numpy(float),
            "ln_frame_median": ft["ln_frame_median"].to_numpy(float)}
    axes.update(geology_axes(keys, args))
    meta_names = [k for k in ("incidence", "epoch_year", "subsolar_lat", "ln_frame_median") if k in axes]
    geo_names = [k for k in ("mola_elev", "themis_night_ir") if k in axes]
    per_axis = lv.attribution(smooth, axes, lon, lat, w=wdeg, cell_deg=args.cell_deg,
                              n_draws=args.perm_draws, seed=args.seed)
    gmeta = lv.group_r2(smooth, axes, meta_names, lon, lat, w=wdeg, cell_deg=args.cell_deg,
                        n_draws=args.perm_draws, seed=args.seed)
    ggeo = (lv.group_r2(smooth, axes, geo_names, lon, lat, w=wdeg, cell_deg=args.cell_deg,
                        n_draws=args.perm_draws, seed=args.seed)
            if geo_names else {"r2": np.nan, "p_value": np.nan, "axes": []})
    for nm, a in per_axis.items():
        side = "metadata" if nm in meta_names else "geology"
        print(f"    {nm:>16} ({side:>8}): R² {a['r2']:.3f}  p {a['p_value']:.4f}", flush=True)
    verdict = lv.trend_verdict(tr1, gmeta, ggeo)
    print(f"\n  metadata group R² {gmeta['r2']:.3f} (p {gmeta['p_value']:.4f}) | "
          f"geology group R² {ggeo['r2']:.3f} (p {ggeo['p_value']:.4f})", flush=True)
    print(f"  TREND-GUARD VERDICT: {verdict['verdict']} -> apply '{verdict['apply']}'\n"
          f"    {verdict['why']}", flush=True)
    if verdict["needs_ruling"]:
        print("    ⚠ §0.1 guard 1: an AMBIGUOUS verdict must NOT silently become full offsets — "
              "Stage D ships both composites and the call goes to Brian (§7 Q3).", flush=True)

    # 7 -------------------------------------------------------------- guard-3/4 diagnostics
    pm = ft["prob_mean"].to_numpy(float)
    ok = np.isfinite(pm) & np.isfinite(o_star)
    corr_pearson = float(np.corrcoef(o_star[ok], pm[ok])[0, 1]) if ok.sum() > 2 else np.nan
    from scipy.stats import spearmanr
    corr_spear = float(spearmanr(o_star[ok], pm[ok]).statistic) if ok.sum() > 2 else np.nan
    lnm = ft["ln_frame_median"].to_numpy(float)
    sd = float(np.nanstd(lnm)) if np.isfinite(lnm).any() else np.nan
    z_rad = (lnm - np.nanmean(lnm)) / sd if np.isfinite(sd) and sd > 0 else np.full(n, np.nan)
    normal = np.abs(z_rad) < 2.0                            # "radiometrically normal" (F02 was −2.23σ)
    big_normal = int(((np.abs(o_star) > BIG_OFFSET) & normal).sum())
    print(f"\n=== §0.1 guards ===\n  guard 4 (mean-flattening): corr(offset, frame-mean P(rich)) "
          f"Pearson {corr_pearson:+.3f} / Spearman {corr_spear:+.3f} (pilot −0.94); "
          f"|o|>{BIG_OFFSET} on radiometrically-normal frames: {big_normal}", flush=True)

    out = pd.DataFrame({
        "PRODUCT_ID": pids, "component": comp, "degree": deg, "offset_source": off_src,
        "edge_weight_tiles": es.weight_sum(n), "n_tiles": ft["n_tiles"].to_numpy(),
        "offset_logit": np.round(o_star, 4),
        "trend_fitted": np.round(smooth, 4), "trend_resid": np.round(resid, 4),
        "offset_residual_only": np.round(o_resid, 4),
        "lofo_offset_pred": np.round(o_hat, 4), "lofo_pred_err": np.round(pred_err, 4),
        "lofo_n_edges": n_used,
        "frame_mean_prob": np.round(pm, 5), "ln_frame_median": np.round(lnm, 4),
        "radiometry_z": np.round(z_rad, 3),
        "incidence": ft["incidence"].to_numpy(), "subsolar_lat": ft["subsolar_lat"].to_numpy(),
        "epoch_year": np.round(ft["epoch_year"].to_numpy(float), 3),
        "lon": np.round(lon, 4), "lat": np.round(lat, 4),
    })
    for nm in geo_names:
        out[nm] = np.round(axes[nm], 2)
    out["flag_big_offset_normal_radiometry"] = (np.abs(o_star) > BIG_OFFSET) & normal
    out["flag_under_pinned"] = (np.abs(o_star) > BIG_OFFSET) & (pred_err > 0.5 * BIG_OFFSET)
    out["flag_isolated"] = deg == 0
    out.to_csv(FIG / "fbuild_stagec_offsets.csv", index=False)

    watch = out[out["flag_big_offset_normal_radiometry"] | out["flag_under_pinned"] |
                out["flag_isolated"]].sort_values("offset_logit", key=np.abs, ascending=False)
    watch.to_csv(FIG / "fbuild_stagec_watchlist.csv", index=False)
    print(f"  guard 3 (under-pinned / unexplained large offsets): {len(watch)} frames on the "
          f"watchlist -> fbuild_stagec_watchlist.csv", flush=True)
    if len(watch):
        print(watch[["PRODUCT_ID", "offset_logit", "degree", "lofo_pred_err", "radiometry_z"]]
              .head(12).to_string(index=False), flush=True)

    guard = pd.DataFrame([{
        "n_frames": n, "n_edges": es.n_edges, "n_components": n_comp,
        "n_offsets_not_solved": n_patched,
        "min_shared_tiles": args.min_tiles, "lambda_star": round(lam_star, 3),
        "baseline_dp": round(base_dp, 4),
        "full_insample_dp": round(float(np.median(dp_star)), 4),
        "full_heldout_cv_dp": float(star["heldout_cv_dp"]),
        "lofo_median_pred_err": round(float(np.nanmedian(pred_err)), 4),
        "max_abs_offset": round(float(np.abs(o_star).max()), 3),
        "sd_offset": round(float(np.std(o_star)), 4),
        "linear_r2": round(tr1["r2"], 4), "linear_p": tr1["p_value"],
        "quad_r2": round(tr2["r2"], 4), "quad_p": tr2["p_value"],
        "null_p95_r2": round(tr1["null_p95_r2"], 4), "n_blocks": tr1["n_blocks"],
        "meta_group_r2": round(float(gmeta["r2"]), 4), "meta_group_p": gmeta["p_value"],
        "meta_axes": ";".join(gmeta["axes"]),
        "geo_group_r2": round(float(ggeo["r2"]), 4) if np.isfinite(ggeo["r2"]) else np.nan,
        "geo_group_p": ggeo["p_value"], "geo_axes": ";".join(ggeo["axes"]),
        "verdict": verdict["verdict"], "apply": verdict["apply"],
        "needs_ruling": verdict["needs_ruling"], "why": verdict["why"],
        "corr_offset_framemean_pearson": round(corr_pearson, 4),
        "corr_offset_framemean_spearman": round(corr_spear, 4),
        "n_big_offset_normal_radiometry": big_normal,
    }])
    guard.to_csv(FIG / "fbuild_trend_guard.csv", index=False)
    pd.DataFrame([{"axis": k, **v} for k, v in per_axis.items()]).to_csv(
        FIG / "fbuild_stagec_attribution.csv", index=False)

    _figures(out, tr1, gmeta, ggeo, per_axis, meta_names, lam_star, dp_star, base_dp)
    print(f"\nStage C done: offsets -> {FIG / 'fbuild_stagec_offsets.csv'} "
          f"(Stage D applies `offset_logit` or `offset_residual_only` per the verdict)", flush=True)
    return 0


# --------------------------------------------------------------------------- figures
def _figures(out, tr1, gmeta, ggeo, per_axis, meta_names, lam_star, dp_star, base_dp):
    o = out["offset_logit"].to_numpy(float)
    lon, lat = out["lon"].to_numpy(float), out["lat"].to_numpy(float)
    vmax = float(np.nanpercentile(np.abs(o), 98)) or 1.0

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    s = ax[0, 0].scatter(lon, lat, c=o, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=14)
    ax[0, 0].set_title(f"solved offsets o_f (λ*={lam_star:.1f}), n={len(o)}", fontsize=9)
    plt.colorbar(s, ax=ax[0, 0], fraction=0.046, label="logit offset")
    s = ax[0, 1].scatter(lon, lat, c=out["trend_fitted"], cmap="coolwarm", vmin=-vmax, vmax=vmax, s=14)
    ax[0, 1].set_title(f"smooth lon/lat plane: R²={tr1['r2']:.3f}, p={tr1['p_value']:.4f} "
                       f"(block-permutation)", fontsize=9)
    plt.colorbar(s, ax=ax[0, 1], fraction=0.046)
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xlabel("lon (deg)")
        a.set_ylabel("lat (deg)")
    ax[1, 0].bar(range(len(per_axis)), [v["r2"] for v in per_axis.values()],
                 color=["tab:blue" if k in meta_names else "tab:orange" for k in per_axis])
    ax[1, 0].set_xticks(range(len(per_axis)))
    ax[1, 0].set_xticklabels(list(per_axis), rotation=35, ha="right", fontsize=7)
    ax[1, 0].set_ylabel("R² vs the smooth field")
    ax[1, 0].set_title(f"attribution — metadata (blue) R²={gmeta['r2']:.3f} vs "
                       f"geology (orange) R²={ggeo['r2']:.3f}", fontsize=9)
    ax[1, 1].hist(o, bins=40, color="0.6")
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].set_xlabel("offset (logit)")
    ax[1, 1].set_title(f"offset distribution — sd {np.std(o):.3f}, |o|max {np.abs(o).max():.2f}",
                       fontsize=9)
    fig.suptitle("PLAN_FBuild Stage C — H4 offsets + trend guard")
    fig.tight_layout()
    fig.savefig(FIG / "fbuild_trend_guard.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    ax[0, 0].scatter(out["frame_mean_prob"], o, s=12, c="tab:blue")
    ax[0, 0].axhline(0, color="0.7", lw=0.8)
    ax[0, 0].set_xlabel("frame-mean P(rich)")
    ax[0, 0].set_ylabel("offset (logit)")
    ax[0, 0].set_title("§0.1 guard 4 — mean-flattening signature (pilot corr −0.94)", fontsize=9)
    ax[0, 1].scatter(out["radiometry_z"], o, s=12,
                     c=np.where(out["flag_big_offset_normal_radiometry"], "tab:red", "tab:blue"))
    ax[0, 1].axhline(0, color="0.7", lw=0.8)
    ax[0, 1].set_xlabel("frame radiometry z (ln frame median)")
    ax[0, 1].set_ylabel("offset (logit)")
    ax[0, 1].set_title("§0.1 guard 3 — large |o| on radiometrically-normal frames (red)", fontsize=9)
    ax[1, 0].scatter(out["degree"], np.abs(o), s=12, c="tab:green")
    ax[1, 0].set_xlabel("graph degree")
    ax[1, 0].set_ylabel("|offset|")
    ax[1, 0].set_title("how well-pinned is each offset", fontsize=9)
    ax[1, 1].hist(dp_star, bins=40, alpha=0.75, label=f"leveled (median {np.median(dp_star):.4f})")
    ax[1, 1].axvline(base_dp, color="tab:red", lw=1.2, label=f"unleveled median {base_dp:.4f}")
    ax[1, 1].set_xlabel("per-edge median |Δp| on co-located tiles")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("gate 2 — co-located disagreement before/after", fontsize=9)
    fig.suptitle("PLAN_FBuild Stage C — offset diagnostics")
    fig.tight_layout()
    fig.savefig(FIG / "fbuild_stagec_offsets.png", dpi=110)
    plt.close(fig)
    print(f"wrote {FIG / 'fbuild_trend_guard.png'} and {FIG / 'fbuild_stagec_offsets.png'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
