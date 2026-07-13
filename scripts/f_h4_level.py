"""H4 (PLAN_StripingArtifact PHASE 2 / PLAN_H4_Leveling) — overlap-constrained POST-HOC
per-frame prediction leveling on the E8_N44 pilot.

H2 (linear subspace) and H3 (in-head consistency penalty) both FAILED: per-frame prediction
variance is not separable from geology by a data-driven *invariance* instrument without
collapsing dynamic range. H4 is the last F-mode lever — a per-frame *additive logit offset*
solved AFTER the head, on the F02-class level component, that by construction cannot touch
within-frame ranking (so it cannot collapse skill the way H3 did).

Pipeline (all inputs already on disk; reuses the H2/H3 per-frame embedding cache):
  1. H1 head predicts the 7 aligned E8_N44 crops -> per-frame P(rich) stack on the shared grid.
  2. Solve offsets o_f minimizing  Σ_edges Σ_colocated w·[(ℓ_i+o_i)-(ℓ_j+o_j)]² + λ·Σ o_f²
     (ℓ = logit p). Exact per-edge sufficient statistic: (δ̄_ij, W_ij) -> 7-unknown weighted LS.
  3. Sweep λ; PICK λ by leave-one-edge-out CV held-out |Δp| (PLAN_H4 §3.2 — the non-circular
     check; post-H4 η² alone would be circular, exactly what killed option D).
  4. Trend guard (§2): fit a plane to o_f vs frame centers; report the smooth component so a
     real regional gradient can't be laundered into offsets.
  5. Score before/after frame-block η² (partition + median) + before/after choropleth, plus
     offset-vs-incidence / offset-vs-epoch scatters (F02 should be the outlier it is).

Baselines on this crop: mosaic raw 0.196 / A1 0.141 / H1(center) partition 0.128 / median
0.081 / pred_overlap 0.073. Reopening bar (combined levers count, Brian 2026-07-09b):
η² ≲ 0.05 at skill ≥ −0.02 with held-out edge-CV dropping below 0.073.

Run (laptop GPU; embedding cached from H2/H3 so this is seconds):
  conda run --no-capture-output -n geospatial python -u scripts/f_h4_level.py \
      --head models/deployable_f_center/86c51a5dca220f63
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.f_h2_eta2 as h2
import scripts.f_pilot_crop as fpc
from rasterio.transform import Affine
from src.ctx_edr import frame_table
from src.fm_embeddings import FangEmbedder
from src.modeling.mlp_head import DeployableHead

FIG = REPO / "reports" / "figures"
EPS = 1e-4
LAMBDAS = [0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


# ------------------------------------------------------------------ math helpers
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q) - np.log(1.0 - q)


def edge_stats(stack: np.ndarray):
    """Per-edge sufficient statistics for the leveling LS.

    Returns edges = list of (i, j, dbar, W, mask) where dbar = weighted-mean(ℓ_j − ℓ_i)
    over co-located tiles (so the LS wants o_i − o_j ≈ dbar), W = tile count, mask = the
    co-located boolean grid (kept for held-out |Δp| in CV). Uniform weights (PLAN_H4 §2).
    """
    n = stack.shape[0]
    logit = _logit(stack)
    valid = np.isfinite(stack)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            if both.sum() < 200:
                continue
            d = logit[j][both] - logit[i][both]
            edges.append((i, j, float(d.mean()), float(both.sum()), both))
    return edges, logit, valid


def solve_offsets(edges, lam: float, n: int) -> np.ndarray:
    """Weighted LS with Tikhonov λ; gauge fixed by median(o) = 0 (PLAN_H4 §2)."""
    ata = np.zeros((n, n))
    atb = np.zeros(n)
    for e in edges:
        i, j, dbar, w = e[0], e[1], e[2], e[3]
        ata[i, i] += w
        ata[j, j] += w
        ata[i, j] -= w
        ata[j, i] -= w
        atb[i] += w * dbar
        atb[j] -= w * dbar
    ata += lam * np.eye(n)
    # min-norm LS: robust to the rank-deficient Laplacian at λ=0 and to an edge-drop
    # disconnecting the graph in leave-one-edge-out CV (each component keeps its own gauge).
    o = np.linalg.lstsq(ata, atb, rcond=None)[0]
    return o - np.median(o)


def median_dp(logit, o, i, j, mask) -> float:
    """Median |Δp| on one edge's co-located tiles after applying offsets o."""
    pi = _sigmoid(logit[i][mask] + o[i])
    pj = _sigmoid(logit[j][mask] + o[j])
    return float(np.median(np.abs(pi - pj)))


def held_out_cv(edges, logit, lam: float, n: int) -> float:
    """Leave-one-edge-out: median over edges of held-out |Δp| (offsets fit without that edge)."""
    dps = []
    for k, (i, j, _, _, mask) in enumerate(edges):
        rest = [e for m, e in enumerate(edges) if m != k]
        o = solve_offsets(rest, lam, n)
        dps.append(median_dp(logit, o, i, j, mask))
    return float(np.median(dps))


def level_stack(stack: np.ndarray, o: np.ndarray) -> np.ndarray:
    out = _sigmoid(_logit(stack) + o[:, None, None])
    out[~np.isfinite(stack)] = np.nan
    return out.astype(np.float32)


# ------------------------------------------------------------------ ctx (mirror of f_h2_eta2.main)
def build_ctx(pids):
    ft = frame_table(fpc.TILE).set_index("PRODUCT_ID")
    cos_i = {p: float(np.cos(np.radians(ft.loc[p, "INCIDENCE"]))) for p in pids}
    nb = np.load(REPO / "reports" / "f_leg_b" / "h2_nuisance_basis.npz")
    lo, hi = float(nb["stretch_lo"]), float(nb["stretch_hi"])
    ctx = {"cos_i": cos_i,
           "minnaert_div": {p: cos_i[p] ** h2.K_MINNAERT for p in pids},
           "log_lohi": (lo, hi)}
    print(f"minnaert_center: k={h2.K_MINNAERT}, log stretch I/F {lo:.4f}..{hi:.4f}", flush=True)
    return ctx, ft


# ------------------------------------------------------------------ trend guard (§2)
def trend_guard(o, ft, pids):
    """Fit a plane o ~ a + b·x + c·y to frame centers; report the smooth component."""
    cen = np.array([[ft.loc[p, "geometry"].centroid.x,
                     ft.loc[p, "geometry"].centroid.y] for p in pids])
    x = (cen[:, 0] - cen[:, 0].mean()) / 1e5
    y = (cen[:, 1] - cen[:, 1].mean()) / 1e5
    A = np.column_stack([np.ones_like(x), x, y])
    coef, *_ = np.linalg.lstsq(A, o, rcond=None)
    fitted = A @ coef
    resid = o - fitted
    var_o = float(np.var(o))
    frac = float(np.var(fitted) / var_o) if var_o > 0 else 0.0
    return fitted, resid, frac, cen


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="models/deployable_f_center/86c51a5dca220f63",
                    help="H1 head dir (default = the operating baseline)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    pids = fpc.crop_pids()
    fpc.WORK.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        fpc.align_frame(pid)
    ctx, ft = build_ctx(pids)

    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)
    frames, ti, tj = h2.embed_frames(pids, ctx, embedder)
    head = DeployableHead.load(Path(args.head))
    stack, transform = h2.head_rasters(head, frames, pids, ti, tj)
    n = len(pids)
    print(f"predicted {n} frames; stack {stack.shape}", flush=True)

    edges, logit, _ = edge_stats(stack)
    print(f"{len(edges)} overlap edges (>=200 co-located tiles)", flush=True)

    # unleveled (o=0) in-sample + baseline held-out |Δp|
    base_dp = float(np.median([median_dp(logit, np.zeros(n), i, j, m)
                               for i, j, _, _, m in edges]))
    before, *_ = h2.score("H1", stack, pids, transform)
    print(f"unleveled: partition η² {before['partition']}  median {before['median']}  "
          f"pred_overlap {before['pred_overlap']}  in-sample|Δp| {base_dp:.4f}", flush=True)

    # λ sweep: in-sample fit, held-out CV, and post-leveling η²
    rows = []
    solved = {}
    for lam in LAMBDAS:
        o = solve_offsets(edges, lam, n)
        solved[lam] = o
        insamp = float(np.median([median_dp(logit, o, i, j, m) for i, j, _, _, m in edges]))
        cv = held_out_cv(edges, logit, lam, n)
        after, *_ = h2.score(f"lam{lam}", level_stack(stack, o), pids, transform)
        rows.append({"lambda": lam, "partition_eta2": after["partition"],
                     "median_eta2": after["median"], "pred_overlap": after["pred_overlap"],
                     "insample_dp": round(insamp, 4), "heldout_cv_dp": round(cv, 4),
                     "max_abs_offset": round(float(np.abs(o).max()), 3)})
        print(f"  λ={lam:>6}: part η² {after['partition']}  med {after['median']}  "
              f"in|Δp| {insamp:.4f}  heldout|Δp| {cv:.4f}  |o|max {np.abs(o).max():.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    # pick λ by held-out CV (tie -> larger λ = more regularized / conservative)
    best = df.sort_values(["heldout_cv_dp", "lambda"], ascending=[True, False]).iloc[0]
    lam_star = float(best["lambda"])
    o_star = solved[lam_star]
    print(f"\nPICK λ*={lam_star} by held-out CV: held-out |Δp| {best['heldout_cv_dp']} "
          f"vs unleveled {base_dp:.4f}  (bar: drop materially below 0.073)", flush=True)

    # trend guard on the chosen offsets
    fitted, resid, frac, cen = trend_guard(o_star, ft, pids)
    guard = frac > 0.5
    print(f"\ntrend guard: smooth (plane) component = {frac:.1%} of offset variance; "
          f"{'SIGNIFICANT — also reporting residual-only (conservative)' if guard else 'small — full offsets safe'}",
          flush=True)
    # conservative, trend-guarded score: apply ONLY the residual offsets (the smooth plane,
    # which a real regional gradient would masquerade as, is added back / left in place).
    r_star = resid - np.median(resid)
    res_score, *_ = h2.score("resid", level_stack(stack, r_star), pids, transform)
    res_insamp = float(np.median([median_dp(logit, r_star, i, j, m) for i, j, _, _, m in edges]))
    print(f"residual-only (trend-guarded): partition η² {res_score['partition']}  "
          f"median {res_score['median']}  in-sample|Δp| {res_insamp:.4f}", flush=True)

    off = pd.DataFrame({"PRODUCT_ID": pids,
                        "incidence": [ft.loc[p, "INCIDENCE"] for p in pids],
                        "image_time": [str(ft.loc[p, "IMAGE_TIME"]) for p in pids],
                        "offset_logit": np.round(o_star, 4),
                        "trend_fitted": np.round(fitted, 4),
                        "trend_resid": np.round(resid, 4)})
    # trend-guard / verdict summary (single source of truth for the notebook)
    guard_df = pd.DataFrame([{
        "lambda_star": lam_star,
        "baseline_partition_eta2": before["partition"], "baseline_median_eta2": before["median"],
        "baseline_dp": round(base_dp, 4),
        "full_partition_eta2": df.loc[df["lambda"] == lam_star, "partition_eta2"].iloc[0],
        "full_median_eta2": df.loc[df["lambda"] == lam_star, "median_eta2"].iloc[0],
        "full_heldout_cv_dp": df.loc[df["lambda"] == lam_star, "heldout_cv_dp"].iloc[0],
        "resid_partition_eta2": res_score["partition"], "resid_median_eta2": res_score["median"],
        "resid_dp": round(res_insamp, 4),
        "smooth_plane_frac": round(frac, 4), "trend_guard_fires": bool(guard),
    }])
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "f_h4_leveling_summary.csv", index=False)
    off.to_csv(FIG / "f_h4_offsets.csv", index=False)
    guard_df.to_csv(FIG / "f_h4_trend_guard.csv", index=False)
    print("\n=== H4 λ SWEEP (baselines: mosaic 0.196 / A1 0.141 / H1 part 0.128 med 0.081; "
          "unleveled in|Δp| {:.4f}) ===".format(base_dp))
    print(df.to_string(index=False))
    print("\n=== per-frame offsets @ λ* ===")
    print(off.to_string(index=False))

    _figures(stack, o_star, pids, transform, off, cen, base_dp, lam_star)


def _figures(stack, o, pids, transform, off, cen, base_dp, lam_star):
    leveled = level_stack(stack, o)
    _, labels_b, part_b, med_b = h2.score("b", stack, pids, transform)
    _, labels_a, part_a, med_a = h2.score("a", leveled, pids, transform)

    def choro(labels, part):
        c = np.full(part.shape, np.nan, dtype=np.float32)
        for i in range(len(pids)):
            sel = (labels == i) & np.isfinite(part)
            if sel.sum() >= 30:
                c[labels == i] = np.nanmean(part[sel])
        return c

    vmax = np.nanpercentile(part_b, 99)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
    panels = [(med_b, "H1 (before): median composite"),
              (choro(labels_b, part_b), "H1 (before): frame-mean choropleth"),
              (med_a, f"H4 (after, λ={lam_star:g}): median composite"),
              (choro(labels_a, part_a), f"H4 (after, λ={lam_star:g}): frame-mean choropleth")]
    for a, (img, t) in zip(ax.ravel(), panels):
        im = a.imshow(img, cmap="magma", vmax=vmax)
        a.set_title(t, fontsize=9)
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("H4 overlap leveling — E8_N44 pilot — P(boulder-rich), P(fa>1e-2)")
    fig.tight_layout()
    fig.savefig(FIG / "f_h4_leveling_choropleth.png", dpi=110)
    plt.close(fig)
    print(f"wrote {FIG / 'f_h4_leveling_choropleth.png'}")

    # offset vs incidence / epoch (F02 = the outlier)
    t = pd.to_datetime(off["image_time"], errors="coerce")
    is_f02 = off["PRODUCT_ID"].str.startswith("F02").values
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, xv, xl in [(ax[0], off["incidence"].values, "incidence (deg)"),
                      (ax[1], t.values, "image time")]:
        a.axhline(0, color="0.7", lw=0.8)
        a.scatter(xv[~is_f02], off["offset_logit"].values[~is_f02], c="tab:blue", label="frame")
        a.scatter(xv[is_f02], off["offset_logit"].values[is_f02], c="tab:red", label="F02")
        a.set_xlabel(xl)
        a.set_ylabel("leveling offset (logit)")
        a.legend(fontsize=8)
    fig.suptitle(f"H4 per-frame offsets @ λ*={lam_star:g}  (F02 = −2.23σ dark frame)")
    fig.tight_layout()
    fig.savefig(FIG / "f_h4_offset_scatter.png", dpi=110)
    plt.close(fig)
    print(f"wrote {FIG / 'f_h4_offset_scatter.png'}")


if __name__ == "__main__":
    main()
