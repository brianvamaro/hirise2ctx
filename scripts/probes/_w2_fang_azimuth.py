"""Azimuth-conditioned read of the Fang-ViT probe (PLAN_CNN.md 5.1 caveat b).

Fang et al.'s own CBIR failure mode: high-incidence scenes produce shadow-dominated
embeddings that match by illumination geometry rather than geomorphology. Two questions
on our S=64 result (t1_gem192 vs Tier-1):

  1. Does the per-image dAUC benefit correlate with illumination geometry?
     Spearman of dAUC vs ctx_incidence_mean and vs circular azimuth distance from the
     cohort median. A strong negative trend = the FM helps least exactly where shadows
     dominate -> the caveat binds. The two azimuth outliers (ESP_076499_1160 at 228.6
     deg, ESP_068483_2280 at 1.7 deg / incidence 4.3 deg) are reported individually.
  2. Do the embeddings themselves ENCODE geometry? Leave-one-image-out ridge on the
     38 image-mean GeM-192 embeddings predicting incidence and sin/cos(azimuth).
     High held-out r = geometry is recoverable from the embedding (caveat confirmed
     as *present*); question 1 decides whether it *harms*.

Figure -> reports/figures/19_w2_fang_azimuth_read.png; numbers printed + JSON next to
the t1_gem192 verdict.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_w2_fang_azimuth.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATASET_DIR = REPO_ROOT / "dataset_v2"
EMB_DIR = DATASET_DIR / "fang_embeddings"
FIG_DIR = REPO_ROOT / "reports" / "figures"
VERDICT = REPO_ROOT / "models/fang_probe/t1_gem192/ed6b211643a2148e/verdict.json"
AZ_OUTLIERS = ("ESP_076499_1160", "ESP_068483_2280")


def circ_dist_deg(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def main() -> int:
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    dauc = pd.Series(v["t1_gem192"]["per_image_dauc"], dtype=float)

    rows = []
    for p in sorted((DATASET_DIR / "features_ctx_illum").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["scale_idx", "ctx_subsolar_az_mean", "ctx_incidence_mean"])
        df = df[df.scale_idx == 3]
        rows.append({"obs_id": p.stem,
                     "az": float(df.ctx_subsolar_az_mean.mean()),
                     "inc": float(df.ctx_incidence_mean.mean())})
    geom = pd.DataFrame(rows).set_index("obs_id")
    g = geom.join(dauc.rename("dauc"), how="inner").dropna()
    az_med = float(g["az"].median())
    g["az_dist"] = circ_dist_deg(g["az"].to_numpy(), az_med)

    print(f"n images with dAUC + geometry: {len(g)}  (cohort az median {az_med:.1f} deg)\n")
    r_inc = stats.spearmanr(g["inc"], g["dauc"])
    r_az = stats.spearmanr(g["az_dist"], g["dauc"])
    print(f"Q1  dAUC vs incidence:        rho={r_inc.statistic:+.3f}  p={r_inc.pvalue:.3f}")
    print(f"Q1  dAUC vs |az - median|:    rho={r_az.statistic:+.3f}  p={r_az.pvalue:.3f}")
    for obs in AZ_OUTLIERS:
        if obs in g.index:
            print(f"    outlier {obs}: az={g.loc[obs, 'az']:.1f} inc={g.loc[obs, 'inc']:.1f} "
                  f"dAUC={g.loc[obs, 'dauc']:+.4f}")

    # ---- Q2: geometry recoverable from image-mean embeddings? LOO ridge ----
    embs, order = [], []
    for obs in g.index:
        z = np.load(EMB_DIR / f"{obs}_P192.npz")
        e = z["gem"][z["valid"].astype(bool)]
        embs.append(e.mean(axis=0))
        order.append(obs)
    X = np.vstack(embs)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    targets = {
        "incidence": g.loc[order, "inc"].to_numpy(),
        "sin_az": np.sin(np.deg2rad(g.loc[order, "az"].to_numpy())),
        "cos_az": np.cos(np.deg2rad(g.loc[order, "az"].to_numpy())),
    }
    from sklearn.linear_model import Ridge

    q2 = {}
    print()
    for name, yt in targets.items():
        preds = np.empty_like(yt)
        for i in range(len(yt)):
            m = np.ones(len(yt), dtype=bool)
            m[i] = False
            preds[i] = Ridge(alpha=100.0).fit(X[m], yt[m]).predict(X[None, i])[0]
        r = stats.pearsonr(yt, preds)
        q2[name] = {"r": float(r.statistic), "p": float(r.pvalue)}
        print(f"Q2  LOO ridge {name:>9s}: held-out r={r.statistic:+.3f}  p={r.pvalue:.4f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, xcol, xlab, rr in ((axes[0], "inc", "CTX incidence (deg)", r_inc),
                               (axes[1], "az_dist", f"|azimuth - {az_med:.0f}| (deg, circular)", r_az)):
        ax.scatter(g[xcol], g["dauc"], s=28, c="#377eb8")
        for obs in AZ_OUTLIERS:
            if obs in g.index:
                ax.scatter(g.loc[obs, xcol], g.loc[obs, "dauc"], s=70, c="#e41a1c", zorder=3)
                ax.annotate(obs.replace("ESP_", ""), (g.loc[obs, xcol], g.loc[obs, "dauc"]),
                            fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel(xlab)
        ax.set_ylabel("per-image dAUC (t1_gem192 - Tier-1)")
        ax.set_title(f"Spearman rho={rr.statistic:+.3f} p={rr.pvalue:.3f}", fontsize=10)
    fig.suptitle("Fang-ViT benefit vs illumination geometry (S=64) -- red = azimuth outliers",
                 fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "19_w2_fang_azimuth_read.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_png.relative_to(REPO_ROOT)}")

    out_json = VERDICT.parent / "azimuth_read.json"
    out_json.write_text(json.dumps({
        "q1_dauc_vs_incidence": {"rho": float(r_inc.statistic), "p": float(r_inc.pvalue)},
        "q1_dauc_vs_az_dist": {"rho": float(r_az.statistic), "p": float(r_az.pvalue)},
        "q2_loo_ridge_geometry_from_embeddings": q2,
        "outliers": {o: {"az": float(g.loc[o, "az"]), "inc": float(g.loc[o, "inc"]),
                         "dauc": float(g.loc[o, "dauc"])}
                     for o in AZ_OUTLIERS if o in g.index},
        "n_images": int(len(g)),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
