"""W1 Rung 4 — CTX content: seam/source structure vs tile-level error.

Joins the re-banked predictions (post-fix) with per-tile CTX source stats
(dataset_v2/features_ctx_illum): ctx_n_sources, ctx_dominant_source_fraction,
ctx_incidence_std. Per image at S=64:

1. seam_frac (tiles with n_sources > 1), and meaningful AUC computed on
   single-source tiles only vs all tiles -- the direct test of the
   "mask seam tiles" Tier 1 reliability-flag candidate.
2. Image-level re-test of the cause-1 correlation on corrected labels:
   Spearman of per-image AUC vs mean_n_sources / dominant_source_fraction /
   std_ctx_incidence.
3. Tile-level error maps for the 8 anti-signal images: y_true binary,
   y_pred, and seam tiles outlined -- looking for seam-aligned error bands.

Writes _w1_rung4_seam_error.md + reports/figures/w1_rung4_errmap_{obs}.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("models/lightgbm_two_stage_balanced/8c7523615964f5cb/scale_S64_target_boulder_count")
ILLUM = Path("dataset_v2/features_ctx_illum")
FIGDIR = Path("reports/figures")
OUT_MD = Path("scripts/probes/_w1_rung4_seam_error.md")
THR = 50.0
ANTI = [
    "ESP_076499_1160", "ESP_055978_2270", "ESP_054000_2255", "ESP_046328_2180",
    "ESP_064510_2260", "ESP_047976_2020", "ESP_049242_2115", "ESP_059686_2235",
]

pred = pd.read_parquet(ART / "predictions.parquet")

rows = []
frames = {}
for obs, g in pred.groupby("obs_id"):
    il = pd.read_parquet(ILLUM / f"{obs}.parquet")
    il = il[il.scale_idx == 3][["ti", "tj", "ctx_n_sources", "ctx_dominant_source_fraction",
                                "ctx_incidence_std", "ctx_incidence_mean"]]
    m = g.merge(il, on=["ti", "tj"], how="left", validate="one_to_one")
    frames[obs] = m
    y = (m.y_true > THR).astype(int)
    single = m.ctx_n_sources == 1

    def safe_auc(mask):
        yt, yp = y[mask], m.y_pred[mask]
        if yt.nunique() < 2:
            return np.nan
        return roc_auc_score(yt, yp)

    rows.append(
        dict(
            obs_id=obs,
            anti=obs in ANTI,
            n=len(m),
            seam_frac=float((~single).mean()),
            mean_n_sources=float(m.ctx_n_sources.mean()),
            dom_frac=float(m.ctx_dominant_source_fraction.mean()),
            inc_std=float(m.ctx_incidence_std.mean()),
            auc_all=safe_auc(np.ones(len(m), bool)),
            auc_single=safe_auc(single.to_numpy()),
            auc_seam=safe_auc((~single).to_numpy()),
        )
    )

df = pd.DataFrame(rows).sort_values("auc_all")
df["delta_single"] = df.auc_single - df.auc_all
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

lines = []
for c in ["mean_n_sources", "dom_frac", "inc_std", "seam_frac"]:
    sub = df[[c, "auc_all"]].dropna()
    rho, p = spearmanr(sub[c], sub.auc_all)
    lines.append(f"- per-image `{c}` vs AUC: Spearman rho={rho:+.3f} p={p:.4f} (n={len(sub)})")
    print(lines[-1])

d = df.delta_single.dropna()
lines.append(f"- single-source-only AUC delta: mean {d.mean():+.4f}, median {d.median():+.4f}, "
             f"improved in {(d > 0).mean():.0%} of {len(d)} images")
print(lines[-1])
anti_d = df[df.anti].delta_single.dropna()
lines.append(f"- anti-signal images only: mean delta {anti_d.mean():+.4f} (n={len(anti_d)})")
print(lines[-1])

# error maps for anti-signal images
for obs in ANTI:
    m = frames[obs]
    ti0, tj0 = m.ti.min(), m.tj.min()
    H, W = m.ti.max() - ti0 + 1, m.tj.max() - tj0 + 1
    def raster(vals, fill=np.nan):
        a = np.full((H, W), fill, dtype=float)
        a[m.ti - ti0, m.tj - tj0] = vals
        return a
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    axes[0].imshow(raster((m.y_true > THR).astype(float)), cmap="RdBu_r", vmin=0, vmax=1)
    axes[0].set_title("y_true (bc>50)")
    yp = np.log1p(np.clip(m.y_pred, 0, None))
    axes[1].imshow(raster(yp), cmap="viridis")
    axes[1].set_title("log1p(y_pred)")
    axes[2].imshow(raster((m.ctx_n_sources > 1).astype(float)), cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("seam tiles (n_sources>1)")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"{obs} — tile error map (S=64), AUC={df.set_index('obs_id').loc[obs, 'auc_all']:.3f}")
    out = FIGDIR / f"w1_rung4_errmap_{obs}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  map -> {out}")

OUT_MD.write_text(
    "# W1 Rung 4 — seam/source structure vs tile-level error (post-fix recipe)\n\n"
    "```\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```\n\n"
    "## Correlations / masking test\n" + "\n".join(lines) + "\n\n"
    "Error maps: reports/figures/w1_rung4_errmap_*.png\n",
    encoding="utf-8",
)
print(f"wrote {OUT_MD}")
