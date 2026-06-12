"""Build notebooks/20_fang_vit_probe.ipynb from Python source.

QA notebook for the W2 Phase 2 Fang-ViT frozen-embedding probe
(PLAN_CNN.md 5.1; DECISIONS.md 2026-06-12 x2). Documents: verdict tables at
both scales + pool ablation, per-image dAUC structure by failure class, the
patch-alignment visual check, the azimuth-conditioned read, and a
side-by-side of FM vs Tier-1 top-ranked tiles on the azimuth outlier.

Inputs (nothing recomputed from raw data):
  - models/fang_probe/*/*/verdict.json (+ predictions.parquet for 5)
  - models/fang_probe/t1_gem192/*/azimuth_read.json
  - models/_sweep_binary/20260611T214042Z/summary.parquet (Tier-1 S=64 AUCs)
  - dataset_v2/w1_dossier.parquet
  - reports/figures/19_w2_fang_patch_alignment_*.png, 19_w2_fang_azimuth_read.png
  - dataset_v2/labels/{obs}.json + cache_v2/ctx_windows/{obs}.tif (5, top-tile patches)

Figures written: reports/figures/20_fang_{verdicts,perimage_dauc,topk_ESP_076499_1160}.png
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "20_fang_vit_probe.ipynb"


def md(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md(
    """# 20 — W2 Phase 2: Fang-ViT frozen-embedding probe

Documents the [PLAN_CNN.md](../PLAN_CNN.md) §5.1 lead bet
(DECISIONS.md 2026-06-12, two entries):
[Fang et al. 2026](https://doi.org/10.1029/2025JH000827) released a ViT-B/16
pretrained (MAE+DINO, self-supervised) on **3.9M crops of the Murray Lab CTX
mosaic itself** ([Zenodo 18180801](https://doi.org/10.5281/zenodo.18180801)).
We extract **frozen GeM(p=3)-pooled embeddings** per tile (own-tile input +
3×3-context input, both bicubic→224) and append them as LightGBM feature
columns in the standard LOIO harness (`fa_gt_1e-2`, 38 v2 images).

**Result: both pre-declared reference gates pass at BOTH scales by the
largest margin of the program** — pooled PR-AUC 0.5651 → **0.7637** at S=64;
the S=32 Tier-1 collapse (0.4840) is **fixed** (0.7639). The 3×3-context
input is the carrier at both scales; GeM > mean > cls; the combined
64+192 cell is the best per-image variant.

**Caveats carried with the claim** (full text in DECISIONS.md):
1. *Transductive pretraining* — the FM saw test **pixels** (never labels)
   during SSL. Deployment estimand = Murray-mosaic inference, which is
   in-corpus everywhere, so LOIO is unbiased **for that estimand**; the
   optional empirical bound is a MOMO (disjoint-corpus) cross-check.
2. *Post-hoc assembly* — promotion requires the standing pre-declared
   confirmation on cohort-expansion images. No seed instability exists in
   this path (deterministic end to end).

Probe-tier code: `scripts/probes/_w2_fang_{embed,probe,patch_visual,azimuth}.py`.
""",
    cell_id="intro",
))

cells.append(code(
    """import json
from pathlib import Path

import sys
REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))
import src.modeling  # noqa: F401  -- Windows DLL bootstrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

FIG = REPO / "reports" / "figures"
PROBE = REPO / "models" / "fang_probe"
DOSSIER = pd.read_parquet(REPO / "dataset_v2/w1_dossier.parquet")
print("probe runs on disk:", sorted(p.name for p in PROBE.iterdir() if p.is_dir()))""",
    cell_id="setup",
))

cells.append(md(
    """## 1. Verdict tables — every probe variant vs its Tier-1 reference

Gates (reference bars, not promotion): pooled ΔPR-AUC ≥ +0.03 vs Tier-1;
per-image ΔAUC median ≥ +0.05 with Wilcoxon p < 0.05 on dossier
validity-passing images.""",
    cell_id="s1_md",
))

cells.append(code(
    """rows = []
for vj in sorted(PROBE.glob("*/*/verdict.json")):
    v = json.loads(vj.read_text(encoding="utf-8"))
    label = [k for k in v if k != "tier1_ref"][0]
    r = v[label]
    scale = 32 if ("gem32" in label or "gem96" in label or "_S32" in vj.parts[-3]) else 64
    rows.append({
        "variant": label, "S": scale,
        "pooled_pr": r["pooled_pr_auc"], "prec@5%": r["prec_at_5"],
        "med_auc": r["med_auc"], "dauc_med(v)": r["dauc_median_v"],
        "win": r["dauc_win_v"], "p": r["wilcoxon_p"],
        "gate_pooled": r["gate_pooled"], "gate_per_image": r["gate_per_image"],
        "t1_pooled_ref": v["tier1_ref"]["pooled_pr_auc"],
    })
vt = pd.DataFrame(rows).sort_values(["S", "pooled_pr"], ascending=[False, False]).reset_index(drop=True)
display(vt.round(4))

fig, ax = plt.subplots(figsize=(9, 4.2))
order = vt.sort_values(["S", "pooled_pr"])
colors = ["#377eb8" if s == 64 else "#4daf4a" for s in order["S"]]
ax.barh(order["variant"] + "  (S=" + order["S"].astype(str) + ")", order["pooled_pr"], color=colors)
for ref, s, c in ((0.5651, 64, "#377eb8"), (0.4840, 32, "#4daf4a")):
    ax.axvline(ref, color=c, ls="--", lw=1.2, label=f"Tier-1 ref S={s} ({ref})")
ax.axvline(0.5955, color="gray", ls=":", lw=1.2, label="W2 fusion best F1(ens) (0.5955)")
ax.set_xlabel("pooled PR-AUC (LOIO, fa_gt_1e-2)")
ax.set_title("Fang-ViT probe variants vs references")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "20_fang_verdicts.png", dpi=150, bbox_inches="tight")
plt.show()""",
    cell_id="s1_code",
))

cells.append(md(
    """## 2. Per-image ΔAUC structure — who gets rescued

ΔAUC = per-image AUC(t1_gem192) − AUC(Tier-1 S=64), colored by the W1
dossier `attributed_cause`. The historical failure classes
(`distribution_shift`, `texture_decorrelated`) are exactly where the FM
helps most; `validity_limited` (images whose AUC is barely measurable) is
the only negative class.""",
    cell_id="s2_md",
))

cells.append(code(
    """vj = sorted(PROBE.glob("t1_gem192/*/verdict.json"))[0]
v = json.loads(vj.read_text(encoding="utf-8"))
d = pd.Series(v["t1_gem192"]["per_image_dauc"], dtype=float).sort_values()
cause = DOSSIER["attributed_cause"].reindex(d.index).fillna("unclassified")
palette = {"ok": "#999999", "ok_geometry_fixed": "#a6cee3", "ok_shadowfeat_fixed": "#b2df8a",
           "ok_validity_limited": "#cccc99", "distribution_shift": "#e41a1c",
           "texture_decorrelated": "#ff7f00", "validity_limited": "#6a3d9a",
           "unclassified": "#000000"}
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.barh([o.replace("ESP_", "") for o in d.index], d.to_numpy(),
        color=[palette[c] for c in cause])
ax.axvline(0, color="k", lw=0.8)
ax.axvline(d.median(), color="#377eb8", ls="--", lw=1.2,
           label=f"median {d.median():+.3f}")
handles = [plt.Rectangle((0, 0), 1, 1, color=v_) for v_ in palette.values()]
ax.legend(handles + [ax.lines[-1]], list(palette) + [f"median {d.median():+.3f}"],
          fontsize=7, loc="lower right")
ax.set_xlabel("per-image dAUC (t1_gem192 - Tier-1, S=64)")
ax.set_title("Fang-ViT per-image benefit by W1 failure class")
fig.tight_layout(); fig.savefig(FIG / "20_fang_perimage_dauc.png", dpi=150, bbox_inches="tight")
plt.show()
print(d.groupby(cause).agg(["mean", "count"]).round(4))""",
    cell_id="s2_code",
))

cells.append(md(
    """## 3. Input-alignment visual check (Brian-requested)

The two ViT inputs per tile: the cached 64-px Stage-4b patch and a 192-px
3×3-context box sliced from the cached CTX window. The center 64×64 of every
192-px slice is asserted **bit-identical** to the cached patch during
extraction (sampled per image, all 38 images pass); these figures are the
eyeball version. Red/blue/green = max-label tile, nearest-to-window-edge
tile (slice-arithmetic stress case), median tile.""",
    cell_id="s3_md",
))

cells.append(code(
    """for obs in ("ESP_042964_2160", "ESP_076499_1160"):
    display(Image(filename=str(FIG / f"19_w2_fang_patch_alignment_{obs}.png"), width=900))""",
    cell_id="s3_code",
))

cells.append(md(
    """## 4. Azimuth-conditioned read — the paper's CBIR caveat

Fang et al. flag that high-incidence scenes produce shadow-dominated
embeddings that match by illumination. In our data the caveat is **present
but harmless**: sin(azimuth) is LOO-recoverable from image-mean embeddings
(r=+0.588, p=1e-4), yet the per-image benefit is geometry-agnostic
(ρ vs incidence −0.06 ns; vs azimuth-distance +0.16 ns) and the cohort's
**biggest winner is the azimuth outlier** ESP_076499_1160 (+0.458) — the
image that rotation augmentation, AdaBN, and z-scoring each only partially
rescued across W1–W2.""",
    cell_id="s4_md",
))

cells.append(code(
    """az = json.loads((vj.parent / "azimuth_read.json").read_text(encoding="utf-8"))
print(json.dumps(az, indent=2))
display(Image(filename=str(FIG / "19_w2_fang_azimuth_read.png"), width=900))""",
    cell_id="s4_code",
))

cells.append(md(
    """## 5. What the re-ranking looks like — top tiles on the azimuth outlier

Held-out top-8 tiles by score on ESP_076499_1160 (the +0.458 image):
t1_gem192 vs Tier-1, each tile shown as its 192-px context box with the
true label. The FM's top picks are nearly all true positives; Tier-1's
are mixed — the same contrast that prec@5% 0.977 vs 0.771 summarizes
cohort-wide.""",
    cell_id="s5_md",
))

cells.append(code(
    """import rasterio

OBS = "ESP_076499_1160"
side = json.loads((REPO / f"dataset_v2/labels/{OBS}.json").read_text(encoding="utf-8"))
with rasterio.open(side["ctx_window_tif"]) as src:
    win = src.read(1).astype(np.uint8)
row0, col0 = int(side["mosaic_row_origin"]), int(side["mosaic_col_origin"])

fm = pd.read_parquet(sorted(PROBE.glob("t1_gem192/*/predictions.parquet"))[0])
fm = fm[fm.obs_id == OBS]
t1 = pd.read_parquet(REPO / "models/lightgbm_classification/99de85c1ad2a72e6/"
                            "scale_S64_tfa_gt_1e-2/predictions.parquet")
t1 = t1[t1.obs_id == OBS]

lo, hi = np.percentile(win[win > 0], [2, 98])
fig, axes = plt.subplots(2, 8, figsize=(15, 4.6))
for r, (df, name) in enumerate(((fm, "t1_gem192"), (t1, "Tier-1"))):
    top = df.nlargest(8, "y_pred")
    for c, (_, t) in enumerate(top.iterrows()):
        rw, cw = int(t.ti) * 64 - row0, int(t.tj) * 64 - col0
        patch = win[rw - 64: rw + 128, cw - 64: cw + 128]
        ax = axes[r, c]
        ax.imshow(patch, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ok = bool(t.y_true)
        ax.set_title(("POS" if ok else "neg") + f"  p={t.y_pred:.2f}",
                     fontsize=8, color=("#1a7a1a" if ok else "#cc2222"))
        ax.set_xticks([]); ax.set_yticks([])
    axes[r, 0].set_ylabel(name, fontsize=11)
fig.suptitle(f"{OBS}: held-out top-8 tiles by score (192-px context shown)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "20_fang_topk_ESP_076499_1160.png", dpi=150, bbox_inches="tight")
plt.show()
print("FM top-8 precision:", fm.nlargest(8, 'y_pred').y_true.mean(),
      "  Tier-1 top-8 precision:", t1.nlargest(8, 'y_pred').y_true.mean())""",
    cell_id="s5_code",
))

cells.append(md(
    """## 6. Disposition

The probe phase is **closed**. The candidate Tier-1 replacement at both
scales is *Tier-1 features + GeM context-input Fang-ViT embeddings →
LightGBM* (t1_gem192 if pooled PR-AUC is binding; t1_gem64_gem192 if
per-image skill is). Next per
[HANDOFF_NEXT_SESSION.md](../HANDOFF_NEXT_SESSION.md): productize the
extraction into `src/` (embed arbitrary CTX windows + tests), pre-declare
confirmation gates on the cohort-expansion images **before** seeing any
new-image numbers, then rebuild the Tier-2 calibrated head on the new
feature set. The conditional-leveler fusion recipe is likely obsolete and
is retired formally after the confirmation read. Fine-tuning the ViT stays
deferred.""",
    cell_id="s6_md",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {NB_PATH}")
