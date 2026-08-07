"""Build notebooks/23_calibration_diagnostic.ipynb from Python source.

Stage-0 documentation for [PLAN_Calibration.md](../PLAN_Calibration.md): characterize
the dynamic-range compression in both products and preview the post-hoc fixes, all
LOIO-honest, calling `src.calibration`. Reads banked predictions, recomputes nothing
heavy.

Figures: reports/figures/23_{tier1_calibration,tier2_compression,tier2_decompression,
tier2_l1_bakeoff,tier2_scale_sweep,prich_vs_abundance,prich_qmatch_confirm,tier2_label_reweight}.png
To regenerate: `python notebooks/_build_23.py` then nbconvert --execute.

§§5-8 (Stage 2) read the banked probe artifacts under models/fang_tier2/l1_bakeoff/
(written by scripts/probes/_diag_tier2_{l1_bakeoff,scale_sweep}.py) — no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "23_calibration_diagnostic.ipynb"


def md(text, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text, cid):
    return {"cell_type": "code", "id": cid, "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md(
    """# 23 — Calibration & de-compression diagnostic

[PLAN_Calibration.md](../PLAN_Calibration.md) **Stage 0** (diagnosis + post-hoc
preview, §§1-4) **and Stage 2** (the expensive de-compression bake-off, §§5-8). Both
deployed products compress toward the middle; this notebook **characterizes** that
compression, **previews** the post-hoc ranking-preserving fixes, and **tests** whether
retraining can beat the floor. Everything is group-aware leave-image-out (LOIO): the
calibrator/head is fit on the other 37 images and applied to the held-out one, via
`src.calibration.loio_calibrate`.

Headlines (computed below):
- **Tier-1** is already well-calibrated (ECE ≈ 0.06); **isotonic** is the calibrator.
- **Tier-2 quantile-matching** recovers the true value distribution (high tail +
  true-zero mass) with ranking intact; isotonic does not.
- **Stage 2:** *every* retraining lever — L1 distributional heads (HL-Gauss/pinball/
  ZILN), LDS reweighting, `min_confidence` label-cleaning, coarser scale — is a **wash,
  dominated, or harmful** on ranking; a Tier-1 *classifier* ranks abundance as well as
  the dedicated regressor ⇒ **the per-tile ceiling is the data**, not the head.
  Quantile-match (L3) stays the product win.
""", "intro"))

cells.append(code(
    """import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists())
sys.path.insert(0, str(REPO))
from src.calibration import (reliability_curve, expected_calibration_error,
    TemperatureScaler, BetaCalibrator, IsotonicCalibrator, quantile_match,
    compression_metrics, loio_calibrate)

FIG = REPO / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
T1 = REPO / "models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet"
T2 = REPO / "models/fang_tier2/tier2_mlp_reg_emb_fractional_area_S32/1e01ad8b17447599/predictions.parquet"
t1 = pd.read_parquet(T1); t2 = pd.read_parquet(T2)
print(f"Tier-1 classifier preds: {len(t1):,} tiles, {t1.obs_id.nunique()} images")
print(f"Tier-2 mlp_reg preds:    {len(t2):,} tiles, {t2.obs_id.nunique()} images")
""", "setup"))

# ---------------- Tier 1 ----------------
cells.append(md(
    """## 1. Tier-1 — is the rich/poor probability compressed?

A reliability diagram bins tiles by predicted probability and plots the empirical
rich-rate against the bin's mean confidence; on the diagonal = calibrated, ECE is the
count-weighted gap. The raw classifier is mildly **over-dispersed**: above the
diagonal for p<0.5 (under-confident lows) and below for p>0.5 (over-confident highs).

We compare three calibrators (all LOIO, all monotone):
- **temperature scaling** — *one* global parameter (`p' = σ(logit(p)/T)`), strictly
  monotone so ROC-AUC is exact. But one knob can only squeeze uniformly toward 0.5,
  so it fixes the over-confident high end **at the cost of** the low end.
- **isotonic** — a *free* monotone map, so it can bend the two ends independently and
  fix both. **The best calibration here** (lowest ECE). Its flat steps create ties (a
  theoretical ranking risk), but with 161k tiles that cost is negligible — see the AUC
  note.
- **beta calibration** — a smooth 3-parameter *strictly*-monotone map
  ([Kull et al. 2017](https://proceedings.mlr.press/v54/kull17a.html)): no ties, but
  3 parameters underfit the reliability curve, so its ECE lands between temperature
  and isotonic. A smooth fallback if isotonic's step artifacts ever matter.

Note on AUC: a monotone map can't change ranking *within* an image, but the LOIO pool
fits a **different** map per held-out image, which reorders tiles *across* images and
dents the *pooled* AUC. At deployment a **single** global map is fit — AUC-exact for
all three (isotonic +0.0003, temperature/beta +0.0000) — so the LOIO pooled-AUC drop
is an artifact, not a real cost. Verdict: **isotonic** for Tier-1.
""", "t1_md"))

cells.append(code(
    """y = t1.y_true.to_numpy()

def ece_split(p):  # overall, lows(<0.5), highs(>=0.5)
    p = np.clip(p, 0, 1); lo = p < 0.5
    return (expected_calibration_error(y, p),
            expected_calibration_error(y[lo], p[lo]),
            expected_calibration_error(y[~lo], p[~lo]))

temp = loio_calibrate(t1, lambda rp, rt, hp: TemperatureScaler().fit(rp, rt).predict(hp))
iso  = loio_calibrate(t1, lambda rp, rt, hp: IsotonicCalibrator().fit(rp, rt).predict(hp))
beta = loio_calibrate(t1, lambda rp, rt, hp: BetaCalibrator().fit(rp, rt).predict(hp))
T_all = TemperatureScaler().fit(t1.y_pred.to_numpy(), y).T
rows = {"raw": t1.y_pred.to_numpy(), f"temperature (T={T_all:.2f})": temp,
        "isotonic": iso, "beta": beta}
print(f"{'variant':>22} {'ECE':>6} {'ECE_low':>8} {'ECE_high':>9} {'AUC_loio':>9}")
for n, p in rows.items():
    e, el, eh = ece_split(p)
    print(f"{n:>22} {e:>6.3f} {el:>8.3f} {eh:>9.3f} {roc_auc_score(y, np.clip(p,0,1)):>9.4f}")
# deployment case: ONE global map -> strictly-monotone calibrators are AUC-exact
gb = {n: c.fit(t1.y_pred.to_numpy(), y).predict(t1.y_pred.to_numpy())
      for n, c in [("temperature", TemperatureScaler()), ("beta", BetaCalibrator())]}
print("global-fit AUC (deployment): " + "  ".join(
    f"{n} {roc_auc_score(y, p):.4f}" for n, p in gb.items())
    + f"  (raw {roc_auc_score(y, t1.y_pred):.4f}) -> LOIO drop is a per-fold artifact")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for p, lbl, c in [(rows["raw"], "raw", "tab:blue"),
                  (temp, "temperature (1 knob)", "tab:orange"),
                  (iso, "isotonic (flexible)", "tab:green"),
                  (beta, "beta (smooth)", "tab:purple")]:
    conf, acc, _ = reliability_curve(y, np.clip(p, 0, 1), n_bins=12)
    ax[0].plot(conf, acc, "o-", color=c, label=lbl, alpha=0.85)
ax[0].plot([0, 1], [0, 1], "k:", lw=1)
ax[0].set_xlabel("mean predicted P(rich)"); ax[0].set_ylabel("empirical rich-rate")
ax[0].set_title("temperature trades lows for highs; isotonic hugs the diagonal"); ax[0].legend()
ax[1].hist(t1.y_pred, bins=40, color="tab:blue", alpha=0.7)
ax[1].set_xlabel("predicted P(rich)"); ax[1].set_ylabel("tiles")
ax[1].set_title(f"probability spread (std {t1.y_pred.std():.2f}) — not collapsed")
fig.tight_layout(); fig.savefig(FIG / "23_tier1_calibration.png", dpi=130)
print("\\nVerdict: Tier-1 is near-calibrated; ISOTONIC fixes both ends best (ECE -> 0.014) "
      "and is AUC-exact at deployment. Beta is a smooth fallback; temperature trades the ends.")
""", "t1_code"))

# ---------------- Tier 2 characterize ----------------
cells.append(md(
    """## 2. Tier-2 — the two-sided compression

The abundance regressor minimizes MSE on a heavy-tailed, zero-inflated target under
a 5 m/px texture floor → it hedges to the conditional mean. The per-bin curve below
shows it **over-predicts the low end** (floors above true zero) and **under-predicts
the high tail**, crossing the diagonal near the rich/poor threshold; the marginal
histogram shows the predicted distribution is far narrower than the truth.
""", "t2_md"))

cells.append(code(
    """yt = t2.y_true.to_numpy(); yp = np.clip(t2.y_pred.to_numpy(), 0, None)
edges = np.array([0, 1e-4, 1e-3, 1e-2, 3e-2, 1.0])
lab = np.clip(np.digitize(yt, edges) - 1, 0, len(edges) - 2)
bt = np.array([yt[lab == b].mean() for b in range(len(edges) - 1)])
bp = np.array([yp[lab == b].mean() for b in range(len(edges) - 1)])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].plot(bt, bp, "o-", color="tab:red", label="mlp_reg")
ax[0].plot([0, bt.max()], [0, bt.max()], "k:", label="perfect")
ax[0].axvline(1e-2, color="grey", ls="--", lw=0.8, label="rich/poor 1e-2")
ax[0].set_xlabel("mean TRUE area-fraction (bin)"); ax[0].set_ylabel("mean PRED")
ax[0].set_title("per-bin: over-predicts lows, under-predicts highs"); ax[0].legend(fontsize=8)
bins = np.logspace(-4, np.log10(max(yt.max(), 1e-3)), 40)
ax[1].hist(np.clip(yt, 1e-4, None), bins=bins, alpha=0.55, label="TRUE", color="k")
ax[1].hist(np.clip(yp, 1e-4, None), bins=bins, alpha=0.55, label="PRED", color="tab:red")
ax[1].set_xscale("log"); ax[1].set_xlabel("area-fraction (log)"); ax[1].set_ylabel("tiles")
ax[1].set_title("marginal: prediction is too narrow"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "23_tier2_compression.png", dpi=130)

m = compression_metrics(yt, yp)
print(f"raw mlp_reg: spearman {m['spearman']:.3f}  top_ratio {m['top_ratio']:.3f}  "
      f"near_zero pred {m['near_zero_pred']:.1%} vs true {m['near_zero_true']:.1%}  "
      f"marginal_L1 {m['marginal_l1']:.4f}")
""", "t2_code"))

# ---------------- Tier 2 decompress ----------------
cells.append(md(
    """## 3. Tier-2 — de-compression preview (LOIO)

Two monotone (ranking-preserving) post-hoc calibrators, fit on the other 37 images:
**isotonic** (fits pred→E[true]) and **quantile-matching** (maps the prediction
distribution onto the truth distribution). Quantile-matching recovers the marginal
by construction; isotonic fits the conditional mean — which *is* the compressed
quantity, so it cannot un-squash the tail.
""", "t2d_md"))

cells.append(md(
    """### What is quantile-matching?

Quantile-matching (a.k.a. the quantile transform, or *histogram matching* in image
processing) replaces each prediction with the **truth value at the same rank**. Fit
on the training images:

1. Sort the training predictions and the training truths, separately.
2. Pair them by rank — smallest prediction ↦ smallest truth, median ↦ median,
   99th-percentile ↦ 99th-percentile, … — which defines a monotone lookup `g`.
3. At inference, push each new prediction `v` through `g`: find the fraction of
   training predictions it exceeds (its quantile), and emit the truth value at that
   quantile (linearly interpolated between the stored rank points).

In code this is just
`quantile_match(v) = np.interp(v, np.sort(ref_pred), np.sort(ref_true))`.

**Why it de-compresses.** By construction the calibrated outputs have the *same
distribution* as the truth — the same fraction of exact zeros and the same heavy
upper tail. The model's squashed range (~0.005–0.04) is stretched back onto the true
range (0–0.29). That is why, at right, the green marginal lands on the gray truth and
the 18% true-zero mass reappears.

**Why ranking is untouched.** `g` is non-decreasing, so it never reorders tiles — the
tile the model ranked highest stays highest. Spearman / NDCG / AUC are therefore
invariant (the small wobble here is only from fitting a *different* `g` per held-out
fold and pooling).

**What it cannot do (the honest limit).** It fixes the *distribution of values*, not
*which tile gets which value*. A genuinely high tile that the model mis-ranks into the
middle still receives a middle value, so the high per-bin point lands near — not on —
the diagonal (top-bin ratio ≈ 0.87). That residual is the ranking / aleatoric floor,
not calibration. And it matches the *global* cohort marginal, so it assumes a new
image resembles the cohort; a truly boulder-poor scene would have its lows wrongly
lifted — handled by fitting per-region or gating on the §2.7 novelty flag (see the
plan).
""", "t2d_how"))

cells.append(code(
    """iso = loio_calibrate(t2, lambda rp, rt, hp: IsotonicCalibrator().fit(rp, rt).predict(hp))
qm  = loio_calibrate(t2, lambda rp, rt, hp: quantile_match(hp, rp, rt))
rows = {"raw mlp_reg": compression_metrics(yt, yp),
        "+ isotonic": compression_metrics(yt, iso),
        "+ quantile-match": compression_metrics(yt, qm)}
scorecard = pd.DataFrame(rows).T[["spearman", "top_ratio", "near_zero_pred", "marginal_l1"]]
print(scorecard.round(3).to_string())
print(f"\\n(true near-zero share = {rows['raw mlp_reg']['near_zero_true']:.1%}; "
      "goal: top_ratio->1, near_zero_pred->that, marginal_l1->0, spearman flat)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for y, lbl, c in [(yp, "raw", "tab:red"), (qm, "quantile-matched", "tab:green")]:
    bp_ = np.array([np.clip(y, 0, None)[lab == b].mean() for b in range(len(edges) - 1)])
    ax[0].plot(bt, bp_, "o-", color=c, label=lbl)
ax[0].plot([0, bt.max()], [0, bt.max()], "k:", label="perfect")
ax[0].set_xlabel("mean TRUE (bin)"); ax[0].set_ylabel("mean PRED")
ax[0].set_title("quantile-match lands the high bin on the diagonal"); ax[0].legend(fontsize=8)
ax[1].hist(np.clip(yt, 1e-4, None), bins=bins, alpha=0.5, label="TRUE", color="k")
ax[1].hist(np.clip(qm, 1e-4, None), bins=bins, alpha=0.5, label="quantile-matched", color="tab:green")
ax[1].set_xscale("log"); ax[1].set_xlabel("area-fraction (log)"); ax[1].set_ylabel("tiles")
ax[1].set_title("calibrated marginal now matches truth"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "23_tier2_decompression.png", dpi=130)
""", "t2d_code"))

cells.append(md(
    """## 4. Stage 0 verdict — what this tells the plan

- **Tier-1** is already well-calibrated (ECE ≈ 0.06, well-spread probabilities);
  **isotonic** trims it the rest of the way (ECE → 0.014, AUC-exact at deployment) —
  a small, safe refinement. *Not* the problem — the headline "mostly rich" maps over
  boulder-rich regions are largely correct.
- **Tier-2** compression is real and two-sided, and the banked structural variants
  (Tweedie, two-stage) compress *more*, not less — it is intrinsic to MSE-on-skewed
  + the texture floor.
- **Quantile-matching** is the Stage-1 win: it recovers the true value distribution
  (high tail + true-zero mass, marginal_L1 → 0) while preserving ranking. **Isotonic
  does not help** (it fits the compressed conditional mean).
- The residual (top-bin ratio ≈ 0.87, not 1.0) is the **ranking / aleatoric floor**:
  matching the marginal cannot place a mis-ranked high tile at the top. That residual
  is what the plan's later stages (better objectives; honest per-tile uncertainty)
  target. See [PLAN_Calibration.md](../PLAN_Calibration.md) for the staged plan and
  the marginal-vs-per-tile-vs-uncertainty framing.

Figures: `reports/figures/23_{tier1_calibration,tier2_compression,tier2_decompression}.png`.
""", "synth"))

# ---------------- Stage 2: L1 bake-off ----------------
cells.append(md(
    """## 5. Stage 2 — L1 distributional bake-off: a wash on ranking

Stage 0 ruled out the *cheap* L1 swaps (log1p, count-Poisson). Stage 2 tests the
*heavy* L1 lever — losses whose optimum is **not** the arithmetic mean, each emitting
a full per-tile distribution so we can also read a non-mean summary (a high quantile,
the mode):

- **HL-Gauss** — histogram loss over Gaussian-smoothed bins; read the distribution's
  mean / mode / P90.
- **pinball** — multi-output P10/P50/P90 quantile regression; the median is a robust
  point, [P10,P90] is the L4 interval.
- **neural ZILN** — zero-inflated log-normal NLL (π, μ, σ); read mixture mean / median
  / P90.

Same emb-only S=32 LOIO. Each point readout is scored by **raw `top_ratio`** (tail
de-compression *without* L3) and by **paired per-image Spearman vs `mlp_reg`** (the
honest must-not-regress test — the median-of-medians glance misleads). The table
below: no head beats `mlp_reg` on rank (best p=0.48); the only keeper is
`pinball.P90`, whose raw `top_ratio ≈ 0.98` gives a tail-calibrated point with no L3
and no ranking cost. The [P10,P90] intervals are under-dispersed (~58 % vs 80 %).
""", "t2_l1_md"))

cells.append(code(
    """import json
from scipy.stats import spearmanr, wilcoxon

BO = REPO / "models/fang_tier2/l1_bakeoff"
base = pd.read_parquet(BO / "preds_mlp_reg.parquet")   # obs_id, ti, tj, y_true(=fa), point

def per_img_rho(df, col):
    return {o: spearmanr(g.y_true, g[col]).correlation for o, g in df.groupby("obs_id")
            if g.y_true.nunique() > 1 and g[col].nunique() > 1}

b = per_img_rho(base, "point")
base_top = compression_metrics(base.y_true, base.point)["top_ratio"]
print(f"{'readout':>16} {'raw_top':>8} {'perimg_rho':>11} {'paired_d':>9} {'wilcoxon_p':>11}")
print(f"{'mlp_reg.mean':>16} {base_top:>8.2f} {np.nanmedian(list(b.values())):>11.3f} {'--':>9} {'--':>11}")

bars = {"mlp_reg.mean": base_top}
for h, cols in {"hlgauss": ["mean", "mode", "p90"], "pinball": ["median", "p90"],
                "ziln": ["mean", "median", "p90"]}.items():
    d = pd.read_parquet(BO / f"preds_{h}.parquet")
    for c in cols:
        top = compression_metrics(d.y_true, d[c])["top_ratio"]
        rr = per_img_rho(d, c)
        keys = [k for k in b if k in rr and np.isfinite(b[k]) and np.isfinite(rr[k])]
        bb = np.array([b[k] for k in keys]); cc = np.array([rr[k] for k in keys])
        print(f"{h+'.'+c:>16} {top:>8.2f} {np.nanmedian(cc):>11.3f} "
              f"{np.median(cc-bb):>+9.3f} {wilcoxon(cc, bb).pvalue:>11.3f}")
        bars[f"{h}.{c}"] = top

cov = json.loads((BO / "coverage.json").read_text())
print("\\n[P10,P90] interval coverage (nominal 80%):  " +
      "   ".join(f"{r['readout']} {r['coverage_p10_p90']:.0%}" for r in cov))

fig, ax = plt.subplots(figsize=(9.5, 4.2))
names = list(bars); colors = ["tab:red" if n == "mlp_reg.mean" else "tab:blue" for n in names]
ax.bar(names, [bars[n] for n in names], color=colors)
ax.axhline(1.0, ls=":", c="k", label="uncompressed (1.0)")
ax.axhline(base_top, ls="--", c="tab:red", lw=0.8, label=f"mlp_reg mean ({base_top:.2f})")
ax.set_ylabel("raw top_ratio"); ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha="right")
ax.set_title("raw tail de-compression by readout (pinball.P90 ~ 1.0) -- but all wash on rank")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "23_tier2_l1_bakeoff.png", dpi=130)
""", "t2_l1_code"))

# ---------------- Stage 2: L2 scale sweep ----------------
cells.append(md(
    """## 6. L2 scale sweep — coarser narrows the marginal, ranking only directional

The one lever the theory says can raise the *ranking* ceiling is shrinking `p(y|x)`:
coarser tiles average over more area → higher SNR. We re-run the same `mlp_reg` at
S=64 (320 m) vs the frozen S=32 (160 m). The marginal de-compresses (raw `top_ratio`
0.66→0.72; pooled rho up), but the per-image ranking gain is only **directional** —
paired Δ +0.025, Wilcoxon p=0.19, not significant at n=38 — and it is partly an
**easier-target artefact** (the true-zero share drops 18 %→6.9 %, so coarse tiles are
a less zero-inflated, easier ranking problem). Honest read: coarsening probably helps
and the Tier-2 map *may* run coarser than Tier-1, but a confident ranking gain needs
the expansion cohort.
""", "scale_md"))

cells.append(code(
    """from scipy.stats import spearmanr, wilcoxon

sw = pd.read_csv(REPO / "models/fang_tier2/l1_bakeoff/scale_sweep.csv")
print(sw.round(4).to_string(index=False))

s32 = pd.read_parquet(REPO / "models/fang_tier2/l1_bakeoff/preds_mlp_reg.parquet")
s64 = pd.read_parquet(REPO / "models/fang_tier2/l1_bakeoff/preds_mlp_reg_S64.parquet")

def per_img(df):
    return {o: spearmanr(g.y_true, g.point).correlation for o, g in df.groupby("obs_id")
            if g.y_true.nunique() > 1}

a, b64 = per_img(s32), per_img(s64)
keys = [k for k in a if k in b64 and np.isfinite(a[k]) and np.isfinite(b64[k])]
aa = np.array([a[k] for k in keys]); bb = np.array([b64[k] for k in keys])
pval = wilcoxon(bb, aa).pvalue
print(f"\\npaired per-image rho  S=32 vs S=64:  n={len(keys)}  paired dmed={np.median(bb-aa):+.3f}"
      f"  S64 wins {int((bb>aa).sum())}/{len(keys)}  Wilcoxon p={pval:.3f}  "
      f"({'NOT ' if pval>=0.05 else ''}significant)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].bar(["S=32\\n(160 m)", "S=64\\n(320 m)"], sw.raw_top_ratio, color=["tab:red", "tab:orange"])
ax[0].axhline(1.0, ls=":", c="k"); ax[0].set_ylabel("raw top_ratio (1.0 = uncompressed)")
ax[0].set_title("coarser tile -> less raw compression of the marginal")
ax[1].scatter(aa, bb, s=20, alpha=0.7)
lim = [min(aa.min(), bb.min()) - 0.02, max(aa.max(), bb.max()) + 0.02]
ax[1].plot(lim, lim, "k:"); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("per-image rho  S=32"); ax[1].set_ylabel("per-image rho  S=64")
ax[1].set_title(f"ranking gain only directional (p={pval:.2f}, n.s.)")
fig.tight_layout(); fig.savefig(FIG / "23_tier2_scale_sweep.png", dpi=130)
""", "scale_code"))

# ---------------- Stage 2: P(rich) vs abundance ----------------
cells.append(md(
    """## 7. Tier-1 P(rich) ≈ Tier-2 regressor as a magnitude ranker — the wall is the data

The cleanest evidence that the per-tile ceiling is set by the *inputs*, not the model:
the Tier-1 rich/poor **classifier** — which never saw the continuous target — ranks
`fractional_area` per-image about as well as the head built specifically to predict it.
Within the rich class the correlation falls (texture barely resolves *how* rich). Two
different model families hitting the same ~0.43 wall ⇒ the magnitude signal in 5 m/px
CTX essentially *is* the rich/poor signal. (Practical upshot: a calibrated `P(rich)`
pushed through quantile-match would approximate the Tier-2 abundance map — a one-model
simplification, tested directly in §7b.)
""", "prich_md"))

cells.append(code(
    """from scipy.stats import spearmanr

t1p = pd.read_parquet(REPO / "models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet").rename(columns={"y_pred": "p_rich"})
parts = []
for p in (REPO / "dataset_v2/labels").glob("*.parquet"):
    d = pd.read_parquet(p)
    parts.append(d[d.tile_size_px == 32][["obs_id", "ti", "tj", "fractional_area", "boulder_count"]])
lab = pd.concat(parts, ignore_index=True)
j = t1p.merge(lab, on=["obs_id", "ti", "tj"]).merge(base[["obs_id", "ti", "tj", "point"]], on=["obs_id", "ti", "tj"])

def pim(col, tcol, mask=None):
    s = j if mask is None else j[mask]
    r = [spearmanr(g[tcol], g[col]).correlation for _, g in s.groupby("obs_id")
         if g[tcol].nunique() > 1 and g[col].nunique() > 1]
    return float(np.nanmedian(r))

rich = j.fractional_area > 1e-2
print(f"P(rich) vs fractional_area  per-image: {pim('p_rich','fractional_area'):.3f}"
      f"   (within rich tiles only: {pim('p_rich','fractional_area', rich):.3f})")
print(f"P(rich) vs boulder_count    per-image: {pim('p_rich','boulder_count'):.3f}")
print(f"\\nAs fractional_area rankers:  P(rich) {pim('p_rich','fractional_area'):.3f}"
      f"   vs   mlp_reg {pim('point','fractional_area'):.3f}   (a dead heat)")

vals = [pim("p_rich", "fractional_area"), pim("point", "fractional_area"),
        pim("p_rich", "fractional_area", rich)]
fig, ax = plt.subplots(figsize=(5.4, 4))
ax.bar(["P(rich)\\n(classifier)", "mlp_reg\\n(regressor)", "P(rich)\\nwithin rich"],
       vals, color=["tab:green", "tab:red", "tab:gray"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
ax.set_ylabel("per-image rho vs fractional_area")
ax.set_title("the magnitude signal IS the rich/poor signal")
fig.tight_layout(); fig.savefig(FIG / "23_prich_vs_abundance.png", dpi=130)
""", "prich_code"))

cells.append(md(
    """### 7b. Confirming the one-model simplification (direct test)

The inference made flesh: push each input through the **same LOIO quantile-match**
onto the `fractional_area` marginal and score the resulting abundance maps. By
construction both hit the **identical marginal** (top_ratio ≈ 0.86, near-zero 18.6 %,
marginal-L1 0.0002), so the only open question is ranking. Result: the maps are **close
but not identical — the dedicated regressor keeps a small, borderline-significant edge**
(pooled 0.642 vs 0.625; paired per-image the regressor wins 24/38, Wilcoxon p ≈ 0.05).
So replacing the Tier-2 head with a quantile-matched `P(rich)` is **viable at a ~0.02
ranking cost, not free**. (Quantile-match also ties the lowest ~18 % of predictions at
the zero floor, which slightly lowers per-image Spearman for both inputs — and the
median-of-medians glance misleads again: it favours `P(rich)` while the paired test
favours the regressor.)
""", "prich_qm_md"))

cells.append(code(
    """from scipy.stats import spearmanr, wilcoxon

def qmatch_abundance(src_col):
    df = pd.DataFrame({"obs_id": j.obs_id, "y_true": j.fractional_area, "y_pred": j[src_col]})
    return loio_calibrate(df, lambda rp, rt, hp: quantile_match(hp, rp, rt))

yt = j.fractional_area.to_numpy(); grp = j.obs_id.to_numpy()
qa = {"qmatch(P_rich)": qmatch_abundance("p_rich"), "qmatch(mlp_reg)": qmatch_abundance("point")}

def per_img(yp):
    return {o: spearmanr(yt[grp == o], yp[grp == o]).correlation
            for o in np.unique(grp) if np.unique(yt[grp == o]).size > 1}

print(f"{'abundance map':>16} {'perimg_rho':>11} {'pooled':>7} {'top':>5} {'near0':>6} {'margL1':>7}")
for name, q in qa.items():
    m = compression_metrics(yt, q)
    print(f"{name:>16} {np.nanmedian(list(per_img(q).values())):>11.3f} {m['spearman']:>7.3f} "
          f"{m['top_ratio']:>5.2f} {m['near_zero_pred']:>6.1%} {m['marginal_l1']:>7.4f}")

a = per_img(qa["qmatch(P_rich)"]); b = per_img(qa["qmatch(mlp_reg)"])
keys = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
aa = np.array([a[k] for k in keys]); bb = np.array([b[k] for k in keys])
pval = wilcoxon(aa, bb).pvalue
print(f"\\npaired per-image rho  qmatch(P_rich) vs qmatch(mlp_reg): n={len(keys)} "
      f"dmed={np.median(aa-bb):+.3f}  P_rich wins {int((aa>bb).sum())}/{len(keys)}  Wilcoxon p={pval:.3f}")
print("-> identical marginal by construction; the regressor keeps a small, borderline ranking edge.")

fig, ax = plt.subplots(figsize=(5, 4.6))
ax.scatter(bb, aa, s=22, alpha=0.75)
lim = [min(aa.min(), bb.min()) - 0.02, max(aa.max(), bb.max()) + 0.02]
ax.plot(lim, lim, "k:"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("per-image rho  qmatch(mlp_reg)"); ax.set_ylabel("per-image rho  qmatch(P_rich)")
ax.set_title(f"one-model simplification: near-equal (paired p={pval:.2f})")
fig.tight_layout(); fig.savefig(FIG / "23_prich_qmatch_confirm.png", dpi=130)
""", "prich_qm_code"))

cells.append(md(
    """## 8. Stage 2b/2c — label-noise & reweighting (both negative)

Two more levers, same LOIO + paired-Wilcoxon discipline:
- **LDS reweighting (Stage 2c, L1+L2):** up-weight the rare high tail by inverse
  smoothed label density. It *does* de-compress the **raw** marginal (top_ratio
  0.67→0.88 as the weighting sharpens) — but buys it with a **significant ranking
  loss** (paired p ≈ 0.015–0.018). Since quantile-match fixes the marginal for free,
  reweighting is **strictly dominated**.
- **min_confidence label-noise (Stage 2b, L2):** regenerate the Stage-4 labels keeping
  only detections with BoulderNet `score ≥ t` (cached Stage-1/2/3 inputs; tile grid is
  detection-independent so the embeddings still join).
  > ⚠ **AMENDED 2026-08-06 (review finding R56). The verdict printed below is a
  > two-factor comparison and is withdrawn.** Each arm was trained on labels regenerated
  > at `score ≥ t` **and scored against those same labels**, so the paired Wilcoxon varied
  > both the predictor *and* the target. Re-scored against **one common target** (the
  > unfiltered `fractional_area`): `conf≥0.5` is a **null** on every project metric
  > (`meaningful_auc` −0.0028 p=0.31; `pr_auc@1e-2` −0.0007 p=0.44; `precision@5%` 0.0000
  > p=0.41; Spearman −0.0034 p=0.43; n=38) — **82 % of the recorded −0.021 is the target
  > moving, not the model**. `conf≥0.7` is directionally harmful but survives no Holm
  > correction. "Monotonically" is contradicted by the scorecard printed below
  > (per-image ρ 0.4333 → **0.4563** → 0.3044), and the `top_ratio` collapse is largely a
  > population artefact: at a matched 36 % top fraction it is 0.664 → 0.623 → 0.519, and
  > after the **shipped** quantile-match layer it is 0.870 → 0.859 → 0.829. What survives
  > is that filtering **thins** the target rather than cleaning it, so nothing shows
  > filtering *helps* and `min_confidence: null` stays. See DECISIONS 2026-08-06b.
""", "label_rw_md"))

cells.append(code(
    """from scipy.stats import spearmanr, wilcoxon
BO = REPO / "models/fang_tier2/l1_bakeoff"

def paired_vs(base_df, df):
    def pim(d):
        return {o: spearmanr(g.y_true, g.y_pred).correlation for o, g in d.groupby("obs_id")
                if g.y_true.nunique() > 1}
    a, b = pim(df), pim(base_df)
    k = [x for x in b if x in a and np.isfinite(a[x]) and np.isfinite(b[x])]
    aa = np.array([a[x] for x in k]); bb = np.array([b[x] for x in k])
    return np.median(aa - bb), int((aa > bb).sum()), len(k), wilcoxon(aa, bb).pvalue

rw = pd.read_csv(BO / "reweight_scorecard.csv"); mc = pd.read_csv(BO / "minconf_scorecard.csv")
print("LDS reweighting (vs unweighted):")
print(rw.round(3).to_string(index=False))
rwb = pd.read_parquet(BO / "preds_reweight_none.parquet")
for s in ["lds_sqrt", "lds_inv"]:
    d, w, n, p = paired_vs(rwb, pd.read_parquet(BO / f"preds_reweight_{s}.parquet"))
    print(f"   {s}: paired d={d:+.3f}  wins {w}/{n}  p={p:.3f}")

print("\\nmin_confidence label-noise (vs unfiltered):")
print(mc.round(3).to_string(index=False))
mcb = pd.read_parquet(BO / "preds_minconf_none.parquet")
for s in ["conf050", "conf070"]:
    d, w, n, p = paired_vs(mcb, pd.read_parquet(BO / f"preds_minconf_{s}.parquet"))
    print(f"   {s}: paired d={d:+.3f}  wins {w}/{n}  p={p:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].plot(rw.scheme, rw.raw_top, "o-", color="tab:red", label="raw top_ratio")
ax[0].plot(rw.scheme, rw.raw_perimg_rho, "s-", color="tab:blue", label="per-img rho")
ax[0].axhline(1.0, ls=":", c="grey"); ax[0].set_ylim(0, 1.1)
ax[0].set_title("LDS reweighting: de-compress bought with ranking"); ax[0].legend(fontsize=8)
ax[1].plot(mc.label, mc.raw_top, "o-", color="tab:red", label="raw top_ratio")
ax[1].plot(mc.label, mc.raw_perimg_rho, "s-", color="tab:blue", label="per-img rho")
ax[1].plot(mc.label, mc.rich_share, "^--", color="tab:green", label="rich share")
ax[1].set_ylim(0, 1.0); ax[1].set_title("min_confidence: rho non-monotone; top_ratio confounded by rich share"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "23_tier2_label_reweight.png", dpi=130)
""", "label_rw_code"))

cells.append(md(
    """## 9. Stage 2 verdict — the per-tile ceiling is the data

- **L1 is ruled out as a ranking lever.** The heavy distributional heads
  (HL-Gauss, pinball, neural-ZILN) join the cheap swaps (log1p, count-Poisson) in
  washing out vs `mlp_reg` per-image (best p=0.48) — exactly what "compression = the
  intrinsic aleatoric floor, not a loss-shape artefact" predicts. Changing the
  targeted functional moves the *value* (de-compresses), never the *rank*.
- **Every L2/2c lever is now exhausted, none beats `mlp_reg`:** coarser scale (S=64)
  is directional-only (paired +0.025, p=0.19); LDS reweighting is **dominated** (de-
  compresses raw, costs ranking p≈0.015); `min_confidence` filtering is
  ~~**harmful** (monotonically worse, conf≥0.7 paired p<0.001)~~ **a null at conf≥0.5 on a
  common target (R56, amended 2026-08-06) — it neither helps nor hurts.** The ~0.43
  per-image ceiling is the **5 m/px CTX magnitude floor**, ~~confirmed five independent
  ways~~ **on the frozen Fang-ViT/GeM-96/S=32 embedding — all five "ways" hold that
  representation fixed (R55), and this one has now been withdrawn as evidence.**
- **Independent proof the wall is the data:** Tier-1 `P(rich)` — a classifier — ranks
  abundance as well as the dedicated regressor (0.437 vs 0.433). The one-model
  simplification (qmatch the `P(rich)` instead of running a Tier-2 head) is **viable but
  not free** — §7b: identical marginal, regressor keeps a ~0.02 / paired p≈0.05 ranking
  edge.
- **So the product story holds:** quantile-match (L3) is the marginal win; the per-tile
  residual is the texture floor, reported honestly via L4 (intervals need recalibration,
  58 % vs 80 % coverage). **The path forward is not a better model** — it is shipping
  Stage 1 (productize qmatch + isotonic into the map) and the §2.3 expansion cohort
  (the only thing that can raise the *ranking* ceiling).

| lever | result |
|---|---|
| L1 cheap swaps (log1p, count) | wash |
| L1 distributional heads (HL-Gauss/pinball/ZILN) | wash |
| L1+L2 reweighting (LDS) | dominated |
| L2 label-noise (min_confidence) | ~~harmful~~ **null at conf≥0.5 (R56, amended 2026-08-06)** |
| L2 coarser scale (S=64) | directional only (p=0.19) |

Figures: `reports/figures/23_{tier2_l1_bakeoff,tier2_scale_sweep,prich_vs_abundance,prich_qmatch_confirm,tier2_label_reweight}.png`.
""", "stage2_synth"))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)",
                                   "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {NB_PATH}")
