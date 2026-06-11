"""W1 Rung 1a — label-shift rescore test (PLAN_ModelUsability.md W1).

For each image, re-join the banked-recipe predictions y_pred(ti, tj) against
label values y_true(ti+di, tj+dj) for offsets di, dj in [-2, +2], and recompute
the per-image meaningful AUC at each offset. If an anti-signal image's AUC
recovers at a nonzero offset, its labels were spatially misaligned (geometry,
not signal). Healthy images give the null distribution for the max-over-25-
offsets inflation, so the anti-signal gains can be judged against chance.

Writes scripts/probes/_w1_shift_rescore.md and a parquet of the full offset
grid for notebook 18.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--artifact-dir", default="models/lightgbm_two_stage_balanced/8c7523615964f5cb/scale_S64_target_boulder_count")
_ap.add_argument("--summary", default="models/_sweep_w0/20260610T221932Z/summary.parquet")
_ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. _postfix")
_args = _ap.parse_args()

ART = Path(_args.artifact_dir)
SUMMARY = Path(_args.summary)
OUT_MD = Path(f"scripts/probes/_w1_shift_rescore{_args.tag}.md")
OUT_PARQUET = Path(f"scripts/probes/_w1_shift_rescore{_args.tag}.parquet")
OFFSETS = range(-2, 3)

pred = pd.read_parquet(ART / "predictions.parquet")
assert (pred.tile_size_px == 64).all()

# Meaningful threshold: read from the W0 summary (bc >= 50 by design); verify.
summ = pd.read_parquet(SUMMARY)
thr_vals = summ.loc[
    (summ.variant == "lightgbm_two_stage_balanced") & (summ.target_col == "boulder_count"),
    "meaningful_threshold",
].unique()
assert len(thr_vals) == 1, thr_vals
THR = float(thr_vals[0])
print(f"meaningful threshold (boulder_count): {THR}")

rows = []
for obs_id, g in pred.groupby("obs_id"):
    labels = g.set_index(["ti", "tj"])["y_true"]
    for di in OFFSETS:
        for dj in OFFSETS:
            # label grid shifted by (di, dj): prediction at (ti, tj) is scored
            # against the label that sat at (ti+di, tj+dj)
            shifted_idx = pd.MultiIndex.from_arrays(
                [g.ti.to_numpy() + di, g.tj.to_numpy() + dj]
            )
            y_true = labels.reindex(shifted_idx).to_numpy()
            ok = ~np.isnan(y_true)
            # strict > to match src/modeling/evaluate.py per_fold_metrics
            y_t = (y_true[ok] > THR).astype(int)
            y_p = g.y_pred.to_numpy()[ok]
            if y_t.sum() == 0 or y_t.sum() == len(y_t):
                auc = np.nan
            else:
                auc = roc_auc_score(y_t, y_p)
            rows.append(
                dict(obs_id=obs_id, di=di, dj=dj, n_overlap=int(ok.sum()),
                     n_pos=int(y_t.sum()), auc=auc)
            )

grid = pd.DataFrame(rows)
grid.to_parquet(OUT_PARQUET, index=False)

center = grid[(grid.di == 0) & (grid.dj == 0)].set_index("obs_id")["auc"]
best = grid.loc[grid.groupby("obs_id")["auc"].idxmax()].set_index("obs_id")
tab = pd.DataFrame(
    {
        "auc_center": center,
        "auc_best": best["auc"],
        "best_di": best["di"],
        "best_dj": best["dj"],
        "gain": best["auc"] - center,
        "n_overlap_best": best["n_overlap"],
    }
).sort_values("auc_center")
tab["anti_signal"] = tab.auc_center < 0.5
tab["recovers_to_gt_0.5"] = tab.anti_signal & (tab.auc_best > 0.5)
tab["recovers_to_gt_0.6"] = tab.anti_signal & (tab.auc_best > 0.6)

healthy_gain = tab.loc[~tab.anti_signal, "gain"]
anti_gain = tab.loc[tab.anti_signal, "gain"]

lines = [
    "# W1 Rung 1a — label-shift rescore test",
    "",
    f"Recipe: two_stage_balanced × boulder_count @ S=64 (`{ART}`); "
    f"meaningful threshold bc > {THR:g} (strict, matching evaluate.py); "
    "offsets di,dj ∈ [-2,+2] (25 cells; "
    "1 tile = 320 m).",
    "",
    "Question: does any anti-signal image's per-image AUC recover when its",
    "label grid is shifted? Recovery at a nonzero offset = geometric",
    "misalignment (rung 1 cause), not absent signal. Healthy images give the",
    "null for max-over-25-offsets inflation.",
    "",
    "```",
    tab.to_string(float_format=lambda v: f"{v:.3f}"),
    "```",
    "",
    f"- Healthy-image (n={len(healthy_gain)}) best-offset gain: "
    f"median {healthy_gain.median():.3f}, max {healthy_gain.max():.3f}",
    f"- Anti-signal (n={len(anti_gain)}) best-offset gain: "
    f"median {anti_gain.median():.3f}, max {anti_gain.max():.3f}",
    f"- Anti-signal recovering past 0.5: {int(tab['recovers_to_gt_0.5'].sum())}/{len(anti_gain)}; "
    f"past 0.6: {int(tab['recovers_to_gt_0.6'].sum())}/{len(anti_gain)}",
    "",
    "Note: best-offset AUC is selected post hoc over 25 cells, so small gains",
    "are expected by chance — judge anti-signal gains against the healthy",
    "null above, and treat only recoveries well past it as geometry evidence.",
]
OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(tab.to_string())
print(f"\nhealthy gain median/max: {healthy_gain.median():.3f}/{healthy_gain.max():.3f}")
print(f"anti gain median/max: {anti_gain.median():.3f}/{anti_gain.max():.3f}")
print(f"wrote {OUT_MD} and {OUT_PARQUET}")
