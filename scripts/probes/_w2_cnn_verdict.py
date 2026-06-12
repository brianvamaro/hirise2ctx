"""W2 Phase 1 verdict: CNN augmentation cells vs the tabular baselines.

Promotion gates (PLAN_CNN.md §4.1, declared before the grid ran): the
augmented CNN must beat the tabular baseline by pooled PR-AUC +0.03 OR
median per-image AUC +0.05 on validity-passing images, paired Wilcoxon
p < 0.05. Mechanism check (H-B): the aug-vs-no-aug contrast must improve
the dossier's distribution_shift images specifically. Honest prior: the
texture_decorrelated images should NOT improve (if they do, suspect leakage).

Baselines:
  - Tier 1 classifier (same target fa_gt_1e-2 -> clean comparison):
    models/_sweep_binary/20260611T214042Z
  - banked GBM recipe (boulder_count; bc>=50 vs fa>1e-2 positive-definition
    caveat applies to cross-target deltas): models/_sweep_w0/20260611T054855Z

Usage: python _w2_cnn_verdict.py <cnn_sweep_dir e.g. models/_sweep_cnn/TS>
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats

TIER1_SUMMARY = REPO_ROOT / "models/_sweep_binary/20260611T214042Z/summary.parquet"
TIER1_PREDS = REPO_ROOT / ("models/lightgbm_classification/99de85c1ad2a72e6/"
                           "scale_S64_tfa_gt_1e-2/predictions.parquet")
GBM_SUMMARY = REPO_ROOT / "models/_sweep_w0/20260611T054855Z/summary.parquet"
DOSSIER = REPO_ROOT / "dataset_v2/w1_dossier.parquet"

GATE_PR_AUC = 0.03
GATE_MEDIAN_AUC = 0.05
ALPHA = 0.05


def pooled_pr_auc_from_preds(df: pd.DataFrame, y_true_col: str = "y_true") -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(df[y_true_col].to_numpy().astype(int),
                                         df["y_pred"].to_numpy()))


def tier1_pooled_pr_auc() -> float:
    """Tier 1 predictions carry no y_true; join fa>1e-2 truth from the packaged table."""
    preds = pd.read_parquet(TIER1_PREDS, columns=["obs_id", "ti", "tj", "y_pred"])
    truth = pd.read_parquet(
        REPO_ROOT / "dataset_v2/packaged/loio_nfold/all.parquet",
        columns=["obs_id", "scale_idx", "ti", "tj", "fractional_area"])
    truth = truth[truth.scale_idx == 3]
    j = preds.merge(truth, on=["obs_id", "ti", "tj"], how="left", validate="one_to_one")
    assert j.fractional_area.notna().all(), "Tier1 prediction tiles missing from all.parquet"
    j["y_true"] = (j.fractional_area > 1e-2).astype(int)
    return pooled_pr_auc_from_preds(j)


def paired(d: pd.Series) -> tuple[float, float, float, float]:
    """(mean, median, win rate, wilcoxon p) of a paired-delta series."""
    d = d.dropna()
    try:
        p = stats.wilcoxon(d, zero_method="wilcox").pvalue
    except ValueError:
        p = float("nan")
    return float(d.mean()), float(d.median()), float((d > 0).mean()), float(p)


def main() -> int:
    sweep_dir = Path(sys.argv[1])
    cnn = pd.read_parquet(sweep_dir / "summary.parquet")
    agg = pd.read_parquet(sweep_dir / "aggregate.parquet")

    dossier = pd.read_parquet(DOSSIER)
    shift_imgs = sorted(dossier[dossier.attributed_cause == "distribution_shift"].index)
    decorr_imgs = sorted(dossier[dossier.attributed_cause == "texture_decorrelated"].index)
    validity_ok = set(dossier[dossier.validity_ok].index) if "validity_ok" in dossier else None

    t1 = pd.read_parquet(TIER1_SUMMARY).set_index("held_out_obs_id")
    gbm = pd.read_parquet(GBM_SUMMARY)
    gbm = gbm[(gbm.variant == "lightgbm_two_stage_balanced")
              & (gbm.target_col == "boulder_count")].set_index("held_out_obs_id")
    t1_pooled = tier1_pooled_pr_auc()

    print(f"CNN sweep: {sweep_dir}")
    print(f"Baselines: Tier1 pooled PR-AUC={t1_pooled:.4f}  "
          f"GBM banked mean PR-AUC={gbm['pr_auc'].mean():.4f} (bc>=50 caveat)")
    print(f"Dossier classes: distribution_shift={shift_imgs}  "
          f"texture_decorrelated={decorr_imgs}\n")

    cells = list(dict.fromkeys(cnn["aug_cell"]))
    cell_a = cnn[cnn.aug_cell == "none"].set_index("held_out_obs_id") if "none" in cells else None

    for cell in cells:
        g = cnn[cnn.aug_cell == cell].set_index("held_out_obs_id")
        arow = agg[agg.aug_cell == cell].iloc[0]
        print(f"================ cell {cell} ================")
        print(f"  aggregate: auc mean={arow['auc_mean']:+.4f} median={arow['auc_median']:+.4f}  "
              f"pooled_pr_auc={arow['pooled_pr_auc']:.4f}  "
              f"pooled_prec@5%={arow['pooled_precision_at_top_5pct']:.4f}")

        for name, base, base_auc_col in (("Tier1 (same target)", t1, "auc"),
                                         ("GBM banked (cross-target)", gbm, "meaningful_auc")):
            common = g.index.intersection(base.index)
            d_auc = g.loc[common, "auc"] - base.loc[common, base_auc_col]
            d_pr = g.loc[common, "pr_auc"] - base.loc[common, "pr_auc"]
            if validity_ok is not None:
                vmask = [o in validity_ok for o in common]
                d_auc_v = d_auc[vmask]
            else:
                d_auc_v = d_auc
            m, md, w, p = paired(d_auc_v)
            m2, md2, w2, p2 = paired(d_pr)
            print(f"  vs {name} (n={len(common)}, validity-passing n={d_auc_v.notna().sum()}):")
            print(f"    d per-image AUC (validity): mean={m:+.4f} median={md:+.4f} "
                  f"win={w:.2f} p={p:.4f}")
            print(f"    d per-fold PR-AUC:          mean={m2:+.4f} median={md2:+.4f} "
                  f"win={w2:.2f} p={p2:.4f}")
            med_base = base.loc[common, base_auc_col][[o in validity_ok for o in common]].median() \
                if validity_ok is not None else base.loc[common, base_auc_col].median()
            med_cnn = g.loc[common, "auc"][[o in validity_ok for o in common]].median() \
                if validity_ok is not None else g.loc[common, "auc"].median()
            gate_auc = (md >= GATE_MEDIAN_AUC) and (p < ALPHA)
            if name.startswith("Tier1"):
                pr_delta_pooled = arow["pooled_pr_auc"] - t1_pooled
                gate_pr = (pr_delta_pooled >= GATE_PR_AUC) and (p2 < ALPHA)
                print(f"    median AUC {med_base:.3f} -> {med_cnn:.3f}; "
                      f"pooled PR-AUC delta {pr_delta_pooled:+.4f}")
                print(f"    GATE: pr_auc {'PASS' if gate_pr else 'fail'} / "
                      f"median_auc {'PASS' if gate_auc else 'fail'} -> "
                      f"{'** PROMOTABLE **' if (gate_pr or gate_auc) else 'no'}")
            else:
                print(f"    median AUC {med_base:.3f} -> {med_cnn:.3f} (caveat: bc>=50 baseline)")

        if cell_a is not None and cell != "none":
            common = g.index.intersection(cell_a.index)
            d = g.loc[common, "auc"] - cell_a.loc[common, "auc"]
            m, md, w, p = paired(d)
            print(f"  vs cell A (aug contrast, n={len(common)}): "
                  f"mean={m:+.4f} median={md:+.4f} win={w:.2f} p={p:.4f}")
            print("    mechanism check -- distribution_shift images (A -> this cell):")
            for obs in shift_imgs:
                if obs in common:
                    print(f"      {obs}: {cell_a.loc[obs, 'auc']:.3f} -> "
                          f"{g.loc[obs, 'auc']:.3f} ({d.loc[obs]:+.3f})")
            print("    leakage check -- texture_decorrelated images (should NOT improve):")
            for obs in decorr_imgs:
                if obs in common:
                    print(f"      {obs}: {cell_a.loc[obs, 'auc']:.3f} -> "
                          f"{g.loc[obs, 'auc']:.3f} ({d.loc[obs]:+.3f})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
