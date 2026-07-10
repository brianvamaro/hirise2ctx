"""H3 (PLAN_StripingArtifact PHASE 2) — assemble the skill-vs-η² Pareto.

Joins the η² sweep (reports/figures/f_h2_eta2_summary.csv, produced by f_h2_eta2.py
over the H3 heads) with the per-λ LOIO skill summaries
(reports/figures/f_leg_b_loio_summary_minnaert_center_h3_lam{λ}.csv) into one table
+ a scatter (skill Δ vs partition η²), and flags the reopening knee: the largest
skill Δ that still reaches partition η² ≤ 0.05 with Δ ≥ −0.02.

Run (after the η² sweep + all per-λ LOIO gates are done):
  conda run --no-capture-output -n geospatial python -u scripts/f_h3_pareto.py \
      --lambdas 0 3 10 30 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/pandas

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIG = REPO / "reports" / "figures"

ETA_BAR = 0.05      # reopening bar on partition η²
SKILL_GATE = -0.02  # min acceptable Δ median per-image AUC vs mosaic baseline


def eta_label(lam: int) -> str:
    return "center" if lam == 0 else f"h3_lam{lam}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", type=int, default=[0, 3, 10, 30, 100])
    args = ap.parse_args()

    # H3 η² sweep is preserved under an _h3 suffix (f_h2_eta2.py writes the generic
    # name, which H2 also uses; the parallel H2/H4 work keeps the generic for H2).
    eta = pd.read_csv(FIG / "f_h2_eta2_summary_h3.csv").set_index("label")
    rows = []
    for lam in args.lambdas:
        r = {"lambda": lam}
        el = eta_label(lam)
        if el in eta.index:
            r["eta2_partition"] = float(eta.loc[el, "partition"])
            r["eta2_median"] = float(eta.loc[el, "median"])
            r["pred_overlap"] = float(eta.loc[el, "pred_overlap"])
        tag = "_minnaert_center" if lam == 0 else f"_minnaert_center_h3_lam{lam}"
        sp = FIG / f"f_leg_b_loio_summary{tag}.csv"
        if sp.exists():
            s = pd.read_csv(sp)
            base = s[s["store"] == "fang_embeddings"]["median_auc"].iloc[0]
            fcol = s[s["store"] != "fang_embeddings"]
            fmed = fcol["median_auc"].iloc[0]
            r["baseline_med_auc"] = round(float(base), 4)
            r["F_med_auc"] = round(float(fmed), 4)
            r["skill_delta"] = round(float(fmed - base), 4)
            r["pooled_pr_auc"] = round(float(fcol["pooled_pr_auc"].iloc[0]), 4)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(FIG / "f_h3_pareto.csv", index=False)
    print("=== H3 skill-vs-η² Pareto (baselines: mosaic raw η² 0.196 / A1 0.141 / "
          "H1 λ=0 0.128 part) ===")
    print(df.to_string(index=False))

    if {"skill_delta", "eta2_partition"} <= set(df.columns):
        ok = df.dropna(subset=["skill_delta", "eta2_partition"])
        reopen = ok[(ok["eta2_partition"] <= ETA_BAR) & (ok["skill_delta"] >= SKILL_GATE)]
        print(f"\nreopening bar: partition η² ≤ {ETA_BAR} AND skill Δ ≥ {SKILL_GATE}")
        if len(reopen):
            knee = reopen.sort_values("skill_delta", ascending=False).iloc[0]
            print(f"  ✅ REOPENS at λ={int(knee['lambda'])}: partition η²="
                  f"{knee['eta2_partition']:.3f}, skill Δ={knee['skill_delta']:+.4f}")
        else:
            print("  ❌ no λ reaches η² ≤ 0.05 at skill ≥ −0.02 (no Pareto point clears both)")

        fig, ax = plt.subplots(figsize=(6.2, 5))
        ax.axhline(SKILL_GATE, color="crimson", ls="--", lw=1, label=f"skill gate {SKILL_GATE}")
        ax.axvline(ETA_BAR, color="seagreen", ls="--", lw=1, label=f"η² bar {ETA_BAR}")
        ax.plot(ok["eta2_partition"], ok["skill_delta"], "-o", color="k", zorder=3)
        for _, row in ok.iterrows():
            ax.annotate(f"λ={int(row['lambda'])}",
                        (row["eta2_partition"], row["skill_delta"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel("frame-block η² (partition composite)  ← better")
        ax.set_ylabel("skill Δ median per-image AUC vs mosaic  ↑ better")
        ax.set_title("H3 consistency-regularized head — skill vs artifact Pareto")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        out = FIG / "f_h3_pareto.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\nwrote {out} + f_h3_pareto.csv")


if __name__ == "__main__":
    main()
