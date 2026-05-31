"""Per-fold variance + win/loss probe for the Stage 6a dev sweep.

Reads the latest `_sweep_stage6a/{timestamp}/summary.parquet` and reports per-fold
mean / std / win-loss counts so we can judge whether the small negative deltas
(Spearman, presence AUC) are real or fold-level noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

OUT_MD = Path(__file__).with_suffix(".md")
SUMMARY = REPO_ROOT / "models/_sweep_stage6a/20260530T213424Z/summary.parquet"

KEY_COLS = [
    "spearman_rho", "presence_auc", "pr_auc",
    "normalised_lift_meaningful", "precision_at_top_5pct", "recall_at_top_5pct",
]


def main() -> int:
    df = pd.read_parquet(SUMMARY)
    lines: list[str] = []
    lines.append(f"# Stage 6a per-fold variance probe")
    lines.append("")
    lines.append(f"Source: `{SUMMARY.relative_to(REPO_ROOT)}`")
    lines.append(f"n_folds = {df['fold_idx'].nunique()}; schemes = {sorted(df['scheme'].unique())}")
    lines.append("")
    lines.append("## Per-scheme aggregates")
    lines.append("")
    agg = df.groupby("scheme")[KEY_COLS].agg(["mean", "std", "min", "max"]).round(4)
    lines.append("```")
    lines.append(agg.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Per-fold deltas (nbr - base)")
    lines.append("")
    lines.append(
        "| metric | mean delta | std delta | wins (nbr>base) | losses | ties | "
    )
    lines.append(
        "|--------|-----------:|----------:|----------------:|-------:|-----:|"
    )
    df_wide = df.pivot_table(index="fold_idx", columns="scheme", values=KEY_COLS)
    for col in KEY_COLS:
        deltas = (
            df_wide[(col, "within_image_4fold_nbr")]
            - df_wide[(col, "within_image_4fold")]
        )
        n_pos = int((deltas > 0).sum())
        n_neg = int((deltas < 0).sum())
        n_zero = int((deltas == 0).sum())
        lines.append(
            f"| {col:<35s} | {deltas.mean():+.4f} | {deltas.std():.4f} | "
            f"{n_pos:>3d} | {n_neg:>3d} | {n_zero:>3d} |"
        )
    lines.append("")
    lines.append("## Per-held-out-image deltas (Spearman + precision@top-5%)")
    lines.append("")
    lines.append(
        "| obs_id | rho_base | rho_nbr | rho_delta | prec5_base | prec5_nbr | prec5_delta |"
    )
    lines.append(
        "|--------|---------:|--------:|----------:|-----------:|----------:|------------:|"
    )
    by_obs = df.groupby(["held_out_obs_id", "scheme"])[
        ["spearman_rho", "precision_at_top_5pct"]
    ].mean().reset_index()
    pivot_rho = by_obs.pivot(index="held_out_obs_id", columns="scheme",
                             values="spearman_rho")
    pivot_p5 = by_obs.pivot(index="held_out_obs_id", columns="scheme",
                            values="precision_at_top_5pct")
    for obs in sorted(pivot_rho.index):
        r_b = pivot_rho.loc[obs, "within_image_4fold"]
        r_n = pivot_rho.loc[obs, "within_image_4fold_nbr"]
        p_b = pivot_p5.loc[obs, "within_image_4fold"]
        p_n = pivot_p5.loc[obs, "within_image_4fold_nbr"]
        lines.append(
            f"| {obs} | {r_b:+.4f} | {r_n:+.4f} | {r_n - r_b:+.4f} | "
            f"{p_b:.4f} | {p_n:.4f} | {p_n - p_b:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
