"""Combine the Stage 6a follow-up sweep results into a single comparison MD.

Reads `models/_sweep_stage6a/20260531T004356Z/aggregate.parquet` (the 4-scheme x
2-scale follow-up sweep) and emits a comparison table with deltas vs the
within_image_4fold baseline, plus an honest verdict per scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

OUT_MD = Path(__file__).with_suffix(".md")
AGG = REPO_ROOT / "models/_sweep_stage6a/20260531T004356Z/aggregate.parquet"

# Metrics that matter (presence_auc excluded per Brian 2026-05-30).
METRICS = [
    ("Spearman rho", "spearman_rho_mean", "+.4f"),
    ("PR-AUC", "pr_auc_mean", ".4f"),
    ("normalised lift @top-K", "normalised_lift_meaningful_mean", ".4f"),
    ("precision @top-5%", "precision_at_top_5pct_mean", ".4f"),
    ("recall @top-5%", "recall_at_top_5pct_mean", ".4f"),
]
SCHEME_BASELINE = "within_image_4fold"
SCHEME_LABELS = {
    "within_image_4fold": "P1+P2 baseline",
    "within_image_4fold_nbr": "+6a default (3x3, mean+max+std)",
    "within_image_4fold_nbr_s5": "+6a 5x5 stencil (mean+max+std)",
    "within_image_4fold_nbr_max": "+6a max-only (3x3, max)",
}


def main() -> int:
    df = pd.read_parquet(AGG)
    lines: list[str] = []
    lines.append("# Stage 6a follow-up sweep -- combined comparison")
    lines.append("")
    lines.append(f"Source: `{AGG.relative_to(REPO_ROOT)}`")
    lines.append("")
    lines.append(
        "Variant: `lightgbm_two_stage_balanced` (P1) + `target_col=boulder_count` (P2).  "
        "Dev = within-image 4-fold on 5 dataset_v2_dev images (20 folds).  "
        "Acceptance criteria (PROMOTION_QUEUE.md Stage 6a): "
        "Spearman delta >= +0.05 AND PR-AUC delta >= +0.03 vs the P1+P2 baseline."
    )
    lines.append("")

    for scale_idx in sorted(df["scale_idx"].unique()):
        tile_size = int(df[df["scale_idx"] == scale_idx]["tile_size_px"].iloc[0])
        scale_df = df[df["scale_idx"] == scale_idx].set_index("scheme")
        lines.append(f"## S = {tile_size} (scale_idx={scale_idx})")
        lines.append("")
        header_cells = ["metric"] + [
            f"{SCHEME_LABELS.get(s, s)}<br>(delta)" for s in SCHEME_LABELS
        ]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["--------"] + [":---:" for _ in SCHEME_LABELS]) + "|")
        base_row = scale_df.loc[SCHEME_BASELINE]
        for label, key, fmt in METRICS:
            row_cells = [label]
            for scheme in SCHEME_LABELS:
                if scheme not in scale_df.index:
                    row_cells.append("-")
                    continue
                val = float(scale_df.loc[scheme, key])
                if scheme == SCHEME_BASELINE:
                    row_cells.append(f"{val:{fmt}}")
                else:
                    delta = val - float(base_row[key])
                    row_cells.append(f"{val:{fmt}}<br>({delta:+.4f})")
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

        verdict_lines = []
        for scheme in SCHEME_LABELS:
            if scheme == SCHEME_BASELINE or scheme not in scale_df.index:
                continue
            rho_delta = (
                float(scale_df.loc[scheme, "spearman_rho_mean"])
                - float(base_row["spearman_rho_mean"])
            )
            pr_delta = (
                float(scale_df.loc[scheme, "pr_auc_mean"])
                - float(base_row["pr_auc_mean"])
            )
            rho_pass = rho_delta >= 0.05
            pr_pass = pr_delta >= 0.03
            verdict = "PASS" if (rho_pass and pr_pass) else "FAIL"
            verdict_lines.append(
                f"- **{SCHEME_LABELS[scheme]}**: "
                f"Spearman {rho_delta:+.4f} ({'>=' if rho_pass else '<'} +0.05), "
                f"PR-AUC {pr_delta:+.4f} ({'>=' if pr_pass else '<'} +0.03) "
                f"--> **{verdict}**"
            )
        lines.append("Acceptance verdict at this scale:")
        lines.append("")
        lines.extend(verdict_lines)
        lines.append("")

    # Best-of-all summary
    lines.append("## Best absolute numbers across the whole grid")
    lines.append("")
    lines.append("| metric | scheme | scale_idx | S | value |")
    lines.append("|--------|--------|----------:|--:|------:|")
    for label, key, fmt in METRICS:
        best_idx = df[key].idxmax()
        row = df.loc[best_idx]
        lines.append(
            f"| {label} | {row['scheme']} | {int(row['scale_idx'])} | "
            f"{int(row['tile_size_px'])} | {float(row[key]):{fmt}} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
