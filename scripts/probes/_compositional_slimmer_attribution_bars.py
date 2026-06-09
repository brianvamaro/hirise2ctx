"""Per-image attribution bar chart for the slimmer compositional writeup.

Single panel (by area, fa >= 1%) version of the original stage7d
two-panel figure (by area + by count). The slimmer doc reports only
the area-based partition, so the by-count panel is dropped here.

Output: reports/figures/compositional_slimmer_attribution_bars.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_v2"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [("baseline", None), ("0.20", 0.20), ("0.10", 0.10), ("0.05", 0.05)]
ORDERED_CATS = ["composition_residual", "dust_attributable", "no_signal", "inconclusive"]
COLORS = {
    "composition_residual": "#2a9d8f",
    "dust_attributable": "#e9c46a",
    "no_signal": "#bdbdbd",
    "inconclusive": "#e76f51",
}
LEGEND_LABELS = {
    "composition_residual": "Composition residual",
    "dust_attributable": "Dust attributable",
    "no_signal": "No signal",
    "inconclusive": "Inconclusive",
}


def _attribution_path(label: str) -> Path:
    if label == "baseline":
        return DATASET / "stage7d_per_image_attribution.parquet"
    return DATASET / f"stage7d_attribution_shadow_{label}.parquet"


def main() -> int:
    attribution = {label: pd.read_parquet(_attribution_path(label))
                   for label, _ in THRESHOLDS}
    threshold_labels = [label for label, _ in THRESHOLDS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = np.zeros(len(threshold_labels))
    for cat in ORDERED_CATS:
        heights = []
        for label in threshold_labels:
            sub = attribution[label]
            sub = sub[sub["partition_rule"] == "P4_area"]
            heights.append(int((sub["attribution"] == cat).sum()))
        print(f"  {cat:>22s}: {heights}")
        # Skip categories with zero count across every threshold; keeps
        # the legend tidy (e.g. 'inconclusive' is currently always 0 for
        # the by-area partition under our standard test settings).
        if sum(heights) == 0:
            continue
        ax.bar(threshold_labels, heights, bottom=bottoms,
               color=COLORS[cat], edgecolor="black", linewidth=0.5,
               label=LEGEND_LABELS[cat])
        bottoms += np.array(heights)

    ax.set_title("Per-image attribution counts")
    ax.set_xlabel("Shadow fraction threshold")
    ax.set_ylabel("Number of eligible images")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = FIG / "compositional_slimmer_attribution_bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
