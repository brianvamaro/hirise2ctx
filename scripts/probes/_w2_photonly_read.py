"""W2 cell E (photometric_only) de-confound read.

Cell C (geometric+photometric) lost to cell A (none) in the Phase 1 grid.
Cell E removes the geometric half. Question: was geometric augmentation the
harmful ingredient (E ~ A or better), or is ANY augmentation harmful at this
scale (E ~ C)? Plus the mechanism check: does photometric-only specifically
help the distribution_shift images (the H-B motivation), as cell C did for
ESP_076499_1160?
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import pandas as pd
from scipy import stats

GRID = REPO_ROOT / "models/_sweep_cnn/20260611T220815Z/summary.parquet"   # cells A-D seed 0
CELL_E = REPO_ROOT / "models/_sweep_cnn/20260612T045007Z/summary.parquet"  # photometric_only seed 0
DOSSIER = REPO_ROOT / "dataset_v2/w1_dossier.parquet"


def paired(d: pd.Series) -> str:
    d = d.dropna()
    try:
        p = stats.wilcoxon(d, zero_method="wilcox").pvalue
    except ValueError:
        p = float("nan")
    return (f"mean={d.mean():+.4f} median={d.median():+.4f} "
            f"win={(d > 0).mean():.2f} p={p:.4f}")


def main() -> int:
    grid = pd.read_parquet(GRID)
    e = pd.read_parquet(CELL_E).set_index("held_out_obs_id")
    dossier = pd.read_parquet(DOSSIER)
    shift = sorted(dossier[dossier.attributed_cause == "distribution_shift"].index)

    for cell in ("none", "photometric"):
        g = grid[grid.aug_cell == cell].set_index("held_out_obs_id")
        common = e.index.intersection(g.index)
        print(f"E (photometric_only) vs {cell}: "
              f"{paired(e.loc[common, 'auc'] - g.loc[common, 'auc'])}")
        print(f"  median AUC {g.loc[common, 'auc'].median():.4f} -> "
              f"{e.loc[common, 'auc'].median():.4f}")
        print("  distribution_shift images:")
        for obs in shift:
            if obs in common:
                print(f"    {obs}: {g.loc[obs, 'auc']:.3f} -> {e.loc[obs, 'auc']:.3f} "
                      f"({e.loc[obs, 'auc'] - g.loc[obs, 'auc']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
