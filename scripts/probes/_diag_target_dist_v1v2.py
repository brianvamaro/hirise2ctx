"""v1-vs-v2 target-distribution shift (PLAN_NewDetections.md §9.1).

For each tile scale, report how zero-inflated `fractional_area` is in the v1
(priority10) vs v2 (vClaire) packaged label set: zero fraction, presence
fractions at the Stage-5b thresholds, and the positive-tail quantiles. The
denser vClaire detections should sharply cut the zero-tile fraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64}
DATASETS = {
    "v1": REPO_ROOT / "dataset" / "packaged" / "loio_9fold" / "all.parquet",
    "v2": REPO_ROOT / "dataset_v2" / "packaged" / "loio_nfold" / "all.parquet",
}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale_idx, tile in SCALE_TILE_PX.items():
        sub = df[df["scale_idx"] == scale_idx]
        fa = sub["fractional_area"].to_numpy()
        n = fa.size
        pos = fa[fa > 0]
        rows.append({
            "scale_idx": scale_idx,
            "tile_size_px": tile,
            "n_tiles": n,
            "zero_frac": float((fa == 0).mean()) if n else float("nan"),
            "frac_gt_0": float((fa > 0).mean()) if n else float("nan"),
            "frac_ge_bc1": float((sub["boulder_count"].to_numpy() >= 1).mean()) if n else float("nan"),
            "frac_gt_1e-3": float((fa > 1e-3).mean()) if n else float("nan"),
            "frac_gt_1e-2": float((fa > 1e-2).mean()) if n else float("nan"),
            "pos_median": float(np.median(pos)) if pos.size else float("nan"),
            "pos_p95": float(np.percentile(pos, 95)) if pos.size else float("nan"),
            "max": float(fa.max()) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    for ver, path in DATASETS.items():
        if not path.exists():
            print(f"[{ver}] missing {path}")
            continue
        df = pd.read_parquet(path, columns=["scale_idx", "fractional_area", "boulder_count"])
        print(f"\n=== {ver}: {path.relative_to(REPO_ROOT)}  ({len(df):,} rows) ===")
        s = summarize(df)
        with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
            print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
