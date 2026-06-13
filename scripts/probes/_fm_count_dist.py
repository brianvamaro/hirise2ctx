"""PLAN_FM 1b: per-tile boulder_count distribution to pick a real count target.

bc_ge_1 is presence (saturated at S=64). Brian wants a count threshold that
actually splits rich/poor (e.g. >=50). This reports the boulder_count quantiles
and the positive rate at candidate thresholds, at both operating scales, so the
threshold is grounded in the data rather than guessed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

LABELS = sorted((REPO_ROOT / "dataset_v2" / "labels").glob("ESP_*.parquet"))


def main() -> int:
    df0 = pd.read_parquet(LABELS[0])
    print("label cols:", list(df0.columns), flush=True)
    scale_col = "tile_size_px" if "tile_size_px" in df0.columns else None
    print("scale col:", scale_col,
          ("scales: " + str(sorted(df0[scale_col].unique()))) if scale_col else "", flush=True)

    frames = [pd.read_parquet(f, columns=[c for c in df0.columns
              if c in ("boulder_count", "fractional_area", "tile_size_px")]) for f in LABELS]
    df = pd.concat(frames, ignore_index=True)

    for scale in (sorted(df["tile_size_px"].unique()) if scale_col else [None]):
        sub = df if scale is None else df[df["tile_size_px"] == scale]
        bc = sub["boulder_count"].to_numpy()
        fa = sub["fractional_area"].to_numpy()
        print(f"\n=== scale px={scale}  n_tiles={len(sub)} ===", flush=True)
        qs = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
        print("boulder_count quantiles:",
              {q: float(np.quantile(bc, q)) for q in qs}, flush=True)
        print(f"  mean={bc.mean():.2f}  max={bc.max():.0f}", flush=True)
        print("pos_rate by count threshold:", flush=True)
        for t in (1, 5, 10, 20, 50, 100, 200):
            print(f"  bc>={t:>3}: {(bc >= t).mean():.4f}", flush=True)
        print("pos_rate by area threshold (ref):", flush=True)
        for t in (1e-3, 1e-2):
            print(f"  fa>{t:g}: {(fa > t).mean():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
