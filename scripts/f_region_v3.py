"""PLAN_FBuild V3 — Stage-B validation on a few frames before the 907 array.

The physical per-row incidence is already validated (V2: reproduces the index center incidence to
~0.1 deg). V3 confirms the rest of the Stage-B path on real output:
  1. CO-LOCATION — overlapping frames land on the SAME global (TI,TJ) tiles (n_overlap > 0). This is
     what the Stage-C H4 solve needs; 0 co-located tiles would mean the cubes aren't grid-aligned and
     the global-tiling is broken (a hard bug to catch BEFORE the full run).
  2. PRE-H4 AGREEMENT — median |Δp| between frames on co-located tiles is sane (~pilot 0.07; H4 then
     reduces it). A huge value (>~0.3) would flag a mapping/head problem.
  3. SANITY — per-frame tile counts + P(rich) distributions look reasonable.

Run (after Stage B on >=2 KNOWN-OVERLAPPING frames — e.g. two E8_N44 pilot frames):
  python scripts/f_region_stageb.py --frames PID1 PID2 [PID3] \
      --cubes-dir $SCRATCH/hirise2ctx/f_region --out-dir $SCRATCH/hirise2ctx/f_region_logits
  python scripts/f_region_v3.py --out-dir $SCRATCH/hirise2ctx/f_region_logits --frames PID1 PID2 [PID3]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Stage-B output dir ({PID}.npz)")
    ap.add_argument("--frames", nargs="+", required=True, help=">=2 overlapping PRODUCT_IDs")
    args = ap.parse_args()
    out = Path(args.out_dir)

    loc: dict[int, dict[str, float]] = defaultdict(dict)
    print("=== per-frame ===")
    for pid in args.frames:
        p = out / f"{pid}.npz"
        if not p.exists():
            print(f"  ⚠ {pid}: no {p.name} (run Stage B on it first)")
            continue
        d = np.load(p)
        TI, TJ, prob = d["TI"], d["TJ"], d["prob"]
        print(f"  {pid}: {TI.size:,} tiles  P(rich) mean {prob.mean():.3f} "
              f"[{prob.min():.3f},{prob.max():.3f}]  rich@0.5 {float((prob>=0.5).mean()):.3f}")
        key = TI.astype(np.int64) * 10_000_000 + TJ.astype(np.int64)
        for k, pr in zip(key.tolist(), prob.tolist()):
            loc[k][pid] = pr

    shared = {k: v for k, v in loc.items() if len(v) >= 2}
    print(f"\n=== co-location ===\n  total distinct tiles {len(loc):,}  "
          f"co-located (>=2 frames) {len(shared):,}")
    if not shared:
        print("  ✗ FAIL: no co-located tiles — global (TI,TJ) grid is NOT aligning frames "
              "(cubes not on a shared lattice?). Fix before the 907 run.")
        return 1

    diffs = []
    for v in shared.values():
        vals = list(v.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diffs.append(abs(vals[i] - vals[j]))
    diffs = np.array(diffs)
    print(f"\n=== pre-H4 overlap agreement (H4 will reduce this) ===")
    print(f"  median |Δp| {np.median(diffs):.4f}  mean {diffs.mean():.4f}  "
          f"p90 {np.percentile(diffs,90):.4f}  (pilot pre-H4 ~0.07)")
    verdict = ("PASS — frames co-locate and agree at pilot-level; Stage C can solve H4"
               if np.median(diffs) < 0.30 else
               "⚠ HIGH pre-H4 |Δp| — inspect the mapping/head before the full run")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
