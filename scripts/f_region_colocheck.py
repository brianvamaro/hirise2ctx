"""Fast co-location check (GEOMETRY only — no embed, no GPU) for PLAN_FBuild Stage C.

The Stage-C H4 solve needs overlapping frames to land on the SAME global (TI,TJ) S=32 tiles. That is
purely a function of the cube transforms + valid masks (do the cam2map cubes share a pixel lattice
so the global-grid rounding co-locates them?) — the embedder is irrelevant. This replicates Stage B's
exact keying [TI=round(wy/160), TJ=round(wx/160) from each tile's CTX-CRS world center] on a decimated
valid mask, and reports pairwise shared-tile counts. Runs in seconds on a login node.

Run (map venv, login node):
  python scripts/f_region_colocheck.py --cubes-dir $SCRATCH/hirise2ctx/f_region --frames PID1 PID2 [PID3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import rasterio

R_MARS_M = 3_396_190.0
TILE_PX = 32
GLOBAL_M = TILE_PX * 5.0


def frame_tiles(path: Path) -> set:
    """Global (TI,TJ) set of the frame's valid S=32 tiles (same keying as f_region_stageb)."""
    with rasterio.open(path) as ds:
        tr = ds.transform
        H, W = ds.height, ds.width
        a = ds.read(1, out_shape=(max(H // TILE_PX, 1), max(W // TILE_PX, 1)))  # ~1 sample per tile
    valid = np.isfinite(a) & (a > 0) & (a > -1e30)
    dr, dc = np.where(valid)                       # decimated row/col == tile index ti/tj
    wy = tr.f + (dr + 0.5) * TILE_PX * tr.e
    wx = tr.c + (dc + 0.5) * TILE_PX * tr.a
    TI = np.round(wy / GLOBAL_M).astype(np.int64)
    TJ = np.round(wx / GLOBAL_M).astype(np.int64)
    return set(zip(TI.tolist(), TJ.tolist()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes-dir", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    args = ap.parse_args()
    cdir = Path(args.cubes_dir)

    sets = {}
    for pid in args.frames:
        path = next((cdir / f"{pid}{e}" for e in (".map.tif", ".tif", ".map.cub")
                     if (cdir / f"{pid}{e}").exists()), None)
        if path is None:
            print(f"  ⚠ {pid}: no raster in {cdir}")
            continue
        sets[pid] = frame_tiles(path)
        print(f"  {pid}: {len(sets[pid]):,} global tiles")

    pids = list(sets)
    max_shared = 0
    print("\n=== pairwise co-located tiles ===")
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            sh = len(sets[pids[i]] & sets[pids[j]])
            max_shared = max(max_shared, sh)
            print(f"  {pids[i]} ∩ {pids[j]} = {sh:,} shared")

    ok = max_shared > 0
    print(f"\nVERDICT: {'PASS — frames co-locate on the global grid; Stage C can build the overlap graph' if ok else 'FAIL — 0 shared tiles: cubes not on a shared lattice; fix keying before the 907 run'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
