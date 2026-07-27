"""PLAN_FBuild Stage B — per-frame embed + infer on the ISIS cubes.

For each of the 907 region frames (a Stage-A `{PRODUCT_ID}.map.cub`), produce raw P(boulder-rich)
per S=32 tile, keyed to a GLOBAL tile grid so overlapping frames co-locate for the Stage-C H4 solve.
Array-task-aware (round-robin stride of region_frame_list.csv), resumable (skips frames whose output
exists). Runs on a Sherlock GPU node in the map venv (same env as map_region.py / run_region_array).

Per frame, per read-window (the map_region window sweep):
  I/F  ÷ cos^k(i(lat))   [per-ROW incidence from PHYSICAL subsolar geometry (V2); no slope fit]
       ÷ per-frame median [H1 centering; median of the cos^k-corrected valid pixels]
       -> FIXED centered-pool log stretch (0.8400..1.1170) -> uint8
  -> FangEmbedder.embed_window (GeM, S=32) -> DeployableHead(deployable_f_center).predict -> P(rich)
  -> key each usable tile to global (TI,TJ) from its CTX-CRS world center (round to the 160 m grid)

Output per frame: {PRODUCT_ID}.npz  (TI, TJ, prob int32/int32/float32) + {PRODUCT_ID}.json (meta).
Logits are NOT stored (H4 logit-transforms at compose time, as the pilot did); calibration is a final
step after H4. Per-row incidence is PHYSICAL — cos(i(φ)) = sinφ·sinφ_s + cosφ·cosφ_s·cos(Δλ) from
V2's region_frame_incidence.csv (center_lon, subsolar_lat, subsolar_lon); no slope fit (V2 confirmed
it reproduces the index center incidence to ~0.1 deg).

Usage (Sherlock gpu node, map venv; via run_f_region_stageb.sbatch):
  TASK_ID=0 N_TASKS=1 python scripts/f_region_stageb.py \
      --cubes-dir $SCRATCH/hirise2ctx/f_region --out-dir $SCRATCH/hirise2ctx/f_region_logits
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP/DLL bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from scripts.map_region import window_offsets
from src.fm_embeddings import FangEmbedder, tile_grid_for_window
from src.mapping import own_tile_zero_fraction
from src.modeling.mlp_head import DeployableHead

K_MINNAERT = 0.580
STRETCH_LO, STRETCH_HI = 0.8400, 1.1170           # centered-pool fixed stretch (H1 training constants)
R_MARS_M = 3_396_190.0                            # CTX equirect sphere (CLAUDE.md)
TILE_PX = 32
GLOBAL_M = TILE_PX * 5.0                           # 160 m global tile grid (all cubes are 5 m/px)
DEFAULT_HEAD = REPO / "models" / "deployable_f_center" / "86c51a5dca220f63"
FRAME_LIST = REPO / "reports" / "figures" / "region_frame_list.csv"
INC_CSV = REPO / "reports" / "figures" / "region_frame_incidence.csv"


def to_uint8_log(arr: np.ndarray) -> np.ndarray:
    """ln(x) in [ln lo, ln hi] -> [1,255]; nodata/<=0 -> 0 (matches f_pilot_crop.to_uint8_log)."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    fin = np.isfinite(arr) & (arr > 0)
    v = np.log(arr[fin])
    llo, lhi = np.log(STRETCH_LO), np.log(STRETCH_HI)
    out[fin] = np.clip((v - llo) / (lhi - llo) * 254.0 + 1.0, 1, 255).astype(np.uint8)
    return out


def lat_deg_of_rows(transform, row_idx: np.ndarray) -> np.ndarray:
    """CTX-equirect world y (m) of pixel rows -> planetocentric latitude (deg)."""
    y = transform.f + (row_idx + 0.5) * transform.e          # e < 0
    return y / (R_MARS_M * np.pi / 180.0)


def cosk_incidence_rows(row_idx, transform, subsolar_lat, dlam_deg) -> np.ndarray:
    """cos^k(incidence(lat)) per row from PHYSICAL subsolar geometry (no slope fit; V2 confirmed it
    reproduces the PDS index center incidence to ~0.1 deg):
        cos(i(φ)) = sinφ·sinφ_s + cosφ·cosφ_s·cos(Δλ),  Δλ = center_lon − subsolar_lon (per-frame const).
    The per-row gradient is exact; H1 median-centering removes the absolute cos^k level, so only the
    within-frame ramp is corrected (real geology/albedo down the frame is untouched)."""
    lat = lat_deg_of_rows(transform, np.asarray(row_idx, float))
    phi = np.radians(lat)
    phis = np.radians(subsolar_lat)
    cosi = np.sin(phi) * np.sin(phis) + np.cos(phi) * np.cos(phis) * np.cos(np.radians(dlam_deg))
    cosi = np.clip(cosi, np.cos(np.radians(89.5)), 1.0)
    return cosi ** K_MINNAERT


def frame_median(ds, transform, subsolar_lat, dlam_deg, stride: int = 16) -> float:
    """H1 centering statistic: median of the cos^k(i(lat))-corrected I/F over valid pixels (decimated)."""
    a = ds.read(1, out_shape=(ds.height // stride, ds.width // stride)).astype(np.float32)
    rows = (np.arange(a.shape[0]) * stride).astype(float)
    cosk = cosk_incidence_rows(rows, transform, subsolar_lat, dlam_deg)
    d = a / cosk[:, None]
    fin = np.isfinite(d) & (a > 0) & (a > -1e30)
    return float(np.median(d[fin])) if fin.any() else float("nan")


def process_frame(ds, transform, subsolar_lat, dlam_deg, med, embedder, head, args):
    """Sweep windows -> global (TI, TJ, prob) for one frame."""
    H, W = ds.height, ds.width
    win, overlap = args.win_px, 2 * TILE_PX
    grid = [(r, c) for r in window_offsets(H, win, overlap, TILE_PX)
            for c in window_offsets(W, win, overlap, TILE_PX)]
    TI, TJ, PROB = [], [], []
    for (row_off, col_off) in grid:
        h = min(win, H - row_off)
        w = min(win, W - col_off)
        if h < 3 * TILE_PX or w < 3 * TILE_PX:
            continue
        iff = ds.read(1, window=Window(col_off, row_off, w, h)).astype(np.float32)
        rows = row_off + np.arange(h, dtype=float)
        cosk = cosk_incidence_rows(rows, transform, subsolar_lat, dlam_deg)
        u8 = to_uint8_log((iff / cosk[:, None]) / med)
        ti, tj = tile_grid_for_window(u8.shape, row_off, col_off, TILE_PX)
        if ti.size == 0:
            continue
        emb, valid = embedder.embed_window(u8, ti, tj, tile_px=TILE_PX, row0=row_off,
                                           col0=col_off, pool="gem", batch=args.batch)
        zf = own_tile_zero_fraction(u8, ti, tj, tile_px=TILE_PX, row0=row_off, col0=col_off)
        usable = valid & (zf <= args.max_zero_fraction)
        if not usable.any():
            continue
        prob = head.predict(emb[usable]).astype(np.float32)
        cy = transform.f + (ti[usable] + 0.5) * TILE_PX * transform.e
        cx = transform.c + (tj[usable] + 0.5) * TILE_PX * transform.a
        TI.append(np.round(cy / GLOBAL_M).astype(np.int64))
        TJ.append(np.round(cx / GLOBAL_M).astype(np.int64))
        PROB.append(prob)
    if not PROB:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    ti = np.concatenate(TI); tj = np.concatenate(TJ); prob = np.concatenate(PROB)
    # dedup co-located tiles from overlapping windows (identical prob; keep mean for safety)
    key = ti.astype(np.int64) * 10_000_000 + tj.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, ti, tj, prob = key[order], ti[order], tj[order], prob[order]
    uniq, first = np.unique(key, return_index=True)
    sums = np.add.reduceat(prob, first)
    counts = np.diff(np.append(first, len(prob)))
    return ti[first], tj[first], (sums / counts).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--win-px", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-zero-fraction", type=float, default=0.3)
    ap.add_argument("--head", default=str(DEFAULT_HEAD))
    ap.add_argument("--frames", nargs="*", default=None, help="explicit PRODUCT_IDs (else the full list)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    task_id = int(__import__("os").environ.get("TASK_ID", 0))
    n_tasks = int(__import__("os").environ.get("N_TASKS", 1))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cubes = Path(args.cubes_dir)

    fl = pd.read_csv(FRAME_LIST)
    inc = pd.read_csv(INC_CSV).set_index("PRODUCT_ID")
    pids = args.frames if args.frames else list(fl["PRODUCT_ID"])
    pids = [p for k, p in enumerate(pids) if k % n_tasks == task_id]

    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)
    head = DeployableHead.load(Path(args.head))
    print(f"task {task_id}/{n_tasks}: {len(pids)} frames  "
          f"head={Path(args.head).name}  (physical per-row incidence)", flush=True)

    for pid in pids:
        out_npz = out_dir / f"{pid}.npz"
        if out_npz.exists():
            continue
        cube = next((cubes / f"{pid}{e}" for e in (".map.cub", ".map.tif", ".tif")
                     if (cubes / f"{pid}{e}").exists()), None)
        if cube is None or pid not in inc.index:
            print(f"  ⚠ {pid}: cube or incidence missing -> skip", flush=True)
            continue
        t0 = time.monotonic()
        with rasterio.open(cube) as ds:
            tr = ds.transform
            slat = float(inc.loc[pid, "subsolar_lat"])
            dlam = float(inc.loc[pid, "center_lon"]) - float(inc.loc[pid, "subsolar_lon"])
            ci = float(inc.loc[pid, "incidence"])           # index center incidence (QA/logging only)
            med = frame_median(ds, tr, slat, dlam)
            if not np.isfinite(med):
                print(f"  ⚠ {pid}: no valid pixels -> skip", flush=True)
                continue
            TI, TJ, prob = process_frame(ds, tr, slat, dlam, med, embedder, head, args)
            crs_wkt = ds.crs.to_wkt() if ds.crs else ""
        np.savez_compressed(out_npz, TI=TI, TJ=TJ, prob=prob)
        (out_dir / f"{pid}.json").write_text(json.dumps({
            "PRODUCT_ID": pid, "index_incidence": ci, "subsolar_lat": slat,
            "dlam_deg": round(dlam, 4), "frame_median": round(med, 6), "n_tiles": int(TI.size),
            "prob_mean": float(prob.mean()) if TI.size else None,
            "global_tile_m": GLOBAL_M, "crs_wkt": crs_wkt,
        }, indent=2), encoding="utf-8")
        print(f"  {pid}: {TI.size:,} tiles  med={med:.4f}  {time.monotonic()-t0:.0f}s", flush=True)
    print(f"task {task_id} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
