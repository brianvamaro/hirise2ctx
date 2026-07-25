"""PLAN_FBuild Stage-B sizing probe — V1 (embed cost) + V5 (within-frame incidence ramp).

Runs AFTER Stage A on the KEEP_CUBES cubes (converted to GeoTIFF, `--frames-dir`). See SHERLOCK_RUN
Part G.

Corrected 2026-07-24 after the first run exposed two methodology bugs on real cam2map output:
  * cam2map canvases are ~50% nodata (the frame is a swath in a lon/lat bbox) AND long frames span
    ~5 deg latitude. The first version (a) read one raster ROW (`read(1, out_shape=(1,h,w))[0]`
    silently sliced a row) and (b) embedded EVERY tile in sampled windows incl. nodata, so both V1
    timing and V5 were garbage.
  * V1 now counts VALID S=32 tiles from the 2D nodata mask and times the embedder on VALID tiles
    only (a geometry-free hardware rate); the 907 extrapolation scales per FRAME-TILE footprint to
    undo the sizing-frame selection bias (probe frames average more tiles than the population).
  * V5 bins median ln(I/F) by latitude band over valid pixels. The MEASURED ramp is dominated by
    real along-track albedo (300 km frames), so the DECISION quantity is the geometry-PREDICTED
    cos^k(i) illumination ramp k*tan(i)*di, di = (di/dlat)*dlat; measured is reported as context.

Run (GPU):
  conda run --no-capture-output -n geospatial python -u scripts/f_build_sizing_probe.py \
      --frames-dir reports/f_build/probe_cubes --timing-csv reports/f_build/probe_cubes/timing.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.fm_embeddings import FangEmbedder, tile_grid_for_window

K_MINNAERT = 0.580
DI_DLAT = 0.635               # deg incidence per deg latitude (audit 2026-07-23 cohort fit)
R_MARS_M = 3_396_190.0
TILE_PX = 32
FIG = REPO / "reports" / "figures"
RAMP_PER_ROW_THRESH = 1.0     # predicted illumination ramp (%) above which per-row cos^k(i(lat)) wins
DECIM = 8                     # decimation stride for the valid-mask / ramp reads


def embedder_rate(embedder, warmups: int = 3) -> float:
    """Geometry-free hardware rate: tiles/s embedding a fully-valid 2048^2 window (warm)."""
    rng = np.random.default_rng(0)
    u8 = rng.integers(60, 200, (2048, 2048)).astype("uint8")
    ti, tj = tile_grid_for_window(u8.shape, 0, 0, TILE_PX)
    rate = float("nan")
    for _ in range(warmups):
        t0 = time.perf_counter()
        embedder.embed_window(u8, ti, tj, tile_px=TILE_PX, row0=0, col0=0, pool="gem", batch=256)
        rate = ti.size / (time.perf_counter() - t0)
    return rate


def frame_stats(path: Path, incidence: float) -> dict:
    """Valid S=32 tile count + V5 latitude-band ramp (measured vs geometry-predicted)."""
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        res = abs(ds.transform.a)
        a = ds.read(1, out_shape=(H // DECIM, W // DECIM)).astype(np.float32)   # 2D (h,w)
    valid = (a > 0) & (a > -1e30) & np.isfinite(a)
    valid_frac = float(valid.mean())
    valid_tiles = int(valid.sum() * DECIM * DECIM / (TILE_PX * TILE_PX))

    # V5: median ln(I/F) per latitude band over valid pixels
    la = np.log(np.where(valid, a, np.nan))
    rows = np.where(valid.any(axis=1))[0]
    measured = np.nan
    dlat = np.nan
    if rows.size > 40:
        edges = np.linspace(rows.min(), rows.max(), 41).astype(int)
        bc, bm = [], []
        for k in range(40):
            fin = la[edges[k]:edges[k + 1]]
            fin = fin[np.isfinite(fin)]
            if fin.size >= 30:
                bm.append(float(np.median(fin))); bc.append((edges[k] + edges[k + 1]) / 2)
        if len(bm) >= 10:
            bc, bm = np.array(bc), np.array(bm)
            slope = np.polyfit(bc, bm, 1)[0]
            measured = abs((np.exp(slope * (bc.max() - bc.min())) - 1) * 100)
            dlat = (bc.max() - bc.min()) * DECIM * res / (R_MARS_M * np.pi / 180)
    predicted = (abs(K_MINNAERT * np.tan(np.radians(incidence)) * np.radians(DI_DLAT * dlat)) * 100
                 if np.isfinite(dlat) else np.nan)
    return {"valid_frac": round(valid_frac, 3), "valid_tiles": valid_tiles,
            "dlat_deg": round(float(dlat), 2) if np.isfinite(dlat) else np.nan,
            "ramp_measured_pct": round(float(measured), 2) if np.isfinite(measured) else np.nan,
            "ramp_predicted_illum_pct": round(float(predicted), 2) if np.isfinite(predicted) else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--frame-list", default=str(REPO / "reports" / "f_build" / "sizing_frame_list.csv"))
    ap.add_argument("--tile-map", default=str(REPO / "reports" / "figures" / "frame_tile_map.csv"))
    ap.add_argument("--timing-csv", default=None)
    ap.add_argument("--gpu-scale", type=float, default=0.5,
                    help="s/frame multiplier from THIS GPU to L40S (RTX 5070 ~2x slower than L40S ~= 0.5)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    fl = pd.read_csv(args.frame_list)
    fdir = Path(args.frames_dir)
    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)
    rate = embedder_rate(embedder)
    print(f"embedder rate (valid tiles, this GPU): {rate:.0f} tiles/s\n")

    rows = []
    for _, fr in fl.iterrows():
        pid = fr["PRODUCT_ID"]
        path = next((fdir / f"{pid}{e}" for e in (".map.tif", ".map.cub", ".tif")
                     if (fdir / f"{pid}{e}").exists()), None)
        if path is None:
            print(f"  ⚠ {pid}: no raster in {fdir} — skipped"); continue
        st = frame_stats(path, float(fr["inc_sel"]))
        rows.append({"PRODUCT_ID": pid, "n_tiles": int(fr["n_tiles"]),
                     "incidence": float(fr["inc_sel"]), **st})
        print(f"  {pid}: valid {st['valid_tiles']:,} tiles (frac {st['valid_frac']}); "
              f"V5 ramp measured {st['ramp_measured_pct']}% / predicted-illum "
              f"{st['ramp_predicted_illum_pct']}% (dlat {st['dlat_deg']}deg)")

    df = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "fbuild_sizing_probe.csv", index=False)

    # ---- V1: bias-corrected 907 extrapolation (scale per frame-tile footprint) ----
    n_ft_total = len(pd.read_csv(args.tile_map))          # total frame x tile rows (~1371)
    tiles_per_ft = df["valid_tiles"].sum() / df["n_tiles"].sum()
    total_tiles = tiles_per_ft * n_ft_total
    gpu_h_local = total_tiles / rate / 3600.0
    gpu_h_l40s = gpu_h_local * args.gpu_scale
    print("\n=== V1 — array sizing (907 frames) ===")
    print(f"valid tiles per frame-tile footprint {tiles_per_ft:,.0f} x {n_ft_total} rows "
          f"-> {total_tiles/1e6:.0f}M valid tiles (plan est. 120-170M)")
    print(f"embed: {gpu_h_local:.0f} GPU-h on this GPU; ~{gpu_h_l40s:.0f} L40S-h "
          f"(gpu_scale {args.gpu_scale}); plan est. 25-40 L40S-h")
    if args.timing_csv and Path(args.timing_csv).exists():
        t = pd.read_csv(args.timing_csv)
        t = t[t["status"] == "ok"] if "status" in t.columns else t
        # ISIS cost scales per frame-tile footprint too (long frames cost more)
        probe_ft = df.set_index("PRODUCT_ID")["n_tiles"]
        t = t.assign(nt=[probe_ft.get(p, 1) for p in t["product_id"]])
        cpu_s_per_ft = (t["t_total"] * 1).sum() / t["nt"].sum()
        print(f"ISIS (Stage A): {cpu_s_per_ft*n_ft_total/3600:,.0f} CPU-h "
              f"({cpu_s_per_ft*n_ft_total/3600/32:,.1f} h wall @32 tasks); plan est. 333 CPU-h")

    # ---- V5 verdict on the geometry-predicted illumination ramp ----
    print("\n=== V5 — within-frame incidence ramp ===")
    print(df[["PRODUCT_ID", "incidence", "n_tiles", "dlat_deg",
              "ramp_measured_pct", "ramp_predicted_illum_pct"]].to_string(index=False))
    worst_pred = df["ramp_predicted_illum_pct"].max()
    verdict = ("PER-FRAME cos^k(i) scalar OK (predicted illumination ramp < 1%)"
               if worst_pred < RAMP_PER_ROW_THRESH else
               "ADOPT per-row cos^k(i(lat)) in Stage B BEFORE the array "
               f"(predicted illumination ramp up to {worst_pred:.1f}% on long frames)")
    print(f"\nworst predicted illumination ramp = {worst_pred:.2f}%  ->  {verdict}")
    print("(measured ramp is geology-dominated over 300 km frames; per-row corrects only the "
          "incidence component and leaves real albedo alone)")
    print(f"\nwrote {FIG / 'fbuild_sizing_probe.csv'}")


if __name__ == "__main__":
    main()
