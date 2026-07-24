"""PLAN_FBuild Stage-B sizing probe — V1 (embed throughput → array sizing) + V5 (within-frame
incidence ramp → per-frame vs per-row cos^k(i)).

Runs AFTER Stage A: `run_f_build_probe.sbatch` (KEEP_CUBES=1) leaves the 5 selected frames as
projected `{pid}.map.cub`s on Sherlock scratch. Convert to GeoTIFF (`isis2std`/`gdal_translate`)
or read the cubes directly (GDAL ISIS3 driver), point `--frames-dir` at them, and pass the ISIS
`--timing-csv` for the cost ledger. Analysis runs on a GPU (Sherlock L40S or the local RTX 5070 —
tiles/frame is hardware-independent; s/frame is scaled to L40S via `--gpu-scale`).

V1 — embed throughput. Sample `--n-windows` full-res core windows spread across each frame, run the
     deploy path (H1 uint8 mapping → FangEmbedder.embed_window → DeployableHead.predict), measure
     usable tiles/window + seconds/window → tiles/frame + s/frame → 907-frame array size + GPU-h.
     (uint8 content barely affects embed cost, so per-window median centering is used for timing;
     the fixed stretch/k only matter to science, checked in V5.)

V5 — within-frame ramp. On a decimated whole-frame read, fit per-row median ln(I/F) vs latitude.
     KEY: a per-frame scalar cos^k(i) (any constant) and the median-centering (also a constant)
     do NOT change the row-wise SLOPE, so the residual within-frame ramp is INVARIANT to the exact
     incidence value — the SeamMap incidence is fine here. Report the measured top-to-bottom ramp
     (%) and the geometry-PREDICTED illumination ramp k·tan(i)·Δi (Δi = di/dlat·Δlat). Verdict per
     PLAN_FBuild V5: residual <~0.5% → per-frame scalar OK; ≥~1% → switch Stage B to per-row
     cos^k(i(lat)) before the 907-frame array. Measured≈predicted ⇒ illumination (per-row fixes it);
     measured≫predicted ⇒ likely real geology (not an artifact; per-row would not help).

Run (GPU; after Stage A):
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run --no-capture-output -n geospatial python -u \
      scripts/f_build_sizing_probe.py --frames-dir reports/f_build/probe_cubes \
      --timing-csv reports/f_build/timing.csv
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
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.f_pilot_crop as fpc   # to_uint8_log, TILE_PX, BATCH
from src.fm_embeddings import FangEmbedder, tile_grid_for_window
from src.mapping import CtxWindow, own_tile_zero_fraction
from src.modeling.mlp_head import DeployableHead

# H1 minnaert_center constants (must match models/deployable_f_center; DECISIONS 2026-07-05b/07).
K_MINNAERT = 0.580
STRETCH_LO, STRETCH_HI = 0.8400, 1.1170
R_MARS_M = 3_396_190.0            # CTX equirect sphere radius (CLAUDE.md)
DI_DLAT = 0.635                   # deg incidence per deg latitude (audit 2026-07-23, cohort fit)
HEAD = REPO / "models" / "deployable_f_center" / "86c51a5dca220f63"
FIG = REPO / "reports" / "figures"
CORE, HALO = 2048, fpc.TILE_PX * 3    # window core + 3x3-context halo


def _read_incidence(pid: str, frame_list: Path) -> float:
    df = pd.read_csv(frame_list)
    row = df[df["PRODUCT_ID"] == pid]
    return float(row["inc_sel"].iloc[0]) if len(row) else np.nan


def v5_ramp(path: Path, incidence: float, stride: int = 8) -> dict:
    """Within-frame top-to-bottom ln(I/F) ramp (slope-invariant to the per-frame cos^k scalar)."""
    with rasterio.open(path) as ds:
        a = ds.read(1, out_shape=(1, ds.height // stride, ds.width // stride))[0].astype(np.float32)
        h_px = ds.height
    a[~np.isfinite(a)] = np.nan
    a[a <= 0] = np.nan
    row_med = np.array([np.nanmedian(r) if np.isfinite(r).sum() > 20 else np.nan
                        for r in np.log(a)])
    ok = np.isfinite(row_med)
    if ok.sum() < 10:
        return {"n_rows_ok": int(ok.sum())}
    rows = np.arange(len(row_med))[ok]
    slope, intercept = np.polyfit(rows, row_med[ok], 1)   # ln(I/F) per decimated-row
    ramp_ln = slope * (rows.max() - rows.min())           # top-to-bottom in ln units
    measured_pct = (np.exp(ramp_ln) - 1.0) * 100.0
    dlat_deg = (h_px * 5.0) / (R_MARS_M * np.pi / 180.0)  # frame latitude extent
    di_rad = np.radians(DI_DLAT * dlat_deg)
    pred_pct = (K_MINNAERT * np.tan(np.radians(incidence)) * di_rad) * 100.0 if np.isfinite(incidence) else np.nan
    return {"n_rows_ok": int(ok.sum()), "dlat_deg": round(float(dlat_deg), 2),
            "measured_ramp_pct": round(float(abs(measured_pct)), 2),
            "predicted_ramp_pct": round(float(abs(pred_pct)), 2) if np.isfinite(pred_pct) else np.nan,
            "incidence": round(float(incidence), 1)}


def _to_uint8(win_if: np.ndarray, incidence: float) -> np.ndarray:
    d = win_if / (np.cos(np.radians(incidence)) ** K_MINNAERT)
    fin = np.isfinite(d) & (d > 0)
    if fin.sum():
        d = d / float(np.median(d[fin]))
    return fpc.to_uint8_log(d, STRETCH_LO, STRETCH_HI)


def v1_throughput(path: Path, incidence: float, embedder, head, n_windows: int) -> dict:
    """Embed sampled core windows → usable tiles/window + s/window → per-frame extrapolation."""
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        ny, nx = max(H // CORE, 1), max(W // CORE, 1)
        # grid-sample up to n_windows core positions spread across the frame (incl. edges)
        gy = np.linspace(0, ny - 1, min(n_windows, ny)).round().astype(int)
        gx = np.linspace(0, nx - 1, max(1, n_windows // max(len(gy), 1))).round().astype(int)
        positions = [(int(iy * CORE), int(ix * CORE)) for iy in np.unique(gy) for ix in np.unique(gx)]
        tiles, secs = [], []
        for (y0, x0) in positions[:n_windows]:
            ry0, rx0 = max(y0 - HALO, 0), max(x0 - HALO, 0)
            h = min(CORE + 2 * HALO, H - ry0)
            w = min(CORE + 2 * HALO, W - rx0)
            if h < 3 * fpc.TILE_PX or w < 3 * fpc.TILE_PX:
                continue
            if_win = ds.read(1, window=Window(rx0, ry0, w, h)).astype(np.float32)
            u8 = _to_uint8(if_win, incidence)
            ti, tj = tile_grid_for_window(u8.shape, 0, 0, fpc.TILE_PX)
            if ti.size == 0:
                continue
            t0 = time.perf_counter()
            emb, valid = embedder.embed_window(u8, ti, tj, tile_px=fpc.TILE_PX, row0=0, col0=0,
                                               pool="gem", batch=fpc.BATCH)
            zf = own_tile_zero_fraction(u8, ti, tj, tile_px=fpc.TILE_PX, row0=0, col0=0)
            usable = valid & (zf <= 0.5)
            if usable.any():
                head.predict(emb[usable])
            secs.append(time.perf_counter() - t0)
            tiles.append(int(usable.sum()))
    if not tiles:
        return {"frame_px": f"{H}x{W}", "n_windows": 0}
    n_core = (int(np.ceil(H / CORE)) * int(np.ceil(W / CORE)))
    tpw, spw = float(np.mean(tiles)), float(np.mean(secs))
    return {"frame_px": f"{H}x{W}", "n_windows": len(tiles),
            "tiles_per_window": round(tpw, 1), "s_per_window": round(spw, 2),
            "tiles_per_s": round(sum(tiles) / max(sum(secs), 1e-6), 1),
            "core_windows": n_core,
            "tiles_per_frame": int(round(tpw * n_core)),
            "s_per_frame_thisgpu": round(spw * n_core, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="dir of {pid}.map.cub or {pid}.map.tif")
    ap.add_argument("--frame-list", default=str(REPO / "reports" / "f_build" / "sizing_frame_list.csv"))
    ap.add_argument("--timing-csv", default=None, help="Stage-A timing.csv for the ISIS cost ledger")
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--gpu-scale", type=float, default=1.0,
                    help="s/frame multiplier from THIS GPU to L40S (RTX 5070→L40S ≈ 1.0; set from parity)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    fdir = Path(args.frames_dir)
    frame_list = Path(args.frame_list)
    pids = [r["PRODUCT_ID"] for _, r in pd.read_csv(frame_list).iterrows()]

    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)
    head = DeployableHead.load(HEAD)

    rows = []
    for pid in pids:
        path = next((fdir / f"{pid}{ext}" for ext in (".map.cub", ".map.tif", ".tif")
                     if (fdir / f"{pid}{ext}").exists()), None)
        if path is None:
            print(f"  ⚠ {pid}: no cube/tif in {fdir} — skipped", flush=True)
            continue
        inc = _read_incidence(pid, frame_list)
        v5 = v5_ramp(path, inc)
        v1 = v1_throughput(path, inc, embedder, head, args.n_windows)
        rows.append({"PRODUCT_ID": pid, **v1, **{f"v5_{k}": v for k, v in v5.items()}})
        print(f"  {pid}: {v1.get('tiles_per_frame','?')} tiles/frame, "
              f"{v1.get('s_per_frame_thisgpu','?')} s/frame; "
              f"V5 ramp measured {v5.get('measured_ramp_pct','?')}% "
              f"(predicted {v5.get('predicted_ramp_pct','?')}%)", flush=True)

    df = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "fbuild_sizing_probe.csv", index=False)

    # ---- V1 array-sizing extrapolation ----
    print("\n=== V1 — array sizing (907 frames) ===", flush=True)
    tpf = df["tiles_per_frame"].mean()
    spf = df["s_per_frame_thisgpu"].mean() * args.gpu_scale
    total_tiles = tpf * 907
    gpu_h = spf * 907 / 3600.0
    print(f"mean tiles/frame {tpf:,.0f}  ->  907 frames ≈ {total_tiles/1e6:,.0f}M tile embeddings "
          f"(plan est. 120–170M)")
    print(f"mean s/frame (L40S-scaled) {spf:,.1f}  ->  {gpu_h:,.1f} GPU-h serial; "
          f"~{gpu_h/6:,.1f} h wall on 6 GPUs (plan est. 25–40 GPU-h)")
    if args.timing_csv and Path(args.timing_csv).exists():
        t = pd.read_csv(args.timing_csv)
        t_ok = t[t["status"] == "ok"] if "status" in t.columns else t
        cpu_h = t_ok["t_total"].mean() * 907 / 3600.0
        scr_gb = t_ok["map_mb"].mean() * 907 / 1000.0
        print(f"ISIS (Stage A): mean {t_ok['t_total'].mean():.0f} s/frame -> {cpu_h:,.0f} CPU-h "
              f"({cpu_h/32:,.1f} h wall @32 tasks); peak scratch if all cubes kept ≈ {scr_gb:,.0f} GB")

    # ---- V5 verdict ----
    print("\n=== V5 — within-frame incidence ramp ===", flush=True)
    worst = df["v5_measured_ramp_pct"].max()
    print(df[["PRODUCT_ID", "v5_incidence", "v5_dlat_deg",
              "v5_measured_ramp_pct", "v5_predicted_ramp_pct"]].to_string(index=False))
    verdict = ("PER-FRAME cos^k(i) OK (residual < ~0.5%)" if worst < 0.5 else
               "SWITCH to per-row cos^k(i(lat)) before the array (residual ≥ ~1%)" if worst >= 1.0 else
               "BORDERLINE (0.5–1%) — inspect measured-vs-predicted; lean per-row for long/grazing frames")
    print(f"\nworst measured within-frame ramp = {worst:.2f}%  ->  {verdict}")
    print("(measured≈predicted ⇒ illumination, per-row fixes it; measured≫predicted ⇒ likely geology)")
    print(f"\nwrote {FIG / 'fbuild_sizing_probe.csv'}")


if __name__ == "__main__":
    main()
