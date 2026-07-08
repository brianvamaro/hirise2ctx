"""F pilot, leg A (PLAN_StripingArtifact 2026-07-03): does per-source-frame inference kill
the frame blocks?

Consumes the 7 calibrated, projected I/F crops from Sherlock
(`reports/f_timing/pilot_crops/{PID}_ifcrop.tif`), aligns each onto the EXACT mosaic-crop 5 m
grid of the A1 payoff test (E8_N44, row_off 1504 / col_off 8992 / 15008 px), maps I/F to the
embedder's uint8 domain under four variants, embeds + predicts per frame with the existing
mosaic-trained heads, composites, and scores the frame-block eta^2 against the SAME-crop
baselines: mosaic raw 0.196, A1 0.141 (scripts/striping_a1_infer_crop.py).

Mappings (I/F -> uint8 [1,255], 0 = nodata):
  affine   — ONE global linear stretch (pooled p2–p98): the "calibrated frames need no
             per-frame handling" bet. THE headline variant.
  lambert  — divide by cos(incidence) per frame (SeamMap metadata) first, then the global
             stretch: removes the real cross-frame illumination signal ctxcal exposes.
  minnaert — divide by cos^k(incidence) with empirical k (fitted log-log; A0: k≈0.66);
             better than Lambert (overcorrects); metadata-only at deploy.
  perframe — per-frame robust (median/IQR -> 125/27.7, the A1 transform): reference point;
             if only this works, F needs per-frame normalization anyway (A1's ceiling
             argument returns, though now on artifact-free radiometry).

Caveat by design: the heads were trained on mosaic-stretch embeddings, so ABSOLUTE
calibration is not scored here — only between-frame structure (eta^2, the artifact) and
frame-overlap agreement (prediction + I/F; the Walter ±2% check).

Run (GPU, ~2.5-3 h total for all 4 mappings):
  conda run -n geospatial python scripts/f_pilot_crop.py                  # all 3 mappings
  conda run -n geospatial python scripts/f_pilot_crop.py --mappings affine --frames 2  # smoke
Phases cache under reports/f_timing/pilot_work/; delete that dir to recompute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/torch imports

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.ctx_edr import frame_table
from src.fm_embeddings import FangEmbedder, tile_grid_for_window
from src.mapping import (CtxWindow, coarsened_transform, own_tile_zero_fraction,
                         tile_origin_transform, tiles_to_raster)
from src.modeling.mlp_head import DeployableHead
from src.striping import eta2, load_frames

TILE = "E8_N44"
R0, C0, SIZE = 1504, 8992, 15008          # the A1-payoff crop (native 5 m px)
CROP_UL = (519317.3, 2837505.5)           # world UL of that crop (probe _f_pilot_bounds)
T5 = Affine(5.0, 0.0, CROP_UL[0], 0.0, -5.0, CROP_UL[1])
TILE_PX, BATCH = 32, 256
CROPS = REPO / "reports" / "f_timing" / "pilot_crops"
WORK = REPO / "reports" / "f_timing" / "pilot_work"
FIG = REPO / "reports" / "figures"
BASE_HEAD = REPO / "models" / "deployable" / "86c51a5dca220f63"
A1_HEAD = REPO / "models" / "deployable_a1" / "86c51a5dca220f63"
BASELINES = {"mosaic_raw": 0.196, "mosaic_a1": 0.141}
A1_M0, A1_S0 = 125.0, 27.7


def crop_pids() -> list[str]:
    return sorted(p.name.replace("_ifcrop.tif", "") for p in CROPS.glob("*_ifcrop.tif"))


# ---------------------------------------------------------------- phase 1: align to crop grid
def aligned_path(pid: str) -> Path:
    return WORK / "aligned" / f"{pid}.npy"


def align_frame(pid: str) -> None:
    out = aligned_path(pid)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(CROPS / f"{pid}_ifcrop.tif") as src:
        dst = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
        reproject(rasterio.band(src, 1), dst, dst_transform=T5, dst_crs=src.crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    np.save(out, dst)
    v = dst[np.isfinite(dst)]
    print(f"  aligned {pid}: valid {v.size / dst.size:.0%}, I/F median {np.median(v):.4f}",
          flush=True)


# ---------------------------------------------------------------- phase 2: I/F -> uint8 maps
def pooled_percentiles(pids, divisor: dict | None = None, step=16) -> tuple[float, float]:
    vals = []
    for pid in pids:
        a = np.load(aligned_path(pid), mmap_mode="r")[::step, ::step]
        a = a[np.isfinite(a)]
        if divisor is not None:
            a = a / divisor[pid]
        vals.append(a)
    pool = np.concatenate(vals)
    return float(np.percentile(pool, 2)), float(np.percentile(pool, 98))


def frame_medians(pids, step=16) -> dict:
    out = {}
    for pid in pids:
        a = np.load(aligned_path(pid), mmap_mode="r")[::step, ::step]
        out[pid] = float(np.median(a[np.isfinite(a)]))
    return out


def to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    fin = np.isfinite(arr)
    out[fin] = np.clip((arr[fin] - lo) / (hi - lo) * 254.0 + 1.0, 1, 255).astype(np.uint8)
    return out


def to_uint8_log(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """log-domain stretch: ln(I/F) lo..hi -> 1..255 (matches f_leg_b_embed --stretch-scale log)."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    fin = np.isfinite(arr) & (arr > 0)
    v = np.log(arr[fin])
    llo, lhi = np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6))
    out[fin] = np.clip((v - llo) / (lhi - llo) * 254.0 + 1.0, 1, 255).astype(np.uint8)
    return out


def mapped_uint8(pid: str, mapping: str, ctx: dict) -> np.ndarray:
    arr = np.load(aligned_path(pid)).astype(np.float32)
    if mapping == "affine":
        lo, hi = ctx["affine_lohi"]
        return to_uint8(arr, lo, hi)
    if mapping == "lambert":
        lo, hi = ctx["lambert_lohi"]
        return to_uint8(arr / ctx["cos_i"][pid], lo, hi)
    if mapping == "minnaert":
        lo, hi = ctx["minnaert_lohi"]
        return to_uint8(arr / ctx["minnaert_div"][pid], lo, hi)
    if mapping == "minnaert_log":
        # deploy recipe (leg B GATE PASS, DECISIONS 2026-07-05b): FIXED training k +
        # log-domain stretch bounds, per-frame cos^k(i) divisor from this frame's incidence
        lo, hi = ctx["log_lohi"]
        return to_uint8_log(arr / ctx["minnaert_div"][pid], lo, hi)
    if mapping == "minnaert_center":
        # H1 (PLAN_StripingArtifact PHASE 2): minnaert_log + per-frame median centering.
        # MUST mirror f_leg_b_embed minnaert_center exactly (÷cos^k, then ÷ this frame's
        # OWN median, then the FIXED centered-space log stretch) — else train/deploy
        # mismatch (the leg-A failure mode). Stretch bounds = training centered pool.
        lo, hi = ctx["log_lohi"]
        d = arr / ctx["minnaert_div"][pid]
        d = d / float(np.nanmedian(d))
        return to_uint8_log(d, lo, hi)
    if mapping == "perframe":
        fin = np.isfinite(arr)
        med = float(np.median(arr[fin]))
        q75, q25 = np.percentile(arr[fin], [75, 25])
        iqr = float(max(q75 - q25, 1e-6))
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[fin] = np.clip((arr[fin] - med) / iqr * A1_S0 + A1_M0, 1, 255).astype(np.uint8)
        return out
    raise ValueError(mapping)


# ---------------------------------------------------------------- phase 3: embed + predict
def predict_frame(data8: np.ndarray, embedder, heads: dict) -> dict:
    """One embedding pass, predictions from every head (predict_window internals, reused)."""
    window = CtxWindow(data=data8, row_off=R0, col_off=C0, transform=tuple(T5)[:6], crs_wkt="")
    ti, tj = tile_grid_for_window(data8.shape, R0, C0, TILE_PX)
    emb, valid = embedder.embed_window(data8, ti, tj, tile_px=TILE_PX, row0=R0, col0=C0,
                                       pool="gem", batch=BATCH)
    zf = own_tile_zero_fraction(data8, ti, tj, tile_px=TILE_PX, row0=R0, col0=C0)
    usable = valid & (zf <= 0.5)
    out = {}
    for name, head in heads.items():
        prob = np.full(ti.size, np.nan)
        if usable.any():
            prob[usable] = head.predict(emb[usable])
        raster, ti_min, tj_min = tiles_to_raster(ti, tj, prob, fill=np.nan)
        out[name] = raster.astype(np.float32)
    tt = tile_origin_transform(window.transform, R0, C0)
    out["transform"] = coarsened_transform(tt, ti_min, tj_min, TILE_PX)
    return out


def preds_path(mapping: str) -> Path:
    return WORK / f"preds_{mapping}.npz"


def run_mapping(mapping: str, pids, ctx, embedder, heads) -> None:
    out = preds_path(mapping)
    if out.exists():
        print(f"[{mapping}] cached", flush=True)
        return
    stacks = {h: [] for h in heads}
    transform = None
    for pid in pids:
        data8 = mapped_uint8(pid, mapping, ctx)
        r = predict_frame(data8, embedder, heads)
        transform = r["transform"]
        for h in heads:
            stacks[h].append(r[h])
        n = int(np.isfinite(r[next(iter(heads))]).sum())
        print(f"[{mapping}] {pid}: {n} tiles", flush=True)
        del data8
    np.savez_compressed(out, pids=np.array(pids),
                        transform=np.array(tuple(transform)[:6]),
                        **{h: np.stack(stacks[h]) for h in heads})
    print(f"[{mapping}] wrote {out}", flush=True)


# ---------------------------------------------------------------- phase 4: composite + score
def frame_labels(pids, shape, transform) -> np.ndarray:
    g = load_frames(TILE)
    idx = {pid: i for i, pid in enumerate(pids)}
    shapes = [(geom, idx[p]) for geom, p in zip(g.geometry, g["PRODUCT_ID"]) if p in idx]
    return rasterize(shapes, out_shape=shape, transform=transform, fill=-1,
                     dtype="int32", all_touched=False)


def coarse_if(pid: str, shape) -> np.ndarray:
    """Aligned I/F block-averaged onto the coarse grid (for the ±2% overlap check)."""
    a = np.load(aligned_path(pid))
    n = SIZE // TILE_PX
    b = a.reshape(n, TILE_PX, n, TILE_PX)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(b, axis=(1, 3)).astype(np.float32)
    frac = np.isfinite(b).mean(axis=(1, 3))
    m[frac < 0.9] = np.nan
    # coarse pred rasters may be trimmed at edges vs n x n; crop to `shape`
    return m[: shape[0], : shape[1]]


def erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=1)


def analyze(mapping: str, rows: list, pair_rows: list) -> dict:
    z = np.load(preds_path(mapping), allow_pickle=False)
    pids = [str(p) for p in z["pids"]]
    transform = Affine(*z["transform"])
    heads = [k for k in z.files if k not in ("pids", "transform")]
    shape = z[heads[0]].shape[1:]
    labels = frame_labels(pids, shape, transform)
    figs = {}
    for h in heads:
        stack = z[h]                                   # (n_frames, H, W)
        valid = np.isfinite(stack)
        # composite (i): the SeamMap partition (each cell from ITS selected frame)
        part = np.full(shape, np.nan, dtype=np.float32)
        part_er = np.full(shape, np.nan, dtype=np.float32)
        for i in range(len(pids)):
            sel = (labels == i) & valid[i]
            part[sel] = stack[i][sel]
            sel_er = (labels == i) & erode(valid[i])
            part_er[sel_er] = stack[i][sel_er]
        # composite (ii): median over overlapping frames (deploy style)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(stack, axis=0).astype(np.float32)
        for comp_name, comp in [("partition", part), ("partition_eroded", part_er),
                                ("median", med)]:
            fin = np.isfinite(comp) & (labels >= 0)
            e = eta2(comp, labels, fin)
            rows.append({"mapping": mapping, "head": h, "composite": comp_name,
                         "eta2": round(float(e), 4), "n_cells": int(fin.sum()),
                         "n_frames": int(np.unique(labels[fin]).size)})
        figs[h] = (part, med)
        # prediction overlap agreement (pairs)
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                both = valid[i] & valid[j]
                if both.sum() < 200:
                    continue
                d = stack[i][both] - stack[j][both]
                r = float(np.corrcoef(stack[i][both], stack[j][both])[0, 1])
                pair_rows.append({"mapping": mapping, "head": h, "kind": "pred",
                                  "pair": f"{pids[i][:3]}~{pids[j][:3]}", "n": int(both.sum()),
                                  "median_absdiff": round(float(np.median(np.abs(d))), 4),
                                  "corr": round(r, 3)})
    return {"labels": labels, "figs": figs, "pids": pids, "transform": transform,
            "shape": shape}


def overlap_if_check(pids, shape, pair_rows) -> None:
    cif = {p: coarse_if(p, shape) for p in pids}
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = cif[pids[i]], cif[pids[j]]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() < 200:
                continue
            ratio = a[both] / b[both]
            pair_rows.append({"mapping": "-", "head": "-", "kind": "IF",
                              "pair": f"{pids[i][:3]}~{pids[j][:3]}", "n": int(both.sum()),
                              "median_absdiff": round(float(np.median(np.abs(ratio - 1))), 4),
                              "corr": round(float(np.corrcoef(a[both], b[both])[0, 1]), 3)})


def render(mapping: str, res: dict) -> None:
    heads = list(res["figs"])
    fig, ax = plt.subplots(len(heads), 3, figsize=(16, 5.2 * len(heads)), squeeze=False)
    for k, h in enumerate(heads):
        part, med = res["figs"][h]
        vmax = np.nanpercentile(part, 99)
        chor = np.full(res["shape"], np.nan, dtype=np.float32)
        for i in range(len(res["pids"])):
            sel = (res["labels"] == i) & np.isfinite(part)
            if sel.sum() >= 30:
                chor[res["labels"] == i] = np.nanmean(part[sel])
        for a, r, t in [(ax[k, 0], part, f"{h}: partition composite"),
                        (ax[k, 1], med, f"{h}: median composite"),
                        (ax[k, 2], chor, f"{h}: frame-mean choropleth")]:
            im = a.imshow(r, cmap="magma", vmax=vmax)
            a.set_title(t, fontsize=10)
            plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"F pilot — per-frame inference on the E8_N44 crop — mapping: {mapping}\n"
                 f"(baselines on this crop: mosaic raw eta²=0.196, A1 eta²=0.141)")
    fig.tight_layout()
    out = FIG / f"f_pilot_{mapping}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    global SIZE, WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--mappings", nargs="+",
                    default=["affine", "lambert", "minnaert", "perframe"])
    ap.add_argument("--frames", type=int, default=0, help="limit frames (smoke test)")
    ap.add_argument("--smoke-size", type=int, default=0,
                    help="shrink the crop to N native px (multiple of 32; smoke test)")
    ap.add_argument("--cpu", action="store_true", help="force CPU embedding (GPU busy)")
    # leg B eta2 confirmation: fixed deploy constants + a retrained head (DECISIONS 2026-07-05b)
    ap.add_argument("--minnaert-k", type=float, default=None,
                    help="FIXED Minnaert k for minnaert_log (training value, e.g. 0.580); "
                         "if omitted, minnaert/minnaert_log fit k on the pilot frames")
    ap.add_argument("--stretch-lohi", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="FIXED I/F stretch bounds for minnaert_log (training p0.5–p99.5)")
    ap.add_argument("--head-dir", default=None,
                    help="load ONE deployable head from here instead of base+a1 "
                         "(e.g. models/deployable_f_wl/<hash>)")
    ap.add_argument("--head-name", default="f_wl", help="label for --head-dir head")
    args = ap.parse_args()

    pids = crop_pids()
    if args.frames:
        pids = pids[: args.frames]
    if args.smoke_size:
        SIZE = args.smoke_size            # sub-crop from the same UL corner; grid stays anchored
        WORK = WORK.parent / "pilot_work_smoke"   # don't poison the full-run cache
    print(f"{len(pids)} frames, mappings: {args.mappings}, size {SIZE}px"
          f"{' (CPU)' if args.cpu else ''}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)

    for pid in pids:
        align_frame(pid)

    ft = frame_table(TILE).set_index("PRODUCT_ID")
    cos_i = {p: float(np.cos(np.radians(ft.loc[p, "INCIDENCE"]))) for p in pids}
    ctx = {"cos_i": cos_i}
    if "affine" in args.mappings:
        ctx["affine_lohi"] = pooled_percentiles(pids)
        print(f"affine stretch: I/F {ctx['affine_lohi'][0]:.4f}..{ctx['affine_lohi'][1]:.4f}",
              flush=True)
    if "lambert" in args.mappings:
        ctx["lambert_lohi"] = pooled_percentiles(pids, divisor=cos_i)
    if "minnaert" in args.mappings:
        # empirical Minnaert exponent from the frames themselves: slope of
        # log(median I/F) vs log(cos i) — metadata-only at deploy once k is fixed
        med = frame_medians(pids)
        k = float(np.polyfit([np.log(cos_i[p]) for p in pids],
                             [np.log(med[p]) for p in pids], 1)[0])
        ctx["k_minnaert"] = k
        ctx["minnaert_div"] = {p: cos_i[p] ** k for p in pids}
        ctx["minnaert_lohi"] = pooled_percentiles(pids, divisor=ctx["minnaert_div"])
        print(f"minnaert k = {k:.3f} (fit from {len(pids)} frame medians)", flush=True)
    if "minnaert_log" in args.mappings or "minnaert_center" in args.mappings:
        # DEPLOY recipe: FIXED training k + log stretch bounds applied to these frames
        # (per-frame cos^k(i) uses each frame's own incidence). No fitting on the pilot.
        # minnaert_center additionally centers per-frame by median (H1) — same k/bounds
        # source, bounds are the CENTERED training pool (~0.84..1.12) for that mapping.
        if args.minnaert_k is None or args.stretch_lohi is None:
            raise SystemExit("minnaert_log/minnaert_center need --minnaert-k and "
                             "--stretch-lohi (the training constants; DECISIONS 2026-07-05b/d)")
        k = float(args.minnaert_k)
        ctx["minnaert_div"] = {p: cos_i[p] ** k for p in pids}
        ctx["log_lohi"] = tuple(args.stretch_lohi)
        print(f"minnaert(k={k:.3f}) log stretch I/F "
              f"{args.stretch_lohi[0]:.4f}..{args.stretch_lohi[1]:.4f}", flush=True)

    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)
    if args.head_dir:
        heads = {args.head_name: DeployableHead.load(Path(args.head_dir))}
        print(f"head: {args.head_name} <- {args.head_dir}", flush=True)
    else:
        heads = {"base": DeployableHead.load(BASE_HEAD)}
        if A1_HEAD.exists():
            heads["a1"] = DeployableHead.load(A1_HEAD)

    for m in args.mappings:
        run_mapping(m, pids, ctx, embedder, heads)

    rows, pair_rows = [], []
    res = None
    for m in args.mappings:
        res = analyze(m, rows, pair_rows)
        render(m, res)
    if res is not None:
        overlap_if_check(res["pids"], res["shape"], pair_rows)

    import pandas as pd

    df = pd.DataFrame(rows)
    dp = pd.DataFrame(pair_rows)
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "f_pilot_eta2_summary.csv", index=False)
    dp.to_csv(FIG / "f_pilot_overlap_pairs.csv", index=False)
    print("\n=== F PILOT eta² (baselines: mosaic raw 0.196 / A1 0.141) ===")
    print(df.to_string(index=False))
    print("\n=== overlap agreement (kind=IF rows: median|ratio-1| vs the ±2% claim) ===")
    if len(dp):
        print(dp.groupby(["kind", "mapping", "head"])["median_absdiff"]
              .median().to_string())
    print(f"\nfull tables: {FIG / 'f_pilot_eta2_summary.csv'}, "
          f"{FIG / 'f_pilot_overlap_pairs.csv'}")


if __name__ == "__main__":
    main()
