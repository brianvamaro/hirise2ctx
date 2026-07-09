"""H2 (PLAN_StripingArtifact PHASE 2) — η² sweep: does removing the frame-nuisance
subspace flatten the per-frame blocks beyond H1?

Embeds the 7 E8_N44 pilot frames ONCE under the H1 minnaert_center mapping (the
expensive GPU pass, cached), then scores a set of heads that differ only in how many
nuisance directions they project out:
  k=0   models/deployable_f_center/<hash>      (H1 baseline — no projection)
  k>0   models/deployable_f_h2_k{K}/<hash>     (H2 — top-k nuisance directions removed)

Because the projection is baked into DeployableHead (travels via load), scoring a head is
just predict()+composite on the cached embeddings — no re-embedding per k.

Metrics per head: frame-block η² (partition + median composite; DECISIONS 2026-06-18d) and
prediction overlap |Δp| vs the input I/F |ratio-1| (the embedder-amplification check).
Baselines on this crop: mosaic raw 0.196 / A1 0.141 / H1 (center) 0.081 median / 0.128 partition.

Run (laptop GPU):
  conda run --no-capture-output -n geospatial python -u scripts/f_h2_eta2.py \
      --heads center:models/deployable_f_center/86c51a5dca220f63 \
              h2_k4:models/deployable_f_h2_k4/86c51a5dca220f63 \
              h2_k16:models/deployable_f_h2_k16/86c51a5dca220f63 \
              h2_k64:models/deployable_f_h2_k64/86c51a5dca220f63
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/torch

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.f_pilot_crop as fpc
from src.ctx_edr import frame_table
from src.fm_embeddings import FangEmbedder, tile_grid_for_window
from src.mapping import (coarsened_transform, own_tile_zero_fraction,
                         tile_origin_transform, tiles_to_raster)
from src.modeling.mlp_head import DeployableHead
from src.striping import eta2

MAPPING = "minnaert_center"
K_MINNAERT = 0.580
FIG = REPO / "reports" / "figures"
EMB_CACHE = fpc.WORK / "h2_emb"           # per-frame embeddings (shared across heads)


def embed_frames(pids, ctx, embedder) -> dict:
    """{pid: {gem, valid, zf}} on the shared pilot tile grid (ti/tj identical for all)."""
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    ti = tj = None
    out = {}
    for pid in pids:
        cache = EMB_CACHE / f"{pid}.npz"
        if cache.exists():
            z = np.load(cache)
            out[pid] = {"gem": z["gem"], "valid": z["valid"], "zf": z["zf"]}
            ti, tj = z["ti"], z["tj"]
            print(f"  {pid}: cached", flush=True)
            continue
        data8 = fpc.mapped_uint8(pid, MAPPING, ctx)
        ti, tj = tile_grid_for_window(data8.shape, fpc.R0, fpc.C0, fpc.TILE_PX)
        emb, valid = embedder.embed_window(data8, ti, tj, tile_px=fpc.TILE_PX,
                                           row0=fpc.R0, col0=fpc.C0, pool="gem",
                                           batch=fpc.BATCH)
        zf = own_tile_zero_fraction(data8, ti, tj, tile_px=fpc.TILE_PX,
                                    row0=fpc.R0, col0=fpc.C0)
        np.savez_compressed(cache, ti=ti, tj=tj, gem=emb.astype(np.float32),
                            valid=valid, zf=zf)
        out[pid] = {"gem": emb.astype(np.float32), "valid": valid, "zf": zf}
        print(f"  {pid}: {int(valid.sum())} valid tiles", flush=True)
        del data8
    return out, ti, tj


def head_rasters(head, frames, pids, ti, tj):
    """Predict every frame with one head -> (stack (n,H,W), transform)."""
    stacks, transform = [], None
    for pid in pids:
        fr = frames[pid]
        usable = fr["valid"] & (fr["zf"] <= 0.5)
        prob = np.full(ti.size, np.nan)
        if usable.any():
            prob[usable] = head.predict(fr["gem"][usable])
        raster, ti_min, tj_min = tiles_to_raster(ti, tj, prob, fill=np.nan)
        stacks.append(raster.astype(np.float32))
        if transform is None:
            tt = tile_origin_transform(tuple(fpc.T5)[:6], fpc.R0, fpc.C0)
            transform = coarsened_transform(tt, ti_min, tj_min, fpc.TILE_PX)
    return np.stack(stacks), transform


def score(label, stack, pids, transform):
    from rasterio.transform import Affine

    shape = stack.shape[1:]
    labels = fpc.frame_labels(pids, shape, Affine(*tuple(transform)[:6]))
    valid = np.isfinite(stack)
    part = np.full(shape, np.nan, dtype=np.float32)
    for i in range(len(pids)):
        sel = (labels == i) & valid[i]
        part[sel] = stack[i][sel]
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(stack, axis=0).astype(np.float32)
    rows = {}
    for name, comp in (("partition", part), ("median", med)):
        fin = np.isfinite(comp) & (labels >= 0)
        rows[name] = round(float(eta2(comp, labels, fin)), 4)
    # prediction overlap disagreement (median over pairs)
    diffs = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            both = valid[i] & valid[j]
            if both.sum() >= 200:
                diffs.append(float(np.median(np.abs(stack[i][both] - stack[j][both]))))
    rows["pred_overlap"] = round(float(np.median(diffs)), 4) if diffs else np.nan
    return rows, labels, part, med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", required=True,
                    help="label:path pairs, e.g. center:models/deployable_f_center/<hash>")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    pids = fpc.crop_pids()
    fpc.WORK.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        fpc.align_frame(pid)

    ft = frame_table(fpc.TILE).set_index("PRODUCT_ID")
    cos_i = {p: float(np.cos(np.radians(ft.loc[p, "INCIDENCE"]))) for p in pids}
    # stretch bounds = the H1 store's centered pool (must match training); read from the
    # nuisance-basis npz so there is one source of truth.
    nb = np.load(REPO / "reports" / "f_leg_b" / "h2_nuisance_basis.npz")
    lo, hi = float(nb["stretch_lo"]), float(nb["stretch_hi"])
    ctx = {"cos_i": cos_i, "minnaert_div": {p: cos_i[p] ** K_MINNAERT for p in pids},
           "log_lohi": (lo, hi)}
    print(f"minnaert_center: k={K_MINNAERT}, log stretch I/F {lo:.4f}..{hi:.4f}", flush=True)

    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)
    frames, ti, tj = embed_frames(pids, ctx, embedder)

    results = []
    figs = {}
    for spec in args.heads:
        label, path = spec.split(":", 1)
        head = DeployableHead.load(Path(path))
        k = head.nuisance_basis.shape[1] if head.nuisance_basis is not None else 0
        stack, transform = head_rasters(head, frames, pids, ti, tj)
        rows, labels, part, med = score(label, stack, pids, transform)
        rows.update(label=label, k=k)
        results.append(rows)
        figs[label] = (labels, part, med, transform)
        print(f"[{label}] k={k}  η² partition {rows['partition']}  median {rows['median']}"
              f"  pred_overlap {rows['pred_overlap']}", flush=True)

    # input I/F overlap for context (one number)
    cif = {p: fpc.coarse_if(p, figs[args.heads[0].split(':',1)[0]][1].shape) for p in pids}
    ifd = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = cif[pids[i]], cif[pids[j]]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() >= 200:
                ifd.append(float(np.median(np.abs(a[both] / b[both] - 1))))
    if_overlap = round(float(np.median(ifd)), 4)

    import pandas as pd
    df = pd.DataFrame(results)[["label", "k", "partition", "median", "pred_overlap"]]
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "f_h2_eta2_summary.csv", index=False)
    print("\n=== H2 η² SWEEP (baselines: mosaic raw 0.196 / A1 0.141 / H1 0.081 med, "
          "0.128 part) ===")
    print(df.to_string(index=False))
    print(f"\ninput I/F overlap |ratio-1| = {if_overlap}  "
          f"(H1 pred_overlap was 0.073; amplification killed if pred_overlap <= this)")

    # choropleth figure: one column per head
    n = len(figs)
    fig, ax = plt.subplots(2, n, figsize=(5.2 * n, 9), squeeze=False)
    for c, (label, (labels, part, med, _)) in enumerate(figs.items()):
        vmax = np.nanpercentile(part, 99)
        chor = np.full(part.shape, np.nan, dtype=np.float32)
        for i in range(len(pids)):
            sel = (labels == i) & np.isfinite(part)
            if sel.sum() >= 30:
                chor[labels == i] = np.nanmean(part[sel])
        for r, (img, t) in enumerate([(med, f"{label}: median composite"),
                                      (chor, f"{label}: frame-mean choropleth")]):
            im = ax[r, c].imshow(img, cmap="magma", vmax=vmax)
            ax[r, c].set_title(t, fontsize=9)
            plt.colorbar(im, ax=ax[r, c], fraction=0.046)
    fig.suptitle("H2 nuisance-subspace removal — E8_N44 pilot — P(boulder-rich), P(fa>1e-2)")
    fig.tight_layout()
    out = FIG / "f_h2_eta2_choropleth.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
