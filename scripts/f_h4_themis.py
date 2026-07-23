"""P3 (PLAN_FBuild §0 / PLAN_H4_Leveling §3.2 #2) — THEMIS night-IR ρ on the leveled pilot map.

The last pre-declared held-out check before the reopening call. Reruns the leg-1 thermal
correlation on the H4-LEVELED E8_N44 pilot composite and compares to the UNLEVELED H1 pilot.
Pre-declared gate: leveled Spearman ρ(P(rich), THEMIS night-IR) **not degraded** vs the
unleveled H1 map (edge-CV + leg-B skill, the decisive checks, already PASS — this is a guard
that leveling didn't launder away the weak real thermal signal).

All inputs are local:
  - per-frame embeddings cached from H2/H3 (`reports/f_timing/pilot_work/h2_emb/*.npz`);
  - the committed H4 offsets (`reports/figures/f_h4_offsets.csv`, λ*=300, full offsets per
    Brian's 2026-07-09b ruling) — reused so this scores the ACTUAL committed leveled map;
  - THEMIS night-IR cached on the CTX clon_0 CRS (`cache_v2/validation/themis_night_ir_region.tif`).

The pilot coarse grid (S=32 × 5 m = 160 m/px) is in the CTX CRS (f_pilot_crop.T5), so THEMIS
co-registers by a straight reproject onto (transform, shape). ρ reported for both the median
composite (deploy style) and the SeamMap partition composite, before & after H4.

Caveat (why this is a guard, not a gate): the footprint is one ~75 km crop where the regional
leg-1 signal was already weak (ρ≈+0.07); THEMIS is acceptance-gate #3 on the real build map
(PLAN_FBuild §5). Small |ρ| here is expected; the decision-relevant quantity is Δρ (after−before).

Run (laptop, seconds — embeddings cached):
  conda run --no-capture-output -n geospatial python -u scripts/f_h4_themis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import rasterio
from rasterio.transform import Affine
from scipy.stats import spearmanr

import scripts.f_h2_eta2 as h2
import scripts.f_h4_level as fhl
import scripts.f_pilot_crop as fpc
from src import validation_retrieve as vr
from src.fm_embeddings import FangEmbedder
from src.modeling.mlp_head import DeployableHead

FIG = REPO / "reports" / "figures"
THEMIS = REPO / "cache_v2" / "validation" / "themis_night_ir_region.tif"
OFFSETS = FIG / "f_h4_offsets.csv"
HEAD = REPO / "models" / "deployable_f_center" / "86c51a5dca220f63"
GATE_TOL = 0.02   # "not degraded" = Δρ >= -0.02 (mirrors the project skill tolerance)


def build_stack():
    """Reconstruct the H1 per-frame P(rich) stack + coarse-grid transform (cached embeddings)."""
    pids = fpc.crop_pids()
    fpc.WORK.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        fpc.align_frame(pid)
    ctx, _ft = fhl.build_ctx(pids)
    embedder = FangEmbedder.load()
    frames, ti, tj = h2.embed_frames(pids, ctx, embedder)
    head = DeployableHead.load(HEAD)
    stack, transform = h2.head_rasters(head, frames, pids, ti, tj)
    return pids, stack, transform


def load_offsets(pids) -> np.ndarray:
    """Committed H4 per-frame offsets (logit) aligned to `pids` order."""
    off = pd.read_csv(OFFSETS).set_index("PRODUCT_ID")["offset_logit"]
    missing = [p for p in pids if p not in off.index]
    if missing:
        raise SystemExit(f"offsets missing for {missing} — rerun scripts/f_h4_level.py")
    return np.array([float(off.loc[p]) for p in pids], dtype=np.float64)


def themis_on_grid(transform, shape, ctx_crs_wkt) -> np.ndarray:
    """Reproject the cached THEMIS night-IR raster onto the pilot coarse grid."""
    if not THEMIS.exists():
        raise SystemExit(f"missing {THEMIS}; run scripts/fetch_validation_data.py "
                         "--product themis_night_ir --match-mosaic")
    with rasterio.open(THEMIS) as ds:
        arr = ds.read(1).astype(np.float32)
        src_tf = ds.transform
        src_crs = ds.crs.to_wkt()
        src_nd = ds.nodata
    return vr.reproject_to_grid(
        arr, src_tf, src_crs,
        dst_crs_wkt=ctx_crs_wkt, dst_transform=Affine(*tuple(transform)[:6]),
        dst_shape=shape, resampling="bilinear", src_nodata=src_nd,
    )


def rho(comp, themis):
    """Spearman ρ over co-valid tiles; returns (rho, n)."""
    m = np.isfinite(comp) & np.isfinite(themis)
    if m.sum() < 50:
        return np.nan, int(m.sum())
    return float(spearmanr(comp[m], themis[m]).statistic), int(m.sum())


def main() -> None:
    pids, stack, transform = build_stack()
    shape = stack.shape[1:]
    o = load_offsets(pids)
    print(f"{len(pids)} frames; stack {stack.shape}; offsets |o|max {np.abs(o).max():.3f}",
          flush=True)

    # before (unleveled H1) and after (H4 full offsets, λ*=300) composites
    _, labels, part_b, med_b = h2.score("before", stack, pids, transform)
    leveled = fhl.level_stack(stack, o)
    _, _, part_a, med_a = h2.score("after", leveled, pids, transform)

    with rasterio.open(fpc.CROPS / f"{pids[0]}_ifcrop.tif") as ds:
        ctx_crs = ds.crs.to_wkt()
    themis = themis_on_grid(transform, shape, ctx_crs)
    tv = themis[np.isfinite(themis)]
    print(f"THEMIS on pilot grid: valid {np.isfinite(themis).mean():.0%}, "
          f"range {tv.min():.1f}..{tv.max():.1f} DN, spread(p2-p98) "
          f"{np.percentile(tv,98)-np.percentile(tv,2):.1f} DN "
          "(mosaic stores scaled brightness-temp DN 1..255, not K)", flush=True)

    rows = []
    for comp_name, before, after in (("median", med_b, med_a),
                                     ("partition", part_b, part_a)):
        rb, nb = rho(before, themis)
        ra, na = rho(after, themis)
        rows.append({"composite": comp_name, "rho_before_H1": round(rb, 4),
                     "rho_after_H4": round(ra, 4), "delta_rho": round(ra - rb, 4),
                     "n_before": nb, "n_after": na,
                     "not_degraded": bool(ra - rb >= -GATE_TOL)})
    df = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "f_h4_themis_rho.csv", index=False)

    print("\n=== P3: THEMIS night-IR ρ (leg-1 harness on the leveled pilot) ===")
    print(df.to_string(index=False))
    # decision is on the deploy-style median composite; gate = not degraded (Δρ >= -tol)
    dep = df[df.composite == "median"].iloc[0]
    passed = bool(dep["not_degraded"])
    print(f"\nregional leg-1 reference ρ ≈ +0.07 (26-tile map). pilot gate tol = ±{GATE_TOL}.")
    print(f"VERDICT (median composite): Δρ = {dep['delta_rho']:+.4f} "
          f"({dep['rho_before_H1']:+.4f} → {dep['rho_after_H4']:+.4f}) → "
          f"{'PASS — THEMIS ρ not degraded by H4' if passed else 'FAIL — H4 degrades THEMIS ρ'}")

    _figure(med_b, med_a, themis)


def _figure(med_b, med_a, themis):
    m = np.isfinite(med_b) & np.isfinite(med_a) & np.isfinite(themis)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, comp, t in ((ax[0], med_b[m], "H1 before"), (ax[1], med_a[m], "H4 after")):
        a.hexbin(themis[m], comp, gridsize=40, cmap="viridis", mincnt=1)
        r = spearmanr(themis[m], comp).statistic
        a.set_title(f"{t}: Spearman ρ = {r:+.3f}", fontsize=10)
        a.set_xlabel("THEMIS night-IR (scaled DN)")
        a.set_ylabel("P(boulder-rich)")
    fig.suptitle("P3 — THEMIS night-IR vs pilot P(rich), median composite (E8_N44)")
    fig.tight_layout()
    out = FIG / "f_h4_themis_rho.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
