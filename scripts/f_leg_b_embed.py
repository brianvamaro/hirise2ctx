"""F pilot leg B — laptop: embed calibrated-frame crops -> fang_embeddings_f* NPZ stores.

For each training obs_id, composites the I/F crops transferred from Sherlock onto the
CTX mosaic pixel grid, converts I/F -> uint8 under the chosen --mapping, and embeds with
the frozen Fang-ViT at S=32 / P=96 / GeM pooling — the recipe used in fang_embeddings/.

Mappings (leg-A lineage; DECISIONS 2026-07-03b/04/04b):
  perframe  per-composite robust norm: median -> 125 DN, IQR -> 27.7 DN
            -> store fang_embeddings_f          (LOIO gate FAIL −0.0499, dim scenes collapse)
  global    ONE fixed affine for all scenes: pooled p2–p98 over all crops -> 1..255
            -> store fang_embeddings_f_global   (control: "calibrated frames need no norm")
  minnaert  divide each crop by cos^k(incidence) first (k fitted from the crops,
            incidence from reports/f_leg_b/frame_incidence.csv), then fixed pooled stretch
            -> store fang_embeddings_f_minnaert (targets the leg-B illumination correlate)
  minnaert_center  H1 (PLAN_StripingArtifact PHASE 2): minnaert ÷cos^k, THEN divide each
            crop by its OWN median so every crop shares a common center, then the fixed
            pooled LOG stretch (contrast scale fitted on the centered pool). Kills the
            residual per-frame level term (F02-class anomalous frames) that survives
            minnaert -> store fang_embeddings_f_minnaert_center

Output matches the existing embedding format: {store}/{obs}_P96.npz with arrays
(ti, tj, valid, gem).  The LOIO gate script (f_leg_b_loio.py --f-store ...) reads a store
and compares skill against the baseline fang_embeddings/ store.

Run (laptop GPU, after transferring obs_crops from Sherlock):
  # expected layout: reports/f_leg_b/obs_crops/{obs_id}_{pid}_ifcrop.tif
  conda run --no-capture-output -n geospatial python -u scripts/f_leg_b_embed.py --mapping minnaert
  conda run --no-capture-output -n geospatial python -u scripts/f_leg_b_embed.py --mapping global --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from src.fm_embeddings import FangEmbedder
from src.striping import A1_REF_MEDIAN, A1_REF_IQR

CROPS_DIR = REPO / "reports" / "f_leg_b" / "obs_crops"   # switched by --crops-dir
LABELS_DIR = REPO / "dataset_v2" / "labels"
FEATURES_DIR = REPO / "dataset_v2" / "features"
INCIDENCE_CSV = REPO / "reports" / "f_leg_b" / "frame_incidence.csv"
STORES = {"perframe": "fang_embeddings_f",
          "global": "fang_embeddings_f_global",
          "minnaert": "fang_embeddings_f_minnaert",
          "minnaert_center": "fang_embeddings_f_minnaert_center"}
TILE_PX = 32      # S=32; context patch = 3*32 = 96 px
BATCH = 96
PX_M = 5.0        # native CTX resolution


# ------------------------------------------------------------------ normalization

def to_uint8_perframe(arr: np.ndarray) -> np.ndarray:
    """Per-frame robust normalization: I/F median -> A1_REF_MEDIAN, IQR -> A1_REF_IQR."""
    fin = np.isfinite(arr)
    if fin.sum() < 50:
        return np.zeros(arr.shape, dtype=np.uint8)
    v = arr[fin]
    med = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(max(q75 - q25, 1e-6))
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[fin] = np.clip(
        (v - med) / iqr * A1_REF_IQR + A1_REF_MEDIAN, 1, 255
    ).astype(np.uint8)
    return out


def to_uint8_affine(arr: np.ndarray, lo: float, hi: float,
                    log: bool = False) -> np.ndarray:
    """Fixed stretch: I/F lo..hi -> 1..255 (0 reserved for nodata).

    log=True stretches ln(I/F) instead — texture amplitude (multiplicative surface
    contrast) then maps to a level-independent number of DN, so dim scenes are not
    pinned against the floor with their texture quantized away.
    """
    fin = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    v = arr[fin]
    if log:
        v = np.log(np.maximum(v, 1e-6))
        lo, hi = np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6))
    out[fin] = np.clip((v - lo) / max(hi - lo, 1e-9) * 254 + 1, 1, 255
                       ).astype(np.uint8)
    return out


def _crop_pid(crop_path: Path, obs_id: str) -> str:
    return crop_path.name[len(obs_id) + 1: -len("_ifcrop.tif")]


def _crop_values(obs_ids: list[str]):
    """Yield (pid, decimated finite positive I/F values) for every crop, one pass."""
    for obs_id in obs_ids:
        for p in sorted(CROPS_DIR.glob(f"{obs_id}_*_ifcrop.tif")):
            with rasterio.open(p) as src:
                a = src.read(1, out_shape=(max(1, src.height // 8),
                                           max(1, src.width // 8))).astype(np.float32)
            v = a[np.isfinite(a) & (a > 0)]
            if v.size >= 100:
                yield _crop_pid(p, obs_id), v


def build_mapping_ctx(mapping: str, obs_ids: list[str],
                      pcts: tuple[float, float] = (2.0, 98.0)) -> dict:
    """Fit the fixed constants a mapping needs, from the crops themselves.

    global   -> {div: 1 per frame, lo, hi}: pooled p{lo}–p{hi} of raw I/F over all crops
    minnaert -> {k, div: cos^k(i) per frame, lo, hi}: k from log(frame median) vs
                log(cos i), then pooled p{lo}–p{hi} of the DIVIDED values
    perframe -> {} (no constants)
    """
    if mapping == "perframe":
        return {}

    ctx: dict = {"div": {}}
    center = mapping == "minnaert_center"
    ctx["center"] = center
    if mapping in ("minnaert", "minnaert_center"):
        import pandas as pd
        inc = pd.read_csv(INCIDENCE_CSV)
        cos_i = {r.PRODUCT_ID: float(np.cos(np.radians(r.incidence)))
                 for r in inc.itertuples()}
        # pass 1: per-frame median I/F (median over its crop medians) -> fit k
        frame_med: dict[str, list[float]] = {}
        for pid, v in _crop_values(obs_ids):
            frame_med.setdefault(pid, []).append(float(np.median(v)))
        pids = sorted(frame_med)
        missing = [p for p in pids if p not in cos_i]
        if missing:
            raise KeyError(f"no incidence in {INCIDENCE_CSV} for: {missing}")
        med = {p: float(np.median(frame_med[p])) for p in pids}
        k = float(np.polyfit([np.log(cos_i[p]) for p in pids],
                             [np.log(med[p]) for p in pids], 1)[0])
        ctx["k"] = k
        ctx["div"] = {p: cos_i[p] ** k for p in pids}
        print(f"minnaert k = {k:.3f} (fit from {len(pids)} frames, "
              f"incidence via {INCIDENCE_CSV.name})", flush=True)

    # sampling pass with final divisors (global: divisor 1) -> pooled p2–p98
    rng = np.random.default_rng(0)
    samples, n_crops = [], 0
    for pid, v in _crop_values(obs_ids):
        v = v / ctx["div"].get(pid, 1.0)
        if center:                       # per-crop log-median centering (H1)
            v = v / float(np.median(v))
        samples.append(rng.choice(v, size=min(v.size, 200_000), replace=False))
        n_crops += 1
    lo, hi = np.percentile(np.concatenate(samples), list(pcts))
    ctx["lo"], ctx["hi"] = float(lo), float(hi)
    print(f"{mapping} stretch: I/F {ctx['lo']:.4f}..{ctx['hi']:.4f} "
          f"(pooled p{pcts[0]:g}–p{pcts[1]:g}, {n_crops} crops)", flush=True)
    return ctx


# ------------------------------------------------------------------ composite

def composite_crops(obs_id: str, row0: int, col0: int, H: int, W: int,
                    mapping: str = "perframe", ctx: dict | None = None,
                    only_pid: str | None = None) -> np.ndarray:
    """Composite all I/F crops for obs_id onto the mosaic pixel grid (H×W uint8).

    Where multiple frames overlap, the last one written wins (crops are sorted by
    filename so the order is deterministic).  minnaert divides each crop by its
    frame's cos^k(i) BEFORE compositing; the final I/F->uint8 conversion is
    per-composite robust (perframe) or the fixed ctx[lo..hi] affine (global/minnaert).

    ``only_pid`` restricts the composite to a SINGLE source frame (H2 nuisance-basis
    build: per-frame embeddings for co-located overlap pairs). Every pixel then gets
    that one frame's per-crop normalization — identical to how it would be treated in
    the full composite where it happened to win — so the per-frame embeddings match
    the composite embeddings the projection is later applied to.
    """
    ctx = ctx or {}
    # NaN canvas: uncovered pixels must stay non-finite so they are EXCLUDED from
    # the median/IQR normalization stats (zeros would pollute them).
    canvas = np.full((H, W), np.nan, dtype=np.float32)

    crops = sorted(CROPS_DIR.glob(f"{obs_id}_*_ifcrop.tif"))
    if only_pid is not None:
        crops = [p for p in crops if _crop_pid(p, obs_id) == only_pid]
    if not crops:
        return np.zeros((H, W), dtype=np.uint8)

    # All crops of one obs_id were extracted onto the identical grid anchored at
    # the obs bounds (f_leg_b_extract.py), so the first crop's transform IS the
    # destination grid; reprojection is a same-grid resample for alignment safety.
    dst_transform = None

    for crop_path in crops:
        with rasterio.open(crop_path) as src:
            arr = src.read(1).astype(np.float32)
            src_crs = src.crs
            src_transform = src.transform
            if dst_transform is None:
                dst_transform = src_transform

        fin = np.isfinite(arr) & (arr > 0)
        if fin.sum() < 50:
            continue
        arr[~fin] = np.nan
        div = ctx.get("div", {}).get(_crop_pid(crop_path, obs_id), 1.0)
        if div != 1.0:
            arr = arr / div
        if ctx.get("center"):            # H1: divide by this crop's own median so
            arr = arr / float(np.nanmedian(arr))   # every crop shares a common center

        dst_if = np.full((H, W), np.nan, dtype=np.float32)
        reproject(source=arr, destination=dst_if,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=src_crs,
                  src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)

        new = np.isfinite(dst_if)
        canvas[new] = dst_if[new]

    # I/F -> uint8 (uncovered NaN pixels -> uint8 0)
    if not np.isfinite(canvas).any():
        return np.zeros((H, W), dtype=np.uint8)
    if mapping == "perframe":
        return to_uint8_perframe(canvas)
    return to_uint8_affine(canvas, ctx["lo"], ctx["hi"],
                           log=ctx.get("scale") == "log")


# ------------------------------------------------------------------ embed one image

def embed_one(obs_id: str, embedder: FangEmbedder, out_dir: Path,
              mapping: str, ctx: dict) -> bool:
    """Embed a single training image from its I/F crops.  Returns True on success."""
    out_path = out_dir / f"{obs_id}_P96.npz"
    if out_path.exists():
        print(f"  {obs_id}: cached", flush=True)
        return True

    sidecar_path = LABELS_DIR / f"{obs_id}.json"
    if not sidecar_path.exists():
        print(f"  {obs_id}: no sidecar JSON; skipping", flush=True)
        return False

    sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    row0 = int(sc["mosaic_row_origin"])
    col0 = int(sc["mosaic_col_origin"])

    # Get window dimensions from the existing ctx_window_tif
    ctx_tif = Path(sc["ctx_window_tif"])
    if not ctx_tif.exists():
        print(f"  {obs_id}: ctx_window_tif missing; skipping", flush=True)
        return False
    with rasterio.open(ctx_tif) as ds:
        H, W = ds.height, ds.width

    # Build composite uint8 window
    window8 = composite_crops(obs_id, row0, col0, H, W, mapping=mapping, ctx=ctx)
    if not window8.any():
        print(f"  {obs_id}: all-zero composite (no crops found?); skipping", flush=True)
        return False

    # Get tile grid from features parquet
    feat_path = FEATURES_DIR / f"{obs_id}.parquet"
    if not feat_path.exists():
        print(f"  {obs_id}: no features parquet; skipping", flush=True)
        return False

    import pandas as pd
    feats = pd.read_parquet(feat_path)
    # Filter to scale_idx=2 (S=32) and unique (ti, tj)
    f32 = feats[feats["scale_idx"] == 2][["ti", "tj"]].drop_duplicates()
    ti = f32["ti"].to_numpy(np.int64)
    tj = f32["tj"].to_numpy(np.int64)

    emb, valid = embedder.embed_window(
        window8, ti, tj, tile_px=TILE_PX, row0=row0, col0=col0, pool="gem",
        batch=BATCH
    )

    n_valid = int(valid.sum())
    print(f"  {obs_id}: {len(ti)} tiles, {n_valid} valid "
          f"({n_valid / max(len(ti), 1):.0%})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ti=ti, tj=tj, valid=valid, gem=emb.astype(np.float32))
    return True


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", choices=sorted(STORES), default="perframe",
                    help="I/F->uint8 mapping (selects the output store)")
    ap.add_argument("--smoke", action="store_true", help="embed only the first 2 obs_ids")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--obs", nargs="+", help="embed specific obs_ids only")
    ap.add_argument("--fit-only", action="store_true",
                    help="fit + print the mapping constants (CPU) and exit — no embedding")
    ap.add_argument("--stretch-pcts", nargs=2, type=float, default=[2.0, 98.0],
                    metavar=("LO", "HI"),
                    help="pooled stretch percentiles (global/minnaert; default 2 98)")
    ap.add_argument("--store-suffix", default="",
                    help="append to the store name (e.g. _w for a stretch variant)")
    ap.add_argument("--crops-dir", default=None,
                    help="crops dir under reports/f_leg_b/ (e.g. obs_crops_cubic)")
    ap.add_argument("--stretch-scale", choices=("linear", "log"), default="linear",
                    help="fixed-stretch domain (global/minnaert); log = level-"
                         "independent texture DN, unpins dim scenes from the floor")
    args = ap.parse_args()
    if args.crops_dir:
        global CROPS_DIR
        CROPS_DIR = REPO / "reports" / "f_leg_b" / args.crops_dir
    out_dir = REPO / "dataset_v2" / (STORES[args.mapping] + args.store_suffix)

    crops = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1]
                    for p in CROPS_DIR.glob("*_ifcrop.tif")})
    if not crops:
        print(f"No I/F crops found in {CROPS_DIR}\n"
              "Transfer obs_crops from Sherlock first:\n"
              "  tar cf obs_crops.tar -C $SCRATCH/hirise2ctx/f_leg_b obs_crops\n"
              "  scp obs_crops.tar laptop:~/hirise2ctx/reports/f_leg_b/\n"
              "  cd reports/f_leg_b && tar xf obs_crops.tar")
        sys.exit(1)

    # Reconstruct obs_ids from crop filenames: {obs_id}_{pid}_ifcrop.tif
    # obs_id is "ESP_XXXXXX_XXXX" (3 parts joined by _)
    obs_ids: list[str] = []
    seen: set[str] = set()
    for p in sorted(CROPS_DIR.glob("*_ifcrop.tif")):
        parts = p.name.replace("_ifcrop.tif", "").split("_")
        # obs_id = first 3 parts: "ESP", "XXXXXX", "XXXX"
        obs_id = "_".join(parts[:3])
        if obs_id not in seen:
            seen.add(obs_id)
            obs_ids.append(obs_id)

    all_obs = list(obs_ids)   # constants are ALWAYS fitted on the full cohort
    if args.obs:
        obs_ids = [o for o in obs_ids if o in set(args.obs)]
    if args.smoke:
        obs_ids = obs_ids[:2]

    print(f"{len(obs_ids)} obs_ids to embed  (mapping={args.mapping} -> {out_dir.name})",
          flush=True)
    ctx = build_mapping_ctx(args.mapping, all_obs, pcts=tuple(args.stretch_pcts))
    if args.mapping != "perframe":
        # minnaert_center (H1) builds on log-minnaert, the passing mapping -> force log
        ctx["scale"] = "log" if args.mapping == "minnaert_center" else args.stretch_scale
    if args.fit_only:
        print("--fit-only: constants fitted, exiting before embedding.")
        return
    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)

    ok = fail = 0
    for obs_id in obs_ids:
        if embed_one(obs_id, embedder, out_dir, args.mapping, ctx):
            ok += 1
        else:
            fail += 1

    print(f"\nembedded: {ok}  failed/skipped: {fail}")
    print(f"store: {out_dir}")
    print(f"\nnext: conda run -n geospatial python scripts/f_leg_b_loio.py "
          f"--f-store {out_dir.name}")


if __name__ == "__main__":
    main()
