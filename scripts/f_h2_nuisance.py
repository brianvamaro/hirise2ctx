"""H2 (PLAN_StripingArtifact PHASE 2) — build the frame-nuisance embedding subspace.

Two frames imaging the SAME ground should embed to the SAME point; any difference is,
by construction, frame-nuisance (radiometry/illumination/epoch the geology can't explain).
So: embed each source frame of the multi-crop TRAINING obs SEPARATELY (minnaert_center,
the H1 mapping), take co-located tile embedding differences across every within-obs frame
pair, and PCA them. The top-k principal directions are the nuisance basis N.

Basis source = the 28 multi-crop training obs, NOT the 7 E8_N44 pilot frames the η² test
scores — so N is learned independently of the artifact-reduction test set (no circularity).

`DeployableHead(nuisance_basis=N[:, :k])` then removes span(N) from every embedding before
the scaler, identically at train and deploy (travels via save/load) — H2's projection.

Output: reports/f_leg_b/h2_nuisance_basis.npz
  basis   (768, 128)  orthonormal columns, descending eigenvalue
  eigvals (128,)       second-moment eigenvalues (variance captured per direction)
  n_diffs, n_pairs, obs_ids, k_minnaert, stretch_lo, stretch_hi

Per-frame embeddings cache under reports/f_leg_b/h2_frame_emb/ so reruns are instant.

Run (laptop GPU, ~10-20 min for the per-frame embed pass, then seconds):
  conda run --no-capture-output -n geospatial python -u scripts/f_h2_nuisance.py
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

from src.fm_embeddings import FangEmbedder
import scripts.f_leg_b_embed as fle

MAPPING = "minnaert_center"
EMB_CACHE = REPO / "reports" / "f_leg_b" / "h2_frame_emb"
OUT = REPO / "reports" / "f_leg_b" / "h2_nuisance_basis.npz"
N_KEEP = 128          # eigenvectors persisted (k-sweep slices this)


def obs_frames() -> dict[str, list[str]]:
    """{obs_id: [pid, ...]} for every obs with >= 2 source-frame crops."""
    by_obs: dict[str, list[str]] = {}
    for p in sorted(fle.CROPS_DIR.glob("*_ifcrop.tif")):
        parts = p.name.replace("_ifcrop.tif", "").split("_")
        obs_id = "_".join(parts[:3])
        pid = fle._crop_pid(p, obs_id)
        by_obs.setdefault(obs_id, []).append(pid)
    return {o: pids for o, pids in by_obs.items() if len(pids) >= 2}


def frame_embeddings(obs_id: str, pids: list[str], embedder, ctx) -> dict[str, dict]:
    """Per-frame {pid: {ti, tj, valid, gem}} on the shared S=32 tile grid of obs_id."""
    sc = json.loads((fle.LABELS_DIR / f"{obs_id}.json").read_text(encoding="utf-8"))
    row0, col0 = int(sc["mosaic_row_origin"]), int(sc["mosaic_col_origin"])
    ctx_tif = Path(sc["ctx_window_tif"])
    with rasterio.open(ctx_tif) as ds:
        H, W = ds.height, ds.width

    import pandas as pd
    feats = pd.read_parquet(fle.FEATURES_DIR / f"{obs_id}.parquet")
    f32 = feats[feats["scale_idx"] == 2][["ti", "tj"]].drop_duplicates()
    ti = f32["ti"].to_numpy(np.int64)
    tj = f32["tj"].to_numpy(np.int64)

    out = {}
    for pid in pids:
        cache = EMB_CACHE / f"{obs_id}__{pid}.npz"
        if cache.exists():
            z = np.load(cache)
            out[pid] = {k: z[k] for k in ("ti", "tj", "valid", "gem")}
            continue
        window8 = fle.composite_crops(obs_id, row0, col0, H, W, mapping=MAPPING,
                                      ctx=ctx, only_pid=pid)
        if not window8.any():
            print(f"    {obs_id} {pid}: empty single-frame composite; skip", flush=True)
            continue
        emb, valid = embedder.embed_window(window8, ti, tj, tile_px=fle.TILE_PX,
                                           row0=row0, col0=col0, pool="gem",
                                           batch=fle.BATCH)
        rec = {"ti": ti, "tj": tj, "valid": valid, "gem": emb.astype(np.float32)}
        EMB_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **rec)
        out[pid] = rec
        print(f"    {obs_id} {pid}: {int(valid.sum())}/{len(ti)} valid tiles", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fit-only", action="store_true",
                    help="print mapping constants and exit (no embedding)")
    ap.add_argument("--stretch-pcts", nargs=2, type=float, default=[0.5, 99.5],
                    metavar=("LO", "HI"),
                    help="pooled stretch percentiles — MUST match the store build "
                         "(H1 fang_embeddings_f_minnaert_center used 0.5 99.5)")
    args = ap.parse_args()

    # all crops fit the mapping constants (parity with the store build)
    all_obs = sorted({"_".join(p.name.split("_")[:3])
                      for p in fle.CROPS_DIR.glob("*_ifcrop.tif")})
    ctx = fle.build_mapping_ctx(MAPPING, all_obs, pcts=tuple(args.stretch_pcts))
    ctx["scale"] = "log"   # minnaert_center forces log (mirror f_leg_b_embed)
    if args.fit_only:
        return

    frames = obs_frames()
    print(f"{len(frames)} multi-crop obs "
          f"({sum(len(v) for v in frames.values())} frames)", flush=True)
    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)

    # second-moment accumulation over co-located difference vectors (no need to
    # store every diff; d d^T is sign-invariant so overlap ordering is irrelevant)
    M = np.zeros((768, 768), dtype=np.float64)
    n_diffs = n_pairs = 0
    for obs_id, pids in frames.items():
        fe = frame_embeddings(obs_id, pids, embedder, ctx)
        pids = [p for p in pids if p in fe]
        for a in range(len(pids)):
            for b in range(a + 1, len(pids)):
                ra, rb = fe[pids[a]], fe[pids[b]]
                # tiles share the grid -> match on (ti, tj), valid in both
                key_a = {(int(t), int(u)): i
                         for i, (t, u, v) in enumerate(zip(ra["ti"], ra["tj"], ra["valid"])) if v}
                idx_a, idx_b = [], []
                for i, (t, u, v) in enumerate(zip(rb["ti"], rb["tj"], rb["valid"])):
                    if v and (int(t), int(u)) in key_a:
                        idx_a.append(key_a[(int(t), int(u))])
                        idx_b.append(i)
                if len(idx_a) < 20:
                    continue
                d = (ra["gem"][idx_a] - rb["gem"][idx_b]).astype(np.float64)
                M += d.T @ d
                n_diffs += len(idx_a)
                n_pairs += 1
        print(f"  {obs_id}: cumulative {n_pairs} pairs / {n_diffs} co-located diffs",
              flush=True)

    if n_diffs == 0:
        sys.exit("no co-located overlap tiles found — cannot build a basis")
    M /= n_diffs
    evals, evecs = np.linalg.eigh(M)                # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    basis = evecs[:, :N_KEEP].astype(np.float32)
    evals = evals[:N_KEEP].astype(np.float64)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, basis=basis, eigvals=evals, n_diffs=n_diffs, n_pairs=n_pairs,
             obs_ids=np.array(sorted(frames)), k_minnaert=ctx.get("k", np.nan),
             stretch_lo=ctx["lo"], stretch_hi=ctx["hi"])

    total = float(np.trace(M))
    print(f"\nnuisance basis: {n_pairs} pairs, {n_diffs} co-located diffs, "
          f"total diff variance {total:.3f}")
    for k in (4, 16, 64):
        frac = float(evals[:k].sum()) / total
        print(f"  top-{k:2d} directions capture {frac:6.1%} of between-frame "
              f"embedding-difference variance")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
