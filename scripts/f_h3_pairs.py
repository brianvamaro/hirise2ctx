"""H3 (PLAN_StripingArtifact PHASE 2) — extract co-located overlap embedding PAIRS.

H2 collected the co-located difference vectors `d = e_i − e_j` and PCA'd them; the
linear-subspace lever failed (2026-07-09). H3 keeps the SAME co-located overlap
tiles but hands the head the raw pairs `(e_i, e_j)` so it can be trained to predict
the same P(rich) on both — the consistency penalty `λ·MSE(sigmoid(net(e_i)),
sigmoid(net(e_j)))` (in `MLPClassifierHead.fit`). Same ground, two frames ⇒ any
prediction difference is artifact by construction (no geology assumption).

Source = the 28 multi-crop TRAINING obs under the H1 `minnaert_center` mapping,
NOT the 7 E8_N44 pilot frames the η² test scores → pairs are independent of the
artifact-reduction test set (no circularity), exactly as H2's basis was.

Reuses `scripts/f_h2_nuisance.py` wholesale for the per-frame embeddings (cached in
reports/f_leg_b/h2_frame_emb/) and the shared-grid co-located matching — only the
reduction differs (store pairs vs accumulate dᵀd).

Output: reports/f_leg_b/h3_consistency_pairs.npz
  ea (N, 768) float32   frame-i embedding of a co-located tile
  eb (N, 768) float32   frame-j embedding of the SAME tile
  n_pairs_obs, n_diffs_total, cap, obs_ids, k_minnaert, stretch_lo, stretch_hi

Run (laptop GPU; instant if h2_frame_emb is already cached):
  conda run --no-capture-output -n geospatial python -u scripts/f_h3_pairs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np

from src.fm_embeddings import FangEmbedder
import scripts.f_h2_nuisance as h2
import scripts.f_leg_b_embed as fle

OUT = REPO / "reports" / "f_leg_b" / "h3_consistency_pairs.npz"
MIN_OVERLAP = 20   # min co-located tiles for a frame pair to contribute (matches H2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cap", type=int, default=40000,
                    help="max pairs kept (random subsample; a regularizer needs "
                         "coverage, not the full ~175k). 0 = keep all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stretch-pcts", nargs=2, type=float, default=[0.5, 99.5],
                    metavar=("LO", "HI"),
                    help="pooled stretch percentiles — MUST match the H1 store build")
    args = ap.parse_args()

    all_obs = sorted({"_".join(p.name.split("_")[:3])
                      for p in fle.CROPS_DIR.glob("*_ifcrop.tif")})
    ctx = fle.build_mapping_ctx(h2.MAPPING, all_obs, pcts=tuple(args.stretch_pcts))
    ctx["scale"] = "log"   # minnaert_center forces log (mirror f_leg_b_embed)

    frames = h2.obs_frames()
    print(f"{len(frames)} multi-crop obs "
          f"({sum(len(v) for v in frames.values())} frames)", flush=True)
    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)

    ea_parts, eb_parts = [], []
    n_pairs_obs = 0
    for obs_id, pids in frames.items():
        fe = h2.frame_embeddings(obs_id, pids, embedder, ctx)
        pids = [p for p in pids if p in fe]
        for a in range(len(pids)):
            for b in range(a + 1, len(pids)):
                ra, rb = fe[pids[a]], fe[pids[b]]
                key_a = {(int(t), int(u)): i
                         for i, (t, u, v) in enumerate(zip(ra["ti"], ra["tj"], ra["valid"])) if v}
                idx_a, idx_b = [], []
                for i, (t, u, v) in enumerate(zip(rb["ti"], rb["tj"], rb["valid"])):
                    if v and (int(t), int(u)) in key_a:
                        idx_a.append(key_a[(int(t), int(u))])
                        idx_b.append(i)
                if len(idx_a) < MIN_OVERLAP:
                    continue
                ea_parts.append(ra["gem"][idx_a].astype(np.float32))
                eb_parts.append(rb["gem"][idx_b].astype(np.float32))
                n_pairs_obs += 1
        print(f"  {obs_id}: cumulative {n_pairs_obs} frame-pairs / "
              f"{sum(len(e) for e in ea_parts)} co-located tile-pairs", flush=True)

    if not ea_parts:
        sys.exit("no co-located overlap tiles found — cannot build pairs")
    ea = np.concatenate(ea_parts, axis=0)
    eb = np.concatenate(eb_parts, axis=0)
    n_total = ea.shape[0]

    if args.cap and n_total > args.cap:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(n_total, size=args.cap, replace=False)
        ea, eb = ea[sel], eb[sel]
        print(f"subsampled {n_total} -> {args.cap} pairs (seed {args.seed})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, ea=ea, eb=eb, n_pairs_obs=n_pairs_obs, n_diffs_total=n_total,
        cap=args.cap, obs_ids=np.array(sorted(frames)),
        k_minnaert=ctx.get("k", np.nan), stretch_lo=ctx["lo"], stretch_hi=ctx["hi"])

    # sanity: mean per-tile prediction-agnostic embedding disagreement magnitude
    dd = np.linalg.norm(ea - eb, axis=1)
    print(f"\nH3 pairs: {n_pairs_obs} frame-pairs, {n_total} co-located tile-pairs "
          f"(kept {ea.shape[0]})")
    print(f"  ‖e_i − e_j‖ median {np.median(dd):.3f}  mean {dd.mean():.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
