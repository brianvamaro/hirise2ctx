"""H4 leg-B skill instrument (PLAN_H4_Leveling §3.1) — does applying H4's per-frame
leveling offsets degrade POOLED skill on the real LOIO predictions?

Per-image AUC is provably ~blind to H4 (an additive per-frame logit offset can't change
within-frame ranking), so the §3.1 instruments are POOLED metrics that DO see cross-frame
level changes: pooled pr_auc@1e-2 and precision@5% on the leg-B common-cohort LOIO
predictions (no presence AUC, per project rule). Per-image median AUC is reported as a
sanity row (expected ≈ unchanged — and exactly unchanged under an obs-level shift).

Pipeline:
  1. Solve per-frame offsets on the 28 multi-crop TRAINING-obs overlap graph (same solver
     as the pilot; bigger graph). Per-frame predictions come from the H1 head run on the
     cached per-frame embeddings `reports/f_leg_b/h2_frame_emb/{obs}__{pid}.npz`.
  2. Map each leg-B obs -> a representative offset = mean of its source frames' offsets
     (EXACT for single-frame obs; an approximation for composite obs, whose store window
     is last-write-wins over frames — the deploy-faithful per-frame-inference LOIO is a
     build-scale rebuild, deferred).
  3. Apply the obs offset in logit space to the F-store LOIO predictions; recompute pooled
     metrics for baseline / H1 (F raw) / H1+H4 (F leveled).

Run (laptop GPU; per-frame embeddings are cached so this is seconds):
  conda run --no-capture-output -n geospatial python -u scripts/f_h4_legb.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.f_h2_nuisance as h2n
import scripts.f_h4_level as h4
from src.modeling.mlp_head import DeployableHead

FIG = REPO / "reports" / "figures"
EMB_CACHE = REPO / "reports" / "f_leg_b" / "h2_frame_emb"
BASELINE = "fang_embeddings"
F_STORE = "fang_embeddings_f_minnaert_center"
MIN_OVERLAP = 20
GATE = -0.02


def frame_logits(head) -> dict:
    """{obs_id: {pid: {ti, tj, logit(valid tiles only, NaN elsewhere)}}} from the H1 head."""
    frames = h2n.obs_frames()
    out = {}
    for obs_id, pids in frames.items():
        rec = {}
        for pid in pids:
            cache = EMB_CACHE / f"{obs_id}__{pid}.npz"
            if not cache.exists():
                continue
            z = np.load(cache)
            valid = z["valid"].astype(bool)
            lg = np.full(z["ti"].shape, np.nan, dtype=np.float64)
            if valid.any():
                lg[valid] = h4._logit(head.predict(z["gem"][valid]))
            rec[pid] = {"ti": z["ti"], "tj": z["tj"], "logit": lg}
        if len(rec) >= 2:
            out[obs_id] = rec
    return out


def build_edges(fl, node_idx):
    """Co-located frame-pair edges (i, j, δ̄=mean(ℓ_j−ℓ_i), W) over the global frame graph."""
    edges = []
    for obs_id, rec in fl.items():
        pids = list(rec)
        for a in range(len(pids)):
            for b in range(a + 1, len(pids)):
                ra, rb = rec[pids[a]], rec[pids[b]]
                key = {(int(t), int(u)): i for i, (t, u) in enumerate(zip(ra["ti"], ra["tj"]))}
                ia, ib = [], []
                for k, (t, u) in enumerate(zip(rb["ti"], rb["tj"])):
                    h = key.get((int(t), int(u)))
                    if h is not None:
                        ia.append(h)
                        ib.append(k)
                if not ia:
                    continue
                la, lb = ra["logit"][ia], rb["logit"][ib]
                m = np.isfinite(la) & np.isfinite(lb)
                if m.sum() < MIN_OVERLAP:
                    continue
                d = float((lb[m] - la[m]).mean())
                edges.append((node_idx[pids[a]], node_idx[pids[b]], d, float(m.sum())))
    return edges


def n_components(edges, n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = set()
    for i, j, *_ in edges:
        seen.add(i)
        seen.add(j)
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(x) for x in seen}), len(seen)


def pooled_metrics(df):
    """pooled pr_auc@1e-2, precision@5%, median per-image AUC on (obs_id, y, p)."""
    y, p = df["y"].to_numpy(), df["p"].to_numpy()
    pr = float(average_precision_score(y, p))
    k = max(1, int(round(0.05 * len(p))))
    top = np.argsort(-p)[:k]
    prec5 = float(y[top].mean())
    aucs = [roc_auc_score(g["y"], g["p"]) for _, g in df.groupby("obs_id")
            if g["y"].nunique() == 2]
    return {"pooled_pr_auc": round(pr, 4), "precision@5%": round(prec5, 4),
            "median_img_auc": round(float(np.median(aucs)), 4), "n_img": len(aucs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="models/deployable_f_center/86c51a5dca220f63")
    args = ap.parse_args()

    head = DeployableHead.load(Path(args.head))
    fl = frame_logits(head)
    pids_all = sorted({pid for rec in fl.values() for pid in rec})
    node_idx = {p: i for i, p in enumerate(pids_all)}
    n = len(pids_all)
    edges = build_edges(fl, node_idx)
    ncomp, nnode = n_components(edges, n)
    print(f"training-obs frame graph: {n} frames, {len(edges)} co-located edges, "
          f"{ncomp} connected component(s) over {nnode} frames with >=1 edge", flush=True)

    o = h4.solve_offsets(edges, lam=300.0, n=n)   # same λ* as the pilot
    offset = {p: float(o[node_idx[p]]) for p in pids_all}

    # obs -> representative offset (mean of its graph-frame offsets; single-frame = exact)
    obs_frames_all = h2n.obs_frames()
    obs_off = {}
    for obs_id, pids in obs_frames_all.items():
        vals = [offset[p] for p in pids if p in offset]
        if vals:
            obs_off[obs_id] = float(np.mean(vals))
    print(f"{len(obs_off)} obs receive an H4 offset "
          f"(range {min(obs_off.values()):+.3f}..{max(obs_off.values()):+.3f})", flush=True)

    preds = pd.read_csv(FIG / "f_leg_b_loio_preds_minnaert_center.csv")
    base = preds[preds.store == BASELINE].copy()
    fh1 = preds[preds.store == F_STORE].copy()
    fh4 = fh1.copy()
    shift = fh4["obs_id"].map(obs_off).fillna(0.0).to_numpy()
    fh4["p"] = h4._sigmoid(h4._logit(fh4["p"].to_numpy()) + shift)
    n_shifted = int((shift != 0).sum())
    print(f"applied offsets to {n_shifted}/{len(fh4)} F-store tile predictions "
          f"({fh4['obs_id'].map(lambda o: o in obs_off).sum()} rows in offset-bearing obs)",
          flush=True)

    rows = []
    for label, d in [("baseline (mosaic)", base), ("H1 (F, unleveled)", fh1),
                     ("H1+H4 (F, leveled)", fh4)]:
        m = pooled_metrics(d)
        m["pipeline"] = label
        rows.append(m)
    tab = pd.DataFrame(rows)[["pipeline", "pooled_pr_auc", "precision@5%",
                              "median_img_auc", "n_img"]]
    FIG.mkdir(parents=True, exist_ok=True)
    tab.to_csv(FIG / "f_h4_legb_summary.csv", index=False)
    pd.DataFrame({"obs_id": list(obs_off), "offset_logit": np.round(list(obs_off.values()), 4)}
                 ).to_csv(FIG / "f_h4_legb_offsets.csv", index=False)

    b = rows[0]
    print("\n=== H4 leg-B pooled skill (no presence AUC; fa>1e-2) ===")
    print(tab.to_string(index=False))
    d_pr = rows[2]["pooled_pr_auc"] - rows[1]["pooled_pr_auc"]
    d_pr_base = rows[2]["pooled_pr_auc"] - b["pooled_pr_auc"]
    d_img = rows[2]["median_img_auc"] - rows[1]["median_img_auc"]
    print(f"\nΔ pooled PR-AUC  (H1+H4 − H1)       = {d_pr:+.4f}")
    print(f"Δ pooled PR-AUC  (H1+H4 − baseline) = {d_pr_base:+.4f}   (gate ≥ {GATE:+.2f})")
    print(f"Δ per-image AUC  (H1+H4 − H1)       = {d_img:+.4f}   "
          f"(sanity: obs-level shift ⇒ within-image ranking unchanged, expect ≈0)")
    print("\nGATE (H4 does not degrade skill vs H1):",
          "PASS" if d_pr >= GATE else "FAIL")


if __name__ == "__main__":
    main()
