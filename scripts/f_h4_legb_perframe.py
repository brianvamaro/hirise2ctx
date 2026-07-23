"""Deploy-faithful per-frame cross-frame skill probe (converts PLAN_FBuild §5 gate #5 to pre-spend).

The 2026-07-15 adversarial review flagged that leg-B's -0.0104 skill cost is a WITHIN-OBS lower
bound: it applied a single OBS-LEVEL mean offset to the pre-composited store prediction, never
exercising the build's actual Stage-D path (per-frame inference -> per-frame H4 offset -> MEAN OF
LEVELED LOGITS composite). Gate #5 does the faithful thing but only DURING the build. This runs it
pre-spend on the cached per-frame logits (`reports/f_leg_b/h2_frame_emb/`) + per-tile labels
(`dataset_v2/labels/{obs}.parquet`, fractional_area>1e-2 at scale_idx==2), reusing the committed
leg-B per-frame offset solve (f_h4_legb machinery, λ*=300).

For each multi-frame training obs it builds three composites over the SAME tiles/head/offsets:
  p_unlev : sigmoid(mean_f logit_f)                    -- build path, no leveling
  p_lev   : sigmoid(mean_f (logit_f + offset_f))       -- build path, H4 leveled  (PLAN_FBuild §5)
  p_legb  : sigmoid(logit(p_unlev) + mean_f offset_f)  -- leg-B's obs-level mean-shift approximation
Then pools pr_auc@1e-2 / precision@5% (no presence AUC). The two decision quantities are both
robust to the head being in-sample here (same head cancels in the differences):
  Δ_deploy = pr_auc(lev) - pr_auc(unlev)   -- the TRUE per-frame H4 skill cost on the build composite
  approx_err = pr_auc(lev) - pr_auc(legb)  -- how badly leg-B's obs-level shift misrepresented it
Gate (mirrors §5 / leg-B): Δ_deploy >= -0.02 AND |approx_err| small ⇒ leg-B's -0.0104 was faithful,
gate #5 de-risked before any ISIS compute.

Run (laptop, seconds — per-frame embeddings + fixed head, no re-embedding):
  conda run --no-capture-output -n geospatial python -u scripts/f_h4_legb_perframe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.f_h4_legb as legb
import scripts.f_h4_level as h4
from src.modeling.mlp_head import DeployableHead

FIG = REPO / "reports" / "figures"
LABELS_DIR = REPO / "dataset_v2" / "labels"
HEAD = REPO / "models" / "deployable_f_center" / "86c51a5dca220f63"
GATE = -0.02
FA_THRESH = 1e-2


def tile_labels(obs_id: str) -> dict:
    """{(ti,tj): y} with y = fractional_area > 1e-2 at scale_idx==2."""
    lab = pd.read_parquet(LABELS_DIR / f"{obs_id}.parquet")
    lab = lab[lab["scale_idx"] == 2]
    y = (lab["fractional_area"].to_numpy() > FA_THRESH).astype(int)
    return {(int(t), int(u)): int(v)
            for t, u, v in zip(lab["ti"].to_numpy(), lab["tj"].to_numpy(), y)}


def composites(rec, offset):
    """Per-tile p_unlev / p_lev / p_legb over the obs's frames. rec = {pid:{ti,tj,logit}}."""
    pids = list(rec)
    # union tile index (frames share the obs grid, but align on (ti,tj) to be safe)
    tiles = {}
    for pid in pids:
        for t, u in zip(rec[pid]["ti"], rec[pid]["tj"]):
            tiles.setdefault((int(t), int(u)), len(tiles))
    nt = len(tiles)
    L = np.full((len(pids), nt), np.nan)
    for a, pid in enumerate(pids):
        idx = [tiles[(int(t), int(u))] for t, u in zip(rec[pid]["ti"], rec[pid]["tj"])]
        L[a, idx] = rec[pid]["logit"]
    offs = np.array([offset[p] for p in pids])
    with np.errstate(invalid="ignore"):
        m_unlev = np.nanmean(L, axis=0)
        m_lev = np.nanmean(L + offs[:, None], axis=0)
    mean_off = float(offs.mean())
    keys = list(tiles)
    p_unlev = h4._sigmoid(m_unlev)
    p_lev = h4._sigmoid(m_lev)
    p_legb = h4._sigmoid(m_unlev + mean_off)
    return keys, p_unlev, p_lev, p_legb, offs.std()


def pooled(df, col):
    y, p = df["y"].to_numpy(), df[col].to_numpy()
    pr = float(average_precision_score(y, p))
    k = max(1, int(round(0.05 * len(p))))
    prec5 = float(y[np.argsort(-p)[:k]].mean())
    return pr, prec5


def main() -> None:
    head = DeployableHead.load(HEAD)
    fl = legb.frame_logits(head)                       # {obs:{pid:{ti,tj,logit}}}, >=2 frames
    pids_all = sorted({pid for rec in fl.values() for pid in rec})
    node_idx = {p: i for i, p in enumerate(pids_all)}
    edges = legb.build_edges(fl, node_idx)
    o = h4.solve_offsets(edges, lam=300.0, n=len(pids_all))
    offset = {p: float(o[node_idx[p]]) for p in pids_all}
    print(f"{len(fl)} multi-frame obs, {len(pids_all)} frames, {len(edges)} edges; "
          f"|offset|max {np.abs(o).max():.3f}", flush=True)

    rows, per_obs = [], []
    for obs_id, rec in fl.items():
        labs = tile_labels(obs_id)
        keys, p_u, p_l, p_b, ostd = composites(rec, offset)
        for k, pu, pl, pb in zip(keys, p_u, p_l, p_b):
            y = labs.get(k)
            if y is None or not np.isfinite(pu):
                continue
            rows.append((obs_id, y, pu, pl, pb))
        per_obs.append({"obs_id": obs_id, "n_frames": len(rec),
                        "offset_std": round(float(ostd), 3)})
    df = pd.DataFrame(rows, columns=["obs_id", "y", "p_unlev", "p_lev", "p_legb"])
    print(f"pooled over {len(df)} labeled tiles in {df.obs_id.nunique()} obs "
          f"(pos rate {df.y.mean():.3f})", flush=True)

    res = {}
    for col in ("p_unlev", "p_lev", "p_legb"):
        pr, prec5 = pooled(df, col)
        res[col] = {"pooled_pr_auc": round(pr, 4), "precision@5%": round(prec5, 4)}
    tab = pd.DataFrame(res).T.rename_axis("composite").reset_index()
    FIG.mkdir(parents=True, exist_ok=True)
    tab.to_csv(FIG / "f_h4_legb_perframe.csv", index=False)
    pd.DataFrame(per_obs).sort_values("offset_std", ascending=False).to_csv(
        FIG / "f_h4_legb_perframe_obs.csv", index=False)

    d_deploy = res["p_lev"]["pooled_pr_auc"] - res["p_unlev"]["pooled_pr_auc"]
    d_legb = res["p_legb"]["pooled_pr_auc"] - res["p_unlev"]["pooled_pr_auc"]
    approx_err = res["p_lev"]["pooled_pr_auc"] - res["p_legb"]["pooled_pr_auc"]
    print("\n=== deploy-faithful per-frame composite skill (build Stage-D path; in-sample head) ===")
    print(tab.to_string(index=False))
    print(f"\nΔ_deploy   pr_auc(lev − unlev)  = {d_deploy:+.4f}   (gate ≥ {GATE:+.2f}) — the TRUE "
          "per-frame H4 cost")
    print(f"Δ_legb     pr_auc(legb − unlev) = {d_legb:+.4f}   (leg-B obs-level approximation)")
    print(f"approx_err pr_auc(lev − legb)   = {approx_err:+.4f}   (how far obs-level missed the "
          "per-frame truth)")
    n_div = int((pd.DataFrame(per_obs)["offset_std"] > 0.05).sum())
    print(f"{n_div}/{len(per_obs)} multi-frame obs have divergent frame offsets (std>0.05) — where "
          "per-frame vs obs-level differ")
    ok = (d_deploy >= GATE) and (abs(approx_err) <= 0.02)
    print("\nVERDICT:",
          "PASS — the true per-frame build composite preserves skill under H4 and leg-B's obs-level "
          "approximation was faithful (gate #5 de-risked pre-build)" if ok else
          f"REVIEW — Δ_deploy {d_deploy:+.4f} (gate {GATE}), approx_err {approx_err:+.4f} "
          "(obs-level approximation was NOT faithful — the -0.0104 understated the per-frame cost)")


if __name__ == "__main__":
    main()
