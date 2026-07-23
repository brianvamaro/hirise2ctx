"""Leave-one-FRAME-out CV for H4 leveling — the honest generalization instrument the pilot lacked.

The 2026-07-15 adversarial review flagged that PLAN_H4_Leveling §3.2 pre-declared ONLY
leave-one-EDGE-out CV, which on the over-determined 7-frame graph is nearly an in-sample check
(dropping one of 15 edges barely moves the solution). The build's real failure mode is whether a
NEW frame's offset — fit from ITS overlaps with already-solved frames — generalizes. That is
leave-one-FRAME-out (LOFO): defined nowhere in the repo. This runs it on the EXISTING pilot cache
(embeddings from f_h2_eta2; committed H1 head), reusing the committed H4 solver (f_h4_level).

For each frame f:
  1. Solve offsets o_ret on the other 6 frames only (edges NOT touching f), at λ*=300.
  2. PREDICT f's offset from its overlap edges to the retained frames:
       edge (i,f): o_i - o_f ≈ dbar  =>  ô_f = o_i - dbar
       edge (f,j): o_f - o_j ≈ dbar  =>  ô_f = o_j + dbar   (weighted mean over f's edges, weight W)
  3. HELD-OUT |Δp|: median co-located |Δp| on f's own edges using ô_f + the retained offsets
     (never fit on f). Compare to the unleveled baseline (0.0738) and the in-sample/edge-CV 0.035.
  4. GENERALIZATION η²: recompute partition η² with o_lofo = [o_ret, ô_f]; if a held-out frame's
     predicted offset generalizes, η² stays near the full 0.0505.
Also reports the cruder "drop-frame regret" (set o_f to gauge, recompute η²) to cross-check the
review's probe numbers (all-but-P21 0.0857, all-but-B03 0.0804, all-but-F02 0.0802).

Gate (pre-declared here, mirroring the review's ask): out-of-frame median |Δp| materially below
0.0738, AND worst-frame LOFO η² <= ~0.06. If both hold the generalization claim is earned pre-build.

Run (laptop, seconds — embeddings cached):
  conda run --no-capture-output -n geospatial python -u scripts/f_h4_lofo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.f_h2_eta2 as h2
import scripts.f_h4_level as fhl
import scripts.f_pilot_crop as fpc
from src.fm_embeddings import FangEmbedder
from src.modeling.mlp_head import DeployableHead

FIG = REPO / "reports" / "figures"
HEAD = REPO / "models" / "deployable_f_center" / "86c51a5dca220f63"
LAM = 300.0                 # the committed λ* (f_h4_trend_guard.csv)
BASELINE_DP = 0.0738        # unleveled in-sample median |Δp| (DECISIONS 2026-07-09b(H4))


def build_stack():
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


def eta2_partition(stack, o, pids, transform) -> float:
    rows, *_ = h2.score("_", fhl.level_stack(stack, o), pids, transform)
    return float(rows["partition"])


def predict_offset(edges, o_ret, f) -> tuple[float, float, int]:
    """Weighted-mean predicted offset for held-out frame f from its edges to retained frames."""
    num = den = 0.0
    ne = 0
    for i, j, dbar, w, _ in edges:
        if i == f:
            est = o_ret[j] + dbar          # o_f - o_j ≈ dbar
        elif j == f:
            est = o_ret[i] - dbar          # o_i - o_f ≈ dbar
        else:
            continue
        num += w * est
        den += w
        ne += 1
    return (num / den if den else 0.0), den, ne


def main() -> None:
    pids, stack, transform = build_stack()
    n = len(pids)
    edges, logit, _ = fhl.edge_stats(stack)
    o_full = fhl.solve_offsets(edges, LAM, n)
    eta_full = eta2_partition(stack, o_full, pids, transform)
    print(f"{n} frames, {len(edges)} edges; full-solve partition η² {eta_full:.4f} "
          f"(committed 0.0505); |o|max {np.abs(o_full).max():.3f}", flush=True)

    rows = []
    heldout_dps, lofo_etas = [], []
    for f in range(n):
        retained = [e for e in edges if e[0] != f and e[1] != f]
        f_edges = [e for e in edges if e[0] == f or e[1] == f]
        o_ret = fhl.solve_offsets(retained, LAM, n)     # frame f -> ~0 (disconnected in this solve)

        o_hat, wsum, ne = predict_offset(edges, o_ret, f)
        o_lofo = o_ret.copy()
        o_lofo[f] = o_hat
        o_lofo = o_lofo - np.median(o_lofo)             # regauge (η² over labels needs a gauge)

        # held-out |Δp| on f's OWN edges (offsets never fit on f)
        dps = [fhl.median_dp(logit, o_lofo, i, j, m) for i, j, _, _, m in f_edges]
        ho_dp = float(np.median(dps)) if dps else np.nan
        heldout_dps.append(ho_dp)

        eta_lofo = eta2_partition(stack, o_lofo, pids, transform)
        lofo_etas.append(eta_lofo)

        # cruder "drop-frame regret": f's offset set to gauge (0), rest = full solve
        o_drop = o_full.copy()
        o_drop[f] = 0.0
        o_drop = o_drop - np.median(o_drop)
        eta_drop = eta2_partition(stack, o_drop, pids, transform)

        rows.append({"heldout_frame": pids[f][:3], "n_edges": ne,
                     "offset_full": round(float(o_full[f]), 3),
                     "offset_pred_LOFO": round(float(o_hat), 3),
                     "pred_err": round(float(abs(o_hat - o_full[f])), 3),
                     "heldout_dp": round(ho_dp, 4),
                     "eta2_LOFO": round(eta_lofo, 4),
                     "eta2_dropframe": round(eta_drop, 4)})

    df = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "f_h4_lofo.csv", index=False)
    print("\n=== LEAVE-ONE-FRAME-OUT CV (held-out frame's offset predicted from its overlaps) ===")
    print(df.to_string(index=False))

    med_ho = float(np.median(heldout_dps))
    worst_eta = float(np.max(lofo_etas))
    med_pred_err = float(df["pred_err"].median())
    print(f"\nbaselines: unleveled |Δp| {BASELINE_DP:.4f}; in-sample/edge-CV |Δp| ~0.035; "
          f"full η² {eta_full:.4f} (bar ~0.05, guard ~0.06)")
    print(f"LOFO: median held-out |Δp| {med_ho:.4f}  | worst-frame η² {worst_eta:.4f}  "
          f"| median |predicted−full offset| {med_pred_err:.3f}")
    dp_ok = med_ho < 0.5 * BASELINE_DP + 0.5 * 0.035     # "materially below 0.0738"
    eta_ok = worst_eta <= 0.06
    print("VERDICT:",
          "PASS — out-of-frame offsets generalize (held-out |Δp| materially below unleveled; "
          "worst LOFO η² within guard)" if (dp_ok and eta_ok) else
          f"MARGINAL/FAIL — dp_ok={dp_ok} (|Δp| {med_ho:.4f} vs 0.0738), "
          f"eta_ok={eta_ok} (worst {worst_eta:.4f} vs 0.06 guard)")


if __name__ == "__main__":
    main()
