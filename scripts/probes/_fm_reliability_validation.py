"""PLAN_FM §2.7 — validate embedding-novelty as a reliability signal (LOIO).

Question: does per-image embedding novelty rank-correlate with where the FROZEN
RECIPE ITSELF underperforms? (Not the old Tier-1 taxonomy — the FM rescued most
of those; we validate against the frozen recipe's OWN per-image AUC.)

Protocol, LOIO-honest by construction (= the deployment case):
  for each held-out image i:
    fit novelty on the OTHER 37 images' valid tiles
    score image i's valid tiles -> aggregate (median) -> per-image novelty[i]
  Spearman(per-image novelty, per-image frozen-recipe AUC)  -- expect NEGATIVE
  (high novelty -> low AUC). Compare Mahalanobis vs kNN; pick the method to wire.

Per-image AUC is read from the banked frozen predictions (the authoritative
numbers the freeze closed on), not recomputed from a re-fit.

Outputs (no map wiring yet -- validation first, per Brian 2026-06-14):
  reports/figures/27_reliability_validation.png
  reports/reliability/per_image_novelty.csv
  prints the verdict table + chosen method.

CPU-only (numpy / sklearn / scipy). No torch, no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.modeling.loaders import load_fang_store  # noqa: E402
from src.reliability import (  # noqa: E402
    MahalanobisNovelty, KNNNovelty, valid_rows, aggregate_per_image,
)

FROZEN_PRED = REPO / "models" / "fang_probe" / "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2" / "predictions.parquet"
DATASET_DIR = REPO / "dataset_v2"   # the frozen store lives here, not dataset/
FIG_DIR = REPO / "reports" / "figures"
OUT_DIR = REPO / "reports" / "reliability"
PX = 96          # frozen recipe input size (S=32 3x3 context)
POOL = "gem"
MAX_QUERY_PER_IMAGE = 3000   # cap scored tiles per image for the median (stable, fast)
SEED = 0


def frozen_per_image_auc(pred_path: Path) -> pd.Series:
    """Per-image ROC-AUC of the frozen recipe at fa_gt_1e-2 (one class -> dropped)."""
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(pred_path)
    aucs = {}
    for obs, g in df.groupby("obs_id"):
        if g["y_true"].nunique() < 2:
            continue   # all-rich or all-poor image: AUC undefined
        aucs[obs] = roc_auc_score(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
    return pd.Series(aucs, name="auc").sort_values()


def loio_novelty(index: pd.DataFrame, matrix: np.ndarray, method_factory,
                 *, rng: np.random.Generator) -> dict[str, float]:
    """Per-image median novelty under LOIO refit (fit on the other 37 images)."""
    obs_all = index["obs_id"].to_numpy()
    images = sorted(np.unique(obs_all))
    out: dict[str, float] = {}
    for held in images:
        held_mask = obs_all == held
        train_rows = index.loc[~held_mask, "row"].to_numpy()
        scorer = method_factory().fit(matrix[train_rows])

        q_rows = index.loc[held_mask, "row"].to_numpy()
        # restrict to valid tiles, then cap for a stable+fast median
        q_rows = q_rows[valid_rows(matrix[q_rows])]
        if q_rows.size > MAX_QUERY_PER_IMAGE:
            q_rows = rng.choice(q_rows, MAX_QUERY_PER_IMAGE, replace=False)
        scores = scorer.score(matrix[q_rows])
        agg = aggregate_per_image(np.full(q_rows.size, held), scores, how="median")
        out.update(agg)
        print(f"  {held}: novelty={agg.get(held, float('nan')):.4f} "
              f"(n_valid_scored={q_rows.size})", flush=True)
    return out


def main() -> None:
    print("Loading frozen embedding store + per-image AUC ...", flush=True)
    index, matrix = load_fang_store(PX, pool=POOL, dataset_dir=DATASET_DIR)
    auc = frozen_per_image_auc(FROZEN_PRED)
    print(f"  store: {matrix.shape[0]} tiles x {matrix.shape[1]} dim, "
          f"{index['obs_id'].nunique()} images; "
          f"valid tiles {int(valid_rows(matrix).sum())}", flush=True)
    print(f"  per-image AUC: {len(auc)} images with both classes "
          f"(range {auc.min():.3f}-{auc.max():.3f}, median {auc.median():.3f})", flush=True)

    rng = np.random.default_rng(SEED)
    methods = {
        "mahalanobis": lambda: MahalanobisNovelty(n_components=256),
        "knn_cos50": lambda: KNNNovelty(k=50, metric="cosine", max_reference=20000, seed=SEED),
    }

    from scipy.stats import spearmanr

    results = {}
    rows = []
    for name, factory in methods.items():
        print(f"\n=== {name} (LOIO refit per image) ===", flush=True)
        nov = loio_novelty(index, matrix, factory, rng=rng)
        common = [o for o in auc.index if o in nov]
        x = np.array([nov[o] for o in common])      # novelty
        y = auc.loc[common].to_numpy()              # frozen-recipe AUC
        rho, p = spearmanr(x, y)
        # how well does it flag the bottom-5 AUC images? (precision@5 of the
        # highest-novelty 5 vs the lowest-AUC 5)
        order_nov = [common[i] for i in np.argsort(-x)]       # most novel first
        worst_auc = set(auc.loc[common].sort_values().index[:5])
        prec_at5 = len(set(order_nov[:5]) & worst_auc) / 5.0
        results[name] = dict(rho=rho, p=p, prec_at5=prec_at5, novelty=nov, common=common)
        print(f"  Spearman(novelty, AUC) = {rho:+.3f}  (p={p:.4f}); "
              f"flag bottom-5 AUC prec@5={prec_at5:.2f}", flush=True)
        for o in common:
            rows.append(dict(obs_id=o, method=name, novelty=nov[o], auc=float(auc.loc[o])))

    # ---- bank per-image table ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_DIR / "per_image_novelty.csv", index=False)

    # ---- figure: novelty vs AUC, one panel per method ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 5.2))
    if len(methods) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        common = r["common"]
        x = np.array([r["novelty"][o] for o in common])
        y = auc.loc[common].to_numpy()
        ax.scatter(x, y, s=28, alpha=0.8, edgecolor="k", linewidth=0.4)
        # annotate the 4 lowest-AUC images
        for o in auc.loc[common].sort_values().index[:4]:
            ax.annotate(o.replace("ESP_", ""), (r["novelty"][o], float(auc.loc[o])),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.set_xlabel(f"per-image novelty ({name}, LOIO median)")
        ax.set_ylabel("frozen-recipe per-image AUC")
        ax.set_title(f"{name}\nSpearman {r['rho']:+.3f} (p={r['p']:.3f}), "
                     f"bottom-5 prec@5={r['prec_at5']:.2f}")
    fig.suptitle("PLAN_FM §2.7 — embedding novelty vs where the frozen recipe underperforms",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "27_reliability_validation.png", dpi=130)
    print(f"\nWrote {FIG_DIR / '27_reliability_validation.png'}", flush=True)

    # ---- verdict ----
    print("\n================ VERDICT ================", flush=True)
    for name, r in results.items():
        verdict = "VALID trust signal" if (r["rho"] < 0 and r["p"] < 0.05) else "weak/insignificant"
        print(f"  {name:14s} rho={r['rho']:+.3f} p={r['p']:.4f} "
              f"prec@5={r['prec_at5']:.2f}  -> {verdict}", flush=True)
    best = min(results, key=lambda n: results[n]["rho"])   # most negative rho
    print(f"  chosen (most negative rho): {best}", flush=True)


if __name__ == "__main__":
    main()
