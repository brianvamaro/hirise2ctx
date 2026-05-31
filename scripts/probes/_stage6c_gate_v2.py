"""Stage 6c v2 — push harder on the gate.

Builds on `_stage6c_gate.py` but with:
  - Richer per-image feature set (6 features vs 3)
  - L1-regularised logreg + LightGBM (max_depth=2, heavily regularised) + ensemble
  - Sweep over the "bad image" cutoff threshold
  - Strategy A (top-K-worst) + pooled-global Strategy B re-evaluated for the
    strongest gate.

Goal: see if a stronger gate can satisfy the strict acceptance criterion
(per-fold mean PR-AUC >= 0.65 AND retained-tile frac >= 70% AND lift +0.10).

Outputs:
  - cache/stage6c/v2_predictor_table.parquet
  - cache/stage6c/v2_gate_summary.parquet
  - scripts/probes/_stage6c_gate_v2.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

# Re-use helpers + paths from v1
from _stage6c_gate import (  # type: ignore
    ROOT, FEATURES_DIR, SUMMARY, BASELINE_SCHEME, S64_IDX,
    PER_TILE_PREDICTIONS, MEANINGFUL_THRESHOLD,
    load_per_image_labels,
    pooled_global_pr_auc,
    pooled_global_with_strategy_a,
    pooled_global_with_strategy_b,
    df_to_md,
)

CACHE = ROOT / "cache" / "stage6c"
WRITEUP = ROOT / "scripts" / "probes" / "_stage6c_gate_v2.md"


# ---------------------------------------------------------------------------
# Richer per-image feature table
# ---------------------------------------------------------------------------


FEATURES_V2 = [
    "mean_n_sources",
    "max_n_sources",
    "std_ctx_incidence",        # mean over tiles of per-tile incidence std
    "max_ctx_incidence_std",    # max over tiles of per-tile incidence std
    "mean_dominant_source_fraction",
    "fraction_stitched_tiles",  # fraction of tiles with ctx_n_sources > 1
]


def aggregate_per_image_features_v2(obs_parquet: Path, scale_idx: int = S64_IDX) -> dict:
    df = pd.read_parquet(obs_parquet)
    df = df[df["scale_idx"] == scale_idx].dropna(
        subset=["ctx_n_sources", "ctx_incidence_std", "ctx_dominant_source_fraction"]
    )
    obs_id = df["obs_id"].iloc[0]
    return {
        "obs_id": obs_id,
        "n_tiles_s64": int(len(df)),
        "mean_n_sources": float(df["ctx_n_sources"].mean()),
        "max_n_sources": float(df["ctx_n_sources"].max()),
        "std_ctx_incidence": float(df["ctx_incidence_std"].mean()),
        "max_ctx_incidence_std": float(df["ctx_incidence_std"].max()),
        "mean_dominant_source_fraction": float(df["ctx_dominant_source_fraction"].mean()),
        "fraction_stitched_tiles": float((df["ctx_n_sources"] > 1).mean()),
    }


def build_predictor_table_v2(features_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    return pd.DataFrame([aggregate_per_image_features_v2(f)
                         for f in sorted(features_dir.glob("*.parquet"))])


# ---------------------------------------------------------------------------
# Gate models (LOIO CV)
# ---------------------------------------------------------------------------


@dataclass
class GateCV:
    name: str
    bad_cutoff: float
    y_bin: np.ndarray
    y_cont: np.ndarray
    p_bad: np.ndarray
    roc_auc: float
    spearman_to_pr_auc: float


def _safe_roc(y_bin: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y_bin, p)) if len(np.unique(y_bin)) == 2 else float("nan")


def _loio_iter(n: int):
    return LeaveOneOut().split(np.arange(n))


def cv_logreg(X: np.ndarray, y_bin: np.ndarray, penalty: str = "l2", C: float = 1.0):
    p_bad = np.zeros(len(X))
    for tr, te in _loio_iter(len(X)):
        if len(np.unique(y_bin[tr])) < 2:
            p_bad[te] = float(y_bin[tr].mean())
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(penalty=penalty, C=C, max_iter=5000, solver="liblinear")
        clf.fit(sc.transform(X[tr]), y_bin[tr])
        p_bad[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p_bad


def cv_ridge_then_logistic(X: np.ndarray, y_cont: np.ndarray, cutoff: float):
    pred = np.zeros(len(X))
    for tr, te in _loio_iter(len(X)):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y_cont[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    s = float(pred.std()) or 1.0
    return 1.0 / (1.0 + np.exp(-(cutoff - pred) / (s / 2)))


def cv_lightgbm(X: np.ndarray, y_bin: np.ndarray):
    """LightGBM with hard regularisation appropriate for n=38."""
    p_bad = np.zeros(len(X))
    for tr, te in _loio_iter(len(X)):
        if len(np.unique(y_bin[tr])) < 2:
            p_bad[te] = float(y_bin[tr].mean())
            continue
        m = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=2,
            num_leaves=3,
            min_data_in_leaf=5,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        m.fit(X[tr], y_bin[tr])
        p_bad[te] = m.predict_proba(X[te])[:, 1]
    return p_bad


def fit_gates(table: pd.DataFrame, cutoff: float) -> list[GateCV]:
    X = table[FEATURES_V2].to_numpy()
    y_cont = table["baseline_pr_auc"].to_numpy()
    y_bin = (y_cont < cutoff).astype(int)

    gates: list[GateCV] = []

    for penalty, C, name in [
        ("l2", 1.0, "logreg_l2_C1"),
        ("l1", 0.5, "logreg_l1_C0.5"),
    ]:
        p = cv_logreg(X, y_bin, penalty=penalty, C=C)
        gates.append(GateCV(
            name=name, bad_cutoff=cutoff, y_bin=y_bin, y_cont=y_cont, p_bad=p,
            roc_auc=_safe_roc(y_bin, p),
            spearman_to_pr_auc=float(stats.spearmanr(p, y_cont).correlation),
        ))

    p = cv_ridge_then_logistic(X, y_cont, cutoff)
    gates.append(GateCV(
        name="ridge_then_logistic", bad_cutoff=cutoff, y_bin=y_bin, y_cont=y_cont, p_bad=p,
        roc_auc=_safe_roc(y_bin, p),
        spearman_to_pr_auc=float(stats.spearmanr(p, y_cont).correlation),
    ))

    p = cv_lightgbm(X, y_bin)
    gates.append(GateCV(
        name="lightgbm_d2", bad_cutoff=cutoff, y_bin=y_bin, y_cont=y_cont, p_bad=p,
        roc_auc=_safe_roc(y_bin, p),
        spearman_to_pr_auc=float(stats.spearmanr(p, y_cont).correlation),
    ))

    # Ensemble = mean of the four predictions.
    p = np.mean([g.p_bad for g in gates], axis=0)
    gates.append(GateCV(
        name="ensemble_mean", bad_cutoff=cutoff, y_bin=y_bin, y_cont=y_cont, p_bad=p,
        roc_auc=_safe_roc(y_bin, p),
        spearman_to_pr_auc=float(stats.spearmanr(p, y_cont).correlation),
    ))

    return gates


# ---------------------------------------------------------------------------
# Strategy A — top-K-worst on per-fold mean
# ---------------------------------------------------------------------------


def eval_top_k(table: pd.DataFrame, p_bad: np.ndarray, k_sweep: list[int]) -> pd.DataFrame:
    order = np.argsort(-p_bad)
    total_tiles = int(table["n_tiles"].sum())
    full_pr_auc_mean = float(table["baseline_pr_auc"].mean())
    full_lift_mean = float(table["baseline_norm_lift"].mean())
    rows = []
    for k in k_sweep:
        if k < 0 or k >= len(table):
            continue
        drop = order[:k]
        keep = np.ones(len(table), dtype=bool); keep[drop] = False
        sub = table.loc[keep]
        kept_tiles = int(sub["n_tiles"].sum())
        rows.append({
            "k_dropped": k,
            "n_kept": len(sub),
            "tile_kept_frac": kept_tiles / total_tiles,
            "kept_pr_auc_mean": float(sub["baseline_pr_auc"].mean()),
            "kept_norm_lift_mean": float(sub["baseline_norm_lift"].mean()),
            "delta_pr_auc": float(sub["baseline_pr_auc"].mean()) - full_pr_auc_mean,
            "delta_norm_lift": float(sub["baseline_norm_lift"].mean()) - full_lift_mean,
        })
    return pd.DataFrame(rows)


STRICT = {"kept_pr_auc_mean": 0.65, "tile_kept_frac": 0.70, "delta_norm_lift": 0.10}


def strict_pass_rows(top_k: pd.DataFrame) -> pd.DataFrame:
    return top_k[
        (top_k["kept_pr_auc_mean"] >= STRICT["kept_pr_auc_mean"])
        & (top_k["tile_kept_frac"] >= STRICT["tile_kept_frac"])
        & (top_k["delta_norm_lift"] >= STRICT["delta_norm_lift"])
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("[1/4] Build rich predictor table …")
    preds = build_predictor_table_v2(FEATURES_DIR)
    labels = load_per_image_labels()
    table = preds.merge(labels, on="obs_id", how="inner")
    table.to_parquet(CACHE / "v2_predictor_table.parquet", index=False)
    print(f"  -> {len(table)} rows, {len(FEATURES_V2)} features")

    print("[2/4] Univariate correlations …")
    uni_rows = []
    for f in FEATURES_V2:
        r = stats.spearmanr(table[f], table["baseline_pr_auc"])
        uni_rows.append({"feature": f, "spearman_rho": r.correlation,
                         "p_value": r.pvalue})
    uni = pd.DataFrame(uni_rows).sort_values("p_value")
    print(uni.to_string(index=False))

    print("[3/4] Sweep gate models × bad-image cutoff …")
    cutoff_sweep = [0.45, 0.50, 0.55, 0.60]
    k_sweep = list(range(0, 21))
    all_results = []
    best_per_fold = None  # (gate_name, cutoff, top_k_row dict, gate obj)
    best_pooled_b = None  # (gate_name, cutoff, delta_b, gate obj, table snapshot)

    per_tile = pd.read_parquet(PER_TILE_PREDICTIONS)
    base_pooled = pooled_global_pr_auc(per_tile)

    for cutoff in cutoff_sweep:
        gates = fit_gates(table, cutoff)
        for g in gates:
            tk = eval_top_k(table, g.p_bad, k_sweep)
            passes = strict_pass_rows(tk)

            # Pooled-global Strategy B (down-weight)
            b = pooled_global_with_strategy_b(per_tile, table, g)
            delta_b = b["pr_auc_global"] - base_pooled["pr_auc_global"]

            # Best per-fold row by kept_pr_auc_mean subject to tile_kept_frac>=0.70
            feasible = tk[tk["tile_kept_frac"] >= 0.70]
            best_feasible = (feasible.sort_values("kept_pr_auc_mean", ascending=False).iloc[0]
                             if not feasible.empty else tk.sort_values("kept_pr_auc_mean", ascending=False).iloc[0])

            all_results.append({
                "cutoff": cutoff,
                "gate": g.name,
                "roc_auc_cv": g.roc_auc,
                "spearman_to_pr_auc": g.spearman_to_pr_auc,
                "strict_passes": int(len(passes)),
                "best_feasible_k": int(best_feasible["k_dropped"]),
                "best_feasible_pr_auc_mean": float(best_feasible["kept_pr_auc_mean"]),
                "best_feasible_tile_frac": float(best_feasible["tile_kept_frac"]),
                "best_feasible_lift": float(best_feasible["delta_norm_lift"]),
                "pooled_baseline_pr_auc": base_pooled["pr_auc_global"],
                "pooled_b_pr_auc": b["pr_auc_global"],
                "pooled_b_delta": delta_b,
            })

            # Track best per-fold strict-feasible.
            if not passes.empty:
                cand = (g.name, cutoff, passes.iloc[0].to_dict(), g)
                if best_per_fold is None or cand[2]["kept_pr_auc_mean"] > best_per_fold[2]["kept_pr_auc_mean"]:
                    best_per_fold = cand

            # Track best pooled-B.
            cand_b = (g.name, cutoff, delta_b, g)
            if best_pooled_b is None or cand_b[2] > best_pooled_b[2]:
                best_pooled_b = cand_b

    summary = pd.DataFrame(all_results).sort_values(
        ["strict_passes", "best_feasible_pr_auc_mean"], ascending=[False, False]
    )
    summary.to_parquet(CACHE / "v2_gate_summary.parquet", index=False)
    print("\nTop 10 gate/cutoff combos by strict-pass + best feasible PR-AUC:")
    print(summary.head(10).to_string(index=False))

    print("[4/4] Write markdown writeup …")
    lines = []
    lines.append("# Stage 6c v2 — pushed gate (richer features + LightGBM + ensemble)")
    lines.append("")
    lines.append("Generated by [`_stage6c_gate_v2.py`](./_stage6c_gate_v2.py).")
    lines.append(f"Features used ({len(FEATURES_V2)}): {', '.join(f'`{f}`' for f in FEATURES_V2)}.")
    lines.append("")
    lines.append("## 1. Univariate Spearman (per-image features ↔ baseline PR-AUC, n=38)")
    lines.append("")
    lines.append(df_to_md(uni))
    lines.append("")
    lines.append("## 2. Gate model × bad-image cutoff sweep")
    lines.append("")
    lines.append(f"For each (gate, cutoff), `best_feasible_*` = best top-K-worst exclusion "
                 f"satisfying `tile_kept_frac >= 0.70`. `strict_passes` = number of K values "
                 f"clearing all three strict criteria.")
    lines.append("")
    cols = ["cutoff", "gate", "roc_auc_cv", "strict_passes", "best_feasible_k",
            "best_feasible_pr_auc_mean", "best_feasible_tile_frac",
            "best_feasible_lift", "pooled_b_delta"]
    lines.append(df_to_md(summary[cols].head(20)))
    lines.append("")

    lines.append("## 3. Outcome")
    lines.append("")
    if best_per_fold is not None:
        gn, ct, row, _ = best_per_fold
        lines.append(f"**Strict per-fold acceptance PASSES** with gate `{gn}` at "
                     f"`bad_cutoff = {ct}`, dropping K = {int(row['k_dropped'])} images: "
                     f"PR-AUC mean = {row['kept_pr_auc_mean']:.3f}, "
                     f"tile_kept_frac = {row['tile_kept_frac']:.3f}, "
                     f"Δ norm lift = {row['delta_norm_lift']:+.3f}.")
    else:
        lines.append("**Strict per-fold acceptance: FAIL across all gate × cutoff × K combinations.**")
    lines.append("")
    if best_pooled_b is not None:
        gn, ct, dlt, _ = best_pooled_b
        lines.append(f"**Best pooled-global Strategy B** (down-weighting): gate `{gn}` at "
                     f"`bad_cutoff = {ct}` gives Δ pooled PR-AUC = {dlt:+.4f} "
                     f"(baseline = {base_pooled['pr_auc_global']:.4f}, "
                     f"gated = {base_pooled['pr_auc_global'] + dlt:.4f}).")
    lines.append("")

    WRITEUP.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {WRITEUP}")

    print("\n=== Headline ===")
    if best_per_fold is not None:
        gn, ct, row, _ = best_per_fold
        print(f"STRICT PASS: {gn} @ cutoff={ct}, K={int(row['k_dropped'])}, "
              f"PR-AUC={row['kept_pr_auc_mean']:.3f}, "
              f"tile_frac={row['tile_kept_frac']:.3f}, "
              f"Δlift={row['delta_norm_lift']:+.3f}")
    else:
        print("STRICT FAIL across all combos.")
    if best_pooled_b is not None:
        gn, ct, dlt, _ = best_pooled_b
        print(f"BEST POOLED-B: {gn} @ cutoff={ct}, Δ PR-AUC = {dlt:+.4f}")


if __name__ == "__main__":
    main()
