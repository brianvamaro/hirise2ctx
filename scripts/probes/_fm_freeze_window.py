"""PLAN_FM 2.1 freeze-window runner: 1b target re-read, 1d pool x head,
1e micro-sweep / cross-head ensemble / calibration, 1g operating-scale read.

Generalizes the head bake-off (`_w2_fang_heads.py`) across tile scale, GeM/
mean/cls pooling, binary target, and MLP architecture, with per-target Tier-1
baselines run in the IDENTICAL harness (cross-target metrics are never compared
directly -- each target reads against its OWN Tier-1, PLAN_FM 2.1b).

Subcommands:
    run    one LOIO cell.  --matrix t1 banks a Tier-1 LightGBM baseline that
           later cells on the same (tile_px, target) resolve automatically;
           --head mlp expands to 3 seeds + mean-prob ensemble (3-seed rule).
    eval   verdict for a banked spec: `a+b+c` = mean-prob ensemble of cell
           labels under models/fang_probe/ (prefix a member with `r:` to use
           its global pct-rank); --transform perimg_rank|blend applies the
           per-image quantile calibration layer (1e).
    pair   paired per-image AUC stats between two specs (freeze evidence).

Usage:
    conda run --no-capture-output -n geospatial python -u \
        scripts/probes/_fm_freeze_window.py run --matrix t1 --head lgbm \
        --tile-px 64 --target bc_ge_1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from scripts.probes._w2_fang_heads import KNNHead, LogRegHead, MLPHead  # noqa: E402
from scripts.probes._w2_fang_probe import (  # noqa: E402  -- probe-tier reuse
    DOSSIER, OUT_ROOT, SCALE_CONFIG, SCHEME, EmbeddingBank, make_fold_iter,
    per_image_auc,
)
from src.modeling.binary_target import get_target
from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, LightGBMClassification

MATRICES = {"t1": ("t1",), "emb": ("ctx",), "t1ctx": ("t1", "ctx")}
DEFAULT_HIDDEN = (256, 64)
DEFAULT_DROPOUT = 0.2
LGBM_PARAMS = LGBMParams(n_estimators=400, learning_rate=0.05, early_stopping_rounds=40)


@dataclass
class MLPArchHead(MLPHead):
    """MLPHead with parameterized width/dropout (1e micro-sweep). With the
    defaults this reproduces the bake-off MLP bit-for-bit (same build order
    after the same manual_seed)."""

    hidden: tuple[int, ...] = DEFAULT_HIDDEN
    dropout: float = DEFAULT_DROPOUT

    def __post_init__(self):
        self.name = f"mlp{arch_tag(self.hidden, self.dropout)}_seed{self.seed}"

    def _build(self, d_in: int):
        import torch.nn as nn

        layers: list = []
        prev = d_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


def arch_tag(hidden: tuple[int, ...], dropout: float) -> str:
    if tuple(hidden) == DEFAULT_HIDDEN and dropout == DEFAULT_DROPOUT:
        return ""
    return f"_h{'x'.join(str(h) for h in hidden)}_d{int(round(dropout * 100)):02d}"


def cell_name(matrix: str, head_label: str, pool: str, tile_px: int, target: str) -> str:
    if matrix == "t1":
        return f"fw_t1_{head_label}_S{tile_px}_{target}"
    return f"fw_{matrix}_{head_label}_{pool}{3 * tile_px}_S{tile_px}_{target}"


# ============================================================================
# Baseline resolution + verdict (target-aware)
# ============================================================================


def resolve_t1_baseline(tile_px: int, target: str) -> tuple[Path, pd.Series]:
    """Tier-1 reference predictions + per-image AUC series for (tile_px, target).

    fa_gt_1e-2 uses the banked refresh runs (exact continuity with every number
    published since 2026-06-11); other targets resolve to the fw_t1 cell banked
    by `run --matrix t1`, with per-image AUC computed from its own predictions
    (identical quantity to the sweep summary's per-fold AUC)."""
    if target == "fa_gt_1e-2":
        _, preds_rel, summary_rel = SCALE_CONFIG[tile_px]
        t1_auc = pd.read_parquet(REPO_ROOT / summary_rel).set_index("held_out_obs_id")["auc"]
        return REPO_ROOT / preds_rel, t1_auc
    hits = sorted(OUT_ROOT.glob(f"fw_t1_lgbm_S{tile_px}_{target}/*/predictions.parquet"))
    if not hits:
        raise SystemExit(
            f"no Tier-1 baseline banked for S={tile_px} / {target}; run\n"
            f"  run --matrix t1 --head lgbm --tile-px {tile_px} --target {target}")
    preds = pd.read_parquet(hits[0], columns=["obs_id", "ti", "tj", "y_true", "y_pred"])
    return hits[0], per_image_auc(preds, "y_pred")


def verdict(label: str, preds: pd.DataFrame, t1_preds_path: Path,
            t1_auc: pd.Series) -> dict:
    """Gate read vs the Tier-1 reference -- same logic as _w2_fang_probe.verdict
    but with the baseline per-image AUC injected (per-target baselines have no
    sweep summary) and the target's positive rate recorded."""
    t1 = pd.read_parquet(t1_preds_path, columns=["obs_id", "ti", "tj", "y_true", "y_pred"])
    t1 = t1.rename(columns={"y_pred": "t1_prob"})
    df = preds[["obs_id", "ti", "tj", "y_true", "y_pred"]].merge(
        t1.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
    assert len(df) == len(preds), f"join loss vs T1: {len(preds)} -> {len(df)}"
    y = df["y_true"].to_numpy().astype(int)
    k = max(1, int(0.05 * y.size))

    dossier = pd.read_parquet(DOSSIER)
    vok = set(dossier[dossier.validity_ok].index)

    out = {}
    for col, lbl in (("t1_prob", "tier1_ref"), ("y_pred", label)):
        s = df[col].to_numpy()
        pr = float(average_precision_score(y, s))
        p5 = float(y[np.argsort(-s)[:k]].mean())
        aucs = per_image_auc(df, col)
        row = {"pooled_pr_auc": pr, "prec_at_5": p5, "med_auc": float(aucs.median()),
               "pos_rate": float(y.mean())}
        if col != "t1_prob":
            d = (aucs - t1_auc).dropna()
            d_v = d[[o in vok for o in d.index]]
            try:
                pval = float(stats.wilcoxon(d_v, zero_method="wilcox").pvalue)
            except ValueError:
                pval = float("nan")
            row.update({
                "dauc_median_v": float(d_v.median()),
                "dauc_win_v": float((d_v > 0).mean()),
                "wilcoxon_p": pval,
                "gate_per_image": bool(d_v.median() >= 0.05 and pval < 0.05),
                "gate_pooled": bool(pr - out["tier1_ref"]["pooled_pr_auc"] >= 0.03),
            })
            cause = dossier["attributed_cause"].reindex(d.index).fillna("unclassified")
            row["dauc_by_cause"] = d.groupby(cause).mean().round(4).to_dict()
            row["per_image_dauc"] = {o: round(float(v), 4) for o, v in d.sort_values().items()}
        out[lbl] = row
    return out


def print_verdict(v: dict) -> None:
    for lbl, row in v.items():
        slim = {k: (round(val, 4) if isinstance(val, float) else val)
                for k, val in row.items() if k not in ("per_image_dauc", "dauc_by_cause")}
        print(f"  {lbl}: {json.dumps(slim, default=str)}", flush=True)


# ============================================================================
# run
# ============================================================================


def run_cell(factory, name: str, *, matrix: str, head: str, pool: str,
             tile_px: int, target_id: str, bank, force: bool) -> pd.DataFrame:
    """One LOIO cell -> banked predictions + (non-baseline) verdict; returns preds."""
    scale_idx = SCALE_CONFIG[tile_px][0]
    target = get_target(target_id)
    snapshot = {
        "variant": name, "task": "classification", "target_id": target_id,
        "scheme": SCHEME, "dataset_dir": "dataset_v2", "scale_idx": scale_idx,
        "tile_size_px": tile_px, "pool": pool if matrix != "t1" else None,
        "sources": list(MATRICES[matrix]), "head": head,
    }
    cfg_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()[:16]
    snapshot["config_hash"] = cfg_hash
    out_dir = OUT_ROOT / name / cfg_hash

    if (out_dir / "predictions.parquet").exists() and not force:
        print(f"=== {name}: cached, skipping (--force to rerun) ===", flush=True)
        return pd.read_parquet(out_dir / "predictions.parquet")

    t0 = time.monotonic()
    print(f"=== {name} (S={tile_px}, target={target_id}, matrix={matrix}) ===", flush=True)
    result = run_loio(
        factory, binarize=target.binarize, task="classification",
        fold_iter=make_fold_iter(bank, MATRICES[matrix], scale_idx,
                                 {"own": tile_px, "ctx": 3 * tile_px}),
        snapshot=snapshot, verbose=True,
    )
    write_run_artifacts(result, out_dir)
    print(f"  [{name}] {time.monotonic() - t0:.0f} s -> {out_dir.relative_to(REPO_ROOT)}",
          flush=True)
    if matrix != "t1":
        t1_path, t1_auc = resolve_t1_baseline(tile_px, target_id)
        v = verdict(name, result.predictions, t1_path, t1_auc)
        (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
        print_verdict(v)
    print(flush=True)
    return result.predictions


def cmd_run(args) -> int:
    pool, tile_px, target_id = args.pool, args.tile_px, args.target
    bank = None
    if args.matrix != "t1":
        bank = EmbeddingBank(pool, pxs=(3 * tile_px,))

    if args.head == "mlp":
        tag = arch_tag(tuple(args.hidden), args.dropout)
        seed_preds = {}
        for s in args.seeds:
            name = cell_name(args.matrix, f"mlp{tag}_seed{s}", pool, tile_px, target_id)
            seed_preds[s] = run_cell(
                lambda s=s: MLPArchHead(seed=s, hidden=tuple(args.hidden), dropout=args.dropout),
                name, matrix=args.matrix, head=f"mlp{tag}", pool=pool,
                tile_px=tile_px, target_id=target_id, bank=bank, force=args.force)
        if len(args.seeds) > 1:
            base = None
            for s in args.seeds:
                p = seed_preds[s][["obs_id", "ti", "tj", "y_true", "y_pred"]].rename(
                    columns={"y_pred": f"p{s}"})
                base = p if base is None else base.merge(
                    p.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
            base["y_pred"] = base[[f"p{s}" for s in args.seeds]].mean(axis=1)
            ens_name = cell_name(args.matrix, f"mlp{tag}_ens{len(args.seeds)}",
                                 pool, tile_px, target_id)
            out_dir = OUT_ROOT / ens_name
            out_dir.mkdir(parents=True, exist_ok=True)
            base[["obs_id", "ti", "tj", "y_true", "y_pred"]].to_parquet(
                out_dir / "predictions.parquet", index=False)
            t1_path, t1_auc = resolve_t1_baseline(tile_px, target_id)
            v = verdict(ens_name, base, t1_path, t1_auc)
            (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
            print(f"=== {ens_name} (mean of {len(args.seeds)} seeds) ===")
            print_verdict(v)
        return 0

    factories = {
        "lgbm": lambda: LightGBMClassification(params=LGBM_PARAMS),
        "logreg": LogRegHead,
        "knn50": KNNHead,
    }
    name = cell_name(args.matrix, args.head, pool, tile_px, target_id)
    preds = run_cell(factories[args.head], name, matrix=args.matrix, head=args.head,
                     pool=pool, tile_px=tile_px, target_id=target_id, bank=bank,
                     force=args.force)
    if args.matrix == "t1":
        y = preds["y_true"].to_numpy().astype(int)
        s = preds["y_pred"].to_numpy()
        k = max(1, int(0.05 * y.size))
        aucs = per_image_auc(preds, "y_pred")
        print(f"  baseline {name}: pooled_pr_auc={average_precision_score(y, s):.4f} "
              f"prec_at_5={y[np.argsort(-s)[:k]].mean():.4f} med_auc={aucs.median():.4f} "
              f"pos_rate={y.mean():.4f}", flush=True)
    return 0


# ============================================================================
# eval / pair (post-hoc, banked predictions only)
# ============================================================================


def load_label(label: str) -> pd.DataFrame:
    direct = OUT_ROOT / label / "predictions.parquet"
    hits = [direct] if direct.exists() else sorted(OUT_ROOT.glob(f"{label}/*/predictions.parquet"))
    if not hits:
        raise SystemExit(f"no banked predictions for label {label!r} under {OUT_ROOT}")
    return pd.read_parquet(hits[0], columns=["obs_id", "ti", "tj", "y_true", "y_pred"])


def load_spec(spec: str) -> pd.DataFrame:
    """`a+b+c` -> mean y_pred over member cells; `r:` prefix takes the member's
    global pct-rank first (rank-mean ensembles across differently calibrated heads)."""
    base = None
    for i, part in enumerate(spec.split("+")):
        as_rank = part.startswith("r:")
        df = load_label(part[2:] if as_rank else part)
        if as_rank:
            df["y_pred"] = df["y_pred"].rank(pct=True)
        df = df.rename(columns={"y_pred": f"p{i}"})
        base = df if base is None else base.merge(
            df.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
    pcols = [c for c in base.columns if c.startswith("p")]
    base["y_pred"] = base[pcols].mean(axis=1)
    return base[["obs_id", "ti", "tj", "y_true", "y_pred"]]


def apply_transform(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    """1e calibration layer: per-image quantile rank (label-free at inference --
    the held-out image's own tile population supplies the ranks), or a 50/50
    blend that keeps half the cross-image level information."""
    df = df.copy()
    if transform == "perimg_rank":
        df["y_pred"] = df.groupby("obs_id")["y_pred"].rank(pct=True)
    elif transform == "blend":
        df["y_pred"] = 0.5 * df.groupby("obs_id")["y_pred"].rank(pct=True) + 0.5 * df["y_pred"]
    elif transform != "none":
        raise SystemExit(f"unknown transform {transform!r}")
    return df


def cmd_eval(args) -> int:
    df = apply_transform(load_spec(args.spec), args.transform)
    label = args.label or f"{args.spec}|{args.transform}"
    t1_path, t1_auc = resolve_t1_baseline(args.tile_px, args.target)
    v = verdict(label, df, t1_path, t1_auc)
    print(f"=== eval {label} (S={args.tile_px}, target={args.target}) ===")
    print_verdict(v)
    if args.save:
        out_dir = OUT_ROOT / args.save
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "predictions.parquet", index=False)
        (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
        print(f"  saved -> {out_dir.relative_to(REPO_ROOT)}")
    return 0


def cmd_pair(args) -> int:
    dossier = pd.read_parquet(DOSSIER)
    vok = set(dossier[dossier.validity_ok].index)
    auc_a = per_image_auc(load_spec(args.a), "y_pred")
    auc_b = per_image_auc(load_spec(args.b), "y_pred")
    d = (auc_b - auc_a).dropna()
    d_v = d[[o in vok for o in d.index]]
    try:
        p = float(stats.wilcoxon(d_v, zero_method="wilcox").pvalue)
    except ValueError:
        p = float("nan")
    print(f"pair: B={args.b}\n  vs A={args.a}\n"
          f"  dAUC(B-A) median(v)={d_v.median():+.4f}  win={float((d_v > 0).mean()):.2f}  "
          f"p={p:.4g}  n(v)={len(d_v)}  median(all)={d.median():+.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one LOIO cell")
    r.add_argument("--matrix", required=True, choices=sorted(MATRICES))
    r.add_argument("--head", required=True, choices=["lgbm", "logreg", "knn50", "mlp"])
    r.add_argument("--pool", default="gem", choices=["gem", "mean", "cls"])
    r.add_argument("--tile-px", type=int, default=64, choices=sorted(SCALE_CONFIG))
    r.add_argument("--target", default="fa_gt_1e-2",
                   choices=["fa_gt_1e-2", "fa_gt_1e-3", "bc_ge_1", "bc_ge_50", "bc_ge_100"])
    r.add_argument("--hidden", nargs="+", type=int, default=list(DEFAULT_HIDDEN))
    r.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="verdict for a banked spec (+ optional calibration)")
    e.add_argument("--spec", required=True)
    e.add_argument("--transform", default="none", choices=["none", "perimg_rank", "blend"])
    e.add_argument("--tile-px", type=int, default=64, choices=sorted(SCALE_CONFIG))
    e.add_argument("--target", default="fa_gt_1e-2",
                   choices=["fa_gt_1e-2", "fa_gt_1e-3", "bc_ge_1", "bc_ge_50", "bc_ge_100"])
    e.add_argument("--label", default=None)
    e.add_argument("--save", default=None)
    e.set_defaults(func=cmd_eval)

    p = sub.add_parser("pair", help="paired per-image AUC stats between two specs")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(func=cmd_pair)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
