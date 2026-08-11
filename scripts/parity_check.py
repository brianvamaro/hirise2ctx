"""Numerical-parity gate for the Sherlock port (PLAN_RegionalMap §4 item 6).

The key de-risk before the multi-hour regional run: prove the GPU box reproduces the
laptop's predictions, so a torch/CUDA/fp16 difference can't silently corrupt the map.

Workflow:
  1. On the LAPTOP (reference machine):  python scripts/parity_check.py --emit-reference
     -> runs `predict_window` on a fixed deterministic CTX window and writes the
        per-tile prob / prob_raw / abundance to `models/deployable/parity_ref.npz`
        (tiny; uploaded to Sherlock alongside the head + calibration.npz).
  2. On SHERLOCK (gpu node, before submitting the full job):  python scripts/parity_check.py
     -> reruns the same fixed window and asserts the predictions match the reference
        within tolerance. Exit 0 = safe to run map_region; nonzero = drift, stop.

The window is fixed by (--tile, --row, --col, --win); the reference npz records them so
the compare run cannot accidentally use a different window. Default is a small interior
window of E4_N44 (data-bearing, ~196 tiles, seconds to embed).

**Scope limit, measured and stated rather than implied (R13).** E4_N44 contains **0 nodata
pixels over the entire 47,420² tile**, so the default reference exercises neither the
own-tile nor the context nodata gate: a regression in either is invisible to this check.
The masking thresholds are now passed explicitly at the production values (0.3 / 0.0) and
recorded in the reference, so at least a *threshold* change is caught. To also cover the
gates, emit a second reference on a tile with a real mosaic gap — E-8_N32 is the one tile in
the shipped 26 with a substantial one (1,280 masked cells) — into its own file, and keep the
clean E4_N44 one as the pure-numerics reference:

    python scripts/parity_check.py --emit-reference --tile E-8_N32 --row R --col C \\
        --reference models/deployable/parity_ref_gap.npz

Do not move the gap into the *only* reference window: a future threshold change would then
break parity ambiguously (numerics drift vs gate change) with nothing to separate the two.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- OpenMP/DLL bootstrap; must precede numpy

import numpy as np

from src.mapping import predict_window, read_tile_window

CTX_TILES = REPO_ROOT / "cache_v2" / "ctx_tiles"
DEFAULT_MODEL_PARENT = REPO_ROOT / "models" / "deployable"
DEFAULT_REF = DEFAULT_MODEL_PARENT / "parity_ref.npz"
DEFAULT_CALIBRATION = DEFAULT_MODEL_PARENT / "calibration.npz"
TILE_PX = 32


def resolve_model_dir(arg: str | None, model_parent: str | Path | None = None) -> Path:
    if arg:
        return Path(arg)
    parent = Path(model_parent) if model_parent is not None else DEFAULT_MODEL_PARENT
    hits = sorted(p for p in parent.glob("*") if (p / "recipe.json").exists())
    if not hits:
        raise SystemExit(f"no deployable head under {parent}")
    return hits[-1]


def run_window(tile: str, row: int, col: int, win: int, model_dir: Path,
               calibration: str, batch: int, ctx_tiles: str | Path | None = None,
               max_zero_fraction: float = 0.3,
               max_context_zero_fraction: float = 0.0):
    """Embed+predict+calibrate the fixed window; return (ti, tj, prob, prob_raw, abundance).

    `ctx_tiles` is a parameter because `--ctx-tiles` was already being *passed* here as an
    eighth positional argument to a seven-parameter function — this driver raised `TypeError`
    on every invocation, emit and check alike, and the flag was silently ignored besides.

    The two masking thresholds are explicit (R13). They used to be taken from the
    `predict_window` signature defaults, which were 0.5 / absent while every production
    driver passed 0.3, so the one gate meant to prove the GPU box reproduces the laptop was
    reproducing a configuration nothing ever shipped.
    """
    ctx_tiles = Path(ctx_tiles) if ctx_tiles is not None else CTX_TILES
    side = json.loads((ctx_tiles / f"{tile}.json").read_text(encoding="utf-8"))
    zip_path = ctx_tiles / f"{tile}.zip"
    if not zip_path.exists():
        raise SystemExit(f"tile zip missing: {zip_path} (re-fetch via ctx_retrieve)")

    from src.calibration import CalibrationLayer
    from src.fm_embeddings import FangEmbedder
    from src.modeling.mlp_head import DeployableHead

    window = read_tile_window(zip_path, side["inner_tif"], row, col, win)
    embedder = FangEmbedder.load()
    head = DeployableHead.load(model_dir)
    calibrator = CalibrationLayer.load(calibration)
    dev = getattr(getattr(embedder, "device", None), "type", "?")
    pred = predict_window(window, embedder, head, tile_px=TILE_PX, batch=batch,
                          max_zero_fraction=max_zero_fraction,
                          max_context_zero_fraction=max_context_zero_fraction,
                          calibrator=calibrator, apply_isotonic=True)
    keep = np.isfinite(pred.prob)
    return {
        "tile": tile, "row": row, "col": col, "win": win, "device": dev,
        "ti": pred.ti[keep].astype(np.int32), "tj": pred.tj[keep].astype(np.int32),
        "prob": pred.prob[keep].astype(np.float64),
        "prob_raw": pred.prob_raw[keep].astype(np.float64),
        "abundance": pred.abundance[keep].astype(np.float64),
        # R13: the gate configuration and what it dropped, so a reference emitted under one
        # masking policy cannot be compared against a run under another without noticing.
        "max_zero_fraction": np.float64(max_zero_fraction),
        "max_context_zero_fraction": np.float64(max_context_zero_fraction),
        "n_masked_nodata": np.int64(pred.n_masked_nodata),
        "n_masked_context_nodata": np.int64(pred.n_masked_context_nodata),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit-reference", action="store_true",
                    help="write the reference npz (run this on the laptop)")
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--tile", default="E4_N44")
    ap.add_argument("--row", type=int, default=20000)
    ap.add_argument("--col", type=int, default=20000)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--model", default=None)
    # Isolation criterion 4: both artifact roots this driver reads are flags.
    ap.add_argument("--ctx-tiles", default=str(CTX_TILES))
    ap.add_argument("--model-parent", default=str(DEFAULT_MODEL_PARENT))
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--max-zero-fraction", type=float, default=0.3,
                    help="own-tile nodata gate; matches every production driver")
    ap.add_argument("--max-context-zero-fraction", type=float, default=0.0,
                    help="R13 context nodata gate; matches scripts/map_region.py")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=2e-3,
                    help="tolerance; fp16 GPU vs fp32 CPU differs at ~1e-3 on probabilities")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model, args.model_parent)

    if args.emit_reference:
        res = run_window(args.tile, args.row, args.col, args.win, model_dir,
                         args.calibration, args.batch, args.ctx_tiles,
                         args.max_zero_fraction, args.max_context_zero_fraction)
        Path(args.reference).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.reference, **{k: v for k, v in res.items()
                                               if k not in ("device",)})
        print(f"[emit] reference window {res['tile']} ({res['row']},{res['col']}) "
              f"win={res['win']} device={res['device']} tiles={res['ti'].size}")
        if not (res["n_masked_nodata"] or res["n_masked_context_nodata"]):
            print("[emit] ⚠ this window masks 0 tiles, so it does NOT exercise either nodata "
                  "gate — see the module docstring for emitting a gap-bearing second reference")
        print(f"[emit] wrote {args.reference}")
        return 0

    ref_path = Path(args.reference)
    if not ref_path.exists():
        raise SystemExit(f"reference missing: {ref_path}  (run --emit-reference on the laptop "
                         "and upload it with the head)")
    ref = np.load(ref_path)
    # R13: reproduce the reference's OWN gate configuration when it records one. Re-running a
    # reference under different thresholds is a policy change, not machine drift, and it must
    # not be reported as either a pass or a numerical failure.
    mzf = float(ref["max_zero_fraction"]) if "max_zero_fraction" in ref else args.max_zero_fraction
    mczf = (float(ref["max_context_zero_fraction"])
            if "max_context_zero_fraction" in ref else args.max_context_zero_fraction)
    res = run_window(str(ref["tile"]), int(ref["row"]), int(ref["col"]), int(ref["win"]),
                     model_dir, args.calibration, args.batch, args.ctx_tiles, mzf, mczf)
    print(f"[check] window {res['tile']} ({res['row']},{res['col']}) win={res['win']} "
          f"device={res['device']}  ref tiles={ref['ti'].size}  this tiles={res['ti'].size}")
    if "max_context_zero_fraction" not in ref:
        print("[check] ⚠ reference predates the R13 gate record; falling back to this run's "
              f"--max-zero-fraction {mzf} / --max-context-zero-fraction {mczf}. Re-emit it to "
              "pin the masking policy.")
    if (mzf, mczf) != (args.max_zero_fraction, args.max_context_zero_fraction):
        print(f"[check] using the reference's thresholds ({mzf}, {mczf}), not this run's "
              f"({args.max_zero_fraction}, {args.max_context_zero_fraction})")

    if not (np.array_equal(ref["ti"], res["ti"]) and np.array_equal(ref["tj"], res["tj"])):
        print("[FAIL] tile grid (ti,tj) differs -> geometry/masking drift")
        return 2

    # Gate on the FAITHFUL quantities: raw P(rich) and abundance (qmatch) are smooth functions
    # of the embedding, so genuine CUDA/fp16 drift shows up here. The isotonic-CALIBRATED prob
    # is a piecewise-constant (step) function of prob_raw, so a sub-tolerance prob_raw diff that
    # straddles a step edge amplifies into ~1e-2 -- a benign artifact, not drift (and isotonic is
    # only optional polish). So report calibrated prob informationally, don't gate on it.
    ok = True
    for key in ("prob_raw", "abundance"):
        a, b = ref[key], res[key]
        max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
        close = np.allclose(a, b, rtol=args.rtol, atol=args.atol)
        print(f"[gate] {key:9s} max|d|={max_abs:.2e}  allclose={close}")
        ok = ok and close

    cal = ref["prob"], res["prob"]
    cal_max = float(np.max(np.abs(cal[0] - cal[1]))) if cal[0].size else 0.0
    print(f"[info] prob(cal) max|d|={cal_max:.2e}  (isotonic step-amplifies the prob_raw diff; "
          "not gated -- optional polish, render --no-isotonic for an exactly-reproducible map)")

    if ok:
        print(f"[PASS] faithful quantities (prob_raw, abundance) match within rtol={args.rtol} "
              f"atol={args.atol} -> safe to run scripts/map_region.py")
        return 0
    print("[FAIL] prob_raw/abundance drifted beyond tolerance -> investigate torch/CUDA/fp16 "
          "before the full run")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
