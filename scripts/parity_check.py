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


def resolve_model_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    hits = sorted(p for p in DEFAULT_MODEL_PARENT.glob("*") if (p / "recipe.json").exists())
    if not hits:
        raise SystemExit(f"no deployable head under {DEFAULT_MODEL_PARENT}")
    return hits[-1]


def run_window(tile: str, row: int, col: int, win: int, model_dir: Path,
               calibration: str, batch: int):
    """Embed+predict+calibrate the fixed window; return (ti, tj, prob, prob_raw, abundance)."""
    side = json.loads((CTX_TILES / f"{tile}.json").read_text(encoding="utf-8"))
    zip_path = CTX_TILES / f"{tile}.zip"
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
                          calibrator=calibrator, apply_isotonic=True)
    keep = np.isfinite(pred.prob)
    return {
        "tile": tile, "row": row, "col": col, "win": win, "device": dev,
        "ti": pred.ti[keep].astype(np.int32), "tj": pred.tj[keep].astype(np.int32),
        "prob": pred.prob[keep].astype(np.float64),
        "prob_raw": pred.prob_raw[keep].astype(np.float64),
        "abundance": pred.abundance[keep].astype(np.float64),
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
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=2e-3,
                    help="tolerance; fp16 GPU vs fp32 CPU differs at ~1e-3 on probabilities")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model)

    if args.emit_reference:
        res = run_window(args.tile, args.row, args.col, args.win, model_dir,
                         args.calibration, args.batch)
        Path(args.reference).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.reference, **{k: v for k, v in res.items()
                                               if k not in ("device",)})
        print(f"[emit] reference window {res['tile']} ({res['row']},{res['col']}) "
              f"win={res['win']} device={res['device']} tiles={res['ti'].size}")
        print(f"[emit] wrote {args.reference}")
        return 0

    ref_path = Path(args.reference)
    if not ref_path.exists():
        raise SystemExit(f"reference missing: {ref_path}  (run --emit-reference on the laptop "
                         "and upload it with the head)")
    ref = np.load(ref_path)
    res = run_window(str(ref["tile"]), int(ref["row"]), int(ref["col"]), int(ref["win"]),
                     model_dir, args.calibration, args.batch)
    print(f"[check] window {res['tile']} ({res['row']},{res['col']}) win={res['win']} "
          f"device={res['device']}  ref tiles={ref['ti'].size}  this tiles={res['ti'].size}")

    ok = True
    if not (np.array_equal(ref["ti"], res["ti"]) and np.array_equal(ref["tj"], res["tj"])):
        print("[FAIL] tile grid (ti,tj) differs -> geometry/masking drift")
        return 2
    for key in ("prob", "prob_raw", "abundance"):
        a, b = ref[key], res[key]
        max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
        close = np.allclose(a, b, rtol=args.rtol, atol=args.atol)
        print(f"[check] {key:9s} max|d|={max_abs:.2e}  allclose={close}")
        ok = ok and close
    if ok:
        print(f"[PASS] predictions match within rtol={args.rtol} atol={args.atol} "
              "-> safe to run scripts/map_region.py")
        return 0
    print("[FAIL] predictions drifted beyond tolerance -> investigate torch/CUDA/fp16 "
          "before the full run")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
