"""Map pilot: predict rich/poor on a CTX region BEYOND HiRISE coverage (PLAN_FM §2.6 B-E).

First real exercise of the productized inference path. A Murray Lab CTX tile is
4 deg x 4 deg (~237 km) while a HiRISE footprint is ~6 km, so almost all of any
cohort tile is unseen by HiRISE. This script windows a region of a cohort tile
*adjacent to but not overlapping* one image's footprint -- same terrain, zero
HiRISE truth -- and runs the frozen deployable head end-to-end:

    cohort tile (cached zip) -> window beyond footprint -> FangEmbedder.embed_window
    -> DeployableHead.predict -> per-tile rich/poor -> 160 m GeoTIFF + PNG.

Proves the embed->predict->render path AND the global-(ti,tj) placement (the
combine pattern) on one tile before any scale-out. No download: it reuses a tile
zip already in `cache_v2/ctx_tiles/`.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/map_pilot.py
    # options: --obs-id ESP_055253_2245  --win-px 3000  --model models/deployable/<hash>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np

from src.mapping import coarsened_transform, predict_window, read_tile_window, write_geotiff

DATASET_DIR = REPO_ROOT / "dataset_v2"
CTX_WINDOWS = REPO_ROOT / "cache_v2" / "ctx_windows"
CTX_TILES = REPO_ROOT / "cache_v2" / "ctx_tiles"
DEFAULT_MODEL_PARENT = REPO_ROOT / "models" / "deployable"
OUT_FIG = REPO_ROOT / "reports" / "figures"
OUT_MAP = REPO_ROOT / "reports" / "map_pilot"
TILE_PX = 32          # frozen S=32
CONTEXT_PAD = 3 * TILE_PX  # 96-px context box -> minimum edge gap


def resolve_model_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    hits = sorted(p for p in DEFAULT_MODEL_PARENT.glob("*") if (p / "recipe.json").exists())
    if not hits:
        raise SystemExit(f"no deployable head under {DEFAULT_MODEL_PARENT}; run scripts/train_deployable_head.py")
    return hits[-1]


def footprint_pixel_box(obs_id: str, inner_transform) -> tuple[int, int, int, int, str]:
    """(row_min, row_max, col_min, col_max, murray_tile) of the image footprint in tile pixels."""
    side = json.loads((CTX_WINDOWS / f"{obs_id}.json").read_text(encoding="utf-8"))
    xmin, ymin, xmax, ymax = side["actual_bounds_target_crs"]
    a, _, c, _, e, f = (inner_transform[i] for i in range(6))
    px_x, px_y = abs(a), abs(e)
    col_min = int(round((xmin - c) / px_x))
    col_max = int(round((xmax - c) / px_x))
    row_min = int(round((f - ymax) / px_y))
    row_max = int(round((f - ymin) / px_y))
    return row_min, row_max, col_min, col_max, side["source_murray_tile"]


def candidate_offsets(fp_box, win: int, tile_h: int, tile_w: int, gap: int):
    """Yield (row_off, col_off, where) placements beyond the footprint, clamped in-tile."""
    r0, r1, c0, c1 = fp_box

    def clamp(v, hi):
        return max(0, min(int(v), hi - win))

    # East / West share the footprint's latitude band; North / South its longitude band.
    yield clamp(r0, tile_h), clamp(c1 + gap, tile_w), "east"
    yield clamp(r0, tile_h), clamp(c0 - gap - win, tile_w), "west"
    yield clamp(r1 + gap, tile_h), clamp(c0, tile_w), "south"
    yield clamp(r0 - gap - win, tile_h), clamp(c0, tile_w), "north"


def overlaps(fp_box, row_off, col_off, win) -> bool:
    r0, r1, c0, c1 = fp_box
    return not (row_off + win <= r0 or row_off >= r1 or col_off + win <= c0 or col_off >= c1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-id", default="ESP_055253_2245", help="cohort image whose tile to map")
    ap.add_argument("--win-px", type=int, default=3000, help="square window side in CTX pixels (5 m/px)")
    ap.add_argument("--gap-px", type=int, default=256, help="min gap from the footprint")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-zero-fraction", type=float, default=0.3,
                    help="reject a candidate window if more than this share of pixels are mosaic nodata")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model)
    card = json.loads((model_dir / "recipe.json").read_text(encoding="utf-8"))
    print(f"=== map pilot: obs={args.obs_id}  win={args.win_px}px  model={model_dir.name} ===")
    print(f"  recipe={card['recipe'].get('cell')}  trained on {card['n_train_images']} images", flush=True)

    tile_sidecar_path = None
    side = json.loads((CTX_WINDOWS / f"{args.obs_id}.json").read_text(encoding="utf-8"))
    murray_tile = side["source_murray_tile"]
    tile_sidecar_path = CTX_TILES / f"{murray_tile}.json"
    if not tile_sidecar_path.exists():
        raise SystemExit(f"tile sidecar missing: {tile_sidecar_path}")
    tile_info = json.loads(tile_sidecar_path.read_text(encoding="utf-8"))
    zip_path = CTX_TILES / f"{murray_tile}.zip"
    if not zip_path.exists():
        raise SystemExit(f"tile zip missing: {zip_path} (re-download via ctx_retrieve.ensure_tile_cached)")
    inner_tif = tile_info["inner_tif"]
    inner_transform = tile_info["inner_transform"]
    tile_h, tile_w = tile_info["inner_shape"]

    fp_box = footprint_pixel_box(args.obs_id, inner_transform)[:4]
    print(f"  tile={murray_tile} {tile_h}x{tile_w}px  footprint rows[{fp_box[0]}:{fp_box[1]}] "
          f"cols[{fp_box[2]}:{fp_box[3]}]", flush=True)

    # Pick the first placement beyond the footprint with acceptable mosaic coverage.
    win = args.win_px
    gap = max(args.gap_px, CONTEXT_PAD)
    chosen = None
    for row_off, col_off, where in candidate_offsets(fp_box, win, tile_h, tile_w, gap):
        if overlaps(fp_box, row_off, col_off, win):
            print(f"  [{where}] off=({row_off},{col_off}) skip: overlaps footprint", flush=True)
            continue
        w = read_tile_window(zip_path, inner_tif, row_off, col_off, win)
        zero_frac = float((w.data == 0).mean())
        print(f"  [{where}] off=({row_off},{col_off}) zero_frac={zero_frac:.3f}", flush=True)
        if zero_frac <= args.max_zero_fraction:
            chosen = (w, where)
            break
    if chosen is None:
        raise SystemExit("no candidate window cleared the nodata threshold; widen --max-zero-fraction or pick another --obs-id")
    window, where = chosen
    print(f"  -> mapping the {where} window", flush=True)

    # --- embed + predict ---
    from src.fm_embeddings import FangEmbedder
    from src.modeling.mlp_head import DeployableHead

    t0 = time.monotonic()
    embedder = FangEmbedder.load()
    head = DeployableHead.load(model_dir)
    pred = predict_window(window, embedder, head, tile_px=TILE_PX,
                          max_zero_fraction=args.max_zero_fraction)
    finite = np.isfinite(pred.prob)
    print(f"  embed+predict {time.monotonic() - t0:.0f}s  tiles={pred.ti.size}  "
          f"valid={pred.n_valid}  nodata_masked={pred.n_masked_nodata}  "
          f"predicted={int(finite.sum())}", flush=True)
    if finite.any():
        p = pred.prob[finite]
        print(f"  prob: mean={p.mean():.3f}  >=0.5 share={float((p >= 0.5).mean()):.3f}  "
              f"min={p.min():.3f}  max={p.max():.3f}", flush=True)

    # --- write GeoTIFF (160 m, tile CRS) ---
    OUT_MAP.mkdir(parents=True, exist_ok=True)
    stem = f"map_pilot_{murray_tile}_{args.obs_id}_{where}"
    tif_path = OUT_MAP / f"{stem}.tif"
    write_geotiff(tif_path, pred.raster, pred.transform, pred.crs_wkt)
    print(f"  GeoTIFF -> {tif_path.relative_to(REPO_ROOT)}", flush=True)

    # --- render PNG ---
    png_path = render_png(window, pred, stem, args.obs_id, murray_tile, where, card)
    print(f"  PNG     -> {png_path.relative_to(REPO_ROOT)}", flush=True)

    (OUT_MAP / f"{stem}.json").write_text(json.dumps({
        "obs_id": args.obs_id, "murray_tile": murray_tile, "placement": where,
        "window_offset_rowcol": [window.row_off, window.col_off], "win_px": win,
        "tile_px": TILE_PX, "model_dir": str(model_dir.relative_to(REPO_ROOT)),
        "recipe_hash": card.get("recipe_hash"), "n_tiles": int(pred.ti.size),
        "n_predicted": int(finite.sum()), "n_nodata_masked": int(pred.n_masked_nodata),
        "rich_share_at_0p5": float((pred.prob[finite] >= 0.5).mean()) if finite.any() else None,
    }, indent=2), encoding="utf-8")
    print(f"  [done] -> reports/map_pilot/{stem}.*")
    return 0


def render_png(window, pred, stem, obs_id, murray_tile, where, card) -> Path:
    """CTX backdrop + predicted rich/poor probability heatmap (160 m) side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    raster = np.ma.masked_invalid(pred.raster)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)

    axes[0].imshow(window.data, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"CTX 5 m/px ({window.data.shape[0]}x{window.data.shape[1]} px)")

    im = axes[1].imshow(raster, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("P(boulder-rich)  fa>1e-2  @160 m")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    rich = np.ma.masked_invalid((pred.raster >= 0.5).astype(float))
    rich.mask = ~np.isfinite(pred.raster)
    axes[2].imshow(rich, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("rich / poor  (P>=0.5)")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Map pilot — {murray_tile} ({where} of {obs_id}, beyond HiRISE coverage)  "
                 f"·  frozen {card['recipe'].get('cell')}", fontsize=11)
    png_path = OUT_FIG / f"{stem}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    raise SystemExit(main())
