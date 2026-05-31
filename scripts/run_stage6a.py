"""Run Stage 6a (spatial-context neighbour features) for one ObsId or all.

Reads existing Stage 4b feature parquets and writes augmented parquets to a parallel
``features_nbr/`` directory next to them.  The augmented frame has every original
column plus ``nbr_<stat>_<feat>`` columns (default stats: mean / max / std).  The
Stage 4b cache is NOT modified.

Usage:
    # Single image:
    conda run -n geospatial python scripts/run_stage6a.py ESP_069669_2220 \
        --dataset-dir dataset_v2_dev

    # All Stage-4b-ready ObsIds in a dataset:
    conda run -n geospatial python scripts/run_stage6a.py --all \
        --dataset-dir dataset_v2_dev

Outputs:
    dataset_v2_dev/features_nbr/{ObsId}.parquet
    dataset_v2_dev/features_nbr/{ObsId}.json     # provenance: stats, stencil, source
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.features import EXCLUDED_FROM_SWEEP, FEATURES_SUBDIR  # noqa: E402
from src.spatial_features import (  # noqa: E402
    DEFAULT_STENCIL_SIZE,
    STATS_SUPPORTED,
    add_neighbour_features,
    select_feature_columns,
)

NBR_FEATURES_SUBDIR = "features_nbr"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _augment_one(
    obs_id: str, *, dataset_dir: Path, stats: tuple[str, ...], stencil_size: int,
    output_subdir: str = NBR_FEATURES_SUBDIR,
) -> dict | None:
    in_parquet = dataset_dir / FEATURES_SUBDIR / f"{obs_id}.parquet"
    if not in_parquet.exists():
        print(f"  {obs_id}: SKIP (Stage 4b output missing at {in_parquet})", flush=True)
        return None
    out_dir = dataset_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"{obs_id}.parquet"
    out_sidecar = out_dir / f"{obs_id}.json"

    t0 = time.monotonic()
    df = pd.read_parquet(in_parquet)
    base_feature_cols = select_feature_columns(df)
    augmented = add_neighbour_features(
        df, feature_cols=base_feature_cols, stats=stats, stencil_size=stencil_size,
    )
    augmented.to_parquet(out_parquet, index=False)
    dt = time.monotonic() - t0

    new_cols = [c for c in augmented.columns if c.startswith("nbr_")]
    per_scale = {
        int(S): int((df["tile_size_px"] == S).sum())
        for S in sorted({int(s) for s in df["tile_size_px"]})
    }
    provenance = {
        "obs_id": obs_id,
        "source_features_parquet": str(in_parquet),
        "source_sha256_short": _file_sha256(in_parquet),
        "stage6a": {
            "stats": list(stats),
            "stencil_size": stencil_size,
            "base_feature_cols": base_feature_cols,
            "n_base_features": len(base_feature_cols),
            "n_new_columns": len(new_cols),
            "new_columns_preview": new_cols[:6] + (["..."] if len(new_cols) > 6 else []),
        },
        "n_tiles_total": int(len(augmented)),
        "per_scale_tile_counts": per_scale,
        "elapsed_seconds": round(dt, 3),
        "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "parquet_path": str(out_parquet),
    }
    out_sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(
        f"  {obs_id}: n_tiles={len(augmented):6d}  base_feats={len(base_feature_cols):3d}  "
        f"new_cols={len(new_cols):4d}  elapsed={dt:.2f}s",
        flush=True,
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6a spatial-context driver")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("obs_id", nargs="?", default=None, help="HiRISE Observation ID")
    g.add_argument("--all", action="store_true",
                   help="Process every ObsId with a Stage 4b parquet in --dataset-dir")
    parser.add_argument(
        "--dataset-dir", default="dataset_v2_dev",
        help="Dataset root (default: dataset_v2_dev)",
    )
    parser.add_argument(
        "--stats", nargs="+", default=list(STATS_SUPPORTED),
        choices=list(STATS_SUPPORTED),
        help=f"Subset of {STATS_SUPPORTED} to compute (default: all)",
    )
    parser.add_argument(
        "--stencil-size", type=int, default=DEFAULT_STENCIL_SIZE,
        help=f"Odd integer >= 3 (default: {DEFAULT_STENCIL_SIZE})",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Optional suffix appended to features_nbr -> features_nbr_{suffix}; "
             "use to keep follow-up variant outputs separate from the default run.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not (dataset_dir / FEATURES_SUBDIR).exists():
        print(f"No Stage 4b output dir at {dataset_dir / FEATURES_SUBDIR}", flush=True)
        return 1
    stats: tuple[str, ...] = tuple(args.stats)
    output_subdir = NBR_FEATURES_SUBDIR + (f"_{args.output_suffix}" if args.output_suffix else "")

    if args.all:
        obs_ids = sorted(
            p.stem for p in (dataset_dir / FEATURES_SUBDIR).glob("*.parquet")
            if p.stem not in EXCLUDED_FROM_SWEEP
        )
        print(
            f"Stage 6a :: {len(obs_ids)} ObsIds in {dataset_dir / FEATURES_SUBDIR} "
            f"(excluding {sorted(EXCLUDED_FROM_SWEEP)})",
            flush=True,
        )
        print(f"  stats={stats}  stencil_size={args.stencil_size}", flush=True)
        t_all = time.monotonic()
        results = [(o, _augment_one(o, dataset_dir=dataset_dir, stats=stats,
                                    stencil_size=args.stencil_size,
                                    output_subdir=output_subdir)) for o in obs_ids]
        dt_all = time.monotonic() - t_all
        solved = [(o, p) for o, p in results if p is not None]
        skipped = [o for o, p in results if p is None]
        print(f"\nSolved {len(solved)} / {len(obs_ids)} in {dt_all:.1f}s", flush=True)
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}", flush=True)
        if solved:
            total_new = solved[0][1]["stage6a"]["n_new_columns"]
            total_base = solved[0][1]["stage6a"]["n_base_features"]
            print(
                f"\nPer image: {total_base} base features -> +{total_new} new columns "
                f"(={total_base} * {len(stats)})",
                flush=True,
            )
        return 0

    obs = args.obs_id
    in_dir = dataset_dir / FEATURES_SUBDIR
    if not (in_dir / f"{obs}.parquet").exists():
        print(f"{obs}: no Stage 4b parquet at {in_dir / f'{obs}.parquet'}", flush=True)
        return 2
    print(f"Stage 6a :: {obs}  stats={stats}  stencil_size={args.stencil_size}",
          flush=True)
    prov = _augment_one(obs, dataset_dir=dataset_dir, stats=stats,
                        stencil_size=args.stencil_size,
                        output_subdir=output_subdir)
    if prov is None:
        return 1
    print(f"  wrote {prov['parquet_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
