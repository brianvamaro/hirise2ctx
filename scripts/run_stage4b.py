"""Run Stage 4b (per-tile CTX texture features) for one ObsId or the full manifest.

Usage:
    # Single image:
    conda run -n geospatial python scripts/run_stage4b.py ESP_069669_2220

    # All ObsIds with Stage 4 labels (auto-skips Stage 4 misses and ESP_057469_2215):
    conda run -n geospatial python scripts/run_stage4b.py --all

Stage 4b reads only existing caches (Stage 2 CTX windows + Stage 4 label parquets); it
does NOT trigger Stage 1/2/3 re-runs. Outputs:
    dataset/features/{ObsId}.parquet              # one row per (scale, ti, tj)
    dataset/features/{ObsId}.json                 # provenance + per-feature-family timings
    dataset/context_patches/{ObsId}_S{32,64}.npy  # bundled patch stacks if enabled

ESP_057469_2215 is excluded from --all (same as Stage 4: 0.1% HiRISE coverage, tile-straddle).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.features import EXCLUDED_FROM_SWEEP, FEATURES_SUBDIR, stage4b_one_image
from src.labeling import LABELS_SUBDIR


def _stage4_ready(output_dir: Path, obs_id: str) -> bool:
    return (
        (output_dir / LABELS_SUBDIR / f"{obs_id}.parquet").exists()
        and (output_dir / LABELS_SUBDIR / f"{obs_id}.json").exists()
    )


def _feature_one(cfg, obs_id: str) -> dict | None:
    if not _stage4_ready(cfg.output_dir, obs_id):
        print(f"  {obs_id}: SKIP (Stage 4 outputs missing)", flush=True)
        return None
    t0 = time.monotonic()
    try:
        prov = stage4b_one_image(
            obs_id,
            cache_dir=cfg.cache_dir,
            output_dir=cfg.output_dir,
            features_cfg=cfg["features"],
            config_hash=cfg.hash,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"  {obs_id}: FAILED ({e})", flush=True)
        return None
    dt = time.monotonic() - t0
    per_scale = prov.get("per_scale_tile_counts", {})
    per_scale_str = "  ".join(f"S={s}:{n}" for s, n in sorted(per_scale.items()))
    glcm_time = sum(
        v.get("glcm", 0.0) for v in prov.get("timings_per_scale_seconds", {}).values()
    )
    print(
        f"  {obs_id}: n_tiles={prov['n_tiles_total']:6d}  {per_scale_str}  "
        f"dn_mode={prov['dn_thresholds']['mode']:3d}  glcm={glcm_time:.1f}s  total={dt:.1f}s",
        flush=True,
    )
    return prov


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4b feature-extraction driver")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("obs_id", nargs="?", default=None, help="HiRISE Observation ID")
    g.add_argument("--all", action="store_true", help="Compute features for all Stage-4-ready ObsIds")
    parser.add_argument("--config", default="config.yaml", help="Path to the pipeline config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = M.load_manifest(cfg.manifest_path)

    if args.all:
        rows = [
            str(row["ObsId"]) for _, row in df.iterrows()
            if str(row["ObsId"]) not in EXCLUDED_FROM_SWEEP
        ]
        print(
            f"Stage 4b :: {len(rows)} of {len(df)} manifest rows "
            f"(excluding {sorted(EXCLUDED_FROM_SWEEP)})", flush=True,
        )
        t_all = time.monotonic()
        results: list[tuple[str, dict | None]] = []
        for obs in rows:
            results.append((obs, _feature_one(cfg, obs)))
        dt_all = time.monotonic() - t_all
        solved = [(o, p) for o, p in results if p is not None]
        skipped = [o for o, p in results if p is None]
        print(f"\nSolved {len(solved)} / {len(rows)} in {dt_all:.1f}s", flush=True)
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}", flush=True)
        if solved:
            total_tiles = sum(p["n_tiles_total"] for _, p in solved)
            print(f"\nTotal feature rows across the sweep: {total_tiles:,}", flush=True)
            # Patch totals (only if any image enabled patches).
            patch_totals: dict[int, tuple[int, int]] = {}
            for _, p in solved:
                cp = p.get("context_patch", {})
                if cp and cp.get("enabled"):
                    for P, n in cp["patch_counts"].items():
                        nbytes = cp["patch_bytes_estimate"][P]
                        cur = patch_totals.get(int(P), (0, 0))
                        patch_totals[int(P)] = (cur[0] + int(n), cur[1] + int(nbytes))
            if patch_totals:
                print("Context patches:", flush=True)
                for P, (n, nbytes) in sorted(patch_totals.items()):
                    print(f"  S={P}: {n:,} patches, {nbytes/1e9:.2f} GB on disk", flush=True)
        return 0

    obs = args.obs_id
    if obs not in df["ObsId"].values:
        print(f"ObsId {obs!r} not in manifest")
        return 2
    print(f"Stage 4b :: {obs}", flush=True)
    prov = _feature_one(cfg, obs)
    if prov is None:
        return 1
    out_parquet = cfg.output_dir / FEATURES_SUBDIR / f"{obs}.parquet"
    out_json = cfg.output_dir / FEATURES_SUBDIR / f"{obs}.json"
    print(f"  wrote {out_parquet}", flush=True)
    print(f"        {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
