"""Run Stage 3 (co-registration) for one ObsId or the full manifest.

Usage:
    # Single image:
    conda run -n geospatial python scripts/run_stage3.py ESP_069669_2220

    # All ObsIds present in the manifest (skips any whose Stage 2 caches are missing):
    conda run -n geospatial python scripts/run_stage3.py --all
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

from src import manifest as M
from src.config import load_config
from src.coregister import stage3_one_image
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR


def _stage2_ready(cache_dir: Path, obs_id: str) -> bool:
    return (
        (cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}.tif").exists()
        and (cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}_hirise_mask.tif").exists()
    )


def _solve_one(cfg, obs_id: str, row) -> dict | None:
    cache_dir = cfg.cache_dir
    if not _stage2_ready(cache_dir, obs_id):
        print(f"  {obs_id}: SKIP (Stage 2 outputs missing)", flush=True)
        return None
    t0 = time.monotonic()
    try:
        prov = stage3_one_image(
            obs_id,
            cache_dir=cache_dir,
            manifest_row=row,
            fft_window_px=int(cfg["coregistration"]["fft_window_px"]),
            upsample_factor=20,
            config_hash=cfg.hash,
        )
    except RuntimeError as e:
        # Common case: HiRISE coverage is too small for any power-of-2 FFT window
        # (e.g. ESP_057469_2215 polygons spill into a neighboring CTX tile, leaving
        # only a 0.1%-coverage thin strip inside the cached window). Log and continue
        # so a single bad ObsId doesn't halt the manifest-wide sweep.
        print(f"  {obs_id}: FAILED ({e})", flush=True)
        return None
    dt = time.monotonic() - t0
    sm = prov["shift_m"]
    print(
        f"  {obs_id}: |shift|={sm['magnitude']:7.1f} m   "
        f"(dx={sm['dx']:+7.1f}, dy={sm['dy']:+7.1f})   "
        f"peak={prov['peak_correlation']:.3f}   "
        f"fft={prov['fft_window']['size_px']}px   "
        f"[{dt:.1f}s]",
        flush=True,
    )
    return prov


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 co-registration driver")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("obs_id", nargs="?", default=None, help="HiRISE Observation ID")
    g.add_argument("--all", action="store_true", help="Solve for every manifest row")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    df = M.load_manifest(cfg.manifest_path)

    if args.all:
        print(f"Stage 3 :: all {len(df)} manifest rows", flush=True)
        results: list[tuple[str, dict | None]] = []
        for _, row in df.iterrows():
            obs = str(row["ObsId"])
            results.append((obs, _solve_one(cfg, obs, row)))
        solved = [(o, p) for o, p in results if p is not None]
        skipped = [o for o, p in results if p is None]
        print(f"\nSolved {len(solved)} / {len(df)}; skipped {len(skipped)}", flush=True)
        if skipped:
            print(f"  Skipped (Stage 2 missing): {', '.join(skipped)}", flush=True)
        if solved:
            print("\nDistribution of |shift|:", flush=True)
            mags = sorted(p["shift_m"]["magnitude"] for _, p in solved)
            print(
                f"  min={mags[0]:.1f}  median={mags[len(mags) // 2]:.1f}  max={mags[-1]:.1f}  m",
                flush=True,
            )
            peaks = sorted(p["peak_correlation"] for _, p in solved)
            print(
                f"  peak: min={peaks[0]:.3f}  median={peaks[len(peaks) // 2]:.3f}  max={peaks[-1]:.3f}",
                flush=True,
            )
        return 0

    obs = args.obs_id
    if obs not in df["ObsId"].values:
        print(f"ObsId {obs!r} not in manifest")
        return 2
    row = df.set_index("ObsId").loc[obs]
    print(f"Stage 3 :: {obs}", flush=True)
    prov = _solve_one(cfg, obs, row)
    if prov is None:
        return 1
    out_json = cfg.cache_dir / "coregistration" / f"{obs}.json"
    print(f"  wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
