"""Sequentially run Stage 2 for every ObsId in the manifest that isn't already cached.

Each unique Murray Lab tile is fetched at most once (the cache key is the tile, not the
ObsId). Each ObsId's CTX window + HiRISE coverage mask is written to cache/ctx_windows/.

Sequential by design: parallel downloads would compete for bandwidth on a single
connection (memory feedback #4 — single-connection plain HTTP beat /vsicurl/ 140× on
this network) and the memory advice forbids running two long IO jobs concurrently.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR, stage2_one_image


def _progress(prefix: str):
    last_pct = [-10]
    started = time.monotonic()

    def cb(downloaded: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(downloaded * 100 / total)
        if pct >= last_pct[0] + 10:
            elapsed = time.monotonic() - started
            mb = downloaded / (1024 * 1024)
            mb_per_s = mb / elapsed if elapsed > 0 else 0.0
            print(f"    [{prefix}] {pct:3d}% ({mb:7.1f} MB) {mb_per_s:5.1f} MB/s", flush=True)
            last_pct[0] = pct

    return cb


def main() -> int:
    cfg = load_config("config.yaml")
    df = M.load_manifest(cfg.manifest_path)
    cfg_retrieve = cfg["ctx_retrieve"]

    print(f"Stage 2 sweep over {len(df)} ObsIds", flush=True)
    t_total = time.monotonic()
    done = 0
    skipped = 0
    for _, row in df.iterrows():
        obs = str(row["ObsId"])
        out_tif = cfg.cache_dir / CTX_WINDOWS_SUBDIR / f"{obs}.tif"
        mask_tif = cfg.cache_dir / CTX_WINDOWS_SUBDIR / f"{obs}_hirise_mask.tif"
        if out_tif.exists() and mask_tif.exists():
            print(f"\n{obs}: already cached, skipping", flush=True)
            skipped += 1
            continue
        print(f"\n{obs}: tile={row['CTX_TileName']}  ({row['BoulderLabel']})", flush=True)
        t0 = time.monotonic()
        prov = stage2_one_image(
            obs,
            cache_dir=cfg.cache_dir,
            manifest_row=row,
            target_crs=cfg["target_crs"],
            url_template=cfg["ctx_mosaic"]["url_template"],
            buffer_m=float(cfg_retrieve["buffer_m"]),
            nominal_width_m=float(cfg_retrieve["nominal_hirise_width_m"]),
            nominal_length_m=float(cfg_retrieve["nominal_hirise_length_m"]),
            config_hash=cfg.hash,
            on_progress=_progress(obs),
        )
        dt = time.monotonic() - t0
        print(
            f"  done in {dt:.1f}s — footprint_source={prov['footprint_source']}, "
            f"coverage={prov['hirise_coverage_fraction']:.3f}, shape={prov['actual_shape']}",
            flush=True,
        )
        done += 1

    elapsed = time.monotonic() - t_total
    print(f"\nSweep complete: {done} freshly fetched, {skipped} already cached, "
          f"{elapsed/60:.1f} min total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
