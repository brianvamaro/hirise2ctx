"""Fetch any manifest .LBL files not yet cached, with retry on transient errors.

One-off probe written 2026-05-26 to complete the per-image MAP_SCALE coverage
(see DECISIONS.md "Open at this date" entry on the 4 missing LBLs). After this
runs, scripts/probes/_boulder_size_audit.py prints a complete table over all 9
polygon-bearing images.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.pds_labels import fetch_label


def main() -> int:
    cfg = load_config("config.yaml")
    df = M.load_manifest(cfg.manifest_path)
    missing = [
        (r["ObsId"], r["LabelURL"])
        for _, r in df.iterrows()
        if not (cfg.cache_dir / "pds_labels" / f"{r['ObsId']}.LBL").exists()
    ]
    if not missing:
        print("All LBLs already cached.")
        return 0
    print(f"Fetching {len(missing)} missing LBLs.")
    failures = []
    for obs, url in missing:
        for attempt in range(3):
            try:
                p = fetch_label(obs, url, cfg.cache_dir)
                print(f"  {obs:>20s}  OK     {p.stat().st_size:>6d} bytes  (attempt {attempt + 1})")
                break
            except Exception as e:
                print(f"  {obs:>20s}  RETRY  attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                time.sleep(2 + 2 * attempt)
        else:
            failures.append(obs)
            print(f"  {obs:>20s}  FAILED after 3 attempts")
    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nAll LBLs fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
