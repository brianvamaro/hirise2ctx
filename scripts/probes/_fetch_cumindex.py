"""Download the PDS CTX CUMINDEX (label + cumulative-index table) from the latest mrox volume.

The CUMINDEX is a flat ASCII table covering every CTX observation (~1.4 M rows in
the 2026-04-30 cut) with PRODUCT_ID, INCIDENCE_ANGLE, EMISSION_ANGLE, PHASE_ANGLE
and related geometry. Stage 6b joins this onto the Murray Lab SeamMap source IDs to
get per-tile illumination geometry.

We cache to ``cache/pds_ctx_cumindex.{lbl,tab}`` per the Stage 6b handoff. SSL is
fixed via ``truststore.inject_into_ssl()`` for Windows / Anaconda (see
[[conda_windows_ssl]]).

Latest verified volume: mrox_5520, 2026-04-30, 87 MB.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import truststore
truststore.inject_into_ssl()

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MROX_VOLUME = "mrox_5520"
BASE_URL = (
    f"https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{MROX_VOLUME}/index"
)

DOWNLOADS = [
    ("cumindex.lbl", CACHE_DIR / "pds_ctx_cumindex.lbl"),
    ("cumindex.tab", CACHE_DIR / "pds_ctx_cumindex.tab"),
]


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size:,} B)")
        return
    print(f"[get ] {url}")
    t0 = time.time()
    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_read = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
        size_hdr = resp.headers.get("Content-Length")
        total = int(size_hdr) if size_hdr else None
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                bytes_read += len(buf)
                if total:
                    pct = 100.0 * bytes_read / total
                    print(
                        f"    {bytes_read/1e6:7.1f} / {total/1e6:.1f} MB "
                        f"({pct:5.1f}%)",
                        end="\r",
                    )
    tmp.replace(dest)
    dt = time.time() - t0
    print(f"\n[ok  ] {dest.name}: {bytes_read:,} B in {dt:.1f} s")


def main() -> int:
    for name, dest in DOWNLOADS:
        _download(f"{BASE_URL}/{name}", dest)
    print("\nDone. Cached:")
    for _, dest in DOWNLOADS:
        if dest.exists():
            print(f"  {dest}  ({dest.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
