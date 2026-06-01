"""Stage 7a bulk fetch -- download every available COLOR.JP2 + COLOR.LBL into
`cache_v2/hirise_color/`. Reads `coverage.parquet` (built by
`scripts/run_stage7a_audit.py`); skips already-cached files (size-checked).

After all JP2s are cached, parses each LBL via `src.colour.parse_color_lbl` and
writes a unified metadata parquet `cache_v2/hirise_color/lbl_metadata.parquet`
for fast downstream loading.

Idempotent: rerunning skips done images and resumes partial downloads via a
`.partial` tempfile. Total download is ~9 GB for the v2 cohort; runtime is
network-bound (typically 5-20 min).

Run via:
    conda run --no-capture-output -n geospatial python -u scripts/run_stage7a_fetch.py
"""
from __future__ import annotations

import functools
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import truststore

truststore.inject_into_ssl()

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import colour  # noqa: E402

CACHE_DIR = Path("cache_v2/hirise_color")
COVERAGE = CACHE_DIR / "coverage.parquet"
METADATA_OUT = CACHE_DIR / "lbl_metadata.parquet"
USER_AGENT = "hirise2ctx/0.1 (stage7a-fetch; brianvamaro@gmail.com)"

# Chunk + timeout chosen to be friendly to the LPL server (single TCP connection per
# file, 1 MB read chunks, generous timeout for the largest 700 MB JP2).
DOWNLOAD_TIMEOUT = 600
CHUNK = 1 << 20


def _try_download_once(url: str, tmp: Path) -> tuple[bool, str]:
    """One download attempt. Returns (success, info-line). Caller cleans `tmp` on fail."""
    if tmp.exists():
        tmp.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f, length=CHUNK)
    except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as e:
        # `OSError` covers WinError 10054 connection-reset surfaces in some Python builds.
        return False, f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    size_mb = tmp.stat().st_size / 1024 / 1024
    return True, f"{size_mb:>7.1f} MB in {dt:>5.1f} s ({size_mb/max(0.01,dt):>5.1f} MB/s)"


def _download(url: str, out_path: Path, expected_bytes: int, *, max_attempts: int = 4) -> str:
    """Returns one of {"cached", "downloaded", "error"}.

    PDS occasionally drops connections mid-stream (WinError 10054); retry with
    exponential backoff (1, 5, 15, 45 s) so a single transient failure doesn't
    abort a multi-GB cohort fetch.
    """
    if out_path.exists() and out_path.stat().st_size == expected_bytes:
        return "cached"
    if out_path.exists():
        # Size mismatch -- restart cleanly rather than try to resume.
        out_path.unlink()
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    backoff = [1, 5, 15, 45]
    for attempt in range(1, max_attempts + 1):
        ok, info = _try_download_once(url, tmp)
        if not ok:
            print(f"    attempt {attempt}/{max_attempts} FAILED: {info}")
            if attempt < max_attempts:
                sleep_s = backoff[min(attempt - 1, len(backoff) - 1)]
                print(f"    retrying in {sleep_s} s ...")
                time.sleep(sleep_s)
            continue
        if tmp.stat().st_size != expected_bytes:
            print(f"    attempt {attempt} SIZE MISMATCH: got {tmp.stat().st_size}, "
                  f"expected {expected_bytes}")
            if attempt < max_attempts:
                sleep_s = backoff[min(attempt - 1, len(backoff) - 1)]
                print(f"    retrying in {sleep_s} s ...")
                time.sleep(sleep_s)
            continue
        print(f"    {info}")
        tmp.replace(out_path)
        return "downloaded"
    if tmp.exists():
        tmp.unlink()
    return "error"


def main() -> int:
    if not COVERAGE.exists():
        print(f"ERROR: {COVERAGE} not found. Run scripts/run_stage7a_audit.py first.")
        return 1
    cov = pd.read_parquet(COVERAGE)
    available = cov[cov["has_color"]].reset_index(drop=True)
    print(f"Available COLOR products: {len(available)}/{len(cov)}")
    total_bytes = int(available["color_jp2_bytes"].sum())
    print(f"Total fetch budget: {total_bytes/1024/1024:.0f} MB\n")

    n_cached_jp2 = n_dl_jp2 = n_err_jp2 = 0
    n_cached_lbl = n_dl_lbl = n_err_lbl = 0
    for i, r in available.iterrows():
        obs_id = r["obs_id"]
        print(f"[{i+1:>2}/{len(available)}] {obs_id}")

        jp2_path = colour.color_jp2_path(Path("cache_v2"), obs_id)
        lbl_path = colour.color_lbl_path(Path("cache_v2"), obs_id)

        status = _download(r["color_lbl_url"], lbl_path, int(r["color_lbl_bytes"]))
        if status == "cached":
            n_cached_lbl += 1
        elif status == "downloaded":
            n_dl_lbl += 1
        else:
            n_err_lbl += 1

        status = _download(r["color_jp2_url"], jp2_path, int(r["color_jp2_bytes"]))
        if status == "cached":
            n_cached_jp2 += 1
        elif status == "downloaded":
            n_dl_jp2 += 1
        else:
            n_err_jp2 += 1

    print(f"\nFetch summary:")
    print(f"  JP2: cached={n_cached_jp2}  downloaded={n_dl_jp2}  errors={n_err_jp2}")
    print(f"  LBL: cached={n_cached_lbl}  downloaded={n_dl_lbl}  errors={n_err_lbl}")

    # Build the unified LBL metadata parquet from every COLOR.LBL we now have on disk.
    print(f"\nParsing LBL metadata into {METADATA_OUT}")
    meta_rows = []
    for obs_id in available["obs_id"]:
        lbl_path = colour.color_lbl_path(Path("cache_v2"), obs_id)
        if not lbl_path.exists():
            print(f"  {obs_id}: LBL missing (download failed); skipping")
            continue
        try:
            lbl = colour.parse_color_lbl(lbl_path)
        except Exception as e:
            print(f"  {obs_id}: LBL parse error: {e}")
            continue
        meta_rows.append({
            "obs_id": lbl.obs_id,
            "incidence_deg": lbl.incidence_deg,
            "emission_deg": lbl.emission_deg,
            "phase_deg": lbl.phase_deg,
            "solar_longitude_deg": lbl.solar_longitude_deg,
            "north_azimuth_deg": lbl.north_azimuth_deg,
            "scaling_factor": lbl.scaling_factor,
            "offset": lbl.offset,
            "map_scale_mpp": lbl.map_scale_mpp,
            "cos_incidence": lbl.cos_incidence,
            "color_lines": lbl.lines,
            "color_line_samples": lbl.line_samples,
            "swath_width_m": round(lbl.line_samples * lbl.map_scale_mpp, 2),
            "bands": lbl.bands,
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(METADATA_OUT)
    print(f"  -> {len(meta_df)} rows")
    print(f"  incidence range: {meta_df['incidence_deg'].min():.1f} - "
          f"{meta_df['incidence_deg'].max():.1f} deg")
    print(f"  swath width range: {meta_df['swath_width_m'].min():.0f} - "
          f"{meta_df['swath_width_m'].max():.0f} m")
    print(f"  map_scale distribution: {meta_df['map_scale_mpp'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
