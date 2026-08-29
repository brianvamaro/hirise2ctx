"""Map-output manifest and sidecar bookkeeping. **STANDARD LIBRARY ONLY — keep it that way.**

Extracted from `scripts/map_region.py` on 2026-08-25 for one reason: the tools that *repair* and
*verify* a shipped map generation must run when the environment is awkward, and they did not.
`scripts/rebuild_map_manifest.py` loaded `map_region.py`, which does `import src.modeling` (the
torch/OpenMP bootstrap), so a repair tool needed CUDA-capable torch to move a few JSON keys
around — and on a Sherlock login node without the module loaded it did not even reach that,
failing under the system Python 2.7 with a non-ASCII SyntaxError.

A recovery tool that shares the heavy dependencies of the thing it recovers is a recovery tool
you cannot use in the situation you built it for. Everything here is `json`, `os`, `pathlib`,
`hashlib`: importable under any Python 3, no numpy, no rasterio, no torch.

`scripts/map_region.py` re-exports these names, so existing callers are unaffected.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

#: JSON files this project writes into a map-output directory that are NOT tile sidecars: the
#: manifest index, `plan.json` from `scripts/plan_map_extent.py` (which lives beside the
#: product it defines), and `union_manifest.json` from `scripts/map_union.py` (which lives in
#: `reports/map_union`, a map-output directory with mosaics but no tile sidecars of its own).
#: Anything else in `*.json` is treated as a tile, deliberately -- see `tile_sidecars`.
MANIFEST_NAMES = ("region_manifest", "a1_manifest", "plan", "union_manifest")


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed. Mirrors `src.mapping.file_sha256`, without the numpy import."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> Path:
    """Write JSON via a **per-process** `.tmp` sibling + rename, so a reader never sees half a
    record *and* two concurrent writers cannot destroy each other's staging file.

    ⚠ **The temp name must be unique per writer, and it was not (fixed 2026-08-25).** This
    staged to a fixed `<path>.tmp`, which is atomic against a *reader* but unsafe against a
    concurrent *writer*: both processes write the same `<path>.tmp`, the first `replace()`
    renames it away, and the second dies on

        FileNotFoundError: '.../region_manifest.json.tmp' -> '.../region_manifest.json'

    Measured: step 11's baseline array, tasks 0 and 1 both assembly-only and both finishing at
    37 s, wrote `region_manifest.json` simultaneously. Task 1 won; **task 0 crashed after its
    tile was fully committed**, losing only its index entry — and its `run_tile_isolated` tally,
    which is why the failure looked like a missing log line rather than a crash. Going from 6
    array tasks to one-tile-per-task is what exposed it; 6 staggered tasks rarely collide, 26
    short ones do.

    With a unique name every writer owns its staging file, so the write always lands and the
    outcome is last-writer-wins rather than a crash. `os.replace` is atomic on both POSIX and
    Windows for a same-directory rename, so a reader still never observes a partial file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def tile_sidecars(out_dir: Path) -> dict[str, Path]:
    r"""Every per-tile sidecar in a map-output directory, keyed by tile. Excludes MANIFEST_NAMES.

    **Deliberately a denylist, not a name pattern.** Matching `E-?\d+_N-?\d+` would be tighter,
    but `tile_result_rows` must index *whatever footprint is on disk* -- that is what makes the
    manifest self-healing after a task dies mid-stride -- and a naming rule reintroduces exactly
    the hardcoded-tile-list assumption the driver is built to avoid. So the rule stays "a JSON
    here is a tile unless it is one of the files we ourselves write that isn't".

    ⚠ **Anything new written into a product directory must be added to MANIFEST_NAMES.**
    `plan.json` was not, and `verify_map_download.py` duly reported it as an unexpected tile with
    no `rasters` record and -- worse -- as a *second grid_id*, firing the R01 "two lattices in one
    product" alarm on a product entirely on one lattice.
    """
    return {p.stem: p for p in sorted(Path(out_dir).glob("*.json"))
            if p.stem not in MANIFEST_NAMES}


def tile_result_rows(out_dir: Path, tiles: list[str] | None = None) -> list[dict]:
    """Reconstruct the manifest's `results` rows from the tile sidecars ON DISK.

    **Why derive rather than carry forward (2026-08-25).** The manifest used to be built by
    reading the previous manifest and merging this run's rows into it. That is a
    read-modify-write with no synchronisation, so a task whose write is overtaken silently
    disappears from the index — and a task that never reaches the write (a wall-clock kill, a
    dead GPU, the `.tmp` collision above) never appears at all. Measured damage on the shipped
    baseline arm: **21 of 26 tiles indexed**, the five gaps being two tiles each from the two
    step-11 tasks that died mid-stride, plus `E0_N36` from the `.tmp` collision. The A1 arm was
    worse — its driver did not merge at all, so **1 of 26**.

    Deriving from the sidecars makes the index **self-healing**: the sidecar is already the
    authority for how a tile was made (it carries the full `run` block, `win_px` included), it
    is written last as R14's completion marker, and so a sidecar on disk *is* a completed tile.
    Any later write therefore repairs every earlier loss, and the worst a future collision can
    do is drop one entry from `runs[]`.

    The trade named plainly: the manifest now tracks **on-disk reality** instead of accumulating
    history, so a tile whose rasters were deleted drops out of the index rather than lingering
    as a claim about files that are gone. For an index of what exists, that is the better
    failure direction.

    `tiles` is an optional *filter*, not the source of truth: passing None indexes whatever is
    on disk, so a driver run on a non-`BLOCK_TILES` footprint still produces a complete index.
    """
    found = tile_sidecars(out_dir)
    want = sorted(found) if tiles is None else sorted(set(found) & set(tiles))
    rows = []
    for t in want:
        try:
            rec = json.loads(found[t].read_text(encoding="utf-8"))
        except ValueError:                      # a sidecar being rewritten right now
            continue
        rows.append({"tile": t, "status": "done",
                     "windows": (rec.get("run") or {}).get("n_windows"),
                     "elapsed_s": rec.get("elapsed_s"),
                     "n_unique_cells": rec.get("n_unique_cells"),
                     "raster_shape": rec.get("raster_shape")})
    return rows


def merge_manifest(path: Path, *, out_dir: Path, grid_id: str, run_record: dict | None,
                   all_tiles: list[str] | None = None,
                   results: list[dict] | None = None) -> dict:
    """Compose a map manifest: `results` derived from the sidecars, `runs` appended.

    `results` may be passed to fold in rows the sidecars cannot express — a `failed` row, whose
    whole point is that no sidecar was written. Those are merged UNDER the derived rows, so a
    tile that has since rendered wins over a stale `failed` row.
    """
    prev = {}
    if Path(path).exists():
        try:
            prev = json.loads(Path(path).read_text(encoding="utf-8"))
        except ValueError:
            prev = {}
    by_tile = {r["tile"]: r for r in tile_result_rows(out_dir, all_tiles)}
    for r in (results or []):
        if isinstance(r, dict) and r.get("tile") and r.get("status") != "done":
            by_tile.setdefault(r["tile"], r)     # keep the sidecar's row if there IS one
    runs = [r for r in (prev.get("runs") or []) if isinstance(r, dict)]
    if not runs and prev.get("model_dir"):       # fold a pre-R14 manifest in as run #0
        runs.append({k: prev.get(k) for k in
                     ("model_dir", "head_digest", "calibration", "calibration_digest",
                      "ctx_tiles", "recipe_hash", "win_px", "calibrated", "raw")})
    if run_record is not None:
        runs.append(run_record)
    doc = {"grid_id": grid_id,
           "tiles": sorted(by_tile), "runs": runs,
           "results": [by_tile[t] for t in sorted(by_tile)]}
    write_json_atomic(path, doc)
    return doc
