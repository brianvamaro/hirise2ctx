"""Shared pytest fixtures."""
from __future__ import annotations

# Windows torch DLL bootstrap: must run before numpy/MKL is imported (src.config pulls in
# numpy), so torch's shm.dll dependency chain resolves when test_modeling_cnn imports torch
# later in the shared pytest process. See src/modeling/__init__.py + DECISIONS.md 2026-05-27.
import src.modeling  # noqa: F401,E402

import os
import shutil
from pathlib import Path

import pytest

from src.config import REPO_ROOT, load_config

from . import live_artifact_guard

# Cache subdirs that may be HARD-LINKED into a staging tree instead of copied.
#
# A hard link is a second name for the *live inode*: anything that truncates, rewrites or
# replaces-in-place the staged name writes straight through to the repository's file. That
# is acceptable only for large immutable source archives whose called code paths provably
# cannot write, replace or invalidate them:
#
#   ctx_tiles/   `ctx_retrieve.ensure_tile_cached` downloads `{tile}.zip` only when the path
#                does not exist and writes `{tile}.json` only when the sidecar does not
#                exist; staging both means neither branch is reachable. Everything else
#                reads through `/vsizip/`, which is read-only. ~44 GB.
#   hirise_jp2/  `hirise_imagery.ensure_jp2_local` returns early when the file exists and is
#                larger than 1 MB (staged JP2s are ~430 MB). Its download path writes a
#                `.partial` sibling and `Path.replace`s it, which swaps a directory entry
#                and would not touch the original inode even if it did run. ~21 GB.
#
# `hirise_decimated/` is deliberately NOT here. That was the residual R77 hole found by the
# 2026-08-06 audit: `read_full_footprint_decimated` reopens that exact path with rasterio
# `"w"` whenever the cached CRS disagrees with the Stage 1 corrected CRS. `ctx_windows/` and
# `reprojected_detections/` are derived products too, and are copied for the same reason.
LINKABLE_IMMUTABLE_SUBDIRS = frozenset({"ctx_tiles", "hirise_jp2"})

# ...and inside those directories, only the archives themselves. A `.aux.xml` is a GDAL PAM
# sidecar: derived, rewritten in place whenever GDAL decides its metadata is dirty, and
# tiny. Copy it.
LINKABLE_ARCHIVE_SUFFIXES = frozenset({".zip", ".jp2"})


@pytest.fixture(scope="session", autouse=True)
def _no_live_artifact_writes():
    """Refuse, for the whole session, any write into a repository artifact root.

    R77's redirects fixed six known call sites; this closes the class. See
    `tests/live_artifact_guard.py` for what is and is not covered.
    """
    live_artifact_guard.install()
    try:
        yield
    finally:
        live_artifact_guard.uninstall()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def read_only_cache(tmp_path):
    """Factory: `read_only_cache(cfg.cache_dir, [subdir, ...], only="ESP_...") -> Path`.

    Stage a throwaway cache that can READ the real caches and can never write to them.

    **R77.** The Stage 1/2/3/4/4b producers take a single `cache_dir` / `output_dir` and
    use it for both input and output, with no dry-run mode. Six tests once passed the live
    tree, so merely running the suite regenerated gitignored artifacts — twice for real.

    Everything is **copied** except the GB-scale archives themselves — files whose suffix
    is in `LINKABLE_ARCHIVE_SUFFIXES` inside a `LINKABLE_IMMUTABLE_SUBDIRS` directory. See
    those constants for the per-directory argument; sidecars living beside an archive
    (`{tile}.json`, `.aux.xml`) are derived and are copied like everything else.

    `only=` (a string, or an iterable of strings, matched as a substring of the filename)
    restricts what gets copied — pass every key the code under test needs, e.g.
    `only=[OBS_ID, TILE_NAME]` when it also reads a Murray-tile-keyed sidecar. Linked
    archives ignore the filter; links are cheap.

    Two post-conditions are enforced automatically: every copied file is a distinct inode
    from its source, and every linked source file is unchanged in size and mtime when the
    test finishes.
    """
    linked_sources: list[tuple[Path, int, int]] = []
    staged_count = 0

    def _wanted(name: str, only) -> bool:
        if only is None:
            return True
        needles = [only] if isinstance(only, str) else list(only)
        return any(n in name for n in needles)

    def _make(real_cache, subdirs, *, only=None) -> Path:
        nonlocal staged_count
        staged_count += 1
        # Each call gets its own root: two staged caches inside one test must not share
        # state, and re-linking an already-staged archive would fail.
        cache = tmp_path / ("cache" if staged_count == 1 else f"cache_{staged_count}")
        cache.mkdir(parents=True, exist_ok=True)
        for sub in subdirs:
            src_dir = Path(real_cache) / sub
            if not src_dir.is_dir():
                continue
            archive_dir = sub in LINKABLE_IMMUTABLE_SUBDIRS
            dst_dir = cache / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if not f.is_file():
                    continue
                linkable = archive_dir and f.suffix.lower() in LINKABLE_ARCHIVE_SUFFIXES
                # `only` filters copies; linked archives are cheap and their names
                # (Murray tile / JP2) do not carry the ObsId being filtered on.
                if not linkable and not _wanted(f.name, only):
                    continue
                dst = dst_dir / f.name
                if linkable:
                    try:
                        os.link(f, dst)
                        st = f.stat()
                        linked_sources.append((f, st.st_size, st.st_mtime_ns))
                        continue
                    except OSError:
                        pass  # cross-volume or link limit: fall through to a copy
                shutil.copy2(f, dst)
                assert not dst.samefile(f), (
                    f"R77: {dst} is the same inode as the live {f}; a producer that "
                    "rewrites it would write through to the repository."
                )
        return cache

    yield _make

    for source, size, mtime_ns in linked_sources:
        st = source.stat()
        assert (st.st_size, st.st_mtime_ns) == (size, mtime_ns), (
            f"R77: the hard-linked source {source} changed during the test "
            f"({size} B/{mtime_ns} -> {st.st_size} B/{st.st_mtime_ns}). Something wrote "
            f"through the link into a live artifact; move '{source.parent.name}' out of "
            "LINKABLE_IMMUTABLE_SUBDIRS."
        )
