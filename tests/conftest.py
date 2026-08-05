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


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def read_only_cache(tmp_path):
    """Factory fixture: `read_only_cache(cfg.cache_dir, [subdir, ...]) -> Path`.

    Build a throwaway cache dir that can READ the real caches but never writes to them.

    **R77.** The Stage 1/2/3/4/4b producers take a single `cache_dir` / `output_dir` and use
    it for both input and output, with no dry-run mode. Six tests passed the *live* tree,
    so merely running the suite regenerated gitignored artifacts — twice for real
    (2026-06-10 and 2026-08-04), the second time migrating a v1 image across the y-sign-fix
    correctness boundary. `git` cannot restore any of it.

    Inputs are **hard-linked**, not copied: the CTX tile zips and HiRISE JP2s are hundreds
    of MB and copying them per test is not viable. A hard link is safe here *only because
    the producers' read and write subdirectories are disjoint* — a producer that truncated
    a linked path would write through to the original inode. Pass only subdirs the code
    under test reads and does not write, and keep that invariant if you add one.
    """
    def _make(real_cache, subdirs) -> Path:
        cache = tmp_path / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        for sub in subdirs:
            src_dir = Path(real_cache) / sub
            if not src_dir.is_dir():
                continue
            dst_dir = cache / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if not f.is_file():
                    continue
                try:
                    os.link(f, dst_dir / f.name)
                except OSError:
                    shutil.copy2(f, dst_dir / f.name)  # cross-volume: fall back
        return cache

    return _make
