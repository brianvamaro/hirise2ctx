"""Runtime guard: fail loudly if test code opens a repository artifact root for writing.

**R77 (residual).** The Stage 1–4b/5 producers take a single `cache_dir` / `output_dir`
and use it for both input and output, with no dry-run mode. Six tests once passed the
*live* tree, so merely running the suite regenerated gitignored artifacts — twice for
real (2026-06-10, and 2026-08-04 when it migrated a v1 image across the y-sign-fix
correctness boundary). `git` cannot restore any of it.

The 2026-08-05 redirects removed the six known call sites. This module removes the
*class* of defect: while it is installed, any attempt to open, truncate, replace, rename
or delete a path under a repository artifact root raises `LiveArtifactWriteError`
**before** the write reaches the filesystem.

This is prevention. A before/after checksum tells you what you already destroyed; this
refuses the syscall. It is deliberately path-based, which leaves exactly one blind spot:
a *hard link* in a temporary tree is another name for a live inode and is not under any
guarded prefix. That blind spot is closed structurally in `conftest.read_only_cache`,
which copies every mutable derived artifact and hard-links only large immutable source
archives whose called code paths cannot write them.

Not guarded, deliberately:

- directory creation (`os.mkdir`, `Path.mkdir`). `Config.cache_dir` / `Config.output_dir`
  mkdir on attribute access, so guarding it would fail on the mere act of reading config,
  and an empty directory cannot destroy data.
- reads of any kind.
"""
from __future__ import annotations

import builtins
import io
import os
import shutil

from src.config import REPO_ROOT

# Gitignored trees holding artifacts git cannot restore, plus `cache_v2_dev`, which is a
# junction to the live `cache_v2` and therefore is NOT an isolated development cache.
ARTIFACT_ROOT_NAMES = (
    "cache",
    "cache_v2",
    "cache_v2_dev",
    "dataset",
    "dataset_v2",
    "models",
    "reports",
)

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


class LiveArtifactWriteError(AssertionError):
    """A test tried to write inside a repository artifact root."""


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _guarded_roots() -> tuple[str, ...]:
    roots: list[str] = []
    for name in ARTIFACT_ROOT_NAMES:
        p = str(REPO_ROOT / name)
        for candidate in (_norm(p), os.path.normcase(os.path.realpath(p))):
            if candidate not in roots:
                roots.append(candidate)
    return tuple(roots)


_ROOTS = _guarded_roots()
_REPO = _norm(str(REPO_ROOT))


def offending_path(raw) -> str | None:
    """Return the absolute path if `raw` lies inside a guarded root, else None.

    Accepts anything path-like. File descriptors, file objects, buffers and GDAL virtual
    filenames (`/vsicurl/...`) are not paths into the repository and return None.
    """
    if raw is None or isinstance(raw, int):
        return None
    try:
        s = os.fspath(raw)
    except TypeError:
        return None
    if isinstance(s, bytes):
        try:
            s = s.decode()
        except UnicodeDecodeError:
            return None
    if not isinstance(s, str) or not s:
        return None
    absolute = os.path.abspath(s)
    normalized = os.path.normcase(absolute)
    for root in _ROOTS:
        if normalized == root or normalized.startswith(root + os.sep):
            return absolute
    # Junction/symlink escape (this is how `cache_v2_dev` reaches `cache_v2`). Only pay
    # for `realpath` when the path is inside the repo at all.
    if normalized.startswith(_REPO + os.sep):
        try:
            real = os.path.normcase(os.path.realpath(s))
        except (OSError, ValueError):
            return None
        for root in _ROOTS:
            if real == root or real.startswith(root + os.sep):
                return absolute
    return None


def _deny(op: str, path: str, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    raise LiveArtifactWriteError(
        f"R77: test tried to write a live repository artifact via {op}{suffix}:\n"
        f"    {path}\n"
        "These trees are gitignored -- git cannot restore them. Point the producer at an "
        "absolute temporary root (note: a relative path in a copied YAML still resolves "
        "against REPO_ROOT, and cache_v2_dev is a junction to the live cache_v2). "
        "To read the real caches, use the `read_only_cache` fixture."
    )


def _check(op: str, raw, detail: str = "") -> None:
    hit = offending_path(raw)
    if hit is not None:
        _deny(op, hit, detail)


# --------------------------------------------------------------------------- patching

_patches: list[tuple[object, str, object]] = []


def _patch(obj, attr: str, factory) -> None:
    original = getattr(obj, attr)
    setattr(obj, attr, factory(original))
    _patches.append((obj, attr, original))


def _wrap_open(original):
    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            _check("open()", file, f"mode={mode!r}")
        return original(file, mode, *args, **kwargs)

    return guarded_open


def _wrap_os_open(original):
    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS:
            _check("os.open()", path, f"flags=0x{flags:x}")
        return original(path, flags, *args, **kwargs)

    return guarded_os_open


def _wrap_unary(original, label):
    def guarded(path, *args, **kwargs):
        _check(label, path)
        return original(path, *args, **kwargs)

    return guarded


def _wrap_binary(original, label, *, check_src=True):
    def guarded(src, dst, *args, **kwargs):
        if check_src:
            _check(label, src, "source")
        _check(label, dst, "destination")
        return original(src, dst, *args, **kwargs)

    return guarded


def _wrap_rasterio_open(original):
    def guarded_rasterio_open(fp, mode="r", *args, **kwargs):
        if mode != "r":
            _check("rasterio.open()", fp, f"mode={mode!r}")
        return original(fp, mode, *args, **kwargs)

    return guarded_rasterio_open


def _wrap_method_patharg(original, label, index: int, name: str):
    """Guard a bound method whose path argument is at positional `index` (self = 0)."""

    def guarded(*args, **kwargs):
        if name in kwargs:
            _check(label, kwargs[name])
        elif len(args) > index:
            _check(label, args[index])
        return original(*args, **kwargs)

    return guarded


def install() -> None:
    """Patch the write entry points reachable from `src/`. Idempotent."""
    if _patches:
        return

    # `pathlib.Path.open/write_text/write_bytes/touch` all funnel into `io.open` or
    # `os.open`; `Path.unlink/rename/replace` funnel into the `os` functions below.
    _patch(builtins, "open", _wrap_open)
    _patch(io, "open", _wrap_open)
    _patch(os, "open", _wrap_os_open)

    for name in ("remove", "unlink", "rmdir", "truncate"):
        _patch(os, name, lambda orig, n=name: _wrap_unary(orig, f"os.{n}()"))
    for name in ("rename", "replace"):
        _patch(os, name, lambda orig, n=name: _wrap_binary(orig, f"os.{n}()"))
    # A new hard link or symlink *into* a guarded root is a write to that root; linking a
    # live file out to a temporary tree is the read the `read_only_cache` fixture does.
    for name in ("link", "symlink"):
        _patch(os, name, lambda orig, n=name: _wrap_binary(orig, f"os.{n}()", check_src=False))

    for name in ("copyfile", "copy", "copy2", "move"):
        _patch(shutil, name, lambda orig, n=name: _wrap_binary(orig, f"shutil.{n}()", check_src=False))
    _patch(shutil, "rmtree", lambda orig: _wrap_unary(orig, "shutil.rmtree()"))

    import numpy as np

    for name in ("save", "savez", "savez_compressed", "savetxt"):
        _patch(np, name, lambda orig, n=name: _wrap_unary(orig, f"numpy.{n}()"))

    # These write through C/C++ filesystem layers that never reach `builtins.open`.
    import rasterio

    _patch(rasterio, "open", _wrap_rasterio_open)

    import pandas as pd

    _patch(pd.DataFrame, "to_parquet",
           lambda orig: _wrap_method_patharg(orig, "DataFrame.to_parquet()", 1, "path"))
    _patch(pd.DataFrame, "to_feather",
           lambda orig: _wrap_method_patharg(orig, "DataFrame.to_feather()", 1, "path"))

    try:
        import geopandas as gpd
    except ImportError:  # pragma: no cover - geopandas is a hard dependency here
        gpd = None
    if gpd is not None:
        _patch(gpd.GeoDataFrame, "to_file",
               lambda orig: _wrap_method_patharg(orig, "GeoDataFrame.to_file()", 1, "filename"))
        _patch(gpd.GeoDataFrame, "to_parquet",
               lambda orig: _wrap_method_patharg(orig, "GeoDataFrame.to_parquet()", 1, "path"))

    try:
        import pyogrio
    except ImportError:  # pragma: no cover
        pyogrio = None
    if pyogrio is not None:
        _patch(pyogrio, "write_dataframe",
               lambda orig: _wrap_method_patharg(orig, "pyogrio.write_dataframe()", 1, "path"))


def uninstall() -> None:
    while _patches:
        obj, attr, original = _patches.pop()
        setattr(obj, attr, original)


def is_installed() -> bool:
    return bool(_patches)
