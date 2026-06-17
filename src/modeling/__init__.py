"""Week 3 modeling package.

Implements the LightGBM (Tweedie / log1p+Huber / two-stage hurdle) and small-CNN
baselines defined in PLAN_modeling.md, both run under a shared leave-one-image-out
cross-validation harness with Spearman rho as the primary metric.

This module sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time so PyTorch can coexist
with numpy/scipy's MKL OpenMP runtime in the `geospatial` conda env (DECISIONS.md
2026-05-27). The env var must be set before torch is imported anywhere; placing it
here -- the package init -- guarantees that any module that uses `src.modeling`
gets the fix automatically.
"""
from __future__ import annotations

import importlib.util as _ilu
import os as _os
from pathlib import Path as _Path

# Windows + Python 3.14 + torch 2.12 wheel: torch's own DLL-search-path initialization
# does not find the libraries shm.dll depends on (libomp, c10, torch_cpu) when the
# Python interpreter was launched via `conda run`. Symptom: `OSError: [WinError 127]
# The specified procedure could not be found. Error loading
# .../torch/lib/shm.dll`. Manually adding torch/lib to the search path via
# os.add_dll_directory BEFORE torch import sidesteps it. Verified via
# scripts/probes/_diag_torch_import.py.
#
# Separately, the conda env carries TWO OpenMP runtimes (numpy MKL's libiomp5md.dll
# + torch's libomp.dll). KMP_DUPLICATE_LIB_OK=TRUE lets them coexist; without it the
# second-to-load OMP runtime aborts the interpreter. Must be set BEFORE any DLL
# containing an OMP runtime is dlopened, which is why both lines run at package
# init time.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# The DLL-search-path workaround below is Windows-only: `os.add_dll_directory` and
# `ctypes.WinDLL` exist only on Windows, and the shm.dll problem is a Windows-loader
# issue. On Linux (e.g. Sherlock) torch's .so loading works normally, so this is a
# no-op there -- guard it rather than crash with `os has no attribute add_dll_directory`.
if _os.name == "nt":
    _spec = _ilu.find_spec("torch")
    if _spec is not None and _spec.origin is not None:
        _torch_lib_dir = _Path(_spec.origin).parent / "lib"
        if _torch_lib_dir.exists():
            _os.add_dll_directory(str(_torch_lib_dir))
            # Explicit shm.dll preload populates its dependency chain into the process so
            # torch's own _load_dll_libraries() finds them at import time. add_dll_directory
            # alone is necessary but not sufficient on this env; the ctypes preload is the
            # actual fix verified in scripts/probes/_diag_torch_import.py.
            import ctypes as _ctypes

            _shm = _torch_lib_dir / "shm.dll"
            if _shm.exists():
                try:
                    _ctypes.WinDLL(str(_shm))
                except OSError:
                    pass

__all__ = []  # populated by sub-modules at use time
