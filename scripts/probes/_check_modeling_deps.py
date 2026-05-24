"""Check which Week 3 modeling deps are present in the geospatial env."""
from __future__ import annotations

import importlib
import importlib.util

PKGS = ["lightgbm", "torch", "sklearn", "scipy", "pyarrow", "numpy", "pandas"]

for name in PKGS:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"  {name:<12s}  MISSING")
        continue
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:<12s}  {ver}")
    except Exception as e:  # pragma: no cover
        print(f"  {name:<12s}  IMPORT_FAILED  {type(e).__name__}: {e}")
