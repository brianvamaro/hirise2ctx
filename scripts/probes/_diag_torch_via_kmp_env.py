"""Verify that setting KMP_DUPLICATE_LIB_OK via os.environ inside Python (before any
torch import) is sufficient to make torch loadable."""
from __future__ import annotations

import os
import sys

print(f"KMP_DUPLICATE_LIB_OK at start: {os.environ.get('KMP_DUPLICATE_LIB_OK', '(unset)')}")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print(f"after setdefault:               {os.environ.get('KMP_DUPLICATE_LIB_OK')}")

print("Importing torch ...")
try:
    import torch  # noqa: F401
    print(f"  torch {torch.__version__} imported OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
