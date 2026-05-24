"""Test whether `import src.modeling` allows subsequent `import torch` to succeed."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("Importing src.modeling (which should run all the DLL setup) ...")
import src.modeling  # noqa: F401
print("  done")

print("Now `import torch` ...")
try:
    import torch
    print(f"  OK -- torch {torch.__version__}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1)
