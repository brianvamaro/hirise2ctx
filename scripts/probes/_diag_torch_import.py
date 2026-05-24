"""Diagnose torch import failure: enumerate what's in torch/lib and try loading shm.dll explicitly.

This is a Windows-Python-3.14 pain point — torch 2.12 wheels may not yet have full 3.14 support.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    print(f"Python {sys.version}")
    # Find torch package dir without importing
    import importlib.util

    spec = importlib.util.find_spec("torch")
    if spec is None:
        print("torch not installed")
        return 1
    pkg_root = Path(spec.origin).parent
    print(f"torch root: {pkg_root}")
    lib_dir = pkg_root / "lib"
    print(f"torch/lib exists: {lib_dir.exists()}")
    if lib_dir.exists():
        dlls = sorted(p.name for p in lib_dir.glob("*.dll"))
        print(f"  {len(dlls)} DLLs in torch/lib")
        # Show first few + shm.dll specifically
        if "shm.dll" in dlls:
            print("  shm.dll: present")
        else:
            print("  shm.dll: MISSING")

    # Try loading shm.dll explicitly via ctypes to see real error
    import ctypes

    shm_path = lib_dir / "shm.dll"
    if shm_path.exists():
        # Pre-add lib dir to DLL search path (torch normally does this internally)
        try:
            os.add_dll_directory(str(lib_dir))
            print("Added torch/lib to DLL search path")
        except Exception as e:
            print(f"add_dll_directory failed: {e}")
        try:
            ctypes.WinDLL(str(shm_path))
            print("shm.dll loaded directly")
        except OSError as e:
            print(f"shm.dll load FAILED: {e}")

    # Look for fbgemm and other deps shm depends on
    for dep in ["fbgemm.dll", "c10.dll", "torch_cpu.dll", "asmjit.dll", "libomp140.x86_64.dll"]:
        p = lib_dir / dep
        print(f"  {dep}: {'present' if p.exists() else 'missing'}")

    # Try the import again with the explicit add_dll_directory in place
    print("\nAttempting import torch ...")
    try:
        import torch  # noqa: F401

        print(f"  torch {torch.__version__} imported OK")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
