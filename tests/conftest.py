"""Shared pytest fixtures."""
from __future__ import annotations

# Windows torch DLL bootstrap: must run before numpy/MKL is imported (src.config pulls in
# numpy), so torch's shm.dll dependency chain resolves when test_modeling_cnn imports torch
# later in the shared pytest process. See src/modeling/__init__.py + DECISIONS.md 2026-05-27.
import src.modeling  # noqa: F401,E402

from pathlib import Path

import pytest

from src.config import REPO_ROOT, load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config("config.yaml")
