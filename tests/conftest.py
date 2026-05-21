"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import REPO_ROOT, load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config("config.yaml")
