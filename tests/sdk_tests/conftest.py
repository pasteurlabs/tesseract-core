# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures and helpers shared by more than one SDK test module."""

import shutil
from pathlib import Path

import pytest

from tesseract_core.sdk import venv_provision


@pytest.fixture
def dummy_api_path(dummy_tesseract_package):
    return dummy_tesseract_package / "tesseract_api.py"


def build_venv(dest: Path, python: str | None = None) -> Path:
    """Build an environment with `from_source`'s helpers, and return its python.

    Skips if uv is missing. Callers own `dest` (~180 MB; delete it promptly).

    Args:
        dest: Directory to create the environment in.
        python: Version to build against, or None for uv's default.

    Returns:
        Path to the environment's interpreter.
    """
    if shutil.which("uv") is None:
        pytest.skip("uv is required to build an environment")

    try:
        python_executable = venv_provision._create_venv(dest, python)
        # No version pins beyond the interpreter: CI rewrites the runtime extras
        # to exact pins on its oldest-dependency axis, and anything added here
        # can conflict with them.
        venv_provision._ensure_runtime(python_executable)
    except RuntimeError as e:
        pytest.skip(f"could not build an environment: {str(e)[-300:]}")

    return python_executable
