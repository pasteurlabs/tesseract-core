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


def build_venv(
    dest: Path, python: str | None = None, requirements: Path | None = None
) -> Path:
    """Build an environment to point `python_executable` at, and return it.

    Built with the same helpers `from_source` uses. If it built environments its
    own way, they could slowly diverge from the real ones, and a test could pass
    against an environment the real resolver would reject.

    Two things stay specific to tests: skipping when uv is missing, and naming a
    Python version, which `from_source` reads from the Tesseract instead.

    Callers own `dest`. A venv with the runtime in it is about 180 MB, so it is
    worth deleting promptly.

    Args:
        dest: Directory to create the environment in.
        python: Version to build against, or None for the running one.
        requirements: A tesseract_requirements.txt to install as well.

    Returns:
        Path to the environment's interpreter.
    """
    if shutil.which("uv") is None:
        pytest.skip("uv is required to build an environment")

    try:
        python_executable = venv_provision._ensure_venv(dest, python)
        if requirements is not None:
            venv_provision._run(
                [
                    *venv_provision._uv(),
                    "pip",
                    "install",
                    "--python",
                    python_executable,
                    *venv_provision._pip_specs(requirements),
                ],
                f"Installing {requirements.name}",
            )
        # No version pins beyond the interpreter: CI rewrites the runtime extras
        # to exact pins on its oldest-dependency axis, and anything added here
        # can conflict with them.
        venv_provision._ensure_runtime(python_executable)
    except RuntimeError as e:
        pytest.skip(f"could not build an environment: {str(e)[-300:]}")

    return python_executable
