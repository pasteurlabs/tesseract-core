# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures and helpers shared by more than one SDK test module."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture
def dummy_api_path(dummy_tesseract_package):
    return dummy_tesseract_package / "tesseract_api.py"


def build_venv(
    dest: Path, python: str | None = None, requirements: Path | None = None
) -> Path:
    """Build an environment to point `python_executable` at, and return it.

    The two things a dedicated process is for -- running a Tesseract on an
    interpreter the caller could not use, and with dependencies the caller does
    not have -- differ only in what goes into the environment, so they share
    this. Callers own `dest`, since a venv with the runtime in it is ~180 MB and
    worth deleting promptly.

    Args:
        dest: Directory to create the environment in.
        python: Version to build against, or None for the running one.
        requirements: A tesseract_requirements.txt to install as well. Specs
            beginning with "." resolve against its directory, as a build would
            read them.

    Returns:
        Path to the environment's interpreter.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build an environment")

    # Importing a tesseract_api.py in-process puts this interpreter's sys.path
    # on PYTHONPATH as a side effect, and any earlier test may have done so. A
    # 3.11 interpreter that inherits it picks up 3.13 packages.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    def run(*args):
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            pytest.skip(
                f"could not build an environment: {result.stderr.strip()[-300:]}"
            )

    run(uv, "venv", str(dest), *(("--python", python) if python else ()))
    # No version pins beyond the interpreter: CI rewrites the runtime extras to
    # exact pins on its oldest-dependency axis, and anything added here can
    # conflict with them.
    run(uv, "pip", "install", "--python", str(dest), f"{REPO_ROOT}[runtime]")

    if requirements is not None:
        for line in requirements.read_text().splitlines():
            spec = line.strip()
            if not spec or spec.startswith("#"):
                continue
            if spec.startswith("."):
                spec = str(requirements.parent / spec)
            run(uv, "pip", "install", "--python", str(dest), spec)

    scripts, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return dest / scripts / exe
