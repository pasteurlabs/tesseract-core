# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the SDK tests.

Chiefly the pieces needed to serve a Tesseract without a container, which both
`test_local_client.py` (the transport) and `test_tesseract.py` (the public API
over it) need.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def dummy_api_path(dummy_tesseract_package):
    return dummy_tesseract_package / "tesseract_api.py"


@pytest.fixture
def sample_inputs():
    return {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3.0, 4.0], dtype=np.float32),
        "s": 2,
    }


def _env_without_pythonpath() -> dict[str, str]:
    """Environment safe to hand to a different interpreter.

    Importing a tesseract_api.py in-process puts this interpreter's sys.path on
    PYTHONPATH as a side effect, and any earlier test in the session may have
    done so. A 3.11 interpreter that inherits it picks up 3.13 packages.
    """
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


@pytest.fixture
def env_without_pythonpath() -> dict[str, str]:
    """`_env_without_pythonpath` as a fixture, for tests that spawn interpreters."""
    return _env_without_pythonpath()


@pytest.fixture(scope="session")
def foreign_venv(tmp_path_factory):
    """A separate environment running a different Python from this one.

    This is what makes subprocess isolation worth more than a nicety: the
    Tesseract need not be installable alongside the caller. Deliberately no
    version pins beyond the interpreter -- CI rewrites the runtime extras to
    exact pins on its oldest-dependency axis, so anything we add here can
    conflict with them.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build a foreign environment")

    # Any supported version that is not the one running the tests.
    ours = f"{sys.version_info.major}.{sys.version_info.minor}"
    foreign = next(v for v in ("3.12", "3.11", "3.13") if v != ours)

    venv_dir = tmp_path_factory.mktemp("foreign_venv") / "env"
    repo_root = Path(__file__).parents[2]
    env = _env_without_pythonpath()

    def run(*args):
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            pytest.skip(
                f"could not build a Python {foreign} environment: "
                f"{result.stderr.strip()[-300:]}"
            )

    run(uv, "venv", str(venv_dir), "--python", foreign)
    run(uv, "pip", "install", "--python", str(venv_dir), f"{repo_root}[runtime]")

    scripts, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return venv_dir / scripts / exe, foreign
