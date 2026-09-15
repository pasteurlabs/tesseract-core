# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by more than one SDK test module.

Only what `test_local_client.py` (the subprocess transport) and
`test_tesseract.py` (the public API over it) both need. Anything used by one
module lives in that module.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture
def dummy_api_path(dummy_tesseract_package):
    return dummy_tesseract_package / "tesseract_api.py"


def env_without_pythonpath() -> dict[str, str]:
    """Environment safe to hand to a different interpreter.

    Importing a tesseract_api.py in-process puts this interpreter's sys.path on
    PYTHONPATH as a side effect, and any earlier test in the session may have
    done so. A 3.11 interpreter that inherits it picks up 3.13 packages.
    """
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


@pytest.fixture(scope="session")
def built_venv(tmp_path_factory):
    """Build an environment to point `python_executable` at, and return it.

    Two things make a dedicated process worth more than a nicety -- a Tesseract
    can run on an interpreter the caller could not use, and with dependencies
    the caller does not have -- and both are the same fixture: build a venv,
    install the runtime into it, hand back its interpreter.

    Results are cached for the session, since building one costs seconds and
    nothing a test does can change it.

    The returned callable takes `python` (a version to build against, or None
    for the running one) and `requirements` (a tesseract_requirements.txt to
    install as well; specs beginning with "." resolve against its directory, as
    the build would read them).
    """
    uv = shutil.which("uv")
    built: dict[tuple, Path] = {}

    def build(python: str | None = None, requirements: Path | None = None) -> Path:
        key = (python, requirements)
        if key in built:
            return built[key]
        if uv is None:
            pytest.skip("uv is required to build an environment")

        venv_dir = tmp_path_factory.mktemp("venv") / "env"
        env = env_without_pythonpath()

        def run(*args):
            result = subprocess.run(args, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                pytest.skip(
                    f"could not build an environment: {result.stderr.strip()[-300:]}"
                )

        run(uv, "venv", str(venv_dir), *(("--python", python) if python else ()))
        # No version pins beyond the interpreter: CI rewrites the runtime extras
        # to exact pins on its oldest-dependency axis, and anything added here
        # can conflict with them.
        run(uv, "pip", "install", "--python", str(venv_dir), f"{REPO_ROOT}[runtime]")

        if requirements is not None:
            specs = [
                line.strip()
                for line in requirements.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
            for spec in specs:
                resolved = (
                    str(requirements.parent / spec) if spec.startswith(".") else spec
                )
                run(uv, "pip", "install", "--python", str(venv_dir), resolved)

        scripts, exe = (
            ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        )
        built[key] = venv_dir / scripts / exe
        return built[key]

    return build
