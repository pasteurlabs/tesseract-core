# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for testing this SDK and runtime against released versions.

Each version given with ``--compat-version`` is installed into its own
environment with uv, which brings both its runtime and its SDK. No Docker is
needed: the old runtime is served with its ``tesseract-runtime serve``, and the
old SDK runs a client script in a subprocess.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest
import requests


def pytest_generate_tests(metafunc):
    if "compat_version" not in metafunc.fixturenames:
        return
    versions = metafunc.config.getoption("compat_versions")
    if versions:
        params = versions
    else:
        params = [
            pytest.param(
                None,
                marks=pytest.mark.skip(
                    reason="Pass --compat-version X.Y.Z to run compatibility tests"
                ),
            )
        ]
    metafunc.parametrize("compat_version", params, scope="module")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}"
        )


def _isolated_env(**extra: str) -> dict[str, str]:
    """The current environment, minus anything that could import this checkout.

    Loading a tesseract_api.py sets PYTHONPATH, which would let the old
    environment pick up this repository's tesseract_core instead of its own.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def old_python(compat_version, tmp_path_factory) -> Path:
    """Interpreter of an environment with ``tesseract-core[runtime]==compat_version``."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install old tesseract-core releases")

    venv = tmp_path_factory.mktemp(f"tesseract-core-{compat_version}")
    _run([uv, "venv", "--python", sys.executable, str(venv)])
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            f"tesseract-core[runtime]=={compat_version}",
        ]
    )
    return python


@pytest.fixture(scope="module")
def compat_api_path(tmp_path_factory) -> Path:
    """A copy of the dummy Tesseract, outside this checkout."""
    dest = tmp_path_factory.mktemp("compat_tesseract")
    shutil.copytree(
        Path(__file__).parent.parent / "dummy_tesseract", dest, dirs_exist_ok=True
    )
    return dest / "tesseract_api.py"


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def old_runtime(old_python, compat_api_path, tmp_path_factory):
    """Serve the dummy Tesseract with the old runtime.

    Yields the server's URL and its output path, which a client needs to read
    json+binref responses.
    """
    output_path = tmp_path_factory.mktemp("old_runtime_output")
    log_path = output_path.parent / f"{output_path.name}.log"
    runtime = old_python.parent / (
        "tesseract-runtime.exe" if os.name == "nt" else "tesseract-runtime"
    )
    port = _free_port()
    env = _isolated_env(
        TESSERACT_API_PATH=str(compat_api_path),
        TESSERACT_OUTPUT_PATH=str(output_path),
    )

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [runtime, "serve", "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            cwd=output_path,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Runtime exited early:\n{log_path.read_text(errors='replace')}"
                    )
                try:
                    if requests.get(f"{url}/health", timeout=1).status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Runtime did not start in time")
                time.sleep(0.2)

            yield url, output_path
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            print(log_path.read_text(errors="replace"))


@pytest.fixture(scope="module")
def run_old_client(old_python):
    """Run ``old_sdk_client.py`` with the old SDK, and return its JSON output."""
    script = Path(__file__).parent / "old_sdk_client.py"

    def run(url: str, server_output_path: Path, args: dict) -> dict:
        # -I keeps this checkout (the current directory) off sys.path
        result = subprocess.run(
            [old_python, "-I", script, url, str(server_output_path)],
            input=json.dumps(args),
            check=False,
            env=_isolated_env(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Old SDK client failed:\n{result.stdout}\n{result.stderr}"
            )
        return json.loads(result.stdout)

    return run
