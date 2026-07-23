# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and CLI options for benchmarks."""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tesseract_core.sdk.docker_client import CLIDockerClient

# Path to the no-op tesseract for benchmarking
NOOP_TESSERACT_PATH = Path(__file__).parent / "tesseract_noop" / "tesseract_api.py"

# Default array sizes when --array-sizes is not specified.
DEFAULT_ARRAY_SIZES = [1000, 100_000, 10_000_000]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--array-sizes",
        default=None,
        help="Comma-separated array sizes (e.g. '100,10000,1000000')",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "docker: requires Docker")
    config.addinivalue_line("markers", "apptainer: requires Apptainer")


@pytest.fixture(autouse=True)
def _require_backend_if_marked(request: pytest.FixtureRequest) -> None:
    """Fail/skip container-marked benchmarks when the backend is unavailable.

    Docker-marked benchmarks fail loudly (Docker is a hard requirement in CI);
    Apptainer-marked benchmarks skip when Apptainer is absent, since it is only
    installed on the dedicated benchmark leg.
    """
    if "docker" in request.keywords and not _check_docker():
        pytest.fail("Docker is required for this benchmark but is not available")
    if "apptainer" in request.keywords and not _check_apptainer():
        pytest.skip("Apptainer is not available")


@functools.cache
def _check_docker() -> bool:
    try:
        CLIDockerClient().info()
        return True
    except Exception:
        return False


@functools.cache
def _check_apptainer() -> bool:
    if shutil.which("apptainer") is None:
        return False
    try:
        subprocess.run(["apptainer", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def array_sizes(request: pytest.FixtureRequest) -> list[int]:
    """Array sizes to benchmark, from --array-sizes or defaults."""
    raw = request.config.getoption("--array-sizes")
    if raw:
        return [int(s.strip()) for s in raw.split(",")]
    return DEFAULT_ARRAY_SIZES


def create_test_array(size: int, dtype: str = "float64") -> np.ndarray:
    """Create a random test array of given size."""
    return np.random.default_rng(42).standard_normal(size).astype(dtype)


@pytest.fixture(scope="session")
def noop_tesseract_image() -> str | None:
    """Build the no-op tesseract image once per session. Returns image name or None."""
    tesseract_dir = NOOP_TESSERACT_PATH.parent
    image_name = "benchmark-noop:latest"
    result = subprocess.run(
        [
            "tesseract",
            "build",
            str(tesseract_dir),
            "--config-override",
            "name=benchmark-noop",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build noop tesseract: {result.stderr}")
    return image_name


# SIF store location for benchmarks, shared between the in-process config and the
# subprocess-based CLI benchmark (which inherits it via the environment).
BENCH_SIF_STORE = Path(__file__).parent / ".apptainer-store"


@pytest.fixture(scope="session")
def noop_sif_image(noop_tesseract_image) -> str | None:
    """Convert the no-op Docker image into a SIF store once per session.

    Returns the store reference (``benchmark-noop:latest``). Requires both Docker
    (to build, via ``noop_tesseract_image``) and Apptainer (to convert); skips if
    Apptainer is unavailable. The store dir is exported via
    ``TESSERACT_APPTAINER_IMAGE_DIR`` so the subprocess CLI benchmark sees it too.
    """
    if not _check_apptainer():
        pytest.skip("Apptainer is not available")
    import os

    from tesseract_core.sdk import config, engine

    if not hasattr(engine, "build_sif"):
        # Older Tesseract without the Apptainer backend (e.g. the baseline branch
        # in the benchmark comparison run). Skip rather than error.
        pytest.skip("This Tesseract version has no Apptainer backend")

    BENCH_SIF_STORE.mkdir(parents=True, exist_ok=True)
    os.environ["TESSERACT_APPTAINER_IMAGE_DIR"] = str(BENCH_SIF_STORE)
    config.update_config(apptainer_image_dir=str(BENCH_SIF_STORE))
    engine.build_sif("benchmark-noop", "latest")
    return "benchmark-noop:latest"


@pytest.fixture
def use_backend(request: pytest.FixtureRequest, monkeypatch):
    """Select the container backend for a parametrized benchmark.

    Used with ``@pytest.mark.parametrize("use_backend", ["docker", "apptainer"],
    indirect=True)``. Sets ``TESSERACT_CONTAINER_BACKEND`` and refreshes the runtime
    config so ``Tesseract.from_image`` / ``tesseract run`` pick up the backend.
    """
    backend = request.param
    from tesseract_core.sdk import config

    if "container_backend" not in config.RuntimeConfig.model_fields:
        # Older Tesseract without the backend abstraction (baseline branch): only
        # Docker exists. Skip non-docker params; run docker ones unchanged.
        if backend != "docker":
            pytest.skip("This Tesseract version has no container backend selection")
        yield backend
        return

    monkeypatch.setenv("TESSERACT_CONTAINER_BACKEND", backend)
    config.update_config(container_backend=backend)
    yield backend
    config.update_config(container_backend="docker")
