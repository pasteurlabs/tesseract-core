# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and CLI options for benchmarks."""

from __future__ import annotations

import functools
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


@pytest.fixture(autouse=True)
def _require_docker_if_marked(request: pytest.FixtureRequest) -> None:
    """Fail Docker-marked benchmarks when Docker is not available."""
    if "docker" not in request.keywords:
        return
    if not _check_docker():
        pytest.fail("Docker is required for this benchmark but is not available")


@functools.cache
def _check_docker() -> bool:
    try:
        CLIDockerClient().info()
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
