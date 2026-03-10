# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for real Tesseract framework overhead.

This module benchmarks actual Tesseract interactions using a no-op Tesseract
that does nothing but decode inputs and encode outputs. This gives realistic
measurements of framework overhead for different interaction modes:

1. Non-containerized via `Tesseract.from_tesseract_api()` - Python-only, no HTTP
2. Containerized via HTTP (`Tesseract.from_image`) - Full Docker + HTTP stack,
   using json+base64 encoding
3. Containerized via CLI (`tesseract run`) - Full Docker + CLI overhead,
   using json+binref encoding

All benchmarks use the same no-op Tesseract defined in tesseract_noop/.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest
from conftest import NOOP_TESSERACT_PATH, create_test_array


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests based on --array-sizes."""
    if "array_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--array-sizes", default=None)
        if raw:
            sizes = [int(s.strip()) for s in raw.split(",")]
        else:
            from conftest import DEFAULT_ARRAY_SIZES

            sizes = DEFAULT_ARRAY_SIZES

        # Filter out very large sizes for containerized benchmarks
        test_name = metafunc.function.__name__
        if test_name in ("test_containerized_http", "test_containerized_cli"):
            sizes = [s for s in sizes if s <= 20_000_000]

        ids = [f"{size:,}" for size in sizes]
        metafunc.parametrize("array_size", sizes, ids=ids)


@pytest.fixture(scope="module")
def tesseract_api_instance(tmp_path_factory):
    """Create a non-containerized Tesseract instance, reused across the module."""
    from tesseract_core.sdk.tesseract import Tesseract

    tmpdir = tmp_path_factory.mktemp("tesseract_api")
    tesseract = Tesseract.from_tesseract_api(
        NOOP_TESSERACT_PATH,
        output_path=tmpdir,
    )
    return tesseract


def test_from_tesseract_api(benchmark, tesseract_api_instance, array_size):
    """Benchmark non-containerized Tesseract via from_tesseract_api()."""
    arr = create_test_array(array_size)
    inputs = {"data": arr}

    benchmark(tesseract_api_instance.apply, inputs)


@pytest.fixture(scope="module")
def http_tesseract_instance(tmp_path_factory, noop_tesseract_image):
    """Create a containerized HTTP Tesseract, reused across the module."""
    from tesseract_core.sdk.tesseract import Tesseract

    tmpdir = tmp_path_factory.mktemp("tesseract_http")
    cm = Tesseract.from_image(
        noop_tesseract_image,
        output_path=tmpdir,
    )
    tesseract = cm.__enter__()
    # Warmup - first request is slow due to container startup
    tesseract.health()
    yield tesseract
    cm.__exit__(None, None, None)


@pytest.mark.docker
def test_containerized_http(benchmark, http_tesseract_instance, array_size):
    """Benchmark containerized Tesseract via HTTP."""
    arr = create_test_array(array_size)
    inputs = {"data": arr}

    benchmark(http_tesseract_instance.apply, inputs)


@pytest.mark.docker
def test_containerized_cli(benchmark, noop_tesseract_image, array_size):
    """Benchmark containerized Tesseract via CLI (`tesseract run`)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        arr = create_test_array(array_size)

        # Write array to binary file for binref encoding
        bin_filename = f"{uuid.uuid4()}.bin"
        bin_path = input_dir / bin_filename
        arr.tofile(bin_path)

        payload = {
            "inputs": {
                "data": {
                    "object_type": "array",
                    "shape": list(arr.shape),
                    "dtype": arr.dtype.name,
                    "data": {
                        "buffer": f"{bin_filename}:0",
                        "encoding": "binref",
                    },
                }
            }
        }

        payload_file = input_dir / f"payload_{array_size}.json"
        payload_file.write_text(json.dumps(payload))

        def run_cli():
            result = subprocess.run(
                [
                    "tesseract",
                    "run",
                    noop_tesseract_image,
                    "apply",
                    f"@{payload_file}",
                    "--input-path",
                    str(input_dir),
                    "--output-path",
                    str(output_dir),
                    "--output-format",
                    "json+binref",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"CLI failed: {result.stderr}")
            return result

        benchmark(run_cli)
