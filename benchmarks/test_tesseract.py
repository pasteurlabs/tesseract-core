# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for real Tesseract framework overhead.

This module benchmarks actual Tesseract interactions using a no-op Tesseract
that does nothing but decode inputs and encode outputs. This gives realistic
measurements of framework overhead for different interaction modes:

1. Non-containerized via `Tesseract.from_tesseract_api()` - Python-only, no HTTP
2. Containerized via HTTP (`Tesseract.from_image`) - Full container + HTTP stack,
   using json+base64 encoding
3. Containerized via CLI (`tesseract run`) - Full container + CLI overhead,
   using json+binref encoding

The containerized benchmarks (2 and 3) run against both the Docker and Apptainer
backends so their numbers can be compared directly (e.g. Apptainer's faster cold
container startup shows up in the CLI benchmark). The Apptainer variants skip when
Apptainer is unavailable.

All benchmarks use the same no-op Tesseract defined in tesseract_noop/.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from conftest import DEFAULT_ARRAY_SIZES, NOOP_TESSERACT_PATH, create_test_array


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests based on --array-sizes."""
    if "array_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--array-sizes", default=None)
        if raw:
            sizes = [int(s.strip()) for s in raw.split(",")]
        else:
            sizes = DEFAULT_ARRAY_SIZES

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


@pytest.fixture
def http_tesseract_instance(
    tmp_path_factory, use_backend, noop_tesseract_image, request
):
    """Create a containerized HTTP Tesseract for the selected backend.

    Function-scoped (rather than module-scoped) because the backend is chosen per
    parametrization; the image build/convert it depends on is still session-cached.
    """
    from tesseract_core.sdk.tesseract import Tesseract

    if use_backend == "apptainer":
        # Ensure the SIF exists in the store for this backend.
        request.getfixturevalue("noop_sif_image")

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


def test_from_tesseract_api(benchmark, tesseract_api_instance, array_size):
    """Benchmark non-containerized Tesseract via from_tesseract_api()."""
    arr = create_test_array(array_size)
    inputs = {"data": arr}

    benchmark(tesseract_api_instance.apply, inputs)


@pytest.mark.parametrize(
    "use_backend",
    [
        pytest.param("docker", marks=pytest.mark.docker),
        pytest.param("apptainer", marks=pytest.mark.apptainer),
    ],
    indirect=True,
)
def test_containerized_http(benchmark, http_tesseract_instance, array_size):
    """Benchmark containerized Tesseract via HTTP, per backend."""
    arr = create_test_array(array_size)
    inputs = {"data": arr}

    benchmark(http_tesseract_instance.apply, inputs)


@pytest.mark.parametrize(
    "use_backend",
    [
        pytest.param("docker", marks=pytest.mark.docker),
        pytest.param("apptainer", marks=pytest.mark.apptainer),
    ],
    indirect=True,
)
def test_containerized_cli(
    benchmark, use_backend, noop_tesseract_image, noop_sif_image, array_size
):
    """Benchmark containerized Tesseract via CLI (`tesseract run`), per backend."""
    image_ref = noop_sif_image if use_backend == "apptainer" else noop_tesseract_image
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
                    image_ref,
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

        def wait_for_cleanup():
            """Let the runtime fully release resources before the next cold start."""
            time.sleep(2)

        # Each invocation spawns a full container. We want clean cold-start
        # timings, so sleep between rounds to let the runtime clean up.
        benchmark.pedantic(
            run_cli,
            setup=wait_for_cleanup,
            rounds=3,
            warmup_rounds=1,
            iterations=1,
        )
