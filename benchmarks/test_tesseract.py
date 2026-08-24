# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for real Tesseract framework overhead.

This module benchmarks actual Tesseract interactions using a no-op Tesseract
that does nothing but decode inputs and encode outputs. This gives realistic
measurements of framework overhead for different interaction modes:

1. Non-containerized via `Tesseract.from_tesseract_api()` - Python-only, no HTTP
2. Containerized via HTTP (`Tesseract.from_image`) - Full Docker + HTTP stack,
   using json+base64 encoding
3. Containerized via HTTP with json+binref encoding and the binref directory on
   a shared-memory tmpfs (/dev/shm), so array payloads are exchanged through
   shared memory rather than base64 in the HTTP body; uses
   experimental_binref_pool=True for warm-buffer writes and zero-copy mmap decode
4. Containerized via CLI (`tesseract run`) - Full Docker + CLI overhead,
   using json+binref encoding with the binref directory on local disk

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


@pytest.fixture(scope="module")
def http_shmem_tesseract_instance(noop_tesseract_image):
    """Create a containerized HTTP Tesseract that exchanges arrays via shmem.

    Uses json+binref encoding with the input and output directories on a
    shared-memory tmpfs (/dev/shm), so array payloads are passed to and from the
    container through shared memory instead of base64 over HTTP.
    ``experimental_binref_pool`` enables the client-side warm-buffer write pool
    and zero-copy lazy mmap
    decode, which is the recommended configuration for this fast path.
    """
    from tesseract_core.sdk.tesseract import Tesseract

    shm_dir = Path("/dev/shm")
    if not shm_dir.is_dir():
        pytest.skip("/dev/shm is not available on this platform")

    input_dir = Path(tempfile.mkdtemp(prefix="tess_shmem_in_", dir=shm_dir))
    output_dir = Path(tempfile.mkdtemp(prefix="tess_shmem_out_", dir=shm_dir))
    cm = Tesseract.from_image(
        noop_tesseract_image,
        input_path=input_dir,
        output_path=output_dir,
        output_format="json+binref",
        experimental_binref_pool=True,
    )
    tesseract = cm.__enter__()
    # Warmup - first request is slow due to container startup
    tesseract.health()
    yield tesseract, output_dir
    cm.__exit__(None, None, None)


def test_from_tesseract_api(benchmark, tesseract_api_instance, array_size):
    """Benchmark non-containerized Tesseract via from_tesseract_api()."""
    arr = create_test_array(array_size)
    inputs = {"data": arr}

    benchmark(tesseract_api_instance.apply, inputs)


@pytest.mark.docker
def test_containerized_http(benchmark, http_tesseract_instance, array_size):
    """Benchmark containerized Tesseract via HTTP, json+base64 encoding."""
    arr = create_test_array(array_size)
    inputs = {"data": arr}

    benchmark(http_tesseract_instance.apply, inputs)


def _purge_binref_outputs(output_dir: Path) -> None:
    """Remove any output .bin files left in the mounted shmem output directory.

    With ``experimental_binref_pool=True`` the client unlinks each output file
    as soon as it is decoded, so little should remain. This runs untimed via the
    benchmark ``teardown`` as a safety net to keep the small /dev/shm tmpfs from
    filling if any file is left behind across the benchmark's many rounds.
    """
    for binref in output_dir.rglob("*.bin"):
        binref.unlink(missing_ok=True)


@pytest.mark.docker
def test_containerized_http_shmem(benchmark, http_shmem_tesseract_instance, array_size):
    """Benchmark containerized Tesseract via HTTP, json+binref over shared memory.

    Same served-container HTTP path as ``test_containerized_http``, but arrays
    are exchanged as binref files on /dev/shm rather than base64 in the request
    body, so no array data travels over HTTP. Runs with
    ``experimental_binref_pool=True``, the recommended configuration for this
    fast path.
    """
    tesseract, output_dir = http_shmem_tesseract_instance
    arr = create_test_array(array_size)
    inputs = {"data": arr}

    benchmark.pedantic(
        tesseract.apply,
        args=(inputs,),
        teardown=lambda *_: _purge_binref_outputs(output_dir),
        rounds=100,
        warmup_rounds=1,
        iterations=1,
    )


def _run_cli_binref_benchmark(benchmark, noop_tesseract_image, array_size, binref_root):
    """Benchmark a containerized CLI apply with json+binref exchange.

    Writes the input array as a .bin file under ``binref_root`` and runs
    ``tesseract run`` with the binref directory as both input and output path.
    ``binref_root`` selects where those .bin files live: a regular temp
    directory (disk) or a shared-memory tmpfs (/dev/shm).
    """
    with tempfile.TemporaryDirectory(dir=binref_root) as tmpdir:
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

        def before_round():
            """Prepare for the next cold-start round.

            Clears the previous round's binref outputs so they don't accumulate
            in the (possibly small) shared-memory output directory, then lets
            Docker fully release resources before the next cold start.
            """
            for binref in output_dir.rglob("*.bin"):
                binref.unlink(missing_ok=True)
            time.sleep(2)

        # Each invocation spawns a full container. We want clean cold-start
        # timings, so sleep between rounds to let Docker clean up.
        benchmark.pedantic(
            run_cli,
            setup=before_round,
            rounds=3,
            warmup_rounds=1,
            iterations=1,
        )


@pytest.mark.docker
def test_containerized_cli(benchmark, noop_tesseract_image, array_size):
    """Benchmark containerized Tesseract via CLI, binref on disk."""
    _run_cli_binref_benchmark(
        benchmark, noop_tesseract_image, array_size, binref_root=None
    )
