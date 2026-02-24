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
import sys
import tempfile
from pathlib import Path

import numpy as np
from utils import BenchmarkSuite, run_benchmark

# Path to the no-op tesseract for benchmarking
NOOP_TESSERACT_PATH = Path(__file__).parent / "tesseract_noop" / "tesseract_api.py"

# Array sizes to benchmark
ARRAY_SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]


def _create_test_array(size: int) -> np.ndarray:
    """Create a test array of given size."""
    return np.random.default_rng().standard_normal(size).astype("float64")


def _build_noop_tesseract() -> str | None:
    """Build the no-op tesseract image for benchmarking.

    Returns the image name, or None if build fails.
    """
    tesseract_dir = NOOP_TESSERACT_PATH.parent
    image_name = "benchmark-noop:latest"

    try:
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
            print(f"Build failed: {result.stderr}")
            return None
        return image_name
    except Exception as e:
        print(f"Build error: {e}")
        return None


def benchmark_from_tesseract_api(
    iterations: int = 50, max_size: int | None = None
) -> BenchmarkSuite:
    """Benchmark non-containerized Tesseract via from_tesseract_api().

    This is the fastest path as it bypasses Docker and HTTP entirely,
    using direct Python calls to the Tesseract API.
    """
    from tesseract_core.sdk.tesseract import Tesseract

    array_sizes = [s for s in ARRAY_SIZES if max_size is None or s <= max_size]

    suite = BenchmarkSuite(
        name="from_tesseract_api",
        metadata={"iterations": iterations, "array_sizes": array_sizes},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tesseract once (includes schema generation overhead)
        tesseract = Tesseract.from_tesseract_api(
            NOOP_TESSERACT_PATH,
            output_path=Path(tmpdir),
        )

        for i, size in enumerate(array_sizes):
            print(f"  [{i + 1}/{len(ARRAY_SIZES)}] Benchmarking size {size:,}...")
            arr = _create_test_array(size)
            inputs = {"data": arr}

            def call_apply(t=tesseract, inp=inputs):
                return t.apply(inp)

            result = run_benchmark(
                name=f"apply_{size:,}",
                func=call_apply,
                iterations=iterations,
            )
            suite.add_result(result)

    return suite


def benchmark_containerized_http(
    iterations: int = 20, max_size: int | None = None
) -> BenchmarkSuite | None:
    """Benchmark containerized Tesseract via HTTP (Tesseract.from_image).

    This measures the full stack: Docker container, HTTP server, and
    all serialization overhead using json+base64 encoding.

    Requires Docker to be available. Returns None if Docker is not available.
    """
    try:
        from tesseract_core.sdk.docker_client import CLIDockerClient
        from tesseract_core.sdk.tesseract import Tesseract

        client = CLIDockerClient()
        # Quick check if Docker is available
        client.info()
    except Exception:
        print("Docker not available, skipping containerized HTTP benchmarks")
        return None

    array_sizes = [s for s in ARRAY_SIZES if max_size is None or s <= max_size]

    suite = BenchmarkSuite(
        name="containerized_http",
        metadata={"iterations": iterations, "array_sizes": array_sizes},
    )

    # Build the benchmark tesseract image
    image_name = _build_noop_tesseract()
    if image_name is None:
        print(
            "Failed to build benchmark tesseract, skipping containerized HTTP benchmarks"
        )
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with Tesseract.from_image(
                image_name, output_path=Path(tmpdir)
            ) as tesseract:
                # Warmup - first request is slow due to container startup
                _ = tesseract.health()

                for i, size in enumerate(array_sizes):
                    if size > 20_000_000:
                        continue  # Skip largest sizes for HTTP benchmarks due to long runtimes

                    print(
                        f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}..."
                    )
                    arr = _create_test_array(size)
                    inputs = {"data": arr}

                    def call_apply(t=tesseract, inp=inputs):
                        return t.apply(inp)

                    result = run_benchmark(
                        name=f"apply_{size:,}",
                        func=call_apply,
                        iterations=iterations,
                        warmup=2,
                    )
                    suite.add_result(result)

    except Exception as e:
        print(f"Error running containerized HTTP benchmarks: {e}")
        return None

    return suite


def benchmark_containerized_cli(
    iterations: int = 10, max_size: int | None = None
) -> BenchmarkSuite | None:
    """Benchmark containerized Tesseract via CLI (`tesseract run`).

    This measures the overhead of invoking Tesseract via the CLI,
    which includes process spawn overhead on top of Docker + HTTP.
    Uses json+binref encoding for both inputs and outputs.

    Requires Docker to be available. Returns None if Docker is not available.
    """
    import uuid

    try:
        from tesseract_core.sdk.docker_client import CLIDockerClient

        client = CLIDockerClient()
        client.info()
    except Exception:
        print("Docker not available, skipping containerized CLI benchmarks")
        return None

    array_sizes = [s for s in ARRAY_SIZES if max_size is None or s <= max_size]

    suite = BenchmarkSuite(
        name="containerized_cli",
        metadata={"iterations": iterations, "array_sizes": array_sizes},
    )

    image_name = _build_noop_tesseract()
    if image_name is None:
        print(
            "Failed to build benchmark tesseract, skipping containerized CLI benchmarks"
        )
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create separate input and output directories to avoid duplicate volume error
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for i, size in enumerate(array_sizes):
            print(f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}...")
            arr = _create_test_array(size)

            # Write array to binary file for binref encoding
            bin_filename = f"{uuid.uuid4()}.bin"
            bin_path = input_dir / bin_filename
            arr.tofile(bin_path)

            # Use binref encoding for input
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

            # Write payload to file
            payload_file = input_dir / f"payload_{size}.json"
            payload_file.write_text(json.dumps(payload))

            def run_cli(
                img=image_name, pf=payload_file, indir=input_dir, outdir=output_dir
            ):
                result = subprocess.run(
                    [
                        "tesseract",
                        "run",
                        img,
                        "apply",
                        f"@{pf}",
                        "--input-path",
                        str(indir),
                        "--output-path",
                        str(outdir),
                        "--output-format",
                        "json+binref",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"CLI failed: {result.stderr}")
                return result

            result = run_benchmark(
                name=f"apply_{size:,}",
                func=run_cli,
                iterations=iterations,
                warmup=1,
            )
            suite.add_result(result)

    return suite


def run_all(
    iterations: int = 50,
    include_containerized: bool = False,
    max_size: int | None = None,
) -> list[BenchmarkSuite]:
    """Run all Tesseract benchmarks.

    Args:
        iterations: Number of iterations per benchmark
        include_containerized: Whether to include Docker-based benchmarks
        max_size: Maximum array size to benchmark (None for no limit)

    Returns:
        List of BenchmarkSuites
    """
    benchmark_funcs = [benchmark_from_tesseract_api]
    if include_containerized:
        benchmark_funcs.extend(
            [benchmark_containerized_http, benchmark_containerized_cli]
        )

    results = []
    for benchmark_func in benchmark_funcs:
        print(f"Running {benchmark_func.__name__}...")
        suite = benchmark_func(iterations=iterations, max_size=max_size)
        results.append(suite)
    return results


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    exclude_docker = "--no-docker" in sys.argv

    print(
        f"Running Tesseract benchmarks (iterations={iterations}, docker={not exclude_docker})..."
    )
    suites = run_all(iterations, include_containerized=not exclude_docker)

    for suite in suites:
        print(f"\n=== {suite.name} ===")
        for result in suite.results:
            print(
                f"  {result.name}: {result.mean_time_s * 1000:.3f}ms (±{result.std_time_s * 1000:.3f}ms)"
            )
