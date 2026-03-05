# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for array encoding and decoding via pydantic model serde.

Tests the performance of json, base64, and binref encoding methods
for arrays of varying sizes, exercising the full pydantic serialization
and validation codepath (model_dump_json / model_validate_json) rather
than calling encode_array / decode_array directly.
"""

from __future__ import annotations

import tempfile

from pydantic import BaseModel
from utils import BenchmarkSuite, create_test_array, run_benchmark

from tesseract_core.runtime.schema_types import Array, Float64

# Default array sizes to benchmark (number of elements)
# Chosen to represent typical workloads from small parameters to large tensors
DEFAULT_ARRAY_SIZES = [
    1,
    10,
    100,
    1000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
]


class ArrayModel(BaseModel):
    """Minimal model with a single variable-length float64 array field."""

    data: Array[(None,), Float64]


def benchmark_encoding(
    iterations: int = 50,
    array_sizes: list[int] | None = None,
    profile: bool = False,
) -> BenchmarkSuite:
    """Run encoding benchmarks for all methods and sizes.

    Args:
        iterations: Number of iterations per benchmark
        array_sizes: Array sizes to benchmark (defaults to DEFAULT_ARRAY_SIZES)
        profile: Whether to profile each invocation with cProfile

    Returns:
        BenchmarkSuite with all results
    """
    if array_sizes is None:
        array_sizes = DEFAULT_ARRAY_SIZES

    suite = BenchmarkSuite(
        name="array_encoding",
        metadata={"iterations": iterations, "array_sizes": array_sizes},
    )

    for i, size in enumerate(array_sizes):
        print(f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}...")
        model = ArrayModel(data=create_test_array(size))

        # JSON encoding
        if size < 20_000_000:  # JSON encoding becomes impractical at very large sizes
            result = run_benchmark(
                name=f"encode_json_{size:,}",
                func=lambda m=model: m.model_dump_json(
                    context={"array_encoding": "json"}
                ),
                iterations=iterations,
                profile=profile,
            )
            suite.add_result(result)

        # Base64 encoding
        result = run_benchmark(
            name=f"encode_base64_{size:,}",
            func=lambda m=model: m.model_dump_json(
                context={"array_encoding": "base64"}
            ),
            iterations=iterations,
            profile=profile,
        )
        suite.add_result(result)

        # Binref encoding (needs temp directory)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_benchmark(
                name=f"encode_binref_{size:,}",
                func=lambda m=model, d=tmpdir: m.model_dump_json(
                    context={"array_encoding": "binref", "base_dir": d}
                ),
                iterations=iterations,
                profile=profile,
            )
            suite.add_result(result)

    return suite


def benchmark_decoding(
    iterations: int = 50,
    array_sizes: list[int] | None = None,
    profile: bool = False,
) -> BenchmarkSuite:
    """Run decoding benchmarks for all methods and sizes.

    Args:
        iterations: Number of iterations per benchmark
        array_sizes: Array sizes to benchmark (defaults to DEFAULT_ARRAY_SIZES)
        profile: Whether to profile each invocation with cProfile

    Returns:
        BenchmarkSuite with all results
    """
    if array_sizes is None:
        array_sizes = DEFAULT_ARRAY_SIZES

    suite = BenchmarkSuite(
        name="array_decoding",
        metadata={"iterations": iterations, "array_sizes": array_sizes},
    )

    for i, size in enumerate(array_sizes):
        print(f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}...")
        model = ArrayModel(data=create_test_array(size))

        # JSON decoding - encode first, then benchmark decoding
        if size < 20_000_000:  # JSON decoding becomes impractical at very large sizes
            json_encoded = model.model_dump_json(context={"array_encoding": "json"})
            result = run_benchmark(
                name=f"decode_json_{size:,}",
                func=lambda s=json_encoded: ArrayModel.model_validate_json(s),
                iterations=iterations,
                profile=profile,
            )
            suite.add_result(result)

        # Base64 decoding
        base64_encoded = model.model_dump_json(context={"array_encoding": "base64"})
        result = run_benchmark(
            name=f"decode_base64_{size:,}",
            func=lambda s=base64_encoded: ArrayModel.model_validate_json(s),
            iterations=iterations,
            profile=profile,
        )
        suite.add_result(result)

        # Binref decoding (needs temp directory with pre-written file)
        with tempfile.TemporaryDirectory() as tmpdir:
            binref_encoded = model.model_dump_json(
                context={"array_encoding": "binref", "base_dir": tmpdir}
            )
            result = run_benchmark(
                name=f"decode_binref_{size:,}",
                func=lambda s=binref_encoded, d=tmpdir: ArrayModel.model_validate_json(
                    s, context={"base_dir": d}
                ),
                iterations=iterations,
                profile=profile,
            )
            suite.add_result(result)

    return suite


def benchmark_roundtrip(
    iterations: int = 50,
    array_sizes: list[int] | None = None,
    profile: bool = False,
) -> BenchmarkSuite:
    """Run roundtrip (encode + decode) benchmarks.

    This measures the full serialization overhead that a Tesseract
    invocation would experience.

    Args:
        iterations: Number of iterations per benchmark
        array_sizes: Array sizes to benchmark (defaults to DEFAULT_ARRAY_SIZES)
        profile: Whether to profile each invocation with cProfile

    Returns:
        BenchmarkSuite with all results
    """
    if array_sizes is None:
        array_sizes = DEFAULT_ARRAY_SIZES

    suite = BenchmarkSuite(
        name="array_roundtrip",
        metadata={"iterations": iterations, "array_sizes": array_sizes},
    )

    for i, size in enumerate(array_sizes):
        print(f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}...")
        model = ArrayModel(data=create_test_array(size))

        # JSON roundtrip
        if size < 20_000_000:  # JSON roundtrip becomes impractical at very large sizes

            def json_roundtrip(m: ArrayModel = model) -> ArrayModel:
                encoded = m.model_dump_json(context={"array_encoding": "json"})
                return ArrayModel.model_validate_json(encoded)

            result = run_benchmark(
                name=f"roundtrip_json_{size:,}",
                func=json_roundtrip,
                iterations=iterations,
                profile=profile,
            )
            suite.add_result(result)

        # Base64 roundtrip
        def base64_roundtrip(m: ArrayModel = model) -> ArrayModel:
            encoded = m.model_dump_json(context={"array_encoding": "base64"})
            return ArrayModel.model_validate_json(encoded)

        result = run_benchmark(
            name=f"roundtrip_base64_{size:,}",
            func=base64_roundtrip,
            iterations=iterations,
            profile=profile,
        )
        suite.add_result(result)

        # Binref roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:

            def binref_roundtrip(
                m: ArrayModel = model, base: str = tmpdir
            ) -> ArrayModel:
                encoded = m.model_dump_json(
                    context={"array_encoding": "binref", "base_dir": base}
                )
                return ArrayModel.model_validate_json(
                    encoded, context={"base_dir": base}
                )

            result = run_benchmark(
                name=f"roundtrip_binref_{size:,}",
                func=binref_roundtrip,
                iterations=iterations,
                profile=profile,
            )
            suite.add_result(result)

    return suite


def run_all(
    iterations: int = 50,
    array_sizes: list[int] | None = None,
    profile: bool = False,
) -> list[BenchmarkSuite]:
    """Run all array encoding benchmarks.

    Args:
        iterations: Number of iterations per benchmark
        array_sizes: Array sizes to benchmark (defaults to DEFAULT_ARRAY_SIZES)
        profile: Whether to profile each invocation with cProfile

    Returns:
        List of BenchmarkSuites for encoding, decoding, and roundtrip
    """
    results = []
    for benchmark_func in [benchmark_encoding, benchmark_decoding, benchmark_roundtrip]:
        print(f"Running {benchmark_func.__name__}...")
        suite = benchmark_func(
            iterations=iterations, array_sizes=array_sizes, profile=profile
        )
        results.append(suite)
    return results


if __name__ == "__main__":
    import sys

    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Running array encoding benchmarks...")
    suites = run_all(iterations)

    for suite in suites:
        print(f"\n=== {suite.name} ===")
        for result in suite.results:
            print(
                f"  {result.name}: {result.mean_time_s * 1000:.3f}ms (±{result.std_time_s * 1000:.3f}ms)"
            )
