# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for array encoding and decoding.

Tests the performance of json, base64, and binref encoding methods
for arrays of varying sizes using the actual encode_array and decode_array
functions from tesseract_core.runtime.array_encoding.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

import numpy as np
from utils import BenchmarkSuite, run_benchmark

from tesseract_core.runtime.array_encoding import (
    decode_array,
    encode_array,
)

# Array sizes to benchmark (number of elements)
# Chosen to represent typical workloads from small parameters to large tensors
ARRAY_SIZES = [
    10,  # Tiny: single parameters, small configs
    100,  # Small: small vectors, configurations
    1_000,  # Medium: typical vectors
    10_000,  # Large: larger vectors, small matrices
    100_000,  # Very large: medium tensors
    1_000_000,  # Huge: large tensors (1M elements)
    10_000_000,  # Massive: very large tensors (10M elements)
    100_000_000,  # Ginormous: very large tensors (100M elements)
]

# Expected shape and dtype for benchmarks (variable-length 1D float64 arrays)
EXPECTED_SHAPE = (None,)
EXPECTED_DTYPE = "float64"


@dataclass
class FakeSerializationInfo:
    """Fake info object for encode_array (serialization)."""

    context: dict = field(default_factory=dict)

    def mode_is_json(self) -> bool:
        return True


@dataclass
class FakeValidationInfo:
    """Fake info object for decode_array (validation/deserialization)."""

    context: dict = field(default_factory=dict)


def _create_test_array(size: int, dtype: str = "float64") -> np.ndarray:
    """Create a test array of given size."""
    return np.random.default_rng().standard_normal(size).astype(dtype)


def benchmark_encoding(
    iterations: int = 50,
) -> BenchmarkSuite:
    """Run encoding benchmarks for all methods and sizes.

    Args:
        iterations: Number of iterations per benchmark

    Returns:
        BenchmarkSuite with all results
    """
    suite = BenchmarkSuite(
        name="array_encoding",
        metadata={"iterations": iterations, "array_sizes": ARRAY_SIZES},
    )

    for i, size in enumerate(ARRAY_SIZES):
        print(f"  [{i + 1}/{len(ARRAY_SIZES)}] Benchmarking size {size:,}...")
        arr = _create_test_array(size)

        # JSON encoding
        if size < 20_000_000:  # JSON encoding becomes impractical at very large sizes
            json_info = FakeSerializationInfo(context={"array_encoding": "json"})
            result = run_benchmark(
                name=f"encode_json_{size:,}",
                func=lambda a=arr, i=json_info: encode_array(
                    a, i, EXPECTED_SHAPE, EXPECTED_DTYPE
                ),
                iterations=iterations,
            )
            suite.add_result(result)

        # Base64 encoding
        base64_info = FakeSerializationInfo(context={"array_encoding": "base64"})
        result = run_benchmark(
            name=f"encode_base64_{size:,}",
            func=lambda a=arr, i=base64_info: encode_array(
                a, i, EXPECTED_SHAPE, EXPECTED_DTYPE
            ),
            iterations=iterations,
        )
        suite.add_result(result)

        # Binref encoding (needs temp directory)
        with tempfile.TemporaryDirectory() as tmpdir:
            binref_info = FakeSerializationInfo(
                context={"array_encoding": "binref", "base_dir": tmpdir}
            )
            result = run_benchmark(
                name=f"encode_binref_{size:,}",
                func=lambda a=arr, i=binref_info: encode_array(
                    a, i, EXPECTED_SHAPE, EXPECTED_DTYPE
                ),
                iterations=iterations,
            )
            suite.add_result(result)

    return suite


def benchmark_decoding(
    iterations: int = 50,
) -> BenchmarkSuite:
    """Run decoding benchmarks for all methods and sizes.

    Args:
        iterations: Number of iterations per benchmark

    Returns:
        BenchmarkSuite with all results
    """
    suite = BenchmarkSuite(
        name="array_decoding",
        metadata={"iterations": iterations, "array_sizes": ARRAY_SIZES},
    )

    for i, size in enumerate(ARRAY_SIZES):
        print(f"  [{i + 1}/{len(ARRAY_SIZES)}] Benchmarking size {size:,}...")
        arr = _create_test_array(size)

        # JSON decoding - encode first, then benchmark decoding
        if size < 20_000_000:  # JSON decoding becomes impractical at very large sizes
            json_encode_info = FakeSerializationInfo(context={"array_encoding": "json"})
            json_encoded = encode_array(
                arr, json_encode_info, EXPECTED_SHAPE, EXPECTED_DTYPE
            )
            json_decode_info = FakeValidationInfo(context={})

            result = run_benchmark(
                name=f"decode_json_{size:,}",
                func=lambda m=json_encoded, i=json_decode_info: decode_array(
                    m, i, EXPECTED_SHAPE, EXPECTED_DTYPE
                ),
                iterations=iterations,
            )
            suite.add_result(result)

        # Base64 decoding
        base64_encode_info = FakeSerializationInfo(context={"array_encoding": "base64"})
        base64_encoded = encode_array(
            arr, base64_encode_info, EXPECTED_SHAPE, EXPECTED_DTYPE
        )
        base64_decode_info = FakeValidationInfo(context={})

        result = run_benchmark(
            name=f"decode_base64_{size:,}",
            func=lambda m=base64_encoded, i=base64_decode_info: decode_array(
                m, i, EXPECTED_SHAPE, EXPECTED_DTYPE
            ),
            iterations=iterations,
        )
        suite.add_result(result)

        # Binref decoding (needs temp directory with pre-written file)
        with tempfile.TemporaryDirectory() as tmpdir:
            binref_encode_info = FakeSerializationInfo(
                context={"array_encoding": "binref", "base_dir": tmpdir}
            )
            binref_encoded = encode_array(
                arr, binref_encode_info, EXPECTED_SHAPE, EXPECTED_DTYPE
            )
            binref_decode_info = FakeValidationInfo(context={"base_dir": tmpdir})

            result = run_benchmark(
                name=f"decode_binref_{size:,}",
                func=lambda m=binref_encoded, i=binref_decode_info: decode_array(
                    m, i, EXPECTED_SHAPE, EXPECTED_DTYPE
                ),
                iterations=iterations,
            )
            suite.add_result(result)

    return suite


def benchmark_roundtrip(
    iterations: int = 50,
) -> BenchmarkSuite:
    """Run roundtrip (encode + decode) benchmarks.

    This measures the full serialization overhead that a Tesseract
    invocation would experience.

    Args:
        iterations: Number of iterations per benchmark

    Returns:
        BenchmarkSuite with all results
    """
    suite = BenchmarkSuite(
        name="array_roundtrip",
        metadata={"iterations": iterations, "array_sizes": ARRAY_SIZES},
    )

    for i, size in enumerate(ARRAY_SIZES):
        print(f"  [{i + 1}/{len(ARRAY_SIZES)}] Benchmarking size {size:,}...")
        arr = _create_test_array(size)

        # JSON roundtrip
        if size < 20_000_000:  # JSON roundtrip becomes impractical at very large sizes

            def json_roundtrip(a: np.ndarray = arr) -> np.ndarray:
                encode_info = FakeSerializationInfo(context={"array_encoding": "json"})
                encoded = encode_array(a, encode_info, EXPECTED_SHAPE, EXPECTED_DTYPE)
                decode_info = FakeValidationInfo(context={})
                return decode_array(
                    encoded, decode_info, EXPECTED_SHAPE, EXPECTED_DTYPE
                )

            result = run_benchmark(
                name=f"roundtrip_json_{size:,}",
                func=json_roundtrip,
                iterations=iterations,
            )
            suite.add_result(result)

        # Base64 roundtrip
        def base64_roundtrip(a: np.ndarray = arr) -> np.ndarray:
            encode_info = FakeSerializationInfo(context={"array_encoding": "base64"})
            encoded = encode_array(a, encode_info, EXPECTED_SHAPE, EXPECTED_DTYPE)
            decode_info = FakeValidationInfo(context={})
            return decode_array(encoded, decode_info, EXPECTED_SHAPE, EXPECTED_DTYPE)

        result = run_benchmark(
            name=f"roundtrip_base64_{size:,}",
            func=base64_roundtrip,
            iterations=iterations,
        )
        suite.add_result(result)

        # Binref roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:

            def binref_roundtrip(a: np.ndarray = arr, base: str = tmpdir) -> np.ndarray:
                encode_info = FakeSerializationInfo(
                    context={"array_encoding": "binref", "base_dir": base}
                )
                encoded = encode_array(a, encode_info, EXPECTED_SHAPE, EXPECTED_DTYPE)
                decode_info = FakeValidationInfo(context={"base_dir": base})
                return decode_array(
                    encoded, decode_info, EXPECTED_SHAPE, EXPECTED_DTYPE
                )

            result = run_benchmark(
                name=f"roundtrip_binref_{size:,}",
                func=binref_roundtrip,
                iterations=iterations,
            )
            suite.add_result(result)

    return suite


def run_all(iterations: int = 50) -> list[BenchmarkSuite]:
    """Run all array encoding benchmarks.

    Args:
        iterations: Number of iterations per benchmark

    Returns:
        List of BenchmarkSuites for encoding, decoding, and roundtrip
    """
    results = []
    for benchmark_func in [benchmark_encoding, benchmark_decoding, benchmark_roundtrip]:
        print(f"Running {benchmark_func.__name__}...")
        suite = benchmark_func(iterations=iterations)
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
