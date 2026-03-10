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
from functools import partial

from pydantic import BaseModel
from utils import (
    DEFAULT_MIN_DURATION_S,
    BenchmarkSuite,
    create_test_array,
    run_benchmark,
)

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


def _encode(model: BaseModel, encoding: str, **context: str) -> str:
    return model.model_dump_json(context={"array_encoding": encoding, **context})


def _decode(encoded: str, **context: str) -> ArrayModel:
    return ArrayModel.model_validate_json(encoded, context=context)


def _roundtrip(model: BaseModel, encoding: str, **context: str) -> ArrayModel:
    encoded = model.model_dump_json(context={"array_encoding": encoding, **context})
    return ArrayModel.model_validate_json(encoded, context=context)


def benchmark_encoding(
    min_iterations: int | None = None,
    array_sizes: list[int] | None = None,
    profile: bool = False,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> BenchmarkSuite:
    """Run encoding benchmarks for all methods and sizes."""
    if array_sizes is None:
        array_sizes = DEFAULT_ARRAY_SIZES

    suite = BenchmarkSuite(
        name="array_encoding",
        metadata={"min_iterations": min_iterations, "array_sizes": array_sizes},
    )

    for i, size in enumerate(array_sizes):
        print(f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}...")
        model = ArrayModel(data=create_test_array(size))

        encodings: list[str] = [
            "json",
            "base64",
            "binref",
        ]
        if size > 20_000_000:
            encodings.remove(
                "json"
            )  # Skip JSON for very large arrays to avoid excessive compute

        with tempfile.TemporaryDirectory() as tmpdir:
            for encoding in encodings:
                ctx = {}
                if encoding == "binref":
                    ctx["base_dir"] = tmpdir
                result = run_benchmark(
                    name=f"encode_{encoding}_{size:,}",
                    func=partial(_encode, model, encoding, **ctx),
                    min_iterations=min_iterations,
                    profile=profile,
                    min_duration_s=min_duration_s,
                )
                suite.add_result(result)

    return suite


def benchmark_decoding(
    min_iterations: int | None = None,
    array_sizes: list[int] | None = None,
    profile: bool = False,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> BenchmarkSuite:
    """Run decoding benchmarks for all methods and sizes."""
    if array_sizes is None:
        array_sizes = DEFAULT_ARRAY_SIZES

    suite = BenchmarkSuite(
        name="array_decoding",
        metadata={"min_iterations": min_iterations, "array_sizes": array_sizes},
    )

    for i, size in enumerate(array_sizes):
        print(f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}...")
        model = ArrayModel(data=create_test_array(size))

        encodings: list[str] = ["json", "base64", "binref"]
        if size > 20_000_000:
            encodings.remove(
                "json"
            )  # Skip JSON for very large arrays to avoid excessive compute

        with tempfile.TemporaryDirectory() as tmpdir:
            for encoding in encodings:
                ctx: dict[str, str] = {}
                if encoding == "binref":
                    ctx["base_dir"] = tmpdir

                encoded = _encode(model, encoding, **ctx)
                result = run_benchmark(
                    name=f"decode_{encoding}_{size:,}",
                    func=partial(_decode, encoded, **ctx),
                    min_iterations=min_iterations,
                    profile=profile,
                    min_duration_s=min_duration_s,
                )
                suite.add_result(result)

    return suite


def benchmark_roundtrip(
    min_iterations: int | None = None,
    array_sizes: list[int] | None = None,
    profile: bool = False,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> BenchmarkSuite:
    """Run roundtrip (encode + decode) benchmarks."""
    if array_sizes is None:
        array_sizes = DEFAULT_ARRAY_SIZES

    suite = BenchmarkSuite(
        name="array_roundtrip",
        metadata={"min_iterations": min_iterations, "array_sizes": array_sizes},
    )

    for i, size in enumerate(array_sizes):
        print(f"  [{i + 1}/{len(array_sizes)}] Benchmarking size {size:,}...")
        model = ArrayModel(data=create_test_array(size))

        encodings: list[str] = ["json", "base64", "binref"]
        if size > 20_000_000:
            encodings.remove(
                "json"
            )  # Skip JSON for very large arrays to avoid excessive compute

        with tempfile.TemporaryDirectory() as tmpdir:
            for encoding in encodings:
                ctx: dict[str, str] = {}
                if encoding == "binref":
                    ctx["base_dir"] = tmpdir

                result = run_benchmark(
                    name=f"roundtrip_{encoding}_{size:,}",
                    func=partial(_roundtrip, model, encoding, **ctx),
                    min_iterations=min_iterations,
                    profile=profile,
                    min_duration_s=min_duration_s,
                )
                suite.add_result(result)

    return suite


def run_all(
    min_iterations: int | None = None,
    array_sizes: list[int] | None = None,
    profile: bool = False,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> list[BenchmarkSuite]:
    """Run all array encoding benchmarks."""
    results = []
    for benchmark_func in [benchmark_encoding, benchmark_decoding, benchmark_roundtrip]:
        print(f"Running {benchmark_func.__name__}...")
        suite = benchmark_func(
            min_iterations=min_iterations,
            array_sizes=array_sizes,
            profile=profile,
            min_duration_s=min_duration_s,
        )
        results.append(suite)
    return results


if __name__ == "__main__":
    import sys

    min_iterations: int | None = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print("Running array encoding benchmarks...")
    suites = run_all(min_iterations=min_iterations)

    for suite in suites:
        print(f"\n=== {suite.name} ===")
        for result in suite.results:
            print(
                f"  {result.name}: {result.mean_time_s * 1000:.3f}ms (±{result.std_time_s * 1000:.3f}ms)"
            )
