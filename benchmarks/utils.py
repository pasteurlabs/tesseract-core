# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark utilities for Tesseract Core."""

from __future__ import annotations

import json
import statistics
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable


def create_test_array(size: int, dtype: str = "float64") -> np.ndarray:
    """Create a random test array of given size."""
    return np.random.default_rng().standard_normal(size).astype(dtype)


class BenchmarkResult(BaseModel):
    """Result of a single benchmark."""

    name: str
    iterations: int
    total_time_s: float
    mean_time_s: float
    median_time_s: float
    std_time_s: float
    min_time_s: float
    max_time_s: float
    times_s: list[float] = Field(default_factory=list, exclude=True)


def run_benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 2,
    profile: bool = False,
) -> BenchmarkResult:
    """Run a benchmark and return results.

    Args:
        name: Name of the benchmark
        func: Function to benchmark (no arguments)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations (not counted)
        profile: Whether to profile each invocation with cProfile

    Returns:
        BenchmarkResult with timing statistics
    """
    from tesseract_core.runtime.profiler import Profiler

    # Warmup
    for _ in range(warmup):
        func()

    # Timed iterations
    profiler = Profiler(enabled=profile)
    times: list[float] = []
    with profiler:
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)

    stats_text = profiler.get_stats()
    if stats_text:
        print(f"\n--- Profile: {name} ---")
        print(stats_text)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_s=sum(times),
        mean_time_s=statistics.mean(times),
        median_time_s=statistics.median(times),
        std_time_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_time_s=min(times),
        max_time_s=max(times),
        times_s=times,
    )


class BenchmarkSuite(BaseModel):
    """Collection of benchmark results."""

    name: str
    results: list[BenchmarkResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)

    def save_json(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: str) -> BenchmarkSuite:
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)
