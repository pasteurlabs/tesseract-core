# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark utilities for Tesseract Core."""

from __future__ import annotations

import math
import statistics
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

# Minimum total benchmark duration in seconds.  run_benchmark uses warmup
# timings to estimate iterations so that the timed phase lasts at least this
# long.  Can be overridden per-call via the *min_duration_s* parameter.
DEFAULT_MIN_DURATION_S = 0.1


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
    min_iterations: int | None = None,
    warmup: int = 2,
    profile: bool = False,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> BenchmarkResult:
    """Run a benchmark and return results.

    The number of iterations is determined automatically from warmup timings
    so that the timed phase runs for at least *min_duration_s* seconds.  If
    *min_iterations* is given, at least that many iterations will be run.

    Args:
        name: Name of the benchmark
        func: Function to benchmark (no arguments)
        min_iterations: Minimum number of timed iterations.  The actual count
            may be higher to satisfy *min_duration_s*.
        warmup: Number of warmup iterations (not counted)
        profile: Whether to profile each invocation with cProfile
        min_duration_s: Target minimum total duration for the timed phase.

    Returns:
        BenchmarkResult with timing statistics
    """
    from tesseract_core.runtime.profiler import Profiler

    # Warmup — also used to estimate per-call time for auto-calibration.
    warmup_times: list[float] = []
    for _ in range(warmup):
        start = time.perf_counter()
        func()
        warmup_times.append(time.perf_counter() - start)

    # Estimate iterations from warmup timings.
    estimated_per_call = statistics.mean(warmup_times) if warmup_times else 0.0
    if estimated_per_call > 0:
        calibrated = max(1, math.ceil(min_duration_s / estimated_per_call))
    else:
        # Function is effectively instant; use a sensible fallback.
        calibrated = max(1, math.ceil(min_duration_s / 1e-6))

    iterations = max(calibrated, min_iterations or 0)

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
