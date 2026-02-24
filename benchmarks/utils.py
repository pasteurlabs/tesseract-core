# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark utilities for Tesseract Core."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    iterations: int
    total_time_s: float
    mean_time_s: float
    std_time_s: float
    min_time_s: float
    max_time_s: float
    times_s: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_s": self.total_time_s,
            "mean_time_s": self.mean_time_s,
            "std_time_s": self.std_time_s,
            "min_time_s": self.min_time_s,
            "max_time_s": self.max_time_s,
        }


def run_benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """Run a benchmark and return results.

    Args:
        name: Name of the benchmark
        func: Function to benchmark (no arguments)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations (not counted)

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed iterations
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    total_time = sum(times)
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_s=total_time,
        mean_time_s=mean_time,
        std_time_s=std_time,
        min_time_s=min(times),
        max_time_s=max(times),
        times_s=times,
    )


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> BenchmarkSuite:
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        suite = cls(name=data["name"], metadata=data.get("metadata", {}))
        for r in data["results"]:
            suite.results.append(
                BenchmarkResult(
                    name=r["name"],
                    iterations=r["iterations"],
                    total_time_s=r["total_time_s"],
                    mean_time_s=r["mean_time_s"],
                    std_time_s=r["std_time_s"],
                    min_time_s=r["min_time_s"],
                    max_time_s=r["max_time_s"],
                )
            )
        return suite


def compare_results(
    baseline: BenchmarkSuite, current: BenchmarkSuite
) -> dict[str, dict[str, float]]:
    """Compare two benchmark suites and return relative speedups.

    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results

    Returns:
        Dictionary mapping benchmark name to comparison metrics:
        - speedup: >1 means current is faster, <1 means slower
        - baseline_mean_s: baseline mean time
        - current_mean_s: current mean time
        - diff_pct: percentage difference (negative = faster)
    """
    baseline_by_name = {r.name: r for r in baseline.results}
    current_by_name = {r.name: r for r in current.results}

    comparisons: dict[str, dict[str, float]] = {}
    for name in baseline_by_name:
        if name in current_by_name:
            base_mean = baseline_by_name[name].mean_time_s
            curr_mean = current_by_name[name].mean_time_s

            if curr_mean > 0:
                speedup = base_mean / curr_mean
            else:
                speedup = float("inf")

            diff_pct = (
                ((curr_mean - base_mean) / base_mean) * 100 if base_mean > 0 else 0
            )

            comparisons[name] = {
                "speedup": speedup,
                "baseline_mean_s": base_mean,
                "current_mean_s": curr_mean,
                "diff_pct": diff_pct,
            }

    return comparisons


def format_comparison_table(comparisons: dict[str, dict[str, float]]) -> str:
    """Format comparison results as a markdown table.

    Args:
        comparisons: Output from compare_results()

    Returns:
        Markdown-formatted table
    """
    lines = [
        "| Benchmark | Baseline | Current | Change | Status |",
        "|-----------|----------|---------|--------|--------|",
    ]

    for name, comp in sorted(comparisons.items()):
        baseline_ms = comp["baseline_mean_s"] * 1000
        current_ms = comp["current_mean_s"] * 1000
        diff_pct = comp["diff_pct"]

        # Determine status emoji
        if diff_pct < -5:
            status = ":rocket: faster"
        elif diff_pct > 5:
            status = ":warning: slower"
        else:
            status = ":white_check_mark: ~same"

        # Format change with sign
        change_str = f"{diff_pct:+.1f}%"

        lines.append(
            f"| {name} | {baseline_ms:.3f}ms | {current_ms:.3f}ms | {change_str} | {status} |"
        )

    return "\n".join(lines)
