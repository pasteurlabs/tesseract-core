#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main benchmark runner for Tesseract Core.

Benchmarks measure real-world Tesseract framework overhead using a no-op
Tesseract that does nothing but decode inputs and encode outputs.

Usage:
    python run_benchmarks.py [options]

Options:
    --iterations N       Number of iterations per benchmark (default: 50)
    --output PATH        Output JSON file for results
    --compare PATH       Compare against baseline JSON file
    --docker             Include containerized benchmarks (requires Docker)
    --array-sizes N,...  Comma-separated list of array sizes to benchmark
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from utils import BenchmarkSuite, compare_results, format_comparison_table


def get_system_info() -> dict:
    """Gather system information for benchmark context."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_tesseract_benchmarks(
    iterations: int, include_docker: bool, array_sizes: list[int] | None = None
) -> list[BenchmarkSuite]:
    """Run Tesseract interaction benchmarks."""
    from bench_tesseract import run_all

    return run_all(
        iterations, include_containerized=include_docker, array_sizes=array_sizes
    )


def run_encoding_benchmarks(
    iterations: int, array_sizes: list[int] | None = None
) -> list[BenchmarkSuite]:
    """Run array encoding benchmarks (isolated, for detailed analysis)."""
    from bench_array_encoding import run_all

    return run_all(iterations, array_sizes=array_sizes)


def merge_suites(suites: list[BenchmarkSuite]) -> BenchmarkSuite:
    """Merge multiple suites into one combined suite."""
    combined = BenchmarkSuite(
        name="tesseract_benchmarks",
        metadata={"suites": [s.name for s in suites]},
    )
    for suite in suites:
        for result in suite.results:
            # Prefix result name with suite name for uniqueness
            result.name = f"{suite.name}/{result.name}"
            combined.add_result(result)
    return combined


def print_results(suites: list[BenchmarkSuite]) -> None:
    """Print benchmark results to stdout."""
    for suite in suites:
        print(f"\n{'=' * 60}")
        print(f"Suite: {suite.name}")
        print("=" * 60)

        for result in suite.results:
            mean_ms = result.mean_time_s * 1000
            std_ms = result.std_time_s * 1000
            print(f"  {result.name:40s} {mean_ms:8.3f}ms (±{std_ms:.3f}ms)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tesseract Core benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=50,
        help="Number of iterations per benchmark (default: 50)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--compare",
        "-c",
        type=str,
        help="Compare against baseline JSON file",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Exclude containerized benchmarks (requires Docker)",
    )
    parser.add_argument(
        "--encoding-only",
        action="store_true",
        help="Only run encoding benchmarks (for detailed analysis)",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        action="store_true",
        help="Output results as markdown table (for CI comments)",
    )
    parser.add_argument(
        "--array-sizes",
        type=str,
        default=None,
        help="Comma-separated list of array sizes to benchmark (e.g., '100,10000,1000000')",
    )

    args = parser.parse_args()

    # Parse array sizes if provided
    array_sizes = None
    if args.array_sizes:
        array_sizes = [int(s.strip()) for s in args.array_sizes.split(",")]

    print(f"Running Tesseract Core benchmarks (iterations={args.iterations})")
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")

    start_time = time.time()

    # Run benchmarks
    if args.encoding_only:
        suites = run_encoding_benchmarks(args.iterations, array_sizes=array_sizes)
    else:
        suites = run_tesseract_benchmarks(
            args.iterations,
            include_docker=not args.no_docker,
            array_sizes=array_sizes,
        )
        suites.extend(run_encoding_benchmarks(args.iterations, array_sizes=array_sizes))

    elapsed = time.time() - start_time
    print(f"\nBenchmarks completed in {elapsed:.1f}s")

    # Print results
    if not args.markdown:
        print_results(suites)

    # Merge and save results
    combined = merge_suites(suites)
    combined.metadata["system"] = get_system_info()
    combined.metadata["iterations"] = args.iterations
    combined.metadata["elapsed_seconds"] = elapsed

    if args.output:
        combined.save_json(args.output)
        print(f"\nResults saved to {args.output}")

    # Compare against baseline if provided
    if args.compare:
        baseline_path = Path(args.compare)
        if baseline_path.exists():
            baseline = BenchmarkSuite.load_json(str(baseline_path))
            comparisons = compare_results(baseline, combined)

            if args.markdown:
                print("\n## Benchmark Results\n")
                print(format_comparison_table(comparisons))
            else:
                print("\n" + "=" * 60)
                print("Comparison against baseline")
                print("=" * 60)
                for name, comp in sorted(comparisons.items()):
                    diff_pct = comp["diff_pct"]
                    status = "🚀" if diff_pct < -5 else ("⚠️" if diff_pct > 5 else "✓")
                    print(f"  {name:40s} {diff_pct:+6.1f}% {status}")
        else:
            print(f"Warning: Baseline file not found: {baseline_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
