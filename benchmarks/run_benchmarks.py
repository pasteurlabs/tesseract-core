#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main benchmark runner for Tesseract Core.

Benchmarks measure real-world Tesseract framework overhead using a no-op
Tesseract that does nothing but decode inputs and encode outputs.

Usage:
    python run_benchmarks.py [options]

Options (all optional):
    -n, --min-iterations N   Minimum iteration count per case
    --min-duration SECS      Min duration per case in seconds (default: 0.1)
    -s, --suite SUITE        Which suite(s) to run; repeatable (default: all)
    -o, --output FILE        Save results to JSON file
    -c, --compare FILE       Compare against a baseline JSON file
    -m, --markdown           Output results as markdown table
    --array-sizes SIZES      Comma-separated array sizes (default: 1..100M)
    --profile                Enable cProfile profiling output

Examples:
    # Run all benchmarks (requires Docker)
    python run_benchmarks.py

    # Run only the HTTP interaction benchmark
    python run_benchmarks.py --suite containerized_http

    # Run only encoding benchmarks during development (no Docker needed)
    python run_benchmarks.py --suite array_encoding --suite array_roundtrip

    # Profile the encoding roundtrip to find bottlenecks
    python run_benchmarks.py --suite array_roundtrip --profile --array-sizes 1000000

    # Run specific array sizes with a longer minimum duration
    python run_benchmarks.py --suite array_encoding --min-duration 0.5 --array-sizes 100,10000

Available suites:
    from_tesseract_api   Non-containerized Tesseract via direct Python calls
    containerized_http   Containerized Tesseract via HTTP (requires Docker)
    containerized_cli    Containerized Tesseract via CLI (requires Docker)
    array_encoding       Array serialization (model_dump_json)
    array_decoding       Array deserialization (model_validate_json)
    array_roundtrip      Full encode + decode roundtrip
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root (e.g., python benchmarks/run_benchmarks.py)
_benchmarks_dir = str(Path(__file__).resolve().parent)
if _benchmarks_dir not in sys.path:
    sys.path.insert(0, _benchmarks_dir)

from utils import DEFAULT_MIN_DURATION_S, BenchmarkSuite  # noqa: E402

# Registry of available benchmark suites.
# Each entry maps a suite name to a factory that returns the runner function.
# Using factories (lambdas) to avoid importing bench modules at top level.
SuiteRunner = Callable[
    [int | None, list[int] | None, bool, float], BenchmarkSuite | None
]

SUITE_REGISTRY: dict[str, Callable[[], SuiteRunner]] = {
    "from_tesseract_api": lambda: (
        __import__("bench_tesseract").benchmark_from_tesseract_api
    ),
    "containerized_http": lambda: (
        __import__("bench_tesseract").benchmark_containerized_http
    ),
    "containerized_cli": lambda: (
        __import__("bench_tesseract").benchmark_containerized_cli
    ),
    "array_encoding": lambda: __import__("bench_array_encoding").benchmark_encoding,
    "array_decoding": lambda: __import__("bench_array_encoding").benchmark_decoding,
    "array_roundtrip": lambda: __import__("bench_array_encoding").benchmark_roundtrip,
}

AVAILABLE_SUITES = list(SUITE_REGISTRY.keys())


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
        "--min-iterations",
        "-n",
        type=int,
        default=None,
        help=(
            "Minimum number of iterations per benchmark case. "
            "The actual count may be higher if needed to reach --min-duration. "
            "By default, iterations are auto-calibrated from warmup timings."
        ),
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION_S,
        help=(
            f"Minimum duration in seconds for each benchmark case "
            f"(default: {DEFAULT_MIN_DURATION_S}). "
            f"Iterations are auto-calibrated from warmup timings to meet this target."
        ),
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
        "--suite",
        "-s",
        type=str,
        action="append",
        choices=AVAILABLE_SUITES,
        help=(
            "Which benchmark suite(s) to run. Can be specified multiple times. "
            "Defaults to all suites."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile profiling and print stats after each suite (useful for finding bottlenecks)",
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

    # Default to all suites if none specified
    selected_suites = args.suite or AVAILABLE_SUITES

    # Parse array sizes if provided
    array_sizes = None
    if args.array_sizes:
        array_sizes = [int(s.strip()) for s in args.array_sizes.split(",")]

    if args.min_iterations is not None:
        print(
            f"Running Tesseract Core benchmarks "
            f"(min_iterations={args.min_iterations}, "
            f"min_duration={args.min_duration}s per case)"
        )
    else:
        print(
            f"Running Tesseract Core benchmarks "
            f"(min_duration={args.min_duration}s per case)"
        )
    print(f"Suites: {', '.join(selected_suites)}")
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")

    start_time = time.time()

    # Run selected benchmarks
    suites: list[BenchmarkSuite] = []

    for suite_name in selected_suites:
        runner = SUITE_REGISTRY[suite_name]()
        print(f"\nRunning {suite_name}...")
        result = runner(
            min_iterations=args.min_iterations,
            array_sizes=array_sizes,
            profile=args.profile,
            min_duration_s=args.min_duration,
        )
        if result is not None:
            suites.append(result)

    elapsed = time.time() - start_time
    print(f"\nBenchmarks completed in {elapsed:.1f}s")

    # Print results
    if not args.markdown:
        print_results(suites)

    # Merge and save results
    combined = merge_suites(suites)
    combined.metadata["system"] = get_system_info()
    combined.metadata["min_iterations"] = args.min_iterations
    combined.metadata["min_duration_s"] = args.min_duration
    combined.metadata["elapsed_seconds"] = elapsed

    if args.output:
        Path(args.output).write_text(combined.model_dump_json(indent=2))
        print(f"\nResults saved to {args.output}")

    # Compare against baseline if provided
    if args.compare:
        from compare_benchmarks import generate_report

        baseline_path = Path(args.compare)
        if not baseline_path.exists():
            print(f"Warning: Baseline file not found: {baseline_path}")
        elif args.output:
            report = generate_report(str(baseline_path), args.output)
            if report:
                print(f"\n{report}")
        else:
            print("Warning: --compare requires --output to save current results first")

    return 0


if __name__ == "__main__":
    sys.exit(main())
