#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare benchmark results and generate a markdown report.

Usage:
    python compare_benchmarks.py baseline.json current.json output.md
"""

import json
import re
import sys
from pathlib import Path


def _parse_benchmark_name(name: str) -> tuple[str, str, int]:
    """Parse benchmark name into (suite, operation, size) for sorting.

    Examples:
        "from_tesseract_api/apply_1" -> ("from_tesseract_api", "apply", 1)
        "array_encoding/encode_json_100" -> ("array_encoding", "encode_json", 100)
    """
    # Split into suite and benchmark parts
    if "/" in name:
        suite, benchmark = name.split("/", 1)
    else:
        suite, benchmark = "", name

    # Extract the numeric size from the end (handles formats like "apply_1,000,000")
    # Remove commas from numbers first
    benchmark_cleaned = benchmark.replace(",", "")
    match = re.search(r"_(\d+)$", benchmark_cleaned)
    if match:
        size = int(match.group(1))
        operation = benchmark_cleaned[: match.start()]
    else:
        size = 0
        operation = benchmark_cleaned

    return (suite, operation, size)


def _sort_benchmark_names(names: list[str]) -> list[str]:
    """Sort benchmark names by suite, then operation, then size numerically."""
    return sorted(names, key=_parse_benchmark_name)


def load_benchmark_file(path: str) -> dict | None:
    """Load benchmark file including metadata.

    Returns None if the file doesn't exist.
    """
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def _generate_current_only_report(current: dict, current_data: dict) -> str:
    """Generate a report with only current benchmark results (no baseline comparison)."""
    lines = [
        "## Benchmark Results",
        "",
        ":information_source: No baseline found - showing current results only.",
        "",
        "Benchmarks use a no-op Tesseract to measure pure framework overhead.",
        "",
        "| Benchmark | Current |",
        "|-----------|---------|",
    ]

    for name in _sort_benchmark_names(list(current.keys())):
        curr_mean = current[name]["mean_time_s"] * 1000
        lines.append(f"| `{name}` | {curr_mean:.3f}ms |")

    # Extract metadata for details section
    iterations = current_data.get("metadata", {}).get("iterations", "N/A")

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Benchmark details</summary>",
            "",
            f"- **Iterations:** {iterations}",
            "- **Runner:** ubuntu-latest",
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def generate_report(baseline_path: str, current_path: str) -> str | None:
    """Generate markdown comparison report.

    Returns None only if current results don't exist.
    If baseline doesn't exist, generates a report with current results only.
    """
    baseline_data = load_benchmark_file(baseline_path)
    current_data = load_benchmark_file(current_path)

    if current_data is None:
        return None

    current = {r["name"]: r for r in current_data["results"]}

    # If no baseline, generate a current-only report
    if baseline_data is None:
        return _generate_current_only_report(current, current_data)

    baseline = {r["name"]: r for r in baseline_data["results"]}

    lines = [
        "## Benchmark Results",
        "",
        "Benchmarks use a no-op Tesseract to measure pure framework overhead.",
        "",
        "| Benchmark | Baseline | Current | Change | Status |",
        "|-----------|----------|---------|--------|--------|",
    ]

    # Find all benchmark names, sorted by suite then size
    all_names = _sort_benchmark_names(list(set(baseline.keys()) | set(current.keys())))

    for name in all_names:
        if name not in baseline:
            # New benchmark, no comparison possible
            curr_mean = current[name]["mean_time_s"] * 1000
            lines.append(f"| `{name}` | - | {curr_mean:.3f}ms | new | :new: |")
            continue

        if name not in current:
            # Removed benchmark
            base_mean = baseline[name]["mean_time_s"] * 1000
            lines.append(
                f"| `{name}` | {base_mean:.3f}ms | - | removed | :wastebasket: |"
            )
            continue

        base_mean = baseline[name]["mean_time_s"] * 1000
        curr_mean = current[name]["mean_time_s"] * 1000

        if base_mean > 0:
            diff_pct = ((curr_mean - base_mean) / base_mean) * 100
        else:
            diff_pct = 0

        if diff_pct < -5:
            status = ":rocket: faster"
        elif diff_pct > 5:
            status = ":warning: slower"
        else:
            status = ":white_check_mark:"

        lines.append(
            f"| `{name}` | {base_mean:.3f}ms | {curr_mean:.3f}ms | {diff_pct:+.1f}% | {status} |"
        )

    # Extract metadata for details section
    iterations = current_data.get("metadata", {}).get("iterations", "N/A")

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Benchmark details</summary>",
            "",
            f"- **Iterations:** {iterations}",
            "- **Runner:** ubuntu-latest",
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main function to compare benchmarks and generate report."""
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} baseline.json current.json output.md",
            file=sys.stderr,
        )
        return 1

    baseline_path = sys.argv[1]
    current_path = sys.argv[2]
    output_path = sys.argv[3]

    report = generate_report(baseline_path, current_path)

    if report is None:
        print("No current benchmark results found.", file=sys.stderr)
        return 1

    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
