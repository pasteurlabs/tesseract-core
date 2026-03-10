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

# Percentage change threshold for flagging a benchmark as notable.
NOTABLE_THRESHOLD_PCT = 5


def _get_median_ms(result: dict) -> float:
    """Get median time in ms, falling back to mean for older result files."""
    return result.get("median_time_s", result["mean_time_s"]) * 1000


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


def _get_runner_description(data: dict) -> str:
    """Get a runner description from benchmark metadata."""
    system = data.get("metadata", {}).get("system", {})
    platform = system.get("platform", "")
    version = system.get("platform_release", "")
    arch = system.get("architecture", "")
    parts = [p for p in (platform, version, arch) if p]
    return " ".join(parts) if parts else "unknown"


def _load_benchmark_file(path: str) -> dict | None:
    """Load benchmark file, returning None if it doesn't exist."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def _generate_current_only_report(current: dict, current_data: dict) -> str:
    """Generate a report when no baseline exists, marking every benchmark as new."""
    all_names = _sort_benchmark_names(list(current.keys()))
    comparisons = [
        _compute_comparison(name, baseline={}, current=current) for name in all_names
    ]

    lines = [
        "## Benchmark Results",
        "",
        ":information_source: No baseline found — all benchmarks marked as new.",
        "",
        "Benchmarks use a no-op Tesseract to measure pure framework overhead.",
        "",
        "| Benchmark | Baseline | Current | Change | Status |",
        "|-----------|----------|---------|--------|--------|",
    ]

    for comp in comparisons:
        lines.append(_format_comparison_row(comp))

    # Extract metadata for details section
    iterations = current_data.get("metadata", {}).get("iterations", "N/A")
    runner = _get_runner_description(current_data)

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Benchmark details</summary>",
            "",
            f"- **Iterations:** {iterations}",
            f"- **Runner:** {runner}",
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def _compute_comparison(name: str, baseline: dict, current: dict) -> dict:
    """Compute comparison metrics for a single benchmark.

    Returns a dict with keys: name, base_median_ms, curr_median_ms, diff_pct, status.
    """
    if name not in baseline:
        curr_median = _get_median_ms(current[name])
        return {
            "name": name,
            "base_median_ms": None,
            "curr_median_ms": curr_median,
            "diff_pct": None,
            "status": ":new:",
            "notable": True,
        }

    if name not in current:
        base_median = _get_median_ms(baseline[name])
        return {
            "name": name,
            "base_median_ms": base_median,
            "curr_median_ms": None,
            "diff_pct": None,
            "status": ":wastebasket:",
            "notable": True,
        }

    base_median = _get_median_ms(baseline[name])
    curr_median = _get_median_ms(current[name])

    if base_median > 0:
        diff_pct = ((curr_median - base_median) / base_median) * 100
    else:
        diff_pct = 0

    if diff_pct < -NOTABLE_THRESHOLD_PCT:
        status = ":rocket: faster"
    elif diff_pct > NOTABLE_THRESHOLD_PCT:
        status = ":warning: slower"
    else:
        status = ":white_check_mark:"

    notable = abs(diff_pct) > NOTABLE_THRESHOLD_PCT
    return {
        "name": name,
        "base_median_ms": base_median,
        "curr_median_ms": curr_median,
        "diff_pct": diff_pct,
        "status": status,
        "notable": notable,
    }


def _format_comparison_row(comp: dict) -> str:
    """Format a single comparison as a markdown table row."""
    base_str = (
        f"{comp['base_median_ms']:.3f}ms" if comp["base_median_ms"] is not None else "-"
    )
    curr_str = (
        f"{comp['curr_median_ms']:.3f}ms" if comp["curr_median_ms"] is not None else "-"
    )
    change_str = (
        f"{comp['diff_pct']:+.1f}%"
        if comp["diff_pct"] is not None
        else "new"
        if comp["status"] == ":new:"
        else "removed"
    )
    return f"| `{comp['name']}` | {base_str} | {curr_str} | {change_str} | {comp['status']} |"


def generate_report(baseline_path: str, current_path: str) -> str | None:
    """Generate markdown comparison report.

    Returns None only if current results don't exist.
    If baseline doesn't exist, generates a report with current results only.
    """
    baseline_data = _load_benchmark_file(baseline_path)
    current_data = _load_benchmark_file(current_path)

    if current_data is None:
        return None

    current = {r["name"]: r for r in current_data["results"]}

    # If no baseline, generate a current-only report
    if baseline_data is None:
        return _generate_current_only_report(current, current_data)

    baseline = {r["name"]: r for r in baseline_data["results"]}

    # Compute all comparisons
    all_names = _sort_benchmark_names(list(set(baseline.keys()) | set(current.keys())))
    comparisons = [_compute_comparison(name, baseline, current) for name in all_names]

    notable = [c for c in comparisons if c["notable"]]
    diffs = [c["diff_pct"] for c in comparisons if c["diff_pct"] is not None]
    num_faster = sum(1 for d in diffs if d < -NOTABLE_THRESHOLD_PCT)
    num_slower = sum(1 for d in diffs if d > NOTABLE_THRESHOLD_PCT)
    num_same = sum(1 for d in diffs if abs(d) <= NOTABLE_THRESHOLD_PCT)

    lines = [
        "## Benchmark Results",
        "",
        "Benchmarks use a no-op Tesseract to measure pure framework overhead.",
        "",
        f":rocket: {num_faster} faster, :warning: {num_slower} slower, :white_check_mark: {num_same} unchanged",
        "",
    ]

    # Show notable changes prominently
    if notable:
        lines.extend(
            [
                "### Notable changes",
                "",
                "| Benchmark | Baseline | Current | Change | Status |",
                "|-----------|----------|---------|--------|--------|",
            ]
        )
        for comp in notable:
            lines.append(_format_comparison_row(comp))
        lines.append("")
    else:
        lines.extend(
            [
                ":white_check_mark: No significant performance changes detected.",
                "",
            ]
        )

    # Full results in collapsed section
    lines.extend(
        [
            "<details>",
            "<summary>Full results</summary>",
            "",
            "| Benchmark | Baseline | Current | Change | Status |",
            "|-----------|----------|---------|--------|--------|",
        ]
    )
    for comp in comparisons:
        lines.append(_format_comparison_row(comp))

    # Extract metadata for details section
    iterations = current_data.get("metadata", {}).get("iterations", "N/A")
    runner = _get_runner_description(current_data)

    lines.extend(
        [
            "",
            f"- **Iterations:** {iterations}",
            f"- **Runner:** {runner}",
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

    if not Path(current_path).exists():
        print(f"Current benchmark file not found: {current_path}", file=sys.stderr)
        return 1

    report = generate_report(baseline_path, current_path)

    if report is None:
        print(
            f"Failed to generate report (current={current_path}, baseline={baseline_path})",
            file=sys.stderr,
        )
        return 1

    try:
        with open(output_path, "w") as f:
            f.write(report)
    except OSError as e:
        print(f"Failed to write report to {output_path}: {e}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
