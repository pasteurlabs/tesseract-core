#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare benchmark results and generate a markdown report.

Usage:
    python compare_benchmarks.py baseline.json current.json output.md
"""

import json
import sys


def load_results(path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return {r["name"]: r for r in data["results"]}


def generate_report(baseline_path: str, current_path: str) -> str:
    """Generate markdown comparison report."""
    baseline = load_results(baseline_path)
    current = load_results(current_path)

    lines = [
        "## Benchmark Results",
        "",
        "Benchmarks use a no-op Tesseract to measure pure framework overhead.",
        "",
        "| Benchmark | Baseline | Current | Change | Status |",
        "|-----------|----------|---------|--------|--------|",
    ]

    for name in sorted(baseline.keys()):
        if name not in current:
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

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Benchmark details</summary>",
            "",
            "- **Iterations:** 30",
            "- **Runner:** ubuntu-latest",
            "- **Suites:** from_tesseract_api, from_tesseract_api_with_output, http_testclient",
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

    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
