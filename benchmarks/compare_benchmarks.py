#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pytest-benchmark JSON results and generate a markdown report.

Usage:
    python compare_benchmarks.py --current current.json --output output.md [--baseline baseline.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Percentage change threshold for flagging a benchmark as notable.
NOTABLE_THRESHOLD_PCT = 10


def _get_median_ms(bench: dict) -> float:
    """Get median time in ms from a pytest-benchmark entry."""
    return bench["stats"]["median"] * 1000


def _short_name(bench: dict) -> str:
    """Derive a short display name from a pytest-benchmark entry.

    Maps pytest-benchmark names like:
        "test_from_tesseract_api[1,000]" -> "api/apply_1,000"
        "test_containerized_http[100]"   -> "http/apply_100"
        "test_containerized_cli[100]"             -> "cli/apply_100"
        "test_encoding[json_100]"        -> "encoding/json_100"
        "test_decoding[base64_10,000]"   -> "decoding/base64_10,000"
        "test_roundtrip[binref_1,000]"   -> "roundtrip/binref_1,000"
    """
    name = bench["name"]

    # Extract test function name and params from "test_func[params]"
    m = re.match(r"test_(\w+)\[(.+)\]$", name)
    if not m:
        return name

    func, params = m.group(1), m.group(2)

    # Tesseract apply benchmarks
    if func == "from_tesseract_api":
        return f"api/apply_{params}"
    if func == "containerized_http":
        return f"http/apply_{params}"
    if func == "containerized_cli":
        return f"cli/apply_{params}"

    # Array encoding benchmarks: test_encoding[json_100] etc.
    if func in ("encoding", "decoding", "roundtrip"):
        return f"{func}/{params}"

    return name


def _parse_short_name(name: str) -> tuple[str, str, int]:
    """Parse short name into (suite, operation, size) for sorting.

    Examples:
        "api/apply_1,000" -> ("api", "apply", 1000)
        "encoding/json_100" -> ("encoding", "json", 100)
    """
    if "/" in name:
        suite, benchmark = name.split("/", 1)
    else:
        suite, benchmark = "", name

    benchmark_cleaned = benchmark.replace(",", "")
    match = re.search(r"_(\d+)$", benchmark_cleaned)
    if match:
        size = int(match.group(1))
        operation = benchmark_cleaned[: match.start()]
    else:
        size = 0
        operation = benchmark_cleaned

    return (suite, operation, size)


def _sort_names(names: list[str]) -> list[str]:
    """Sort benchmark names by suite, then operation, then size numerically."""
    return sorted(names, key=_parse_short_name)


def _get_runner_description(data: dict) -> str:
    """Get a runner description from pytest-benchmark machine_info."""
    machine = data.get("machine_info", {})
    system = machine.get("system", "")
    release = machine.get("release", "")
    machine_arch = machine.get("machine", "")
    parts = [p for p in (system, release, machine_arch) if p]
    return " ".join(parts) if parts else "unknown"


def _load_benchmark_file(path: str | None) -> dict | None:
    """Load benchmark file, returning None if it doesn't exist."""
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def _index_benchmarks(data: dict) -> dict[str, dict]:
    """Index benchmarks by short name, keyed to their stats."""
    result = {}
    for bench in data.get("benchmarks", []):
        name = _short_name(bench)
        result[name] = bench
    return result


def _compute_comparison(name: str, baseline: dict, current: dict) -> dict:
    """Compute comparison metrics for a single benchmark."""
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
        diff_pct = 0.0

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


def _generate_current_only_report(current: dict[str, dict], current_data: dict) -> str:
    """Generate a report when no baseline exists, marking every benchmark as new."""
    all_names = _sort_names(list(current.keys()))
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

    runner = _get_runner_description(current_data)
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Benchmark details</summary>",
            "",
            f"- **Runner:** {runner}",
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def generate_report(baseline_path: str | None, current_path: str) -> str | None:
    """Generate markdown comparison report.

    Returns None only if current results don't exist.
    If baseline doesn't exist, generates a report with current results only.
    """
    baseline_data = _load_benchmark_file(baseline_path)
    current_data = _load_benchmark_file(current_path)

    if current_data is None:
        return None

    current = _index_benchmarks(current_data)

    if baseline_data is None:
        return _generate_current_only_report(current, current_data)

    baseline = _index_benchmarks(baseline_data)

    all_names = _sort_names(list(set(baseline.keys()) | set(current.keys())))
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

    runner = _get_runner_description(current_data)
    lines.extend(
        [
            "",
            f"- **Runner:** {runner}",
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main function to compare benchmarks and generate report."""
    parser = argparse.ArgumentParser(description="Compare benchmark results.")
    parser.add_argument("--baseline", default=None, help="Baseline benchmark JSON file")
    parser.add_argument("--current", required=True, help="Current benchmark JSON file")
    parser.add_argument("--output", required=True, help="Output markdown report path")
    args = parser.parse_args()

    if not Path(args.current).exists():
        print(f"Current benchmark file not found: {args.current}", file=sys.stderr)
        return 1

    report = generate_report(args.baseline, args.current)

    if report is None:
        print(
            f"Failed to generate report (current={args.current}, baseline={args.baseline})",
            file=sys.stderr,
        )
        return 1

    try:
        with open(args.output, "w") as f:
            f.write(report)
    except OSError as e:
        print(f"Failed to write report to {args.output}: {e}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
