#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compare pytest-benchmark JSON results and generate a markdown report.

Each side takes one JSON file per pass of the benchmark suite. A benchmark's
median can move between two runs of the same code by more than it varies
between rounds within one run (which CPU cores a server process lands on, for
example), so a single pass per side is not enough to tell a change from noise.
With several passes, a benchmark is flagged only if every current pass is more
than NOTABLE_THRESHOLD_PCT slower (or faster) than every baseline pass.

Usage:
    python compare_benchmarks.py --current current_1.json [current_2.json ...] \
        --output output.md [--baseline baseline_1.json [baseline_2.json ...]]
"""

import argparse
import json
import re
import statistics
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
        "test_containerized_http_shmem[100]" -> "http-shmem/apply_100"
        "test_containerized_cli[100]"    -> "cli/apply_100"
        "test_subprocess[100]"           -> "subprocess/apply_100"
        "test_subprocess_shmem[100]"     -> "subprocess-shmem/apply_100"
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
    if func == "containerized_http_shmem":
        return f"http-shmem/apply_{params}"
    if func == "containerized_cli":
        return f"cli/apply_{params}"
    if func == "subprocess":
        return f"subprocess/apply_{params}"
    if func == "subprocess_shmem":
        return f"subprocess-shmem/apply_{params}"

    # Array encoding benchmarks: test_roundtrip[json_100] etc.
    if func == "roundtrip":
        return f"{func}/{params}"

    return name


def _parse_short_name(name: str) -> tuple[str, str, int]:
    """Parse short name into (suite, operation, size) for sorting.

    Examples:
        "api/apply_1,000" -> ("api", "apply", 1000)
        "roundtrip/json_100" -> ("roundtrip", "json", 100)
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


def _metadata_lines(current_data: dict, pr_number: str | None) -> list[str]:
    """Provenance bullets describing where the results came from."""
    lines = []
    if pr_number is not None:
        lines.append(f"- **PR:** #{pr_number}")
    lines.append(f"- **Runner:** {_get_runner_description(current_data)}")
    lines.append(f"- **Python:** {_python_description(current_data)}")
    return lines


def _python_description(data: dict) -> str:
    """Describe the Python interpreter a benchmark run used."""
    machine = data.get("machine_info", {})
    implementation = machine.get("python_implementation", "")
    version = machine.get("python_version", "")
    return " ".join(p for p in (implementation, version) if p) or "unknown"


def _load_passes(paths: list[str] | None) -> list[dict]:
    """Load the benchmark files of one side, skipping any that don't exist."""
    passes = []
    for path in paths or []:
        if Path(path).exists():
            with open(path) as f:
                passes.append(json.load(f))
    return passes


def _index_passes(passes: list[dict]) -> dict[str, list[float]]:
    """Map each benchmark's short name to its median time in ms in every pass."""
    result: dict[str, list[float]] = {}
    for data in passes:
        for bench in data.get("benchmarks", []):
            result.setdefault(_short_name(bench), []).append(_get_median_ms(bench))
    return result


def _spread_pct(medians: list[float]) -> float | None:
    """Range of the per-pass medians, relative to their median."""
    if len(medians) < 2:
        return None
    center = statistics.median(medians)
    if center <= 0:
        return None
    return (max(medians) - min(medians)) / center * 100


def _compute_comparison(
    name: str, baseline: dict[str, list[float]], current: dict[str, list[float]]
) -> dict:
    """Compute comparison metrics for a single benchmark.

    ``baseline`` and ``current`` map names to per-pass medians. A change is
    notable only if every current pass is beyond every baseline pass by more
    than the threshold, so a single outlier pass on either side cannot produce
    a flag.
    """
    base = baseline.get(name)
    curr = current.get(name)
    spreads = [_spread_pct(medians) for medians in (base, curr) if medians]
    spreads = [s for s in spreads if s is not None]
    comp = {
        "name": name,
        "base_median_ms": statistics.median(base) if base else None,
        "curr_median_ms": statistics.median(curr) if curr else None,
        "spread_pct": max(spreads) if spreads else None,
        "diff_pct": None,
    }

    if base is None:
        return {**comp, "status": ":new:", "notable": True}
    if curr is None:
        return {**comp, "status": ":wastebasket:", "notable": True}

    if comp["base_median_ms"] > 0:
        comp["diff_pct"] = (comp["curr_median_ms"] / comp["base_median_ms"] - 1) * 100
    else:
        comp["diff_pct"] = 0.0

    threshold = NOTABLE_THRESHOLD_PCT / 100
    if min(curr) > max(base) * (1 + threshold):
        status = ":warning: slower"
    elif max(curr) < min(base) * (1 - threshold):
        status = ":rocket: faster"
    else:
        status = ":white_check_mark:"

    return {**comp, "status": status, "notable": status != ":white_check_mark:"}


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
    spread_str = f"{comp['spread_pct']:.1f}%" if comp["spread_pct"] is not None else "-"
    return (
        f"| `{comp['name']}` | {base_str} | {curr_str} | {change_str} | "
        f"{spread_str} | {comp['status']} |"
    )


_TABLE_HEADER = [
    "| Benchmark | Baseline | Current | Change | Spread | Status |",
    "|-----------|----------|---------|--------|--------|--------|",
]

_COLUMN_NOTE = (
    "Times are the median over passes of each pass's median. Spread is the "
    "range of the per-pass medians on the noisier side, relative to its median."
)


def _passes_description(num_baseline: int, num_current: int) -> str:
    """Describe how many passes each side ran."""
    if num_baseline == num_current:
        return f"{num_current} pass{'es' if num_current != 1 else ''} per side"
    return f"{num_baseline} baseline and {num_current} current passes"


def _generate_current_only_report(
    current: dict[str, list[float]],
    current_data: dict,
    pr_number: str | None = None,
    notice: str = ":information_source: No baseline found — all benchmarks marked as new.",
) -> str:
    """Generate a report without a usable baseline, marking every benchmark as new."""
    all_names = _sort_names(list(current.keys()))
    comparisons = [
        _compute_comparison(name, baseline={}, current=current) for name in all_names
    ]

    lines = [
        "## Benchmark Results",
        "",
        notice,
        "",
        "Benchmarks use a no-op Tesseract to measure pure framework overhead.",
        "",
        *_TABLE_HEADER,
    ]

    for comp in comparisons:
        lines.append(_format_comparison_row(comp))

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Benchmark details</summary>",
            "",
            *_metadata_lines(current_data, pr_number),
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def generate_report(
    baseline_paths: list[str] | None,
    current_paths: list[str],
    pr_number: str | None = None,
) -> str | None:
    """Generate markdown comparison report.

    Returns None only if current results don't exist.
    If baseline doesn't exist, generates a report with current results only.
    """
    baseline_passes = _load_passes(baseline_paths)
    current_passes = _load_passes(current_paths)

    if not current_passes:
        return None

    current_data = current_passes[0]
    current = _index_passes(current_passes)

    if not baseline_passes:
        return _generate_current_only_report(current, current_data, pr_number)

    # Timings from different interpreters can differ by more than the notable
    # threshold, which the report would then attribute to the PR.
    baseline_pythons = {_python_description(d) for d in baseline_passes}
    current_pythons = {_python_description(d) for d in current_passes}
    if len(baseline_pythons | current_pythons) > 1:
        return _generate_current_only_report(
            current,
            current_data,
            pr_number,
            notice=(
                f":warning: Baseline ran on {', '.join(sorted(baseline_pythons))} but "
                f"current ran on {', '.join(sorted(current_pythons))}, so the results "
                "are not comparable. Showing current results only."
            ),
        )

    baseline = _index_passes(baseline_passes)

    all_names = _sort_names(list(set(baseline.keys()) | set(current.keys())))
    comparisons = [_compute_comparison(name, baseline, current) for name in all_names]

    notable = [c for c in comparisons if c["notable"]]
    statuses = [c["status"] for c in comparisons if c["diff_pct"] is not None]
    num_faster = statuses.count(":rocket: faster")
    num_slower = statuses.count(":warning: slower")
    num_same = statuses.count(":white_check_mark:")

    lines = [
        "## Benchmark Results",
        "",
        "Benchmarks use a no-op Tesseract to measure pure framework overhead.",
        "",
        f":rocket: {num_faster} faster, :warning: {num_slower} slower, :white_check_mark: {num_same} unchanged",
        "",
    ]

    if notable:
        lines.extend(["### Notable changes", "", *_TABLE_HEADER])
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
            *_TABLE_HEADER,
        ]
    )
    for comp in comparisons:
        lines.append(_format_comparison_row(comp))

    method_note = (
        f"Ran {_passes_description(len(baseline_passes), len(current_passes))}, "
        "alternating between baseline and current. A benchmark is flagged only "
        "if every current pass differs from every baseline pass by more than "
        f"{NOTABLE_THRESHOLD_PCT}% in the same direction. {_COLUMN_NOTE}"
    )
    lines.extend(
        [
            "",
            method_note,
            "",
            *_metadata_lines(current_data, pr_number),
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main function to compare benchmarks and generate report."""
    parser = argparse.ArgumentParser(description="Compare benchmark results.")
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=None,
        help="Baseline benchmark JSON files, one per pass",
    )
    parser.add_argument(
        "--current",
        nargs="+",
        required=True,
        help="Current benchmark JSON files, one per pass",
    )
    parser.add_argument("--output", required=True, help="Output markdown report path")
    parser.add_argument(
        "--pr-number", default=None, help="PR number to record in the report"
    )
    args = parser.parse_args()

    if not any(Path(path).exists() for path in args.current):
        print(f"Current benchmark files not found: {args.current}", file=sys.stderr)
        return 1

    report = generate_report(args.baseline, args.current, args.pr_number)

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
