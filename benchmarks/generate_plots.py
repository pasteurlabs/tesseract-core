#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate performance plots from pytest-benchmark JSON output.

This script generates plots showing Tesseract overhead for various workload
sizes, helping users understand if their workload is a good fit.

Usage:
    python generate_plots.py --benchmark-file PATH [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

_outline = [pe.withStroke(linewidth=3, foreground="white")]

here = os.path.dirname(os.path.abspath(__file__))


def load_benchmark_results(benchmark_file: Path) -> dict:
    """Load benchmark results from JSON file."""
    with open(benchmark_file) as f:
        return json.load(f)


def _parse_benchmark_name(name: str) -> tuple[str, str]:
    """Parse a pytest-benchmark name into (test_func, params).

    Example: "test_from_tesseract_api[1,000]" -> ("from_tesseract_api", "1,000")
    """
    m = re.match(r"test_(\w+)\[(.+)\]$", name)
    if not m:
        return (name, "")
    return m.group(1), m.group(2)


def extract_suite_data(results: dict, suite_name: str) -> dict[int, float]:
    """Extract median times by array size for a given Tesseract suite.

    Returns dict mapping array size to median time in ms.
    """
    # Map suite names to test function names
    suite_names = ("from_tesseract_api", "containerized_http", "containerized_cli")
    if suite_name not in suite_names:
        return {}

    target_func = suite_name
    if target_func is None:
        return {}

    data = {}
    for bench in results.get("benchmarks", []):
        func, params = _parse_benchmark_name(bench["name"])
        if func == target_func:
            size = int(params.replace(",", ""))
            median_time_ms = bench["stats"]["median"] * 1000
            data[size] = median_time_ms
    return data


def extract_encoding_data(
    results: dict,
) -> dict[str, dict[int, float]]:
    """Extract roundtrip encoding data by method and size.

    Returns dict mapping encoding method to {size: median_time_ms}.
    """
    data: dict[str, dict[int, float]] = {}
    for bench in results.get("benchmarks", []):
        func, params = _parse_benchmark_name(bench["name"])
        if func != "roundtrip":
            continue

        # params is like "json_100" or "base64_1,000,000"
        parts = params.rsplit("_", 1)
        if len(parts) != 2:
            continue

        method, size_str = parts
        size = int(size_str.replace(",", ""))
        median_time_ms = bench["stats"]["median"] * 1000

        if method not in data:
            data[method] = {}
        data[method][size] = median_time_ms

    return data


def _format_time_label(time_ms: float) -> str:
    """Format time in ms with appropriate precision for bar labels."""
    if time_ms >= 100:
        return f"{time_ms:.0f}"
    elif time_ms >= 10:
        return f"{time_ms:.1f}"
    elif time_ms >= 1:
        return f"{time_ms:.2f}"
    elif time_ms >= 0.1:
        return f"{time_ms:.2f}"
    else:
        return f"{time_ms:.3f}"


def _size_to_label(bytes_size: int) -> str:
    """Convert byte size to human-readable label."""
    if bytes_size < 1024:
        data_label = f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        data_label = f"{bytes_size // 1024}kB"
    elif bytes_size < 1024 * 1024 * 1024:
        data_label = f"{bytes_size // (1024 * 1024)}MB"
    else:
        data_label = f"{bytes_size // (1024 * 1024 * 1024)}GB"
    return data_label


def generate_guidance_plot(output_path: Path, benchmark_results: dict) -> None:
    """Generate a plot showing when Tesseract overhead is acceptable.

    This helps users understand if their workload is a good fit based on
    computation time and I/O data size. Shows curves for different interaction modes
    and data sizes.
    """
    _, ax = plt.subplots(figsize=(12, 7))

    computation_times = np.logspace(-4, 4, 100)

    from_api_data = extract_suite_data(benchmark_results, "from_tesseract_api")
    containerized_http_data = extract_suite_data(
        benchmark_results, "containerized_http"
    )
    containerized_cli_data = extract_suite_data(benchmark_results, "containerized_cli")

    # Colorblind-safe palette (blue / orange / purple)
    modes = [
        ("Non-containerized, in-memory", from_api_data, "#0072B2"),
        ("Containerized, json+base64 via HTTP", containerized_http_data, "#E69F00"),
        ("Containerized, json+binref via CLI", containerized_cli_data, "#9467BD"),
    ]

    io_sizes = [
        (1024, ":"),  # 1kB, dotted line
        (1024 * 1024, "--"),  # 1MB, dashed line
        (1024 * 1024 * 1024, "-"),  # 1GB, solid line
    ]

    for mode_name, mode_data, color in modes:
        if not mode_data:
            continue

        sizes_arr = np.array(sorted(mode_data.keys()), dtype=float)
        times_arr = np.array([mode_data[int(s)] for s in sizes_arr])
        # Affine model: overhead = intercept + slope * size
        # Use min observed time as intercept (fixed overhead floor),
        # then fit slope through the residuals.
        intercept = float(np.min(times_arr))
        residuals = times_arr - intercept
        slope = float(np.dot(sizes_arr, residuals) / np.dot(sizes_arr, sizes_arr))

        for size, linestyle in io_sizes:
            array_size = size // 8  # Convert bytes to number of float64 elements
            overhead_ms = intercept + slope * array_size

            data_label = _size_to_label(size)

            overhead_pct = (overhead_ms / computation_times * 1e-3) * 100
            ax.semilogx(
                computation_times,
                overhead_pct,
                label=f"{mode_name} ({data_label})",
                linewidth=1.5,
                color=color,
                linestyle=linestyle,
            )

    # Add guidance regions (colorblind-safe)
    ax.axhspan(0, 10, alpha=0.15, color="#0072B2")
    ax.axhspan(10, 50, alpha=0.15, color="#E69F00")
    ax.axhspan(50, 100, alpha=0.15, color="#D55E00")

    ax.text(
        1.2e-4,
        5,
        "Excellent fit (<10% overhead)",
        fontsize=10,
        color="#0072B2",
        ha="left",
        va="center",
        path_effects=_outline,
    )
    ax.text(
        1.2e-4,
        30,
        "Good fit (10-50% overhead)",
        fontsize=10,
        color="#E69F00",
        ha="left",
        va="center",
        path_effects=_outline,
    )
    ax.text(
        1.2e-4,
        75,
        "Consider alternatives (>50%)",
        fontsize=10,
        color="#D55E00",
        ha="left",
        va="center",
        path_effects=_outline,
    )

    time_markers = [
        (1e-3, "1ms"),
        (1, "1s"),
        (60, "1min"),
        (3600, "1hr"),
    ]
    for time_s, label in time_markers:
        ax.axvline(x=time_s, color="0.4", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.text(time_s, 101, label, fontsize=9, ha="center", va="bottom", color="black")

    ax.set_xlabel("Computation Time (s)", fontsize=12)
    ax.set_ylabel("Tesseract Overhead (%)", fontsize=12)
    ax.set_title("When is Tesseract a Good Fit?", fontsize=18, pad=45)
    ax.text(
        0.5,
        1.08,
        "Overhead as percentage of computation time, by interaction mode and I/O size",
        fontsize=12,
        ha="center",
        va="center",
        transform=ax.transAxes,
        path_effects=_outline,
    )
    ax.legend(loc="upper center", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xlim(1e-4, 1e4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")


def generate_encoding_comparison_plot(
    output_path: Path, benchmark_results: dict
) -> None:
    """Generate plot comparing array encoding methods.

    This shows the performance of json, base64, and binref encodings
    for arrays of varying sizes.
    """
    _, ax = plt.subplots(figsize=(10, 6))

    sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    size_labels = ["10", "100", "1K", "10K", "100K", "1M", "10M"]

    roundtrip_data = extract_encoding_data(benchmark_results)

    if not roundtrip_data:
        print(f"Warning: No array encoding data found, skipping {output_path}")
        return

    json_times = [roundtrip_data.get("json", {}).get(s, 0) for s in sizes]
    base64_times = [roundtrip_data.get("base64", {}).get(s, 0) for s in sizes]
    binref_times = [roundtrip_data.get("binref", {}).get(s, 0) for s in sizes]

    x = np.arange(len(sizes))
    width = 0.25

    bars1 = ax.bar(x - width, json_times, width, label="JSON", color="#D55E00")
    bars2 = ax.bar(x, base64_times, width, label="Base64", color="#0072B2")
    bars3 = ax.bar(x + width, binref_times, width, label="Binref", color="#009E73")

    ax.set_xlabel("Array Size (elements)", fontsize=12)
    ax.set_title("Array Encoding Performance", fontsize=18, pad=30)
    ax.text(
        0.5,
        1.04,
        "Encode + Decode roundtrip, lower is better",
        fontsize=12,
        ha="center",
        va="center",
        transform=ax.transAxes,
        path_effects=_outline,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.legend(
        loc="upper left",
        title="Encoding",
        title_fontproperties={"weight": "600"},
        facecolor="white",
        edgecolor="white",
        alignment="left",
    )
    ax.set_yscale("log")
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y", which="major")
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    _format_time_label(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    path_effects=_outline,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")


def generate_overhead_comparison_plot(
    output_path: Path, benchmark_results: dict
) -> None:
    """Generate plot comparing overhead across interaction modes.

    This shows the breakdown of overhead by array size for each mode.
    """
    _, ax = plt.subplots(figsize=(10, 6))

    sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    size_labels = ["10", "100", "1K", "10K", "100K", "1M", "10M", "100M"]

    from_api_data = extract_suite_data(benchmark_results, "from_tesseract_api")
    containerized_http_data = extract_suite_data(
        benchmark_results, "containerized_http"
    )
    containerized_cli_data = extract_suite_data(benchmark_results, "containerized_cli")

    from_api = [from_api_data.get(s, 0) for s in sizes]
    containerized_http = [containerized_http_data.get(s, 0) for s in sizes]
    containerized_cli = [containerized_cli_data.get(s, 0) for s in sizes]

    x = np.arange(len(sizes))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        from_api,
        width,
        label="Non-containerized, in-memory",
        color="#0072B2",
    )
    bars2 = ax.bar(
        x,
        containerized_http,
        width,
        label="Containerized, json+base64 via HTTP",
        color="#E69F00",
    )
    bars3 = ax.bar(
        x + width,
        containerized_cli,
        width,
        label="Containerized, json+binref via CLI",
        color="#9467BD",
    )

    ax.set_xlabel("Array Size (elements)", fontsize=12)
    ax.set_title("Tesseract Overhead by Interaction Mode", fontsize=18, pad=30)
    ax.text(
        0.5,
        1.04,
        "No-op Tesseract, lower is better",
        fontsize=12,
        ha="center",
        va="center",
        transform=ax.transAxes,
        path_effects=_outline,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 2e5)
    ax.legend(
        loc="upper left",
        title="Interaction mode",
        title_fontproperties={"weight": "600"},
        facecolor="white",
        edgecolor="white",
        alignment="left",
    )
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y", which="major")
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    _format_time_label(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    path_effects=_outline,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots for documentation"
    )
    parser.add_argument(
        "benchmark_file",
        type=str,
        help="Path to pytest-benchmark JSON results file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(here, "plots"),
        help="Output directory for plots",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = load_benchmark_results(Path(args.benchmark_file))

    generate_guidance_plot(output_dir / "benchmark_guidance.png", benchmark_results)
    generate_overhead_comparison_plot(
        output_dir / "benchmark_overhead.png", benchmark_results
    )
    generate_encoding_comparison_plot(
        output_dir / "benchmark_encoding.png", benchmark_results
    )

    print(f"\nPlots generated in {output_dir}/")


if __name__ == "__main__":
    main()
