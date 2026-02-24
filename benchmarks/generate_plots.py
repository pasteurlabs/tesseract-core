#!/usr/bin/env python
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate performance plots for documentation.

This script generates plots showing Tesseract overhead for various workload
sizes, helping users understand if their workload is a good fit.

Usage:
    python generate_plots.py --benchmark-file PATH [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(benchmark_file: Path) -> dict:
    """Load benchmark results from JSON file."""
    with open(benchmark_file) as f:
        return json.load(f)


def extract_suite_data(results: dict, suite_name: str) -> dict[int, float]:
    """Extract mean times by array size for a given suite.

    Returns dict mapping array size to mean time in ms.
    """
    data = {}
    for result in results.get("results", []):
        name = result["name"]
        if name.startswith(f"{suite_name}/apply_"):
            size_str = name.split("apply_")[1]
            size = int(size_str.replace(",", ""))
            mean_time_ms = result["mean_time_s"] * 1000
            data[size] = mean_time_ms
    return data


def _format_time_label(time_ms: float) -> str:
    """Format time in ms with appropriate precision for bar labels."""
    if time_ms >= 1000:
        return f"{time_ms:.0f}"
    elif time_ms >= 100:
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

    # Extract overhead values from benchmark results for all three modes
    from_api_data = extract_suite_data(benchmark_results, "from_tesseract_api")
    containerized_http_data = extract_suite_data(
        benchmark_results, "containerized_http"
    )
    containerized_cli_data = extract_suite_data(benchmark_results, "containerized_cli")

    # Use same categories as overhead plot with consistent colors
    modes = [
        ("Non-containerized, in-memory", from_api_data, "#2ecc71"),
        ("Containerized, json+base64 via HTTP", containerized_http_data, "#3498db"),
        ("Containerized, json+binref via CLI", containerized_cli_data, "#e74c3c"),
    ]

    # Representative I/O sizes with different linestyles
    io_sizes = [
        (1024, ":"),  # 1kB, dotted line
        (1024 * 1024, "--"),  # 1MB, dashed line
        (1024 * 1024 * 1024, "-"),  # 1GB, solid line
    ]

    for mode_name, mode_data, color in modes:
        if not mode_data:
            continue

        for size, linestyle in io_sizes:
            # Get actual overhead from benchmarks
            available = sorted(mode_data.keys())
            array_size = size // 8  # Convert bytes to number of float64 elements
            closest = min(available, key=lambda x: abs(x - array_size))
            overhead_ms = mode_data[closest]

            # Scale overhead linearly to match target
            overhead_ms *= array_size / closest

            data_label = _size_to_label(size)

            # Plot overhead percentage curve
            overhead_pct = (overhead_ms / computation_times * 1e-3) * 100
            ax.semilogx(
                computation_times,
                overhead_pct,
                label=f"{mode_name} ({data_label})",
                linewidth=1.5,
                color=color,
                linestyle=linestyle,
            )

    # Add guidance regions
    ax.axhspan(0, 10, alpha=0.15, color="green")
    ax.axhspan(10, 50, alpha=0.15, color="yellow")
    ax.axhspan(50, 100, alpha=0.15, color="red")

    ax.text(
        1.2e-4,
        5,
        "Excellent fit (<10% overhead)",
        fontsize=10,
        color="darkgreen",
        ha="left",
        va="center",
    )
    ax.text(
        1.2e-4,
        30,
        "Good fit (10-50% overhead)",
        fontsize=10,
        color="darkorange",
        ha="left",
        va="center",
    )
    ax.text(
        1.2e-4,
        75,
        "Consider alternatives (>50%)",
        fontsize=10,
        color="darkred",
        ha="left",
        va="center",
    )

    # Add human-readable time markers
    time_markers = [
        (1e-3, "1ms"),
        (1, "1s"),
        (60, "1min"),
        (3600, "1hr"),
    ]
    for time_s, label in time_markers:
        ax.axvline(x=time_s, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.text(time_s, 102, label, fontsize=9, ha="center", va="bottom", color="gray")

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

    # Array sizes to display
    sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    size_labels = ["10", "100", "1K", "10K", "100K", "1M", "10M"]

    # Extract roundtrip data (most representative of real usage)
    roundtrip_data = {}
    for result in benchmark_results.get("results", []):
        name = result["name"]
        if name.startswith("array_roundtrip/roundtrip_"):
            # Parse: array_roundtrip/roundtrip_{method}_{size}
            parts = name.split("roundtrip_")[1]
            method, size_str = parts.rsplit("_", 1)
            size = int(size_str.replace(",", ""))
            mean_time_ms = result["mean_time_s"] * 1000
            if method not in roundtrip_data:
                roundtrip_data[method] = {}
            roundtrip_data[method][size] = mean_time_ms

    if not roundtrip_data:
        print(f"Warning: No array encoding data found, skipping {output_path}")
        return

    json_times = [roundtrip_data.get("json", {}).get(s, 0) for s in sizes]
    base64_times = [roundtrip_data.get("base64", {}).get(s, 0) for s in sizes]
    binref_times = [roundtrip_data.get("binref", {}).get(s, 0) for s in sizes]

    x = np.arange(len(sizes))
    width = 0.25

    bars1 = ax.bar(x - width, json_times, width, label="JSON", color="#e74c3c")
    bars2 = ax.bar(x, base64_times, width, label="Base64", color="#3498db")
    bars3 = ax.bar(x + width, binref_times, width, label="Binref", color="#2ecc71")

    ax.set_xlabel("Array Size (elements)", fontsize=12)
    ax.set_ylabel("Roundtrip Time (ms)", fontsize=12)
    ax.set_title(
        "Array Encoding Performance\n(Encode + Decode roundtrip, lower is better)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    # Add value labels on bars
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

    # Array sizes to display
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
        color="#2ecc71",
    )
    bars2 = ax.bar(
        x,
        containerized_http,
        width,
        label="Containerized, json+base64 via HTTP",
        color="#3498db",
    )
    bars3 = ax.bar(
        x + width,
        containerized_cli,
        width,
        label="Containerized, json+binref via CLI",
        color="#e74c3c",
    )

    ax.set_xlabel("Array Size (elements)", fontsize=12)
    ax.set_ylabel("Overhead (ms)", fontsize=12)
    ax.set_title(
        "Tesseract Overhead by Interaction Mode\n(No-op Tesseract, lower is better)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e5)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
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
        help="Path to benchmark results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/img",
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
