# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization script for the MATLAB spring-mass-damper solver.

This script demonstrates how to visualize the output of the matlab-springmass
Tesseract, showing displacement and velocity time histories with system
characteristics annotated.

Usage:
    # Run with a local Tesseract (requires built image)
    python visualize.py

    # Or connect to a running server
    python visualize.py --url http://localhost:8000
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_simulation(tesseract, params: dict) -> dict:
    """Run the spring-mass-damper simulation with given parameters."""
    return tesseract.apply(inputs=params)


def plot_response(
    time,
    displacement,
    velocity,
    steady_state,
    damping_ratio,
    natural_frequency,
    output_path: Path | None = None,
):
    """Plot displacement and velocity time histories.

    Args:
        time: Time array [s]
        displacement: Displacement array [m]
        velocity: Velocity array [m/s]
        steady_state: Analytical steady-state displacement [m]
        damping_ratio: Damping ratio [-]
        natural_frequency: Natural frequency [rad/s]
        output_path: If provided, save the figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Displacement
    ax1.plot(time, displacement, "b-", linewidth=2, label="x(t)")
    ax1.axhline(
        y=steady_state,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Steady state = {steady_state:.4f} m",
    )
    ax1.set_ylabel("Displacement [m]")
    ax1.set_title(
        f"Spring-Mass-Damper Response "
        f"(\u03b6 = {damping_ratio:.3f}, "
        f"\u03c9_n = {natural_frequency:.2f} rad/s)"
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Velocity
    ax2.plot(time, velocity, "g-", linewidth=2, label="v(t)")
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Velocity [m/s]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved response plot to {output_path}")

    return fig, (ax1, ax2)


def plot_phase_portrait(displacement, velocity, output_path: Path | None = None):
    """Plot phase portrait (velocity vs displacement).

    Args:
        displacement: Displacement array [m]
        velocity: Velocity array [m/s]
        output_path: If provided, save the figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(displacement, velocity, "b-", linewidth=1.5)
    ax.plot(displacement[0], velocity[0], "go", markersize=10, label="Start")
    ax.plot(displacement[-1], velocity[-1], "rs", markersize=10, label="End")

    ax.set_xlabel("Displacement [m]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Phase Portrait")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved phase portrait to {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MATLAB spring-mass-damper solver output"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of running Tesseract server (if not provided, uses from_image)",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Mass [kg]",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="Damping coefficient [N*s/m]",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=10.0,
        help="Spring stiffness [N/m]",
    )
    parser.add_argument(
        "--force",
        type=float,
        default=1.0,
        help="Step force amplitude [N]",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=10.0,
        help="Simulation end time [s]",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful for headless environments)",
    )
    args = parser.parse_args()

    from tesseract_core import Tesseract

    params = {
        "mass": args.mass,
        "damping": args.damping,
        "stiffness": args.stiffness,
        "force_amplitude": args.force,
        "t_end": args.t_end,
        "n_output_points": 500,
    }

    # Create output directory
    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = this_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Run simulation
    print("Running spring-mass-damper simulation...")
    print(
        f"  m={params['mass']} kg, c={params['damping']} N*s/m, "
        f"k={params['stiffness']} N/m, F0={params['force_amplitude']} N"
    )

    if args.url:
        tesseract = Tesseract.from_url(args.url)
        output = run_simulation(tesseract, params)
    else:
        with Tesseract.from_image("matlab-springmass") as tesseract:
            output = run_simulation(tesseract, params)

    # Extract results
    time = np.array(output["time"])
    displacement = np.array(output["displacement"])
    velocity = np.array(output["velocity"])
    steady_state = output["steady_state"]
    damping_ratio = output["damping_ratio"]
    natural_frequency = output["natural_frequency"]

    print(
        f"  Damping ratio: {damping_ratio:.4f} "
        f"({'underdamped' if damping_ratio < 1 else 'overdamped' if damping_ratio > 1 else 'critical'})"
    )
    print(f"  Natural frequency: {natural_frequency:.4f} rad/s")
    print(f"  Steady state: {steady_state:.6f} m")

    # Create visualizations
    print("\nGenerating visualizations...")

    plot_response(
        time,
        displacement,
        velocity,
        steady_state,
        damping_ratio,
        natural_frequency,
        output_path=plots_dir / "response.png",
    )

    plot_phase_portrait(
        displacement,
        velocity,
        output_path=plots_dir / "phase_portrait.png",
    )

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
