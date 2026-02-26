# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization script for the Fortran heat equation solver.

This script demonstrates how to visualize the output of the fortran_heat
Tesseract, showing temperature evolution over time as both a static heatmap
and an animated GIF.

Usage:
    # Run with a local Tesseract (requires built image)
    python visualize.py

    # Or connect to a running server
    python visualize.py --url http://localhost:8000
"""

import argparse
import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def run_simulation(tesseract, params: dict) -> dict:
    """Run the heat equation simulation with given parameters."""
    return tesseract.apply(inputs=params)


def plot_temperature_evolution(x, time, temperature, output_path: Path | None = None):
    """Create a heatmap showing temperature evolution over space and time.

    Args:
        x: Spatial coordinates array, shape (n_points,)
        time: Time values array, shape (n_steps+1,)
        temperature: Temperature history, shape (n_steps+1, n_points)
        output_path: If provided, save the figure to this path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap with time on y-axis, space on x-axis
    im = ax.imshow(
        temperature,
        aspect="auto",
        origin="lower",
        extent=[x.min(), x.max(), time.min(), time.max()],
        cmap="hot",
        norm=Normalize(vmin=temperature.min(), vmax=temperature.max()),
    )

    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Time t [s]")
    ax.set_title("1D Heat Equation: Temperature Evolution")

    fig.colorbar(im, ax=ax, label="Temperature [K]")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")

    return fig, ax


def plot_temperature_profiles(
    x, time, temperature, n_profiles: int = 5, output_path: Path | None = None
):
    """Plot temperature profiles at selected time steps.

    Args:
        x: Spatial coordinates array
        time: Time values array
        temperature: Temperature history
        n_profiles: Number of time profiles to show
        output_path: If provided, save the figure to this path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select evenly spaced time indices
    n_steps = len(time) - 1
    indices = np.linspace(0, n_steps, n_profiles, dtype=int)

    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))

    for i, idx in enumerate(indices):
        ax.plot(
            x,
            temperature[idx],
            color=colors[i],
            linewidth=2,
            label=f"t = {time[idx]:.3f} s",
        )

    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature Profiles at Different Times")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved profiles to {output_path}")

    return fig, ax


def make_animation(x, time, temperature):
    """Create an animation of temperature evolution.

    Args:
        x: Spatial coordinates array
        time: Time values array
        temperature: Temperature history, shape (n_steps+1, n_points)

    Returns:
        matplotlib.animation.FuncAnimation object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up static elements
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(temperature.min() - 5, temperature.max() + 5)
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.grid(True, alpha=0.3)

    # Create line object for animation
    (line,) = ax.plot([], [], "b-", linewidth=2)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12)
    ax.set_title("1D Heat Equation Simulation")

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(frame):
        line.set_data(x, temperature[frame])
        time_text.set_text(f"t = {time[frame]:.4f} s")
        return line, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time),
        init_func=init,
        interval=50,  # 50ms between frames = 20 fps
        blit=True,
    )

    plt.tight_layout()
    return anim


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Fortran heat equation solver output"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of running Tesseract server (if not provided, uses from_image)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=51,
        help="Number of spatial grid points",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="Number of time steps",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Thermal diffusivity [m^2/s]",
    )
    parser.add_argument(
        "--t-left",
        type=float,
        default=100.0,
        help="Left boundary temperature [K]",
    )
    parser.add_argument(
        "--t-right",
        type=float,
        default=0.0,
        help="Right boundary temperature [K]",
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save animation as GIF",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful for headless environments)",
    )
    args = parser.parse_args()

    # Import Tesseract SDK
    from tesseract_core import Tesseract

    # Set up simulation parameters
    params = {
        "n_points": args.n_points,
        "n_steps": args.n_steps,
        "alpha": args.alpha,
        "length": 1.0,
        "dt": 0.005,  # Time step (must satisfy CFL: alpha*dt/dx^2 <= 0.5)
        "t_left": args.t_left,
        "t_right": args.t_right,
        "initial_temperature": 20.0,  # Room temperature initial condition
    }

    # Create output directory
    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = this_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Run simulation
    print("Running heat equation simulation...")
    print(f"  Parameters: n_points={params['n_points']}, n_steps={params['n_steps']}")
    print(
        f"  Boundary conditions: T_left={params['t_left']}K, T_right={params['t_right']}K"
    )

    if args.url:
        tesseract = Tesseract.from_url(args.url)
        output = run_simulation(tesseract, params)
    else:
        with Tesseract.from_image("fortran-heat") as tesseract:
            output = run_simulation(tesseract, params)

    # Extract results
    x = np.array(output["x"])
    time = np.array(output["time"])
    temperature = np.array(output["temperature"])

    print(f"  Simulation complete: {len(time)} time steps, {len(x)} spatial points")
    print(
        f"  Final temperature range: [{temperature[-1].min():.1f}, {temperature[-1].max():.1f}] K"
    )

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Heatmap of temperature evolution
    plot_temperature_evolution(
        x, time, temperature, output_path=plots_dir / "temperature_heatmap.png"
    )

    # 2. Temperature profiles at different times
    plot_temperature_profiles(
        x,
        time,
        temperature,
        n_profiles=6,
        output_path=plots_dir / "temperature_profiles.png",
    )

    # 3. Animation
    print("Creating animation...")
    anim = make_animation(x, time, temperature)

    if args.save_animation:
        gif_path = plots_dir / "heat_evolution.gif"
        writer = animation.PillowWriter(fps=20)
        anim.save(gif_path, writer=writer)
        print(f"Saved animation to {gif_path}")

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
