# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract wrapper for a Fortran 1D heat equation solver.

This example demonstrates how to wrap Fortran simulation code
as a Tesseract using subprocess-based integration. The solver uses an
explicit finite difference scheme to solve the 1D transient heat equation:

    dT/dt = alpha * d^2T/dx^2

"""

import struct
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from tesseract_core.runtime import Array, Float64


class InputSchema(BaseModel):
    """Input parameters for the 1D transient heat equation solver.

    Solves: dT/dt = alpha * d^2T/dx^2
    with Dirichlet boundary conditions on a domain [0, length].
    """

    n_points: int = Field(
        default=51,
        ge=3,
        le=1001,
        description="Number of spatial grid points. Must be >= 3.",
    )
    n_steps: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of time steps to simulate.",
    )
    alpha: float = Field(
        default=0.01,
        gt=0.0,
        description="Thermal diffusivity [m^2/s]. Material property that characterizes "
        "how quickly heat diffuses through a material.",
    )
    length: float = Field(
        default=1.0,
        gt=0.0,
        description="Domain length [m].",
    )
    dt: float = Field(
        default=0.001,
        gt=0.0,
        description="Time step size [s].",
    )
    t_left: float = Field(
        default=100.0,
        description="Temperature at left boundary (x=0) [K or degC].",
    )
    t_right: float = Field(
        default=0.0,
        description="Temperature at right boundary (x=length) [K or degC].",
    )
    initial_temperature: float = Field(
        default=0.0,
        description="Initial temperature throughout the domain [K or degC].",
    )

    @model_validator(mode="after")
    def check_stability(self) -> Self:
        """Check CFL stability condition for explicit scheme.

        The explicit finite difference scheme is stable only when
        r = alpha * dt / dx^2 <= 0.5 (von Neumann stability analysis).
        """
        dx = self.length / (self.n_points - 1)
        r = self.alpha * self.dt / (dx * dx)
        if r > 0.5:
            raise ValueError(
                f"Stability condition violated: r = alpha*dt/dx^2 = {r:.4f} > 0.5. "
                f"Either reduce dt, reduce alpha, or increase n_points to satisfy "
                f"the CFL condition for the explicit scheme."
            )
        return self


class OutputSchema(BaseModel):
    """Output from the heat equation solver."""

    x: Array[(None,), Float64] = Field(
        description="Spatial coordinates [m]. Shape: (n_points,)",
    )
    temperature: Array[(None, None), Float64] = Field(
        description="Temperature history [K or degC]. Shape: (n_steps+1, n_points). "
        "First row is the initial condition, last row is the final state.",
    )
    final_temperature: Array[(None,), Float64] = Field(
        description="Final temperature profile [K or degC]. Shape: (n_points,)",
    )
    time: Array[(None,), Float64] = Field(
        description="Time values for each saved step [s]. Shape: (n_steps+1,)",
    )


def _get_solver_path() -> Path:
    """Get path to the compiled Fortran solver executable."""
    return Path("/app/fortran/heat_solver")


def _write_input_file(filepath: Path, inputs: InputSchema) -> None:
    """Write solver input parameters to a text file."""
    with open(filepath, "w") as f:
        f.write(f"n_points {inputs.n_points}\n")
        f.write(f"n_steps {inputs.n_steps}\n")
        f.write(f"alpha {inputs.alpha}\n")
        f.write(f"length {inputs.length}\n")
        f.write(f"dt {inputs.dt}\n")
        f.write(f"t_left {inputs.t_left}\n")
        f.write(f"t_right {inputs.t_right}\n")
        f.write(f"initial_temperature {inputs.initial_temperature}\n")


def _read_output_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read solver output from binary file.

    The Fortran solver writes:
    - n_points (int32)
    - n_steps (int32)
    - x array (float64, n_points)
    - T_history array (float64, n_points x (n_steps+1), column-major)

    Returns:
        x: Spatial coordinates array, shape (n_points,)
        T_history: Temperature history array, shape (n_steps+1, n_points)
    """
    with open(filepath, "rb") as f:
        # Read header (Fortran writes native int, typically 4 bytes)
        n_points = struct.unpack("i", f.read(4))[0]
        n_steps = struct.unpack("i", f.read(4))[0]

        # Read spatial coordinates
        x = np.frombuffer(f.read(n_points * 8), dtype=np.float64)

        # Read temperature history
        # Fortran stores arrays in column-major order
        T_flat = np.frombuffer(f.read(n_points * (n_steps + 1) * 8), dtype=np.float64)
        # Reshape from column-major (Fortran) to row-major (C/NumPy)
        # In Fortran: T_history(n_points, n_steps+1) stored column-major
        # We want: T_history[n_steps+1, n_points] in row-major
        T_history = T_flat.reshape((n_points, n_steps + 1), order="F").T

    return x, T_history


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the Fortran heat equation solver.

    This function:
    1. Writes input parameters to a temporary file
    2. Calls the compiled Fortran executable via subprocess
    3. Reads the binary output file and returns results as arrays
    """
    solver_path = _get_solver_path()

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_file = tmpdir_path / "input.txt"
        output_file = tmpdir_path / "output.bin"

        # Write input parameters
        _write_input_file(input_file, inputs)

        # Run Fortran solver with real-time output streaming
        process = subprocess.Popen(
            [str(solver_path), str(input_file), str(output_file)],
            stdout=None,  # Inherit stdout - streams to parent process
            stderr=None,  # Inherit stderr - streams to parent process
        )
        returncode = process.wait()

        if returncode != 0:
            raise RuntimeError(f"Fortran solver failed with return code {returncode}.")

        # Read output
        x, T_history = _read_output_file(output_file)

    # Compute time array
    time = np.arange(inputs.n_steps + 1) * inputs.dt

    return OutputSchema(
        x=x,
        temperature=T_history,
        final_temperature=T_history[-1, :],
        time=time,
    )
