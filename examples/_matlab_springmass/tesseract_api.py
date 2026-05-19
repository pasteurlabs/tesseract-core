# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract wrapper for a MATLAB spring-mass-damper ODE solver.

This example demonstrates how to wrap MATLAB simulation code
as a Tesseract using the official MathWorks MATLAB Docker image.
The solver uses MATLAB's ode45 (Dormand-Prince Runge-Kutta) to
solve the damped harmonic oscillator equation:

    m * x'' + c * x' + k * x = F(t)

with a unit step force input F(t) = F0 for t >= 0.
"""

import glob
import json
import math
import os
import pwd
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float64

# Path to the MATLAB binary. Auto-discovers /opt/matlab/<release>/bin/matlab
# (uses the highest version if multiple are installed). Override with the
# MATLAB_BIN env var for a non-standard path.
_matlab_candidates = sorted(glob.glob("/opt/matlab/*/bin/matlab"))
MATLAB_BIN = os.environ.get(
    "MATLAB_BIN",
    _matlab_candidates[-1] if _matlab_candidates else "/opt/matlab/R2026a/bin/matlab",
)

# If MATLAB_USERNAME is set, rename the 'ubuntu' user in /etc/passwd (and
# /etc/group) to that username at startup. MATLAB's file-based license check
# picks the 'ubuntu' entry out of /etc/passwd regardless of the running UID,
# so renaming it lets the check see the correct user without baking the
# username into the image. Requires the container to run as root, e.g.
# `tesseract run --user 0:0 --env MATLAB_USERNAME=<your-license-user> ...`.
_MATLAB_USERNAME = os.environ.get("MATLAB_USERNAME")
if _MATLAB_USERNAME:
    # /etc/passwd is what MATLAB's license check reads directly. /tmp/passwd
    # is what the tesseract template's libnss_wrapper redirects NSS lookups
    # to (used by pwd.getpwnam below). Both must be updated.
    for _path in ("/etc/passwd", "/etc/group", "/tmp/passwd", "/tmp/group"):
        try:
            with open(_path) as _f:
                _content = _f.read()
            _new = re.sub(
                r"^ubuntu:", f"{_MATLAB_USERNAME}:", _content, flags=re.MULTILINE
            )
            if _new != _content:
                with open(_path, "w") as _f:
                    _f.write(_new)
        except (PermissionError, OSError, FileNotFoundError):
            # Not running as root, file doesn't exist, or rename already done.
            pass

# Path to the .m source file copied into the container at build time.
SOLVER_DIR = Path("/tesseract/matlab")


class InputSchema(BaseModel):
    """Input parameters for the spring-mass-damper ODE solver.

    Solves: m * x'' + c * x' + k * x = F0 (step force)
    with initial conditions x(0) = 0, x'(0) = 0.
    """

    mass: float = Field(
        default=1.0,
        gt=0.0,
        description="Mass [kg].",
    )
    damping: float = Field(
        default=0.5,
        ge=0.0,
        description="Damping coefficient [N*s/m]. Set to 0 for undamped oscillation.",
    )
    stiffness: float = Field(
        default=10.0,
        gt=0.0,
        description="Spring stiffness [N/m].",
    )
    force_amplitude: float = Field(
        default=1.0,
        description="Step force amplitude [N]. Applied as a constant force for t >= 0.",
    )
    t_end: float = Field(
        default=10.0,
        gt=0.0,
        description="Simulation end time [s].",
    )
    n_output_points: int = Field(
        default=200,
        ge=10,
        le=10000,
        description="Number of output time points.",
    )


class OutputSchema(BaseModel):
    """Output from the spring-mass-damper solver."""

    time: Array[(None,), Float64] = Field(
        description="Time values [s]. Shape: (n_output_points,)",
    )
    displacement: Array[(None,), Float64] = Field(
        description="Displacement x(t) [m]. Shape: (n_output_points,)",
    )
    velocity: Array[(None,), Float64] = Field(
        description="Velocity x'(t) [m/s]. Shape: (n_output_points,)",
    )
    steady_state: float = Field(
        description="Analytical steady-state displacement F0/k [m].",
    )
    damping_ratio: float = Field(
        description="Damping ratio zeta = c / (2 * sqrt(k * m)) [-]. "
        "< 1: underdamped, = 1: critically damped, > 1: overdamped.",
    )
    natural_frequency: float = Field(
        description="Natural frequency omega_n = sqrt(k / m) [rad/s].",
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the MATLAB spring-mass-damper solver.

    This function:
    1. Writes input parameters to a temporary JSON file
    2. Calls MATLAB via `matlab -batch` to run the solver
    3. Reads the JSON output file and returns results
    """
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_file = tmpdir_path / "input.json"
        output_file = tmpdir_path / "output.json"

        # If we'll spawn MATLAB as a different user, hand them the tmpdir.
        if _MATLAB_USERNAME:
            try:
                _pw = pwd.getpwnam(_MATLAB_USERNAME)
                os.chown(tmpdir, _pw.pw_uid, _pw.pw_gid)
            except (KeyError, PermissionError):
                pass

        # Write input parameters as JSON
        input_data = {
            "mass": inputs.mass,
            "damping": inputs.damping,
            "stiffness": inputs.stiffness,
            "force_amplitude": inputs.force_amplitude,
            "t_end": inputs.t_end,
            "n_output_points": inputs.n_output_points,
        }
        input_file.write_text(json.dumps(input_data))

        # Build the MATLAB command.
        # addpath so MATLAB can find spring_mass_damper.m,
        # then call it with input/output file paths.
        matlab_cmd = (
            f"addpath('{SOLVER_DIR}'); "
            f"spring_mass_damper('{input_file}', '{output_file}')"
        )

        # If MATLAB_USERNAME is set, run MATLAB as that user (UID/GID resolved
        # via getpwnam after the /etc/passwd rename above). Otherwise inherit.
        popen_kwargs = {}
        if _MATLAB_USERNAME:
            try:
                pw = pwd.getpwnam(_MATLAB_USERNAME)
                env = os.environ.copy()
                env.update(
                    {
                        "HOME": pw.pw_dir,
                        "USER": _MATLAB_USERNAME,
                        "LOGNAME": _MATLAB_USERNAME,
                    }
                )
                popen_kwargs.update(user=pw.pw_uid, group=pw.pw_gid, env=env)
            except KeyError:
                pass

        process = subprocess.Popen(
            [MATLAB_BIN, "-batch", matlab_cmd],
            stdout=None,  # Inherit stdout — streams to parent process
            stderr=None,  # Inherit stderr — streams to parent process
            **popen_kwargs,
        )
        returncode = process.wait()

        if returncode != 0:
            raise RuntimeError(f"MATLAB solver failed with return code {returncode}.")

        # Read output JSON
        output_data = json.loads(output_file.read_text())

    time = np.array(output_data["time"], dtype=np.float64)
    displacement = np.array(output_data["displacement"], dtype=np.float64)
    velocity = np.array(output_data["velocity"], dtype=np.float64)

    # Compute analytical quantities
    steady_state = inputs.force_amplitude / inputs.stiffness
    natural_frequency = math.sqrt(inputs.stiffness / inputs.mass)
    damping_ratio = inputs.damping / (2.0 * math.sqrt(inputs.stiffness * inputs.mass))

    return OutputSchema(
        time=time,
        displacement=displacement,
        velocity=velocity,
        steady_state=steady_state,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
    )
