# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return a large simulation trajectory without holding it all in memory.

A transient solver often produces far more output than fits comfortably in RAM:
a 2D field snapshotted at every timestep is a 3D array that grows without bound
as the simulation runs. The usual options are both bad -- keep every snapshot in
memory and risk running out, or stream them to disk during the solve but then
read the whole thing *back* into memory just to hand it to the Tesseract runtime,
which promptly writes it out again.

``BinrefArray`` removes that round-trip. The solver writes each snapshot straight
to a binref buffer on disk as it is computed, and returns a lightweight
reference. When the client asks for ``json+binref`` output, the on-disk bytes are
forwarded verbatim -- never read back into the server's memory. The output field
is an ordinary ``Array``, so nothing about the schema is special.

This example runs a tiny explicit heat-diffusion solver on a 2D grid, checkpoints
every timestep, and returns the trajectory plus the final field.
"""

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float64
from tesseract_core.runtime.experimental import BinrefArray, BinrefWriter


class InputSchema(BaseModel):
    size: int = Field(default=16, description="Side length of the square grid.")
    steps: int = Field(default=20, description="Number of timesteps to integrate.")
    diffusivity: float = Field(default=0.1, description="Diffusion coefficient.")


class OutputSchema(BaseModel):
    # Ordinary Array fields -- they happen to be fed on-disk references. A client
    # requesting json+binref gets the bytes straight from disk, with no server
    # round-trip through memory.
    trajectory: list[Array[(None, None), Float64]] = Field(
        description="The temperature field at every timestep."
    )
    final: Array[(None, None), Float64] = Field(
        description="The temperature field after the last step."
    )


def _step(field: np.ndarray, diffusivity: float) -> np.ndarray:
    """One explicit forward-Euler diffusion step with a 5-point Laplacian."""
    laplacian = (
        np.roll(field, 1, 0)
        + np.roll(field, -1, 0)
        + np.roll(field, 1, 1)
        + np.roll(field, -1, 1)
        - 4.0 * field
    )
    return field + diffusivity * laplacian


def apply(inputs: InputSchema) -> OutputSchema:
    # A hot square in the middle of a cold grid.
    field = np.zeros((inputs.size, inputs.size), dtype=np.float64)
    lo, hi = inputs.size // 4, 3 * inputs.size // 4
    field[lo:hi, lo:hi] = 1.0

    # Checkpoint every timestep to disk as the solve proceeds. BinrefWriter packs
    # all the snapshots into a few shared buffers instead of one file each, and
    # never keeps more than the current field in memory.
    with BinrefWriter() as checkpoints:
        trajectory = [checkpoints.write(field)]
        for _ in range(inputs.steps):
            field = _step(field, inputs.diffusivity)
            trajectory.append(checkpoints.write(field))

    # The final field gets its own buffer.
    final = BinrefArray.write(field)

    # Every array in the output already lives on disk; returning them copies no
    # array data back into memory.
    return OutputSchema(trajectory=trajectory, final=final)
