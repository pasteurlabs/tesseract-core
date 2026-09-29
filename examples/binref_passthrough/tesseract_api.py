# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return a simulation trajectory from disk without reading it back into memory.

A tiny explicit heat-diffusion solver writes the 2D field to a binref buffer at
every timestep and returns ``BinrefArray`` references to those buffers. With
``json+binref`` output, the runtime forwards the on-disk bytes to the client
as-is. The output fields are ordinary ``Array`` types.
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

    # Write each snapshot to disk as the solve proceeds. BinrefWriter packs them
    # into a few shared buffers instead of one file each.
    checkpoints = BinrefWriter()
    trajectory = [checkpoints.write(field)]
    for _ in range(inputs.steps):
        field = _step(field, inputs.diffusivity)
        trajectory.append(checkpoints.write(field))

    final = BinrefArray.write(field)
    return OutputSchema(trajectory=trajectory, final=final)
