# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pyvista as pv
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float32


class MeshInfo(BaseModel):
    dimensions: list[int]
    n_points: int
    n_cells: int
    bounds: list[float]
    center: list[float]


class InputSchema(BaseModel):
    vertices: Array[(None, 3), Float32] = Field(
        description="3D vector of the mesh vertices."
    )


class OutputSchema(BaseModel):
    pv_mesh: MeshInfo = Field(description="PyVista mesh object info.")


def apply(inputs: InputSchema) -> OutputSchema:
    """Convert input vertices into a PyVista mesh."""
    grid = pv.StructuredGrid(*inputs.vertices)
    info = MeshInfo(
        dimensions=grid.dimensions,
        n_points=grid.n_points,
        n_cells=grid.n_cells,
        bounds=grid.bounds,
        center=grid.center,
    )
    return OutputSchema(pv_mesh=info)
