# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pydantic import BaseModel, Field, model_validator

from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType

#
# Schemas
#


class VolumetricMeshData(BaseModel):
    """Mock mesh schema; shapes not validated."""

    n_points: int
    n_cells: int

    points: Differentiable[Array[(None, 3), Float32]]
    num_points_per_cell: Array[(None,), Float32]  # should have length == n_cells
    cell_connectivity: Array[(None,), Int32]  # length == sum(num_points_per_cell)

    cell_data: dict[str, Array[(None, None), Float32]]
    point_data: dict[str, Array[(None, None), Float32]]

    @model_validator(mode="after")
    def validate_num_points_per_cell(self):
        if not isinstance(self.num_points_per_cell, np.ndarray):
            return self
        if len(self.num_points_per_cell) != self.n_cells:
            raise ValueError(f"Length of num_points_per_cell must be {self.n_cells}")
        return self

    @model_validator(mode="after")
    def validate_cell_connectivity(self):
        if not isinstance(self.cell_connectivity, np.ndarray):
            return self
        expected_len = sum(self.num_points_per_cell)
        if len(self.cell_connectivity) != expected_len:
            raise ValueError(f"Length of cell_connectivity must be {expected_len}")
        return self


class InputSchema(BaseModel):
    mesh: VolumetricMeshData = Field(
        description="The mesh you want summary statistics of"
    )


class SummaryStatistics(BaseModel):
    first_point_coordinates: Differentiable[Array[(3,), Float32]] = Field(
        description="Coordinates of the first point defined in the mesh."
    )
    barycenter: Differentiable[Array[(3,), Float32]] = Field(
        "Mean of all the points defined in the input mesh."
    )


class OutputSchema(BaseModel):
    statistics: SummaryStatistics = Field(description="Summary statistics of the mesh.")


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    points = inputs.mesh.points

    statistics = SummaryStatistics(
        first_point_coordinates=points[0],
        barycenter=points.mean(axis=0),
    )

    return OutputSchema(statistics=statistics)


#
# Optional endpoints
#


def abstract_eval(abstract_inputs):
    input_points_shapedtype = abstract_inputs.mesh.points

    # get dimension of vector space points live in from input
    dim = input_points_shapedtype.shape[1]
    dtype = input_points_shapedtype.dtype
    return {
        "statistics": {
            "first_point_coordinates": ShapeDType(shape=(dim,), dtype=dtype),
            "barycenter": ShapeDType(shape=(dim,), dtype="float32"),
        }
    }


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    # TODO: Make this work with other inputs as well
    assert tuple(jac_inputs) == ("mesh.points",)

    jac = {}

    if "statistics.barycenter" in jac_outputs:
        jac_dbar_dpoints = np.zeros((3, inputs.mesh.n_points, 3))
        jac_dbar_dpoints[0, :, 0] = 1.0 / inputs.mesh.n_points
        jac_dbar_dpoints[1, :, 1] = 1.0 / inputs.mesh.n_points
        jac_dbar_dpoints[2, :, 2] = 1.0 / inputs.mesh.n_points
        jac["statistics.barycenter"] = {"mesh.points": jac_dbar_dpoints}

    if "statistics.first_point_coordinates" in jac_outputs:
        jac_dfpc_dpoints = np.zeros((3, inputs.mesh.n_points, 3))
        jac_dfpc_dpoints[0, 0, 0] = 1.0
        jac_dfpc_dpoints[1, 0, 1] = 1.0
        jac_dfpc_dpoints[2, 0, 2] = 1.0

        jac["statistics.first_point_coordinates"] = {"mesh.points": jac_dfpc_dpoints}

    return jac
