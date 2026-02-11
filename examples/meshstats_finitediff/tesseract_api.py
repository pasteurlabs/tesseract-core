# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example Tesseract demonstrating finite difference gradient computation on mesh data.

This example shows how to use finite_difference_jacobian, finite_difference_jvp,
and finite_difference_vjp to make a Tesseract with complex nested schemas
differentiable without writing explicit gradient code.

This is a variant of the meshstats example that uses finite differences instead
of hand-written Jacobian implementations.
"""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from tesseract_core.runtime import (
    Array,
    Differentiable,
    Float64,
    Int32,
    ShapeDType,
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)

#
# Schemas
#

FDAlgorithm = Literal["central", "forward", "stochastic"]


class VolumetricMeshData(BaseModel):
    """Mock mesh schema; shapes not validated."""

    n_points: int
    n_cells: int

    points: Differentiable[Array[(None, 3), Float64]]
    num_points_per_cell: Array[(None,), Float64]  # should have length == n_cells
    cell_connectivity: Array[(None,), Int32]  # length == sum(num_points_per_cell)

    cell_data: dict[str, Array[(None, None), Float64]]
    point_data: dict[str, Array[(None, None), Float64]]

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
    fd_algorithm: FDAlgorithm = Field(
        default="central",
        description=(
            "Finite difference algorithm to use for gradient computation. "
            "Options: 'central' (most accurate), 'forward' (faster), "
            "'stochastic' (SPSA, scales to high dimensions)."
        ),
    )


class SummaryStatistics(BaseModel):
    first_point_coordinates: Differentiable[Array[(3,), Float64]] = Field(
        description="Coordinates of the first point defined in the mesh."
    )
    barycenter: Differentiable[Array[(3,), Float64]] = Field(
        description="Mean of all the points defined in the input mesh."
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
    """Calculate output shape of apply from the shape of its inputs."""
    input_points_shapedtype = abstract_inputs.mesh.points

    # get dimension of vector space points live in from input
    dim = input_points_shapedtype.shape[1]
    dtype = input_points_shapedtype.dtype
    return {
        "statistics": {
            "first_point_coordinates": ShapeDType(shape=(dim,), dtype=dtype),
            "barycenter": ShapeDType(shape=(dim,), dtype=dtype),
        }
    }


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    """Compute the Jacobian using finite differences.

    This implementation uses the finite_difference_jacobian helper which automatically
    computes numerical derivatives by perturbing each input element.
    """
    return finite_difference_jacobian(
        apply,
        inputs,
        jac_inputs,
        jac_outputs,
        algorithm=inputs.fd_algorithm,
        eps=1e-6,
    )


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Compute the Jacobian-vector product using finite differences.

    The JVP computes J @ v efficiently using directional derivatives,
    without explicitly forming the full Jacobian matrix.
    """
    return finite_difference_jvp(
        apply,
        inputs,
        jvp_inputs,
        jvp_outputs,
        tangent_vector,
        algorithm=inputs.fd_algorithm,
        eps=1e-6,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Compute the vector-Jacobian product using finite differences.

    The VJP computes v @ J which is useful for backpropagation.
    """
    return finite_difference_vjp(
        apply,
        inputs,
        vjp_inputs,
        vjp_outputs,
        cotangent_vector,
        algorithm=inputs.fd_algorithm,
        eps=1e-6,
    )
