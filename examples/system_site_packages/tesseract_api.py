# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solve the Poisson equation on a unit square using Firedrake.

Firedrake is NOT listed in tesseract_requirements.txt — it's only available
because system_site_packages is enabled and Firedrake is pre-installed in the
base image.
"""

from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    mesh_resolution: int = Field(
        description="Number of cells along each edge of the unit square mesh.",
        ge=1,
    )


class OutputSchema(BaseModel):
    num_dofs: int = Field(description="Number of degrees of freedom in the solution.")
    l2_error: float = Field(
        description="L2 norm of the error against the exact solution."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Solve -laplacian(u) = f on the unit square with known exact solution."""
    from firedrake import (
        DirichletBC,
        Function,
        FunctionSpace,
        SpatialCoordinate,
        TestFunction,
        TrialFunction,
        UnitSquareMesh,
        dx,
        errornorm,
        grad,
        inner,
        pi,
        sin,
        solve,
    )

    mesh = UnitSquareMesh(inputs.mesh_resolution, inputs.mesh_resolution)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    exact = sin(pi * x) * sin(pi * y)
    f = 2 * pi**2 * sin(pi * x) * sin(pi * y)

    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    bc = DirichletBC(V, 0, "on_boundary")

    u_h = Function(V)
    solve(a == L, u_h, bcs=bc)

    l2_err = errornorm(exact, u_h, norm_type="L2")

    return OutputSchema(num_dofs=V.dof_count, l2_error=l2_err)
