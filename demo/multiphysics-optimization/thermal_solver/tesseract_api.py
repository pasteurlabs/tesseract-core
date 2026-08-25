# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thermal solver Tesseract: 2D steady-state heat equation on a rectangular plate.

Solves -k * laplacian(T) = q(x, y) with Dirichlet boundary conditions using
a finite-difference discretization on a regular grid. The heat source is a
Gaussian blob with parameterized location and intensity.

When displacement is provided (from a structural solver), the mesh is deformed
accordingly, introducing geometry-dependent coupling for two-way
thermoelastic problems.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Grid resolution (interior nodes). Kept moderate for demo speed.
NX = 30
NY = 30
DX = 1.0 / (NX + 1)  # grid spacing on the unit square
DY = 1.0 / (NY + 1)


class InputSchema(BaseModel):
    source_x: Differentiable[Float32] = Field(
        description="Heat source x-location (0-1, fraction of plate width)"
    )
    source_y: Differentiable[Float32] = Field(
        description="Heat source y-location (0-1, fraction of plate height)"
    )
    source_intensity: Differentiable[Float32] = Field(
        description="Heat source intensity", default=10.0
    )
    source_width: Float32 = Field(description="Heat source Gaussian width", default=0.1)
    displacement: Differentiable[Array[(NX, NY, 2), Float32]] = Field(
        description="Displacement field from structural solver (NX x NY x 2). "
        "Zero for the first coupling iteration.",
        default=None,
    )
    conductivity: Float32 = Field(description="Thermal conductivity", default=1.0)
    boundary_temp: Float32 = Field(
        description="Dirichlet boundary temperature", default=0.0
    )


class OutputSchema(BaseModel):
    temperature: Differentiable[Array[(NX, NY), Float32]] = Field(
        description="Steady-state temperature field on interior nodes"
    )


def _make_grid(nx: int, ny: int):
    """Create a unit-square grid of interior node positions."""
    x = jnp.linspace(0, 1, nx + 2)[1:-1]
    y = jnp.linspace(0, 1, ny + 2)[1:-1]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def _gaussian_source(X, Y, cx, cy, intensity, width):
    """Gaussian heat source centered at (cx, cy)."""
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    return intensity * jnp.exp(-r2 / (2 * width**2))


def _neg_laplacian(u):
    """-(u_xx + u_yy) on the interior grid with zero (homogeneous) boundaries."""
    uxx = jnp.diff(u, n=2, axis=0, prepend=0.0, append=0.0)
    uyy = jnp.diff(u, n=2, axis=1, prepend=0.0, append=0.0)
    return -(uxx / DX**2 + uyy / DY**2)


# Once conductivity and boundary values are moved to the RHS, the operator is just
# the unit Laplacian -- it depends only on the (fixed) grid shape, so it is a
# compile-time constant. We build it and Cholesky-factor it ONCE at import, then
# reuse the factorization on every solve via lineax's `state=` (factor once, solve
# many). Each solve is then only the O(N^2) back-substitution.
NEG_LAPLACIAN_OP = lx.FunctionLinearOperator(
    _neg_laplacian,
    jax.ShapeDtypeStruct((NX, NY), jnp.float32),
    tags=(lx.positive_semidefinite_tag,),
)
SOLVER = lx.Cholesky()
NEG_LAPLACIAN_FACTOR = SOLVER.init(NEG_LAPLACIAN_OP, options={})


def _solve_poisson(rhs, boundary_value=0.0):
    """Solve  -(u_xx + u_yy) = rhs with Dirichlet bc's reusing global factorization."""
    # A Dirichlet value on an edge adds value/h² to the RHS at the adjacent interior
    # nodes (a corner picks up a contribution from both of its edges).
    b = rhs
    b = b.at[0, :].add(boundary_value / DX**2)
    b = b.at[-1, :].add(boundary_value / DX**2)
    b = b.at[:, 0].add(boundary_value / DY**2)
    b = b.at[:, -1].add(boundary_value / DY**2)
    return lx.linear_solve(
        NEG_LAPLACIAN_OP, b, SOLVER, state=NEG_LAPLACIAN_FACTOR
    ).value


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    X, Y = _make_grid(NX, NY)

    # Deform grid if displacement is provided
    displacement = inputs.get("displacement")
    if displacement is not None:
        X = X + displacement[:, :, 0]
        Y = Y + displacement[:, :, 1]

    source = _gaussian_source(
        X,
        Y,
        inputs["source_x"],
        inputs["source_y"],
        inputs["source_intensity"],
        inputs["source_width"],
    )

    # Steady-state heat equation -k*laplacian(T) = q. Dividing by the (constant)
    # conductivity leaves the unit Laplacian, whose factorization is reused above.
    k = inputs["conductivity"]
    temperature = _solve_poisson(source / k, inputs["boundary_temp"])

    return {"temperature": temperature.astype(jnp.float32)}


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs: Any) -> Any:
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs: Any) -> Any:
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict,
    jvp_inputs: tuple[str],
    jvp_outputs: tuple[str],
    tangent_vector: dict,
) -> Any:
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> Any:
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
) -> Any:
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> Any:
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
) -> Any:
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> Any:
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))
