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
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Grid resolution (interior nodes). Kept moderate for demo speed.
NX = 30
NY = 30


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


def _solve_heat_jacobi(source, conductivity, boundary_temp, n_iters=500):
    """Solve 2D Poisson equation via damped Jacobi iteration.

    This is intentionally simple — not the fastest solver, but fully
    differentiable through JAX and easy to understand.
    """
    nx, ny = source.shape
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    dx2 = dx**2
    dy2 = dy**2
    coeff = 1.0 / (2.0 * conductivity * (1.0 / dx2 + 1.0 / dy2))

    T = jnp.full((nx + 2, ny + 2), boundary_temp)

    def iteration(T, _):
        T_padded = T
        laplacian = (T_padded[2:, 1:-1] + T_padded[:-2, 1:-1]) / dx2 + (
            T_padded[1:-1, 2:] + T_padded[1:-1, :-2]
        ) / dy2
        T_interior = coeff * (conductivity * laplacian + source)
        T_new = T_padded.at[1:-1, 1:-1].set(T_interior)
        return T_new, None

    T_final, _ = jax.lax.scan(iteration, T, None, length=n_iters)
    return T_final[1:-1, 1:-1]


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

    temperature = _solve_heat_jacobi(
        source,
        inputs["conductivity"],
        inputs["boundary_temp"],
    )

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
