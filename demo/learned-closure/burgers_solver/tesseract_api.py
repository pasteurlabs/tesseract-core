# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-timestep Burgers' equation solver Tesseract.

Solves one explicit Euler step of the 1D viscous Burgers' equation:

    u^{n+1} = u^n + dt * (-u * du/dx + nu * d²u/dx²)

The viscosity field nu is provided as an input — the solver does not compute it.
This clean interface (state + material field → next state) is the same contract
that a Fortran solver with an adjoint could implement. The outer time-stepping
loop and closure evaluation live in the caller, enabling per-timestep closure
calls and end-to-end gradient flow through both solver and closure.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Default grid size
N = 128

# --- Grid setup (fixed for this Tesseract) ---
DX = 1.0 / (N - 1)


class InputSchema(BaseModel):
    u: Differentiable[Array[(N,), Float64]] = Field(
        description="Current velocity field on the grid"
    )
    nu: Differentiable[Array[(N,), Float64]] = Field(
        description="Viscosity field at each grid point (must be positive)"
    )
    dt: float = Field(description="Time step size", default=1e-4)


class OutputSchema(BaseModel):
    u_next: Differentiable[Array[(N,), Float64]] = Field(
        description="Velocity field after one time step"
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    u = inputs["u"]
    nu = inputs["nu"]
    dt = inputs["dt"]

    # Spatial derivatives via central differences
    dudx = jnp.zeros_like(u)
    dudx = dudx.at[1:-1].set((u[2:] - u[:-2]) / (2 * DX))

    d2udx2 = jnp.zeros_like(u)
    d2udx2 = d2udx2.at[1:-1].set((u[2:] - 2 * u[1:-1] + u[:-2]) / (DX**2))

    # Burgers' equation: du/dt = -u * du/dx + nu * d²u/dx²
    dudt = -u * dudx + nu * d2udx2

    # Forward Euler step
    u_next = u + dt * dudt

    # Enforce boundary conditions (Dirichlet: hold boundary values)
    u_next = u_next.at[0].set(u[0])
    u_next = u_next.at[-1].set(u[-1])

    return {"u_next": u_next}


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs: Any) -> Any:
    return {"u_next": {"shape": [N], "dtype": "float64"}}


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
