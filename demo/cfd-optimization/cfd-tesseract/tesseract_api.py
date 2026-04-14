# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(BaseModel):
    v0: Differentiable[
        Array[
            (
                None,
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="3D Array defining the initial velocity field")
    density: float = Field(description="Density of the fluid")
    viscosity: float = Field(description="Viscosity of the fluid")
    inner_steps: int = Field(
        description="Number of solver steps for each timestep", default=25
    )
    outer_steps: int = Field(description="Number of timesteps steps", default=10)
    max_velocity: float = Field(description="Maximum velocity", default=2.0)
    cfl_safety_factor: float = Field(description="CFL safety factor", default=0.5)
    domain_size_x: float = Field(description="Domain size x", default=1.0)
    domain_size_y: float = Field(description="Domain size y", default=1.0)


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None, None, None), Float32]] = Field(
        description="3D Array defining the final velocity field"
    )


def cfd_fwd(
    v0: jnp.ndarray,
    density: float,
    viscosity: float,
    inner_steps: int,
    outer_steps: int,
    max_velocity: float,
    cfl_safety_factor: float,
    domain_size_x: float,
    domain_size_y: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute the final velocity field using the semi-implicit Navier-Stokes equations.

    Args:
        v0: Initial velocity field.
        density: Density of the fluid.
        viscosity: Viscosity of the fluid.
        inner_steps: Number of solver steps for each timestep.
        outer_steps: Number of timesteps steps.
        max_velocity: Maximum velocity.
        cfl_safety_factor: CFL safety factor.
        domain_size_x: Domain size in x direction.
        domain_size_y: Domain size in y direction.

    Returns:
        Final velocity field.
    """
    vx0 = v0[..., 0]
    vy0 = v0[..., 1]
    bc = cfd.boundaries.HomogeneousBoundaryConditions(
        (
            (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
            (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
        )
    )

    # reconstruct grid from input
    grid = cfd.grids.Grid(
        vx0.shape, domain=((0.0, domain_size_x), (0.0, domain_size_y))
    )

    vx0 = cfd.grids.GridArray(vx0, grid=grid, offset=(1.0, 0.5))
    vy0 = cfd.grids.GridArray(vy0, grid=grid, offset=(0.5, 1.0))

    # reconstruct GridVariable from input
    vx0 = cfd.grids.GridVariable(vx0, bc)
    vy0 = cfd.grids.GridVariable(vy0, bc)
    v0 = (vx0, vy0)

    # Choose a time step.
    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety_factor, viscosity, grid
    )

    # Define a step function and use it to compute a trajectory.
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid
        ),
        steps=inner_steps,
    )
    rollout_fn = cfd.funcutils.trajectory(step_fn, outer_steps)
    _, trajectory = jax.device_get(rollout_fn(v0))
    vxn = trajectory[0].array.data[-1]
    vyn = trajectory[1].array.data[-1]
    return jnp.stack([vxn, vyn], axis=-1)


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    vn = cfd_fwd(**inputs)
    return dict(result=vn)


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(inputs.model_dump())


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
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

    def wrapped_apply(dynamic_inputs):
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
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]
