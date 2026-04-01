# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lorenz 96 Tesseract with differentiable forcing parameter F.

This is a variant of the Lorenz 96 Tesseract from the 4D-Var demo, modified
to mark the forcing parameter F as Differentiable. This allows JAX autodiff
(and thus NumPyro's NUTS) to compute gradients w.r.t. F for Bayesian
parameter inference.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(BaseModel):
    """Input schema for forecasting of Lorenz 96 system."""

    state: Differentiable[Array[(None,), Float32]] = Field(
        description="A state vector for the Lorenz 96 system"
    )
    F: Differentiable[Array[(), Float32]] = Field(
        description="Forcing parameter for Lorenz 96", default=8.0
    )
    dt: float = Field(description="Time step for integration", default=0.05)
    n_steps: int = Field(description="Number of integration steps", default=1)


class OutputSchema(BaseModel):
    """Output schema for forecasting of Lorenz 96 system."""

    result: Differentiable[Array[(None, None), Float32]] = Field(
        description="A trajectory of predictions after integration"
    )


def lorenz96_step(state: jnp.ndarray, F: float, dt: float) -> jnp.ndarray:
    """Perform one step of RK4 integration for the Lorenz 96 system."""

    def lorenz96_derivatives(x: jnp.ndarray) -> jnp.ndarray:
        N = x.shape[0]
        ip1 = (jnp.arange(N) + 1) % N
        im1 = (jnp.arange(N) - 1) % N
        im2 = (jnp.arange(N) - 2) % N
        d = (x[ip1] - x[im2]) * x[im1] - x + F
        return d

    k1 = lorenz96_derivatives(state)
    k2 = lorenz96_derivatives(state + dt * k1 / 2)
    k3 = lorenz96_derivatives(state + dt * k2 / 2)
    k4 = lorenz96_derivatives(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def lorenz96_multi_step(
    state: jnp.ndarray, F: float, dt: float, n_steps: int
) -> jnp.ndarray:
    """Perform multiple steps of Lorenz 96 integration using scan."""

    def step_fn(state: jnp.ndarray, _: Any) -> tuple:
        return lorenz96_step(state, F, dt), state

    _, trajectory = jax.lax.scan(step_fn, state, None, length=n_steps)
    return trajectory


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    trajectory = lorenz96_multi_step(**inputs)
    return dict(result=trajectory)


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs: Any) -> Any:
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
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
) -> dict:
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
