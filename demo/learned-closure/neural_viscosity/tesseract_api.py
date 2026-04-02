# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neural viscosity closure Tesseract.

A small MLP that predicts spatially-varying viscosity from local flow features.
Used as a learned closure inside a PDE solver — the solver calls this Tesseract
at every timestep to get the viscosity field, and gradients flow back through
both during training.

The network weights are passed as explicit inputs (not internal state) so that
an external optimizer can differentiate through the full solver-closure pipeline.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Network architecture constants
HIDDEN_DIM = 32
N_HIDDEN_LAYERS = 2


class InputSchema(BaseModel):
    u: Differentiable[Array[(None,), Float64]] = Field(
        description="Velocity field at grid points"
    )
    dudx: Differentiable[Array[(None,), Float64]] = Field(
        description="Velocity gradient du/dx at grid points"
    )
    x: Array[(None,), Float64] = Field(description="Spatial coordinates of grid points")
    # Network weights as flat arrays for easy composition
    w1: Differentiable[Array[(3, HIDDEN_DIM), Float64]] = Field(
        description="First layer weights (3 input features -> hidden)"
    )
    b1: Differentiable[Array[(HIDDEN_DIM,), Float64]] = Field(
        description="First layer bias"
    )
    w2: Differentiable[Array[(HIDDEN_DIM, HIDDEN_DIM), Float64]] = Field(
        description="Second layer weights"
    )
    b2: Differentiable[Array[(HIDDEN_DIM,), Float64]] = Field(
        description="Second layer bias"
    )
    w3: Differentiable[Array[(HIDDEN_DIM, 1), Float64]] = Field(
        description="Output layer weights (hidden -> 1)"
    )
    b3: Differentiable[Array[(1,), Float64]] = Field(description="Output layer bias")


class OutputSchema(BaseModel):
    nu: Differentiable[Array[(None,), Float64]] = Field(
        description="Predicted viscosity at each grid point (always positive)"
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    u = inputs["u"]
    dudx = inputs["dudx"]
    x = inputs["x"]

    # Stack features: [u, dudx, x] at each grid point -> (N, 3)
    features = jnp.stack([u, dudx, x], axis=-1)

    # Forward pass through MLP
    h = features @ inputs["w1"] + inputs["b1"]
    h = jnp.tanh(h)
    h = h @ inputs["w2"] + inputs["b2"]
    h = jnp.tanh(h)
    out = h @ inputs["w3"] + inputs["b3"]

    # Sigmoid * scale to keep viscosity in a physically reasonable range.
    # Range [0, nu_max] prevents CFL violations in the explicit solver.
    nu_max = 0.05
    nu = nu_max * jax.nn.sigmoid(out[:, 0])

    return {"nu": nu}


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
