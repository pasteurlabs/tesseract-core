# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Any

import jax.numpy as jnp
import jax.tree
from jax import ShapeDtypeStruct, eval_shape, jacrev, jit, jvp, vjp
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


def gaussian_rbf(x: float, c: float, length_scale: float) -> float:
    return jnp.exp(-((x - c) ** 2) / (2 * length_scale**2))


def mse_error(
    x_centers: Array[(None,), Float32],
    weights: Array[(None,), Float32],
    length_scale: float,
    x_target: Array[(None,), Float32],
    y_target: Array[(None,), Float32],
) -> float:
    # Initialize the RBF expansion result to zero
    y_hat = 0
    # Compute the RBF expansion
    for coeff, center in zip(weights, x_centers, strict=False):
        y_hat += coeff * gaussian_rbf(x_target, center, length_scale)

    return {"mse": jnp.mean((y_target - y_hat) ** 2)}


@jit
def apply_jit(inputs: dict) -> dict:
    ordered_keys = ["x_centers", "weights", "length_scale", "x_target", "y_target"]
    return mse_error(*(inputs[key] for key in ordered_keys))


#
# Schemas
#


class InputSchema(BaseModel):
    x_centers: Array[(None,), Float32] = Field(description="Centers of the RBFs")
    weights: Differentiable[Array[(None,), Float32]] = Field(
        description="Coefficients for the RBFs expansion"
    )
    length_scale: Differentiable[Float32] = Field(
        description="Length scale for the RBFs kernel.", default=0.1
    )
    x_target: Differentiable[Array[(None,), Float32]] = Field(
        description="x-coordinates of the target points"
    )
    y_target: Differentiable[Array[(None,), Float32]] = Field(
        description="y-coordinates of the target points"
    )

    @model_validator(mode="after")
    def validate_shape_targets(self) -> Self:
        if self.x_target.shape != self.y_target.shape:
            raise ValueError(
                f"x_target and y_target must have the same shape. "
                f"Got {self.x_target.shape} and {self.y_target.shape} instead."
            )
        return self

    @model_validator(mode="after")
    def validate_shape_weights(self) -> Self:
        if self.x_centers.shape != self.weights.shape:
            raise ValueError(
                f"x_centers and weights must have the same shape. "
                f"Got {self.x_centers.shape} and {self.weights.shape} instead."
            )
        return self


class OutputSchema(BaseModel):
    mse: Differentiable[Float32] = Field(
        description="Mean squared error of the RBFs interpolation"
    )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(inputs.model_dump())


#
# Jax-handled AD endpoints (no need to modify)
#


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
    jaxified_inputs = jax.tree.map(
        lambda x: ShapeDtypeStruct(**x),
        abstract_inputs.model_dump(),
        is_leaf=lambda x: (x.keys() == {"shape", "dtype"}),
    )
    jax_shapes = eval_shape(apply_jit, jaxified_inputs)
    return jax.tree.map(
        lambda sd: {"shape": sd.shape, "dtype": str(sd.dtype)}, jax_shapes
    )


#
# Helper functions
#


@partial(jit, static_argnames=["jac_inputs", "jac_outputs"])
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jacrev(filtered_apply)(flatten_with_paths(inputs, include_paths=jac_inputs))


@partial(jit, static_argnames=["jvp_inputs", "jvp_outputs"])
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@partial(jit, static_argnames=["vjp_inputs", "vjp_outputs"])
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]
