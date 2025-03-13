# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class Vector_and_Scalar(BaseModel):
    v: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector"
    )
    s: Differentiable[Float32] = Field(description="A scalar", default=1.0)

    def scale(self) -> Differentiable[Array[(None,), Float32]]:
        return self.s * self.v


class InputSchema(BaseModel):
    a: Vector_and_Scalar = Field(
        description="An arbitrary vector and a scalar to multiply it by"
    )
    b: Vector_and_Scalar = Field(
        description="An arbitrary vector and a scalar to multiply it by "
        "must be of same shape as b"
    )
    norm_ord: int = Field(
        description="Order of norm (see numpy.linalg.norm)",
        default=2,
    )

    @model_validator(mode="after")
    def validate_shape_inputs(self) -> Self:
        if self.a.v.shape != self.b.v.shape:
            raise ValueError(
                f"a.v and b.v must have the same shape. "
                f"Got {self.a.v.shape} and {self.b.v.shape} instead."
            )
        return self


class Result_and_Norm(BaseModel):
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="Vector s_a·a + s_b·b"
    )
    normed_result: Differentiable[Array[(None,), Float32]] = Field(
        description="Normalized Vector s_a·a + s_b·b/|s_a·a + s_b·b|"
    )


class OutputSchema(BaseModel):
    vector_add: Result_and_Norm
    vector_min: Result_and_Norm


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    a_scaled = inputs["a"]["s"] * inputs["a"]["v"]
    b_scaled = inputs["b"]["s"] * inputs["b"]["v"]
    add_result = a_scaled + b_scaled
    min_result = a_scaled - b_scaled
    return {
        "vector_add": {
            "result": add_result,
            "normed_result": add_result
            / jnp.linalg.norm(add_result, ord=inputs["norm_ord"]),
        },
        "vector_min": {
            "result": min_result,
            "normed_result": min_result
            / jnp.linalg.norm(min_result, ord=inputs["norm_ord"]),
        },
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Multiplies a vector `a` by `s`, and sums the result to `b`."""
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtye_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtye_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtye_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtye_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtye_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: {"shape": x.shape, "dtype": str(x.dtype)}
        if is_shapedtye_struct(x)
        else x,
        jax_shapes,
        is_leaf=is_shapedtye_struct,
    )


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


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


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
