# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import (
    LRUCache,
    filter_func,
    flatten_with_paths,
    hash_pytree_leaves,
    set_at_path,
)

#
# Cache configuration
#
# When apply() is called, this template runs jax.vjp internally and caches the
# resulting backward function. A subsequent vector_jacobian_product call on the
# same inputs reuses that cached backward, skipping the redundant forward pass
# that vjp would otherwise repeat.
#
# Under typical tesseract-jax usage (gradient-based code calling the Tesseract
# through tesseract-jax's custom_vjp), apply() runs first on every gradient
# evaluation — even under plain jax.grad — so the cache reliably hits on the
# subsequent vjp call. This makes the recipe close to a free win for any
# gradient-based workflow: Adam/SGD, L-BFGS / line search, value_and_grad-style
# code, etc. Typical speedup is ~10-20% on moderate-to-deep networks.
#
# The cache also helps when a single apply is followed by multiple vjp calls
# on the same input — for example, jax.jacrev (which decomposes into one vjp
# per output basis vector) or CG-style inverse-problem solvers (each inner CG
# step computes J^T u at the same iterate). Each subsequent vjp on the same
# input reuses the cached residuals.
#
# Poor fit:
#   - Very small models where the forward pass is microseconds: the cache
#     machinery overhead exceeds the saved work.
#
# Set _CACHE_SIZE = 0 to disable caching entirely; apply() and vjp() then
# bypass the cache machinery and run their forward passes directly.
# Increase _CACHE_SIZE for workloads that interleave several apply() calls
# before their corresponding vjp() calls.
#
_CACHE_SIZE = 1

#
# Schemata
#


class Vector_and_Scalar(BaseModel):
    v: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector"
    )
    s: Differentiable[Float32] = Field(description="A scalar", default=1.0)

    def scale(self) -> np.ndarray:
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


#
# Required endpoints
#


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
    inputs_dict = inputs.model_dump()

    if _CACHE_SIZE <= 0:
        return apply_jit(inputs_dict)

    def _apply_for_vjp(inputs_dict):
        out = apply_jit(inputs_dict)
        diff_out, static_out = eqx.partition(out, eqx.is_array)
        return diff_out, static_out

    diff_primals, vjp_func, static_primals = jax.vjp(
        _apply_for_vjp, inputs_dict, has_aux=True
    )
    out = eqx.combine(diff_primals, static_primals)

    cotangent_template = jax.tree.map(jnp.zeros_like, diff_primals)
    _vjp_cache.put(_hash_inputs(inputs_dict), (vjp_func, cotangent_template))

    return out


#
# Jax-handled gradient endpoints (no need to modify)
#
_vjp_cache = LRUCache(maxsize=_CACHE_SIZE)


def _hash_inputs(inputs_dict: dict) -> bytes:
    """Compute a SHA-256 hash of a pytree of inputs for cache key comparison."""
    leaves, treedef = jax.tree.flatten(inputs_dict)
    return hash_pytree_leaves(leaves, treedef)


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
    inputs_dict = inputs.model_dump()

    if (
        _CACHE_SIZE > 0
        and (cached := _vjp_cache.get(_hash_inputs(inputs_dict))) is not None
    ):
        vjp_func, cotangent_template = cached
        full_cotangent = jax.tree.map(jnp.zeros_like, cotangent_template)
        full_cotangent = set_at_path(full_cotangent, cotangent_vector)
        (all_input_cotangents,) = vjp_func(full_cotangent)
        return flatten_with_paths(all_input_cotangents, include_paths=vjp_inputs)

    return vjp_jit(inputs_dict, tuple(vjp_inputs), tuple(vjp_outputs), cotangent_vector)


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
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtye_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtye_struct,
    )


#
# Helper functions
#


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
