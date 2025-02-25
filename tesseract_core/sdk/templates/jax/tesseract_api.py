# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for {{name}}
# Generated by tesseract {{version}} on {{timestamp}}

from functools import partial
from typing import Any

import jax.tree
from jax import ShapeDtypeStruct, eval_shape, jacrev, jit, jvp, vjp
from pydantic import BaseModel

from tesseract_core.runtime import Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

#
# Schemata
#


class InputSchema(BaseModel):
    example: Differentiable[Float32]


class OutputSchema(BaseModel):
    example: Differentiable[Float32]


#
# Required endpoints
#


# TODO: Add or import your function here, must be JAX-jittable and
# take/return a single pytree as an input/output conforming respectively
# to Input/OutputSchema
@jit
def apply_jit(inputs: dict) -> dict:
    return inputs


def apply(inputs: InputSchema) -> OutputSchema:
    # Optional: Insert any pre-processing/setup that doesn't require tracing
    # and is only required when specifically running your apply function
    # and not your differentiable endpoints.
    # For example, you might want to set up a logger or mlflow server.
    # Pre-processing should not modify any input that could impact the
    # differentiable outputs in a nonlinear way (a constant shift
    # should be safe)

    out = apply_jit(inputs.model_dump())

    # Optional: Insert any post-processing that doesn't require tracing
    # For example, you might want to save to disk or modify a non-differentiable
    # output. Again, do not modify any differentiable output in a non-linear way.
    return out


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
