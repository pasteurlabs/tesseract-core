# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torch.utils._pytree import tree_map
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

#
# Schemata
#


class Vector_and_Scalar(BaseModel):
    v: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector"
    )
    s: Differentiable[Float32] = Field(description="A scalar", default=1.0)

    # we lose the ability to use methods such as this when using model_dump
    # unless we reconstruct nested models
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


def evaluate(inputs: Any) -> Any:
    a_scaled = inputs["a"]["s"] * inputs["a"]["v"]
    b_scaled = inputs["b"]["s"] * inputs["b"]["v"]
    add_result = a_scaled + b_scaled
    min_result = a_scaled - b_scaled
    return {
        "vector_add": {
            "result": add_result,
            "normed_result": add_result / torch.linalg.norm(add_result, ord=2),
        },
        "vector_min": {
            "result": min_result,
            "normed_result": min_result / torch.linalg.norm(min_result, ord=2),
        },
    }


def apply(inputs: InputSchema) -> OutputSchema:
    # Optional: Insert any pre-processing/setup that doesn't require tracing
    # and is only required when specifically running your apply function
    # and not your differentiable endpoints.
    # For example, you might want to set up a logger or mlflow server.
    # Pre-processing should not modify any input that could impact the
    # differentiable outputs in a nonlinear way (a constant shift
    # should be safe)

    # Convert to pytorch tensors
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    out = evaluate(tensor_inputs)

    # Optional: Insert any post-processing that doesn't require tracing
    # For example, you might want to save to disk or modify a non-differentiable
    # output. Again, do not modify any differentiable output in a non-linear way.
    return out


#
# Pytorch-handled AD endpoints (no need to modify)
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    # convert all numbers and arrays to torch tensors
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())

    # flatten the dict such that they can be accessed by paths and obtain the values
    pos_inputs = flatten_with_paths(tensor_inputs, jac_inputs).values()

    # create a positional function that accepts a flat sequence of values and returns a tuple
    filtered_pos_eval = filter_func(
        evaluate,
        tensor_inputs,
        jac_outputs,
        input_paths=jac_inputs,
    )

    def filtered_pos_eval_flat(*args):
        res = filtered_pos_eval(*args)
        return tuple(res[k] for k in jac_outputs)

    # calculate the jacobian
    jacobian = torch.autograd.functional.jacobian(
        filtered_pos_eval_flat, tuple(pos_inputs)
    )

    # rebuild the dictionary from the list of results
    jac_dict = {}
    for dy, dys in zip(jac_outputs, jacobian, strict=True):
        jac_dict[dy] = {}
        for dx, dxs in zip(jac_inputs, dys, strict=True):
            jac_dict[dy][dx] = dxs

    return jac_dict


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    # Make ordering of tangent_vector identical to jvp_inputs
    tangent_vector = {key: tangent_vector[key] for key in jvp_inputs}

    # convert all numbers and arrays to torch tensors
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    pos_tangent = tree_map(to_tensor, tangent_vector).values()

    # flatten the dictionaries such that they can be accessed by paths
    pos_inputs = flatten_with_paths(tensor_inputs, jvp_inputs).values()

    # create a positional function that accepts a list of values
    filtered_pos_eval = filter_func(
        evaluate, tensor_inputs, jvp_outputs, input_paths=jvp_inputs
    )

    return torch.func.jvp(filtered_pos_eval, tuple(pos_inputs), tuple(pos_tangent))[1]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    # Make ordering of cotangent_vector identical to vjp_inputs
    cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}

    # convert all numbers and arrays to torch tensors
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    tensor_cotangent = tree_map(to_tensor, cotangent_vector)

    # flatten the dictionaries such that they can be accessed by paths
    pos_inputs = flatten_with_paths(tensor_inputs, vjp_inputs).values()

    # create a positional function that accepts a list of values
    filtered_pos_func = filter_func(
        evaluate, tensor_inputs, vjp_outputs, input_paths=vjp_inputs
    )

    _, vjp_func = torch.func.vjp(filtered_pos_func, *pos_inputs)

    vjp_vals = vjp_func(tensor_cotangent)
    return dict(zip(vjp_inputs, vjp_vals, strict=True))


to_tensor = lambda x: torch.tensor(x) if isinstance(x, np.generic | np.ndarray) else x
