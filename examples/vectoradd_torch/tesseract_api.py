# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torch.utils._pytree import tree_map
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_pos_func, flatten_with_paths

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

    # Convert to pytorch tensors to enable torch.jit
    tensor_inputs = tree_map(convert_to_tensors, inputs.model_dump())
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
    tensor_inputs = tree_map(convert_to_tensors, inputs.model_dump())

    # flatten the dictionaries such that they can be accessed by paths
    path_inputs = flatten_with_paths(tensor_inputs, jac_inputs)

    # transform the dictionaries into a list of values for a positional function
    pos_inputs = path_inputs.values()
    keys = path_inputs.keys()

    # create a positional function that accepts a list of values and returns a set of tuples
    filtered_pos_eval = filter_pos_func(
        evaluate, tensor_inputs, jac_outputs, keys, output_to_tuple=True
    )

    # calculate the jacobian
    jacobian = torch.autograd.functional.jacobian(filtered_pos_eval, tuple(pos_inputs))

    # rebuild the dictionary from the list of results
    res_dict = {}
    for dy, dys in zip(jac_outputs, jacobian):
        res_dict[dy] = {}
        for dx, dxs in zip(jac_inputs, dys):
            res_dict[dy][dx] = dxs

    return res_dict


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent: dict[str, Any],
):
    # convert all numbers and arrays to torch tensors
    tensor_inputs = tree_map(convert_to_tensors, inputs.model_dump())
    tensor_tangent = tree_map(convert_to_tensors, tangent)

    # flatten the dictionaries such that they can be accessed by paths
    path_inputs = flatten_with_paths(tensor_inputs, jvp_inputs)

    # transform the dictionaries into a list of values for a positional function
    pos_inputs = path_inputs.values()
    keys_inputs = path_inputs.keys()

    pos_tangent = tensor_tangent.values()

    # create a positional function that accepts a list of values
    filtered_pos_eval = filter_pos_func(
        evaluate, tensor_inputs, jvp_outputs, keys_inputs
    )

    tangent = torch.func.jvp(filtered_pos_eval, tuple(pos_inputs), tuple(pos_tangent))

    return tangent[1]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    # Make ordering of vjp in and output args deterministic
    # Necessacy as torch.vjp function requires inputs and outputs to be in the same order
    cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}

    # convert all numbers and arrays to torch tensors
    tensor_inputs = tree_map(convert_to_tensors, inputs.model_dump())
    tensor_cotangent = tree_map(convert_to_tensors, cotangent_vector)

    # flatten the dictionaries such that they can be accessed by paths
    path_inputs = flatten_with_paths(tensor_inputs, vjp_inputs)

    # transform the dictionaries into a list of values for a positional function
    pos_inputs = path_inputs.values()
    keys_inputs = path_inputs.keys()

    # create a positional function that accepts a list of values
    filtered_pos_func = filter_pos_func(
        evaluate, tensor_inputs, vjp_outputs, keys_inputs
    )

    _, vjp_func = torch.func.vjp(filtered_pos_func, *pos_inputs)

    res = vjp_func(tensor_cotangent)

    # rebuild the dictionary from the list of results
    res_dict = {}
    for key, value in zip(vjp_inputs, res):
        res_dict[key] = value

    return res_dict


def convert_to_tensors(data):
    """Convert all numbers and arrays to torch tensors."""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data.copy())
    elif isinstance(data, (np.floating, float)):
        return torch.tensor(data)
    elif isinstance(data, (np.integer, int)):
        return torch.tensor(data)
    elif isinstance(data, (np.bool_, bool)):
        return torch.tensor(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
