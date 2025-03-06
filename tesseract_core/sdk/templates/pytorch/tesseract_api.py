# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for {{name}}
# Generated by tesseract {{version}} on {{timestamp}}

from typing import Any, Callable

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from tesseract_core.runtime import Differentiable, Float32
from tesseract_core.runtime.tree_transforms import (
    flatten_with_paths,
    set_at_path,
)

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


# TODO: Add or import your function here, must be differentiable and
# take/return a single pytree as an input/output conforming respectively
# to Input/OutputSchema
def evaluate(inputs: Any) -> Any:
    print(inputs)
    return {
        "example": inputs["example"] + inputs["b"],
        "c": inputs["example"] * inputs["b"],
    }


def apply(inputs: InputSchema) -> OutputSchema:
    # Convert to pytorch tensors to enable torch.jit
    inputs = convert_to_tensors(inputs.model_dump())

    # Optional: Insert any pre-processing/setup that doesn't require tracing
    # and is only required when specifically running your apply function
    # and not your differentiable endpoints.
    # For example, you might want to set up a logger or mlflow server.
    # Pre-processing should not modify any input that could impact the
    # differentiable outputs in a nonlinear way (a constant shift
    # should be safe)

    out = evaluate(inputs)

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
    tensor_inputs = convert_to_tensors(inputs.model_dump())

    # flatten the dictionaries such that they can be accessed by paths
    path_inputs = flatten_with_paths(tensor_inputs, jac_inputs)

    # transform the dictionaries into a list of values for a positional function
    pos_inputs, treedef = tree_flatten(path_inputs)

    # create a positional function that accepts a list of values and returns a set of tuples
    filtered_pos_eval = filter_pos_func(
        evaluate, tensor_inputs, jac_outputs, treedef, True
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
    tensor_inputs = convert_to_tensors(inputs.model_dump())
    tensor_tangent = convert_to_tensors(tangent)

    # flatten the dictionaries such that they can be accessed by paths
    path_inputs = flatten_with_paths(tensor_inputs, jvp_inputs)

    # transform the dictionaries into a list of values for a positional function
    pos_inputs, treedef = tree_flatten(path_inputs)
    pos_tangent, _ = tree_flatten(tensor_tangent)

    # sort

    # create a positional function that accepts a list of values
    filtered_pos_eval = filter_pos_func(evaluate, tensor_inputs, jvp_outputs, treedef)

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
    vjp_inputs = list(vjp_inputs)
    vjp_outputs = list(vjp_outputs)
    vjp_inputs.sort()
    vjp_outputs.sort()

    # sort the cotangent vector
    cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}

    # convert all numbers and arrays to torch tensors
    tensor_inputs = convert_to_tensors(inputs.model_dump())
    tensor_cotangent = convert_to_tensors(cotangent_vector)

    # flatten the dictionaries such that they can be accessed by paths
    path_inputs = flatten_with_paths(tensor_inputs, vjp_inputs)

    # transform the dictionaries into a list of values for a positional function
    pos_inputs, treedef = tree_flatten(path_inputs)

    # create a positional function that accepts a list of values
    filtered_pos_func = filter_pos_func(evaluate, tensor_inputs, vjp_outputs, treedef)

    _, vjp_func = torch.func.vjp(filtered_pos_func, *pos_inputs)

    res = vjp_func(tensor_cotangent)

    # rebuild the dictionary from the list of results
    res_dict = {}
    for key, value in zip(vjp_inputs, res):
        res_dict[key] = value

    return res_dict


def filter_pos_func(
    func: Callable[[dict], dict],
    default_inputs: dict,
    output_paths: set[str],
    pytree: TreeSpec,
    output_to_tuple: bool = False,
) -> Callable:
    """Returns a reduced func with default inputs that operates on positional arguments.

    The returned function will accept a tuple of positional arguments,
    convert them back to a dictionary and update the default inputs
    with the new values at each path. It will then call the original function with the updated inputs
    and return a dictionary `{output_path: value}`.
    """

    # function that accepts positional arguments
    def filtered_pos_func(*args):
        # convert back to dictionary
        new_inputs = tree_unflatten(args, pytree)

        # partially update the default inputs with the new values
        updated_inputs = set_at_path(default_inputs, new_inputs)

        path_outputs = flatten_with_paths(func(updated_inputs), output_paths)

        if output_to_tuple:
            return tuple(tree_flatten(path_outputs)[0])

        return path_outputs

    return filtered_pos_func


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
    elif isinstance(data, dict):
        return {key: convert_to_tensors(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_tensors(item) for item in data]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def convert_to_numpy(data):
    """Convert all numbers and arrays to numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, dict):
        return {key: convert_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(item) for item in data]
    else:
        return data
