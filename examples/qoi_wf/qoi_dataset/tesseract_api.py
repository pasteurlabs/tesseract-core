# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator, ConfigDict
from torch.utils._pytree import tree_map
from typing_extensions import Self
from scripts.dataset import CADDataset

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

from tesseract_core.runtime.experimental import InputFileReference, OutputFileReference

#
# Schemata
#

class InputSchema(BaseModel):

    config: InputFileReference = Field(
        description="Configuration file"
    )

    sim_folder: str | Path = Field(
        description="Folder path containing CAD files and simulation results",
    )

    dataset_folder: str | Path = Field(
        description="Folder path where postprocessed simulations will be dumped into"
    )

class OutputSchema(BaseModel):
    data: list[OutputFileReference] = Field(
        description="List of npz files containing point-cloud data, simulation parameters and/or QoIs",
    )
   

def evaluate(inputs: Any) -> Any:
    from scripts.process.npz import NPZProcessor

    processor = NPZProcessor(root=inputs["sim_folder"], out_dir=inputs["dataset_folder"], config_path=inputs["config"])
    processed_files = processor.build()

    return {
        "data" : processed_files
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


# def vector_jacobian_product(
#     inputs: InputSchema,
#     vjp_inputs: set[str],
#     vjp_outputs: set[str],
#     cotangent_vector: dict[str, Any],
# ):
#     # Cast to tuples for consistent ordering in positional function
#     vjp_inputs = tuple(vjp_inputs)
#     # Make ordering of cotangent_vector identical to vjp_inputs
#     cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}

#     # convert all numbers and arrays to torch tensors
#     tensor_inputs = tree_map(to_tensor, inputs.model_dump())
#     tensor_cotangent = tree_map(to_tensor, cotangent_vector)

#     # flatten the dictionaries such that they can be accessed by paths
#     pos_inputs = flatten_with_paths(tensor_inputs, vjp_inputs).values()

#     # create a positional function that accepts a list of values
#     filtered_pos_func = filter_func(
#         evaluate, tensor_inputs, vjp_outputs, input_paths=vjp_inputs
#     )

#     _, vjp_func = torch.func.vjp(filtered_pos_func, *pos_inputs)

#     vjp_vals = vjp_func(tensor_cotangent)
#     return dict(zip(vjp_inputs, vjp_vals, strict=True))


to_tensor = lambda x: torch.tensor(x) if isinstance(x, np.generic | np.ndarray) else x


#
# Required endpoints
#
