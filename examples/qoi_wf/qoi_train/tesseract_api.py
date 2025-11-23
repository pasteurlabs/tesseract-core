# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
import yaml

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torch.utils._pytree import tree_map
from typing_extensions import Self
from pathlib import Path
from scripts.dataset import CADDataset, create_raw_splits, create_scaled_datasets
from scripts.scaler import ScalingPipeline
from scripts.train import create_training_args_from_config, train_hybrid_models

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

    data: list[str | Path] = Field(
        description="List of npz files containing point-cloud data, simulation parameters and/or QoIs"
    ) # TODO: Change input type to be list[InputFileReference] (as outputs are inside the qoi_dataset... How can we make this?)

    

class OutputSchema(BaseModel):

    trained_model: OutputFileReference = Field(
        description="Pickle file containing weights of trained model"
    )
    scaler: OutputFileReference = Field(
        description="Pickle file containing the scaling method for the dataset"
    ) 
   

def evaluate(inputs: Any) -> Any:

    raw_dataset = CADDataset(files=inputs["data"], config_path=inputs["config"])

    with open(inputs["config"], "r") as f:
        config = yaml.safe_load(f)

    train_samples, val_samples, test_samples, split_info = create_raw_splits(
        dataset=raw_dataset,
        train_ratio=config["model_spec"]["train_ratio"],
        val_ratio=config["model_spec"]["val_ratio"],
        test_ratio=config["model_spec"]["test_ratio"],
        seed=config["random_seed"],
    )

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scaling pipeline from config
    scaling_pipeline = ScalingPipeline(inputs["config"])
    scaling_pipeline.fit(train_samples)

    scaled_train = scaling_pipeline.transform_samples(train_samples)
    scaled_val = scaling_pipeline.transform_samples(val_samples)
    scaled_test = scaling_pipeline.transform_samples(test_samples)

    # Save the scaler to a pickle file using the save method
    scaler_path = scaling_pipeline.save(output_dir / "scaler.pkl")

    train_dataset, val_dataset, test_dataset = create_scaled_datasets(
        scaled_train, scaled_val, scaled_test
    )

    model_folder = output_dir / "models"
    training_args = create_training_args_from_config(
        config, train_dataset, val_dataset, split_info, model_folder
    )

    hybrid_model_configs = config.get("hybrid_models", None)
    hybrid_training_config = config.get("hybrid_training", {})
    print("\nStarting hybrid model training...")
    model_path = train_hybrid_models(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model_configs=hybrid_model_configs,
        training_config=hybrid_training_config,
        save_dir=model_folder,
        config_path=inputs["config"],
        split_info=split_info,
    )

    return {
        "trained_model": Path(model_path),  # placeholder for now
        "scaler": scaler_path
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

