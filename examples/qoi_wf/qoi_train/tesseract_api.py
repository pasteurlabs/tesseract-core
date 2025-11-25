# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field
from scripts.dataset import CADDataset, create_raw_splits, create_scaled_datasets
from scripts.scaler import ScalingPipeline
from scripts.train import train_hybrid_models
from torch.utils._pytree import tree_map

from tesseract_core.runtime.experimental import InputFileReference, OutputFileReference

#
# Schemata
#


class InputSchema(BaseModel):
    config: InputFileReference = Field(description="Configuration file")

    data: list[str] = Field(
        description="List of npz file paths (can be absolute paths from dependent workflows)"
    )


class OutputSchema(BaseModel):
    trained_models: list[OutputFileReference] = Field(
        description="Pickle file containing weights of trained model"
    )
    scalers: list[OutputFileReference] = Field(
        description="Pickle file containing the scaling method for the dataset"
    )


def evaluate(inputs: Any) -> Any:
    # Convert all inputs to Path objects (handles strings, InputFileReference, and Path)
    config_path = Path(str(inputs["config"]))
    data_files = [Path(str(f)) for f in inputs["data"]]

    raw_dataset = CADDataset(files=data_files, config_path=config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_samples, val_samples, test_samples, split_info = create_raw_splits(
        dataset=raw_dataset,
        train_ratio=config["model_spec"]["train_ratio"],
        val_ratio=config["model_spec"]["val_ratio"],
        test_ratio=config["model_spec"]["test_ratio"],
        seed=config["random_seed"],
    )

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scaling pipeline from config
    scaling_pipeline = ScalingPipeline(config_path)
    scaling_pipeline.fit(train_samples)

    scaled_train = scaling_pipeline.transform_samples(train_samples)
    scaled_val = scaling_pipeline.transform_samples(val_samples)
    scaled_test = scaling_pipeline.transform_samples(test_samples)

    train_dataset, val_dataset, test_dataset = create_scaled_datasets(
        scaled_train, scaled_val, scaled_test
    )

    model_folder = output_dir / "models"

    hybrid_model_configs = config.get("hybrid_models", None)
    hybrid_training_config = config.get("hybrid_training", {})
    print("\nStarting hybrid model training...")
    results = train_hybrid_models(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model_configs=hybrid_model_configs,
        training_config=hybrid_training_config,
        save_dir=model_folder,
        config_path=config_path,
        split_info=split_info,
        scaler=scaling_pipeline,
    )

    print(results)
    # Extract model paths (exclude scaler_path from results dict)
    model_paths = [Path(info["model_path"]) for _, info in results.items()]

    # Get the scaler path from results (saved in experiment folder by train_hybrid_models)
    scaler_paths = [Path(info["scaler_path"]) for _, info in results.items()]

    return {
        "trained_models": model_paths,
        "scalers": scaler_paths,
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


def to_tensor(x: Any) -> torch.Tensor | Any:
    """Convert numpy arrays/scalars to torch tensors, pass through other types."""
    if isinstance(x, np.generic | np.ndarray):
        return torch.tensor(x)
    return x


#
# Required endpoints
#
