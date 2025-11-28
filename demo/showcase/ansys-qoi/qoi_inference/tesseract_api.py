# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field
from torch.utils._pytree import tree_map

from tesseract_core.runtime import Array, Float32
from tesseract_core.runtime.experimental import InputFileReference

#
# Schemata
#


class InputSchema(BaseModel):
    config: str = Field(description="Configuration file")

    data_folder: str = Field(
        description="Folder containing the list of npz files containing point-cloud data, simulation parameters and/or QoIs"
    )  
    trained_model: str = Field(
        description="Pickle file containing weights of trained model"
    )
    scaler: str = Field(
        description="Pickle file containing the scaling method for the dataset"
    )


class OutputSchema(BaseModel):
    qoi: Array[(None, None), Float32] = Field(
        description="QoIs - 2D array where each row is a prediction",
    )


def evaluate(inputs: Any) -> Any:
    from process.dataset import CADDataset, ScaledCADDataset, cad_collate
    from process.models import HybridPointCloudTreeModel
    from process.scaler import ScalingPipeline
    config_path = Path(inputs["config"])
    data_folder_path = Path(inputs["data_folder"])
    files = [str(p.resolve()) for p in data_folder_path.glob("*.npz")]

    data_files = [Path(f) for f in files]

    raw_dataset = CADDataset(files=data_files, config_path=config_path)

    with open(inputs["config"]) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path("/tesseract/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the scaling pipeline from saved pickle file
    scaling_pipeline = ScalingPipeline.load(Path(inputs["scaler"]))

    # Get all inference samples from the dataset
    inference_samples = [raw_dataset[i] for i in range(len(raw_dataset))]

    # Transform samples using the loaded scaler
    scaled_inference_samples = scaling_pipeline.transform_samples(inference_samples)

    # Create scaled dataset for inference
    inference_dataset = ScaledCADDataset(scaled_inference_samples)

    # Create data loader with collate function
    batch_size = config.get("inference", {}).get("batch_size", 32)
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=cad_collate,
    )

    # Load the trained model
    print("Loading trained model...")
    model = HybridPointCloudTreeModel()
    model.load(inputs["trained_model"])

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(inference_loader)

    # Convert predictions to 2D torch tensor (stacking all predictions)
    qoi_predictions = torch.tensor(predictions, dtype=torch.float32)

    # Save predictions to multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as CSV
    csv_path = output_dir / f"predictions_{timestamp}.csv"
    predictions_array = qoi_predictions.numpy()

    # Determine number of QoI outputs
    n_samples, n_qoi = predictions_array.shape

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        header = ["sample_id"] + [f"qoi_{i}" for i in range(n_qoi)]
        writer.writerow(header)
        # Write data
        for i, pred in enumerate(predictions_array):
            writer.writerow([i, *pred.tolist()])

    print(f"Saved predictions to {csv_path}")
    print(
        f"Predictions shape: {predictions_array.shape} ({n_samples} samples, {n_qoi} QoI outputs)"
    )

    return {"qoi": qoi_predictions}


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
