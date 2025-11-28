# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract API for QoI model inference."""

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


class InputSchema(BaseModel):
    """Input schema for QoI model inference."""

    config: str = Field(description="Configuration file")

    data_folder: str = Field(
        description="Folder containing npz files with point cloud data and simulation parameters"
    )
    trained_model: str = Field(
        description="Pickle file containing weights of trained model"
    )
    scaler: str = Field(
        description="Pickle file containing the scaling method for the dataset"
    )


class OutputSchema(BaseModel):
    """Output schema for QoI model inference."""

    qoi: Array[(None, None), Float32] = Field(
        description="QoIs - 2D array where each row is a prediction",
    )


def evaluate(inputs: Any) -> Any:
    """Run inference on QoI prediction models."""
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
    """Apply model inference process to input data.

    Args:
        inputs: Input schema containing config, data paths, model and scaler

    Returns:
        Output schema with QoI predictions
    """
    # Convert to pytorch tensors
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    out = evaluate(tensor_inputs)
    return out


def to_tensor(x: Any) -> torch.Tensor | Any:
    """Convert numpy arrays/scalars to torch tensors, pass through other types."""
    if isinstance(x, np.generic | np.ndarray):
        return torch.tensor(x)
    return x
