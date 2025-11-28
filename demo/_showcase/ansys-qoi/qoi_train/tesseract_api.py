# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract API for QoI model training."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field
from torch.utils._pytree import tree_map

from tesseract_core.runtime.experimental import OutputFileReference


class InputSchema(BaseModel):
    """Input schema for QoI model training."""

    config: str = Field(description="Configuration file")

    data_folder: str = Field(
        description="Folder containing npz files containing point cloud data information, "
        "simulation parameters and QoI"
    )


class OutputSchema(BaseModel):
    """Output schema for QoI model training."""

    trained_models: list[OutputFileReference] = Field(
        description="Pickle file containing weights of trained model"
    )
    scalers: list[OutputFileReference] = Field(
        description="Pickle file containing the scaling method for the dataset"
    )


def evaluate(inputs: Any) -> Any:
    """Train QoI prediction models on the dataset."""
    from process.dataset import CADDataset, create_raw_splits, create_scaled_datasets
    from process.scaler import ScalingPipeline
    from process.train import train_hybrid_models

    # Convert all inputs to Path objects (handles strings, InputFileReference, and Path)
    config_path = Path(inputs["config"])
    data_folder_path = Path(inputs["data_folder"])
    files = [str(p.resolve()) for p in data_folder_path.glob("*.npz")]

    data_files = [Path(f) for f in files]

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
    output_dir = Path("/tesseract/outputs")
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
    """Apply model training process to input data.

    Args:
        inputs: Input schema containing config and data paths

    Returns:
        Output schema with trained models and scalers
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
