# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils._pytree import tree_map


class InputSchema(BaseModel):
    """Input schema for QoI dataset generation."""

    config: str = Field(description="Configuration file")

    sim_folder: str = Field(
        description="Folder path containing CAD files and simulation results",
    )

    dataset_folder: str = Field(
        description="Folder path where postprocessed simulations will be dumped into"
    )


class OutputSchema(BaseModel):
    """Output schema for QoI dataset generation."""

    data: list[str | Path] = Field(
        description="List of npz files containing point-cloud data, simulation parameters and/or QoIs",
    )


def evaluate(inputs: Any) -> Any:
    """Process simulation data and generate NPZ dataset files."""
    from process.npz import NPZProcessor

    processor = NPZProcessor(
        root=Path(inputs["sim_folder"]),
        out_dir=Path(inputs["dataset_folder"]),
        config_path=Path(inputs["config"]),
    )
    processed_files = processor.build()

    return {"data": processed_files}


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply dataset generation process to input data.

    Args:
        inputs: Input schema containing config and data paths

    Returns:
        Output schema with list of generated NPZ files
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
