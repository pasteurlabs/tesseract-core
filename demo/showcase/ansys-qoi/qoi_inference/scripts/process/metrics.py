"""Model evaluation metrics for regression tasks."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass
class ModelMetrics:
    """Comprehensive regression metrics.

    Includes both absolute and normalized (scale-independent) metrics.
    """

    # Absolute metrics
    mse: float
    mae: float
    r2: float
    rmse: float
    mape: float = None
    max_error: float = None

    # Normalized metrics (scale-independent)
    nmse: float = None  # Normalized by variance
    nrmse: float = None  # Normalized by std
    nmae: float = None  # Normalized by mean

    def __post_init__(self):
        if self.rmse is None:
            self.rmse = np.sqrt(self.mse)

    def __str__(self):
        parts = [f"R²: {self.r2:.4f}"]

        # Show normalized metrics (scale-independent)
        if self.nmse is not None:
            parts.append(f"NMSE: {self.nmse:.4f}")
        if self.nrmse is not None:
            parts.append(f"NRMSE: {self.nrmse:.4f}")
        if self.nmae is not None:
            parts.append(f"NMAE: {self.nmae:.4f}")

        # Absolute metrics
        parts.append(
            f"| MSE: {self.mse:.6f}, RMSE: {self.rmse:.6f}, MAE: {self.mae:.6f}"
        )

        if self.mape is not None:
            parts.append(f"MAPE: {self.mape:.2f}%")
        if self.max_error is not None:
            parts.append(f"Max: {self.max_error:.6f}")

        return ", ".join(parts)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "mse": float(self.mse),
            "mae": float(self.mae),
            "r2": float(self.r2),
            "rmse": float(self.rmse),
            "mape": float(self.mape) if self.mape is not None else None,
            "max_error": float(self.max_error) if self.max_error is not None else None,
            "nmse": float(self.nmse) if self.nmse is not None else None,
            "nrmse": float(self.nrmse) if self.nrmse is not None else None,
            "nmae": float(self.nmae) if self.nmae is not None else None,
        }


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> ModelMetrics:
    """Compute comprehensive regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        ModelMetrics with all computed metrics
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Basic metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Normalized metrics
    y_var = np.var(y_true)
    y_mean = np.mean(y_true)

    nmse = mse / (y_var + 1e-8) if y_var > 1e-8 else None
    nrmse = rmse / (np.sqrt(y_var) + 1e-8) if y_var > 1e-8 else None
    nmae = mae / (np.abs(y_mean) + 1e-8) if np.abs(y_mean) > 1e-8 else None

    # MAPE (avoid division by zero)
    nonzero_mask = np.abs(y_true) > 1e-8
    mape = (
        np.mean(
            np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
        )
        * 100
        if np.any(nonzero_mask)
        else None
    )
    # Maximum absolute error
    max_error = np.max(np.abs(y_true - y_pred))

    return ModelMetrics(
        mse=mse,
        mae=mae,
        r2=r2,
        rmse=rmse,
        mape=mape,
        max_error=max_error,
        nmse=nmse,
        nrmse=nrmse,
        nmae=nmae,
    )
