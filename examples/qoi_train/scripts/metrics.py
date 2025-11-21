from dataclasses import dataclass
import torch
import numpy as np
from typing import Union


@dataclass
class ModelMetrics:
    """Unified metrics for both torch and sklearn models."""
    mse: float
    mae: float
    r2: float
    rmse: float
    mape: float
    max_error: float
    # Normalized metrics (scale-independent)
    nmse: float = None  # Normalized MSE
    nrmse: float = None  # Normalized RMSE
    nmae: float = None  # Normalized MAE

    def __post_init__(self):
        if self.rmse is None:
            self.rmse = np.sqrt(self.mse)

    def __str__(self):
        metrics_str = f"R²: {self.r2:.4f}"

        # Show normalized metrics if available (these are scale-independent)
        if self.nmse is not None:
            metrics_str += f", NMSE: {self.nmse:.4f}"
        if self.nrmse is not None:
            metrics_str += f", NRMSE: {self.nrmse:.4f}"
        if self.nmae is not None:
            metrics_str += f", NMAE: {self.nmae:.4f}"

        # Also show absolute metrics
        metrics_str += f" | MSE: {self.mse:.6f}, RMSE: {self.rmse:.6f}, MAE: {self.mae:.6f}"

        if self.mape is not None:
            metrics_str += f", MAPE: {self.mape:.2f}%"
        if self.max_error is not None:
            metrics_str += f", Max: {self.max_error:.6f}"

        return metrics_str

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = {
            'mse': float(self.mse),
            'mae': float(self.mae),
            'r2': float(self.r2),
            'rmse': float(self.rmse),
            'mape': float(self.mape) if self.mape is not None else None,
            'max_error': float(self.max_error) if self.max_error is not None else None,
        }

        # Add normalized metrics
        if self.nmse is not None:
            result['nmse'] = float(self.nmse)
        if self.nrmse is not None:
            result['nrmse'] = float(self.nrmse)
        if self.nmae is not None:
            result['nmae'] = float(self.nmae)

        return result


def compute_metrics(y_true: Union[np.ndarray, torch.Tensor],
                   y_pred: Union[np.ndarray, torch.Tensor],
                   include_additional: bool = True) -> ModelMetrics:
    """
    Compute comprehensive metrics for model evaluation.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        include_additional: Whether to include MAPE and max_error

    Returns:
        ModelMetrics object with all computed metrics including normalized versions
    """
    # Convert to numpy if torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to handle multi-dimensional outputs
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Basic metrics
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(mse)

    # R² (coefficient of determination)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero

    # Normalized metrics (scale-independent)
    # These are normalized by the variance of the true values
    y_true_var = np.var(y_true_flat)
    y_true_range = np.max(y_true_flat) - np.min(y_true_flat)
    y_true_mean = np.mean(y_true_flat)

    # NMSE: Normalized by variance (0 is perfect, >1 is worse than predicting mean)
    nmse = mse / (y_true_var + 1e-8) if y_true_var > 1e-8 else None

    # NRMSE: Multiple common normalizations
    # Option 1: Normalized by range (percentage of range)
    nrmse_range = rmse / (y_true_range + 1e-8) if y_true_range > 1e-8 else None

    # Option 2: Normalized by mean (coefficient of variation of RMSE)
    nrmse_mean = rmse / (np.abs(y_true_mean) + 1e-8) if np.abs(y_true_mean) > 1e-8 else None

    # Option 3: Normalized by std (most common for comparing models)
    nrmse_std = rmse / (np.sqrt(y_true_var) + 1e-8) if y_true_var > 1e-8 else None

    # Use std-normalized version as default (most commonly used)
    nrmse = nrmse_std

    # NMAE: Normalized by mean absolute value
    nmae = mae / (np.abs(y_true_mean) + 1e-8) if np.abs(y_true_mean) > 1e-8 else None

    # Additional metrics
    mape = None
    max_error = None

    if include_additional:
        # Mean Absolute Percentage Error (avoid division by zero)
        nonzero_mask = np.abs(y_true_flat) > 1e-8
        if np.any(nonzero_mask):
            mape = np.mean(np.abs((y_true_flat[nonzero_mask] - y_pred_flat[nonzero_mask]) / y_true_flat[nonzero_mask])) * 100

        # Maximum absolute error
        max_error = np.max(np.abs(y_true_flat - y_pred_flat))

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
