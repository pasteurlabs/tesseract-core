import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .dataset import RawDataSample
from .process.utils import compute_bbox_stats
from .utils import compute_stats_from_samples, pca_align


@dataclass
class ScalerConfig:
    """Configuration for different scaling strategies."""

    point_strategy: str = (
        "unit_cube"  # none, center_only, unit_sphere, unit_cube, standardize
    )
    param_strategy: str = "standardize"  # none, standardize, normalize
    qoi_strategy: str = "standardize"  # none, standardize, normalize

    # PCA alignment
    pca_align: bool = True


@dataclass
class ScaledDataSample:
    """Scaled data sample ready for training."""

    xyz: np.ndarray  # (N, 3) transformed point coordinates
    normals: Optional[np.ndarray]  # (N, 3) transformed normal vectors or None
    params: np.ndarray  # (P,) scaled parameter values
    qoi: np.ndarray  # (Q,) scaled quantity of interest values
    source_idx: Optional[int] = None  # Index into the original dataset files


class DataScaler:
    """Handles all scaling operations for the CAD dataset."""

    def __init__(self, config: ScalerConfig):
        self.config = config
        self.fitted = False

        # Fitted parameters
        self.point_stats_ = {}
        self.param_stats_ = {}
        self.qoi_stats_ = {}

    def fit(self, global_stats: dict):
        """Fit scaler parameters based on global dataset statistics."""
        self.point_stats_ = global_stats["point_cloud"]
        self.param_stats_ = global_stats["params"]
        self.qoi_stats_ = global_stats["qoi"]

        self.fitted = True
        print("DataScaler fitted successfully!")

    # ========== POINT CLOUD SCALING METHODS ==========

    def _scale_points_none(self, xyz: np.ndarray) -> np.ndarray:
        """No scaling applied."""
        return xyz.copy()

    def _scale_points_center_only(self, xyz: np.ndarray) -> np.ndarray:
        """Center at origin only."""
        centroid = xyz.mean(axis=0)
        return xyz - centroid

    def _scale_points_unit_sphere(self, xyz: np.ndarray) -> np.ndarray:
        """Center and scale to fit in unit sphere."""
        centroid = xyz.mean(axis=0)
        xyz_centered = xyz - centroid
        # Use maximum distance from center
        max_dist = np.max(np.linalg.norm(xyz_centered, axis=1))
        scale_factor = max_dist + 1e-8
        return xyz_centered / scale_factor

    def _scale_points_unit_cube(self, xyz: np.ndarray) -> np.ndarray:
        """Center and scale to fit in unit cube."""
        centroid = xyz.mean(axis=0)
        xyz_centered = xyz - centroid
        # Use maximum range in any dimension
        bbox_dict, _ = compute_bbox_stats(xyz_centered)
        max_side = bbox_dict["max_side"]
        scale_factor = max_side + 1e-8
        return xyz_centered / scale_factor

    def _scale_points_standardize(self, xyz: np.ndarray) -> np.ndarray:
        """Standardize using global statistics (z-score normalization)."""
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        mean = self.point_stats_["mean"]
        std = self.point_stats_["std"]
        return (xyz - mean) / (std + 1e-8)

    def transform_points(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform point coordinates and compute global features.

        Returns: (transformed_xyz)
        """
        # Apply point scaling based on strategy
        if self.config.point_strategy == "none":
            xyz_out = self._scale_points_none(xyz)
        elif self.config.point_strategy == "center_only":
            xyz_out = self._scale_points_center_only(xyz)
        elif self.config.point_strategy == "unit_sphere":
            xyz_out = self._scale_points_unit_sphere(xyz)
        elif self.config.point_strategy == "unit_cube":
            xyz_out = self._scale_points_unit_cube(xyz)
        elif self.config.point_strategy == "standardize":
            xyz_out = self._scale_points_standardize(xyz)
        else:
            raise ValueError(f"Unknown point strategy: {self.config.point_strategy}")

        return xyz_out.astype(np.float32)

    def pca_align_points(
        self, xyz: np.ndarray, normals: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply PCA alignment to points and normals."""
        if not self.config.pca_align:
            return xyz, normals

        xyz_aligned, R_pca = pca_align(xyz)

        normals_aligned = None
        if normals is not None:
            normals_aligned = (normals @ R_pca).astype(np.float32)
            # Re-normalize after rotation
            normals_aligned = normals_aligned / (
                np.linalg.norm(normals_aligned, axis=1, keepdims=True) + 1e-9
            )

        return xyz_aligned, normals_aligned

    # ========== SHARED SCALING METHODS ==========

    def _transform_data(
        self, data: np.ndarray, strategy: str, stats: dict
    ) -> np.ndarray:
        """Generic transform method for params and qoi."""
        if len(data) == 0:
            return data.copy()

        if strategy == "none":
            return data.copy()
        elif strategy == "standardize":
            mean, std = stats["mean"], stats["std"]
            return (data - mean) / (std + 1e-8)
        elif strategy == "normalize":
            min_val, max_val = stats["min"], stats["max"]
            return (data - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def transform_params(self, params: np.ndarray) -> np.ndarray:
        """Transform parameter values."""
        return self._transform_data(
            params, self.config.param_strategy, self.param_stats_
        )

    def transform_qoi(self, qoi: np.ndarray) -> np.ndarray:
        """Transform QoI values."""
        return self._transform_data(qoi, self.config.qoi_strategy, self.qoi_stats_)


class ScalingPipeline:
    """Simple scaling pipeline for CAD dataset."""

    def __init__(self, config_path: Path):
        self.scaler = self._create_scaler_from_config(config_path)
        self.fitted = False

    def _create_scaler_from_config(self, config_path: Path) -> DataScaler:
        """Create scaler from YAML config file."""
        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        scaling_config = config.get("scaling", {})

        # Handle naming variations
        param_strategy = scaling_config.get("params", "standardize")
        if param_strategy == "standard":
            param_strategy = "standardize"

        qoi_strategy = scaling_config.get("qoi", "standardize")
        if qoi_strategy == "standard":
            qoi_strategy = "standardize"

        scaler_config = ScalerConfig(
            point_strategy=scaling_config.get("points", "unit_cube"),
            param_strategy=param_strategy,
            qoi_strategy=qoi_strategy,
            pca_align=scaling_config.get("pca_align", True),
        )

        return DataScaler(scaler_config)

    def fit(self, train_samples: list[RawDataSample]) -> None:
        """Fit the scaler on training data."""
        train_stats = compute_stats_from_samples(train_samples)
        self.scaler.fit(train_stats)
        self.fitted = True

    def transform_sample(self, raw_sample: RawDataSample) -> ScaledDataSample:
        """Transform a single sample."""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        # PCA align
        xyz_aligned, normals_aligned = self.scaler.pca_align_points(
            raw_sample.xyz, raw_sample.normals
        )

        # Transform
        xyz_scaled = self.scaler.transform_points(xyz_aligned)
        params_scaled = self.scaler.transform_params(raw_sample.params)
        qoi_scaled = self.scaler.transform_qoi(raw_sample.qoi)

        return ScaledDataSample(
            xyz=xyz_scaled,
            normals=normals_aligned,
            params=params_scaled,
            qoi=qoi_scaled,
            source_idx=raw_sample.source_idx,
        )

    def transform_samples(
        self, raw_samples: list[RawDataSample]
    ) -> list[ScaledDataSample]:
        """Transform a list of samples."""
        return [self.transform_sample(sample) for sample in raw_samples]

    def save(self, filepath: Path) -> Path:
        """Save the fitted scaling pipeline to a pickle file.

        Args:
            filepath: Path where to save the scaler pickle file

        Returns:
            Path to the saved file
        """
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before saving")

        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        print(f"ScalingPipeline saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "ScalingPipeline":
        """Load a fitted scaling pipeline from a pickle file.

        Args:
            filepath: Path to the saved scaler pickle file

        Returns:
            Loaded ScalingPipeline instance
        """
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)

        if not isinstance(pipeline, cls):
            raise TypeError(f"Loaded object is not a ScalingPipeline, got {type(pipeline)}")

        # Ensure fitted flag is set to True after loading
        pipeline.fitted = True

        print(f"ScalingPipeline loaded from {filepath}")
        return pipeline
