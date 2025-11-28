"""Point cloud processing from STL mesh files."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import load_mesh, sample_points, sample_points_with_spheres


@dataclass
class SphereSamplingConfig:
    """Configuration for focused sampling in spherical regions of interest."""

    enabled: bool = False
    radius: float = 0.2
    fraction: float = 0.3
    centers: list[list[float]] = field(default_factory=list)

    def to_spheres(self) -> list[tuple[np.ndarray, float]]:
        """Convert center coordinates to (center_array, radius) tuples."""
        return [(np.array(c, dtype=np.float32), self.radius) for c in self.centers]


@dataclass
class PointConfig:
    """Configuration for point cloud sampling and augmentation."""

    n_points: int = 2048
    sampling_method: str = "poisson"  # Options: "poisson", "uniform"
    sphere_sampling: SphereSamplingConfig = field(default_factory=SphereSamplingConfig)

    # Augmentation settings (for training)
    apply_augmentation: bool = False
    augment_rotation_deg: float = 15.0
    augment_jitter_std: float = 0.0
    augment_jitter_clip: float = 0.0
    augment_translation_range: float = 0.005
    augment_enable_scaling: bool = False


@dataclass
class PointProcessor:
    """Extracts point clouds with normals from STL mesh files."""

    cfg: PointConfig

    def download(self, folder: Path) -> tuple[np.ndarray, np.ndarray]:
        """Sample points from the STL file in the given folder.

        Args:
            folder: Directory containing a single .stl file

        Returns:
            (points, normals): Arrays of shape (N, 3) each
        """
        stl_path = self._find_stl(folder)
        mesh = load_mesh(stl_path)

        if self.cfg.sphere_sampling.enabled:
            return sample_points_with_spheres(
                mesh,
                n_points=self.cfg.n_points,
                method=self.cfg.sampling_method,
                spheres=self.cfg.sphere_sampling.to_spheres(),
                sphere_fraction=self.cfg.sphere_sampling.fraction,
            )

        return sample_points(
            mesh, n_points=self.cfg.n_points, method=self.cfg.sampling_method
        )

    @staticmethod
    def _find_stl(folder: Path) -> Path:
        """Find the single STL file in folder, raising an error if none or multiple exist."""
        stls = list(folder.glob("*.stl"))
        if len(stls) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 STL file in {folder}, found {len(stls)}"
            )
        return stls[0]
