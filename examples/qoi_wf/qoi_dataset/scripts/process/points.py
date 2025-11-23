import numpy as np
from dataclasses import dataclass, field 
from pathlib import Path
from typing import Tuple
from .utils import load_mesh, sample_points, sample_points_with_spheres



@dataclass
class SphereSamplingConfig:
    enabled: bool = False
    radius: float = 0.2
    fraction: float = 0.3
    centers: list[list[float]] = field(default_factory=list)

    def to_spheres(self):
        """Convert to list of (center_np, radius) tuples for sampling code."""
        return [(np.array(c, dtype=np.float32), self.radius) for c in self.centers]


@dataclass
class PointConfig:
    """Configuration for point cloud processing."""

    # NPZ creation config
    n_points: int = 2048
    sampling_method: str = "poisson"  # poisson, uniform

    sphere_sampling: SphereSamplingConfig = field(default_factory=SphereSamplingConfig)

    # Augmentation config
    apply_augmentation: bool = False
    augment_rotation_deg: float = 15.0
    augment_jitter_std: float = 0.0
    augment_jitter_clip: float = 0.0
    augment_translation_range: float = 0.005
    augment_enable_scaling: bool = False

@dataclass
class PointProcessor:
    """
    Processes point clouds from STL files based on configuration settings.
    """

    def __init__(self, config: PointConfig):
        self.cfg = config

    def download(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process point cloud data from a folder containing STL files.

        Args:
            folder: Path to folder containing STL file

        Returns:
            Tuple of (points, normals) as numpy arrays
        """
        # load config
        n_points = self.cfg.n_points
        method = self.cfg.sampling_method

        # logic
        stl = self._find_single_stl(folder)
        mesh = load_mesh(stl)
        if self.cfg.sphere_sampling.enabled:
            spheres = self.cfg.sphere_sampling.to_spheres()
            fraction = self.cfg.sphere_sampling.fraction
            
            pts_raw, nrm = sample_points_with_spheres(
                mesh, 
                n_points=n_points, 
                method=method,
                spheres=spheres,
                sphere_fraction=fraction,
            )
        else:
            pts_raw, nrm = sample_points(mesh, n_points=n_points, method=method)
        return pts_raw, nrm

    @staticmethod
    def _find_single_stl(folder: Path) -> Path:
        stls = list(folder.glob("*.stl"))
        if len(stls) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 STL in {folder}, found {len(stls)}"
            )
        return stls[0]
