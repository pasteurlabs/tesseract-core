import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from .utils import extract_cad_sketch


@dataclass
class GeometryParamsConfig:
    """Configuration for geometry processing."""

    # File config
    file: str = "design_table_custom.csv"


@dataclass
class GeometryParamsProcessor:
    """
    Processes geometry data from configuration files.
    """

    def __init__(self, config: GeometryParamsConfig):
        self.cfg = config

    def download(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process geometry data from a folder containing configuration files.

        Args:
            folder: Path to folder containing configuration files

        Returns:
            Tuple of (geometry_names, geometry) as numpy arrays
        """
        filename = folder / self.cfg.file
        geom_names, geom_values = extract_cad_sketch(filename)
        return np.array(geom_names), np.array(geom_values)


@dataclass
class BCParamsConfig:
    """Configuration for parameters processing."""

    # File config
    file: str = "metadata.json.series"
    variations: List[str] = None

    # Processing config
    normalize: bool = False
    log_transform: bool = False


@dataclass
class BCParamsProcessor:
    """
    Processes parameters from JSON metadata files based on configuration settings.
    """

    def __init__(self, config: BCParamsConfig):
        self.cfg = config

    def download(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process parameter data from a folder containing metadata JSON.

        Args:
            folder: Path to folder containing metadata JSON file

        Returns:
            Tuple of (param_names, params) as numpy arrays
        """
        # load config
        filename = str(self.cfg.file)
        var_keys = list(self.cfg.variations)

        # logic
        metadata_json = self._load_params_json(folder / filename)
        params_names = np.array([v for v in var_keys])
        params = np.array([metadata_json["variations"][v] for v in var_keys])

        return params_names, params

    @staticmethod
    def _load_params_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
