"""Parameter extraction from metadata and geometry files."""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from .utils import extract_cad_sketch


@dataclass
class GeometryParamsConfig:
    """Configuration for geometry parameter extraction."""
    file: str = "design_table_custom.csv"


@dataclass
class GeometryParamsProcessor:
    """Extracts geometry parameters from CAD design tables."""

    cfg: GeometryParamsConfig

    def download(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract geometry parameters from design table CSV.

        Args:
            folder: Directory containing the design table file

        Returns:
            (param_names, param_values): Arrays of parameter names and values
        """
        file_path = folder / self.cfg.file
        names, values = extract_cad_sketch(file_path)
        return np.array(names), np.array(values)


@dataclass
class BCParamsConfig:
    """Configuration for boundary condition parameter extraction."""
    file: str = "metadata.json.series"
    variations: List[str] = None

    # Optional processing flags (unused currently, for future extensions)
    normalize: bool = False
    log_transform: bool = False


@dataclass
class BCParamsProcessor:
    """Extracts boundary condition parameters from simulation metadata."""

    cfg: BCParamsConfig

    def download(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract BC parameters from metadata JSON file.

        Args:
            folder: Directory containing metadata JSON

        Returns:
            (param_names, param_values): Arrays of parameter names and values
        """
        file_path = folder / self.cfg.file
        metadata = self._load_json(file_path)

        param_names = np.array(self.cfg.variations)
        param_values = np.array([metadata["variations"][key] for key in self.cfg.variations])

        return param_names, param_values

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load JSON file from disk."""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
