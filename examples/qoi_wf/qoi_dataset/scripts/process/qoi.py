"""Quantity of Interest (QoI) extraction from simulation reports."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from .utils import SurfaceIntegralReport


@dataclass
class QoiConfig:
    """Configuration for QoI data extraction."""
    files: List[str] = field(default_factory=lambda: ["all_pressure.txt"])


@dataclass
class QoiProcessor:
    """Extracts quantities of interest from simulation report files."""

    cfg: QoiConfig

    def download(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract QoI data from report files in the given folder.

        Args:
            folder: Directory containing report files

        Returns:
            (qoi_names, qoi_values): Arrays of QoI names and their values
        """
        qoi_data = {}

        for filename in self.cfg.files:
            file_path = folder / filename
            try:
                report = SurfaceIntegralReport.from_file(file_path)
                file_stem = Path(filename).stem

                # Add file suffix to distinguish QoI from different files
                for key, value in report.values.items():
                    qoi_data[f"{key}_{file_stem}"] = value

            except Exception as e:
                print(f"⚠️  Error reading {filename}: {e}")
                continue

        return np.array(list(qoi_data.keys())), np.array(list(qoi_data.values()))
