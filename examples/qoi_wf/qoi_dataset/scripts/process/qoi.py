import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from .utils import SurfaceIntegralReport


@dataclass
class QoiConfig:
    """Configuration for QoI processing."""

    # File config
    files: List[str] = field(default_factory=lambda: ["all_pressure.txt"])


@dataclass
class QoiProcessor:
    """
    Processes Quantities of Interest (QoI) from pressure report files.
    """

    def __init__(self, config: QoiConfig):
        self.cfg = config

    def download(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process QoI data from a folder containing report files.

        Args:
            folder: Path to folder containing report files

        Returns:
            Tuple of (qoi_names, qoi) as numpy arrays
        """
        qoi_data = {}
        filenames = self.cfg.files
        for file in filenames:
            file_path = folder / file
           
            try:
                qoi_report = SurfaceIntegralReport.from_file(file_path)
                file_stem = Path(file).stem
                # Accumulate QoI data from each file
                for key, value in qoi_report.values.items():
                    qoi_data[f"{key}_{file_stem}"] = value
                        
            except Exception as e:
                print(f"❌ Error processing file {file}: {e}")
                continue

        qoi_names = np.array(list(qoi_data.keys()))
        qoi = np.array(list(qoi_data.values()))
        return qoi_names, qoi
