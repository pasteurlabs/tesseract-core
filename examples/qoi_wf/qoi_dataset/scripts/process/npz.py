import yaml
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from .points import PointProcessor, PointConfig
from .params import (
    GeometryParamsConfig,
    GeometryParamsProcessor,
    BCParamsConfig,
    BCParamsProcessor,
)
from .qoi import QoiProcessor, QoiConfig


@dataclass
class NPZProcessor:
    """
    Create one NPZ per immediate subfolder in `root`, written to `out_dir`.
    Uses individual processor classes for different data types.
    """

    root: Path
    out_dir: Path
    config_path: Path

    def __post_init__(self):
        """Load configuration and initialize processors."""
        self.cfg = self._load_config()
        self._init_processors()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from config.yaml file."""
        cfg_path = self.config_path

        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Missing {self.config_path} in {cfg_path.parent} — please create it."
            )
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _init_processors(self):
        """Initialize individual processors with their configurations."""
        # Initialize point processor
        from .points import SphereSamplingConfig

        point_spec = self.cfg["point_spec"]  # Will raise KeyError if missing

        # Parse sphere sampling config if present
        sphere_config = SphereSamplingConfig()
        if "sphere_sampling" in point_spec:
            sphere_spec = point_spec["sphere_sampling"]
            sphere_config = SphereSamplingConfig(
                enabled=sphere_spec.get("enabled", False),
                radius=sphere_spec.get("radius", 0.2),
                fraction=point_spec.get("sphere_sampling_fraction", sphere_spec.get("fraction", 0.3)),
                centers=sphere_spec.get("centers", []),
            )

        point_config = PointConfig(
            n_points=point_spec["n_points"],
            sampling_method=point_spec["sampling_method"],
            sphere_sampling=sphere_config,
        )
        self.point_processor = PointProcessor(point_config)

        # Initialize params processor
        params_spec = self.cfg["params_spec"]  # Will raise KeyError if missing
        params_config = BCParamsConfig(
            file=params_spec["file"],
            variations=params_spec["variations"],
        )
        self.params_processor = BCParamsProcessor(params_config)
        # Initialize QoI processor
        qoi_spec = self.cfg["qoi_spec"]  # Will raise KeyError if missing
        qoi_config = QoiConfig(files=qoi_spec["files"])
        self.qoi_processor = QoiProcessor(qoi_config)

        # Initialize geometry processor
        geometry_spec = self.cfg["geometry_spec"]  # Will raise KeyError if missing
        geometry_config = GeometryParamsConfig(
            file=geometry_spec["file"],
        )
        self.geometry_processor = GeometryParamsProcessor(geometry_config)

    def build(self) -> list[str | Path]:
        """Process all folders and create NPZ files."""
        folders = [p for p in self.root.iterdir() if p.is_dir()]

        skipped_count = 0
        processed_count = 0

        output_paths = []

        for folder in folders:
            try:
                folder_id = int(folder.name.split("_")[-1])
                out_path = self.out_dir / f"{folder_id}.npz"

                # Process different data types using individual processors
                points, normals = self.point_processor.download(folder)
                param_names, params = self.params_processor.download(folder)
                qoi_names, qoi = self.qoi_processor.download(folder)

                # Validate QoI data - skip if empty
                if qoi is None or len(qoi) == 0 or qoi.size == 0:
                    print(f"⚠️  Skipping {folder.name}: Empty QoI data")
                    skipped_count += 1
                    continue

                if qoi_names is None or len(qoi_names) == 0:
                    print(f"⚠️  Skipping {folder.name}: Empty QoI names")
                    skipped_count += 1
                    continue

                # Geometry processing (optional, may fail for some folders)
                try:
                    geometry_names, geometry = self.geometry_processor.download(folder)
                except (FileNotFoundError, NotImplementedError):
                    geometry_names, geometry = None, None

                # Create NPZ file
                self.dump_npz(
                    out_path,
                    points,
                    normals,
                    param_names,
                    params,
                    qoi_names,
                    qoi,
                    geometry_names,
                    geometry,
                )
                processed_count += 1

                output_paths.append(out_path)

            except Exception as e:
                print(f"❌ Failed to process folder {folder}: {e}")
                skipped_count += 1
                continue

        print(f"\n✅ Processed: {processed_count} folders")
        if skipped_count > 0:
            print(f"⚠️  Skipped: {skipped_count} folders (empty QoI or errors)")

        return output_paths

    def dump_npz(
        self,
        out_path: Path,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        param_names: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        qoi_names: Optional[np.ndarray] = None,
        qoi: Optional[np.ndarray] = None,
        geometry_names: Optional[np.ndarray] = None,
        geometry: Optional[np.ndarray] = None,
    ):
        """Save all processed data to a compressed NPZ file."""
        print(f"Creating NPZ: {out_path}")

        payload: dict[str, Any] = {
            "points": np.asarray(points, dtype=np.float32),  # [N,3]
        }

        if normals is not None:
            payload["normals"] = np.asarray(normals, dtype=np.float32)

        if params is not None:
            payload["bc_params"] = np.asarray(params, dtype=np.float32)
            payload["bc_param_names"] = np.asarray(param_names, dtype=object)

        if qoi is not None:
            payload["qoi"] = np.asarray(qoi, dtype=np.float32)
            payload["qoi_names"] = np.asarray(qoi_names, dtype=object)

        if geometry is not None:
            payload["geom_params"] = np.asarray(geometry, dtype=np.float32)
            payload["geom_param_names"] = np.asarray(geometry_names, dtype=object)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), **payload)

    def process_single_folder(self, folder: Path, out_path: Path) -> None:
        """Process a single folder and create its NPZ file."""
        try:
            # Process different data types using individual processors
            points, normals = self.point_processor.download(folder)
            param_names, params = self.params_processor.download(folder)
            qoi_names, qoi = self.qoi_processor.download(folder)

            # Validate QoI data - raise error if empty
            if qoi is None or len(qoi) == 0 or qoi.size == 0:
                raise ValueError(f"Empty QoI data for folder {folder.name}")

            if qoi_names is None or len(qoi_names) == 0:
                raise ValueError(f"Empty QoI names for folder {folder.name}")

            # Geometry processing (optional)
            try:
                geometry_names, geometry = self.geometry_processor.download(folder)
            except (FileNotFoundError, NotImplementedError):
                geometry_names, geometry = None, None

            # Create NPZ file
            self.dump_npz(
                out_path,
                points,
                normals,
                param_names,
                params,
                qoi_names,
                qoi,
                geometry_names,
                geometry,
            )

        except Exception as e:
            print(f"❌ Failed to process folder {folder}: {e}")
            raise
