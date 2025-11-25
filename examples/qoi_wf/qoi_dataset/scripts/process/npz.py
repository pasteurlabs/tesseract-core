"""NPZ dataset generation from simulation folders."""

import yaml
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from .points import PointProcessor, PointConfig, SphereSamplingConfig
from .params import GeometryParamsConfig, GeometryParamsProcessor, BCParamsConfig, BCParamsProcessor
from .qoi import QoiProcessor, QoiConfig


@dataclass
class NPZProcessor:
    """
    Processes simulation folders and creates compressed NPZ files.

    Each immediate subfolder in `root` is processed to create one NPZ file in `out_dir`.
    Configuration is loaded from `config_path` to specify what data to extract.
    """

    root: Path
    out_dir: Path
    config_path: Path

    def __post_init__(self):
        """Load configuration and initialize data processors."""
        self.cfg = self._load_config()
        self._init_processors()

    def _load_config(self) -> dict[str, Any]:
        """Load and validate YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _init_processors(self):
        """Initialize data processors based on configuration."""
        # Point processor (required)
        point_spec = self.cfg["point_spec"]
        sphere_config = self._parse_sphere_config(point_spec)

        point_config = PointConfig(
            n_points=point_spec["n_points"],
            sampling_method=point_spec["sampling_method"],
            sphere_sampling=sphere_config,
        )
        self.point_processor = PointProcessor(point_config)

        # BC params processor (optional)
        self.params_processor = self._init_optional_processor(
            "bc_params_spec",
            lambda spec: BCParamsProcessor(BCParamsConfig(
                file=spec["file"],
                variations=spec["variations"],
            ))
        )

        # QoI processor (optional)
        self.qoi_processor = self._init_optional_processor(
            "qoi_spec",
            lambda spec: QoiProcessor(QoiConfig(files=spec["files"]))
        )

        # Geometry processor (optional)
        self.geometry_processor = self._init_optional_processor(
            "geometry_params_spec",
            lambda spec: GeometryParamsProcessor(GeometryParamsConfig(file=spec["file"]))
        )

    def _parse_sphere_config(self, point_spec: dict) -> SphereSamplingConfig:
        """Parse sphere sampling configuration from point spec."""
        if "sphere_sampling" not in point_spec:
            return SphereSamplingConfig()

        sphere_spec = point_spec["sphere_sampling"]
        return SphereSamplingConfig(
            enabled=sphere_spec.get("enabled", False),
            radius=sphere_spec.get("radius", 0.2),
            fraction=point_spec.get("sphere_sampling_fraction", sphere_spec.get("fraction", 0.3)),
            centers=sphere_spec.get("centers", []),
        )

    def _init_optional_processor(self, config_key: str, factory_fn):
        """Initialize an optional processor if its config exists."""
        spec = self.cfg.get(config_key)
        return factory_fn(spec) if spec is not None else None

    def build(self) -> list[str | Path]:
        """
        Process all simulation folders and create NPZ files.

        Returns:
            List of paths to created NPZ files
        """
        folders = [p for p in self.root.iterdir() if p.is_dir()]
        output_paths = []
        skipped_count = 0
        processed_count = 0

        for folder in folders:
            try:
                folder_id = int(folder.name.split("_")[-1])
                out_path = self.out_dir / f"{folder_id}.npz"

                # Extract data from folder
                points, normals = self.point_processor.download(folder)
                param_names, params = self._extract_optional(self.params_processor, folder)
                qoi_names, qoi = self._extract_optional(self.qoi_processor, folder)

                # Validate QoI if configured (skip folder if empty)
                if self.qoi_processor is not None:
                    if qoi is None or len(qoi) == 0 or qoi_names is None or len(qoi_names) == 0:
                        print(f"⚠️  Skipping {folder.name}: Empty QoI data")
                        skipped_count += 1
                        continue

                # Extract geometry (optional, may fail)
                geometry_names, geometry = self._extract_optional(
                    self.geometry_processor,
                    folder,
                    catch_exceptions=True
                )

                # Save to NPZ
                self.dump_npz(out_path, points, normals, param_names, params, qoi_names, qoi, geometry_names, geometry)
                output_paths.append(out_path)
                processed_count += 1

            except Exception as e:
                print(f"❌ Failed to process {folder.name}: {e}")
                skipped_count += 1
                continue

        print(f"\n✅ Processed: {processed_count} folders")
        if skipped_count > 0:
            print(f"⚠️  Skipped: {skipped_count} folders")

        return output_paths

    def _extract_optional(self, processor, folder: Path, catch_exceptions: bool = False):
        """Extract data using optional processor, returning None if processor not configured."""
        if processor is None:
            return None, None

        try:
            return processor.download(folder)
        except (FileNotFoundError, NotImplementedError):
            if catch_exceptions:
                return None, None
            raise

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
        """
        Save processed data to compressed NPZ file.

        Args:
            out_path: Output file path
            points: Point cloud coordinates (N, 3)
            normals: Point normals (N, 3) or None
            param_names: Parameter names or None
            params: Parameter values or None
            qoi_names: QoI names or None
            qoi: QoI values or None
            geometry_names: Geometry parameter names or None
            geometry: Geometry parameter values or None
        """
        print(f"Creating NPZ: {out_path}")

        payload: dict[str, Any] = {"points": np.asarray(points, dtype=np.float32)}

        # Add optional data fields
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
        """
        Process a single folder and create its NPZ file.

        Args:
            folder: Simulation folder to process
            out_path: Output NPZ file path

        Raises:
            ValueError: If QoI data is empty (when QoI processor is configured)
            Exception: Other processing errors
        """
        try:
            # Extract data
            points, normals = self.point_processor.download(folder)
            param_names, params = self._extract_optional(self.params_processor, folder)
            qoi_names, qoi = self._extract_optional(self.qoi_processor, folder)

            # Validate QoI if configured (raise error if empty)
            if self.qoi_processor is not None:
                if qoi is None or len(qoi) == 0 or qoi_names is None or len(qoi_names) == 0:
                    raise ValueError(f"Empty QoI data for folder {folder.name}")

            # Extract geometry (optional)
            geometry_names, geometry = self._extract_optional(
                self.geometry_processor,
                folder,
                catch_exceptions=True
            )

            # Save to NPZ
            self.dump_npz(out_path, points, normals, param_names, params, qoi_names, qoi, geometry_names, geometry)

        except Exception as e:
            print(f"❌ Failed to process {folder.name}: {e}")
            raise
