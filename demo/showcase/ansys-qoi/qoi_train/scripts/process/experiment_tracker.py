"""Experiment tracking for model training runs."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ExperimentTracker:
    """Track experiments with automatic folder organization and metadata logging."""

    def __init__(
        self,
        base_dir: Path,
        experiment_type: str,  # "sklearn", "hybrid", "neural", etc.
        experiment_name: str | None = None,
        config_path: Path | None = None,
    ):
        """Initialize experiment tracker.

        Args:
            base_dir: Base directory for all models (e.g., "models_basic_run")
            experiment_type: Type of experiment ("sklearn", "hybrid", "neural", "pc")
            experiment_name: Optional custom name for the experiment
            config_path: Path to config file used for this experiment
        """
        self.base_dir = Path(base_dir)
        self.experiment_type = experiment_type

        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            dir_name = f"experiment_{experiment_type}_{experiment_name}_{timestamp}"
        else:
            dir_name = f"experiment_{experiment_type}_{timestamp}"

        self.run_dir = self.base_dir / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.predictions_dir = self.run_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)

        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.models_dir = self.run_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Initialize metadata
        self.metadata = {
            "experiment_type": experiment_type,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "run_dir": str(self.run_dir),
            "created_at": datetime.now().isoformat(),
        }

        self.training_history = []
        self.model_metrics = {}

        # Copy config file if provided
        if config_path:
            self.log_config(config_path)

        print(f"📁 Created experiment directory: {self.run_dir.name}")

    def log_config(self, config_path: Path):
        """Copy config file to experiment directory."""
        dest = self.run_dir / "config.yaml"
        shutil.copy(config_path, dest)
        self.metadata["config_file"] = str(config_path)
        print(f"  ✅ Saved config to: {dest.relative_to(self.base_dir)}")

    def log_dataset_info(self, split_info: dict[str, Any]):
        """Log dataset split information."""
        dataset_info_path = self.run_dir / "dataset_info.json"
        with open(dataset_info_path, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_info = {}
            for key, value in split_info.items():
                if isinstance(value, (list, np.ndarray)):
                    serializable_info[key] = (
                        [int(x) for x in value] if len(value) > 0 else []
                    )
                else:
                    serializable_info[key] = value
            json.dump(serializable_info, f, indent=2)
        print(
            f"  ✅ Saved dataset info to: {dataset_info_path.relative_to(self.base_dir)}"
        )

    def log_training_step(self, step: int, metrics: dict[str, float]):
        """Log metrics for a training step/epoch."""
        log_entry = {"step": step, **metrics}
        self.training_history.append(log_entry)

    def save_training_history(self):
        """Save complete training history to JSON."""
        if not self.training_history:
            return

        history_path = self.run_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        print(
            f"  ✅ Saved training history to: {history_path.relative_to(self.base_dir)}"
        )

    def log_model_metrics(
        self, model_name: str, metrics: dict[str, float], split: str = "test"
    ):
        """Log evaluation metrics for a specific model.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metric names and values
            split: Dataset split ("train", "val", "test")
        """
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = {}

        self.model_metrics[model_name][f"{split}_metrics"] = metrics

    def save_model_metrics(self):
        """Save all model metrics to JSON."""
        if not self.model_metrics:
            return

        metrics_path = self.run_dir / "model_metrics.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        serializable_metrics = convert_to_native(self.model_metrics)

        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"  ✅ Saved model metrics to: {metrics_path.relative_to(self.base_dir)}")

    def get_model_path(self, model_name: str, extension: str = ".pkl") -> Path:
        """Get path for saving a model file."""
        return self.models_dir / f"{model_name}{extension}"

    def get_predictions_path(self, filename: str) -> Path:
        """Get path for saving prediction files."""
        return self.predictions_dir / filename

    def get_plot_path(self, filename: str) -> Path:
        """Get path for saving plot files."""
        return self.plots_dir / filename

    def save_metadata(self):
        """Save experiment metadata to JSON."""
        metadata_path = self.run_dir / "experiment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(
            f"  ✅ Saved experiment metadata to: {metadata_path.relative_to(self.base_dir)}"
        )

    def add_metadata(self, key: str, value: Any):
        """Add custom metadata to the experiment."""
        self.metadata[key] = value

    def finalize(self):
        """Finalize the experiment by saving all metadata and summaries."""
        # Save all accumulated data
        self.save_training_history()
        self.save_model_metrics()
        self.save_metadata()

        # Create a README for the experiment
        self._create_readme()

        print(f"\n✅ Experiment finalized: {self.run_dir.name}")
        print(f"   Location: {self.run_dir}")

    def _create_readme(self):
        """Create a README file summarizing the experiment."""
        readme_path = self.run_dir / "README.md"

        with open(readme_path, "w") as f:
            f.write(f"# Experiment: {self.run_dir.name}\n\n")
            f.write(f"**Type:** {self.experiment_type}\n")
            f.write(f"**Created:** {self.metadata['created_at']}\n\n")

            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.run_dir.name}/\n")
            f.write("├── config.yaml              # Configuration used for this run\n")
            f.write("├── experiment_metadata.json # Experiment metadata\n")
            f.write("├── dataset_info.json        # Dataset split information\n")
            f.write("├── training_history.json    # Training metrics per epoch\n")
            f.write("├── model_metrics.json       # Final evaluation metrics\n")
            f.write("├── models/                  # Trained model files\n")
            f.write("├── predictions/             # Prediction CSV files\n")
            f.write("└── plots/                   # Visualization plots\n")
            f.write("```\n\n")

            # Add model metrics if available
            if self.model_metrics:
                f.write("## Model Performance\n\n")
                for model_name, metrics_dict in self.model_metrics.items():
                    f.write(f"### {model_name}\n\n")
                    for split, metrics in metrics_dict.items():
                        f.write(f"**{split}:**\n")
                        for metric_name, value in metrics.items():
                            if isinstance(value, float):
                                f.write(f"- {metric_name}: {value:.6f}\n")
                            else:
                                f.write(f"- {metric_name}: {value}\n")
                        f.write("\n")

            f.write("## Reproducing Results\n\n")
            f.write(
                "To reproduce this experiment, use the saved `config.yaml` with the same dataset splits:\n\n"
            )
            f.write("```python\n")
            f.write("# Load the config\n")
            f.write(f"config_path = Path('{self.run_dir / 'config.yaml'}')\n\n")
            f.write("# Load dataset split information\n")
            f.write(f"with open('{self.run_dir / 'dataset_info.json'}') as f:\n")
            f.write("    split_info = json.load(f)\n\n")
            f.write("# Use the same train/val/test indices to ensure reproducibility\n")
            f.write("```\n")

        print(f"  ✅ Created README: {readme_path.relative_to(self.base_dir)}")


def load_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Load an experiment's metadata and results.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary containing experiment data
    """
    experiment_data = {}

    # Load metadata
    metadata_path = experiment_dir / "experiment_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            experiment_data["metadata"] = json.load(f)

    # Load config
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            experiment_data["config"] = yaml.safe_load(f)

    # Load dataset info
    dataset_info_path = experiment_dir / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            experiment_data["dataset_info"] = json.load(f)

    # Load training history
    history_path = experiment_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            experiment_data["training_history"] = json.load(f)

    # Load model metrics
    metrics_path = experiment_dir / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            experiment_data["model_metrics"] = json.load(f)

    return experiment_data
