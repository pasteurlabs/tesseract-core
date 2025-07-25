# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics, Parameters, and Artifacts (MPA) library for Tesseract Core."""

import csv
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import datetime
from io import UnsupportedOperation
from pathlib import Path
from typing import Any, Optional, Union

import requests

from tesseract_core.runtime.config import get_config


class BaseBackend(ABC):
    """Base class for MPA backends."""

    def __init__(self) -> None:
        self.log_dir = os.getenv("LOG_DIR")
        if not self.log_dir:
            self.log_dir = Path(get_config().output_path) / "logs"
        else:
            self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create a unique run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self.log_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)

    @abstractmethod
    def log_parameter(self, key: str, value: Any) -> None:
        """Log a parameter."""
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        """Log an artifact."""
        pass

    @abstractmethod
    def start_run(self) -> None:
        """Start a new run."""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current run."""
        pass


class FileBackend(BaseBackend):
    """MPA backend that writes to local files."""

    def __init__(self) -> None:
        super().__init__()
        # Initialize log files
        self.params_file = self.run_dir / "parameters.json"
        self.metrics_file = self.run_dir / "metrics.csv"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        # Initialize parameters dict and metrics list
        self.parameters = {}
        self.metrics = []

        # Initialize CSV file with headers
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "key", "value", "step"])

    def log_parameter(self, key: str, value: Any) -> None:
        """Log a parameter to JSON file."""
        self.parameters[key] = value
        with open(self.params_file, "w") as f:
            json.dump(self.parameters, f, indent=2, default=str)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to CSV file."""
        timestamp = datetime.now().isoformat()
        step_value = (
            step
            if step is not None
            else len([m for m in self.metrics if m["key"] == key])
        )

        metric_entry = {
            "timestamp": timestamp,
            "key": key,
            "value": value,
            "step": step_value,
        }
        self.metrics.append(metric_entry)

        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, key, value, step_value])

    def log_artifact(self, local_path: str) -> None:
        """Copy artifact to the artifacts directory."""
        source_path = Path(local_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")

        dest_path = self.artifacts_dir / source_path.name
        shutil.copy2(source_path, dest_path)

    def start_run(self) -> None:
        """Start a new run. File backend doesn't need special start logic."""
        pass

    def end_run(self) -> None:
        """End the current run. File backend doesn't need special end logic."""
        pass


class MLflowBackend(BaseBackend):
    """MPA backend that writes to an MLflow tracking server."""

    def __init__(self) -> None:
        super().__init__()
        try:
            os.environ["GIT_PYTHON_REFRESH"] = (
                "quiet"  # Suppress potential MLflow git warnings
            )
            self._ensure_mlflow_reachable()

            import mlflow

            self.mlflow = mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow is required for MLflowBackend but is not installed"
            ) from exc

    def _ensure_mlflow_reachable(self) -> None:
        try:
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if mlflow_tracking_uri and (
                mlflow_tracking_uri.startswith("http://")
                or mlflow_tracking_uri.startswith("https://")
            ):
                response = requests.get(mlflow_tracking_uri, timeout=5)
                response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to connect to MLflow tracking server at {mlflow_tracking_uri}. "
                "Please make sure an MLflow server is running and MLFLOW_TRACKING_URI is set correctly. "
                "Alternatively, switch to local file-based by setting MLFLOW_TRACKING_URI to an empty string."
            ) from e

    def log_parameter(self, key: str, value: Any) -> None:
        """Log a parameter to MLflow."""
        self.mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to MLflow."""
        self.mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Log an artifact to MLflow."""
        self.mlflow.log_artifact(local_path)

    def start_run(self) -> None:
        """Start a new MLflow run."""
        self.mlflow.start_run()

    def end_run(self) -> None:
        """End the current MLflow run."""
        self.mlflow.end_run()


def _create_backend() -> BaseBackend:
    """Create the appropriate backend based on environment."""
    if os.getenv("MLFLOW_TRACKING_URI"):
        return MLflowBackend()
    else:
        return FileBackend()


# Context variable for the current backend instance
_current_backend: ContextVar[BaseBackend] = ContextVar("current_backend")


def _get_current_backend() -> BaseBackend:
    """Get the current backend instance from context variable."""
    try:
        return _current_backend.get()
    except LookupError as exc:
        raise RuntimeError(
            "No active MPA run. Use 'with mpa.start_run():' to start a run."
        ) from exc


# Public API functions that work with the current context
def log_parameter(key: str, value: Any) -> None:
    """Log a parameter to the current run context."""
    _get_current_backend().log_parameter(key, value)


def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    """Log a metric to the current run context."""
    _get_current_backend().log_metric(key, value, step)


def log_artifact(local_path: str) -> None:
    """Log an artifact to the current run context."""
    _get_current_backend().log_artifact(local_path)


@contextmanager
def start_run() -> Generator[None, None, None]:
    """Context manager for starting and ending a run."""
    backend = _create_backend()
    token = _current_backend.set(backend)
    backend.start_run()

    logfile = backend.run_dir / "tesseract.log"

    try:
        with stdio_to_logfile(logfile):
            yield
    finally:
        backend.end_run()
        _current_backend.reset(token)
