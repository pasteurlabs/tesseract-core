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
from tesseract_core.runtime.logs import LogPipe


class BaseBackend(ABC):
    """Base class for MPA backends."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        if base_dir is None:
            base_dir = get_config().output_path
        self.log_dir = Path(base_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

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

    def __init__(self, base_dir: Optional[str] = None) -> None:
        super().__init__(base_dir)
        # Initialize log files
        self.params_file = self.log_dir / "parameters.json"
        self.metrics_file = self.log_dir / "metrics.csv"
        self.artifacts_dir = self.log_dir / "artifacts"
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

    def __init__(self, base_dir: Optional[str] = None) -> None:
        super().__init__(base_dir)
        os.environ["GIT_PYTHON_REFRESH"] = (
            "quiet"  # Suppress potential MLflow git warnings
        )

        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow is required for MLflowBackend but is not installed"
            ) from exc

        self._ensure_mlflow_reachable()
        self.mlflow = mlflow

        config = get_config()
        tracking_uri = config.mlflow_tracking_uri

        if not tracking_uri.startswith(("http://", "https://")):
            # If it's a db file URI, convert to local path
            tracking_uri = tracking_uri.replace("sqlite:///", "")

            # Relative paths are resolved against the base output path
            if not Path(tracking_uri).is_absolute():
                tracking_uri = (Path(get_config().output_path) / tracking_uri).resolve()

            tracking_uri = f"sqlite:///{tracking_uri}"

        mlflow.set_tracking_uri(tracking_uri)

    def _ensure_mlflow_reachable(self) -> None:
        """Check if the MLflow tracking server is reachable."""
        config = get_config()
        mlflow_tracking_uri = config.mlflow_tracking_uri
        if mlflow_tracking_uri.startswith(("http://", "https://")):
            try:
                response = requests.get(mlflow_tracking_uri, timeout=5)
                response.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(
                    f"Failed to connect to MLflow tracking server at {mlflow_tracking_uri}. "
                    "Please make sure an MLflow server is running and TESSERACT_MLFLOW_TRACKING_URI is set correctly, "
                    "or switch to file-based logging by setting TESSERACT_MLFLOW_TRACKING_URI to an empty string."
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


def _create_backend(base_dir: Optional[str]) -> BaseBackend:
    """Create the appropriate backend based on environment."""
    config = get_config()
    if config.mlflow_tracking_uri:
        return MLflowBackend(base_dir)
    else:
        return FileBackend(base_dir)


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
def redirect_stdio(logfile: Union[str, Path]) -> Generator[None, None, None]:
    """Context manager for redirecting stdout and stderr to a custom pipe.

    Writes messages to both the original stderr and the given logfile.
    """
    from tesseract_core.runtime.core import redirect_fd

    try:
        # Check if a file descriptor is available
        sys.stdout.fileno()
        sys.stderr.fileno()
    except UnsupportedOperation:
        # Don't redirect if stdout/stderr are not file descriptors
        # (This likely means that streams are already redirected)
        yield
        return

    with ExitStack() as stack:
        f = stack.enter_context(open(logfile, "w"))

        # Duplicate the original stderr file descriptor before any redirection
        orig_stderr_fd = os.dup(sys.stderr.fileno())
        orig_stderr_file = os.fdopen(orig_stderr_fd, "w")
        stack.callback(orig_stderr_file.close)

        # Use `print` instead of `.write` so we get appropriate newlines and flush behavior
        write_to_stderr = lambda msg: print(msg, file=orig_stderr_file, flush=True)
        write_to_file = lambda msg: print(msg, file=f, flush=True)
        pipe_fd = stack.enter_context(LogPipe(write_to_stderr, write_to_file))

        # Redirect file descriptors at OS level
        stack.enter_context(redirect_fd(sys.stdout, pipe_fd))
        stack.enter_context(redirect_fd(sys.stderr, pipe_fd))
        yield


@contextmanager
def start_run(base_dir: Optional[str] = None) -> Generator[None, None, None]:
    """Context manager for starting and ending a run."""
    backend = _create_backend(base_dir)
    token = _current_backend.set(backend)
    backend.start_run()

    logfile = backend.log_dir / "tesseract.log"

    try:
        with redirect_stdio(logfile):
            yield
    finally:
        backend.end_run()
        _current_backend.reset(token)
