# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MPA library."""

import csv
import json

import pytest

from tesseract_core.runtime import mpa
from tesseract_core.runtime.config import update_config
from tesseract_core.runtime.mpa import (
    log_artifact,
    log_metric,
    log_parameter,
    start_run,
)


def test_start_run_context_manager():
    """Test that start_run works as a context manager."""
    with start_run():
        log_parameter("test_param", "value")
        log_metric("test_metric", 0.5)


def test_no_active_run_error():
    """Test that logging functions raise error when no run is active."""
    with pytest.raises(RuntimeError, match="No active MPA run"):
        log_parameter("test", "value")

    with pytest.raises(RuntimeError, match="No active MPA run"):
        log_metric("test", 0.5)

    with pytest.raises(RuntimeError, match="No active MPA run"):
        log_artifact("test.txt")


def test_nested_runs():
    """Test that nested runs work correctly."""
    with start_run():
        log_parameter("outer", "value1")

        with start_run():
            log_parameter("inner", "value2")
            log_metric("inner_metric", 1.0)

        # Should still work in outer context
        log_parameter("outer2", "value3")


def test_file_backend_default():
    """Test that FileBackend is used by default."""
    backend = mpa._create_backend(None)
    assert isinstance(backend, mpa.FileBackend)


def test_file_backend_empty_mlflow_uri():
    """Test that FileBackend is used when mlflow_tracking_uri is empty."""
    update_config(mlflow_tracking_uri="")
    backend = mpa._create_backend(None)
    assert isinstance(backend, mpa.FileBackend)


def test_uses_custom_base_directory(tmpdir):
    outdir = tmpdir / "mpa_test"
    backend = mpa.FileBackend(base_dir=str(outdir))
    assert backend.log_dir == outdir / "logs"


def test_log_parameter_content():
    """Test parameter logging creates correct JSON content."""
    backend = mpa.FileBackend()
    backend.log_parameter("model_name", "test_model")
    backend.log_parameter("epochs", 10)
    backend.log_parameter("learning_rate", 0.001)

    # Verify JSON file content
    assert backend.params_file.exists()
    with open(backend.params_file) as f:
        params = json.load(f)

    assert params["model_name"] == "test_model"
    assert params["epochs"] == 10
    assert params["learning_rate"] == 0.001


def test_log_metric_content():
    """Test metric logging creates correct CSV content."""
    backend = mpa.FileBackend()
    backend.log_metric("accuracy", 0.95)
    backend.log_metric("loss", 0.05, step=1)

    # Verify CSV file content
    assert backend.metrics_file.exists()
    with open(backend.metrics_file) as f:
        reader = csv.DictReader(f)
        metrics = list(reader)

    assert len(metrics) == 2

    # First metric (auto-generated step)
    assert metrics[0]["key"] == "accuracy"
    assert float(metrics[0]["value"]) == 0.95
    assert int(metrics[0]["step"]) == 0
    assert "timestamp" in metrics[0]

    # Second metric (explicit step)
    assert metrics[1]["key"] == "loss"
    assert float(metrics[1]["value"]) == 0.05
    assert int(metrics[1]["step"]) == 1


def test_log_artifact_content(tmpdir):
    """Test artifact logging copies files correctly."""
    backend = mpa.FileBackend()

    # Create a test file with specific content
    test_file = tmpdir / "model_summary.txt"
    test_content = "Model: CNN\nAccuracy: 95.2%\nLoss: 0.048"
    test_file.write_text(test_content, encoding="utf-8")

    # Log the artifact
    backend.log_artifact(str(test_file))

    # Verify file was copied with correct content
    copied_file = backend.artifacts_dir / "model_summary.txt"
    assert copied_file.exists()
    assert copied_file.read_text() == test_content


def test_log_artifact_missing_file():
    """Test that logging non-existent artifact raises error."""
    backend = mpa.FileBackend()

    with pytest.raises(FileNotFoundError, match="Artifact file not found"):
        backend.log_artifact("non_existent_file.txt")


def test_mlflow_backend_creation(tmpdir):
    """Test that MLflowBackend is created when mlflow_tracking_uri is set."""
    pytest.importorskip("mlflow")  # Skip if MLflow is not installed
    mlflow_dir = tmpdir / "mlflow_backend_test"
    update_config(mlflow_tracking_uri=f"file://{mlflow_dir}")
    backend = mpa._create_backend(None)
    assert isinstance(backend, mpa.MLflowBackend)


def test_mlflow_log_calls(tmpdir):
    """Test MLflow backend logging functions with temporary directory."""
    pytest.importorskip("mlflow")  # Skip if MLflow is not installed
    mlflow_dir = tmpdir / "mlflow_logging_test"
    update_config(mlflow_tracking_uri=f"file://{mlflow_dir}")

    with start_run():
        log_parameter("model_type", "neural_network")
        log_parameter("epochs", 100)

        log_metric("accuracy", 0.85)
        log_metric("loss", 0.25, step=1)

        artifact_file = tmpdir / "model_config.json"
        artifact_file.write_text("Test content", encoding="utf-8")
        log_artifact(str(artifact_file))

    # Verify MLflow directory structure was created
    assert mlflow_dir.exists()
    # MLflow creates experiment directories, so we should see some structure
    mlflow_contents = list(mlflow_dir.listdir())
    assert len(mlflow_contents) > 0
