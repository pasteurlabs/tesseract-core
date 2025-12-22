# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for MLflow HTTP client with real MLflow server."""

import time
from pathlib import Path

import mlflow
import pytest

from tesseract_core.runtime.mlflow_client import MLflowHTTPClient


def test_create_and_end_run(mlflow_server):
    """Test creating and ending a run."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        run_id = client.create_run(
            run_name="test_create_run", tags={"test": "true", "purpose": "endtoend"}
        )

        assert run_id is not None
        assert isinstance(run_id, str)
        assert client.run_id == run_id

        # Verify run was created using official MLflow client
        run = mlflow.get_run(run_id)
        assert run.info.run_id == run_id
        assert run.info.status == "RUNNING"
        assert run.data.tags["test"] == "true"
        assert run.data.tags["purpose"] == "endtoend"

        # End the run
        client.end_run(status="FINISHED")
        assert client.run_id is None

        # Verify run was ended using official MLflow client
        run = mlflow.get_run(run_id)
        assert run.info.status == "FINISHED"


def test_log_parameters(mlflow_server):
    """Test logging parameters."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        run_id = client.create_run(run_name="test_log_params")

        # Log various parameter types
        client.log_param("learning_rate", 0.001)
        client.log_param("optimizer", "adam")
        client.log_param("batch_size", 32)
        client.log_param("epochs", 100)

        # Verify params were logged using official MLflow client
        run = mlflow.get_run(run_id)
        params = run.data.params

        assert params["learning_rate"] == "0.001"
        assert params["optimizer"] == "adam"
        assert params["batch_size"] == "32"
        assert params["epochs"] == "100"

        client.end_run()


def test_log_metrics(mlflow_server):
    """Test logging metrics with steps."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        run_id = client.create_run(run_name="test_log_metrics")

        # Log metrics with steps
        for step in range(5):
            client.log_metric("accuracy", 0.5 + step * 0.1, step=step)
            client.log_metric("loss", 1.0 - step * 0.15, step=step)

        # Verify metrics were logged using official MLflow client
        # Get metric history to verify all steps were logged
        mlflow_client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        accuracy_history = mlflow_client.get_metric_history(run_id, "accuracy")
        loss_history = mlflow_client.get_metric_history(run_id, "loss")

        assert len(accuracy_history) == 5
        assert len(loss_history) == 5

        # Verify values at specific steps
        for i, metric in enumerate(accuracy_history):
            assert metric.step == i
            assert abs(metric.value - (0.5 + i * 0.1)) < 0.001

        for i, metric in enumerate(loss_history):
            assert metric.step == i
            assert abs(metric.value - (1.0 - i * 0.15)) < 0.001

        client.end_run()


def test_log_batch(mlflow_server):
    """Test batch logging for efficiency."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        run_id = client.create_run(run_name="test_batch_logging")

        # Log batch of items
        timestamp = int(time.time() * 1000)
        client.log_batch(
            metrics=[
                {"key": "train_acc", "value": 0.95, "timestamp": timestamp, "step": 0},
                {"key": "val_acc", "value": 0.92, "timestamp": timestamp, "step": 0},
            ],
            params=[
                {"key": "model_type", "value": "cnn"},
                {"key": "layers", "value": "3"},
            ],
            tags=[
                {"key": "environment", "value": "test"},
            ],
        )

        # Verify all items were logged using official MLflow client
        run = mlflow.get_run(run_id)

        assert run.data.params["model_type"] == "cnn"
        assert run.data.params["layers"] == "3"

        assert "train_acc" in run.data.metrics
        assert "val_acc" in run.data.metrics
        assert abs(run.data.metrics["train_acc"] - 0.95) < 0.001
        assert abs(run.data.metrics["val_acc"] - 0.92) < 0.001

        assert run.data.tags["environment"] == "test"

        client.end_run()


def test_context_manager_normal_exit(mlflow_server):
    """Test context manager ends run with FINISHED on normal exit."""
    mlflow.set_tracking_uri(mlflow_server)

    with MLflowHTTPClient(tracking_uri=mlflow_server) as client:
        run_id = client.create_run(run_name="test_context_normal")
        client.log_param("test", "value")

        # Run should be active
        assert client.run_id == run_id

    # After exiting context, verify run was marked as FINISHED using official MLflow client
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["test"] == "value"


def test_context_manager_exception_exit(mlflow_server):
    """Test context manager ends run with FAILED on exception."""
    mlflow.set_tracking_uri(mlflow_server)
    run_id = None

    with pytest.raises(ValueError):
        with MLflowHTTPClient(tracking_uri=mlflow_server) as client:
            run_id = client.create_run(run_name="test_context_exception")
            client.log_param("test", "value")
            raise ValueError("Test exception")

    # Verify run was marked as FAILED using official MLflow client
    assert run_id is not None
    run = mlflow.get_run(run_id)
    assert run.info.status == "FAILED"
    assert run.data.params["test"] == "value"


def test_timestamps_in_milliseconds(mlflow_server):
    """Test that timestamps are correctly handled in milliseconds."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        run_id = client.create_run(run_name="test_timestamps")

        # Log metric with explicit timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        client.log_metric("test_metric", 0.5, timestamp=timestamp_ms)

        # Verify the timestamp was preserved using official MLflow client
        mlflow_client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        metric_history = mlflow_client.get_metric_history(run_id, "test_metric")

        assert len(metric_history) == 1
        # MLflow should have stored the timestamp in milliseconds
        assert metric_history[0].timestamp == timestamp_ms
        assert abs(metric_history[0].value - 0.5) < 0.001

        client.end_run()


def test_experiment_id_as_string(mlflow_server):
    """Test that experiment_id is correctly handled as string."""
    # MLflow quirk: experiment_id must be a string, not an int
    mlflow.set_tracking_uri(mlflow_server)
    client = MLflowHTTPClient(
        tracking_uri=mlflow_server,
        experiment_id="0",  # Default experiment
    )

    with client:
        run_id = client.create_run(run_name="test_experiment_id")

        # Verify run was created in the correct experiment using official MLflow client
        run = mlflow.get_run(run_id)
        assert run.info.experiment_id == "0"

        client.end_run()


def test_tag_conversion_to_list_format(mlflow_server):
    """Test that tags dict is converted to MLflow's list format."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        # Pass tags as dict
        run_id = client.create_run(
            run_name="test_tag_conversion",
            tags={
                "team": "ml-platform",
                "project": "tesseract",
                "version": "1.0",
            },
        )

        # Verify tags were stored correctly using official MLflow client
        run = mlflow.get_run(run_id)
        tags = run.data.tags

        assert tags["team"] == "ml-platform"
        assert tags["project"] == "tesseract"
        assert tags["version"] == "1.0"

        client.end_run()


def test_multiple_runs_sequential(mlflow_server):
    """Test creating multiple runs sequentially."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        # First run
        run_id_1 = client.create_run(run_name="test_multi_run_1")
        client.log_param("run_number", "1")
        client.end_run()

        # Second run
        run_id_2 = client.create_run(run_name="test_multi_run_2")
        client.log_param("run_number", "2")
        client.end_run()

        # Verify both runs exist using official MLflow client
        assert run_id_1 != run_id_2

        run_1 = mlflow.get_run(run_id_1)
        run_2 = mlflow.get_run(run_id_2)

        assert run_1.data.params["run_number"] == "1"
        assert run_2.data.params["run_number"] == "2"
        assert run_1.info.status == "FINISHED"
        assert run_2.info.status == "FINISHED"


def test_error_handling_no_active_run(mlflow_server):
    """Test that operations without active run raise clear errors."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)

    with client:
        # Try to log without creating a run
        with pytest.raises(RuntimeError, match="No active run"):
            client.log_param("key", "value")

        with pytest.raises(RuntimeError, match="No active run"):
            client.log_metric("metric", 0.5)

        with pytest.raises(RuntimeError, match="No active run"):
            client.get_run()

        with pytest.raises(RuntimeError, match="No active run"):
            client.end_run()


def test_update_run_status(mlflow_server):
    """Test updating run status directly."""
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        run_id = client.create_run(run_name="test_update_status")

        # Update to FAILED status
        client.update_run(run_id, status="FAILED")

        # Verify status was updated using official MLflow client
        run = mlflow.get_run(run_id)
        assert run.info.status == "FAILED"


@pytest.mark.parametrize(
    "suffix,content,mode",
    [
        (".txt", "Model configuration\nLayers: 3\nActivation: ReLU\n", "w"),
        (".json", '{"model": "cnn", "layers": 3, "activation": "ReLU"}', "w"),
        (".yaml", "model:\n  type: cnn\n  layers: 3\n  activation: ReLU\n", "w"),
        (".csv", "epoch,loss,accuracy\n1,0.5,0.8\n2,0.3,0.9\n", "w"),
        (".html", "<html><body><h1>Model Report</h1></body></html>", "w"),
        (
            ".png",
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89",
            "wb",
        ),
    ],
)
def test_log_artifact(tmp_path, mlflow_server, suffix, content, mode):
    """Test artifact logging implementation correctness.

    This test validates that our HTTP client correctly implements the MLflow
    artifact upload protocol, matching http_artifact_repo.py:
    - Uses PUT request with file data
    - Includes Content-Type header based on MIME type
    - Constructs URL as: /api/2.0/mlflow-artifacts/artifacts/{run_id}/{path}

    Tests multiple MIME types to ensure proper Content-Type header handling.
    Verifies artifacts are correctly logged using the official MLflow SDK.
    """
    client = MLflowHTTPClient(tracking_uri=mlflow_server)
    mlflow.set_tracking_uri(mlflow_server)

    with client:
        run_id = client.create_run(run_name=f"test_artifact_upload_{suffix}")

        # Create test file with a predictable name
        file_path = tmp_path / f"file{suffix}"
        with open(file_path, mode=mode) as f:
            f.write(content)

        artifact_filename = Path(file_path).name

        # Log artifact using our HTTP client
        client.log_artifact(file_path)

        # Verify artifact was logged using official MLflow SDK
        mlflow_client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        artifacts = mlflow_client.list_artifacts(run_id=run_id)

        # Check that the artifact exists
        assert len(artifacts) > 0, f"No artifacts found for run {run_id}"
        artifact_names = [artifact.path for artifact in artifacts]
        assert artifact_filename in artifact_names, (
            f"Artifact {artifact_filename} not found in {artifact_names}"
        )

        # Download and verify the artifact content matches
        download_path = mlflow_client.download_artifacts(run_id, artifact_filename)

        # Verify content
        if mode == "wb":
            # Binary file
            with open(download_path, "rb") as f:
                downloaded_content = f.read()
            assert downloaded_content == content, "Binary artifact content mismatch"
        else:
            # Text file
            with open(download_path) as f:
                downloaded_content = f.read()
            assert downloaded_content == content, "Text artifact content mismatch"

        client.end_run()
