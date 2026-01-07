# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MPA module."""

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import ValidationError

from tesseract_core.runtime import mpa
from tesseract_core.runtime.config import update_config
from tesseract_core.runtime.mpa import (
    log_artifact,
    log_metric,
    log_parameter,
    start_run,
)


class Always400Handler(BaseHTTPRequestHandler):
    """HTTP request handler that always returns 400."""

    def do_GET(self):
        """Handle GET requests with 400."""
        self.send_response(400)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bad Request")

    def do_POST(self):
        """Handle POST requests with 400."""
        self.send_response(400)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bad Request")

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


@pytest.fixture(scope="module")
def dummy_mlflow_server():
    """Start a dummy HTTP server that always returns 400."""
    server = HTTPServer(("localhost", 0), Always400Handler)
    port = server.server_address[1]

    # Start server in a background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield f"http://localhost:{port}"
    finally:
        # Shutdown server
        server.shutdown()


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


def test_mlflow_backend_creation_fails_with_unreachable_server(dummy_mlflow_server):
    """Test that MLflowBackend creation fails when server returns 400."""
    update_config(mlflow_tracking_uri=dummy_mlflow_server)
    with pytest.raises(
        RuntimeError, match="Failed to connect to MLflow tracking server"
    ):
        mpa._create_backend(None)


def test_mlflow_non_http_scheme_raises_error(mocker):
    """Test that non-HTTP/HTTPS schemes raise an error."""
    # Mock to avoid actually trying to reach the server
    mocker.patch.object(mpa.MLflowBackend, "_ensure_mlflow_reachable")

    update_config(mlflow_tracking_uri="sqlite:///mlflow.db")
    with pytest.raises(
        ValueError, match="MLflow logging only supports accessing MLflow via HTTP/HTTPS"
    ):
        mpa.MLflowBackend()


def test_mlflow_run_extra_args(mocker, dummy_mlflow_server):
    """Test passing a dict with basic tags."""
    pytest.importorskip("mlflow")

    kwargs = {"tags": {"env": "prod", "team": "ml"}}
    kwargs_str = repr(kwargs)

    # Mock the mlflow module to avoid actual MLflow calls
    mocked_mlflow = mocker.patch("tesseract_core.runtime.mpa.mlflow")

    # Mock the reachability check since dummy server returns 400
    mocker.patch.object(mpa.MLflowBackend, "_ensure_mlflow_reachable")

    update_config(
        mlflow_tracking_uri=dummy_mlflow_server, mlflow_run_extra_args=kwargs_str
    )

    backend = mpa.MLflowBackend()

    # Make sure kwargs are forwarded correctly to mlflow.start_run
    backend.start_run()
    mocked_mlflow.start_run.assert_called_with(**kwargs)


def test_mlflow_run_extra_args_parsing():
    # This is actually a test for config.py but we add it here for now

    with pytest.raises(ValidationError):
        # Not a valid Python object
        update_config(mlflow_run_extra_args="{'unbalanced dict': True")

    with pytest.raises(ValidationError):
        # Not a dict
        update_config(mlflow_run_extra_args="['this is a list']")

    with pytest.raises(ValidationError):
        # Not str keys
        update_config(mlflow_run_extra_args="{0: 'hey there'}")

    # All good
    update_config(mlflow_run_extra_args="{'hey there': 'general kenobi'}")

    # Passing dicts directly is fine, too
    update_config(mlflow_run_extra_args={"hey there": "general kenobi"})
