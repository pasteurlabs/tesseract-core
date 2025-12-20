# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MPA library."""

import csv
import json
import os
import sqlite3

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
    mlflow_db_file = tmpdir / "mlflow.db"
    update_config(mlflow_tracking_uri=f"sqlite:///{mlflow_db_file}")
    backend = mpa._create_backend(None)
    assert isinstance(backend, mpa.MLflowBackend)


def test_mlflow_log_calls(tmpdir):
    """Test MLflow backend logging functions with temporary directory."""
    pytest.importorskip("mlflow")  # Skip if MLflow is not installed
    mlflow_db_file = tmpdir / "mlflow.db"
    update_config(mlflow_tracking_uri=f"sqlite:///{mlflow_db_file}")

    with start_run():
        log_parameter("model_type", "neural_network")
        log_parameter("epochs", 100)

        log_metric("accuracy", 0.85)
        log_metric("loss", 0.25, step=1)

        artifact_file = tmpdir / "model_config.json"
        artifact_file.write_text("Test content", encoding="utf-8")
        log_artifact(str(artifact_file))

    # Verify MLflow database file was created
    assert mlflow_db_file.exists()

    # Query the database to verify content was logged
    with sqlite3.connect(str(mlflow_db_file)) as conn:
        cursor = conn.cursor()

        # Check parameters were logged
        cursor.execute("SELECT key, value FROM params")
        params = dict(cursor.fetchall())
        assert params["model_type"] == "neural_network"
        assert params["epochs"] == "100"

        # Check metrics were logged
        cursor.execute("SELECT key, value, step FROM metrics ORDER BY step")
        metrics = cursor.fetchall()
        assert len(metrics) == 2
        assert metrics[0] == ("accuracy", 0.85, 0)  # step defaults to 0
        assert metrics[1] == ("loss", 0.25, 1)

        # Check artifacts were logged (MLflow stores artifact info in runs table)
        cursor.execute("SELECT artifact_uri FROM runs")
        artifact_uris = [row[0] for row in cursor.fetchall()]
        assert len(artifact_uris) > 0  # At least one run with artifacts

        # Verify the artifact file was actually copied to the artifact location
        artifact_found = False
        for artifact_uri in artifact_uris:
            if artifact_uri and os.path.exists(artifact_uri):
                try:
                    artifact_files = os.listdir(artifact_uri)
                    if "model_config.json" in artifact_files:
                        artifact_found = True
                        break
                except OSError:
                    continue

        assert artifact_found


def test_build_tracking_uri_with_credentials():
    pytest.importorskip("mlflow")
    update_config(
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_tracking_username="testuser",
        mlflow_tracking_password="testpass",
    )
    tracking_uri = mpa.MLflowBackend._build_tracking_uri()
    assert tracking_uri == "http://testuser:testpass@localhost:5000"


def test_build_tracking_uri_without_credentials():
    pytest.importorskip("mlflow")
    update_config(
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_tracking_username="",
        mlflow_tracking_password="",
    )
    tracking_uri = mpa.MLflowBackend._build_tracking_uri()
    assert tracking_uri == "http://localhost:5000"


def test_build_tracking_uri_url_encoded_credentials():
    pytest.importorskip("mlflow")
    update_config(
        mlflow_tracking_uri="https://mlflow.example.com",
        mlflow_tracking_username="user@example.com",
        mlflow_tracking_password="p@ss:w0rd!",
    )
    tracking_uri = mpa.MLflowBackend._build_tracking_uri()
    assert (
        tracking_uri == "https://user%40example.com:p%40ss%3Aw0rd%21@mlflow.example.com"
    )


def test_build_tracking_uri_with_path_and_query():
    pytest.importorskip("mlflow")
    update_config(
        mlflow_tracking_uri="http://localhost:5000/api/mlflow?param=value",
        mlflow_tracking_username="testuser",
        mlflow_tracking_password="testpass",
    )
    tracking_uri = mpa.MLflowBackend._build_tracking_uri()
    assert (
        tracking_uri == "http://testuser:testpass@localhost:5000/api/mlflow?param=value"
    )


def test_build_tracking_uri_username_without_password():
    pytest.importorskip("mlflow")
    update_config(
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_tracking_username="testuser",
        mlflow_tracking_password="",
    )
    with pytest.raises(
        RuntimeError,
        match="If one of TESSERACT_MLFLOW_TRACKING_USERNAME and TESSERACT_MLFLOW_TRACKING_PASSWORD is defined",
    ):
        mpa.MLflowBackend._build_tracking_uri()


def test_build_tracking_uri_password_without_username():
    pytest.importorskip("mlflow")
    update_config(
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_tracking_username="",
        mlflow_tracking_password="testpass",
    )
    with pytest.raises(
        RuntimeError,
        match="If one of TESSERACT_MLFLOW_TRACKING_USERNAME and TESSERACT_MLFLOW_TRACKING_PASSWORD is defined",
    ):
        mpa.MLflowBackend._build_tracking_uri()


def test_build_tracking_uri_sqlite_ignores_credentials():
    pytest.importorskip("mlflow")
    update_config(
        mlflow_tracking_uri="sqlite:///mlflow.db",
        mlflow_tracking_username="testuser",
        mlflow_tracking_password="testpass",
    )
    tracking_uri = mpa.MLflowBackend._build_tracking_uri()
    assert "testuser" not in tracking_uri
    assert "testpass" not in tracking_uri
    assert tracking_uri.startswith("sqlite:///")


def test_parse_run_extra_args_basic_tags():
    """Test parsing dict with basic tags."""
    pytest.importorskip("mlflow")
    args_str = '{"tags": {"env": "prod", "team": "ml"}}'
    args = mpa.MLflowBackend._parse_run_extra_args(args_str)
    assert args == {"tags": {"env": "prod", "team": "ml"}}


def test_parse_run_extra_args_multiple_params():
    """Test parsing dict with multiple parameters."""
    pytest.importorskip("mlflow")
    args_str = (
        '{"tags": {"foo": "bar"}, "run_name": "myrun", "description": "test run"}'
    )
    args = mpa.MLflowBackend._parse_run_extra_args(args_str)
    assert args == {
        "tags": {"foo": "bar"},
        "run_name": "myrun",
        "description": "test run",
    }


def test_parse_run_extra_args_empty_string():
    """Test parsing empty string returns empty dict."""
    pytest.importorskip("mlflow")
    args = mpa.MLflowBackend._parse_run_extra_args("")
    assert args == {}


def test_parse_run_extra_args_whitespace_only():
    """Test parsing whitespace-only string returns empty dict."""
    pytest.importorskip("mlflow")
    args = mpa.MLflowBackend._parse_run_extra_args("   ")
    assert args == {}


def test_parse_run_extra_args_invalid_syntax():
    """Test parsing invalid syntax raises ValueError."""
    pytest.importorskip("mlflow")
    with pytest.raises(
        ValueError,
        match="TESSERACT_MLFLOW_RUN_EXTRA_ARGS must be a valid Python dict string",
    ):
        mpa.MLflowBackend._parse_run_extra_args("{invalid syntax")


def test_parse_run_extra_args_non_dict_type():
    """Test parsing non-dict type raises ValueError."""
    pytest.importorskip("mlflow")
    with pytest.raises(
        ValueError, match="TESSERACT_MLFLOW_RUN_EXTRA_ARGS must evaluate to a dict"
    ):
        mpa.MLflowBackend._parse_run_extra_args('["not", "a", "dict"]')


def test_parse_run_extra_args_nested_structures():
    """Test parsing nested dict structures."""
    pytest.importorskip("mlflow")
    args_str = '{"tags": {"env": "prod", "meta": {"team": "ml", "version": "1.0"}}}'
    args = mpa.MLflowBackend._parse_run_extra_args(args_str)
    assert args == {"tags": {"env": "prod", "meta": {"team": "ml", "version": "1.0"}}}
