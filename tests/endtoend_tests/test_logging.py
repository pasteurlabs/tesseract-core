# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Tesseract logging, MPA, MLflow integration, and log streaming."""

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from textwrap import dedent

import mlflow
import pytest
import requests

from tesseract_core.sdk.cli import app


@pytest.fixture(scope="module")
def logging_test_image(
    cli_runner, dummy_tesseract_location, tmpdir_factory, docker_cleanup_module
):
    tesseract_api = dedent(
        """
    from pydantic import BaseModel

    print("Hello from tesseract_api.py!")

    class InputSchema(BaseModel):
        message: str = "Hello, Tesseractor!"

    class OutputSchema(BaseModel):
        out: str

    def apply(inputs: InputSchema) -> OutputSchema:
        print("Hello from apply!")
        return OutputSchema(out=f"Received message: {inputs.message}")
    """
    )

    workdir = tmpdir_factory.mktemp("mpa_test_image")

    # Write the API file
    with open(workdir / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)

    shutil.copy(
        dummy_tesseract_location / "tesseract_config.yaml",
        workdir / "tesseract_config.yaml",
    )

    # Build the Tesseract
    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(workdir), "--tag", "logging_test_image"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup_module["images"].append(img_tag)
    return img_tag


@pytest.fixture(scope="module")
def logging_with_mlflow_test_image(
    cli_runner, tmpdir_factory, dummy_tesseract_location, docker_cleanup_module
):
    tesseract_api = dedent(
        """
    from pydantic import BaseModel
    import mlflow
    import sys

    class InputSchema(BaseModel):
        pass

    class OutputSchema(BaseModel):
        pass

    def apply(inputs: InputSchema) -> OutputSchema:
        sys.__stderr__.write("DUMMY_STDERR_OUTPUT\\n")
        mlflow.start_run()
        return OutputSchema()
    """
    )

    workdir = tmpdir_factory.mktemp("logging_with_mlflow_test_image")

    # Write the API file
    with open(workdir / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)
    # Add mlflow dependency
    with open(workdir / "tesseract_requirements.txt", "w") as f:
        f.write("mlflow\n")

    shutil.copy(
        dummy_tesseract_location / "tesseract_config.yaml",
        workdir / "tesseract_config.yaml",
    )

    # Build the Tesseract
    result = cli_runner.invoke(
        app,
        [
            "--loglevel",
            "debug",
            "build",
            str(workdir),
            "--tag",
            "logging_with_mlflow_test_image",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup_module["images"].append(img_tag)
    return img_tag


@pytest.fixture(scope="module")
def mpa_test_image(
    cli_runner, dummy_tesseract_location, tmpdir_factory, docker_cleanup_module
):
    tesseract_api = dedent(
        """
    from pydantic import BaseModel
    from tesseract_core.runtime.experimental import log_artifact, log_metric, log_parameter

    class InputSchema(BaseModel):
        pass

    class OutputSchema(BaseModel):
        pass

    def apply(inputs: InputSchema) -> OutputSchema:
        steps = 5
        param_value = "test_param"
        # Log parameters
        log_parameter("test_parameter", param_value)
        log_parameter("steps_config", steps)

        # Log metrics over multiple steps
        for step in range(steps):
            log_metric("squared_step", step ** 2, step=step)

        # Create and log an artifact
        artifact_content = "Test artifact content"

        artifact_path = "/tmp/test_artifact.txt"
        with open(artifact_path, "w") as f:
            f.write(artifact_content)

        log_artifact(artifact_path)

        return OutputSchema()
        """
    )
    workdir = tmpdir_factory.mktemp("mpa_test_image")

    # Write the API file
    with open(workdir / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)
    # Add mlflow dependency
    with open(workdir / "tesseract_requirements.txt", "w") as f:
        f.write("mlflow\n")

    shutil.copy(
        dummy_tesseract_location / "tesseract_config.yaml",
        workdir / "tesseract_config.yaml",
    )

    # Build the Tesseract
    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(workdir), "--tag", "mpa_test_image"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup_module["images"].append(img_tag)
    return img_tag


def test_logging_tesseract_run(logging_test_image, tmpdir):
    # Run the Tesseract and capture logs
    # Use subprocess because pytest messes with stdout/stderr
    run_res = subprocess.run(
        [
            "tesseract",
            "run",
            logging_test_image,
            "apply",
            '{"inputs": {"message": "Test message"}}',
            "--output-path",
            tmpdir,
        ],
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr
    assert "Hello from tesseract_api.py!\nHello from apply!" == run_res.stderr.strip()

    results = json.loads(run_res.stdout.strip())
    assert results["out"] == "Received message: Test message"

    logdir = Path(tmpdir) / "logs"

    log_file = logdir / "tesseract.log"
    assert log_file.exists()

    with open(log_file) as f:
        log_content = f.read()
    assert "Hello from apply!" == log_content.strip()


def test_logging_tesseract_serve(
    logging_test_image, tmpdir, docker_cleanup, docker_client
):
    serve_res = subprocess.run(
        [
            "tesseract",
            "serve",
            logging_test_image,
            "--output-path",
            tmpdir,
        ],
        capture_output=True,
        text=True,
    )
    assert serve_res.returncode == 0, serve_res.stderr
    assert serve_res.stdout

    serve_meta = json.loads(serve_res.stdout)
    container_name = serve_meta["container_name"]
    docker_cleanup["containers"].append(container_name)
    container = docker_client.containers.get(container_name)

    run_id = str(uuid.uuid4())
    res = requests.post(
        f"http://{container.host_ip}:{container.host_port}/apply",
        params={"run_id": run_id},
        json={"inputs": {}},
    )
    assert res.status_code == 200, res.text

    log_file = Path(tmpdir) / f"run_{run_id}/logs/tesseract.log"
    assert log_file.exists()

    with open(log_file) as f:
        log_content = f.read()
    assert "Hello from apply!" == log_content.strip()


def test_logging_with_mlflow(logging_with_mlflow_test_image, tmpdir):
    # This test covers a bug where mlflow would mess with stderr capturing
    # We ensure that stderr output from the Tesseract is captured exactly once
    # in stderr output and log file, even when mlflow is used.
    run_res = subprocess.run(
        [
            "tesseract",
            "run",
            logging_with_mlflow_test_image,
            "apply",
            '{"inputs": {}}',
            "--output-path",
            tmpdir,
        ],
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr
    assert run_res.stderr.count("DUMMY_STDERR_OUTPUT") == 1, run_res.stderr

    log_file = Path(tmpdir) / "logs" / "tesseract.log"
    assert log_file.exists()

    with open(log_file) as f:
        log_content = f.read()

    assert log_content.count("DUMMY_STDERR_OUTPUT") == 1, log_content


def test_mpa_file_backend(tmpdir, mpa_test_image):
    """Test the MPA (Metrics, Parameters, and Artifacts) submodule with file backend."""
    import csv

    outdir = Path(tmpdir)

    run_cmd = [
        "tesseract",
        "run",
        mpa_test_image,
        "apply",
        '{"inputs": {}}',
        "--output-path",
        outdir,
    ]

    run_res = subprocess.run(
        run_cmd,
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr

    log_dir = outdir / "logs"
    assert log_dir.exists()

    # Verify parameters file
    params_file = log_dir / "parameters.json"
    assert params_file.exists()
    with open(params_file) as f:
        params = json.load(f)
        assert params["test_parameter"] == "test_param"
        assert params["steps_config"] == 5

    # Verify metrics file
    metrics_file = log_dir / "metrics.csv"
    assert metrics_file.exists()

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        metrics = list(reader)

        # Should have 5 metrics: 5 squared_step (0, 1, 4, 9, 16)
        assert len(metrics) == 5

        # Check squared_step values
        squared_metrics = [m for m in metrics if m["key"] == "squared_step"]
        assert len(squared_metrics) == 5
        for i, metric in enumerate(squared_metrics):
            assert float(metric["value"]) == i**2
            assert int(metric["step"]) == i

    # Verify artifacts directory and artifact file
    artifacts_dir = log_dir / "artifacts"
    assert artifacts_dir.exists()

    artifact_file = artifacts_dir / "test_artifact.txt"
    assert artifact_file.exists()

    with open(artifact_file) as f:
        artifact_data = f.read()
        assert artifact_data == "Test artifact content"


def test_mpa_mlflow_backend(mlflow_server, mpa_test_image):
    """Test the MPA (Metrics, Parameters, and Artifacts) submodule with MLflow backend, using a local MLflow server."""
    # Hardcode some values specific to docker-compose config in extra/mlflow/mlflow-docker-compose.yaml

    # Inside containers, tracking URIs look like http://{service_name}:{internal_port}
    mlflow_server_local = "http://mlflow-server:5000"
    # Network name as specified in MLflow docker compose config
    network_name = "tesseract-mlflow-server"

    # Run the Tesseract, logging to running MLflow server
    run_cmd = [
        "tesseract",
        "run",
        "--network",
        network_name,
        "--env",
        f"TESSERACT_MLFLOW_TRACKING_URI={mlflow_server_local}",
        mpa_test_image,
        "apply",
        '{"inputs": {}}',
    ]
    run_res = subprocess.run(
        run_cmd,
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr

    # Use MLflow client to verify content was logged
    mlflow.set_tracking_uri(mlflow_server)

    # Get the most recent run (the one we just created)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Get the default experiment (experiment_id="0")
    experiment = client.get_experiment("0")
    assert experiment is not None, "Default experiment not found"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0, "No runs found in MLflow"

    # Get the most recent run
    print(runs)
    run = runs[0]
    run_id = run.info.run_id

    # Check parameters were logged
    params = run.data.params
    assert params["test_parameter"] == "test_param"
    assert params["steps_config"] == "5"  # MLflow stores params as strings

    # Check metrics were logged
    metrics_history = client.get_metric_history(run_id, "squared_step")
    assert len(metrics_history) == 5

    # Verify some of the squared_step values
    assert metrics_history[0].value == 0.0
    assert metrics_history[0].step == 0
    assert metrics_history[1].value == 1.0
    assert metrics_history[1].step == 1
    assert metrics_history[4].value == 16.0
    assert metrics_history[4].step == 4

    # Check artifacts were logged
    artifacts = client.list_artifacts(run_id)
    assert len(artifacts) > 0, "Expected at least one artifact to be logged"


# ============================================================================
# Log Streaming End-to-End Tests
# ============================================================================


LOG_STREAMING_TESSERACT_API = dedent(
    """
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel


class InputSchema(BaseModel):
    num_lines: int = 3
    signal_dir: str  # Directory where signal files are created


class OutputSchema(BaseModel):
    lines_printed: int


def apply(inputs: InputSchema) -> OutputSchema:
    signal_dir = Path(inputs.signal_dir)

    for i in range(inputs.num_lines):
        # Print log line
        print(f"Log line {i}", file=sys.stderr)
        sys.stderr.flush()

        # Wait for signal file before continuing (except for last line)
        if i < inputs.num_lines - 1:
            signal_file = signal_dir / f"continue_{i}.signal"
            timeout = 10  # seconds
            start = time.time()
            while not signal_file.exists():
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for {signal_file}")
                time.sleep(0.01)

    return OutputSchema(lines_printed=inputs.num_lines)
"""
)


# Simple tesseract API for testing stream_logs=False and stream_logs=print
SIMPLE_LOGGING_TESSERACT_API = dedent(
    """\
from pydantic import BaseModel


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    message: str


def apply(inputs: InputSchema) -> OutputSchema:
    print("Hello from apply!")
    return OutputSchema(message="done")
"""
)


# Tesseract API with delays between log lines for testing real-time streaming
TIMED_LOGGING_TESSERACT_API = dedent(
    """\
import time
from pydantic import BaseModel


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    message: str


def apply(inputs: InputSchema) -> OutputSchema:
    print("Line 1")
    time.sleep(0.5)
    print("Line 2")
    time.sleep(0.5)
    print("Line 3")
    return OutputSchema(message="done")
"""
)


@pytest.fixture(scope="module")
def log_streaming_test_image(
    cli_runner, dummy_tesseract_location, tmpdir_factory, docker_cleanup_module
):
    """Build a Tesseract that prints logs and waits for signal files."""
    workdir = tmpdir_factory.mktemp("log_streaming_test_image")

    # Write the API file
    with open(workdir / "tesseract_api.py", "w") as f:
        f.write(LOG_STREAMING_TESSERACT_API)

    shutil.copy(
        dummy_tesseract_location / "tesseract_config.yaml",
        workdir / "tesseract_config.yaml",
    )

    # Build the Tesseract
    result = cli_runner.invoke(
        app,
        [
            "--loglevel",
            "debug",
            "build",
            str(workdir),
            "--tag",
            "log_streaming_test_image",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup_module["images"].append(img_tag)
    return img_tag


@pytest.fixture(scope="module")
def log_streaming_tesseract_api_path(dummy_tesseract_location, tmpdir_factory):
    """Create a tesseract_api.py for testing log streaming with from_tesseract_api."""
    workdir = tmpdir_factory.mktemp("log_streaming_tesseract_api")
    api_path = Path(str(workdir)) / "tesseract_api.py"
    with open(api_path, "w") as f:
        f.write(LOG_STREAMING_TESSERACT_API)

    return api_path


@pytest.fixture(scope="module")
def simple_logging_tesseract_api_path(tmpdir_factory):
    """Create a simple tesseract_api.py for testing stream_logs options."""
    workdir = tmpdir_factory.mktemp("simple_logging_tesseract_api")
    api_path = Path(str(workdir)) / "tesseract_api.py"
    with open(api_path, "w") as f:
        f.write(SIMPLE_LOGGING_TESSERACT_API)

    return api_path


@pytest.fixture(scope="module")
def timed_logging_tesseract_api_path(tmpdir_factory):
    """Create a tesseract_api.py that prints with delays for timing tests."""
    workdir = tmpdir_factory.mktemp("timed_logging_tesseract_api")
    api_path = Path(str(workdir)) / "tesseract_api.py"
    with open(api_path, "w") as f:
        f.write(TIMED_LOGGING_TESSERACT_API)

    return api_path


def test_log_streaming_sdk_from_image(log_streaming_test_image, tmpdir):
    """Test log streaming via Python SDK using Tesseract.from_image().

    Uses file-based synchronization to prove logs are truly streaming:
    the Tesseract blocks after each log line until we create a signal file,
    which we only do after receiving the log via the callback.
    """
    import threading

    from tesseract_core import Tesseract

    output_path = Path(tmpdir)
    signal_dir = output_path / "signals"
    signal_dir.mkdir()

    # Container path for signals - output_path is mounted at /tesseract/output_data
    container_signal_dir = "/tesseract/output_data/signals"

    captured_logs = []
    log_lock = threading.Lock()

    def capture_log(line: str) -> None:
        with log_lock:
            captured_logs.append(line)
            # When we receive a log line, create the signal file to unblock the Tesseract
            for i in range(10):  # Check for any log line number
                if f"Log line {i}" in line:
                    signal_file = signal_dir / f"continue_{i}.signal"
                    signal_file.touch()
                    break

    with Tesseract.from_image(
        log_streaming_test_image, output_path=output_path
    ) as tesseract:
        result = tesseract.apply(
            {"num_lines": 3, "signal_dir": container_signal_dir},
            stream_logs=capture_log,
        )

    assert result["lines_printed"] == 3

    # Check that we captured all log lines
    assert len(captured_logs) == 3
    for i in range(3):
        assert f"Log line {i}" in captured_logs[i]


def test_log_streaming_sdk_from_tesseract_api(log_streaming_tesseract_api_path, tmpdir):
    """Test log streaming via Python SDK using Tesseract.from_tesseract_api().

    Uses file-based synchronization to prove logs are truly streaming.
    """
    import threading

    from tesseract_core import Tesseract

    output_path = Path(tmpdir)
    signal_dir = output_path / "signals"
    signal_dir.mkdir()

    captured_logs = []
    log_lock = threading.Lock()

    def capture_log(line: str) -> None:
        with log_lock:
            captured_logs.append(line)
            # When we receive a log line, create the signal file to unblock the Tesseract
            for i in range(10):
                if f"Log line {i}" in line:
                    signal_file = signal_dir / f"continue_{i}.signal"
                    signal_file.touch()
                    break

    with Tesseract.from_tesseract_api(
        log_streaming_tesseract_api_path, output_path=output_path
    ) as tesseract:
        result = tesseract.apply(
            {"num_lines": 3, "signal_dir": str(signal_dir)},
            stream_logs=capture_log,
        )

    assert result["lines_printed"] == 3

    # Check that we captured all log lines via the callback
    assert len(captured_logs) == 3
    for i in range(3):
        assert f"Log line {i}" in captured_logs[i]

    # Also verify the log file was created
    run_dirs = list(output_path.glob("run_*/"))
    assert len(run_dirs) >= 1
    log_file = run_dirs[0] / "logs" / "tesseract.log"
    assert log_file.exists()


def test_log_streaming_cli(log_streaming_test_image, tmpdir):
    """Test that CLI streams logs to stderr in real-time (default behavior).

    Uses file-based synchronization to prove logs are truly streaming:
    the Tesseract blocks after each log line until we create a signal file,
    which we only do after receiving the log via stderr.
    """
    import threading

    output_path = Path(tmpdir)
    signal_dir = output_path / "signals"
    signal_dir.mkdir()

    # Container path for signals - output_path is mounted at /tesseract/output_data
    container_signal_dir = "/tesseract/output_data/signals"

    payload = json.dumps(
        {"inputs": {"num_lines": 3, "signal_dir": container_signal_dir}}
    )

    captured_logs = []
    log_lock = threading.Lock()

    def run_tesseract():
        proc = subprocess.Popen(
            [
                "tesseract",
                "run",
                log_streaming_test_image,
                "apply",
                payload,
                "-o",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Read stderr in real-time
        while True:
            line = proc.stderr.readline()
            if line:
                line = line.strip()
                with log_lock:
                    captured_logs.append(line)
                # When we receive a log line, create signal file to unblock the Tesseract
                for i in range(10):
                    if f"Log line {i}" in line:
                        signal_file = signal_dir / f"continue_{i}.signal"
                        signal_file.touch()
                        break
            elif proc.poll() is not None:
                break

        stdout, remaining_stderr = proc.communicate()
        if remaining_stderr:
            for line in remaining_stderr.strip().split("\n"):
                if line:
                    with log_lock:
                        captured_logs.append(line)

        return proc.returncode, stdout

    returncode, stdout = run_tesseract()

    assert returncode == 0, f"Command failed. Captured logs: {captured_logs}"
    result = json.loads(stdout.strip())
    assert result["lines_printed"] == 3

    # Verify we captured all log lines
    log_lines = [log for log in captured_logs if "Log line" in log]
    assert len(log_lines) == 3, (
        f"Expected 3 log lines, got {len(log_lines)}: {log_lines}"
    )

    for i in range(3):
        assert any(f"Log line {i}" in line for line in log_lines), (
            f"Missing 'Log line {i}' in {log_lines}"
        )


def test_stream_logs_false_no_stderr(simple_logging_tesseract_api_path, tmpdir):
    """Test that stream_logs=False does not print logs to stderr.

    This is a regression test for the issue where logs were always printed
    to stderr even when stream_logs=False.
    """
    # Run in subprocess to capture real stderr (pytest captures it otherwise)
    test_script = f'''
import sys
from pathlib import Path
from tesseract_core import Tesseract

output_path = Path("{tmpdir}")

with Tesseract.from_tesseract_api(
    "{simple_logging_tesseract_api_path}", output_path=output_path
) as tesseract:
    result = tesseract.apply({{}}, stream_logs=False)

assert result["message"] == "done"
'''

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    # The key assertion: stderr should NOT contain log output
    assert "Hello from apply!" not in result.stderr, (
        f"Logs should not appear in stderr with stream_logs=False, "
        f"but got: {result.stderr}"
    )

    # Verify logs were still written to file
    run_dirs = list(Path(tmpdir).glob("run_*/"))
    assert len(run_dirs) >= 1
    log_file = run_dirs[0] / "logs" / "tesseract.log"
    assert log_file.exists()
    log_content = log_file.read_text()
    assert "Hello from apply!" in log_content


def test_stream_logs_print_no_infinite_loop(simple_logging_tesseract_api_path, tmpdir):
    """Test that stream_logs=print does not cause infinite recursion.

    This is a regression test for the issue where using `print` as the log sink
    caused infinite recursion because print writes to stdout which was redirected
    to the pipe being read.
    """
    # Run in subprocess with timeout to detect infinite loops
    test_script = f'''
import sys
from pathlib import Path
from tesseract_core import Tesseract

output_path = Path("{tmpdir}")

with Tesseract.from_tesseract_api(
    "{simple_logging_tesseract_api_path}", output_path=output_path
) as tesseract:
    result = tesseract.apply({{}}, stream_logs=print)

assert result["message"] == "done"
print("TEST_COMPLETED_SUCCESSFULLY", file=sys.__stderr__)
'''

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        timeout=30,  # Should complete quickly; timeout catches infinite loops
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    # Verify script completed (didn't hang in infinite loop)
    assert "TEST_COMPLETED_SUCCESSFULLY" in result.stderr
    # Verify the log was actually printed via the print sink
    assert "Hello from apply!" in result.stdout


def test_stream_logs_realtime(timed_logging_tesseract_api_path, tmpdir):
    """Test that logs are streamed in real-time, not buffered until the end.

    This test verifies that log lines appear incrementally as they are printed,
    with timing gaps between them, rather than all appearing at once when the
    tesseract completes.
    """
    import time

    from tesseract_core import Tesseract

    output_path = Path(tmpdir)
    timestamps = []

    def timed_sink(msg):
        timestamps.append((time.time(), msg))

    with Tesseract.from_tesseract_api(
        str(timed_logging_tesseract_api_path), output_path=output_path
    ) as tesseract:
        start = time.time()
        result = tesseract.apply({}, stream_logs=timed_sink)

    assert result["message"] == "done"

    # Check we got all 3 log lines
    log_lines = [(t, m) for t, m in timestamps if "Line" in m]
    assert len(log_lines) == 3, (
        f"Expected 3 log lines, got {len(log_lines)}: {log_lines}"
    )

    # Check that logs arrived incrementally (with gaps between them)
    # Each line should arrive ~0.5s after the previous one
    for i in range(1, len(log_lines)):
        gap = log_lines[i][0] - log_lines[i - 1][0]
        # Allow some tolerance but ensure there's a real gap (at least 0.3s)
        assert gap >= 0.3, (
            f"Gap between line {i} and {i + 1} was only {gap:.3f}s, "
            f"expected ~0.5s. Logs may not be streaming in real-time."
        )

    # First log should arrive quickly (within 0.5s of start)
    first_log_time = log_lines[0][0] - start
    assert first_log_time < 0.5, (
        f"First log took {first_log_time:.3f}s to arrive, expected < 0.5s"
    )
