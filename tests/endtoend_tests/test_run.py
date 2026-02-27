# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for running Tesseracts."""

import json
import subprocess

import numpy as np
import pytest
import requests
from common import build_tesseract, encode_array, image_exists

from tesseract_core.sdk.cli import app


@pytest.fixture(scope="module")
def built_image_name(
    docker_client,
    docker_cleanup_module,
    shared_dummy_image_name,
    dummy_tesseract_location,
):
    """Build the dummy Tesseract image for the tests."""
    image_name = build_tesseract(
        docker_client, dummy_tesseract_location, shared_dummy_image_name
    )
    assert image_exists(docker_client, image_name)
    docker_cleanup_module["images"].append(image_name)
    yield image_name


def test_tesseract_list(cli_runner, built_image_name):
    # Test List Command
    list_res = cli_runner.invoke(
        app,
        [
            "list",
        ],
        # Ensure that the output is not truncated
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert list_res.exit_code == 0, list_res.stderr
    assert built_image_name.split(":")[0] in list_res.stdout


def test_tesseract_run_stdout(cli_runner, built_image_name):
    test_commands = ("openapi-schema", "health")

    for command in test_commands:
        run_res = cli_runner.invoke(
            app,
            [
                "run",
                built_image_name,
                command,
            ],
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        assert run_res.stdout

        try:
            json.loads(run_res.stdout)
        except json.JSONDecodeError:
            print(f"failed to decode {command} stdout as JSON")
            print(run_res.stdout)
            raise


def test_run_with_memory(cli_runner, built_image_name):
    """Ensure we can run a Tesseract command with memory limits."""
    run_res = cli_runner.invoke(
        app,
        [
            "run",
            built_image_name,
            "health",
            "--memory",
            "512m",
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    # Verify the command executed successfully
    result = json.loads(run_res.stdout)
    assert result["status"] == "ok"


@pytest.mark.parametrize("method", ["run", "serve"])
@pytest.mark.parametrize("array_format", ["json", "json+base64", "json+binref"])
def test_io_path_interactions(
    docker_cleanup, built_image_name, tmp_path, method, array_format
):
    """Ensure that input / output paths work across different methods of interaction and file formats."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    encoding = array_format.split("+")[-1]
    example_inputs = {
        "inputs": {
            "a": encode_array(np.array([1, 2]), encoding=encoding, basedir=input_dir),
            "b": encode_array(np.array([3, 4]), encoding=encoding, basedir=input_dir),
            "s": 1.0,
            "normalize": True,
        },
    }

    if method == "serve":
        run_res = subprocess.run(
            [
                "tesseract",
                "serve",
                built_image_name,
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-f",
                array_format,
            ],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0, run_res.stderr
        assert run_res.stdout

        serve_meta = json.loads(run_res.stdout)
        container_name = serve_meta["container_name"]
        docker_cleanup["containers"].append(container_name)

        req = requests.post(
            f"http://localhost:{serve_meta['containers'][0]['port']}/apply",
            json=example_inputs,
        )
        assert req.status_code == 200, req.text
        result = req.json()

    elif method == "run":
        run_res = subprocess.run(
            [
                "tesseract",
                "run",
                built_image_name,
                "apply",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-f",
                array_format,
                json.dumps(example_inputs),
            ],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0, run_res.stderr
        assert run_res.stdout
        result = json.loads(run_res.stdout)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure result payload is as expected
    assert "result" in result
    assert result["result"]["data"]["encoding"] == encoding

    if array_format == "json+binref":
        binref_path = result["result"]["data"]["buffer"].rsplit(":", maxsplit=1)[0]
        binref_file = output_dir / binref_path
        assert binref_file.exists(), f"Expected binref file {binref_file} to exist"

    # Ensure logs are written to the output directory
    if method == "serve":
        run_dirs = list(output_dir.glob("run_*/"))
        assert len(run_dirs) == 1, f"Expected one run directory, found: {run_dirs}"
        output_dir = run_dirs[0]

    log_dir = output_dir / "logs"
    assert log_dir.exists(), f"Expected log directory {log_dir} to exist"
    assert (log_dir / "tesseract.log").exists(), "Expected tesseract.log to exist"

    # Also try overriding the output format via Accept header
    if method == "serve":
        req = requests.post(
            f"http://localhost:{serve_meta['containers'][0]['port']}/apply",
            json=example_inputs,
            headers={"Accept": "application/json+base64"},
        )
        assert req.status_code == 200, req.text
        result = req.json()
        assert "result" in result
        assert result["result"]["data"]["encoding"] == "base64"
