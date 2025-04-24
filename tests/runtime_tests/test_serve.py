# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from textwrap import dedent

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from tesseract_core.runtime.serve import create_rest_api

test_input = {
    "a": [1.0, 2.0, 3.0],
    "b": [1, 1, 1],
    "s": 2.5,
}


def is_wsl():
    """Check if the current environment is WSL."""
    kernel = platform.uname().release
    return "Microsoft" in kernel or "WSL" in kernel


def array_from_json(json_data):
    encoding = json_data["data"]["encoding"]
    if encoding == "base64":
        decoded_buffer = base64.b64decode(json_data["data"]["buffer"])
        array = np.frombuffer(decoded_buffer, dtype=json_data["dtype"]).reshape(
            json_data["shape"]
        )
    elif encoding == "json":
        array = np.array(json_data["data"]["buffer"], dtype=json_data["dtype"]).reshape(
            json_data["shape"]
        )

    return array


def model_to_json(model):
    return json.loads(model.model_dump_json())


@contextmanager
def serve_in_subprocess(api_file, port, num_workers=1, timeout=30.0):
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from tesseract_core.runtime.serve import serve; "
                f"serve(host='localhost', port={port}, num_workers={num_workers})",
            ],
            env=dict(os.environ, TESSERACT_API_PATH=api_file),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # wait for server to start
        while True:
            try:
                response = requests.get(f"http://localhost:{port}/health")
            except requests.exceptions.ConnectionError:
                pass
            else:
                if response.status_code == 200:
                    break

            time.sleep(0.1)
            timeout -= 0.1

            if timeout < 0:
                raise TimeoutError("Server did not start in time")

        yield f"http://localhost:{port}"

    finally:
        proc.terminate()
        stdout, stderr = proc.communicate()
        print(stdout.decode())
        print(stderr.decode())
        proc.wait(timeout=5)


@pytest.fixture
def http_client(dummy_tesseract_module):
    """A test HTTP client."""
    rest_api = create_rest_api(dummy_tesseract_module)
    return TestClient(rest_api)


@pytest.mark.parametrize(
    "format",
    [
        "json",
        "json+base64",
        pytest.param("json+binref", marks=pytest.mark.xfail),  # FIXME
        "msgpack",
    ],
)
def test_create_rest_api_apply_endpoint(http_client, dummy_tesseract_module, format):
    """Test we can get an Apply endpoint from generated API."""
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)

    response = http_client.post(
        "/apply",
        json={"inputs": model_to_json(test_inputs)},
        headers={"Accept": f"application/{format}"},
    )

    assert response.status_code == 200, response.text

    if format in {"json", "json+base64"}:
        result = array_from_json(response.json()["result"])
        assert np.array_equal(result, np.array([3.5, 6.0, 8.5]))
    elif format == "msgpack":
        assert (
            response.content
            == b"\x81\xa6result\x85\xc4\x02nd\xc3\xc4\x04type\xa3<f4\xc4\x04kind\xc4\x00\xc4\x05shape\x91\x03\xc4\x04data\xc4\x0c\x00\x00`@\x00\x00\xc0@\x00\x00\x08A"  # noqa: E501
        )
    elif format == "json+binref":
        raise NotImplementedError()


def test_create_rest_api_jacobian_endpoint(http_client, dummy_tesseract_module):
    """Test we can get a Jacobian endpoint from generated API."""
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)

    response = http_client.post(
        "/jacobian",
        json={
            "inputs": model_to_json(test_inputs),
            "jac_inputs": ["a", "b"],
            "jac_outputs": ["result"],
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    expected = dummy_tesseract_module.jacobian(test_inputs, {"a", "b"}, {"result"})

    assert result.keys() == expected.keys()
    assert np.array_equal(
        array_from_json(result["result"]["a"]), expected["result"]["a"]
    )


def test_create_rest_api_generates_health_endpoint(http_client):
    """Test we can get health endpoint from generated API."""
    response = http_client.get("/health")
    assert response.json() == {"status": "ok"}


def test_get_input_schema(http_client):
    response = http_client.get("/input_schema")

    assert response.status_code == 200, response.text


def test_get_output_schema(http_client):
    response = http_client.get("/output_schema")

    assert response.status_code == 200, response.text


def test_post_abstract_eval(http_client):
    payload = {
        "inputs": {
            "a": {"dtype": "float64", "shape": [4]},
            "b": {"dtype": "float64", "shape": [4]},
            "s": 1.0,
            "normalize": False,
        }
    }
    response = http_client.post("/abstract_eval", json=payload)

    assert response.status_code == 200, response.text
    assert response.json() == {"result": {"shape": [4], "dtype": "float64"}}


def test_post_abstract_eval_throws_validation_errors(http_client):
    response = http_client.post("/abstract_eval", json={"what": {"is": "this"}})

    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    error_types = [e["type"] for e in errors]

    assert "missing" in error_types
    assert "extra_forbidden" in error_types


def test_get_openapi_schema(http_client):
    response = http_client.get("/openapi.json")

    assert response.status_code == 200, response.text
    assert response.json()["info"]["title"] == "Tesseract"
    assert response.json()["paths"]


@pytest.mark.skipif(
    is_wsl(),
    reason="flaky on Windows",
)
def test_threading_sanity(tmpdir, free_port):
    """Test with a Tesseract that requires to be run in the main thread.

    This is important so we don't require users to be aware of threading issues.
    """
    TESSERACT_API = dedent(
        """
    import threading
    from pydantic import BaseModel

    assert threading.current_thread() == threading.main_thread()

    class InputSchema(BaseModel):
        pass

    class OutputSchema(BaseModel):
        pass

    def apply(input: InputSchema) -> OutputSchema:
        assert threading.current_thread() == threading.main_thread()
        return OutputSchema()
    """
    )

    api_file = tmpdir / "tesseract_api.py"

    with open(api_file, "w") as f:
        f.write(TESSERACT_API)

    # We can't run the server in the same process because it will use threading under the hood
    # so we need to spawn a new process instead
    with serve_in_subprocess(api_file, free_port) as url:
        response = requests.post(f"{url}/apply", json={"inputs": {}})
        assert response.status_code == 200, response.text


@pytest.mark.skipif(
    is_wsl(),
    reason="flaky on Windows",
)
def test_multiple_workers(tmpdir, free_port):
    """Test that the server can be run with multiple worker processes."""
    TESSERACT_API = dedent(
        """
    import time
    import multiprocessing
    from pydantic import BaseModel

    class InputSchema(BaseModel):
        pass

    class OutputSchema(BaseModel):
        pid: int

    def apply(input: InputSchema) -> OutputSchema:
        return OutputSchema(pid=multiprocessing.current_process().pid)
    """
    )

    api_file = tmpdir / "tesseract_api.py"

    with open(api_file, "w") as f:
        f.write(TESSERACT_API)

    with serve_in_subprocess(api_file, free_port, num_workers=2) as url:
        # Fire back-to-back requests to the server and check that they are handled
        # by different workers (i.e. different PIDs)
        post_request = lambda _: requests.post(f"{url}/apply", json={"inputs": {}})

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Fire a lot of requests in parallel
            futures = executor.map(post_request, range(100))
            responses = list(futures)

        # Check that all responses are 200
        for response in responses:
            assert response.status_code == 200, response.text

        # Check that not all pids are the same
        # (i.e. the requests were handled by different workers)
        pids = set(response.json()["pid"] for response in responses)
        assert len(pids) > 1, "All requests were handled by the same worker"


def test_debug_mode(dummy_tesseract_module, monkeypatch):
    import tesseract_core.runtime.config
    from tesseract_core.runtime.config import update_config

    def apply_that_raises(inputs):
        raise ValueError("This is a test error")

    orig_config = tesseract_core.runtime.config._current_config

    monkeypatch.setattr(dummy_tesseract_module, "apply", apply_that_raises)

    try:
        update_config(debug=False, tesseract_api_path=dummy_tesseract_module.__file__)
        rest_api = create_rest_api(dummy_tesseract_module)
        http_client = TestClient(rest_api, raise_server_exceptions=False)

        response = http_client.post(
            "/apply",
            json={
                "inputs": model_to_json(
                    dummy_tesseract_module.InputSchema.model_validate(test_input)
                )
            },
        )
        assert response.status_code == 500, response.text
        assert response.text == "Internal Server Error"
        assert "This is a test error" not in response.text
    finally:
        tesseract_core.runtime.config._current_config = orig_config

    try:
        update_config(debug=True, tesseract_api_path=dummy_tesseract_module.__file__)
        rest_api = create_rest_api(dummy_tesseract_module)
        http_client = TestClient(rest_api, raise_server_exceptions=False)

        response = http_client.post(
            "/apply",
            json={
                "inputs": model_to_json(
                    dummy_tesseract_module.InputSchema.model_validate(test_input)
                )
            },
        )
        assert response.status_code == 500, response.text
        assert "This is a test error" in response.text
        assert "Traceback" in response.text
    finally:
        tesseract_core.runtime.config._current_config = orig_config
