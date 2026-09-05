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
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from typeguard import suppress_type_checks

from tesseract_core.runtime.config import get_config
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


def array_from_json(json_data, base_dir=None):
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
    elif encoding == "binref":
        binfile, offset = json_data["data"]["buffer"].split(":")
        length = np.prod(json_data["shape"]) * np.dtype(json_data["dtype"]).itemsize
        with open(base_dir / binfile, "rb") as f:
            f.seek(int(offset))
            decoded_buffer = f.read(length)
        array = np.frombuffer(decoded_buffer, dtype=json_data["dtype"]).reshape(
            json_data["shape"]
        )
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return array


def model_to_json(model):
    return json.loads(model.model_dump_json())


@contextmanager
def serve_in_subprocess(api_file, port, num_workers=1, timeout=30.0):
    proc = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from tesseract_core.runtime.serve import serve; "
                f"serve(host='localhost', port={port}, num_workers={num_workers})",
            ],
            env=dict(os.environ, TESSERACT_API_PATH=str(api_file)),
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
        if proc is not None:
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
        "json+binref",
    ],
)
def test_create_rest_api_apply_endpoint(http_client, dummy_tesseract_module, format):
    """Test we can get an Apply endpoint from generated API."""
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)

    response = http_client.post(
        "/apply",
        json={"inputs": model_to_json(test_inputs)},
        headers={"Accept": f"application/{format}"},
        params={"run_id": "test_job"},
    )

    assert response.status_code == 200, response.text

    result = array_from_json(response.json()["result"], Path(get_config().output_path))
    assert np.array_equal(result, np.array([3.5, 6.0, 8.5]))


def test_apply_rejects_experimental_cuda_ipc_by_default(dummy_tesseract_module):
    """The experimental json+cuda_ipc format is refused unless explicitly enabled.

    A Tesseract must not produce CUDA IPC handles in production unless the
    ``enable_experimental_cuda_ipc`` runtime flag is set.
    """
    from tesseract_core.runtime.config import update_config

    update_config(enable_experimental_cuda_ipc=False)
    client = TestClient(
        create_rest_api(dummy_tesseract_module), raise_server_exceptions=False
    )
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)
    response = client.post(
        "/apply",
        json={"inputs": model_to_json(test_inputs)},
        headers={"Accept": "application/json+cuda_ipc"},
    )
    assert response.status_code == 406, response.text
    detail = response.json()["detail"]
    assert "json+cuda_ipc" in detail["message"]
    assert detail["available_formats"] == ["json", "json+base64", "json+binref"]


def test_apply_unknown_output_format_is_not_acceptable(dummy_tesseract_module):
    """An Accept header naming a format the runtime does not know is a client error.

    The 406 body lists what the server does offer, so a client can fall back.
    """
    client = TestClient(
        create_rest_api(dummy_tesseract_module), raise_server_exceptions=False
    )
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)
    response = client.post(
        "/apply",
        json={"inputs": model_to_json(test_inputs)},
        headers={"Accept": "application/json+nonsense"},
    )
    assert response.status_code == 406, response.text
    detail = response.json()["detail"]
    assert "json+nonsense" in detail["message"]
    assert set(detail["available_formats"]) >= {"json", "json+base64", "json+binref"}


def test_apply_rejects_an_accept_header_with_nothing_on_offer(dummy_tesseract_module):
    """A header naming only types the server cannot produce is a 406."""
    client = TestClient(
        create_rest_api(dummy_tesseract_module), raise_server_exceptions=False
    )
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)
    response = client.post(
        "/apply",
        json={"inputs": model_to_json(test_inputs)},
        headers={"Accept": "text/html"},
    )
    assert response.status_code == 406, response.text


@pytest.mark.parametrize(
    "accept,expected_encoding",
    [
        # Whatever a client happens to send, the server answers in a format it
        # actually offers, and only refuses when it can offer none.
        ("application/json, text/plain, */*", "json"),
        ("text/html,application/xhtml+xml,*/*;q=0.8", "json"),
        ("application/*", "json"),
        ("APPLICATION/JSON", "json"),
        ("application/json;q=0.9", "json"),
        ("", "json"),
        ("text/html, application/json+base64", "base64"),
        ("application/json+nonsense, application/json+binref", "binref"),
        # Higher q wins, even when it is listed second.
        ("application/json;q=0.2, application/json+base64;q=0.8", "base64"),
    ],
)
def test_apply_negotiates_the_best_offered_format(
    http_client, dummy_tesseract_module, accept, expected_encoding
):
    """Media ranges are honoured by q-value, and wildcards fall back to the default."""
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)
    response = http_client.post(
        "/apply",
        json={"inputs": model_to_json(test_inputs)},
        headers={"Accept": accept},
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["data"]["encoding"] == expected_encoding
    # The Content-Type names the format produced, not the header sent.
    assert response.headers["content-type"].startswith("application/json")


def test_unacceptable_output_format_is_rejected_before_running(
    dummy_tesseract_module, monkeypatch
):
    """Format negotiation fails before the endpoint does any work."""
    calls = []

    def apply_that_records(inputs):
        calls.append(inputs)
        return dummy_tesseract_module.OutputSchema(result=np.zeros(3))

    monkeypatch.setattr(dummy_tesseract_module, "apply", apply_that_records)
    client = TestClient(
        create_rest_api(dummy_tesseract_module), raise_server_exceptions=False
    )
    test_inputs = dummy_tesseract_module.InputSchema.model_validate(test_input)
    response = client.post(
        "/apply",
        json={"inputs": model_to_json(test_inputs)},
        headers={"Accept": "application/json+nonsense"},
    )
    assert response.status_code == 406, response.text
    assert calls == []


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
    # The run_id query parameter is intended to be hidden
    assert "run_id" not in response.json()


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
    is_wsl() or sys.platform == "win32",
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
    from tesseract_core.runtime.config import update_config

    def apply_that_raises(inputs):
        raise ValueError("This is a test error")

    monkeypatch.setattr(dummy_tesseract_module, "apply", apply_that_raises)

    update_config(debug=False, api_path=dummy_tesseract_module.__file__)
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
    assert "/test" not in rest_api.openapi()["paths"]

    update_config(debug=True, api_path=dummy_tesseract_module.__file__)
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
    assert "/test" in rest_api.openapi()["paths"]


def test_custom_validation_error_over_http(dummy_tesseract_module):
    """Custom array validation errors are properly reconstructed as ValidationErrors via the SDK."""
    from pydantic import ValidationError as PydanticValidationError

    from tesseract_core import Tesseract

    rest_api = create_rest_api(dummy_tesseract_module)
    test_client = TestClient(rest_api, raise_server_exceptions=False)

    tess = Tesseract.from_url("http://testserver")
    test_client.headers["Content-Type"] = "application/json"
    tess._client._session = test_client

    # Send a properly structured encoded array with non-numeric data in the buffer
    # to trigger a custom PydanticCustomError (array_decode_error) on the server
    non_numeric_array = {
        "object_type": "array",
        "shape": [2],
        "dtype": "float32",
        "data": {"buffer": ["not", "numeric"], "encoding": "json"},
    }
    valid_array = {
        "object_type": "array",
        "shape": [2],
        "dtype": "float32",
        "data": {"buffer": [1, 2], "encoding": "json"},
    }

    # The injected TestClient returns httpx responses instead of requests ones,
    # which trips typeguard's return type check on HTTPClient._send
    with suppress_type_checks(), pytest.raises(PydanticValidationError) as exc_info:
        tess.apply({"a": non_numeric_array, "b": valid_array, "s": 1})

    err_types = {e["type"] for e in exc_info.value.errors()}
    assert "array_decode_error" in err_types
