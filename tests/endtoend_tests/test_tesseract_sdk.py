# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import socket

import numpy as np
import pytest
from common import build_tesseract, image_exists

from tesseract_core import Tesseract
from tesseract_core.sdk import engine

expected_endpoints = {
    "apply",
    "jacobian",
    "health",
    "input_schema",
    "output_schema",
    "abstract_eval",
    "jacobian_vector_product",
    "vector_jacobian_product",
}


@pytest.fixture(scope="module")
def built_image_name(docker_client, shared_dummy_image_name, dummy_tesseract_location):
    """Build the dummy Tesseract image for the tests."""
    image_name = build_tesseract(dummy_tesseract_location, shared_dummy_image_name)
    assert image_exists(docker_client, image_name)
    yield image_name


def test_available_endpoints(built_image_name):
    with Tesseract.from_image(built_image_name) as vecadd:
        assert set(vecadd.available_endpoints) == expected_endpoints


def test_apply(built_image_name, dummy_tesseract_location, free_port):
    inputs = {"a": [1, 2], "b": [3, 4], "s": 1}

    # Test URL access
    tesseract_url = f"http://localhost:{free_port}"
    served_tesseract = engine.serve([built_image_name], ports=[str(free_port)])
    try:
        vecadd = Tesseract(tesseract_url)
        out = vecadd.apply(inputs)
    finally:
        engine.teardown(served_tesseract)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Test from_image (context manager)
    with Tesseract.from_image(built_image_name) as vecadd:
        out = vecadd.apply(inputs)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Test from_image (serve + teardown)
    vecadd = Tesseract.from_image(built_image_name)
    try:
        vecadd.serve()
        out = vecadd.apply(inputs)
    finally:
        vecadd.teardown()

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Test from_tesseract_api
    with Tesseract.from_tesseract_api(
        dummy_tesseract_location / "tesseract_api.py"
    ) as vecadd:
        out = vecadd.apply(inputs)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))


def test_apply_with_error(built_image_name):
    # pass two inputs with different shapes, which raises an internal error
    inputs = {"a": [1, 2, 3], "b": [3, 4], "s": 1}

    with Tesseract.from_image(built_image_name) as vecadd:
        with pytest.raises(RuntimeError) as excinfo:
            vecadd.apply(inputs)

    assert "assert a.shape == b.shape" in str(excinfo.value)

    # get logs
    logs = vecadd.server_logs()
    assert "assert a.shape == b.shape" in logs


@pytest.fixture(scope="module")
def served_tesseract_remote(built_image_name):
    # Find a free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    free_port = sock.getsockname()[1]
    sock.close()
    # Serve the Tesseract image
    tesseract_url = f"http://localhost:{free_port}"
    served_tesseract = engine.serve([built_image_name], ports=[str(free_port)])
    try:
        yield tesseract_url
    finally:
        engine.teardown(served_tesseract)


@pytest.fixture(scope="module")
def served_tesseract_from_image(built_image_name):
    with Tesseract.from_image(built_image_name) as vecadd:
        yield vecadd


@pytest.fixture(scope="module")
def served_tesseract_module(dummy_tesseract_location):
    vecadd = Tesseract.from_tesseract_api(dummy_tesseract_location / "tesseract_api.py")
    yield vecadd


@pytest.mark.parametrize(
    "endpoint_name", sorted(expected_endpoints | {"openapi_schema"})
)
def test_all_endpoints(
    endpoint_name,
    served_tesseract_module,
    served_tesseract_from_image,
    served_tesseract_remote,
):
    """Test that all endpoints can be invoked without errors."""
    inputs = {"a": [1, 2], "b": [3, 4], "s": 1}

    if endpoint_name == "apply":
        inputs = {"inputs": inputs}
    elif endpoint_name == "jacobian":
        inputs = {"inputs": inputs, "jac_inputs": ["a"], "jac_outputs": ["result"]}
    elif endpoint_name == "jacobian_vector_product":
        inputs = {
            "inputs": inputs,
            "jvp_inputs": ["a"],
            "jvp_outputs": ["result"],
            "tangent_vector": {"a": np.array([1.0, 1.0])},
        }
    elif endpoint_name == "vector_jacobian_product":
        inputs = {
            "inputs": inputs,
            "vjp_inputs": ["a"],
            "vjp_outputs": ["result"],
            "cotangent_vector": {"result": np.array([1.0, 1.0])},
        }
    elif endpoint_name == "abstract_eval":
        inputs = {
            "abstract_inputs": {
                "a": {"shape": [2], "dtype": "float32"},
                "b": {"shape": [2], "dtype": "float32"},
            }
        }
    else:
        inputs = {}

    # Test from_tesseract_api
    out = getattr(served_tesseract_module, endpoint_name)
    if callable(out):
        out(**inputs)

    # Test from_image
    out = getattr(served_tesseract_from_image, endpoint_name)
    if callable(out):
        out(**inputs)

    # Test URL access
    vecadd = Tesseract(served_tesseract_remote)
    out = getattr(vecadd, endpoint_name)
    if callable(out):
        out(**inputs)
