# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common import build_tesseract, image_exists

from tesseract_core import Tesseract
from tesseract_core.sdk import engine


@pytest.fixture(scope="module")
def built_image_name(docker_client, shared_dummy_image_name, dummy_tesseract_location):
    """Build the dummy Tesseract image for the tests."""
    image_name = build_tesseract(dummy_tesseract_location, shared_dummy_image_name)
    assert image_exists(docker_client, image_name)
    yield image_name


def test_available_endpoints(built_image_name):
    with Tesseract.from_image(built_image_name) as vecadd:
        assert set(vecadd.available_endpoints) == {
            "apply",
            "jacobian",
            "health",
            "input_schema",
            "output_schema",
            "abstract_eval",
            "jacobian_vector_product",
            "vector_jacobian_product",
        }


def test_apply(built_image_name, free_port):
    inputs = {"a": [1, 2], "b": [3, 4], "s": 1}

    # Test URL access
    tesseract_url = f"http://localhost:{free_port}"
    served_tesseract = engine.serve([built_image_name], port=str(free_port))
    try:
        vecadd = Tesseract(tesseract_url)
        out = vecadd.apply(inputs)
    finally:
        engine.teardown(served_tesseract)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Test from_image
    with Tesseract.from_image(built_image_name) as vecadd:
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
