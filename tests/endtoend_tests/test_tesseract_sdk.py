# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common import build_tesseract, image_exists

from tesseract_core import Tesseract


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


def test_apply(built_image_name):
    input = {"a": [1, 2], "b": [3, 4], "s": 1}

    with Tesseract.from_image(built_image_name) as vecadd:
        out = vecadd.apply(input)

    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))
    assert set(out.keys()) == {"result"}
