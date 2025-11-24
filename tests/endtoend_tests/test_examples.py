# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for all unit Tesseracts.

Each unit Tesseract is tested by building a Docker image from it and then
running the Tesseract CLI + HTTP interface to test its functionality.

Add test cases for specific unit Tesseracts to the TEST_CASES dictionary.
"""

import base64
import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import requests
from common import build_tesseract, encode_array, image_exists


def json_normalize(obj: str):
    """Normalize JSON str for comparison."""
    return json.dumps(json.loads(obj), separators=(",", ":"))


def assert_contains_array_allclose(
    output: dict | list,
    array_expected: npt.ArrayLike,
    rtol=1e-4,
    atol=1e-4,
):
    """Check the output pytree for an array close to array_expected."""

    def _get_array_leaves(tree):
        if isinstance(tree, dict) and tree.get("object_type") == "array":
            shape = tree["shape"]
            dtype = tree["dtype"]
            data = tree["data"]
            if data["encoding"] == "base64":
                buffer = base64.b64decode(data["buffer"])
                array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
            else:
                raise NotImplementedError("only base64 encoding supported")
            yield array
        elif isinstance(tree, dict):
            for val in tree.values():
                yield from _get_array_leaves(val)
        elif isinstance(tree, list):
            for val in tree:
                yield from _get_array_leaves(val)
        else:
            raise AssertionError(f"Got unexpected type {type(tree)}")

    output_arrays = list(_get_array_leaves(output))
    expected = np.asarray(array_expected)
    for array in output_arrays:
        if np.allclose(array, expected, rtol=rtol, atol=atol):
            return

    raise AssertionError(
        f"Expected array not found in output.\nExpected: {expected}\nFound arrays: {output_arrays}"
    )


@dataclass
class SampleRequest:
    endpoint: str
    payload: dict
    expected_status_code: int = 200
    output_contains_pattern: str | list[str] | None = None
    output_contains_array: npt.ArrayLike | None = None
    output_format: str = "json+base64"


@dataclass
class Config:
    test_with_random_inputs: bool = False
    sample_requests: list[SampleRequest] = None
    volume_mounts: list[str] = None
    input_path: str = None
    output_path: str = None


# Add config and test cases for specific unit Tesseracts here
TEST_CASES = {
    "empty": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(endpoint="apply", payload={"inputs": {}}),
        ],
    ),
    "py310": Config(
        test_with_random_inputs=True,
    ),
    "helloworld": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {"name": "Ozzy"}},
                output_contains_pattern="Hello Ozzy!",
            ),
        ],
    ),
    "pip_custom_step": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {"name": "Ozzy"}},
                output_contains_pattern="Hello Ozzy!",
            ),
        ],
    ),
    "pyvista-arm64": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "vertices": encode_array(
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
                        )
                    }
                },
                output_contains_pattern="pv_mesh",
            ),
        ],
    ),
    "localpackage": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {"name": "Ozzy"}},
                output_contains_pattern="Hello Ozzy!",
            ),
        ],
    ),
    "vectoradd": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "a": encode_array([1, 2, 3]),
                        "b": encode_array([4, 5, 6]),
                    },
                },
                output_contains_pattern=[encode_array([5.0, 7.0, 9.0], as_json=True)],
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {
                        "a": encode_array([1, 2, 3]),
                        "b": encode_array([4, 5, 6]),
                    },
                    "jac_inputs": ["s", "a"],
                    "jac_outputs": ["result"],
                },
                output_contains_pattern=['"s":', '"a":'],
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {
                        "a": encode_array([1, 2, 3]),
                        "b": encode_array([4, 5, 6]),
                    },
                    "jac_inputs": ["invalid", "inputs"],
                    "jac_outputs": ["result"],
                },
                expected_status_code=422,
                output_contains_pattern="Input should be",
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {
                        "a": encode_array([1, 2, 3]),
                        "b": encode_array([4, 5, 6]),
                    },
                    "jac_inputs": ["s", "a"],
                    "jac_outputs": ["invalid", "outputs"],
                },
                expected_status_code=422,
                output_contains_pattern="Input should be",
            ),
            SampleRequest(
                endpoint="check-gradients",
                payload={
                    "inputs": {
                        "a": encode_array([1, 2, 3]),
                        "b": encode_array([4, 5, 6]),
                    },
                },
            ),
        ],
    ),
    "vectoradd_jax": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 3},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                },
                output_contains_array=np.array([7.0, 11.0, 15.0], dtype="float32"),
            ),
            SampleRequest(
                endpoint="abstract_eval",
                payload={
                    "inputs": {
                        "a": {
                            "v": {"shape": [3], "dtype": "float32"},
                            "s": {"shape": [], "dtype": "float32"},
                        },
                        "b": {
                            "v": {"shape": [3], "dtype": "float32"},
                            "s": {"shape": [], "dtype": "float32"},
                        },
                    }
                },
            ),
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {}},
                expected_status_code=422,
                output_contains_pattern="missing",
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 2},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                    "jac_inputs": ["a.s", "a.v"],
                    "jac_outputs": ["vector_add.result"],
                },
                output_contains_pattern=['"a.s":', '"a.v":'],
                output_contains_array=np.array([[1, 2, 3]], dtype="float32"),
            ),
            SampleRequest(
                endpoint="jacobian_vector_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 2},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                    "jvp_inputs": ["a.v"],
                    "jvp_outputs": ["vector_add.result"],
                    "tangent_vector": {"a.v": encode_array([0.1, 0.2, 0.3])},
                },
                output_contains_array=np.array([0.2, 0.4, 0.6], dtype="float32"),
            ),
            SampleRequest(
                endpoint="jacobian_vector_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 2},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                    "jvp_inputs": ["a.s"],
                    "jvp_outputs": ["vector_add.result"],
                    "tangent_vector": {"a.s": 0.5},
                },
                output_contains_array=np.array([0.5, 1.0, 1.5], dtype="float32"),
            ),
            SampleRequest(
                endpoint="vector_jacobian_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1.0, 2.0, 3.0]), "s": 2},
                        "b": {"v": encode_array([4.0, 5.0, 6.0]), "s": 1},
                    },
                    "vjp_inputs": ["a.v"],
                    "vjp_outputs": ["vector_add.result"],
                    "cotangent_vector": {
                        "vector_add.result": encode_array([0.1, 0.2, 0.3]),
                    },
                },
                output_contains_array=np.array([0.2, 0.4, 0.6], dtype="float32"),
            ),
            SampleRequest(
                endpoint="vector_jacobian_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1.0, 2.0, 3.0]), "s": 2},
                        "b": {"v": encode_array([4.0, 5.0, 6.0]), "s": 1},
                    },
                    "vjp_inputs": ["a.s"],
                    "vjp_outputs": ["vector_add.result"],
                    "cotangent_vector": {
                        "vector_add.result": encode_array([0.1, 0.2, 0.3]),
                    },
                },
                output_contains_array=np.array([1.4], dtype="float32"),
            ),
            SampleRequest(
                endpoint="check-gradients",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 2},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                },
            ),
        ],
    ),
    "vectoradd_torch": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 3},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                },
                output_contains_array=np.array([7.0, 11.0, 15.0], dtype="float32"),
            ),
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {}},
                expected_status_code=422,
                output_contains_pattern="missing",
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 2},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                    "jac_inputs": ["a.s", "a.v"],
                    "jac_outputs": ["vector_add.result"],
                },
                output_contains_pattern=['"a.s":', '"a.v":'],
                output_contains_array=np.array([[1, 2, 3]], dtype="float32"),
            ),
            SampleRequest(
                endpoint="jacobian_vector_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 2},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                    "jvp_inputs": ["a.v"],
                    "jvp_outputs": ["vector_add.result"],
                    "tangent_vector": {"a.v": encode_array([0.1, 0.2, 0.3])},
                },
                output_contains_array=np.array([0.2, 0.4, 0.6], dtype="float32"),
            ),
            SampleRequest(
                endpoint="jacobian_vector_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1, 2, 3]), "s": 2},
                        "b": {"v": encode_array([4, 5, 6]), "s": 1},
                    },
                    "jvp_inputs": ["a.s"],
                    "jvp_outputs": ["vector_add.result"],
                    "tangent_vector": {"a.s": 0.5},
                },
                output_contains_array=np.array([0.5, 1.0, 1.5], dtype="float32"),
            ),
            SampleRequest(
                endpoint="vector_jacobian_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1.0, 2.0, 3.0]), "s": 2},
                        "b": {"v": encode_array([4.0, 5.0, 6.0]), "s": 1},
                    },
                    "vjp_inputs": ["a.v"],
                    "vjp_outputs": ["vector_add.result"],
                    "cotangent_vector": {
                        "vector_add.result": encode_array([0.1, 0.2, 0.3]),
                    },
                },
                output_contains_array=np.array([0.2, 0.4, 0.6], dtype="float32"),
            ),
            SampleRequest(
                endpoint="vector_jacobian_product",
                payload={
                    "inputs": {
                        "a": {"v": encode_array([1.0, 2.0, 3.0]), "s": 2},
                        "b": {"v": encode_array([4.0, 5.0, 6.0]), "s": 1},
                    },
                    "vjp_inputs": ["a.s"],
                    "vjp_outputs": ["vector_add.result"],
                    "cotangent_vector": {
                        "vector_add.result": encode_array([0.1, 0.2, 0.3]),
                    },
                },
                output_contains_array=np.array([1.4], dtype="float32"),
            ),
        ],
    ),
    "univariate": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {"x": 0.0, "y": 0.0},
                },
                output_contains_array=np.array([1.0], dtype="float64"),
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {"x": 0.0, "y": 0.0},
                    "jac_inputs": ["x"],
                    "jac_outputs": ["result"],
                },
                output_contains_pattern=[
                    f'"result":{{"x":{encode_array(np.float32(-2.0), as_json=True)}}}'
                ],
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {"x": 0.0, "y": 0.0},
                    "jac_inputs": ["y"],
                    "jac_outputs": ["result"],
                },
                output_contains_pattern=[
                    f'"result":{{"y":{encode_array(np.float32(0.0), as_json=True)}}}'
                ],
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "inputs": {"x": 0.0, "y": 0.0},
                    "jac_inputs": ["hey"],
                    "jac_outputs": ["result"],
                },
                expected_status_code=422,
                output_contains_pattern="Input should be",
            ),
            SampleRequest(
                endpoint="jacobian_vector_product",
                payload={
                    "inputs": {"x": 0.0, "y": 0.0},
                    "jvp_inputs": ["x", "y"],
                    "jvp_outputs": ["result"],
                    "tangent_vector": {"x": 1.0, "y": 0.0},
                },
                output_contains_pattern=f'"result":{encode_array(np.float32(-2.0), as_json=True)}',
            ),
            SampleRequest(
                endpoint="vector_jacobian_product",
                payload={
                    "inputs": {"x": 0.0, "y": 0.0},
                    "vjp_inputs": ["x", "y"],
                    "vjp_outputs": ["result"],
                    "cotangent_vector": {"result": 1.0},
                },
                output_contains_pattern=[
                    f'"x":{encode_array(np.float32(-2.0), as_json=True)}',
                    f'"y":{encode_array(np.float32(0.0), as_json=True)}',
                ],
            ),
            SampleRequest(
                endpoint="check-gradients",
                payload={
                    "inputs": {"x": 0.0, "y": 0.0},
                },
            ),
        ],
    ),
    "package_data": Config(
        test_with_random_inputs=True,
    ),
    "cuda": Config(
        test_with_random_inputs=True,
    ),
    "meshstats": Config(
        test_with_random_inputs=False,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "mesh": {
                            "n_points": 5,
                            "n_cells": 2,
                            "points": encode_array(
                                [
                                    [0.0, 666.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 1.0, 0.0],
                                    [0.5, 0.5, 1.0],
                                ]
                            ),
                            "num_points_per_cell": encode_array([4, 4]),
                            "cell_connectivity": encode_array([0, 1, 2, 3, 1, 2, 3, 4]),
                            "cell_data": {
                                "temperature": encode_array(
                                    [[100.0, 105.0], [110.0, 115.0]]
                                ),
                                "pressure": encode_array([[1.0, 1.2], [1.1, 1.3]]),
                            },
                            "point_data": {
                                "displacement": encode_array(
                                    [
                                        [0.0, 0.1, 0.2],
                                        [0.1, 0.0, 0.2],
                                        [0.2, 0.1, 0.0],
                                        [0.1, 0.2, 0.1],
                                        [0.2, 0.1, 0.1],
                                    ]
                                ),
                                "velocity": encode_array(
                                    [
                                        [0.0, 0.0, 0.0],
                                        [0.1, 0.0, 0.0],
                                        [0.0, 0.1, 0.0],
                                        [0.0, 0.0, 0.1],
                                        [0.1, 0.1, 0.1],
                                    ]
                                ),
                            },
                        }
                    },
                },
                output_contains_pattern=encode_array(
                    np.array([0.0, 666.0, 0.0], dtype="float32"), as_json=True
                ),
            ),
            SampleRequest(
                endpoint="abstract_eval",
                payload={
                    "inputs": {
                        "mesh": {
                            "n_points": 5,
                            "n_cells": 8,
                            "points": {
                                "shape": [5, 3],
                                "dtype": "float32",
                            },
                            "num_points_per_cell": {
                                "shape": [2],
                                "dtype": "int32",
                            },
                            "cell_connectivity": {
                                "shape": [8],
                                "dtype": "int32",
                            },
                            "cell_data": {
                                "temperature": {
                                    "shape": [2, 2],
                                    "dtype": "float32",
                                },
                                "pressure": {
                                    "shape": [2, 2],
                                    "dtype": "float32",
                                },
                            },
                            "point_data": {
                                "displacement": {
                                    "shape": [5, 3],
                                    "dtype": "float32",
                                },
                                "velocity": {
                                    "shape": [5, 3],
                                    "dtype": "float32",
                                },
                            },
                        }
                    },
                },
                output_contains_pattern='"shape":[3],"dtype":"float32"',
            ),
            SampleRequest(
                endpoint="jacobian",
                payload={
                    "jac_inputs": ["mesh.points"],
                    "jac_outputs": ["statistics.barycenter"],
                    "inputs": {
                        "mesh": {
                            "n_points": 5,
                            "n_cells": 2,
                            "points": encode_array(
                                [
                                    [0.0, 666.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 1.0, 0.0],
                                    [0.5, 0.5, 1.0],
                                ]
                            ),
                            "num_points_per_cell": encode_array([4, 4]),
                            "cell_connectivity": encode_array([0, 1, 2, 3, 1, 2, 3, 4]),
                            "cell_data": {
                                "temperature": encode_array(
                                    [[100.0, 105.0], [110.0, 115.0]]
                                ),
                                "pressure": encode_array([[1.0, 1.2], [1.1, 1.3]]),
                            },
                            "point_data": {
                                "displacement": encode_array(
                                    [
                                        [0.0, 0.1, 0.2],
                                        [0.1, 0.0, 0.2],
                                        [0.2, 0.1, 0.0],
                                        [0.1, 0.2, 0.1],
                                        [0.2, 0.1, 0.1],
                                    ]
                                ),
                                "velocity": encode_array(
                                    [
                                        [0.0, 0.0, 0.0],
                                        [0.1, 0.0, 0.0],
                                        [0.0, 0.1, 0.0],
                                        [0.0, 0.0, 0.1],
                                        [0.1, 0.1, 0.1],
                                    ]
                                ),
                            },
                        }
                    },
                },
                output_contains_pattern='"shape":[3,5,3]',
            ),
            SampleRequest(
                endpoint="check-gradients",
                payload={
                    "inputs": {
                        "mesh": {
                            "n_points": 5,
                            "n_cells": 2,
                            "points": encode_array(
                                [
                                    [0.0, 2.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 1.0, 0.0],
                                    [0.5, 0.5, 1.0],
                                ]
                            ),
                            "num_points_per_cell": encode_array([4, 4]),
                            "cell_connectivity": encode_array([0, 1, 2, 3, 1, 2, 3, 4]),
                            "cell_data": {
                                "temperature": encode_array(
                                    [[100.0, 105.0], [110.0, 115.0]]
                                ),
                                "pressure": encode_array([[1.0, 1.2], [1.1, 1.3]]),
                            },
                            "point_data": {
                                "displacement": encode_array(
                                    [
                                        [0.0, 0.1, 0.2],
                                        [0.1, 0.0, 0.2],
                                        [0.2, 0.1, 0.0],
                                        [0.1, 0.2, 0.1],
                                        [0.2, 0.1, 0.1],
                                    ]
                                ),
                                "velocity": encode_array(
                                    [
                                        [0.0, 0.0, 0.0],
                                        [0.1, 0.0, 0.0],
                                        [0.0, 0.1, 0.0],
                                        [0.0, 0.0, 0.1],
                                        [0.1, 0.1, 0.1],
                                    ]
                                ),
                            },
                        }
                    },
                },
            ),
        ],
    ),
    "dataloader": Config(
        test_with_random_inputs=False,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "data": "@/tesseract/input_data/sample_*.json",
                    },
                },
                output_contains_pattern=[
                    '{"data":[{"object_type":"array","shape":[3,3],"dtype":"float32","data":{"buffer":',
                ],
            ),
            SampleRequest(
                endpoint="check-gradients",
                payload={
                    "inputs": {
                        "data": "@/tesseract/input_data/sample_*.json",
                    },
                },
            ),
        ],
        volume_mounts=["testdata:/tesseract/input_data:ro"],
    ),
    "conda": Config(
        test_with_random_inputs=False,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {"message": "Hey!"}},
                output_contains_pattern=[r'{"cowsays":"  ____\n| Hey! |\n  ====\n'],
            )
        ],
    ),
    "required_files": Config(
        test_with_random_inputs=False,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {}},
                output_contains_pattern=[r'{"a":1.0,"b":100.0}'],
            ),
        ],
        input_path="input",
    ),
    "filereference": Config(
        test_with_random_inputs=False,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "data": [
                            "sample_0.json",
                            "sample_1.json",
                            "sample_2.json",
                            "sample_3.json",
                            "sample_4.json",
                            "sample_5.json",
                            "sample_6.json",
                            "sample_7.json",
                            "sample_8.json",
                            "sample_9.json",
                        ]
                    }
                },
                output_contains_pattern=["sample_0.copy"],
            )
        ],
        input_path="testdata",
        output_path="output",
    ),
    "metrics": Config(
        test_with_random_inputs=True,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={"inputs": {}},
                # Just verify it runs without error - output will be empty
            ),
        ],
    ),
    "qp_solve": Config(
        test_with_random_inputs=False,
        sample_requests=[
            SampleRequest(
                endpoint="apply",
                payload={
                    "inputs": {
                        "Q": encode_array(np.eye(2)),
                        "q": encode_array(np.ones(2)),
                        "G": encode_array(np.array([[-1.0, 1.0]]).reshape((1, 2))),
                        "h": encode_array(np.array([-1]).reshape((1,))),
                        "target_kappa": 1e-1,
                        "solver_tol": 1e-4,
                    }
                },
                output_contains_array=np.array([-0.5, -1.5], dtype="float32"),
            )
        ],
    ),
    "tesseractreference": Config(  # Can't test requests standalone; needs target Tesseract. Covered in separate test.
        test_with_random_inputs=False, sample_requests=[]
    ),
}


@pytest.fixture
def unit_tesseract_config(unit_tesseract_names, unit_tesseract_path):
    for tesseract in TEST_CASES:
        if tesseract not in unit_tesseract_names:
            raise ValueError(
                f"A test case in TEST_CASES refers to a nonexistent Tesseract {tesseract}."
            )

    if unit_tesseract_path.name not in TEST_CASES:
        raise ValueError(
            f"No test case found for Tesseract {unit_tesseract_path.name}."
        )

    return TEST_CASES[unit_tesseract_path.name]


def print_debug_info(result):
    """Print debug info from result of a CLI command if it failed."""
    if result.exit_code == 0:
        return
    print(result.output)
    if result.exc_info:
        traceback.print_exception(*result.exc_info)


def fix_fake_arrays(fakedata, seed=42):
    is_array = (
        lambda x: isinstance(x, dict) and "shape" in x and "dtype" in x and "data" in x
    )
    rng = np.random.RandomState(seed)

    def _walk(data):
        if is_array(data):
            # Use broadcasting to minimize surface for shape errors
            data["shape"] = (1,) * len(data["shape"])

            new_data = rng.uniform(0, 100, data["shape"]).astype(data["dtype"])

            if data["data"]["encoding"] == "base64":
                data["data"]["buffer"] = base64.b64encode(new_data.tobytes()).decode()
            elif data["data"]["encoding"] == "json":
                data["data"]["buffer"] = new_data.flatten().tolist()
            elif data["data"]["encoding"] == "binref":
                # FIXME: overriding with base64 to not have to create files
                # in the tesseract
                data["data"]["encoding"] = "base64"
                data["data"]["buffer"] = base64.b64encode(new_data.tobytes()).decode()
        elif isinstance(data, dict):
            for key, value in data.items():
                data[key] = _walk(value)
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                data[idx] = _walk(value)

        return data

    return _walk(fakedata)


def example_from_json_schema(schema):
    """Generate a random example JSON object from a JSON schema."""
    import jsf

    faker = jsf.JSF(schema)
    payload = faker.generate()
    payload = fix_fake_arrays(payload)
    return payload


def test_unit_tesseract_endtoend(
    cli_runner,
    docker_client,
    dummy_image_name,
    unit_tesseract_path,
    unit_tesseract_config,
    free_port,
    docker_cleanup,
):
    """Test that unit Tesseract images can be built and used to serve REST API."""
    from tesseract_core.sdk.cli import app

    # Stage 1: Build
    img_name = build_tesseract(
        docker_client,
        unit_tesseract_path,
        dummy_image_name,
        tag="sometag",
    )
    assert image_exists(docker_client, img_name)
    docker_cleanup["images"].append(img_name)

    # Stage 2: Test CLI usage
    result = cli_runner.invoke(
        app,
        [
            "run",
            img_name,
            "openapi-schema",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    openapi_schema = json.loads(result.stdout)

    def _input_schema_from_openapi(openapi_schema):
        input_schema = openapi_schema["components"]["schemas"]["ApplyInputSchema"]
        # For some reason, jsf can't handle #/components/schemas/<x> references,
        # so we convert them to #$defs/<x>
        input_schema.update({"$defs": openapi_schema["components"]["schemas"]})
        input_schema["$defs"].pop("ApplyInputSchema", None)
        input_schema = json.loads(
            json.dumps(input_schema).replace("components/schemas", "$defs")
        )
        return input_schema

    input_schema = _input_schema_from_openapi(openapi_schema)

    mount_args, io_args = [], []

    if unit_tesseract_config.volume_mounts:
        for mnt in unit_tesseract_config.volume_mounts:
            # Assume that the mount is relative to the Tesseract path
            local_path, *other = mnt.split(":")
            local_path = Path(local_path)
            if not local_path.is_absolute():
                local_path = unit_tesseract_path / local_path
            mnt = ":".join([str(local_path), *other])
            mount_args.extend(["--volume", mnt])

    if unit_tesseract_config.input_path:
        io_args.extend(
            [
                "--input-path",
                str(unit_tesseract_path / unit_tesseract_config.input_path),
            ]
        )
    if unit_tesseract_config.output_path:
        io_args.extend(
            [
                "--output-path",
                str(unit_tesseract_path / unit_tesseract_config.output_path),
            ]
        )

    if unit_tesseract_config.test_with_random_inputs:
        random_input = example_from_json_schema(input_schema)

        result = cli_runner.invoke(
            app,
            [
                "run",
                img_name,
                *mount_args,
                "apply",
                json.dumps(random_input),
                *io_args,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    if unit_tesseract_config.sample_requests:
        for request in unit_tesseract_config.sample_requests:
            print(f"Running request: {request}")
            cli_cmd = request.endpoint.replace("_", "-")

            if cli_cmd == "check-gradients":
                args = [
                    "run",
                    img_name,
                    cli_cmd,
                    json.dumps(request.payload),
                ]
            else:
                args = [
                    "run",
                    img_name,
                    *mount_args,
                    cli_cmd,
                    json.dumps(request.payload),
                    *io_args,
                    "--output-format",
                    request.output_format,
                ]

            result = cli_runner.invoke(app, args, env={"TERM": "dumb"})
            if request.expected_status_code == 200:
                print_debug_info(result)
                assert result.exit_code == 0, result.exception
                if cli_cmd in ("check-gradients",):
                    # Result is text
                    output = result.stdout
                else:
                    # Result is JSON output
                    output = json_normalize(result.stdout)
            else:
                # Result is an error message
                assert result.exit_code != 0
                assert result.exc_info is not None
                output = "".join(traceback.format_exception(*result.exc_info))

            if request.output_contains_pattern is not None:
                patterns = request.output_contains_pattern
                if isinstance(patterns, str):
                    patterns = [patterns]

                for pattern in patterns:
                    assert pattern in output

            if request.output_contains_array is not None:
                array = request.output_contains_array
                output_json = json.loads(output)
                assert_contains_array_allclose(output_json, array)

    # Stage 3: Test HTTP server
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            img_name,
            "-p",
            free_port,
            *mount_args,
            *io_args,
        ],
        catch_exceptions=False,
    )

    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    serve_meta = json.loads(run_res.stdout)
    container_name = serve_meta["container_name"]
    docker_cleanup["containers"].append(container_name)

    # Now test server (send requests and validate outputs)
    response = requests.get(f"http://localhost:{free_port}/openapi.json")
    assert response.status_code == 200
    openapi_schema = response.json()
    input_schema = _input_schema_from_openapi(openapi_schema)

    if unit_tesseract_config.test_with_random_inputs:
        payload_from_schema = example_from_json_schema(input_schema)
        response = requests.post(
            f"http://localhost:{free_port}/apply", json=payload_from_schema
        )
        assert response.status_code == 200, response.text

    sample_requests = unit_tesseract_config.sample_requests or []
    for request in sample_requests:
        if request.endpoint in ("check-gradients",):
            # Not supported in HTTP mode
            continue

        headers = {
            "Accept": f"application/{request.output_format}",
        }
        response = requests.post(
            f"http://localhost:{free_port}/{request.endpoint}",
            json=request.payload,
            headers=headers,
        )
        output = response.text
        assert response.status_code == request.expected_status_code, output

        if request.output_contains_pattern is not None:
            patterns = request.output_contains_pattern
            if isinstance(patterns, str):
                patterns = [patterns]

            for pattern in patterns:
                assert pattern in output

        if request.output_contains_array is not None:
            array = request.output_contains_array
            output_json = json.loads(output)
            assert_contains_array_allclose(output_json, array)
