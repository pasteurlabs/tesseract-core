# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import socket
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tesseract_core import Tesseract
from tesseract_core.sdk import engine

expected_endpoints = {
    "apply",
    "jacobian",
    "health",
    "abstract_eval",
    "jacobian_vector_product",
    "vector_jacobian_product",
    "test",
}


def test_available_endpoints(built_image_name):
    with Tesseract.from_image(built_image_name) as vecadd:
        assert set(vecadd.available_endpoints) == expected_endpoints


@pytest.mark.parametrize("output_format", ["json", "json+base64"])
def test_apply(built_image_name, dummy_tesseract_location, free_port, output_format):
    inputs = {"a": [1, 2], "b": [3, 4], "s": 1}

    # Test URL access
    tesseract_url = f"http://localhost:{free_port}"
    served_tesseract, _ = engine.serve(
        built_image_name, port=str(free_port), output_format=output_format
    )
    try:
        vecadd = Tesseract.from_url(tesseract_url)
        out = vecadd.apply(inputs)
    finally:
        engine.teardown(served_tesseract)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Test from_image (context manager)
    with Tesseract.from_image(built_image_name, output_format=output_format) as vecadd:
        out = vecadd.apply(inputs)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Test from_image (serve + teardown)
    vecadd = Tesseract.from_image(built_image_name, output_format=output_format)
    try:
        vecadd.serve()
        out = vecadd.apply(inputs)
    finally:
        vecadd.teardown()

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Test from_tesseract_api
    with Tesseract.from_tesseract_api(
        dummy_tesseract_location / "tesseract_api.py", output_format=output_format
    ) as vecadd:
        out = vecadd.apply(inputs)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))


def test_apply_with_error(built_image_name):
    # pass two inputs with different shapes, which raises a validation error
    inputs = {"a": [1, 2, 3], "b": [3, 4], "s": 1}

    with Tesseract.from_image(built_image_name) as vecadd:
        with pytest.raises(ValidationError) as excinfo:
            vecadd.apply(inputs)

    assert "a and b must have the same shape" in str(excinfo.value)


@pytest.fixture(scope="module")
def served_tesseract_remote(built_image_name):
    # Find a free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    free_port = sock.getsockname()[1]
    sock.close()
    # Serve the Tesseract image
    tesseract_url = f"http://localhost:{free_port}"
    served_tesseract, _ = engine.serve(
        built_image_name, port=str(free_port), debug=True
    )
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
    "endpoint_name",
    sorted(expected_endpoints | {"openapi_schema"}),
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
    elif endpoint_name == "test":
        inputs = {
            "test_spec": {
                "endpoint": "apply",
                "payload": {
                    "inputs": {
                        "a": {
                            "object_type": "array",
                            "shape": [3],
                            "dtype": "int64",
                            "data": {
                                "buffer": "AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA",
                                "encoding": "base64",
                            },
                        },
                        "b": {
                            "object_type": "array",
                            "shape": [3],
                            "dtype": "int64",
                            "data": {
                                "buffer": "BAAAAAAAAAAFAAAAAAAAAAYAAAAAAAAA",
                                "encoding": "base64",
                            },
                        },
                    }
                },
                "expected_outputs": {
                    "result": np.array([7.0, 11.0, 15.0], dtype="float32")
                },
                "atol": 1e-8,
                "rtol": 0.00001,
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
    vecadd = Tesseract.from_url(served_tesseract_remote)
    out = getattr(vecadd, endpoint_name)
    if callable(out):
        out(**inputs)


def test_signature_consistency():
    """Test that from_image and engine.serve have the same signature."""
    allowed_diff = [
        # debug mode is always enabled in from_image
        "debug",
        # setting output format is not meaningful (arrays are decoded automatically)
        "output_format",
        # stream_logs is SDK-only for streaming logs to a callback
        "stream_logs",
        # timeout is a client-side HTTP parameter, not relevant for serve
        "timeout",
        # experimental_binref_pool is a client-side HTTP encode/decode
        # optimization, not relevant for serve
        "experimental_binref_pool",
    ]

    from_image_sig = dict(inspect.signature(Tesseract.from_image).parameters)
    serve_sig = dict(inspect.signature(engine.serve).parameters)

    for param in allowed_diff:
        from_image_sig.pop(param, None)
        serve_sig.pop(param, None)

    assert set(from_image_sig.keys()) == set(serve_sig.keys())

    for key in from_image_sig:
        assert from_image_sig[key].default == serve_sig[key].default, (
            f"Default value mismatch for parameter '{key}': "
            f"{from_image_sig[key].default} != {serve_sig[key].default}"
        )


def test_apply_with_binref_format(built_image_name, tmp_path):
    """Test that json+binref output format works with Tesseract.from_image (Issue #423)."""
    inputs = {"a": [1, 2], "b": [3, 4], "s": 1}
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with Tesseract.from_image(
        built_image_name,
        output_path=output_dir,
        output_format="json+binref",
    ) as vecadd:
        out = vecadd.apply(inputs)

    assert set(out.keys()) == {"result"}
    np.testing.assert_array_equal(out["result"], np.array([4.0, 6.0]))

    # Verify that binary files were created in the output directory (in run subdirectories)
    bin_files = list(output_dir.glob("**/*.bin"))
    assert len(bin_files) > 0, "Expected binary output files to be created"


def test_apply_with_shmem_binref_pool(built_image_name):
    """Test json+binref exchange over /dev/shm with experimental_binref_pool=True end-to-end.

    Exercises the full opt-in fast path: client-side warm-buffer input pool
    (``_BinrefWritePool``) writing into a shared-memory input dir, and the
    zero-copy lazy mmap decode of the (shared-memory) output.
    """
    shm_dir = Path("/dev/shm")
    if not shm_dir.is_dir():
        pytest.skip("/dev/shm is not available on this platform")

    # Inputs must be real arrays (not plain lists) so the binref-input
    # encoding path (gated on ``hasattr(x, "__array__")``) actually triggers.
    inputs = {"a": np.array([1, 2]), "b": np.array([3, 4]), "s": 1}
    expected = np.array([4.0, 6.0])

    with (
        tempfile.TemporaryDirectory(
            prefix="tess_shmem_pool_in_", dir=shm_dir
        ) as input_dir,
        tempfile.TemporaryDirectory(
            prefix="tess_shmem_pool_out_", dir=shm_dir
        ) as output_dir,
    ):
        with Tesseract.from_image(
            built_image_name,
            input_path=input_dir,
            output_path=output_dir,
            output_format="json+binref",
            experimental_binref_pool=True,
        ) as vecadd:
            out = vecadd.apply(inputs)
            assert set(out.keys()) == {"result"}
            np.testing.assert_array_equal(out["result"], expected)

            # The pool's warm slot files live under the shared-memory input
            # dir for the client's lifetime (checked in/out between requests,
            # not deleted per-request like the fresh-file fallback).
            pool_files = list(Path(input_dir).glob("pool_*.bin"))
            assert len(pool_files) > 0, (
                "Expected pooled binref input files under /dev/shm"
            )

            # Run more requests than the pool's default max_slots (4) to
            # exercise both the warm-slot-hit path and the exhausted-pool
            # fallback-to-fresh-file path, and check every result is correct.
            for _ in range(8):
                out = vecadd.apply(inputs)
                np.testing.assert_array_equal(out["result"], expected)
