# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test this SDK and runtime against released ones.

See docs/content/reference/version-compatibility.md for the policy this checks.

Skipped unless versions are given, e.g. ``pytest tests/compat_tests
--compat-version 1.13.0``. Each test runs in one direction: this SDK against an
old runtime, or an old SDK against this runtime. Results are compared with
those of the dummy Tesseract called in-process.
"""

from contextlib import nullcontext

import numpy as np
import pytest
import requests

from tesseract_core import Tesseract

ARGS = {
    "inputs": {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [0.5, -1.0, 2.0, 0.0],
        "s": 2.5,
        "normalize": True,
    },
    "abstract_inputs": {
        "a": {"shape": [4], "dtype": "float32"},
        "b": {"shape": [4], "dtype": "float32"},
    },
    "tangent_vector": {"a": [1.0, 0.0, -1.0, 2.0]},
    "cotangent_vector": {"result": [0.0, 1.0, 1.0, -1.0]},
    "invalid_inputs": {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]},
}

ENDPOINTS = (
    "apply",
    "abstract_eval",
    "jacobian",
    "jacobian_vector_product",
    "vector_jacobian_product",
)


def _to_arrays(tree):
    if isinstance(tree, dict):
        return {k: _to_arrays(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return np.asarray(tree, dtype=np.float32)
    return tree


def _call(tess: Tesseract, endpoint: str) -> dict:
    """Call ``endpoint`` as old_sdk_client.py does."""
    inputs = _to_arrays(ARGS["inputs"])
    if endpoint == "apply":
        return tess.apply(inputs)
    if endpoint == "abstract_eval":
        return tess.abstract_eval(ARGS["abstract_inputs"])
    if endpoint == "jacobian":
        return tess.jacobian(inputs, ["a", "b"], ["result"])
    if endpoint == "jacobian_vector_product":
        return tess.jacobian_vector_product(
            inputs, ["a"], ["result"], _to_arrays(ARGS["tangent_vector"])
        )
    if endpoint == "vector_jacobian_product":
        return tess.vector_jacobian_product(
            inputs, ["a"], ["result"], _to_arrays(ARGS["cotangent_vector"])
        )
    raise ValueError(endpoint)


def _assert_tree_close(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_close(actual[key], expected[key])
    elif isinstance(expected, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)


def _version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:2])


@pytest.fixture(scope="module")
def reference(compat_api_path) -> dict:
    """Results of the dummy Tesseract called in-process with this runtime."""
    tess = Tesseract.from_tesseract_api(compat_api_path)
    return {endpoint: _call(tess, endpoint) for endpoint in ENDPOINTS}


# This SDK against an old runtime


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_sdk_endpoints(old_runtime, reference, endpoint):
    url, _ = old_runtime
    with Tesseract.from_url(url) as tess:
        assert tess.health()["status"] == "ok"
        _assert_tree_close(_call(tess, endpoint), reference[endpoint])


@pytest.mark.parametrize("output_format", ["json", "json+base64", "json+binref"])
def test_sdk_output_formats(old_runtime, reference, output_format):
    url, output_path = old_runtime
    with Tesseract.from_url(url, server_output_path=output_path) as tess:
        view = tess.with_encoding(output_format=output_format)
        for endpoint in ("apply", "jacobian"):
            _assert_tree_close(_call(view, endpoint), reference[endpoint])


@pytest.mark.parametrize("compression", ["none", "lz4"])
def test_sdk_compression(old_runtime, reference, compression):
    url, _ = old_runtime
    with Tesseract.from_url(url) as tess:
        if tess.server_capabilities.compressions is None:
            # Runtimes before 1.15 do not advertise compressions
            expect_warning = pytest.warns(UserWarning, match="advertise")
        else:
            expect_warning = nullcontext()
        with expect_warning:
            view = tess.with_encoding(
                output_format="json+base64", compression=compression
            )
            _assert_tree_close(_call(view, "apply"), reference["apply"])


def test_runtime_honours_compression(old_runtime, compat_version):
    """The Accept header's compression parameter reaches the response.

    Runtime 1.13 ignores it, as documented.
    """
    url, _ = old_runtime
    response = requests.post(
        f"{url}/apply",
        json={"inputs": ARGS["inputs"]},
        headers={"Accept": "application/json+base64; compression=lz4"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["result"]["data"]
    if _version(compat_version) >= (1, 14):
        assert data.get("compression") == "lz4"
    else:
        assert data.get("compression") is None


def test_sdk_gpu_transport_none(old_runtime, reference):
    url, _ = old_runtime
    with Tesseract.from_url(url) as tess:
        assert tess.resolve_gpu_transport() == "none"
        view = tess.with_encoding(gpu_transport="none")
        _assert_tree_close(_call(view, "apply"), reference["apply"])


def test_sdk_refuses_gpu_transport_before_sending(old_runtime):
    """No supported old runtime on a CPU server can pass GPU arrays by reference."""
    url, _ = old_runtime
    with Tesseract.from_url(url) as tess, pytest.raises(ValueError, match="cuda_ipc"):
        tess.with_encoding(gpu_transport="cuda_ipc")


def test_sdk_invalid_inputs(old_runtime):
    url, _ = old_runtime
    with (
        Tesseract.from_url(url) as tess,
        pytest.raises(ValueError, match="must have the same shape"),
    ):
        tess.apply(_to_arrays(ARGS["invalid_inputs"]))


# An old SDK against this runtime


@pytest.mark.parametrize(
    "output_format,compression",
    [
        ("json", None),
        ("json+base64", None),
        ("json+base64", "lz4"),
        ("json+binref", None),
        ("json+binref", "lz4"),
    ],
)
def test_old_sdk(
    run_old_client,
    compat_version,
    serve_in_subprocess,
    compat_api_path,
    free_port,
    tmp_path,
    reference,
    output_format,
    compression,
):
    env = {
        "TESSERACT_OUTPUT_FORMAT": output_format,
        "TESSERACT_OUTPUT_PATH": str(tmp_path),
    }
    if compression is not None:
        env["TESSERACT_COMPRESSION"] = compression
    with serve_in_subprocess(compat_api_path, free_port, env=env) as url:
        results = run_old_client(url, tmp_path, ARGS)

    assert results.pop("sdk_version") == compat_version
    assert results.pop("health")["status"] == "ok"
    assert "must have the same shape" in results.pop("invalid_inputs_error")
    _assert_tree_close(results, reference)
