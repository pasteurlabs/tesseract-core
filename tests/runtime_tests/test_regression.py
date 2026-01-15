# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for regression testing utilities."""

import numpy as np
import pytest
from pydantic import BaseModel, RootModel

from tesseract_core.runtime import Array, Differentiable, Float32, Float64, Int64, UInt8
from tesseract_core.runtime.core import create_endpoints
from tesseract_core.runtime.experimental import LazySequence
from tesseract_core.runtime.testing.regression import TestSpec as RegressionTestSpec
from tesseract_core.runtime.testing.regression import regress_test_case


class SubModel(BaseModel):
    foo: Float32
    bar: list[Differentiable[Array[..., Int64]]]


class SubRootModel(RootModel):
    root: Float32


class NestedModel(BaseModel):
    testdiffarr: Differentiable[Array[(5, None), Float64]]
    testfoo: list[SubModel] | None
    testbar: dict[str, SubModel]
    testbaz: Array[(1, 2, 3), UInt8]
    testset: set[int]
    testtuple: tuple[int, str]
    testlazysequence: LazySequence[tuple[str, Differentiable[Array[(None,), Float32]]]]
    testrootmodel: SubRootModel


@pytest.fixture
def complex_tesseract_module(dummy_tesseract_module):
    """Fixture for a complex Tesseract module with nested models."""
    dummy_tesseract_module.OutputSchema = NestedModel
    return dummy_tesseract_module


def make_valid_nested_output():
    """Create a valid NestedModel-compatible output dict."""
    return {
        "testdiffarr": np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            dtype=np.float64,
        ),
        "testfoo": [
            {"foo": np.float32(1.0), "bar": [np.array([1, 2], dtype=np.int64)]},
            {"foo": np.float32(2.0), "bar": [np.array([3, 4], dtype=np.int64)]},
        ],
        "testbar": {
            "key1": {
                "foo": np.float32(10.0),
                "bar": [np.array([10, 20], dtype=np.int64)],
            },
        },
        "testbaz": np.zeros((1, 2, 3), dtype=np.uint8),
        "testset": {1, 2, 3},
        "testtuple": (42, "hello"),
        "testlazysequence": [("item1", np.array([1.0, 2.0], dtype=np.float32))],
        "testrootmodel": np.float32(99.0),
    }


def make_apply_payload():
    """Create a valid apply payload for testing."""
    return {
        "inputs": {
            "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "s": 2,
        }
    }


# Tests for tree structure validation via regress_test_case


def test_matching_structure_passes(complex_tesseract_module, monkeypatch):
    """Test that matching expected outputs pass validation."""
    expected_output = make_valid_nested_output()

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**expected_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "passed"
    assert result.message == ""


def test_missing_dict_key_fails(complex_tesseract_module, monkeypatch):
    """Test that missing dict keys in output structure cause failure."""
    expected_output = make_valid_nested_output()
    actual_output = make_valid_nested_output()
    actual_output["testbar"]["extra_key"] = {
        "foo": np.float32(5.0),
        "bar": [np.array([1], dtype=np.int64)],
    }

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**actual_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Key mismatch at testbar" in result.message


def test_list_length_mismatch_fails(complex_tesseract_module, monkeypatch):
    """Test that list length differences cause failure."""
    expected_output = make_valid_nested_output()
    actual_output = make_valid_nested_output()
    actual_output["testfoo"].append(
        {
            "foo": np.float32(3.0),
            "bar": [np.array([5, 6], dtype=np.int64)],
        }
    )

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**actual_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Mismatch in length of list at testfoo" in result.message


def test_array_value_mismatch_fails(complex_tesseract_module, monkeypatch):
    """Test that array value differences cause failure."""
    expected_output = make_valid_nested_output()
    actual_output = make_valid_nested_output()
    actual_output["testdiffarr"] = np.array(
        [
            [999.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        dtype=np.float64,
    )

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**actual_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Values are not sufficiently close" in result.message


def test_nested_scalar_mismatch_fails(complex_tesseract_module, monkeypatch):
    """Test that nested scalar value differences cause failure."""
    expected_output = make_valid_nested_output()
    actual_output = make_valid_nested_output()
    actual_output["testfoo"][0]["foo"] = np.float32(999.0)

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**actual_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Values are not sufficiently close" in result.message


def test_root_model_value_mismatch_fails(complex_tesseract_module, monkeypatch):
    """Test that RootModel value differences cause failure."""
    expected_output = make_valid_nested_output()
    actual_output = make_valid_nested_output()
    actual_output["testrootmodel"] = np.float32(0.0)

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**actual_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Values are not sufficiently close" in result.message


def test_lazy_sequence_structure(complex_tesseract_module, monkeypatch):
    """Test that LazySequence structures are validated correctly."""
    expected_output = make_valid_nested_output()
    actual_output = make_valid_nested_output()
    actual_output["testlazysequence"] = [
        ("item1", np.array([999.0, 999.0], dtype=np.float32))
    ]

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**actual_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Values are not sufficiently close" in result.message


def test_none_vs_populated_list_fails(complex_tesseract_module, monkeypatch):
    """Test that None vs populated list causes appropriate failure."""
    expected_output = make_valid_nested_output()
    expected_output["testfoo"] = None
    actual_output = make_valid_nested_output()

    def mock_apply(inputs):
        return complex_tesseract_module.OutputSchema(**actual_output)

    monkeypatch.setattr(complex_tesseract_module, "apply", mock_apply)
    endpoints = {
        func.__name__: func for func in create_endpoints(complex_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload=make_apply_payload(),
        expected_outputs=expected_output,
    )

    result = regress_test_case(complex_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"


# Tests for regress_test_case function


def test_regress_test_case_success(dummy_tesseract_module):
    """Test regress_test_case with matching inputs/outputs."""
    endpoints = {
        func.__name__: func for func in create_endpoints(dummy_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload={
            "inputs": {
                "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
                "s": 2,
            }
        },
        expected_outputs={"result": np.array([6.0, 9.0, 12.0], dtype=np.float32)},
    )

    result = regress_test_case(dummy_tesseract_module, endpoints, test_spec)
    assert result.status == "passed"
    assert result.message == ""


def test_regress_test_case_value_mismatch(dummy_tesseract_module):
    """Test that value mismatches return failed status."""
    endpoints = {
        func.__name__: func for func in create_endpoints(dummy_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload={
            "inputs": {
                "a": np.array([1.0, 2.0], dtype=np.float32),
                "b": np.array([4.0, 5.0], dtype=np.float32),
                "s": 2,
            }
        },
        expected_outputs={"result": np.array([999.0, 999.0], dtype=np.float32)},
    )

    result = regress_test_case(dummy_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Values are not sufficiently close" in result.message


def test_regress_test_case_expected_exception(dummy_tesseract_module):
    """Test that expected exceptions pass the test."""
    endpoints = {
        func.__name__: func for func in create_endpoints(dummy_tesseract_module)
    }

    test_spec_dict = {
        "endpoint": "apply",
        "payload": {
            "inputs": {
                "a": np.array([1.0, 2.0]),
                "b": np.array([4.0]),  # Wrong shape - triggers ValidationError
                "s": 2,
            }
        },
        "expected_exception": "pydantic.ValidationError",
    }

    result = regress_test_case(
        dummy_tesseract_module, endpoints, RegressionTestSpec(**test_spec_dict)
    )
    assert result.status == "passed"
    assert result.message == ""

    # Wrong expected exception - should return failed status
    test_spec_dict["expected_exception"] = "IndexError"
    result = regress_test_case(
        dummy_tesseract_module, endpoints, RegressionTestSpec(**test_spec_dict)
    )
    assert result.status == "failed"
    assert "inputs do not conform to InputSchema" in result.message


def test_regress_test_case_unexpected_exception(dummy_tesseract_module):
    """Test that unexpected exceptions return failed status."""
    endpoints = {
        func.__name__: func for func in create_endpoints(dummy_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload={
            "inputs": {
                "a": np.array([1.0, 2.0]),
                "b": np.array([4.0]),  # Wrong shape - triggers ValidationError
                "s": 2,
            }
        },
        expected_outputs={"result": np.array([1.0])},
    )

    result = regress_test_case(dummy_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "inputs do not conform to InputSchema" in result.message


def test_regress_test_case_array_discrepancy_message(dummy_tesseract_module):
    """Test that array discrepancy messages contain useful information."""
    endpoints = {
        func.__name__: func for func in create_endpoints(dummy_tesseract_module)
    }

    test_spec = RegressionTestSpec(
        endpoint="apply",
        payload={
            "inputs": {
                "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "b": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "s": 1,
            }
        },
        # Actual output will be [1, 2, 3, 4], expected has one wrong value
        expected_outputs={"result": np.array([1.0, 2.0, 999.0, 4.0], dtype=np.float32)},
    )

    result = regress_test_case(dummy_tesseract_module, endpoints, test_spec)
    assert result.status == "failed"
    assert "Values are not sufficiently close" in result.message
    assert "Shape: (4,)" in result.message
    assert "1 / 4" in result.message  # 1 out of 4 elements differ
    assert "25.0%" in result.message


# Tests for TestSpec validation


def test_testspec_requires_exactly_one_outcome():
    """Test that TestSpec requires exactly one of expected_outputs or expected_exception."""
    # Both provided - should raise
    with pytest.raises(ValueError, match="Cannot specify both"):
        RegressionTestSpec(
            endpoint="apply",
            payload={"a": 1},
            expected_outputs={"result": 2},
            expected_exception=ValueError,
        )

    # Neither provided - should raise
    with pytest.raises(ValueError, match="Must specify either"):
        RegressionTestSpec(
            endpoint="apply",
            payload={"a": 1},
        )

    # Only expected_outputs - should pass
    spec = RegressionTestSpec(
        endpoint="apply",
        payload={"a": 1},
        expected_outputs={"result": 2},
    )
    assert spec.expected_outputs == {"result": 2}
    assert spec.expected_exception is None

    # Only expected_exception - should pass
    spec = RegressionTestSpec(
        endpoint="apply",
        payload={"a": 1},
        expected_exception=ValueError,
    )
    assert spec.expected_exception is ValueError
    assert spec.expected_outputs is None


def test_testspec_parses_exception_from_string():
    """Test that TestSpec can parse exception types from strings."""
    # String exception name
    spec = RegressionTestSpec(
        endpoint="apply",
        payload={"a": 1},
        expected_exception="ValueError",
    )
    assert spec.expected_exception is ValueError

    # Exception type directly
    spec = RegressionTestSpec(
        endpoint="apply",
        payload={"a": 1},
        expected_exception=ValueError,
    )
    assert spec.expected_exception is ValueError

    # ValidationError (from pydantic)
    spec = RegressionTestSpec(
        endpoint="apply",
        payload={"a": 1},
        expected_exception="pydantic.ValidationError",
    )
    from pydantic import ValidationError

    assert spec.expected_exception is ValidationError


def test_testspec_invalid_exception_type():
    """Test that invalid exception types raise errors."""
    with pytest.raises(
        ValueError,
        match=r"Non-builtin exception 'NonExistentException' must be specified in 'packagename.exceptionname'",
    ):
        RegressionTestSpec(
            endpoint="apply",
            payload={"a": 1},
            expected_exception="NonExistentException",
        )

    with pytest.raises(ValueError, match="Failed to import module"):
        RegressionTestSpec(
            endpoint="apply",
            payload={"a": 1},
            expected_exception="nonexistentpackage.NonExistentException",
        )

    with pytest.raises(ValueError, match="Module 'pydantic' has no attribute"):
        RegressionTestSpec(
            endpoint="apply",
            payload={"a": 1},
            expected_exception="pydantic.NonExistentException",
        )
