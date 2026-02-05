# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for regression testing utilities.

This module tests the regression testing framework via the public API (regress_test_case)
and includes focused unit tests for implementation details that affect debugging experience.

Structure:
- Section 1: Model definitions (NestedModel, SubModel, SubRootModel - same as test_schema_generation.py)
- Section 2: Helper functions (make_valid_nested_output, make_apply_payload)
- Section 3: Public API tests via regress_test_case (separate functions, with error assertions)
- Section 4: TestSpec validation tests
- Section 5: Unit tests for _validate_tree_structure internals
"""

import numpy as np
import pytest
from pydantic import BaseModel, RootModel

from tesseract_core.runtime import Array, Differentiable, Float32, Float64, Int64, UInt8
from tesseract_core.runtime.core import create_endpoints
from tesseract_core.runtime.experimental import LazySequence
from tesseract_core.runtime.testing.regression import TestSpec as RegressionTestSpec
from tesseract_core.runtime.testing.regression import (
    _array_discrepancy_msg,
    _validate_tree_structure,
    regress_test_case,
)

# =============================================================================
# Section 1: Model definitions (consistent with test_schema_generation.py)
# =============================================================================


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


# =============================================================================
# Section 2: Helper functions and fixtures
# =============================================================================


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


# =============================================================================
# Section 3: Public API tests via regress_test_case
# =============================================================================


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
    assert "Unexpected keys:" in result.message
    assert "extra_key" in result.message


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
    assert "Expected: 2" in result.message
    assert "Obtained: 3" in result.message


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
    assert "testdiffarr" in result.message
    assert "Shape:" in result.message


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
    assert "testfoo.[0].{foo}" in result.message
    assert "Expected:" in result.message
    assert "Obtained:" in result.message


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
    assert "testrootmodel" in result.message


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
    assert "testlazysequence" in result.message


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
    assert "Type mismatch at testfoo" in result.message


def test_nested_path_in_error_message(complex_tesseract_module, monkeypatch):
    """Verify error messages include full path for debugging nested structures."""
    expected_output = make_valid_nested_output()
    actual_output = make_valid_nested_output()
    # Mutate deeply nested field: testbar -> {key1} -> bar -> [0] -> array value
    actual_output["testbar"]["key1"]["bar"][0] = np.array([999, 888], dtype=np.int64)

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
    assert "testbar.{key1}.{bar}.[0]" in result.message


# Tests for regress_test_case function (basic endpoint tests)


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


# =============================================================================
# Section 4: TestSpec validation tests
# =============================================================================


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


# =============================================================================
# Section 5: Unit tests for _validate_tree_structure internals
# =============================================================================


class TestValidateTreeStructureInternals:
    """Unit tests for implementation details that affect debugging experience.

    These tests directly test _validate_tree_structure to verify:
    - Path formatting in error messages
    - Leaf collection structure
    - Empty container handling
    - Dict vs model attribute distinction
    """

    def test_path_formatting_dict_keys(self):
        """Verify dict keys are formatted with braces: {key}."""
        with pytest.raises(AssertionError, match=r"\{foo\}"):
            _validate_tree_structure(
                {"foo": "value"},
                {"foo": 123},  # Type mismatch to trigger error
            )

    def test_path_formatting_list_indices(self):
        """Verify list indices are formatted with brackets: [idx]."""
        with pytest.raises(AssertionError, match=r"\[1\]"):
            _validate_tree_structure(
                [1, "wrong"],
                [1, 2],  # Type mismatch at index 1
            )

    def test_path_formatting_nested_structure(self):
        """Verify nested paths combine dict keys, list indices, and model attrs."""
        from tesseract_core.runtime.schema_generation import DICT_INDEX_SENTINEL

        # Path pattern indicating a dict at top level
        path_patterns = {
            (DICT_INDEX_SENTINEL, "inner", "value"): float,
        }

        with pytest.raises(AssertionError) as exc_info:
            _validate_tree_structure(
                {"key": {"inner": {"value": "wrong"}}},
                {"key": {"inner": {"value": 1.0}}},
                path_patterns,
            )

        # Should show dict key in braces
        assert "{key}" in str(exc_info.value)

    def test_leaf_collection_returns_correct_structure(self):
        """Verify leaves dict structure for algorithm correctness."""
        leaves = _validate_tree_structure(
            {"scalar": 42, "nested": {"inner": 100}},
            {"scalar": 42, "nested": {"inner": 100}},
        )

        # Leaves should contain path tuples mapping to (obtained, expected) tuples
        assert ("{scalar}",) in leaves
        assert leaves[("{scalar}",)] == (42, 42)
        assert ("{nested}", "{inner}") in leaves
        assert leaves[("{nested}", "{inner}")] == (100, 100)

    def test_leaf_collection_with_lists(self):
        """Verify list indices appear in leaf paths."""
        leaves = _validate_tree_structure(
            [10, 20, 30],
            [10, 20, 30],
        )

        assert ("[0]",) in leaves
        assert ("[1]",) in leaves
        assert ("[2]",) in leaves
        assert leaves[("[1]",)] == (20, 20)

    def test_empty_dict_returns_empty_leaves(self):
        """Verify empty containers return empty dicts without raising."""
        result = _validate_tree_structure({}, {})
        assert result == {}

    def test_empty_list_returns_empty_leaves(self):
        """Verify empty lists return empty dicts without raising."""
        result = _validate_tree_structure([], [])
        assert result == {}

    def test_numpy_array_shape_in_path(self):
        """Verify arrays are treated as leaves and shape mismatches report path."""
        with pytest.raises(AssertionError, match=r"Shape mismatch.*\{arr\}"):
            _validate_tree_structure(
                {"arr": np.array([1, 2, 3])},
                {"arr": np.array([1, 2])},
            )

    def test_numpy_array_dtype_mismatch(self):
        """Verify dtype mismatches are caught and reported."""
        with pytest.raises(AssertionError, match=r"dtype mismatch"):
            _validate_tree_structure(
                {"arr": np.array([1, 2], dtype=np.int32)},
                {"arr": np.array([1, 2], dtype=np.float64)},
            )

    def test_type_mismatch_error_format(self):
        """Verify type mismatch errors include expected and obtained types."""
        with pytest.raises(AssertionError) as exc_info:
            _validate_tree_structure(
                {"value": "string"},
                {"value": 123},
            )

        error_msg = str(exc_info.value)
        assert "Type mismatch" in error_msg
        assert "Expected:" in error_msg
        assert "Obtained:" in error_msg
        assert "int" in error_msg
        assert "str" in error_msg

    def test_dict_key_mismatch_error_format(self):
        """Verify key mismatch errors include missing and unexpected keys."""
        with pytest.raises(AssertionError) as exc_info:
            _validate_tree_structure(
                {"a": 1, "b": 2, "extra": 3},
                {"a": 1, "b": 2, "missing": 4},
            )

        error_msg = str(exc_info.value)
        assert "Key mismatch" in error_msg
        assert "Missing" in error_msg
        assert "Unexpected" in error_msg
        assert "missing" in error_msg
        assert "extra" in error_msg

    def test_list_length_mismatch_error_format(self):
        """Verify list length errors include expected and obtained lengths."""
        with pytest.raises(AssertionError) as exc_info:
            _validate_tree_structure(
                [1, 2, 3, 4],
                [1, 2],
            )

        error_msg = str(exc_info.value)
        assert "Mismatch in length" in error_msg
        assert "Expected: 2" in error_msg
        assert "Obtained: 4" in error_msg


class TestArrayDiscrepancyMsg:
    """Unit tests for statistics formatting in error messages."""

    def test_statistics_format_max_mean_median(self):
        """Verify max/mean/median statistics format for debugging."""
        diff_ids = np.array([[0], [1]])
        obtained = np.array([10.0, 20.0])
        expected = np.array([1.0, 2.0])

        msg = _array_discrepancy_msg(
            size=10,
            shape=(10,),
            diff_ids=diff_ids,
            obtained_array=obtained,
            expected_array=expected,
            threshold=100,
        )

        assert "Max:" in msg
        assert "Mean:" in msg
        assert "Median:" in msg

    def test_percentage_calculation(self):
        """Verify percentage of differing elements is calculated correctly."""
        diff_ids = np.array([[0], [1]])  # 2 differences
        obtained = np.array([10.0, 20.0])
        expected = np.array([1.0, 2.0])

        msg = _array_discrepancy_msg(
            size=4,  # 4 total elements
            shape=(4,),
            diff_ids=diff_ids,
            obtained_array=obtained,
            expected_array=expected,
            threshold=100,
        )

        # 2 / 4 = 50%
        assert "2 / 4" in msg
        assert "50.0%" in msg

    def test_individual_errors_format(self):
        """Verify individual error table format."""
        diff_ids = np.array([[0], [2]])
        obtained = np.array([5.0, 15.0])
        expected = np.array([1.0, 10.0])

        msg = _array_discrepancy_msg(
            size=5,
            shape=(5,),
            diff_ids=diff_ids,
            obtained_array=obtained,
            expected_array=expected,
            threshold=100,
        )

        # Table header
        assert "Index" in msg
        assert "Obtained" in msg
        assert "Expected" in msg
        assert "Difference" in msg

    def test_threshold_limits_output(self):
        """Verify threshold limits number of displayed differences."""
        # More differences than threshold
        num_diffs = 150
        diff_ids = np.array([[i] for i in range(num_diffs)])
        obtained = np.arange(num_diffs, dtype=np.float64)
        expected = np.zeros(num_diffs, dtype=np.float64)

        msg = _array_discrepancy_msg(
            size=200,
            shape=(200,),
            diff_ids=diff_ids,
            obtained_array=obtained,
            expected_array=expected,
            threshold=100,  # Limit to 100
        )

        assert "Only showing first 100 mismatches" in msg

    def test_shape_in_message(self):
        """Verify shape is included in the message."""
        diff_ids = np.array([[0, 0], [1, 1]])
        obtained = np.array([10.0, 20.0])
        expected = np.array([1.0, 2.0])

        msg = _array_discrepancy_msg(
            size=9,
            shape=(3, 3),
            diff_ids=diff_ids,
            obtained_array=obtained,
            expected_array=expected,
            threshold=100,
        )

        assert "Shape: (3, 3)" in msg

    def test_relative_errors_when_nonzero_expected(self):
        """Verify relative error statistics when expected values are nonzero."""
        diff_ids = np.array([[0], [1]])
        obtained = np.array([2.0, 4.0])
        expected = np.array([1.0, 2.0])

        msg = _array_discrepancy_msg(
            size=2,
            shape=(2,),
            diff_ids=diff_ids,
            obtained_array=obtained,
            expected_array=expected,
            threshold=100,
        )

        # Should have relative error stats
        assert "abs(obtained - expected) / abs(expected)" in msg

    def test_zero_expected_skips_relative_errors(self):
        """Verify relative errors are skipped when all expected values are zero."""
        diff_ids = np.array([[0], [1]])
        obtained = np.array([1.0, 2.0])
        expected = np.array([0.0, 0.0])

        msg = _array_discrepancy_msg(
            size=2,
            shape=(2,),
            diff_ids=diff_ids,
            obtained_array=obtained,
            expected_array=expected,
            threshold=100,
        )

        assert "all expected values are zero" in msg
