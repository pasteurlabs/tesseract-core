# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for regression testing utilities."""

import json

import numpy as np
import pytest

from tesseract_core.runtime.testing.regression import (
    _array_discrepancy_msg,
    _validate_tree_structure,
    iter_regression_tests,
    regress_test_case,
)


class TestValidateTreeStructure:
    """Tests for _validate_tree_structure function."""

    def test_passes(self):
        """Test cases where structure validation should pass."""
        # Empty containers (these return empty leaf dicts, but don't raise)
        result = _validate_tree_structure({}, {})
        assert result == {}
        result = _validate_tree_structure([], [])
        assert result == {}

        # Simple containers
        _validate_tree_structure({"a": 1}, {"a": 2})
        _validate_tree_structure([1, 2, 3], [4, 5, 6])
        _validate_tree_structure((1, 2, 3), (4, 5, 6))

        # Nested dicts
        _validate_tree_structure({"a": {"b": 1}}, {"a": {"b": 2}})

        # Nested list
        _validate_tree_structure([[1, 2], [3, 4]], [[5, 6], [7, 8]])

        # Mixed nesting
        _validate_tree_structure(
            {"results": [1, 2, 3], "meta": {"count": 3}},
            {"results": [4, 5, 6], "meta": {"count": 6}},
        )
        _validate_tree_structure([{"a": 1}, {"b": 2}], [{"a": 3}, {"b": 4}])

        # Numpy arrays - same shape and dtype
        _validate_tree_structure(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([4, 5, 6], dtype=np.int64),
        )
        _validate_tree_structure(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        )

    def test_type_mismatches(self):
        """Test type mismatch detection."""
        # Root level type mismatches
        with pytest.raises(AssertionError, match="Type mismatch"):
            _validate_tree_structure({}, [])

        with pytest.raises(AssertionError, match="Type mismatch"):
            _validate_tree_structure(1, "1")

        with pytest.raises(AssertionError, match="Type mismatch"):
            _validate_tree_structure([1, 2], 1)

        # Nested type mismatches
        with pytest.raises(AssertionError, match="Type mismatch"):
            _validate_tree_structure({"a": 1}, {"a": "1"})

        with pytest.raises(AssertionError, match="Type mismatch"):
            _validate_tree_structure({"x": [1, 2]}, {"x": {"y": 1}})

    def test_dict_key_mismatches(self):
        """Test dictionary key mismatch detection."""
        # Missing keys
        with pytest.raises(AssertionError, match="Key mismatch"):
            _validate_tree_structure({"a": 1}, {"a": 1, "b": 2})

        # Extra keys
        with pytest.raises(AssertionError, match="Key mismatch"):
            _validate_tree_structure({"a": 1, "b": 2}, {"a": 1})

        # Different keys entirely
        with pytest.raises(AssertionError, match="Key mismatch"):
            _validate_tree_structure({"a": 1}, {"c": 1})

        # Nested key mismatches
        with pytest.raises(AssertionError, match="Key mismatch"):
            _validate_tree_structure({"x": {"a": 1}}, {"x": {"a": 1, "b": 2}})

    def test_sequence_length_mismatches(self):
        """Test sequence length mismatch detection."""
        with pytest.raises(AssertionError, match="Mismatch in length"):
            _validate_tree_structure([1, 2], [1, 2, 3])

        # Nested length mismatches
        with pytest.raises(AssertionError, match="Mismatch in length"):
            _validate_tree_structure({"results": [1, 2]}, {"results": [1, 2, 3]})

    def test_numpy_mismatches(self):
        """Test numpy array shape and dtype mismatch detection."""
        # Shape mismatches
        with pytest.raises(AssertionError, match="Shape mismatch"):
            _validate_tree_structure(np.array([1, 2, 3]), np.array([[1, 2, 3]]))

        with pytest.raises(AssertionError, match="Shape mismatch"):
            _validate_tree_structure(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3, 4]))

        # Dtype mismatches
        with pytest.raises(AssertionError, match="dtype mismatch"):
            _validate_tree_structure(
                np.array([1, 2, 3], dtype=np.int64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError, match="dtype mismatch"):
            _validate_tree_structure(
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([1.0, 2.0], dtype=np.float64),
            )

    def test_path_tracking(self):
        """Test that error messages contain correct path information."""
        # Deep nesting path - dicts are formatted as {key} when path_patterns=None
        with pytest.raises(AssertionError, match=r"\{foo\}.*\{bar\}.*\[2\].*\{x\}"):
            _validate_tree_structure(
                {"foo": {"bar": [1, 2, {"x": 1}]}},
                {"foo": {"bar": [1, 2, {"x": "1"}]}},
            )

        # List index path
        with pytest.raises(AssertionError, match=r"\[1\].*\{b\}"):
            _validate_tree_structure([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": "2"}])

    def test_dict_vs_model_formatting(self):
        """Test that schema patterns distinguish dict keys from model attributes."""
        from tesseract_core.runtime.schema_generation import DICT_INDEX_SENTINEL

        # Schema says this is a dict[str, int]
        path_patterns = {(DICT_INDEX_SENTINEL,): int}
        tree1 = {"foo": 1, "bar": 2}
        tree2 = {"foo": 10, "bar": 20}
        leaves = _validate_tree_structure(tree1, tree2, path_patterns)

        # Dict keys formatted as {key}
        assert ("{foo}",) in leaves
        assert ("{bar}",) in leaves

        # Schema says this is a model with attribute "foo"
        path_patterns = {("foo",): int}
        tree1 = {"foo": 1}
        tree2 = {"foo": 10}
        leaves = _validate_tree_structure(tree1, tree2, path_patterns)

        # Model attributes formatted without braces
        assert ("foo",) in leaves

    def test_leaf_collection(self):
        """Test that leaf values are collected with correct paths."""
        tree1 = {
            "scalar": 42,
            "array": np.array([1, 2, 3]),
            "nested": {"inner": 3.14},
        }
        tree2 = {
            "scalar": 100,
            "array": np.array([4, 5, 6]),
            "nested": {"inner": 2.71},
        }

        leaves = _validate_tree_structure(tree1, tree2)

        assert len(leaves) == 3
        # Dicts are formatted as {key} when path_patterns=None
        assert ("{scalar}",) in leaves
        assert ("{array}",) in leaves
        assert ("{nested}", "{inner}") in leaves
        assert leaves[("{scalar}",)] == (42, 100)
        assert leaves[("{nested}", "{inner}")] == (3.14, 2.71)


class TestRegressTestCase:
    """Tests for regress_test_case function."""

    def test_success(self, dummy_tesseract_module):
        """Test regress_test_case with matching inputs/outputs."""
        from tesseract_core.runtime.core import create_endpoints

        endpoints = {
            func.__name__: func for func in create_endpoints(dummy_tesseract_module)
        }

        test_spec = {
            "endpoint": "apply",
            "inputs": {
                "inputs": {
                    "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
                    "s": 2,
                }
            },
            "expected_outputs": {
                "result": np.array([6.0, 9.0, 12.0], dtype=np.float32)
            },
        }

        # Should not raise
        regress_test_case(dummy_tesseract_module, endpoints, test_spec)

    def test_value_mismatch(self, dummy_tesseract_module):
        """Test that value mismatches raise AssertionError."""
        from tesseract_core.runtime.core import create_endpoints

        endpoints = {
            func.__name__: func for func in create_endpoints(dummy_tesseract_module)
        }

        test_spec = {
            "endpoint": "apply",
            "inputs": {
                "inputs": {
                    "a": np.array([1.0, 2.0], dtype=np.float32),
                    "b": np.array([4.0, 5.0], dtype=np.float32),
                    "s": 2,
                }
            },
            "expected_outputs": {
                "result": np.array([999.0, 999.0], dtype=np.float32)  # Wrong values
            },
        }

        with pytest.raises(AssertionError, match="Values are not sufficiently close"):
            regress_test_case(dummy_tesseract_module, endpoints, test_spec)

    def test_expected_exception(self, dummy_tesseract_module):
        """Test that expected exceptions pass the test."""
        from tesseract_core.runtime.core import create_endpoints

        endpoints = {
            func.__name__: func for func in create_endpoints(dummy_tesseract_module)
        }

        test_spec = {
            "endpoint": "apply",
            "inputs": {
                "inputs": {
                    "a": np.array([1.0, 2.0]),
                    "b": np.array([4.0]),  # Wrong shape - triggers AssertionError
                    "s": 2,
                }
            },
            "expected_exception": "AssertionError",
        }

        # Should not raise (test passes because exception was expected)
        regress_test_case(dummy_tesseract_module, endpoints, test_spec)

    def test_unexpected_exception(self, dummy_tesseract_module):
        """Test that unexpected exceptions are propagated."""
        from tesseract_core.runtime.core import create_endpoints

        endpoints = {
            func.__name__: func for func in create_endpoints(dummy_tesseract_module)
        }

        test_spec = {
            "endpoint": "apply",
            "inputs": {
                "inputs": {
                    "a": np.array([1.0, 2.0]),
                    "b": np.array([4.0]),  # Wrong shape - triggers AssertionError
                    "s": 2,
                }
            },
            # No expected_exception specified
            "expected_outputs": {"result": np.array([1.0])},
        }

        with pytest.raises(AssertionError):
            regress_test_case(dummy_tesseract_module, endpoints, test_spec)


class TestIterRegressionTests:
    """Tests for iter_regression_tests iterator."""

    def test_multiple_files(self, tmp_path, dummy_tesseract_module):
        """Test iter_regression_tests with multiple test files."""
        # Create test files
        test_dir = tmp_path / "tests"
        test_dir.mkdir()

        # Passing test - JSON lists will be converted to numpy arrays by Pydantic
        test1 = test_dir / "test1.json"
        test1.write_text(
            json.dumps(
                {
                    "endpoint": "apply",
                    "inputs": {
                        "inputs": {
                            "a": [1.0, 2.0],
                            "b": [3.0, 4.0],
                            "s": 1,
                        }
                    },
                    "expected_outputs": {"result": [4.0, 6.0]},
                }
            )
        )

        # Failing test
        test2 = test_dir / "test2.json"
        test2.write_text(
            json.dumps(
                {
                    "endpoint": "apply",
                    "inputs": {
                        "inputs": {
                            "a": [1.0],
                            "b": [2.0],
                            "s": 1,
                        }
                    },
                    "expected_outputs": {
                        "result": [999.0]  # Wrong values
                    },
                }
            )
        )

        results = list(iter_regression_tests(dummy_tesseract_module, test_dir))

        assert len(results) == 2
        assert results[0].status == "passed"
        assert results[1].status == "failed"
        assert "Values are not sufficiently close" in results[1].message


class TestArrayDiscrepancyMsg:
    """Tests for _array_discrepancy_msg formatting."""

    def test_formatting(self):
        """Test array discrepancy message formatting."""
        obtained = np.array([1.0, 2.0, 999.0, 4.0])
        expected = np.array([1.0, 2.0, 3.0, 4.0])

        diff_mask = obtained != expected
        diff_ids = np.nonzero(diff_mask)
        # Convert to list of tuples (each index as a tuple)
        diff_ids_list = list(zip(*diff_ids, strict=False))

        msg = _array_discrepancy_msg(
            size=4,
            shape=(4,),
            diff_ids=diff_ids_list,
            obtained_array=obtained[diff_mask],
            expected_array=expected[diff_mask],
            threshold=100,
        )

        # Check that message contains key information
        assert "Shape: (4,)" in msg
        assert "1 / 4" in msg  # 1 out of 4 elements differ
        assert "25.0%" in msg
