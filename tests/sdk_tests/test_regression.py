# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for regression testing utilities."""

import numpy as np
import pytest

from tesseract_core.runtime.regression import _validate_tree_structure


def test_validate_tree_structure_passes():
    """Test cases where structure validation should pass."""
    # Empty containers
    _validate_tree_structure({}, {})
    _validate_tree_structure([], [])

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
        np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)
    )
    _validate_tree_structure(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
    )


def test_validate_tree_structure_type_mismatches():
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


def test_validate_tree_structure_dict_key_mismatches():
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


def test_validate_tree_structure_sequence_length_mismatches():
    """Test sequence length mismatch detection."""
    with pytest.raises(AssertionError, match="Mismatch in length"):
        _validate_tree_structure([1, 2], [1, 2, 3])

    # Nested length mismatches
    with pytest.raises(AssertionError, match="Mismatch in length"):
        _validate_tree_structure({"results": [1, 2]}, {"results": [1, 2, 3]})


def test_validate_tree_structure_numpy_mismatches():
    """Test numpy array shape and dtype mismatch detection."""
    # Shape mismatches
    with pytest.raises(AssertionError, match="Shape mismatch"):
        _validate_tree_structure(np.array([1, 2, 3]), np.array([[1, 2, 3]]))

    with pytest.raises(AssertionError, match="Shape mismatch"):
        _validate_tree_structure(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3, 4]))

    # Dtype mismatches
    with pytest.raises(AssertionError, match="dtype mismatch"):
        _validate_tree_structure(
            np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 3], dtype=np.float64)
        )

    with pytest.raises(AssertionError, match="dtype mismatch"):
        _validate_tree_structure(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_validate_tree_structure_path_tracking():
    """Test that error messages contain correct path information."""
    # Deep nesting path
    with pytest.raises(AssertionError, match=r"\.foo\.bar\[2\]"):
        _validate_tree_structure(
            {"foo": {"bar": [1, 2, {"x": 1}]}}, {"foo": {"bar": [1, 2, {"x": "1"}]}}
        )

    # List index path
    with pytest.raises(AssertionError, match=r"\[1\]\.b"):
        _validate_tree_structure([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": "2"}])
