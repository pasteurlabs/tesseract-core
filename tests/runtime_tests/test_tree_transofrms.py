# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel
from tesseract_core.runtime.tree_transforms import path_to_index_op, get_at_path


class TestPathToIndexOp:
    """Test cases for path_to_index_op function."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Basic sequence indexing
            ("[0]", ("seq", 0)),
            ("[123]", ("seq", 123)),
            # Basic dictionary indexing
            ("{key}", ("dict", "key")),
            ("{bar_baz}", ("dict", "bar_baz")),
            # Dictionary indexing with special characters
            ("{key with spaces}", ("dict", "key with spaces")),
            ("{key-with-dashes}", ("dict", "key-with-dashes")),
            ("{key.with.dots}", ("dict", "key.with.dots")),
            ("{123}", ("dict", "123")),
            ("{a123}", ("dict", "a123")),
            ("{ }", ("dict", " ")),
            # Basic attribute access
            ("attr", ("getattr", "attr")),
            # Attribute access with underscores and numbers
            ("attr123", ("getattr", "attr123")),
            ("attr_123", ("getattr", "attr_123")),
            ("_private", ("getattr", "_private")),
            # Unicode letters are correct in attribute names
            ("café", ("getattr", "café")),
            ("фыя", ("getattr", "фыя")),
            # All Unicode characters are correct in dictionary keys
            ("{café}", ("dict", "café")),
            ("{🔑key}", ("dict", "🔑key")),
        ],
    )
    def test_valid_paths(self, path, expected):
        """Test valid path patterns return expected results."""
        assert path_to_index_op(path) == expected

    @pytest.mark.parametrize(
        "invalid_path",
        [
            "",
            "[]",
            "{}",
            "[pi]",
            "[1.5]",
            # Negative indices not supported
            "[-1]",
            # Invalid attribute names
            "attr-with-dashes",
            "attr with spaces",
            "123startswithadigit",
            "attr!",
            "🚀rocket",
            "done✅",
            "-starts-with-dash",
            " starts-with-space",
            "!starts-with-symbol",
        ],
    )
    def test_invalid_paths(self, invalid_path):
        """Test invalid path patterns raise ValueError."""
        with pytest.raises(ValueError, match="Invalid path"):
            path_to_index_op(invalid_path)


class TestGetAtPath:
    """Test cases for get_at_path function."""

    @pytest.fixture
    def sample_tree(self):
        """Create a sample nested tree for testing."""
        class SimpleObj:
            def __init__(self):
                self.attr = "object_attribute"
                self.nested = {"key": "nested_value"}
                self.list_data = [1, 2, {"inner": "list_dict"}]

        class TestModel(BaseModel):
            field1: str
            field2: dict
            
        model = TestModel(field1="pydantic_value", field2={"nested": "pydantic_nested"})

        return {
            "root_key": "root_value",
            "numbers": [10, 20, 30],
            "nested_dict": {
                "level1": {
                    "level2": "deep_value"
                }
            },
            "mixed_list": [
                {"item": "first"},
                {"item": "second", "extra": [100, 200]}
            ],
            "object": SimpleObj(),
            "pydantic_model": model,
            "dict_with_special_keys": {
                "key with spaces": "space_value",
                "key-with-dashes": "dash_value",
                "123": "numeric_key"
            }
        }
    
    def test_empty_path_returns_root(self, sample_tree):
        """Test that empty path returns the entire tree."""
        result = get_at_path(sample_tree, "")
        assert result is sample_tree

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Simple dictionary access
            ("root_key", "root_value"),
            # List indexing
            ("numbers.[0]", 10),
            ("numbers.[1]", 20),
            ("numbers.[2]", 30),
            # Dictionary access
            ("nested_dict.{level1}", {"level2": "deep_value"}),
            ("nested_dict.{level1}.{level2}", "deep_value"),
            # Attribute syntax also works for dicts
            ("nested_dict.level1", {"level2": "deep_value"}),
            ("nested_dict.level1.level2", "deep_value"),
            # Mixed list and dict access
            ("mixed_list.[0].item", "first"),
            ("mixed_list.[1].item", "second"),
            ("mixed_list.[1].extra.[0]", 100),
            ("mixed_list.[1].extra.[1]", 200),
            # Object attribute access
            ("object.attr", "object_attribute"),
            ("object.nested.key", "nested_value"),
            ("object.list_data.[0]", 1),
            ("object.list_data.[2].inner", "list_dict"),
            # Dictionary keys with special characters
            ("dict_with_special_keys.{key with spaces}", "space_value"),
            ("dict_with_special_keys.{key-with-dashes}", "dash_value"),
            ("dict_with_special_keys.{123}", "numeric_key"),
            # Pydantic model field access
            ("pydantic_model.field1", "pydantic_value"),
            ("pydantic_model.field2.nested", "pydantic_nested"),
        ],
    )
    def test_valid_paths(self, sample_tree, path, expected):
        """Test get_at_path with valid paths."""
        result = get_at_path(sample_tree, path)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_path,expected_error",
        [
            # Non-existent keys in dictionary
            ("nonexistent", KeyError),
            # String has no attribute
            ("root_key.nonexistent", AttributeError),
            # Nested dict access
            ("nested_dict.nonexistent", KeyError),
            # Invalid list indices
            ("numbers.[99]", IndexError),
            # Negative indices not supported
            ("numbers.[-1]", ValueError),
            # Sequence doesn't have an attribute
            ("numbers.invalid_syntax", AttributeError),
            # Object attribute that doesn't exist
            ("object.nonexistent_attr", AttributeError),
        ],
    )
    def test_invalid_paths(self, sample_tree, invalid_path, expected_error):
        """Test get_at_path with invalid paths raises appropriate errors."""
        with pytest.raises(expected_error):
            get_at_path(sample_tree, invalid_path)