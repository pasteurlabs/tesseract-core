# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tesseract_core.runtime.tree_transforms import path_to_index_op


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
