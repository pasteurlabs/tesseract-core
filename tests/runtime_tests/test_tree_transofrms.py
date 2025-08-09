import pytest
from pydantic import BaseModel

from tesseract_core.runtime.tree_transforms import (
    filter_func,
    flatten_with_paths,
    get_at_path,
    path_to_index_op,
    set_at_path,
)


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


@pytest.fixture
def sample_tree():
    """Create a sample nested tree for get/set at path testing."""

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
        "nested_dict": {"level1": {"level2": "deep_value"}},
        "mixed_list": [{"item": "first"}, {"item": "second", "extra": [100, 200]}],
        "object": SimpleObj(),
        "pydantic_model": model,
        "dict_with_special_keys": {
            "key with spaces": "space_value",
            "key-with-dashes": "dash_value",
            "123": "numeric_key",
        },
    }


class TestGetAtPath:
    """Test cases for get_at_path function."""

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


class TestSetAtPath:
    """Test cases for set_at_path function."""

    def test_set_at_path_creates_deep_copy(self, sample_tree):
        """Test that set_at_path creates a deep copy and doesn't modify original."""
        original_value = sample_tree["root_key"]
        result = set_at_path(sample_tree, {"root_key": "new_value"})

        # Original should be unchanged
        assert sample_tree["root_key"] == original_value
        # Result should have new value
        assert result["root_key"] == "new_value"
        # Should be different objects
        assert result is not sample_tree

    @pytest.mark.parametrize(
        "path_values,verification_path,expected_value",
        [
            # Simple dictionary access
            ({"root_key": "new_root_value"}, "root_key", "new_root_value"),
            # List indexing
            ({"numbers.[0]": 999}, "numbers.[0]", 999),
            ({"numbers.[1]": 888}, "numbers.[1]", 888),
            ({"numbers.[2]": 777}, "numbers.[2]", 777),
            # Dictionary access with {key} syntax
            (
                {"nested_dict.{level1}.{level2}": "new_deep_value"},
                "nested_dict.{level1}.{level2}",
                "new_deep_value",
            ),
            # Attribute syntax for dicts (dict fallback)
            (
                {"nested_dict.level1.level2": "fallback_value"},
                "nested_dict.level1.level2",
                "fallback_value",
            ),
            # Mixed list and dict access
            ({"mixed_list.[0].item": "new_first"}, "mixed_list.[0].item", "new_first"),
            ({"mixed_list.[1].extra.[0]": 555}, "mixed_list.[1].extra.[0]", 555),
            # Object attribute access
            (
                {"object.attr": "new_object_attribute"},
                "object.attr",
                "new_object_attribute",
            ),
            (
                {"object.nested.key": "new_nested_value"},
                "object.nested.key",
                "new_nested_value",
            ),
            (
                {"object.list_data.[2].inner": "new_list_dict"},
                "object.list_data.[2].inner",
                "new_list_dict",
            ),
            # Dictionary keys with special characters
            (
                {"dict_with_special_keys.{key with spaces}": "new_space_value"},
                "dict_with_special_keys.{key with spaces}",
                "new_space_value",
            ),
            (
                {"dict_with_special_keys.{123}": "new_numeric_value"},
                "dict_with_special_keys.{123}",
                "new_numeric_value",
            ),
            # Pydantic model field access
            (
                {"pydantic_model.field1": "new_pydantic_value"},
                "pydantic_model.field1",
                "new_pydantic_value",
            ),
            (
                {"pydantic_model.field2.nested": "new_pydantic_nested"},
                "pydantic_model.field2.nested",
                "new_pydantic_nested",
            ),
        ],
    )
    def test_set_single_path(
        self, sample_tree, path_values, verification_path, expected_value
    ):
        """Test setting values at various single paths."""
        result = set_at_path(sample_tree, path_values)
        actual_value = get_at_path(result, verification_path)
        assert actual_value == expected_value

    def test_set_multiple_paths(self, sample_tree):
        """Test setting multiple paths in a single call."""
        path_values = {
            "root_key": "multi_new_root",
            "numbers.[0]": 111,
            "numbers.[1]": 222,
            "nested_dict.level1.level2": "multi_deep_value",
            "object.attr": "multi_object_attr",
            "mixed_list.[0].item": "multi_first",
        }

        result = set_at_path(sample_tree, path_values)

        # Verify all changes were applied
        assert get_at_path(result, "root_key") == "multi_new_root"
        assert get_at_path(result, "numbers.[0]") == 111
        assert get_at_path(result, "numbers.[1]") == 222
        assert get_at_path(result, "nested_dict.level1.level2") == "multi_deep_value"
        assert get_at_path(result, "object.attr") == "multi_object_attr"
        assert get_at_path(result, "mixed_list.[0].item") == "multi_first"

        # Verify unchanged paths remain the same
        assert get_at_path(result, "numbers.[2]") == get_at_path(
            sample_tree, "numbers.[2]"
        )
        assert get_at_path(result, "mixed_list.[1].item") == get_at_path(
            sample_tree, "mixed_list.[1].item"
        )

    def test_set_complex_nested_structures(self, sample_tree):
        """Test setting complex nested data structures."""
        new_complex_dict = {"new_key": "new_value", "new_list": [1, 2, 3]}
        new_complex_list = [{"nested": "data"}, 42, [7, 8, 9]]

        path_values = {
            "nested_dict.{level1}": new_complex_dict,
            "mixed_list.[1].extra": new_complex_list,
        }

        result = set_at_path(sample_tree, path_values)

        assert get_at_path(result, "nested_dict.{level1}") == new_complex_dict
        assert get_at_path(result, "mixed_list.[1].extra") == new_complex_list
        assert get_at_path(result, "nested_dict.{level1}.new_key") == "new_value"
        assert get_at_path(result, "mixed_list.[1].extra.[0].nested") == "data"

    def test_set_empty_values_dict(self, sample_tree):
        """Test that setting with empty values dict returns unchanged copy."""
        result = set_at_path(sample_tree, {})

        # Should be a deep copy
        assert result is not sample_tree
        # But with same values
        assert get_at_path(result, "root_key") == get_at_path(sample_tree, "root_key")
        assert get_at_path(result, "numbers") == get_at_path(sample_tree, "numbers")

    @pytest.mark.parametrize(
        "invalid_path_values,expected_error",
        [
            # String has no attribute
            ({"root_key.nonexistent": "value"}, AttributeError),
            # Invalid list indices
            ({"numbers.[99]": "value"}, IndexError),
            ({"numbers.[-1]": "value"}, ValueError),
            # Invalid path syntax
            ({"numbers.invalid_syntax": "value"}, AttributeError),
            # Object attribute that doesn't exist
            ({"object.nonexistent_attr": "value"}, AttributeError),
        ],
    )
    def test_set_invalid_paths(self, sample_tree, invalid_path_values, expected_error):
        """Test that setting invalid paths raises appropriate errors."""
        with pytest.raises(expected_error):
            set_at_path(sample_tree, invalid_path_values)

    def test_set_creates_new_dict_keys(self, sample_tree):
        """Test that setting non-existent dict keys creates them (dict fallback)."""
        # Despite the fact that attributes do not exist,
        # the set should succeed and create new keys due to attribute -> dict fallback
        result = set_at_path(sample_tree, {"nonexistent": "new_value"})
        assert get_at_path(result, "nonexistent") == "new_value"

        result = set_at_path(
            sample_tree, {"nested_dict.nonexistent": "nested_new_value"}
        )
        assert get_at_path(result, "nested_dict.nonexistent") == "nested_new_value"

    def test_set_preserves_unmodified_branches(self, sample_tree):
        """Test that setting values preserves unmodified branches of the tree."""
        original_mixed_list = sample_tree["mixed_list"]
        original_object = sample_tree["object"]

        result = set_at_path(sample_tree, {"root_key": "changed_root"})

        # Changed path should be different
        assert result["root_key"] != sample_tree["root_key"]

        # Unmodified branches should be deep copies but with same content
        assert result["mixed_list"] is not original_mixed_list
        assert result["mixed_list"] == original_mixed_list
        assert result["object"] is not original_object
        assert result["object"].attr == original_object.attr


class TestFlattenWithPaths:
    """Test cases for flatten_with_paths function."""

    @pytest.mark.parametrize(
        "include_paths,expected_keys",
        [
            # Single path
            ({"root_key"}, {"root_key"}),
            # Multiple simple paths
            (
                {"root_key", "numbers.[0]", "numbers.[1]"},
                {"root_key", "numbers.[0]", "numbers.[1]"},
            ),
            # Mixed path types
            (
                {
                    "root_key",
                    "numbers.[0]",
                    "nested_dict.{level1}.{level2}",
                    "object.attr",
                },
                {
                    "root_key",
                    "numbers.[0]",
                    "nested_dict.{level1}.{level2}",
                    "object.attr",
                },
            ),
            # Dictionary keys with special characters
            (
                {
                    "dict_with_special_keys.{key with spaces}",
                    "dict_with_special_keys.{123}",
                },
                {
                    "dict_with_special_keys.{key with spaces}",
                    "dict_with_special_keys.{123}",
                },
            ),
            # Pydantic model access
            (
                {"pydantic_model.field1", "pydantic_model.field2.nested"},
                {"pydantic_model.field1", "pydantic_model.field2.nested"},
            ),
            # Complex nested paths
            (
                {
                    "mixed_list.[0].item",
                    "mixed_list.[1].extra.[0]",
                    "object.list_data.[2].inner",
                },
                {
                    "mixed_list.[0].item",
                    "mixed_list.[1].extra.[0]",
                    "object.list_data.[2].inner",
                },
            ),
            # All supported path syntax types at once
            (
                {
                    "root_key",  # Simple dict key
                    "numbers.[1]",  # List index
                    "nested_dict.{level1}",  # Dict key with braces
                    "nested_dict.level1.level2",  # Nested dict with fallback
                    "object.attr",  # Object attribute
                    "dict_with_special_keys.{key with spaces}",  # Special characters
                    "pydantic_model.field1",  # Pydantic field
                },
                {
                    "root_key",
                    "numbers.[1]",
                    "nested_dict.{level1}",
                    "nested_dict.level1.level2",
                    "object.attr",
                    "dict_with_special_keys.{key with spaces}",
                    "pydantic_model.field1",
                },
            ),
        ],
    )
    def test_flatten_valid_paths(self, sample_tree, include_paths, expected_keys):
        """Test flatten_with_paths with valid path sets."""
        result = flatten_with_paths(sample_tree, include_paths)

        # Should return a dictionary with exactly the expected keys
        assert set(result.keys()) == expected_keys

        # Each value should match what get_at_path returns
        for path in include_paths:
            assert result[path] == get_at_path(sample_tree, path)

    def test_flatten_empty_paths(self, sample_tree):
        """Test flatten_with_paths with empty path set."""
        result = flatten_with_paths(sample_tree, set())
        assert result == {}

    def test_flatten_single_path(self, sample_tree):
        """Test flatten_with_paths with a single path."""
        include_paths = {"nested_dict.level1.level2"}
        result = flatten_with_paths(sample_tree, include_paths)

        assert len(result) == 1
        assert result["nested_dict.level1.level2"] == "deep_value"

    @pytest.mark.parametrize(
        "invalid_paths,expected_error",
        [
            # Non-existent keys
            ({"nonexistent"}, KeyError),
            ({"nested_dict.nonexistent"}, KeyError),
            # Invalid list indices
            ({"numbers.[99]"}, IndexError),
            ({"numbers.[-1]"}, ValueError),
            # Invalid path syntax
            ({"numbers.invalid_syntax"}, AttributeError),
            # Object attributes that don't exist
            ({"object.nonexistent_attr"}, AttributeError),
        ],
    )
    def test_flatten_invalid_paths(self, sample_tree, invalid_paths, expected_error):
        """Test flatten_with_paths with invalid paths raises appropriate errors."""
        with pytest.raises(expected_error):
            flatten_with_paths(sample_tree, invalid_paths)

    def test_flatten_mixed_valid_invalid_paths(self, sample_tree):
        """Test that flatten_with_paths fails if any path is invalid, even if others are valid."""
        include_paths = {
            "root_key",  # Valid
            "numbers.[0]",  # Valid
            "nonexistent",  # Invalid
        }

        with pytest.raises(KeyError):
            flatten_with_paths(sample_tree, include_paths)

    def test_flatten_duplicate_paths(self, sample_tree):
        """Test flatten_with_paths behavior with duplicate paths in the set."""
        # Sets naturally handle duplicates, but this tests the behavior explicitly
        include_paths = {"root_key"}
        include_paths_with_duplicate = {"root_key"}  # Set will deduplicate

        result1 = flatten_with_paths(sample_tree, include_paths)
        result2 = flatten_with_paths(sample_tree, include_paths_with_duplicate)

        assert result1 == result2
        assert len(result1) == 1


class TestFilterFunc:
    """Test cases for filter_func function."""

    @pytest.fixture
    def sample_func(self):
        """A simple function that operates on the sample tree structure."""

        def func(tree):
            return {
                "root_key": tree["root_key"].upper(),
                "numbers": [x * 2 for x in tree["numbers"]],
                "computed_sum": sum(tree["numbers"]),
            }

        return func

    def test_filter_func_with_dict_input(self, sample_tree, sample_func):
        """Test filter_func with dictionary input (no input_paths specified)."""
        filtered_func = filter_func(sample_func, sample_tree)

        # Call with dictionary input
        result = filtered_func({"root_key": "hello"})

        # Should return full output since no output_paths specified
        assert result["root_key"] == "HELLO"
        assert result["numbers"] == [20, 40, 60]  # original [10, 20, 30] * 2
        assert result["computed_sum"] == 60

    def test_filter_func_with_positional_args(self, sample_tree, sample_func):
        """Test filter_func with positional arguments."""
        input_paths = ["root_key", "numbers.[0]"]
        filtered_func = filter_func(sample_func, sample_tree, input_paths=input_paths)

        # Call with positional arguments
        result = filtered_func("hello", 99)

        # Should use "hello" for root_key and 99 for numbers.[0]
        assert result["root_key"] == "HELLO"
        assert result["numbers"] == [198, 40, 60]  # [99, 20, 30] * 2
        assert result["computed_sum"] == 149

    def test_filter_func_with_output_filtering(self, sample_tree, sample_func):
        """Test filter_func with output path filtering."""
        output_paths = {"root_key", "computed_sum"}
        filtered_func = filter_func(sample_func, sample_tree, output_paths=output_paths)

        result = filtered_func({"root_key": "world"})

        # Should only return specified output paths
        assert set(result.keys()) == {"root_key", "computed_sum"}
        assert result["root_key"] == "WORLD"
        assert result["computed_sum"] == 60

    def test_filter_func_positional_args_mismatch(self, sample_tree, sample_func):
        """Test that filter_func raises error when positional args don't match input_paths."""
        input_paths = ["root_key", "numbers.[0]"]
        filtered_func = filter_func(sample_func, sample_tree, input_paths=input_paths)

        # Wrong number of arguments
        with pytest.raises(AssertionError, match="Mismatch between number"):
            filtered_func("hello")  # Missing second argument

    def test_filter_func_dict_input_wrong_args(self, sample_tree, sample_func):
        """Test that filter_func raises error with wrong number of dict arguments."""
        filtered_func = filter_func(sample_func, sample_tree)

        # Should expect exactly one dict argument
        with pytest.raises(ValueError, match="Expected a single dictionary argument"):
            filtered_func({"root_key": "hello"}, {"extra": "arg"})

    @pytest.mark.parametrize(
        "invalid_arg",
        [
            "not_a_dict",
            ["not", "a", "dict"],
            42,
            None,
            object(),
        ],
    )
    def test_filter_func_dict_input_wrong_type(
        self, sample_tree, sample_func, invalid_arg
    ):
        """Test that filter_func raises error when argument is not a dictionary."""
        filtered_func = filter_func(sample_func, sample_tree)

        # Should expect a dictionary argument
        with pytest.raises(TypeError, match="Expected argument to be a dictionary"):
            filtered_func(invalid_arg)
