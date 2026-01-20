# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel


def path_to_index_op(
    path: str,
) -> (
    tuple[Literal["seq"], int]
    | tuple[Literal["dict"], str]
    | tuple[Literal["getattr"], str]
):
    """Converts a path string to a tuple of operation and index."""
    seq_idx_re = re.match(r"^\[(\d+)\]$", path)
    if seq_idx_re:
        return ("seq", int(seq_idx_re.group(1)))

    dict_idx_re = re.match(r"^\{(.+)\}$", path)
    if dict_idx_re:
        return ("dict", dict_idx_re.group(1))

    # Use Python's built-in identifier validation for attribute names
    if path.isidentifier():
        return ("getattr", path)

    raise ValueError(f"Invalid path: {path}")


def get_at_path(tree: Any, path: str) -> Any:
    """Gets the value at a path in a nested pytree.

    Paths can have a structure like `a.b.[0].c.{key}` where:
    - `a` is an attribute / key of the input tree
    - `b.[0]` is the first element of the iterable `b`
    - `c.{key}` is the value of the key `key` in the mapping `c`
    """
    # Empty path means "the root of the tree"
    if not path:
        return tree

    split_path = path.split(".")

    def _get_recursive(tree: Any, path: list[str]) -> Any:
        if not path:
            return tree

        key, path = path[0], path[1:]
        method, idx = path_to_index_op(key)
        if method in ("seq", "dict"):
            if hasattr(tree, "__getitem__"):
                return _get_recursive(tree[idx], path)
            else:
                # For types like sets that don't support indexing, we have to iterate manually
                # and hope the ordering is consistent between iterations
                for i, val in enumerate(tree):
                    if i == idx:
                        return _get_recursive(val, path)
                raise IndexError(f"Index {idx} out of range for {tree}")
        elif method == "getattr":
            if hasattr(tree, key):
                return _get_recursive(getattr(tree, key), path)
            elif isinstance(tree, Mapping):
                # If the key is not an attribute, try to access it as a key in a dictionary
                # This is useful for accessing keys of models that have been dumped to dictionaries
                return _get_recursive(tree[key], path)
            else:
                raise AttributeError(f"Attribute {key} not found in {tree}")
        else:
            raise AssertionError(f"Invalid method: {method}")

    return _get_recursive(tree, split_path)


def set_at_path(tree: Any, values: dict[str, Any]) -> Any:
    """Sets the value at a collection of paths in a nested pytree.

    `values` argument is a flat dictionary with paths as keys and values as values.

    Paths can have a structure like `a.b.[0].c.{key}` where:
    - `a` is an attribute / key of the input tree
    - `b.[0]` is the first element of the iterable `b`
    - `c.{key}` is the value of the key `key` in the mapping `c`
    """
    tree = deepcopy(tree)

    def _set_recursive(tree: Any, path: list[str], value: Any) -> Any:
        key, path = path[0], path[1:]
        method, idx = path_to_index_op(key)
        if method in ("seq", "dict"):
            if not path:
                tree[idx] = value
                return
            return _set_recursive(tree[idx], path, value)
        elif method == "getattr":
            if hasattr(tree, key):
                if not path:
                    setattr(tree, key, value)
                    return
                return _set_recursive(getattr(tree, key), path, value)
            elif isinstance(tree, dict):
                # If the key is not an attribute, try to access it as a key in a dictionary
                # This is useful for accessing keys of models that have been dumped to dictionaries
                if not path:
                    tree[key] = value
                    return
                return _set_recursive(tree[key], path, value)
            else:
                raise AttributeError(f"Attribute {key} not found in {tree}")
        else:
            raise AssertionError(f"Invalid method: {method}")

    for path, value in values.items():
        split_path = path.split(".")
        _set_recursive(tree, split_path, value)

    return tree


def flatten_with_paths(
    tree: Mapping | Sequence | BaseModel,
    include_paths: Iterable[str],
) -> dict[str, Any]:
    """Filter and flatten a nested PyTree by extracting only the specified paths.

    Returns a dictionary with the specified keys and corresponding values.
    """
    out = {}
    for path in include_paths:
        out[path] = get_at_path(tree, path)
    return out


def expand_path_pattern(path_pattern: str, inputs: dict[str, Any]) -> list[str]:
    """Expand a path pattern to a list of all matching paths in the given pytree.

    For example, given the path pattern `a.[].{}`, and the inputs `{"a": [{"b": 1}, {"c": 2}]}`,
    this function would return `["a.[0].{b}", "a.[1].{c}"]`.

    Silently ignores parts of the pattern that do not match the inputs, assuming they are optional.
    """
    parts = path_pattern.split(".")

    def _handle_part(
        parts: Sequence[str], current_inputs: Any, current_path: list[str]
    ) -> list[str]:
        """Recursively expand each part separately."""
        if not parts or not hasattr(current_inputs, "__iter__"):
            return [".".join(current_path)]

        paths = []
        part = parts[0]

        if part == "[]":
            # sequence access
            for i, val in enumerate(current_inputs):
                subpaths = _handle_part(parts[1:], val, [*current_path, f"[{i}]"])
                paths.extend(subpaths)
        elif part == "{}":
            # dictionary access
            for key in current_inputs:
                subpaths = _handle_part(
                    parts[1:], current_inputs[key], [*current_path, f"{{{key}}}"]
                )
                paths.extend(subpaths)
        elif part in current_inputs:
            subpaths = _handle_part(
                parts[1:], current_inputs[part], [*current_path, part]
            )
            paths.extend(subpaths)
        return paths

    return _handle_part(parts, inputs, [])


def filter_func(
    func: Callable[[dict], dict],
    default_inputs: dict,
    output_paths: Iterable[str] | None = None,
    input_paths: Sequence[str] | None = None,
) -> Callable:
    """Modifies a function that operates on pytrees to operate on flat {path: value} or positional args instead.

    The returned function accepts either a dictionary `{input_path: value}` if `input_paths` is None or
    positional arguments in the same order as input_paths.
    The function will update the default inputs with the new values.
    It will then call the original function with the updated inputs and return a dictionary
    `{output_path: value}` if output_paths is not None, or the full unmodified output otherwise.

    Args:
        func: The original function that accepts a single pytree as input and returns a single pytree as output.
        default_inputs: The default input pytree to the function. Also used to determine the structure of the inputs.
        output_paths: The subset of paths of the outputs that the modified function returns.
            If None, the full output is returned unmodified.
        input_paths: The paths that positional arguments correspond to.
            If None, a single dictionary argument is expected by the modified function.
    """

    def filtered_func(*args: Any) -> dict:
        if input_paths:
            if len(input_paths) != len(args):
                raise ValueError(
                    f"Mismatch between number of given paths {len(input_paths)} and args {len(args)}."
                )
            new_inputs = dict(zip(input_paths, args, strict=True))
        else:
            if len(args) != 1:
                raise ValueError("Expected a single dictionary argument")
            if not isinstance(args[0], dict):
                raise TypeError("Expected argument to be a dictionary")
            new_inputs = args[0]

        updated_inputs = set_at_path(default_inputs, new_inputs)
        outputs = func(updated_inputs)
        if output_paths:
            outputs = flatten_with_paths(outputs, output_paths)
        return outputs

    return filtered_func
