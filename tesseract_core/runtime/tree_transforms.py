# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Optional, Union

from pydantic import BaseModel


def path_to_index_op(path: str) -> tuple[str, Union[int, str]]:
    """Converts a path string to a tuple of operation and index."""
    seq_idx_re = re.match(r"\[(\d+)\]", path)
    if seq_idx_re:
        return ("seq", int(seq_idx_re.group(1)))

    dict_idx_re = re.match(r"\{(.+)\}", path)
    if dict_idx_re:
        return ("dict", dict_idx_re.group(1))

    getattr_re = re.match(r"(\w+)", path)
    if getattr_re:
        return ("getattr", getattr_re.group(1))

    raise ValueError(f"Invalid path: {path}")


def get_at_path(tree: Any, path: str) -> Any:
    """Gets the value at a path in a nested pytree.

    Paths can have a structure like `a.b.[0].c.{key}` where:
    - `a` is an attribute / key of the input tree
    - `b.[0]` is the first element of the list `b`
    - `c.{key}` is the value of the key `key` in the dictionary `c`
    """
    split_path = path.split(".")

    def _get_recursive(tree: Any, path: list[str]) -> Any:
        if not path:
            return tree

        key, path = path[0], path[1:]
        method, idx = path_to_index_op(key)
        if method in ("seq", "dict"):
            return _get_recursive(tree[idx], path)
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
    - `b.[0]` is the first element of the list `b`
    - `c.{key}` is the value of the key `key` in the dictionary `c`
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
            elif isinstance(tree, Mapping):
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
    tree: Union[Mapping, Sequence, BaseModel],
    include_paths: set[str],
) -> dict[str, Any]:
    """Filter and flatten a nested PyTree by extracting only the specified paths.

    Returns a dictionary with the specified keys and corresponding values.
    """
    out = {}
    for path in include_paths:
        out[path] = get_at_path(tree, path)
    return out


def filter_func(
    func: Callable[[dict], dict],
    default_inputs: dict,
    output_paths: set[str],
    input_paths: Optional[set[str]] = None,
    output_to_tuple: bool = False,
) -> Callable:
    """Returns a reduced func with default inputs that operates on positional args or {path: value} dicts.

    The returned function accepts either a dictionary `{input_path: value}` if `input_paths` is None or
    positional arguments in the same order as input_paths.
    The function will update the default inputs with the new values.
    It will then call the original function with the updated inputs and return a dictionary
    `{output_path: value}` or a tuple of values if `output_to_tuple` is True.

    Args:
        func: The original function that accepts a dictionary of inputs
        default_inputs: The default inputs to the function
        output_paths: The paths of the outputs that the function returns
        input_paths: The keys to reconstruct positional to dictionaries.
            If None, the returned function accepts a dictionary arguments.
        output_to_tuple: If True, the returned function will return a tuple of outputs
    """

    def filtered_func(new_inputs: dict) -> dict:
        updated_inputs = set_at_path(default_inputs, new_inputs)

        path_outputs = flatten_with_paths(func(updated_inputs), output_paths)

        return tuple(path_outputs.values()) if output_to_tuple else path_outputs

    if not input_paths:
        return filtered_func

    else:
        # function that accepts positional arguments
        def filtered_pos_func(*args):
            # convert back to dictionary
            new_inputs = dict(zip(input_paths, args))

            # call the filtered function that accepts dictionaries
            return filtered_func(new_inputs)

        return filtered_pos_func
