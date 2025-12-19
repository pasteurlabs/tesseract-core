from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _validate_tree_structure(tree: Any, template: Any, path: str = "") -> None:
    """Recursively validates a tree-like structure against a template.

    Compares types, dictionary keys, sequence lengths, array shapes and dtypes.
    Does not compare values, raises AssertionError on first mismatch.
    """
    assert type(tree) is type(template), (
        f"Type mismatch at {path}:\n"
        f"  Expected: {type(template).__name__}, "
        f"  Obtained: {type(tree).__name__}\n\n"
    )

    if isinstance(template, Mapping):  # Dictionary-like structure
        tree_keys = set(tree.keys())
        template_keys = set(template.keys())

        assert tree_keys == template_keys, (
            f"Key mismatch at {path}:\n"
            f"  Missing keys: {template_keys - tree_keys}\n"
            f"  Unexpected keys: {tree_keys - template_keys}\n"
            f"  Matching keys: {template_keys & tree_keys}\n\n"
        )

        for key in template_keys:
            _validate_tree_structure(tree[key], template[key], f"{path}.{key}")

    elif isinstance(template, Sequence) and not isinstance(
        template, (str, bytes)
    ):  # List, tuple, etc.
        assert len(tree) == len(template), (
            f"Mismatch in length of {type(template).__name__} at {path}:\n"
            f"  Expected: {len(template)}\n"
            f"  Obtained: {len(tree)}\n\n"
        )

        for i, (tree_branch, template_branch) in enumerate(
            zip(tree, template, strict=True)
        ):
            _validate_tree_structure(tree_branch, template_branch, f"{path}[{i}]")

    elif isinstance(template, np.ndarray):
        assert tree.shape == template.shape, (
            f"Shape mismatch for array at {path}: \n"
            f"  Expected: {template.shape}\n"
            f"  Obtained: {tree.shape}\n\n"
        )
        assert tree.dtype == template.dtype, (
            f"dtype mismatch for array at {path}:\n"
            f"  Expected: {template.dtype}\n"
            f"  Obtained: {tree.dtype}\n\n"
        )

    # If nothing above matched do nothing
    return
