# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression testing utilities for Tesseract endpoints."""

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import ValidationError

from ..core import create_endpoints
from ..schema_generation import DICT_INDEX_SENTINEL, get_all_model_path_patterns
from .common import get_input_schema, get_output_schema

ROWFORMAT = "{:>15s}  {:>20s}  {:>20s}  {:>20s}\n"


class RegressionTestResult(NamedTuple):
    """Result of a regression test.

    Attributes:
        test_file: Path to json file containing test cases
        endpoint: name of endpoint function tested
        status: Whether the regression passed, failed or an unexpected error occurred
        message: Any relevant failure/exception message
    """

    test_file: Path
    endpoint: str
    status: Literal["passed", "failed", "error"]
    message: str | None


class _NoException(Exception):
    """Sentinel exception that cannot be instantiated - represents 'no exception expected'."""

    def __new__(cls) -> None:
        raise TypeError(
            f"{cls.__name__} is intended as a sentinel and cannot be instantiated or raised"
        )


def _parse_exception_type(exception_name: str | None) -> type[Exception]:
    """Parse exception name string to exception class.

    Args:
        exception_name: Name of exception (e.g. "ValueError", "ValidationError")

    Returns:
        Exception class, or NoException if None provided.
    """
    if exception_name is None:
        return _NoException

    # Common exceptions mapping
    exception_mapping = {
        "ValidationError": ValidationError,
        # Add other common non-builtin exceptions here as needed
    }

    # Check custom mapping first
    if exception_name in exception_mapping:
        return exception_mapping[exception_name]

    # Try to get from builtins
    import builtins

    if hasattr(builtins, exception_name):
        exc_class = getattr(builtins, exception_name)
        if isinstance(exc_class, type) and issubclass(exc_class, BaseException):
            return exc_class

    raise ValueError(f"Unknown exception type: {exception_name}")


def _validate_tree_structure(
    tree: Any,
    template: Any,
    path_patterns: dict[tuple, type] | None = None,
    path: tuple[str | object, ...] = (),
) -> dict[tuple[str, ...], tuple[Any, Any]]:
    """Recursively validate a tree-like structure against a template and collect leaf values.

    Compares types, dictionary keys, sequence lengths, array shapes and dtypes.
    Does not compare values, raises AssertionError on first mismatch. Uses schema
    path patterns to distinguish between dict keys (formatted as "{key}") and model
    attributes (formatted as "attribute").

    Args:
        tree: The tree structure to validate (obtained values).
        template: The template structure to validate against (expected values).
        path_patterns: Optional schema patterns mapping path tuples to types,
            used to distinguish dicts from model attributes.
        path: Current path in the tree traversal (for error messages).

    Returns:
        Dict mapping path tuples to (tree_leaf, template_leaf) tuples.

    Raises:
        AssertionError: If structures don't match (type, keys, length, shape, dtype).
    """
    assert type(tree) is type(template), (
        f"Type mismatch at {path}:\n"
        f"  Expected: {type(template).__name__}, "
        f"  Obtained: {type(tree).__name__}"
    )

    if isinstance(template, Mapping):  # Dictionary-like structure
        # assume Mapping is a regular dict unless path patterns has specific attribute names
        # (in which case the dict is just a json representation of a pydantic BaseModel)
        is_dict = True
        if path_patterns:
            arbitrary_key = next(iter(path_patterns.keys()))
            # If the key is an empty tuple there is a flaw in our logic
            # Instead of raising an assertion error we let the IndexError
            # propagate upwards for visibiility
            is_dict = arbitrary_key[0] == DICT_INDEX_SENTINEL

        key_or_attribute = "key" if is_dict else "optional attribute"

        tree_keys = set(tree.keys())
        template_keys = set(template.keys())

        assert tree_keys == template_keys, (
            f"{key_or_attribute.capitalize()} mismatch at {'.'.join(path)}:\n"
            f"  Missing {key_or_attribute}s: {template_keys - tree_keys}\n"
            f"  Unexpected {key_or_attribute}s: {tree_keys - template_keys}\n"
            f"  Matching {key_or_attribute.split(' ')[-1]}s: {template_keys & tree_keys}"
        )

        leaves = {}
        for key in template_keys:
            if is_dict:
                # Drop sentinel
                if path_patterns:
                    next_path_patterns = {k[1:]: v for k, v in path_patterns.items()}
                else:
                    next_path_patterns = None
            else:
                # Filter to paths beginning with relevant attribute
                next_path_patterns = {
                    k[1:]: v
                    for k, v in path_patterns.items()
                    if len(k) > 1 and k[0] == key
                }
            leaves.update(
                _validate_tree_structure(
                    tree[key],
                    template[key],
                    next_path_patterns,
                    (*path, f"{{{key}}}" if is_dict else f"{key}"),
                )
            )
        return leaves

    elif isinstance(template, Sequence) and not isinstance(
        template, (str, bytes)
    ):  # List, tuple, etc.
        assert len(tree) == len(template), (
            f"Mismatch in length of {type(template).__name__} at {'.'.join(path)}:\n"
            f"  Expected: {len(template)}\n"
            f"  Obtained: {len(tree)}"
        )

        # Drop sentinel
        if path_patterns:
            next_path_patterns = {k[1:]: v for k, v in path_patterns.items()}
        else:
            next_path_patterns = None

        leaves = {}
        for i, (tree_branch, template_branch) in enumerate(
            zip(tree, template, strict=True)
        ):
            leaves.update(
                _validate_tree_structure(
                    tree_branch, template_branch, next_path_patterns, (*path, f"[{i}]")
                )
            )
        return leaves

    elif isinstance(template, np.ndarray):
        assert tree.shape == template.shape, (
            f"Shape mismatch for array at {'.'.join(path)}: \n"
            f"  Expected: {template.shape}\n"
            f"  Obtained: {tree.shape}"
        )
        assert tree.dtype == template.dtype, (
            f"dtype mismatch for array at {'.'.join(path)}:\n"
            f"  Expected: {template.dtype}\n"
            f"  Obtained: {tree.dtype}"
        )

    # Validation complete return path to leaf
    return {path: (tree, template)}


def _array_discrepancy_msg(
    size: int,
    shape: tuple[int],
    diff_ids: list,
    obtained_array: np.ndarray,
    expected_array: np.ndarray,
    threshold: int = 100,
) -> str:
    """Format detailed discrepancy message for arrays that don't match.

    DISCLAIMER: Logic/messsaging borrows heavily from pytest-regressions
    https://github.com/ESSS/pytest-regressions/blob/master/src/pytest_regressions/ndarrays_regression.py

    Args:
        size: Total size of the full arrays.
        shape: Shape of the full arrays.
        diff_ids: Indices where arrays differ.
        obtained_array: Array values that were obtained (only differing elements).
        expected_array: Array values that were expected (only differing elements).
        threshold: Maximum number of individual differences to display.

    Returns:
        Formatted error message with statistics and individual differences.
    """
    # Summary
    error_msg = f"Shape: {shape}\n"
    pct = 100 * len(diff_ids) / size
    error_msg += f"  Number of differences: {len(diff_ids)} / {size} ({pct:.1f}%)\n"
    if np.issubdtype(obtained_array.dtype, np.number) and len(diff_ids) > 1:
        error_msg += "  Statistics are computed for differing elements only.\n"

        abs_errors = abs(obtained_array - expected_array)
        error_msg += "  Stats for abs(obtained - expected):\n"
        error_msg += f"    Max:     {abs_errors.max()}\n"
        error_msg += f"    Mean:    {abs_errors.mean()}\n"
        error_msg += f"    Median:  {np.median(abs_errors)}\n"

        expected_nonzero = np.array(np.nonzero(expected_array)).T
        rel_errors = abs(
            (obtained_array[expected_nonzero] - expected_array[expected_nonzero])
            / expected_array[expected_nonzero]
        )
        if len(rel_errors) == 0:
            error_msg += "  Relative errors are not reported because all expected values are zero.\n"
        else:
            error_msg += "  Stats for abs(obtained - expected) / abs(expected):\n"
            if len(rel_errors) != len(abs_errors):
                pct = 100 * len(rel_errors) / len(abs_errors)
                error_msg += "    Number of (differing) non-zero expected results: "
                error_msg += f"{len(rel_errors)} / {len(abs_errors)} ({pct:.1f}%)\n"
                error_msg += "    Relative errors are computed for the non-zero expected results.\n"
            else:
                rel_errors = abs((obtained_array - expected_array) / expected_array)
            error_msg += f"    Max:     {rel_errors.max()}\n"
            error_msg += f"    Mean:    {rel_errors.mean()}\n"
            error_msg += f"    Median:  {np.median(rel_errors)}\n"

    # Details results
    error_msg += "  Individual errors:\n"
    if len(diff_ids) > threshold:
        error_msg += f"    Only showing first {threshold} mismatches.\n"
        diff_ids = diff_ids[:threshold]
        obtained_array = obtained_array[:threshold]
        expected_array = expected_array[:threshold]
    error_msg += ROWFORMAT.format(
        "Index",
        "Obtained",
        "Expected",
        "Difference",
    )
    for diff_id, obtained, expected in zip(
        diff_ids, obtained_array, expected_array, strict=True
    ):
        diff_id_str = ", ".join(str(i) for i in diff_id)
        if len(diff_id) != 1:
            diff_id_str = f"({diff_id_str})"
        error_msg += ROWFORMAT.format(
            diff_id_str,
            str(obtained),
            str(expected),
            (str(obtained - expected) if isinstance(obtained, np.number) else ""),
        )
    error_msg += "\n"
    return error_msg


def regress_test_case(
    api_module: ModuleType,
    endpoint_functions: dict[str, Callable],
    test_spec: dict,
    *,
    base_dir: Path | None = None,
    threshold: int = 100,
) -> None:
    """Run a single regression test from a test specification.

    Args:
        api_module: Module containing the Tesseract API.
        endpoint_functions: Dict mapping endpoint names to endpoint functions.
        test_spec: Test specification dict loaded from JSON file. Expected keys:
            - endpoint: Name of the endpoint to test.
            - inputs: Input data conforming to InputSchema.
            - expected_outputs: Expected output data (required if no exception expected).
            - expected_exception: Optional exception type name (e.g., "ValueError").
            - expected_exception_regex: Optional regex pattern to match exception message.
            - atol: Optional absolute tolerance for numeric comparisons (default: 1e-8).
            - rtol: Optional relative tolerance for numeric comparisons (default: 1e-5).
        base_dir: Optional base directory for resolving relative paths in schemas.
        threshold: Maximum number of array discrepancies to display in error messages.

    Raises:
        AssertionError: If the test fails (output mismatch, wrong exception, etc.).
    """
    assert test_spec["endpoint"] in endpoint_functions, (
        f"Endpoint {test_spec['endpoint']} not found in {api_module.__name__}\n"
        f"  Available endpoints: {list(endpoint_functions.keys())}"
    )
    endpoint_func = endpoint_functions[test_spec["endpoint"]]

    # Parse expected exception
    expected_exception = _parse_exception_type(test_spec.get("expected_exception"))
    expected_exception_regex = test_spec.get("expected_exception_regex")

    if expected_exception_regex is not None and not isinstance(
        expected_exception_regex, str
    ):
        raise AssertionError(
            f"expected_exception_regex must be a string, got {type(expected_exception_regex).__name__}."
        )

    # Read and validate expected_outputs when no exception expected
    if expected_exception is _NoException:
        assert "expected_outputs" in test_spec, (
            "expected_outputs missing when no exception expected"
        )
        expected_outputs = test_spec["expected_outputs"]

        OutputSchema = get_output_schema(endpoint_func)
        try:
            OutputSchema.model_validate(
                expected_outputs, context={"base_dir": base_dir}
            )
        except ValidationError as e:
            error_str = "\n".join(f"  {line}" for line in str(e).splitlines())
            raise AssertionError(
                "expected_outputs does not conform to OutputSchema "
                f"(perhaps the OutputSchema has recently changed?):\n{error_str}"
            ) from None

    # Load + dump inputs to ensure they are valid + normalized
    assert "inputs" in test_spec, "inputs missing"

    # Try to validate inputs and run endpoint, catching the expected exception
    InputSchema = get_input_schema(endpoint_functions["apply"])
    try:
        loaded_inputs = InputSchema.model_validate(
            test_spec["inputs"], context={"base_dir": base_dir}
        )
    except expected_exception as e:
        if expected_exception_regex:
            if not re.search(expected_exception_regex, str(e)):
                raise AssertionError(
                    f"Exception message does not match regex.\n"
                    f"  Expected pattern: {expected_exception_regex}\n"
                    f"  Actual message: {e}"
                ) from None
        # Test passed - exception was as expected
        return
    except ValidationError as e:
        # Format each line with 2-space indent
        error_str = "\n".join(f"  {line}" for line in str(e).splitlines())
        raise AssertionError(
            "inputs do not conform to InputSchema "
            f"(perhaps the InputSchema has recently changed?):\n{error_str}"
        ) from None
    except Exception as e:
        # Got unexpected exception type
        if expected_exception is _NoException:
            raise
        else:
            raise AssertionError(
                f"Expected {expected_exception}, but got {type(e).__name__}: {e}"
            ) from None

    try:
        obtained_outputs = endpoint_func(loaded_inputs).model_dump()
        # If we get here with no exception but expected one, that's a failure
        if expected_exception is not _NoException:
            raise AssertionError(
                f"Expected {expected_exception}, but no exception was raised"
            ) from None
    except expected_exception as e:
        if expected_exception_regex:
            if not re.search(expected_exception_regex, str(e)):
                raise AssertionError(
                    f"Exception message does not match regex.\n"
                    f"  Expected pattern: {expected_exception_regex}\n"
                    f"  Actual message: {e}"
                ) from None
        # Test passed - exception was as expected
        return
    except Exception as e:
        # Got unexpected exception type
        if expected_exception is _NoException:
            raise
        else:
            raise AssertionError(
                f"Expected {expected_exception}, but got {type(e).__name__}: {e}"
            ) from None

    # Validate structure of outputs
    # The output schema provides a guide to distinguish between dicts (with {keys}) and models (with attributes)
    path_patterns = get_all_model_path_patterns(OutputSchema)

    obtained_expected_flat = _validate_tree_structure(
        obtained_outputs, expected_outputs, path_patterns
    )

    atol = test_spec.get("atol", 1e-8)
    rtol = test_spec.get("rtol", 1e-5)

    discrepancies = []
    for path, (obtained_val, expected_val) in obtained_expected_flat.items():
        is_inexact_numeric = False
        if isinstance(expected_val, float):
            is_inexact_numeric = True
        elif isinstance(expected_val, (np.number, np.ndarray)):
            if np.issubdtype(expected_val.dtype, np.inexact):
                is_inexact_numeric = True

        if isinstance(expected_val, np.ndarray) and expected_val.ndim == 0:
            # convert scalar arrays to np.numbers for simpler discrepancy message logic
            expected_val = expected_val[()]
            obtained_val = obtained_val[()]

        if isinstance(expected_val, np.ndarray):
            if is_inexact_numeric:
                not_close_mask = ~np.isclose(
                    obtained_val,
                    expected_val,
                    equal_nan=True,
                    atol=atol,
                    rtol=rtol,
                )
            else:
                not_close_mask = obtained_val != expected_val

            if np.any(not_close_mask):
                diff_ids = np.array(np.nonzero(not_close_mask)).T
                array_msg = _array_discrepancy_msg(
                    expected_val.size,
                    expected_val.shape,
                    diff_ids,
                    obtained_val[not_close_mask],
                    expected_val[not_close_mask],
                    threshold,
                )
                discrepancies.append(f"{'.'.join(path)}\n{array_msg}")
        else:
            if is_inexact_numeric:
                close = np.allclose(obtained_val, expected_val, atol=atol, rtol=rtol)
            else:
                close = obtained_val == expected_val

            if not close:
                if isinstance(expected_val, (int, float, np.number)):
                    difference_if_numeric = (
                        f"\n  Difference: {obtained_val - expected_val}"
                    )
                else:
                    difference_if_numeric = ""

                discrepancies.append(
                    f"{'.'.join(path)}:\n"
                    f"  Expected: {expected_val}\n"
                    f"  Obtained: {obtained_val}"
                    f"{difference_if_numeric}"
                )
    if discrepancies:
        raise AssertionError(
            "Values are not sufficiently close.\n\n" + "\n\n".join(discrepancies)
        )


def iter_regression_tests(
    api_module: ModuleType,
    *test_case_paths: Path,
    base_dir: Path | None = None,
    threshold: int = 100,
) -> Iterator[RegressionTestResult]:
    """Iteratively run regression tests from multiple test case paths.

    This function yields test results as they complete, allowing for streaming
    progress indicators (e.g., pytest-like ".....F...E..x" output).

    Args:
        api_module: Module containing the Tesseract API.
        *test_case_paths: Paths to .json test files or directories containing them.
            Directories are recursively searched for .json files.
        base_dir: Optional base directory for resolving relative paths in schemas.
        threshold: Maximum number of array discrepancies to display in error messages.

    Yields:
        RegressionTestResult for each test case, with status "passed", "failed", or "error".

    Raises:
        ValueError: If no test files are found or paths are invalid.
        FileNotFoundError: If a specified path does not exist.

    Example:
        >>> import my_tesseract_module
        >>> for result in iter_regression_tests(my_tesseract_module, Path("tests/")):
        ...     if result.status == "passed":
        ...         print(f"✓ {result.test_file.name}")
        ...     else:
        ...         print(f"✗ {result.test_file.name}: {result.message}")
    """
    # Get available endpoints
    endpoint_functions = {func.__name__: func for func in create_endpoints(api_module)}

    # Test for existence and expand directories to json files
    test_files: list[Path] = []
    for path in test_case_paths:
        if path.is_file():
            if path.suffix == ".json":
                test_files.append(path)
            else:
                raise ValueError(
                    f"Test case path must be a .json file or directory: {path}"
                )
        elif path.is_dir():
            test_files.extend(sorted(path.glob("*.json")))
        else:
            raise FileNotFoundError(f"Test case path does not exist: {path}")

    if not test_files:
        raise ValueError("No test files found in provided paths")

    for test_file in test_files:
        try:
            with open(test_file) as f:
                spec = json.load(f)
            endpoint = spec["endpoint"]
        except KeyError:
            yield RegressionTestResult(
                test_file,
                "unknown",
                "error",
                "Endpoint not specified, this is mandatory.",
            )
            continue
        except Exception as e:
            # If we can't even read the file, report as error
            yield RegressionTestResult(
                test_file, "unknown", "error", f"Failed to read test file: {e}"
            )
            continue

        try:
            regress_test_case(
                api_module,
                endpoint_functions,
                spec,
                base_dir=base_dir,
                threshold=threshold,
            )
            status = "passed"
            error_msg = ""
        except AssertionError as e:
            status = "failed"
            error_msg = str(e)
        except Exception as e:
            # Unexpected error - bug in test or code
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
        yield RegressionTestResult(test_file, endpoint, status, error_msg)
