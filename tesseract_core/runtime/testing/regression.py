# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression testing utilities for Tesseract endpoints."""

import builtins
import importlib
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
)

from ..config import get_config
from ..core import get_input_schema, get_output_schema
from ..schema_generation import DICT_INDEX_SENTINEL, get_all_model_path_patterns

ROWFORMAT = "{:>15s}  {:>20s}  {:>20s}  {:>20s}\n"

logger = logging.getLogger("tesseract")


class TestCliConfig(BaseModel):
    """CLI configuration overrides for test execution.

    Contains original CLI arguments for test configuration.
    Only includes safe config options that don't pose security risks.
    """

    input_path: str | None = None  # Original --input-path CLI arg
    volume_mounts: list[str] | None = None  # Original --volume/-v CLI args
    user: str | None = None  # Original --user CLI arg


class TestSpec(BaseModel):
    """Test specification for a single regression test.

    Must provide exactly one of:
    - expected_outputs: For testing successful execution
    - expected_exception: For testing exception handling
    """

    endpoint: str
    payload: dict
    expected_outputs: dict | None = None
    expected_exception: type[Exception] | None = None
    expected_exception_regex: str | None = None
    atol: float = 1e-8
    rtol: float = 1e-5
    cli_config: TestCliConfig | None = None

    @field_validator("expected_exception", mode="before")
    @classmethod
    def parse_exception_type(
        cls, v: str | type[Exception] | None
    ) -> type[Exception] | None:
        """Parse exception from string or type.

        Allows JSON files to specify exceptions as strings (e.g., "ValueError")
        while Python code can pass exception types directly.
        """
        if v is None or isinstance(v, type):
            return v

        if not isinstance(v, str):
            raise ValueError(
                f"expected_exception must be a string or exception type, got {type(v).__name__}"
            )

        # Use existing _parse_exception_type helper
        # But return None for _NoException sentinel
        parsed = _parse_exception_type(v)
        return None if parsed is _NoException else parsed

    @model_validator(mode="after")
    def validate_expected_outcome(self) -> "TestSpec":
        """Ensure exactly one of expected_outputs or expected_exception is provided."""
        has_outputs = self.expected_outputs is not None
        has_exception = self.expected_exception is not None

        if has_outputs and has_exception:
            raise ValueError(
                "Cannot specify both 'expected_outputs' and 'expected_exception'. "
                "Provide exactly one."
            )

        if not has_outputs and not has_exception:
            raise ValueError(
                "Must specify either 'expected_outputs' or 'expected_exception'. "
                "Provide exactly one."
            )

        return self


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


class TestOutputSchema(BaseModel):
    """Output schema for the test endpoint.

    Attributes:
        status: Test result status:
            - "passed": Test passed all validations
            - "failed": Test failed validation (output mismatch)
            - "error": Endpoint raised unexpected exception
        message: Empty for passed tests, error details for failed/error
        endpoint: Name of the tested endpoint
    """

    status: Literal["passed", "failed", "error"]
    message: str
    endpoint: str


class _NoException(Exception):
    """Sentinel exception that cannot be instantiated - represents 'no exception expected'."""

    def __new__(cls) -> None:
        raise TypeError(
            f"{cls.__name__} is intended as a sentinel and cannot be instantiated or raised"
        )


def _parse_exception_type(exception_name: str | None) -> type[Exception]:
    """Parse exception name string to exception class.

    Args:
        exception_name: Name of exception. Must be either:
            - A builtin exception name (e.g., "ValueError", "TypeError")
            - A fully qualified exception in format "packagename.exceptionname"
              (e.g., "pydantic.ValidationError", "requests.exceptions.HTTPError")

    Returns:
        Exception class, or NoException if None provided.

    Raises:
        ValueError: If exception format is invalid or cannot be imported.
    """
    if exception_name is None:
        return _NoException

    # Check if it's a builtin exception first
    if hasattr(builtins, exception_name):
        exc_class = getattr(builtins, exception_name)
        if isinstance(exc_class, type) and issubclass(exc_class, Exception):
            return exc_class

    # For non-builtin exceptions, require packagename.exceptionname format
    if "." not in exception_name:
        raise ValueError(
            f"Non-builtin exception '{exception_name}' must be specified in "
            f"'packagename.exceptionname' format (e.g., 'pydantic.ValidationError')"
        )

    # Split into module path and exception class name
    module_name, class_name = exception_name.rsplit(".", 1)

    # Attempt to import the exception from the package
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(
            f"Failed to import module '{module_name}' for exception '{exception_name}': {e}"
        ) from e

    # Get the exception class from the module
    if not hasattr(module, class_name):
        raise ValueError(
            f"Module '{module_name}' has no attribute '{class_name}'. "
            f"Available attributes: {', '.join(dir(module))}"
        )

    exc_class = getattr(module, class_name)

    # Verify it's actually an exception class
    if not (isinstance(exc_class, type) and issubclass(exc_class, Exception)):
        raise ValueError(
            f"'{exception_name}' is not a valid exception class. "
            f"Found type: {type(exc_class).__name__}"
        )

    return exc_class


def _validate_tree_structure(
    tree: Any,
    template: Any,
    path_patterns: dict[tuple, type] | None = None,
    path: tuple[str, ...] = (),
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
        ValueError: If structures don't match (type, keys, length, shape, dtype).
    """
    if type(tree) is not type(template):
        raise ValueError(
            f"Type mismatch at {'.'.join(path)}:\n"
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

        if tree_keys != template_keys:
            raise ValueError(
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
                # is_dict is only False when path_patterns is not None (set at line 246); assert to satisfy pyright
                assert path_patterns is not None
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
        if len(tree) != len(template):
            raise ValueError(
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
        if tree.shape != template.shape:
            raise ValueError(
                f"Shape mismatch for array at {'.'.join(path)}: \n"
                f"  Expected: {template.shape}\n"
                f"  Obtained: {tree.shape}"
            )
        if tree.dtype != template.dtype:
            raise ValueError(
                f"dtype mismatch for array at {'.'.join(path)}:\n"
                f"  Expected: {template.dtype}\n"
                f"  Obtained: {tree.dtype}"
            )

    # Validation complete return path to leaf
    return {path: (tree, template)}


def _array_discrepancy_msg(
    size: int,
    shape: tuple[int, ...],
    diff_ids: list | np.ndarray,
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


def _compare_leaf_values(
    path: tuple[str, ...],
    obtained_val: Any,
    expected_val: Any,
    atol: float,
    rtol: float,
    threshold: int,
) -> str | None:
    """Compare a single leaf pair and return a discrepancy message, or None if matching."""
    is_inexact_numeric = False
    if isinstance(expected_val, float):
        is_inexact_numeric = True
    elif isinstance(expected_val, (np.number, np.ndarray)):
        if np.issubdtype(expected_val.dtype, np.inexact):
            is_inexact_numeric = True

    if isinstance(expected_val, np.ndarray) and expected_val.ndim == 0:
        expected_val = expected_val[()]
        obtained_val = obtained_val[()]

    if isinstance(expected_val, np.ndarray):
        if is_inexact_numeric:
            not_close_mask = ~np.isclose(
                obtained_val, expected_val, equal_nan=True, atol=atol, rtol=rtol
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
            return f"{'.'.join(path)}\n{array_msg}"
    else:
        if is_inexact_numeric:
            close = np.allclose(obtained_val, expected_val, atol=atol, rtol=rtol)
        else:
            close = obtained_val == expected_val

        if not close:
            if isinstance(expected_val, (int, float, np.number)):
                difference_if_numeric = f"\n  Difference: {obtained_val - expected_val}"
            else:
                difference_if_numeric = ""

            return (
                f"{'.'.join(path)}:\n"
                f"  Expected: {expected_val}\n"
                f"  Obtained: {obtained_val}"
                f"{difference_if_numeric}"
            )

    return None


def regress_test_case(
    api_module: ModuleType,
    endpoint_functions: dict[str, Callable],
    test_spec: TestSpec,
    *,
    base_dir: Path | None = None,
    threshold: int = 100,
) -> TestOutputSchema:
    """Run a single regression test from a test specification.

    Args:
        api_module: Module containing the Tesseract API.
        endpoint_functions: Dict mapping endpoint names to endpoint functions.
        test_spec: Test specification as a TestSpec model with fields:
            - endpoint: Name of the endpoint to test.
            - payload: Input data conforming to InputSchema.
            - expected_outputs: Expected output data (required if no exception expected).
            - expected_exception: Optional exception type (e.g., ValueError, ValidationError).
            - expected_exception_regex: Optional regex pattern to match exception message.
            - atol: Optional absolute tolerance for numeric comparisons (default: 1e-8).
            - rtol: Optional relative tolerance for numeric comparisons (default: 1e-5).
        base_dir: Optional base directory for resolving relative paths in schemas.
        threshold: Maximum number of array discrepancies to display in error messages.

    Returns:
        TestOutputSchema with status:
            - "passed": Test passed all validations
            - "failed": Test failed validation
            - "error": Endpoint raised unexpected exception
    """
    if test_spec.endpoint not in endpoint_functions:
        return TestOutputSchema(
            status="failed",
            message=(
                f"Endpoint {test_spec.endpoint} not found in {api_module.__name__}\n"
                f"  Available endpoints: {list(endpoint_functions.keys())}"
            ),
            endpoint=test_spec.endpoint,
        )
    endpoint_func = endpoint_functions[test_spec.endpoint]

    expected_exception = (
        test_spec.expected_exception if test_spec.expected_exception else _NoException
    )
    expected_exception_regex = test_spec.expected_exception_regex

    # Validate expected_outputs when no exception expected
    if expected_exception is _NoException:
        OutputSchema = get_output_schema(endpoint_func)

        # Build context for validation
        validation_context = {"base_dir": base_dir}
        if test_spec.endpoint == "jacobian":
            validation_context["output_keys"] = test_spec.payload.get("jac_outputs", [])
            validation_context["input_keys"] = test_spec.payload.get("jac_inputs", [])
        elif test_spec.endpoint == "jacobian_vector_product":
            validation_context["output_keys"] = test_spec.payload.get("jvp_outputs", [])
        elif test_spec.endpoint == "vector_jacobian_product":
            validation_context["input_keys"] = test_spec.payload.get("vjp_inputs", [])

        try:
            expected_outputs = OutputSchema.model_validate(
                test_spec.expected_outputs, context=validation_context
            ).model_dump()
        except ValidationError as e:
            error_str = "\n".join(f"  {line}" for line in str(e).splitlines())
            return TestOutputSchema(
                status="failed",
                message=(
                    "expected_outputs does not conform to OutputSchema "
                    f"(perhaps the OutputSchema has recently changed?):\n{error_str}"
                ),
                endpoint=test_spec.endpoint,
            )

    # Validate inputs
    InputSchema = get_input_schema(endpoint_func)
    try:
        loaded_inputs = InputSchema.model_validate(
            test_spec.payload, context={"base_dir": base_dir}
        )
    except expected_exception as e:
        if expected_exception_regex and not re.search(expected_exception_regex, str(e)):
            return TestOutputSchema(
                status="failed",
                message=(
                    f"Exception message does not match regex.\n"
                    f"  Expected pattern: {expected_exception_regex}\n"
                    f"  Actual message: {e}"
                ),
                endpoint=test_spec.endpoint,
            )
        return TestOutputSchema(
            status="passed", message="", endpoint=test_spec.endpoint
        )
    except ValidationError as e:
        # Format each line with 2-space indent
        error_str = "\n".join(f"  {line}" for line in str(e).splitlines())
        return TestOutputSchema(
            status="failed",
            message=(
                "inputs do not conform to InputSchema "
                f"(perhaps the InputSchema has recently changed?):\n{error_str}"
            ),
            endpoint=test_spec.endpoint,
        )
    except Exception as e:
        if expected_exception is _NoException:
            return TestOutputSchema(
                status="error",
                message=f"{type(e).__name__}: {e}",
                endpoint=test_spec.endpoint,
            )
        else:
            return TestOutputSchema(
                status="failed",
                message=f"Expected {expected_exception.__name__}, but got {type(e).__name__}: {e}",
                endpoint=test_spec.endpoint,
            )

    # Call endpoint - only try-except needed for endpoint execution
    try:
        obtained_outputs = endpoint_func(loaded_inputs).model_dump()
    except expected_exception as e:
        if expected_exception_regex and not re.search(expected_exception_regex, str(e)):
            return TestOutputSchema(
                status="failed",
                message=(
                    f"Exception message does not match regex.\n"
                    f"  Expected pattern: {expected_exception_regex}\n"
                    f"  Actual message: {e}"
                ),
                endpoint=test_spec.endpoint,
            )
        return TestOutputSchema(
            status="passed", message="", endpoint=test_spec.endpoint
        )
    except Exception as e:
        if expected_exception is _NoException:
            return TestOutputSchema(
                status="error",
                message=f"{type(e).__name__}: {e}",
                endpoint=test_spec.endpoint,
            )
        else:
            return TestOutputSchema(
                status="failed",
                message=f"Expected {expected_exception.__name__}, but got {type(e).__name__}: {e}",
                endpoint=test_spec.endpoint,
            )

    # If we expected an exception but didn't get one
    if expected_exception is not _NoException:
        return TestOutputSchema(
            status="failed",
            message=f"Expected {expected_exception.__name__}, but no exception was raised",
            endpoint=test_spec.endpoint,
        )

    # Validate structure
    path_patterns = get_all_model_path_patterns(OutputSchema)
    try:
        obtained_expected_flat = _validate_tree_structure(
            obtained_outputs, expected_outputs, path_patterns
        )
    except ValueError as e:
        return TestOutputSchema(
            status="failed", message=str(e), endpoint=test_spec.endpoint
        )

    # Compare values
    atol = test_spec.atol
    rtol = test_spec.rtol
    discrepancies = []

    for path, (obtained_val, expected_val) in obtained_expected_flat.items():
        msg = _compare_leaf_values(
            path, obtained_val, expected_val, atol, rtol, threshold
        )
        if msg is not None:
            discrepancies.append(msg)

    if discrepancies:
        return TestOutputSchema(
            status="failed",
            message="Values are not sufficiently close.\n\n"
            + "\n\n".join(discrepancies),
            endpoint=test_spec.endpoint,
        )

    return TestOutputSchema(status="passed", message="", endpoint=test_spec.endpoint)


def make_test_endpoint(
    api_module: ModuleType,
    endpoints: list[Callable],
) -> Callable[[TestSpec], TestOutputSchema]:
    """Create a test endpoint closure for regression testing.

    Args:
        api_module: Module containing the Tesseract API.
        endpoints: List of endpoint callables. Captured by reference so the
            returned closure always sees the current contents (including the
            ``test`` endpoint itself, which is appended after this factory
            returns).

    Returns:
        A ``test`` function suitable for inclusion in the endpoints list.
    """

    def test(payload: TestSpec) -> TestOutputSchema:
        """Run a single regression test against a Tesseract endpoint.

        Tests an endpoint by calling it with specified inputs and comparing outputs
        against expected values or verifying expected exceptions are raised.

        Args:
            payload: Test specification containing:
                - endpoint: Name of endpoint to test (e.g., "apply", "jacobian")
                - payload: Input data for the endpoint
                - expected_outputs: Expected output data (mutually exclusive with expected_exception)
                - expected_exception: Expected exception type or name (mutually exclusive with expected_outputs)
                - expected_exception_regex: Optional regex pattern for exception message
                - atol: Absolute tolerance for numeric comparisons (default: 1e-8)
                - rtol: Relative tolerance for numeric comparisons (default: 1e-5)

        Returns:
            TestOutputSchema with:
                - status: "passed" | "failed" | "error"
                - message: Empty for passed tests, error details for failed/error
                - endpoint: Name of the tested endpoint

        Note:
            This endpoint is designed for testing and CI/CD workflows.
            All outcomes return HTTP 200 with status in the response body regardless of success/failure.
        """
        logger.warning(
            "The 'test' endpoint is experimental and may change, be replaced, or be deprecated in future versions."
        )

        config = get_config()
        base_dir = Path(config.input_path) if config.input_path else None

        return regress_test_case(
            api_module,
            endpoint_functions={func.__name__: func for func in endpoints},
            test_spec=payload,
            base_dir=base_dir,
            threshold=100,
        )

    return test
