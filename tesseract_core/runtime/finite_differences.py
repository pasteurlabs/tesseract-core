# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback
from collections.abc import Iterator, Sequence
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    get_args,
)

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel
from rich.progress import Progress

from .core import create_endpoints
from .tree_transforms import get_at_path, set_at_path

ADEndpointName = Literal[
    "jacobian", "jacobian_vector_product", "vector_jacobian_product"
]


class GradientCheckResult(NamedTuple):
    """Result of a gradient check (Jacobian row).

    Attributes:
        in_path: The input path of the gradient check.
        out_path: The output path of the gradient check.
        idx: The row index of the gradient check.
        grad_val: The value of the gradient at the given index.
        ref_val: The value of the reference gradient at the given index.
        exception: The exception raised during the gradient check, if any.
    """

    in_path: str
    out_path: str
    idx: tuple[int, ...]
    grad_val: Optional[ArrayLike]
    ref_val: Optional[ArrayLike]
    exception: Optional[str]


def get_input_schema(endpoint_function: Callable) -> type[BaseModel]:
    """Get the input schema of an endpoint function."""
    schema = endpoint_function.__annotations__["payload"]
    if not issubclass(schema, BaseModel):
        raise AssertionError(f"Expected BaseModel, got {schema}")
    return schema


def get_output_schema(endpoint_function: Callable) -> type[BaseModel]:
    """Get the output schema of an endpoint function."""
    schema = endpoint_function.__annotations__["return"]
    if not issubclass(schema, BaseModel):
        raise AssertionError(f"Expected BaseModel, got {schema}")
    return schema


def expand_path_pattern(path_pattern: str, inputs: dict[str, Any]) -> list[str]:
    """Expand a path pattern to a list of all matching paths in the given pytree.

    For example, given the path pattern `a.[].{}`, and the inputs `{"a": [{"b": 1}, {"c": 2}]}`,
    this function would return `["a.[0].{b}", "a.[1].{c}"]`.
    """
    parts = path_pattern.split(".")

    def _handle_part(
        parts: Sequence[str], current_inputs: Any, current_path: list[str]
    ) -> list[str]:
        """Recursively expand each part separately."""
        if not parts:
            return [".".join(current_path)]

        paths = []
        part = parts[0]

        if part == "[]":
            # sequence access
            for i, _ in enumerate(current_inputs):
                subpaths = _handle_part(
                    parts[1:], current_inputs[i], [*current_path, f"[{i}]"]
                )
                paths.extend(subpaths)
        elif part == "{}":
            # dictionary access
            for key in current_inputs:
                subpaths = _handle_part(
                    parts[1:], current_inputs[key], [*current_path, f"{{{key}}}"]
                )
                paths.extend(subpaths)
        else:
            subpaths = _handle_part(
                parts[1:], current_inputs[part], [*current_path, part]
            )
            paths.extend(subpaths)
        return paths

    return _handle_part(parts, inputs, [])


def get_differentiable_paths(
    apply_endpoint_fn: Callable, inputs: dict[str, Any], outputs: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Get the paths of all differentiable leaves present in the given inputs and outputs."""
    InputSchema = get_input_schema(apply_endpoint_fn)
    OutputSchema = get_output_schema(apply_endpoint_fn)

    diffable_input_paths = InputSchema.differentiable_arrays
    diffable_output_paths = OutputSchema.differentiable_arrays

    ad_inputs = []
    for pattern in diffable_input_paths:
        ad_inputs.extend(expand_path_pattern(pattern, inputs))

    ad_outputs = []
    for pattern in diffable_output_paths:
        ad_outputs.extend(expand_path_pattern(pattern, outputs))

    return ad_inputs, ad_outputs


def _cached_jacobian(fn: Callable) -> Callable:
    """Cache the result of the jacobian computation based on input_path, output_path, and input_idx."""
    cache = {}

    @wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        _, _, input_path, output_path, input_idx, *_ = args
        key = (input_path, output_path, tuple(input_idx))
        if key not in cache:
            try:
                cache[key] = fn(*args, **kwargs)
            except Exception as e:
                cache[key] = e
        if isinstance(cache[key], Exception):
            raise cache[key]
        return cache[key]

    _wrapper.clear_cache = cache.clear
    return _wrapper


@_cached_jacobian
def _jacobian_via_apply(
    endpoints_func: dict[str, Callable],
    inputs: dict[str, Any],
    input_path: Sequence[str],
    output_path: Sequence[str],
    input_idx: tuple[int, ...],
    eps: float = 1e-4,
) -> ArrayLike:
    """Compute a Jacobian row using central finite differences."""
    apply_fn = endpoints_func["apply"]
    ApplySchema = get_input_schema(apply_fn)

    def _perturbed_apply(inputs: dict[str, Any], eps: float) -> dict[str, Any]:
        input_val = get_at_path(inputs, input_path).copy()
        if input_idx:
            # array
            input_val[input_idx] += eps
        else:
            # scalar
            input_val += eps
        inputs_plus = set_at_path(inputs, {input_path: input_val})
        return apply_fn(
            ApplySchema.model_validate({"inputs": inputs_plus})
        ).model_dump()

    output_plus = _perturbed_apply(inputs, eps)
    output_minus = _perturbed_apply(inputs, -eps)

    central_diff = (
        get_at_path(output_plus, output_path) - get_at_path(output_minus, output_path)
    ) / (2 * eps)
    return central_diff


@_cached_jacobian
def _jacobian_via_jacobian(
    endpoints_func: dict[str, Callable],
    inputs: dict[str, Any],
    input_path: Sequence[str],
    output_path: Sequence[str],
    input_idx: tuple[int, ...],
) -> ArrayLike:
    """Compute a Jacobian row using the jacobian endpoint."""
    jac_fn = endpoints_func["jacobian"]

    def _jacobian(inputs: dict[str, Any]) -> dict[str, Any]:
        JacSchema = get_input_schema(jac_fn)
        return jac_fn(
            JacSchema.model_validate(
                {
                    "inputs": inputs,
                    "jac_inputs": [input_path],
                    "jac_outputs": [output_path],
                }
            )
        ).model_dump()

    output = _jacobian(inputs)
    output_val = output[output_path][input_path]
    # Jacobian output has shape (*output_shape, *input_shape), where we slice into input_shape
    # while passing through output_shape.
    jac_slice = (slice(None),) * (output_val.ndim - len(input_idx)) + tuple(input_idx)
    return output_val[jac_slice]


@_cached_jacobian
def _jacobian_via_jvp(
    endpoints_func: dict[str, Callable],
    inputs: dict[str, Any],
    input_path: Sequence[str],
    output_path: Sequence[str],
    input_idx: tuple[int, ...],
) -> ArrayLike:
    """Compute a Jacobian row using the jacobian_vector_product endpoint."""
    jvp_fn = endpoints_func["jacobian_vector_product"]
    JvpSchema = get_input_schema(jvp_fn)

    tangent = np.zeros_like(get_at_path(inputs, input_path))
    tangent[input_idx] = 1
    jvp = jvp_fn(
        JvpSchema.model_validate(
            {
                "inputs": inputs,
                "jvp_inputs": [input_path],
                "jvp_outputs": [output_path],
                "tangent_vector": {input_path: tangent},
            }
        )
    ).model_dump()
    return jvp[output_path]


@_cached_jacobian
def _jacobian_via_vjp(
    endpoints_func: dict[str, Callable],
    inputs: dict[str, Any],
    input_path: Sequence[str],
    output_path: Sequence[str],
    input_idx: tuple[int, ...],
) -> ArrayLike:
    """Compute a Jacobian row using the vector_jacobian_product endpoint."""
    apply_fn = endpoints_func["apply"]
    ApplySchema = get_input_schema(apply_fn)
    outputs = apply_fn(ApplySchema.model_validate({"inputs": inputs})).model_dump()

    vjp_fn = endpoints_func["vector_jacobian_product"]
    VjpSchema = get_input_schema(vjp_fn)
    jac_row = np.zeros_like(get_at_path(outputs, output_path))

    for col_idx in np.ndindex(jac_row.shape):
        cotangent = np.zeros_like(jac_row)
        cotangent[col_idx] = 1
        vjp = vjp_fn(
            VjpSchema.model_validate(
                {
                    "inputs": inputs,
                    "vjp_inputs": [input_path],
                    "vjp_outputs": [output_path],
                    "cotangent_vector": {output_path: cotangent},
                }
            )
        ).model_dump()
        jac_row[col_idx] = vjp[input_path][input_idx]

    return jac_row


def _sample_indices(
    inputs: dict[str, Any],
    diff_inputs: list[str],
    diff_outputs: list[str],
    max_evals: int,
    rng: np.random.RandomState,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Sample combinations of (input_path, output_path, row_idx) to check.

    row_idx are sampled at random, proportional to the size of the input.
    """
    input_shapes = {path: get_at_path(inputs, path).shape for path in diff_inputs}
    total_elements = sum(np.prod(shape) for shape in input_shapes.values())

    idx_per_input = {}
    for path, shape in input_shapes.items():
        if not shape:
            idx_per_input[path] = [()]
            continue
        n_evals = max(1, int(max_evals * np.prod(shape) / total_elements))
        idx_tuple = np.unravel_index(rng.choice(int(np.prod(shape)), n_evals), shape)
        idx_per_input[path] = list(zip(*idx_tuple))

    items_to_check = []
    for in_path in diff_inputs:
        for idx in idx_per_input[in_path]:
            idx = tuple(int(i) for i in idx)
            for out_path in diff_outputs:
                items_to_check.append((in_path, out_path, idx))

    return items_to_check


def check_endpoint_gradients(
    endpoint_functions: dict[str, Callable],
    inputs: dict[str, Any],
    endpoint: str,
    *,
    diff_inputs: list[str],
    diff_outputs: list[str],
    max_evals: int,
    eps: float,
    rtol: float,
    rng: np.random.RandomState,
    show_progress: bool,
) -> tuple[list[GradientCheckResult], int]:
    """Check gradients of an endpoint against a finite difference approximation."""
    failures = []

    if endpoint == "jacobian":
        _jacobian_via_grad = _jacobian_via_jacobian
    elif endpoint == "jacobian_vector_product":
        _jacobian_via_grad = _jacobian_via_jvp
    elif endpoint == "vector_jacobian_product":
        _jacobian_via_grad = _jacobian_via_vjp
    else:
        raise AssertionError(f"Unknown endpoint {endpoint}")

    items_to_check = _sample_indices(inputs, diff_inputs, diff_outputs, max_evals, rng)
    num_evals = 0

    try:
        with Progress(disable=not show_progress) as progress:
            subtask = progress.add_task(
                f"Checking gradients for {endpoint}...", total=len(items_to_check)
            )

            for in_path, out_path, idx in items_to_check:
                num_evals += 1

                failure = None
                try:
                    result_apply = _jacobian_via_apply(
                        endpoint_functions,
                        inputs,
                        in_path,
                        out_path,
                        idx,
                        eps=eps,
                    )
                    result_grad = _jacobian_via_grad(
                        endpoint_functions,
                        inputs,
                        in_path,
                        out_path,
                        idx,
                    )
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)
                    exc_info = f"{type(e).__name__}: '{e}' in file {tb[-1].filename}, line {tb[-1].lineno}"
                    failure = GradientCheckResult(
                        in_path=in_path,
                        out_path=out_path,
                        idx=idx,
                        ref_val=None,
                        grad_val=None,
                        exception=exc_info,
                    )
                else:
                    if not np.allclose(result_apply, result_grad, atol=1e-8, rtol=rtol):
                        failure = GradientCheckResult(
                            in_path=in_path,
                            out_path=out_path,
                            idx=idx,
                            ref_val=result_apply,
                            grad_val=result_grad,
                            exception=None,
                        )

                if failure:
                    failures.append(failure)
                    progress.update(
                        subtask,
                        description=f"Checking gradients for {endpoint}... (failures: {len(failures)})",
                    )

                progress.update(subtask, advance=1)

    except BaseException as e:
        # Sometimes, Pydantic re-raises exceptions as Pydantic<...>Exception so we check the string representation
        is_interrupt = isinstance(e, KeyboardInterrupt) or "KeyboardInterrupt" in str(e)
        if not is_interrupt:
            raise
        print("Interrupted")

    return failures, num_evals


def check_gradients(
    api_module: ModuleType,
    inputs: dict[str, Any],
    *,
    input_paths: Optional[Sequence[str]] = None,
    output_paths: Optional[Sequence[str]] = None,
    base_dir: Optional[Path] = None,
    endpoints: Optional[Sequence[ADEndpointName]] = None,
    max_evals: int = 1000,
    eps: float = 1e-4,
    rtol: float = 0.1,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> Iterator[tuple[str, list[GradientCheckResult], int]]:
    """Check gradients of endpoints against a finite difference approximation.

    Args:
        api_module: The module containing the Tesseract endpoints.
        inputs: The inputs to apply to evaluate gradients at.
        input_paths: The input paths to check. If not provided, all differentiable paths are checked.
        output_paths: The output paths to check. If not provided, all differentiable paths are checked.
        base_dir: The base directory to resolve relative paths.
        endpoints: The AD endpoints to check. If not provided, all available endpoints are checked.
        max_evals: The target number of ``apply`` evaluations to perform.
        eps: The epsilon to use for finite differences, as a fraction of the maximum absolute value of each input.
        rtol: The relative tolerance to use for comparison.
        seed: The random seed to use for sampling. If not provided, a random seed is used.
        show_progress: Whether to show a progress bar.
    """
    # We apply a global cache to these functions to avoid hashing `inputs` multiple times,
    # so we need to clear the cache before each run.
    _jacobian_via_apply.clear_cache()
    _jacobian_via_jacobian.clear_cache()
    _jacobian_via_jvp.clear_cache()
    _jacobian_via_vjp.clear_cache()

    # Get available endpoints
    endpoint_functions = {func.__name__: func for func in create_endpoints(api_module)}
    available_endpoints = [
        func_name
        for func_name in endpoint_functions
        if func_name in get_args(ADEndpointName)
    ]

    if not available_endpoints:
        raise ValueError(f"No AD endpoints found in {api_module.__name__}")

    if not endpoints:
        endpoints = available_endpoints

    for endpoint in endpoints:
        if endpoint not in available_endpoints:
            raise ValueError(f"Endpoint {endpoint} not found in {api_module.__name__}")

    # Load + dump inputs to ensure they are valid + normalized
    InputSchema = get_input_schema(endpoint_functions["apply"])
    loaded_inputs = InputSchema.model_validate(inputs, context={"base_dir": base_dir})
    inputs = loaded_inputs.inputs.model_dump()
    outputs = endpoint_functions["apply"](loaded_inputs).model_dump()

    # Get differentiable paths
    diff_inputs, diff_outputs = get_differentiable_paths(
        endpoint_functions["apply"],
        inputs,
        outputs,
    )

    if not input_paths:
        input_paths = diff_inputs

    for path in input_paths:
        if path not in diff_inputs:
            raise ValueError(
                f"Input path {path} not found in differentiable paths ({diff_inputs})"
            )

    if not output_paths:
        output_paths = diff_outputs

    for path in output_paths:
        if path not in diff_outputs:
            raise ValueError(
                f"Output path {path} not found in differentiable paths ({diff_outputs})"
            )

    # Check gradients for each endpoint separately
    rng = np.random.RandomState(seed)

    for endpoint in endpoints:
        failures, num_evals = check_endpoint_gradients(
            endpoint_functions,
            inputs,
            endpoint,
            diff_inputs=input_paths,
            diff_outputs=output_paths,
            max_evals=max_evals,
            eps=eps,
            rtol=rtol,
            rng=rng,
            show_progress=show_progress,
        )
        yield endpoint, failures, num_evals
