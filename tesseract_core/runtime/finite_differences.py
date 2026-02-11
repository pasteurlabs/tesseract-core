# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback
from collections.abc import Callable, Iterator, Sequence
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    Literal,
    NamedTuple,
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
    grad_val: ArrayLike | None
    ref_val: ArrayLike | None
    exception: str | None


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


def _perturb_input(
    inputs_dict: dict[str, Any],
    input_path: str,
    idx: tuple[int, ...],
    eps: float,
) -> dict[str, Any]:
    """Perturb a single element of an input array by eps.

    Args:
        inputs_dict: The input dictionary to perturb.
        input_path: Path to the input array to perturb.
        idx: Index within the array to perturb (empty tuple for scalars).
        eps: Perturbation magnitude (can be negative).

    Returns:
        A new dictionary with the perturbed input.
    """
    input_val = np.asarray(get_at_path(inputs_dict, input_path)).copy()
    if idx:
        input_val[idx] += eps
    else:
        input_val = input_val + eps
    return set_at_path(inputs_dict, {input_path: input_val})


def _compute_central_diff_row(
    apply_fn: Callable,
    inputs_dict: dict[str, Any],
    input_schema: type[BaseModel],
    input_path: str,
    output_path: str,
    idx: tuple[int, ...],
    eps: float,
) -> ArrayLike:
    """Compute a single Jacobian row using central finite differences.

    This is the core computation shared between gradient checking and the
    finite_difference_jacobian helper.

    Args:
        apply_fn: Function that takes validated inputs and returns outputs.
        inputs_dict: Dictionary of input values.
        input_schema: Pydantic schema for validating inputs.
        input_path: Path to the input being differentiated.
        output_path: Path to the output to differentiate.
        idx: Index within the input array (empty tuple for scalars).
        eps: Perturbation magnitude.

    Returns:
        The gradient of output_path with respect to input_path[idx].
    """
    inputs_plus = _perturb_input(inputs_dict, input_path, idx, eps)
    inputs_minus = _perturb_input(inputs_dict, input_path, idx, -eps)

    outputs_plus = apply_fn(input_schema.model_validate(inputs_plus)).model_dump()
    outputs_minus = apply_fn(input_schema.model_validate(inputs_minus)).model_dump()

    return (
        get_at_path(outputs_plus, output_path) - get_at_path(outputs_minus, output_path)
    ) / (2 * eps)


def _compute_forward_diff_row(
    apply_fn: Callable,
    inputs_dict: dict[str, Any],
    base_outputs: dict[str, Any],
    input_schema: type[BaseModel],
    input_path: str,
    output_path: str,
    idx: tuple[int, ...],
    eps: float,
) -> ArrayLike:
    """Compute a single Jacobian row using forward finite differences.

    Args:
        apply_fn: Function that takes validated inputs and returns outputs.
        inputs_dict: Dictionary of input values.
        base_outputs: Pre-computed outputs at the base point.
        input_schema: Pydantic schema for validating inputs.
        input_path: Path to the input being differentiated.
        output_path: Path to the output to differentiate.
        idx: Index within the input array (empty tuple for scalars).
        eps: Perturbation magnitude.

    Returns:
        The gradient of output_path with respect to input_path[idx].
    """
    inputs_plus = _perturb_input(inputs_dict, input_path, idx, eps)
    outputs_plus = apply_fn(input_schema.model_validate(inputs_plus)).model_dump()

    return (
        get_at_path(outputs_plus, output_path) - get_at_path(base_outputs, output_path)
    ) / eps


@_cached_jacobian
def _jacobian_via_apply(
    endpoints_func: dict[str, Callable],
    inputs: dict[str, Any],
    input_path: str,
    output_path: str,
    input_idx: tuple[int, ...],
    eps: float = 1e-4,
) -> ArrayLike:
    """Compute a Jacobian row using central finite differences."""
    apply_fn = endpoints_func["apply"]
    ApplySchema = get_input_schema(apply_fn)

    # Wrap the apply function to match expected signature
    def wrapped_apply(validated_inputs: Any) -> Any:
        return apply_fn(validated_inputs)

    # Create a schema that wraps inputs in {"inputs": ...}
    class WrappedSchema(BaseModel):
        inputs: dict

        @classmethod
        def model_validate(cls, obj: Any) -> Any:
            return ApplySchema.model_validate({"inputs": obj})

    return _compute_central_diff_row(
        wrapped_apply, inputs, WrappedSchema, input_path, output_path, input_idx, eps
    )


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
        idx_per_input[path] = list(zip(*idx_tuple, strict=True))

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
    input_paths: Sequence[str] | None = None,
    output_paths: Sequence[str] | None = None,
    base_dir: Path | None = None,
    endpoints: Sequence[ADEndpointName] | None = None,
    max_evals: int = 1000,
    eps: float = 1e-4,
    rtol: float = 0.1,
    seed: int | None = None,
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


FDAlgorithm = Literal["central", "forward", "stochastic"]


def finite_difference_jacobian(
    apply_fn: Callable,
    inputs: BaseModel,
    jac_inputs: set[str],
    jac_outputs: set[str],
    *,
    algorithm: FDAlgorithm = "central",
    eps: float = 1e-4,
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Compute the Jacobian of a Tesseract apply function using finite differences.

    This function provides a generic way to make any Tesseract differentiable
    by computing gradients numerically. It can be used directly as the implementation
    of a ``jacobian`` endpoint.

    Args:
        apply_fn: The Tesseract's apply function with signature ``apply(inputs) -> outputs``.
        inputs: The input data at which to compute the Jacobian.
        jac_inputs: Set of input paths to differentiate with respect to.
        jac_outputs: Set of output paths to compute derivatives of.
        algorithm: The finite difference algorithm to use:
            - ``"central"``: Central differences ``(f(x+eps) - f(x-eps)) / (2*eps)``.
              Most accurate but requires 2 function evaluations per input element.
            - ``"forward"``: Forward differences ``(f(x+eps) - f(x)) / eps``.
              Less accurate but requires only 1 extra function evaluation per input element.
            - ``"stochastic"``: Simultaneous Perturbation Stochastic Approximation (SPSA).
              Approximates the full Jacobian using random perturbation directions.
              Scales better to high-dimensional inputs. See:
              Spall, J. C. (1992). Multivariate stochastic approximation using a
              simultaneous perturbation gradient approximation. IEEE Transactions
              on Automatic Control, 37(3), 332-341.
        eps: Perturbation magnitude for finite differences.
        num_samples: Number of random samples for the stochastic algorithm.
            Only used when ``algorithm="stochastic"``. Defaults to the total number
            of input elements if not specified.
        seed: Random seed for reproducibility (only used with ``algorithm="stochastic"``).

    Returns:
        A nested dictionary with structure ``{output_path: {input_path: jacobian_array}}``,
        where each jacobian_array has shape ``(*output_shape, *input_shape)``.

    Example:
        In a Tesseract's ``tesseract_api.py``::

            from tesseract_core.runtime import finite_difference_jacobian


            def jacobian(
                inputs: InputSchema,
                jac_inputs: set[str],
                jac_outputs: set[str],
            ):
                return finite_difference_jacobian(
                    apply, inputs, jac_inputs, jac_outputs
                )
    """
    inputs_dict = inputs.model_dump()

    # Get reference outputs for shape information and for forward differences
    base_outputs = apply_fn(inputs).model_dump()

    # Build the result structure
    result: dict[str, dict[str, ArrayLike]] = {}
    for out_path in jac_outputs:
        result[out_path] = {}
        out_val = get_at_path(base_outputs, out_path)
        out_shape = np.asarray(out_val).shape

        for in_path in jac_inputs:
            in_val = get_at_path(inputs_dict, in_path)
            in_arr = np.asarray(in_val)
            in_shape = in_arr.shape

            # Initialize Jacobian with shape (*output_shape, *input_shape)
            jac_shape = (*out_shape, *in_shape) if in_shape else out_shape
            result[out_path][in_path] = np.zeros(jac_shape, dtype=np.float64)

    if algorithm == "stochastic":
        _compute_jacobian_stochastic(
            apply_fn,
            inputs,
            inputs_dict,
            base_outputs,
            jac_inputs,
            jac_outputs,
            result,
            eps=eps,
            num_samples=num_samples,
            seed=seed,
        )
    else:
        _compute_jacobian_elementwise(
            apply_fn,
            inputs,
            inputs_dict,
            base_outputs,
            jac_inputs,
            jac_outputs,
            result,
            eps=eps,
            use_central=(algorithm == "central"),
        )

    return result


def _compute_jacobian_elementwise(
    apply_fn: Callable,
    inputs: BaseModel,
    inputs_dict: dict,
    base_outputs: dict,
    jac_inputs: set[str],
    jac_outputs: set[str],
    result: dict[str, dict[str, ArrayLike]],
    *,
    eps: float,
    use_central: bool,
) -> None:
    """Compute Jacobian by perturbing each input element individually."""
    input_schema = type(inputs)

    for in_path in jac_inputs:
        in_val = get_at_path(inputs_dict, in_path)
        in_arr = np.asarray(in_val)
        in_shape = in_arr.shape

        # Handle scalars
        indices = list(np.ndindex(in_shape)) if in_shape else [()]

        for idx in indices:
            for out_path in jac_outputs:
                if use_central:
                    grad = _compute_central_diff_row(
                        apply_fn,
                        inputs_dict,
                        input_schema,
                        in_path,
                        out_path,
                        idx,
                        eps,
                    )
                else:
                    grad = _compute_forward_diff_row(
                        apply_fn,
                        inputs_dict,
                        base_outputs,
                        input_schema,
                        in_path,
                        out_path,
                        idx,
                        eps,
                    )

                if idx:
                    result[out_path][in_path][(Ellipsis, *idx)] = grad
                else:
                    result[out_path][in_path][...] = grad


def _compute_jacobian_stochastic(
    apply_fn: Callable,
    inputs: BaseModel,
    inputs_dict: dict,
    base_outputs: dict,
    jac_inputs: set[str],
    jac_outputs: set[str],
    result: dict[str, dict[str, ArrayLike]],
    *,
    eps: float,
    num_samples: int | None,
    seed: int | None,
) -> None:
    """Compute Jacobian using Simultaneous Perturbation Stochastic Approximation (SPSA).

    This algorithm estimates the Jacobian by:
    1. Generating random perturbation directions (Rademacher distributed: ±1)
    2. Computing the gradient approximation using these directions
    3. Averaging over multiple samples to reduce variance

    The key insight is that SPSA requires only 2 function evaluations per sample,
    regardless of the input dimension, making it efficient for high-dimensional inputs.
    """
    rng = np.random.RandomState(seed)

    # Collect all input arrays and their metadata
    input_info = {}
    total_input_elements = 0
    for in_path in jac_inputs:
        in_val = get_at_path(inputs_dict, in_path)
        in_arr = np.asarray(in_val)
        input_info[in_path] = {
            "array": in_arr,
            "shape": in_arr.shape,
            "size": in_arr.size if in_arr.shape else 1,
        }
        total_input_elements += input_info[in_path]["size"]

    # Default number of samples
    if num_samples is None:
        num_samples = total_input_elements

    # Collect output shapes
    output_info = {}
    for out_path in jac_outputs:
        out_val = get_at_path(base_outputs, out_path)
        out_arr = np.asarray(out_val)
        output_info[out_path] = {
            "shape": out_arr.shape,
            "size": out_arr.size if out_arr.shape else 1,
        }

    # Accumulate Jacobian estimates
    for _ in range(num_samples):
        # Generate random perturbation directions (Rademacher: ±1)
        perturbations = {}
        for in_path, info in input_info.items():
            if info["shape"]:
                perturbations[in_path] = rng.choice([-1, 1], size=info["shape"]).astype(
                    np.float64
                )
            else:
                perturbations[in_path] = np.float64(rng.choice([-1, 1]))

        # Compute perturbed inputs
        inputs_plus_dict = inputs_dict.copy()
        inputs_minus_dict = inputs_dict.copy()

        for in_path, delta in perturbations.items():
            in_arr = input_info[in_path]["array"]
            inputs_plus_dict = set_at_path(
                inputs_plus_dict, {in_path: in_arr + eps * delta}
            )
            inputs_minus_dict = set_at_path(
                inputs_minus_dict, {in_path: in_arr - eps * delta}
            )

        # Evaluate function at perturbed points
        outputs_plus = apply_fn(
            type(inputs).model_validate(inputs_plus_dict)
        ).model_dump()
        outputs_minus = apply_fn(
            type(inputs).model_validate(inputs_minus_dict)
        ).model_dump()

        # Update Jacobian estimate for each (output, input) pair
        for out_path in jac_outputs:
            out_plus = np.asarray(get_at_path(outputs_plus, out_path))
            out_minus = np.asarray(get_at_path(outputs_minus, out_path))
            output_diff = (out_plus - out_minus) / (2 * eps)

            for in_path in jac_inputs:
                delta = perturbations[in_path]

                # SPSA gradient estimate: (f(x+eps*delta) - f(x-eps*delta)) / (2*eps*delta)
                # For Jacobian: J[i,j] contribution = output_diff[i] / delta[j]
                # We accumulate and average over samples
                if input_info[in_path]["shape"]:
                    # For each output element, divide by each input perturbation
                    # Result shape: (*output_shape, *input_shape)
                    jac_contrib = np.outer(output_diff.ravel(), 1.0 / delta.ravel())
                    jac_contrib = jac_contrib.reshape(
                        *output_info[out_path]["shape"], *input_info[in_path]["shape"]
                    )
                else:
                    # Scalar input
                    jac_contrib = output_diff / delta

                result[out_path][in_path] += jac_contrib / num_samples


def finite_difference_jvp(
    apply_fn: Callable,
    inputs: BaseModel,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, ArrayLike],
    *,
    algorithm: Literal["central", "forward"] = "central",
    eps: float = 1e-4,
) -> dict[str, ArrayLike]:
    """Compute the Jacobian-vector product (JVP) using finite differences.

    The JVP computes ``J @ v`` where ``J`` is the Jacobian and ``v`` is the tangent vector.
    This is done efficiently using directional derivatives without computing the full Jacobian.

    Args:
        apply_fn: The Tesseract's apply function with signature ``apply(inputs) -> outputs``.
        inputs: The input data at which to compute the JVP.
        jvp_inputs: Set of input paths to differentiate with respect to.
        jvp_outputs: Set of output paths to compute derivatives of.
        tangent_vector: Dictionary mapping input paths to tangent arrays.
        algorithm: The finite difference algorithm (``"central"`` or ``"forward"``).
        eps: Perturbation magnitude.

    Returns:
        Dictionary mapping output paths to JVP result arrays.

    Example:
        In a Tesseract's ``tesseract_api.py``::

            from tesseract_core.runtime import finite_difference_jvp


            def jacobian_vector_product(
                inputs: InputSchema,
                jvp_inputs: set[str],
                jvp_outputs: set[str],
                tangent_vector: dict[str, Any],
            ):
                return finite_difference_jvp(
                    apply, inputs, jvp_inputs, jvp_outputs, tangent_vector
                )
    """
    inputs_dict = inputs.model_dump()

    # Construct directional perturbation
    inputs_plus_dict = inputs_dict.copy()
    inputs_minus_dict = inputs_dict.copy()

    for in_path in jvp_inputs:
        in_val = np.asarray(get_at_path(inputs_dict, in_path))
        tangent = np.asarray(tangent_vector[in_path])

        inputs_plus_dict = set_at_path(
            inputs_plus_dict, {in_path: in_val + eps * tangent}
        )
        if algorithm == "central":
            inputs_minus_dict = set_at_path(
                inputs_minus_dict, {in_path: in_val - eps * tangent}
            )

    # Evaluate at perturbed points
    outputs_plus = apply_fn(type(inputs).model_validate(inputs_plus_dict)).model_dump()

    if algorithm == "central":
        outputs_minus = apply_fn(
            type(inputs).model_validate(inputs_minus_dict)
        ).model_dump()

        result = {}
        for out_path in jvp_outputs:
            out_plus = np.asarray(get_at_path(outputs_plus, out_path))
            out_minus = np.asarray(get_at_path(outputs_minus, out_path))
            result[out_path] = (out_plus - out_minus) / (2 * eps)
    else:
        base_outputs = apply_fn(inputs).model_dump()
        result = {}
        for out_path in jvp_outputs:
            out_plus = np.asarray(get_at_path(outputs_plus, out_path))
            out_base = np.asarray(get_at_path(base_outputs, out_path))
            result[out_path] = (out_plus - out_base) / eps

    return result


def finite_difference_vjp(
    apply_fn: Callable,
    inputs: BaseModel,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, ArrayLike],
    *,
    algorithm: Literal["central", "forward"] = "central",
    eps: float = 1e-4,
) -> dict[str, ArrayLike]:
    """Compute the vector-Jacobian product (VJP) using finite differences.

    The VJP computes ``v @ J`` where ``J`` is the Jacobian and ``v`` is the cotangent vector.
    Unlike JVP, this requires computing the relevant rows of the Jacobian explicitly.

    Args:
        apply_fn: The Tesseract's apply function with signature ``apply(inputs) -> outputs``.
        inputs: The input data at which to compute the VJP.
        vjp_inputs: Set of input paths to differentiate with respect to.
        vjp_outputs: Set of output paths to compute derivatives of.
        cotangent_vector: Dictionary mapping output paths to cotangent arrays.
        algorithm: The finite difference algorithm (``"central"`` or ``"forward"``).
        eps: Perturbation magnitude.

    Returns:
        Dictionary mapping input paths to VJP result arrays.

    Example:
        In a Tesseract's ``tesseract_api.py``::

            from tesseract_core.runtime import finite_difference_vjp


            def vector_jacobian_product(
                inputs: InputSchema,
                vjp_inputs: set[str],
                vjp_outputs: set[str],
                cotangent_vector: dict[str, Any],
            ):
                return finite_difference_vjp(
                    apply, inputs, vjp_inputs, vjp_outputs, cotangent_vector
                )
    """
    inputs_dict = inputs.model_dump()
    input_schema = type(inputs)
    base_outputs = apply_fn(inputs).model_dump()

    # Initialize result
    result: dict[str, np.ndarray] = {}
    for in_path in vjp_inputs:
        in_val = get_at_path(inputs_dict, in_path)
        in_arr = np.asarray(in_val)
        result[in_path] = np.zeros_like(in_arr, dtype=np.float64)

    # VJP = sum over outputs of cotangent[output] @ J[output, input]
    # We need to compute each row of J (one per input element) and contract with cotangent
    for in_path in vjp_inputs:
        in_val = get_at_path(inputs_dict, in_path)
        in_arr = np.asarray(in_val)
        in_shape = in_arr.shape

        indices = list(np.ndindex(in_shape)) if in_shape else [()]

        for idx in indices:
            # Compute gradient and contract with cotangent for each output
            vjp_value = 0.0
            for out_path in vjp_outputs:
                if algorithm == "central":
                    grad = _compute_central_diff_row(
                        apply_fn,
                        inputs_dict,
                        input_schema,
                        in_path,
                        out_path,
                        idx,
                        eps,
                    )
                else:
                    grad = _compute_forward_diff_row(
                        apply_fn,
                        inputs_dict,
                        base_outputs,
                        input_schema,
                        in_path,
                        out_path,
                        idx,
                        eps,
                    )
                cotangent = np.asarray(cotangent_vector[out_path])
                vjp_value += np.sum(cotangent * grad)

            if idx:
                result[in_path][idx] = vjp_value
            else:
                result[in_path] = np.float64(vjp_value)

    return result
