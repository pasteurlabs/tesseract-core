# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
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

from ..core import create_endpoints, get_input_schema, get_output_schema
from ..tree_transforms import escape_dict_key, get_at_path, set_at_path, split_path

GradientEndpointName = Literal[
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


def expand_path_pattern(path_pattern: str, inputs: dict[str, Any]) -> list[str]:
    """Expand a path pattern to a list of all matching paths in the given pytree.

    For example, given the path pattern `a.[].{}`, and the inputs `{"a": [{"b": 1}, {"c": 2}]}`,
    this function would return `["a.[0].{b}", "a.[1].{c}"]`.
    """
    parts = split_path(path_pattern)

    def _handle_part(
        parts: Sequence[str], current_inputs: Any, current_path: list[str]
    ) -> list[str]:
        """Recursively expand each part separately."""
        if not parts:
            return [".".join(current_path)]

        if current_inputs is None:
            # An optional container (e.g. `list | None`) that was not supplied.
            return []

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
                    parts[1:],
                    current_inputs[key],
                    [*current_path, f"{{{escape_dict_key(str(key))}}}"],
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


def _by_index(input_path: Any, output_path: Any, input_idx: Any) -> tuple:
    """Key a per-index Jacobian row on the triple it depends on."""
    return (input_path, output_path, tuple(input_idx))


class _VjpSweepTarget(NamedTuple):
    """Target specification for a shared VJP sweep.

    When ``sampled_outputs`` is None, all output elements are evaluated
    exhaustively. When it is a tuple of coordinates, only those coordinates are
    evaluated.
    """

    wanted_inputs: tuple[tuple[int, ...], ...]
    sampled_outputs: tuple[tuple[int, ...], ...] | None = None


def _by_sampled_set(
    input_path: Any,
    output_path: Any,
    target: _VjpSweepTarget,
) -> tuple:
    """Key a VJP sweep on the path pair and the sampled input/output set it answers for."""
    return (
        input_path,
        output_path,
        tuple(target.wanted_inputs),
        target.sampled_outputs,
    )


def _cached_function(*, key_fn: Callable) -> Callable:
    """Memoise on a key derived from the call's path arguments.

    ``key_fn`` receives ``(input_path, output_path, fifth_positional)`` and
    returns the cache key, so the caller decides the granularity.
    """

    def _decorate(fn: Callable) -> Callable:
        cache = {}

        @wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            _, _, input_path, output_path, fifth, *_ = args
            key = key_fn(input_path, output_path, fifth)
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

    return _decorate


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


@_cached_function(key_fn=_by_index)
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


@_cached_function(key_fn=_by_index)
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


@_cached_function(key_fn=_by_index)
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


@_cached_function(key_fn=_by_sampled_set)
def _vjp_sweep(
    endpoints_func: dict[str, Callable],
    inputs: dict[str, Any],
    input_path: Sequence[str],
    output_path: Sequence[str],
    target: _VjpSweepTarget,
    outputs: dict[str, Any],
) -> dict[tuple[int, ...], ArrayLike]:
    """Sweep one-hot cotangents over the output and keep the wanted rows.

    One VJP call with a one-hot cotangent returns the gradient with respect
    to every element of ``input_path``, so a single sweep over the output
    elements answers for all sampled indices at once.

    When ``target.sampled_outputs`` is None, all output elements are swept
    exhaustively. When it is a tuple of coordinates, only those coordinates are
    evaluated.
    """
    vjp_fn = endpoints_func["vector_jacobian_product"]
    VjpSchema = get_input_schema(vjp_fn)
    template = np.zeros_like(get_at_path(outputs, output_path))

    if target.sampled_outputs is None:
        rows = {idx: np.zeros_like(template) for idx in target.wanted_inputs}
        for col_idx in np.ndindex(template.shape):
            cotangent = np.zeros_like(template)
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
            grad = vjp[input_path]
            for idx in target.wanted_inputs:
                rows[idx][col_idx] = grad[idx]
        return rows

    rows = {
        idx: np.zeros(len(target.sampled_outputs), dtype=template.dtype)
        for idx in target.wanted_inputs
    }
    for col_pos, col_idx in enumerate(target.sampled_outputs):
        cotangent = np.zeros_like(template)
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
        grad = vjp[input_path]
        for idx in target.wanted_inputs:
            rows[idx][col_pos] = grad[idx]
    return rows


def _jacobian_via_vjp(
    endpoints_func: dict[str, Callable],
    inputs: dict[str, Any],
    input_path: Sequence[str],
    output_path: Sequence[str],
    input_idx: tuple[int, ...],
    outputs: dict[str, Any],
    sampled_input_idx: tuple[tuple[int, ...], ...] = (),
    sampled_output_idx: tuple[tuple[int, ...], ...] | None = None,
) -> ArrayLike:
    """Return one Jacobian row from the sweep shared by this path pair.

    ``sampled_input_idx`` is the set this pair will be asked for. It goes
    into the cache key so that one sweep serves every index in it, which is
    the granularity the sweep actually answers at, while callers keep asking
    for a single row like the other three helpers.
    """
    wanted = tuple(dict.fromkeys((*sampled_input_idx, tuple(input_idx))))
    target = _VjpSweepTarget(wanted_inputs=wanted, sampled_outputs=sampled_output_idx)
    return _vjp_sweep(endpoints_func, inputs, input_path, output_path, target, outputs)[
        tuple(input_idx)
    ]


_jacobian_via_vjp.clear_cache = _vjp_sweep.clear_cache


def _sample_coordinates(
    shape: tuple[int, ...],
    count: int,
    rng: np.random.RandomState,
    *,
    replace: bool = True,
) -> list[tuple[int, ...]]:
    """Sample coordinate tuples from an array of the given shape.

    Args:
        shape: Shape of the array to sample from.
        count: Number of coordinates to sample.
        rng: Random number generator to use.
        replace: Whether to sample with or without replacement.

    Returns:
        List of coordinate tuples within the given shape.
    """
    if not shape:
        return [()] * count
    total_elements = int(np.prod(shape, dtype=int))
    if total_elements == 0:
        return []
    flat_indices = rng.choice(total_elements, size=count, replace=replace)
    unraveled = np.unravel_index(flat_indices, shape)
    coords = zip(*(tuple(int(x) for x in dim) for dim in unraveled), strict=True)
    return list(coords)


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
        idx_per_input[path] = _sample_coordinates(shape, n_evals, rng, replace=True)

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
    outputs: dict[str, Any],
    *,
    diff_inputs: list[str],
    diff_outputs: list[str],
    max_evals: int,
    max_output_samples: int | None = None,
    eps: float | Mapping[str, float],
    rtol: float,
    rng: np.random.RandomState,
    output_rng: np.random.RandomState | None = None,
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

    # Indices this run will ask for, per path pair. The VJP sweep answers for
    # all of them in one pass, so it needs to know them up front.
    sampled_by_pair: dict[tuple[str, str], tuple] = {}
    for in_path, out_path, idx in items_to_check:
        sampled_by_pair.setdefault((in_path, out_path), ())
        sampled_by_pair[(in_path, out_path)] += (tuple(idx),)

    sampled_outputs_by_pair: dict[
        tuple[str, str], tuple[tuple[int, ...], ...] | None
    ] = {}
    if endpoint == "vector_jacobian_product" and max_output_samples is not None:
        actual_output_rng = output_rng if output_rng is not None else rng
        for in_path, out_path in sampled_by_pair:
            out_val = get_at_path(outputs, out_path)
            out_shape = np.shape(out_val)
            n_output_elements = int(np.prod(out_shape, dtype=int)) if out_shape else 1
            if max_output_samples < n_output_elements:
                sampled_outputs_by_pair[(in_path, out_path)] = tuple(
                    _sample_coordinates(
                        out_shape,
                        max_output_samples,
                        actual_output_rng,
                        replace=False,
                    )
                )
            else:
                sampled_outputs_by_pair[(in_path, out_path)] = None
    else:
        for in_path, out_path in sampled_by_pair:
            sampled_outputs_by_pair[(in_path, out_path)] = None

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
                        eps=eps[in_path] if isinstance(eps, Mapping) else eps,
                    )
                    if endpoint == "vector_jacobian_product":
                        col_indices = sampled_outputs_by_pair.get((in_path, out_path))
                        grad_kwargs = {
                            "outputs": outputs,
                            "sampled_input_idx": sampled_by_pair[(in_path, out_path)],
                            "sampled_output_idx": col_indices,
                        }
                    else:
                        col_indices = None
                        grad_kwargs = {}
                    result_grad = _jacobian_via_grad(
                        endpoint_functions,
                        inputs,
                        in_path,
                        out_path,
                        idx,
                        **grad_kwargs,
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
                    if col_indices is not None:
                        reference = np.asarray(result_apply)
                        ref_val = np.asarray([reference[c] for c in col_indices])
                        grad_val = np.asarray(result_grad)
                    else:
                        ref_val = result_apply
                        grad_val = result_grad

                    if not np.allclose(ref_val, grad_val, atol=1e-8, rtol=rtol):
                        failure = GradientCheckResult(
                            in_path=in_path,
                            out_path=out_path,
                            idx=idx,
                            ref_val=ref_val,
                            grad_val=grad_val,
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
    endpoints: Sequence[GradientEndpointName] | None = None,
    max_evals: int = 1000,
    max_output_samples: int | None = None,
    eps: float | Mapping[str, float] = 1e-4,
    rtol: float = 0.1,
    seed: int | None = None,
    show_progress: bool = True,
) -> Iterator[tuple[str, list[GradientCheckResult], int]]:
    """Returns an iterator that checks gradients of endpoints against a finite difference approximation.

    Args:
        api_module: The module containing the Tesseract endpoints.
        inputs: The inputs to apply to evaluate gradients at.
        input_paths: The input paths to check. If not provided, all differentiable paths are checked.
        output_paths: The output paths to check. If not provided, all differentiable paths are checked.
        base_dir: The base directory to resolve relative paths.
        endpoints: The gradient endpoints to check. If not provided, all available endpoints are checked.
        max_evals: The target number of ``apply`` evaluations to perform.
        max_output_samples: Maximum number of output elements sampled when checking the
            vector_jacobian_product endpoint. If None, all output elements are checked
            exhaustively.
        eps: The step size to use for finite differences, as an absolute
            perturbation. A single float is applied unscaled to every
            differentiated input; a mapping gives one step per input path,
            which is what inputs of differing magnitude need, and must name
            every path being checked.
        rtol: The relative tolerance to use for comparison.
        seed: The random seed to use for sampling. If not provided, a random seed is used.
        show_progress: Whether to show a progress bar.
    """
    if max_output_samples is not None and max_output_samples <= 0:
        raise ValueError("max_output_samples must be greater than 0")

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
        if func_name in get_args(GradientEndpointName)
    ]

    if not available_endpoints:
        raise ValueError(f"No gradient endpoints found in {api_module.__name__}")

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

    if isinstance(eps, Mapping):
        missing = [path for path in input_paths if path not in eps]
        if missing:
            raise ValueError(
                f"eps is missing a step size for input path(s): {', '.join(missing)}"
            )
        unknown = [path for path in eps if path not in input_paths]
        if unknown:
            raise ValueError(
                f"eps names input path(s) that are not being checked: "
                f"{', '.join(unknown)}"
            )

    for path in output_paths:
        if path not in diff_outputs:
            raise ValueError(
                f"Output path {path} not found in differentiable paths ({diff_outputs})"
            )

    # Check gradients for each endpoint separately
    rng = np.random.RandomState(seed)
    output_rng = np.random.RandomState(seed)

    for endpoint in endpoints:
        failures, num_evals = check_endpoint_gradients(
            endpoint_functions,
            inputs,
            endpoint,
            outputs=outputs,
            diff_inputs=input_paths,
            diff_outputs=output_paths,
            max_evals=max_evals,
            max_output_samples=max_output_samples,
            eps=eps,
            rtol=rtol,
            rng=rng,
            output_rng=output_rng,
            show_progress=show_progress,
        )
        yield endpoint, failures, num_evals
