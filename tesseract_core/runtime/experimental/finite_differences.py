# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite difference utilities for automatic differentiation.

These provide a simple way to make any Tesseract differentiable without
implementing analytical gradients.

.. note::

    These are experimental and the API may change in future releases.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel

from tesseract_core.runtime.testing.finite_differences import (
    _compute_central_diff_row,
    _compute_forward_diff_row,
)
from tesseract_core.runtime.tree_transforms import get_at_path, set_at_path

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
        algorithm: The finite difference algorithm to use. Options are
            ``"central"`` (central differences, most accurate, 2 evaluations per element),
            ``"forward"`` (forward differences, faster, 1 extra evaluation per element), or
            ``"stochastic"`` (SPSA algorithm, scales better to high-dimensional inputs).
        eps: Perturbation magnitude for finite differences.
        num_samples: Number of random samples for the stochastic algorithm.
            Only used when ``algorithm="stochastic"``. Defaults to ``max(10, sqrt(n))``
            where ``n`` is the total number of input elements, providing O(sqrt(n))
            cost instead of O(n) for full finite differences.
        seed: Random seed for reproducibility (only used with ``algorithm="stochastic"``).

    Returns:
        A nested dictionary with structure ``{output_path: {input_path: jacobian_array}}``,
        where each jacobian_array has shape ``(*output_shape, *input_shape)``.

    Example:
        In a Tesseract's ``tesseract_api.py``::

            from tesseract_core.runtime.experimental import finite_difference_jacobian


            def jacobian(
                inputs: InputSchema,
                jac_inputs: set[str],
                jac_outputs: set[str],
            ):
                return finite_difference_jacobian(
                    apply, inputs, jac_inputs, jac_outputs
                )

    .. note::

        This function is experimental and its API may change in future releases.
        It is useful for prototyping or when analytical gradients are difficult
        to derive, but numerical differentiation is generally less accurate and
        more computationally expensive than analytical methods.
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
            algorithm=algorithm,
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
    algorithm: FDAlgorithm,
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
                elif algorithm == "forward":
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
                else:
                    raise ValueError(f"Unknown algorithm {algorithm}")

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

    SPSA requires only 2 function evaluations per sample, regardless of the input
    dimension, making it efficient for high-dimensional inputs.
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

    # Default number of samples: use sqrt(n) which balances cost vs accuracy.
    # This gives O(sqrt(n)) evaluations instead of O(n) for full FD,
    # while still providing reasonable gradient estimates.
    if num_samples is None:
        num_samples = max(10, int(np.sqrt(total_input_elements)))

    # If num_samples >= total_input_elements, stochastic is no cheaper than
    # elementwise but less accurate. Fall back to central differences.
    if num_samples >= total_input_elements:
        _compute_jacobian_elementwise(
            apply_fn,
            inputs,
            inputs_dict,
            base_outputs,
            jac_inputs,
            jac_outputs,
            result,
            eps=eps,
            algorithm="central",
        )
        return

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
            perturbation_size = None if info["shape"] == () else info["shape"]
            perturbations[in_path] = rng.choice(
                np.array([-1, 1], dtype=np.float64), size=perturbation_size
            )

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
    algorithm: FDAlgorithm = "central",
    eps: float = 1e-4,
) -> dict[str, ArrayLike]:
    """Compute the Jacobian-vector product (JVP) using finite differences.

    The JVP computes ``J @ v`` where ``J`` is the Jacobian and ``v`` is the tangent vector.
    This is done efficiently using directional derivatives without computing the full Jacobian.

    Note: The ``"stochastic"`` algorithm is treated as ``"central"`` for JVP computation.
    JVP naturally requires only O(1) function evaluations regardless of input dimension
    (2 for central, 1 for forward), so stochastic estimation provides no benefit.

    Args:
        apply_fn: The Tesseract's apply function with signature ``apply(inputs) -> outputs``.
        inputs: The input data at which to compute the JVP.
        jvp_inputs: Set of input paths to differentiate with respect to.
        jvp_outputs: Set of output paths to compute derivatives of.
        tangent_vector: Dictionary mapping input paths to tangent arrays.
        algorithm: The finite difference algorithm to use. Options are
            ``"central"`` (most accurate, default) or ``"forward"`` (faster).
            The ``"stochastic"`` option is accepted but treated as ``"central"``.
        eps: Perturbation magnitude.

    Returns:
        Dictionary mapping output paths to JVP result arrays.

    Example:
        In a Tesseract's ``tesseract_api.py``::

            from tesseract_core.runtime.experimental import finite_difference_jvp


            def jacobian_vector_product(
                inputs: InputSchema,
                jvp_inputs: set[str],
                jvp_outputs: set[str],
                tangent_vector: dict[str, Any],
            ):
                return finite_difference_jvp(
                    apply, inputs, jvp_inputs, jvp_outputs, tangent_vector
                )

    .. note::

        This function is experimental and its API may change in future releases.
    """
    inputs_dict = inputs.model_dump()

    # Stochastic algorithm is treated as central for JVP since JVP already
    # achieves O(1) function evaluations via directional derivatives
    if algorithm == "stochastic":
        algorithm = "central"

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
    algorithm: FDAlgorithm = "central",
    eps: float = 1e-4,
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, ArrayLike]:
    """Compute the vector-Jacobian product (VJP) using finite differences.

    The VJP computes ``v @ J`` where ``J`` is the Jacobian and ``v`` is the cotangent vector.

    Note: For ``"central"`` and ``"forward"`` algorithms, the VJP is computed by
    explicitly computing the Jacobian rows and contracting with the cotangent vector.
    This requires O(n_inputs) function evaluations, the same cost as computing the
    full Jacobian. For high-dimensional inputs, consider using ``algorithm="stochastic"``.

    Args:
        apply_fn: The Tesseract's apply function with signature ``apply(inputs) -> outputs``.
        inputs: The input data at which to compute the VJP.
        vjp_inputs: Set of input paths to differentiate with respect to.
        vjp_outputs: Set of output paths to compute derivatives of.
        cotangent_vector: Dictionary mapping output paths to cotangent arrays.
        algorithm: The finite difference algorithm to use. Options are
            ``"central"`` (most accurate), ``"forward"`` (faster), or
            ``"stochastic"`` (SPSA, better for high-dimensional inputs).
        eps: Perturbation magnitude.
        num_samples: Number of random samples for the stochastic algorithm.
            Only used when ``algorithm="stochastic"``. Defaults to ``max(10, sqrt(n))``
            where ``n`` is the total number of input elements.
        seed: Random seed for reproducibility (only used with ``algorithm="stochastic"``).

    Returns:
        Dictionary mapping input paths to VJP result arrays.

    Example:
        In a Tesseract's ``tesseract_api.py``::

            from tesseract_core.runtime.experimental import finite_difference_vjp


            def vector_jacobian_product(
                inputs: InputSchema,
                vjp_inputs: set[str],
                vjp_outputs: set[str],
                cotangent_vector: dict[str, Any],
            ):
                return finite_difference_vjp(
                    apply, inputs, vjp_inputs, vjp_outputs, cotangent_vector
                )

    .. note::

        This function is experimental and its API may change in future releases.
    """
    inputs_dict = inputs.model_dump()
    input_schema = type(inputs)

    # Initialize result
    result: dict[str, np.ndarray] = {}
    for in_path in vjp_inputs:
        in_val = get_at_path(inputs_dict, in_path)
        in_arr = np.asarray(in_val)
        result[in_path] = np.zeros_like(in_arr, dtype=np.float64)

    if algorithm == "stochastic":
        _compute_vjp_stochastic(
            apply_fn,
            inputs,
            inputs_dict,
            vjp_inputs,
            vjp_outputs,
            cotangent_vector,
            result,
            eps=eps,
            num_samples=num_samples,
            seed=seed,
        )
    else:
        # Only needed for forward differences
        base_outputs = apply_fn(inputs).model_dump() if algorithm == "forward" else None

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
                    elif algorithm == "forward":
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
                    else:
                        raise ValueError(f"Unknown algorithm {algorithm}")
                    cotangent = np.asarray(cotangent_vector[out_path])
                    vjp_value += np.sum(cotangent * grad)

                if idx:
                    result[in_path][idx] = vjp_value
                else:
                    result[in_path] = np.float64(vjp_value)

    return result


def _compute_vjp_stochastic(
    apply_fn: Callable,
    inputs: BaseModel,
    inputs_dict: dict,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, ArrayLike],
    result: dict[str, np.ndarray],
    *,
    eps: float,
    num_samples: int | None,
    seed: int | None,
) -> None:
    """Compute VJP using Simultaneous Perturbation Stochastic Approximation (SPSA).

    For VJP, we want to compute v @ J for each input, which is:
        VJP[in_path] = sum over out_path of: cotangent[out_path] * J[out_path, in_path]

    Using SPSA, we can estimate this efficiently by:
    1. Generating random perturbation directions delta (Rademacher: ±1)
    2. Computing output differences: (f(x + eps*delta) - f(x - eps*delta)) / (2*eps)
    3. Computing gradient estimate: (output_diff · cotangent) / delta

    This requires only 2 function evaluations per sample, regardless of dimension.
    """
    rng = np.random.RandomState(seed)

    # Collect input metadata
    input_info = {}
    total_input_elements = 0
    for in_path in vjp_inputs:
        in_val = get_at_path(inputs_dict, in_path)
        in_arr = np.asarray(in_val)
        input_info[in_path] = {
            "array": in_arr,
            "shape": in_arr.shape,
            "size": in_arr.size if in_arr.shape else 1,
        }
        total_input_elements += input_info[in_path]["size"]

    # Default number of samples: use sqrt(n) which balances cost vs accuracy.
    # This gives O(sqrt(n)) evaluations instead of O(n) for full FD,
    # while still providing reasonable gradient estimates.
    if num_samples is None:
        num_samples = max(10, int(np.sqrt(total_input_elements)))

    # If num_samples >= total_input_elements, stochastic is no cheaper than
    # elementwise but less accurate. Fall back to central differences.
    if num_samples >= total_input_elements:
        input_schema = type(inputs)
        for in_path in vjp_inputs:
            in_val = get_at_path(inputs_dict, in_path)
            in_arr = np.asarray(in_val)
            in_shape = in_arr.shape
            indices = list(np.ndindex(in_shape)) if in_shape else [()]
            for idx in indices:
                vjp_value = 0.0
                for out_path in vjp_outputs:
                    grad = _compute_central_diff_row(
                        apply_fn,
                        inputs_dict,
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
        return

    # Accumulate VJP estimates
    for _ in range(num_samples):
        # Generate random perturbation directions (Rademacher: ±1)
        perturbations = {}
        for in_path, info in input_info.items():
            perturbation_size = None if info["shape"] == () else info["shape"]
            perturbations[in_path] = rng.choice(
                np.array([-1, 1], dtype=np.float64), size=perturbation_size
            )

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

        # Compute weighted output difference (weighted by cotangent)
        # This is the directional derivative in the direction of the cotangent
        weighted_output_diff = 0.0
        for out_path in vjp_outputs:
            out_plus = np.asarray(get_at_path(outputs_plus, out_path))
            out_minus = np.asarray(get_at_path(outputs_minus, out_path))
            output_diff = (out_plus - out_minus) / (2 * eps)
            cotangent = np.asarray(cotangent_vector[out_path])
            weighted_output_diff += np.sum(cotangent * output_diff)

        # Update VJP estimate for each input
        # VJP contribution = weighted_output_diff / delta
        for in_path in vjp_inputs:
            delta = perturbations[in_path]
            vjp_contrib = weighted_output_diff / delta
            result[in_path] += vjp_contrib / num_samples
