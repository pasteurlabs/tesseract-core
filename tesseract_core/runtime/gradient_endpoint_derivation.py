# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Gradient fallback utilities for deriving missing gradient endpoints from existing ones.
# These are experimental and the API may change in future releases.

from collections.abc import Callable
from typing import Any

import numpy as np

from .tree_transforms import get_at_path


def vjp_from_jacobian(
    jacobian_fn: Callable,
    inputs: Any,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute VJP as v^T @ J using the full Jacobian.

    Args:
        jacobian_fn: The api_module.jacobian callable.
        inputs: Validated InputSchema instance.
        vjp_inputs: set[str] of input path strings to differentiate w.r.t.
        vjp_outputs: set[str] of output path strings to differentiate.
        cotangent_vector: dict mapping output paths to cotangent arrays.

    Returns:
        dict mapping input paths to gradient arrays.
    """
    jac = jacobian_fn(inputs=inputs, jac_inputs=vjp_inputs, jac_outputs=vjp_outputs)

    # For each input dx, sum v^T @ J[dy][dx] over all requested outputs dy.
    # tensordot with axes=v.ndim contracts all of v against the leading axes of J,
    # leaving shape (*dx_shape) — the gradient w.r.t. dx.
    return {
        dx: sum(
            np.tensordot(
                np.asarray(cotangent_vector[dy]),
                np.asarray(jac[dy][dx]),
                axes=np.asarray(cotangent_vector[dy]).ndim,
            )
            for dy in vjp_outputs
        )
        for dx in vjp_inputs
    }


def jvp_from_jacobian(
    jacobian_fn: Callable,
    inputs: Any,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute JVP as J @ t using the full Jacobian.

    Args:
        jacobian_fn: The api_module.jacobian callable.
        inputs: Validated InputSchema instance.
        jvp_inputs: set[str] of input path strings.
        jvp_outputs: set[str] of output path strings.
        tangent_vector: dict mapping input paths to tangent arrays.

    Returns:
        dict mapping output paths to JVP result arrays.
    """
    jac = jacobian_fn(inputs=inputs, jac_inputs=jvp_inputs, jac_outputs=jvp_outputs)

    # For each output dy, sum J[dy][dx] @ t[dx] over all requested inputs dx.
    # tensordot with axes=t.ndim contracts the trailing axes of J against all of t,
    # leaving shape (*dy_shape) — the tangent propagated to dy.
    return {
        dy: sum(
            np.tensordot(
                np.asarray(jac[dy][dx]),
                np.asarray(tangent_vector[dx]),
                axes=np.asarray(tangent_vector[dx]).ndim,
            )
            for dx in jvp_inputs
        )
        for dy in jvp_outputs
    }


def jacobian_from_vjp(
    vjp_fn: Callable,
    eval_fn: Callable,
    inputs: Any,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute the Jacobian by calling VJP with one-hot cotangent vectors.

    Requires M calls to VJP, where M is the total number of output elements.

    Args:
        vjp_fn: The api_module.vector_jacobian_product callable.
        eval_fn: Either api_module.apply or api_module.abstract_eval, used to
            determine output shapes and dtypes. abstract_eval is preferred as it
            avoids a full forward pass.
        inputs: Validated InputSchema instance.
        jac_inputs: set[str] of input path strings.
        jac_outputs: set[str] of output path strings.

    Returns:
        dict[str, dict[str, np.ndarray]] with structure {output_path: {input_path: array}}
        where each array has shape ``(*output_shape, *input_shape)``.
    """
    # eval_fn is called once to learn output shapes and dtypes without running
    # the full computation. abstract_eval is preferred over apply for this reason.
    raw_outputs = eval_fn(inputs=inputs)
    outputs_dict = (
        raw_outputs.model_dump() if hasattr(raw_outputs, "model_dump") else raw_outputs
    )

    # Pre-allocate jac[dy][dx] with shape (*dy_shape, *dx_shape).
    jac = {}
    output_shapes = {}
    output_dtypes = {}
    for dy in jac_outputs:
        dy_out = get_at_path(outputs_dict, dy)
        output_shapes[dy] = tuple(dy_out.shape)
        output_dtypes[dy] = dy_out.dtype
        jac[dy] = {}
        for dx in jac_inputs:
            dx_val = np.asarray(get_at_path(inputs, dx))
            jac[dy][dx] = np.zeros(
                (*output_shapes[dy], *dx_val.shape), dtype=output_dtypes[dy]
            )

    # Sweep one-hot cotangents over every output element.
    # Each probe e_i recovers row i of J (i.e. dF_i/dx for all x),
    # because VJP(e_i) = e_i^T @ J = J[i, :].
    # The M probes are independent and could be parallelised.
    for dy in jac_outputs:
        dy_shape = output_shapes[dy]
        for nd_idx in np.ndindex(*dy_shape) if dy_shape else [()]:
            # Build the one-hot cotangent for output element nd_idx.
            cotangent = {dy: np.zeros(dy_shape, dtype=output_dtypes[dy])}
            if dy_shape:
                cotangent[dy][nd_idx] = 1.0
            else:
                cotangent[dy] = np.array(1.0, dtype=output_dtypes[dy])
            grad = vjp_fn(
                inputs=inputs,
                vjp_inputs=jac_inputs,
                vjp_outputs={dy},
                cotangent_vector=cotangent,
            )
            for dx in jac_inputs:
                # grad[dx] is the gradient w.r.t. dx, i.e. row nd_idx of J.
                if dy_shape:
                    jac[dy][dx][nd_idx] = np.asarray(grad[dx])
                else:
                    jac[dy][dx] = np.asarray(grad[dx])
    return jac


def jacobian_from_jvp(
    jvp_fn: Callable,
    inputs: Any,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute the Jacobian by calling JVP with one-hot tangent vectors.

    Requires N calls to JVP, where N is the total number of input elements.
    Output shapes are inferred from the first JVP probe.

    Args:
        jvp_fn: The api_module.jacobian_vector_product callable.
        inputs: Validated InputSchema instance.
        jac_inputs: set[str] of input path strings.
        jac_outputs: set[str] of output path strings.

    Returns:
        dict[str, dict[str, np.ndarray]] with structure {output_path: {input_path: array}}
        where each array has shape ``(*output_shape, *input_shape)``.
    """
    jac: dict[str, dict[str, Any]] = {dy: {} for dy in jac_outputs}

    # Sweep one-hot tangents over every input element.
    # Each probe e_j recovers column j of J (i.e. dF/dx_j for all F),
    # because JVP(e_j) = J @ e_j = J[:, j].
    # The N probes are independent and could be parallelised.
    for dx in jac_inputs:
        dx_val = np.asarray(get_at_path(inputs, dx))
        dx_shape = dx_val.shape
        for nd_idx in np.ndindex(*dx_shape) if dx_shape else [()]:
            # Build the one-hot tangent for input element nd_idx.
            tangent = {dx: np.zeros_like(dx_val)}
            if dx_shape:
                tangent[dx][nd_idx] = 1.0
            else:
                tangent[dx] = np.array(1.0, dtype=dx_val.dtype)
            result = jvp_fn(
                inputs=inputs,
                jvp_inputs={dx},
                jvp_outputs=jac_outputs,
                tangent_vector=tangent,
            )
            for dy in jac_outputs:
                dy_result = np.asarray(result[dy])
                # Allocate on the first probe; output shape is unknown until here.
                if dx not in jac[dy]:
                    jac[dy][dx] = np.zeros(
                        (*dy_result.shape, *dx_shape), dtype=dy_result.dtype
                    )
                # dy_result is column nd_idx of J; (...,) selects all output axes.
                if dx_shape:
                    jac[dy][dx][(..., *nd_idx)] = dy_result
                else:
                    jac[dy][dx] = dy_result
    return jac
