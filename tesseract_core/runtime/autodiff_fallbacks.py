# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fallback implementations of autodiff endpoints derived from other available endpoints."""

import numpy as np

from .tree_transforms import get_at_path


def vjp_from_jacobian(jacobian_fn, inputs, vjp_inputs, vjp_outputs, cotangent_vector):
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
    out = {}
    for dx in vjp_inputs:
        grad = None
        for dy in vjp_outputs:
            J = np.asarray(jac[dy][dx])          # shape: (*dy_shape, *dx_shape)
            v = np.asarray(cotangent_vector[dy])  # shape: (*dy_shape)
            # Contract all of v against the first v.ndim axes of J → shape (*dx_shape)
            term = np.tensordot(v, J, axes=v.ndim)
            grad = term if grad is None else grad + term
        out[dx] = grad
    return out


def jvp_from_jacobian(jacobian_fn, inputs, jvp_inputs, jvp_outputs, tangent_vector):
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
    out = {}
    for dy in jvp_outputs:
        result = None
        for dx in jvp_inputs:
            J = np.asarray(jac[dy][dx])        # shape: (*dy_shape, *dx_shape)
            t = np.asarray(tangent_vector[dx])  # shape: (*dx_shape)
            # Contract the last t.ndim axes of J with all of t → shape (*dy_shape)
            term = np.tensordot(J, t, axes=t.ndim)
            result = term if result is None else result + term
        out[dy] = result
    return out


def jacobian_from_vjp(vjp_fn, apply_fn, inputs, jac_inputs, jac_outputs):
    """Compute the Jacobian by calling VJP with one-hot cotangent vectors.

    Requires M calls to VJP, where M is the total number of output elements.

    Args:
        vjp_fn: The api_module.vector_jacobian_product callable.
        apply_fn: The api_module.apply callable (used to determine output shapes).
        inputs: Validated InputSchema instance.
        jac_inputs: set[str] of input path strings.
        jac_outputs: set[str] of output path strings.

    Returns:
        dict[str, dict[str, np.ndarray]] with structure {output_path: {input_path: array}}
        where each array has shape (*output_shape, *input_shape).
    """
    raw_outputs = apply_fn(inputs=inputs)
    outputs_dict = raw_outputs.model_dump() if hasattr(raw_outputs, "model_dump") else raw_outputs

    jac = {}
    output_vals = {}
    for dy in jac_outputs:
        dy_val = np.asarray(get_at_path(outputs_dict, dy))
        output_vals[dy] = dy_val
        jac[dy] = {}
        for dx in jac_inputs:
            dx_val = np.asarray(get_at_path(inputs, dx))
            jac[dy][dx] = np.zeros((*dy_val.shape, *dx_val.shape), dtype=dy_val.dtype)

    for dy in jac_outputs:
        dy_val = output_vals[dy]
        dy_shape = dy_val.shape
        for nd_idx in (np.ndindex(*dy_shape) if dy_shape else [()]):
            cotangent = {dy: np.zeros_like(dy_val)}
            if dy_shape:
                cotangent[dy][nd_idx] = 1.0
            else:
                cotangent[dy] = np.array(1.0, dtype=dy_val.dtype)
            grad = vjp_fn(
                inputs=inputs,
                vjp_inputs=jac_inputs,
                vjp_outputs={dy},
                cotangent_vector=cotangent,
            )
            for dx in jac_inputs:
                if dy_shape:
                    jac[dy][dx][nd_idx] = np.asarray(grad[dx])
                else:
                    jac[dy][dx] = np.asarray(grad[dx])
    return jac


def jacobian_from_jvp(jvp_fn, apply_fn, inputs, jac_inputs, jac_outputs):
    """Compute the Jacobian by calling JVP with one-hot tangent vectors.

    Requires N calls to JVP, where N is the total number of input elements.

    Args:
        jvp_fn: The api_module.jacobian_vector_product callable.
        apply_fn: The api_module.apply callable (used to determine output shapes).
        inputs: Validated InputSchema instance.
        jac_inputs: set[str] of input path strings.
        jac_outputs: set[str] of output path strings.

    Returns:
        dict[str, dict[str, np.ndarray]] with structure {output_path: {input_path: array}}
        where each array has shape (*output_shape, *input_shape).
    """
    raw_outputs = apply_fn(inputs=inputs)
    outputs_dict = raw_outputs.model_dump() if hasattr(raw_outputs, "model_dump") else raw_outputs

    jac = {}
    for dy in jac_outputs:
        dy_val = np.asarray(get_at_path(outputs_dict, dy))
        jac[dy] = {}
        for dx in jac_inputs:
            dx_val = np.asarray(get_at_path(inputs, dx))
            jac[dy][dx] = np.zeros((*dy_val.shape, *dx_val.shape), dtype=dy_val.dtype)

    for dx in jac_inputs:
        dx_val = np.asarray(get_at_path(inputs, dx))
        dx_shape = dx_val.shape
        for nd_idx in (np.ndindex(*dx_shape) if dx_shape else [()]):
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
                if dx_shape:
                    # Set the nd_idx-th column: jac[dy][dx][..., *nd_idx] = dy_result
                    jac[dy][dx][(...,) + nd_idx] = dy_result
                else:
                    jac[dy][dx] = dy_result
    return jac
