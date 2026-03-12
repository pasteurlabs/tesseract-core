# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Autodiff fallback utilities for deriving missing autodiff endpoints from existing ones.
# These are experimental and the API may change in future releases.

from collections.abc import Callable
from typing import Any

import numpy as np


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
