# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lorenz 96 Tesseract with finite-difference gradients.

This variant of the Lorenz 96 Tesseract uses numerical finite differences
for all gradient endpoints, simulating the scenario where the simulator
source code is unavailable or not differentiable (e.g., a compiled Fortran
binary, a commercial solver, etc.).

The apply function is identical to the JAX variant — only the gradient
strategy differs.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)


class InputSchema(BaseModel):
    """Input schema for forecasting of Lorenz 96 system."""

    state: Differentiable[Array[(None,), Float32]] = Field(
        description="A state vector for the Lorenz 96 system"
    )
    F: Differentiable[Array[(), Float32]] = Field(
        description="Forcing parameter for Lorenz 96", default=8.0
    )
    dt: float = Field(description="Time step for integration", default=0.05)
    n_steps: int = Field(description="Number of integration steps", default=1)


class OutputSchema(BaseModel):
    """Output schema for forecasting of Lorenz 96 system."""

    result: Differentiable[Array[(None, None), Float32]] = Field(
        description="A trajectory of predictions after integration"
    )


# --- Pure NumPy Lorenz 96 solver (no JAX dependency) ---


def _lorenz96_derivatives(x: np.ndarray, F: float) -> np.ndarray:
    """Compute Lorenz 96 tendencies."""
    N = x.shape[0]
    ip1 = (np.arange(N) + 1) % N
    im1 = (np.arange(N) - 1) % N
    im2 = (np.arange(N) - 2) % N
    return (x[ip1] - x[im2]) * x[im1] - x + F


def _lorenz96_step(state: np.ndarray, F: float, dt: float) -> np.ndarray:
    """One RK4 step."""
    k1 = _lorenz96_derivatives(state, F)
    k2 = _lorenz96_derivatives(state + dt * k1 / 2, F)
    k3 = _lorenz96_derivatives(state + dt * k2 / 2, F)
    k4 = _lorenz96_derivatives(state + dt * k3, F)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _lorenz96_multi_step(
    state: np.ndarray, F: float, dt: float, n_steps: int
) -> np.ndarray:
    """Integrate Lorenz 96 for n_steps, returning the full trajectory."""
    trajectory = np.empty((n_steps, state.shape[0]), dtype=np.float32)
    for i in range(n_steps):
        next_state = _lorenz96_step(state, F, dt)
        trajectory[i] = state
        state = next_state
    return trajectory


# --- Tesseract endpoints ---


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: integrate the Lorenz 96 system."""
    state = np.asarray(inputs.state, dtype=np.float64)
    F = float(inputs.F)
    trajectory = _lorenz96_multi_step(state, F, inputs.dt, inputs.n_steps)
    return OutputSchema(result=trajectory.astype(np.float32))


def abstract_eval(abstract_inputs: Any) -> Any:
    """Calculate output shape from input shape."""
    state_sd = abstract_inputs.state
    n_dim = state_sd.shape[0]
    # n_steps is a static int, available as a concrete value
    n_steps = abstract_inputs.n_steps
    return {
        "result": ShapeDType(shape=(n_steps, n_dim), dtype="float32"),
    }


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> Any:
    """Jacobian via central finite differences."""
    return finite_difference_jacobian(
        apply, inputs, jac_inputs, jac_outputs, algorithm="central", eps=1e-5
    )


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> Any:
    """JVP via finite differences."""
    return finite_difference_jvp(
        apply,
        inputs,
        jvp_inputs,
        jvp_outputs,
        tangent_vector,
        algorithm="central",
        eps=1e-5,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> Any:
    """VJP via finite differences."""
    return finite_difference_vjp(
        apply,
        inputs,
        vjp_inputs,
        vjp_outputs,
        cotangent_vector,
        algorithm="central",
        eps=1e-5,
    )
