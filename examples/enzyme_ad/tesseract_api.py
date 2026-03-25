# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract wrapping a Fortran heat equation solver differentiated by Enzyme.

This example demonstrates how to obtain exact automatic derivatives of a
Fortran simulation without writing any adjoint code.  The pipeline is:

    Fortran source
      -> LFortran (LLVM IR)
      -> Enzyme (LLVM AD pass)
      -> shared library with forward, JVP, and VJP entry points
      -> Python ctypes -> Tesseract API

The solver computes a single explicit Euler step of the 1D heat equation:

    dT/dt = alpha * d^2T/dx^2

Enzyme generates machine-precision derivatives through the compiled Fortran
code, enabling gradient-based optimization of thermal parameters (alpha, dx,
dt) or initial conditions (T_in) with respect to any output.
"""

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float64

# ── Shared library loading ──────────────────────────────────────────

_LIB_PATH = Path("/tesseract/enzyme/libheat_ad.so")
_lib = ctypes.CDLL(str(_LIB_PATH))

# void heat_step_forward(int n, double* T_in, double* T_out,
#                        double alpha, double dx, double dt)
_lib.heat_step_forward.restype = None
_lib.heat_step_forward.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]

# void heat_step_vjp(int n,
#     double* T_in, double* dT_in, double* T_out, double* dT_out,
#     double alpha, double* dalpha, double dx, double* ddx,
#     double dt, double* ddt)
_lib.heat_step_vjp.restype = None
_lib.heat_step_vjp.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
]

# void heat_step_jvp(int n,
#     double* T_in, double* dT_in, double* T_out, double* dT_out,
#     double alpha, double dalpha, double dx, double ddx,
#     double dt, double ddt)
_lib.heat_step_jvp.restype = None
_lib.heat_step_jvp.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    """Get a ctypes double pointer from a contiguous float64 array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# ── Schemas ─────────────────────────────────────────────────────────


class InputSchema(BaseModel):
    """Input for a single explicit Euler step of the 1D heat equation."""

    T_in: Differentiable[Array[(None,), Float64]] = Field(
        description="Temperature profile at current time step [K]. Shape: (n,). "
        "Boundary values T_in[0] and T_in[-1] are held fixed (Dirichlet).",
    )
    alpha: Differentiable[Float64] = Field(
        default=0.01,
        description="Thermal diffusivity [m^2/s].",
        gt=0.0,
    )
    dx: Differentiable[Float64] = Field(
        default=0.25,
        description="Grid spacing [m].",
        gt=0.0,
    )
    dt: Differentiable[Float64] = Field(
        default=0.001,
        description="Time step size [s].",
        gt=0.0,
    )

    @model_validator(mode="after")
    def check_stability(self) -> Self:
        """Verify CFL stability condition: r = alpha * dt / dx^2 <= 0.5."""
        r = self.alpha * self.dt / (self.dx**2)
        if r > 0.5:
            raise ValueError(
                f"CFL stability condition violated: r = {r:.4f} > 0.5. "
                f"Reduce dt or alpha, or increase dx."
            )
        return self

    @model_validator(mode="after")
    def check_min_points(self) -> Self:
        """Need at least 3 points for interior stencil."""
        if len(self.T_in) < 3:
            raise ValueError("T_in must have at least 3 points.")
        return self


class OutputSchema(BaseModel):
    """Output: temperature profile after one heat equation step."""

    T_out: Differentiable[Array[(None,), Float64]] = Field(
        description="Temperature profile after one time step [K]. Shape: (n,).",
    )


# ── Required endpoints ──────────────────────────────────────────────


def apply(inputs: InputSchema) -> OutputSchema:
    """Compute one explicit Euler step of the 1D heat equation."""
    T_in = np.ascontiguousarray(inputs.T_in, dtype=np.float64)
    n = len(T_in)
    T_out = np.zeros(n, dtype=np.float64)

    _lib.heat_step_forward(
        n, _as_ptr(T_in), _as_ptr(T_out), inputs.alpha, inputs.dx, inputs.dt
    )

    return OutputSchema(T_out=T_out)


# ── Optional endpoints (AD via Enzyme) ──────────────────────────────


def _run_vjp(inputs: InputSchema, cotangent_T_out: np.ndarray):
    """Run Enzyme reverse-mode AD and return all gradients."""
    T_in = np.ascontiguousarray(inputs.T_in, dtype=np.float64)
    n = len(T_in)

    # Shadow arrays (Enzyme accumulates gradients into these)
    dT_in = np.zeros(n, dtype=np.float64)
    T_out = np.zeros(n, dtype=np.float64)
    dT_out = np.array(cotangent_T_out, dtype=np.float64)
    dalpha = ctypes.c_double(0.0)
    ddx = ctypes.c_double(0.0)
    ddt = ctypes.c_double(0.0)

    _lib.heat_step_vjp(
        n,
        _as_ptr(T_in),
        _as_ptr(dT_in),
        _as_ptr(T_out),
        _as_ptr(dT_out),
        inputs.alpha,
        ctypes.byref(dalpha),
        inputs.dx,
        ctypes.byref(ddx),
        inputs.dt,
        ctypes.byref(ddt),
    )

    return dT_in, dalpha.value, ddx.value, ddt.value


def _run_jvp(inputs: InputSchema, tangent_T_in, tangent_alpha, tangent_dx, tangent_dt):
    """Run Enzyme forward-mode AD and return output tangent."""
    T_in = np.ascontiguousarray(inputs.T_in, dtype=np.float64)
    n = len(T_in)

    dT_in = np.ascontiguousarray(tangent_T_in, dtype=np.float64)
    T_out = np.zeros(n, dtype=np.float64)
    dT_out = np.zeros(n, dtype=np.float64)

    _lib.heat_step_jvp(
        n,
        _as_ptr(T_in),
        _as_ptr(dT_in),
        _as_ptr(T_out),
        _as_ptr(dT_out),
        inputs.alpha,
        float(tangent_alpha),
        inputs.dx,
        float(tangent_dx),
        inputs.dt,
        float(tangent_dt),
    )

    return dT_out


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Reverse-mode AD via Enzyme: compute v^T @ J."""
    cotangent_T_out = cotangent_vector.get("T_out", np.zeros_like(inputs.T_in))
    dT_in, dalpha, ddx, ddt = _run_vjp(inputs, cotangent_T_out)

    result = {}
    if "T_in" in vjp_inputs:
        result["T_in"] = dT_in
    if "alpha" in vjp_inputs:
        result["alpha"] = dalpha
    if "dx" in vjp_inputs:
        result["dx"] = ddx
    if "dt" in vjp_inputs:
        result["dt"] = ddt
    return result


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Forward-mode AD via Enzyme: compute J @ v."""
    n = len(inputs.T_in)
    tangent_T_in = tangent_vector.get("T_in", np.zeros(n, dtype=np.float64))
    tangent_alpha = tangent_vector.get("alpha", 0.0)
    tangent_dx = tangent_vector.get("dx", 0.0)
    tangent_dt = tangent_vector.get("dt", 0.0)

    dT_out = _run_jvp(inputs, tangent_T_in, tangent_alpha, tangent_dx, tangent_dt)

    result = {}
    if "T_out" in jvp_outputs:
        result["T_out"] = dT_out
    return result
