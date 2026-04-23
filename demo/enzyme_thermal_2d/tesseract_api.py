# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract wrapping a 2D Fortran thermal solver differentiated by Enzyme.

This example demonstrates how to obtain exact automatic derivatives of a
production-style Fortran thermal simulation without writing any adjoint code.

The solver computes transient 2D heat conduction with:
  - Temperature-dependent conductivity: k(T) = k0 + k1*T
  - Mixed boundary conditions: Dirichlet (hot wall), convection, insulated
  - Volumetric heat source
  - Multi-step explicit time integration

Enzyme generates machine-precision derivatives through the entire time-stepping
loop, enabling gradient-based optimization of material properties, boundary
conditions, or initial conditions with respect to any output.
"""

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float64

# -- Shared library loading ------------------------------------------------

_LIB_PATH = Path("/tesseract/enzyme/libthermal_2d_ad.so")
_lib = ctypes.CDLL(str(_LIB_PATH))

# void thermal_2d_forward(int nx, int ny, int n_steps,
#     double* T_init, double* T_final,
#     double k0, double k1, double rho, double cp,
#     double h_conv, double T_inf, double T_hot,
#     double* Q, double Lx, double Ly, double dt)
_lib.thermal_2d_forward.restype = None
_lib.thermal_2d_forward.argtypes = [
    ctypes.c_int,  # nx
    ctypes.c_int,  # ny
    ctypes.c_int,  # n_steps
    ctypes.POINTER(ctypes.c_double),  # T_init
    ctypes.POINTER(ctypes.c_double),  # T_final
    ctypes.c_double,  # k0
    ctypes.c_double,  # k1
    ctypes.c_double,  # rho
    ctypes.c_double,  # cp
    ctypes.c_double,  # h_conv
    ctypes.c_double,  # T_inf
    ctypes.c_double,  # T_hot
    ctypes.POINTER(ctypes.c_double),  # Q
    ctypes.c_double,  # Lx
    ctypes.c_double,  # Ly
    ctypes.c_double,  # dt
]

# void thermal_2d_vjp(int nx, int ny, int n_steps,
#     double* T_init, double* dT_init, double* T_final, double* dT_final,
#     double k0, double* dk0, double k1, double* dk1,
#     double rho, double* drho, double cp, double* dcp,
#     double h_conv, double* dh_conv, double T_inf, double* dT_inf,
#     double T_hot, double* dT_hot,
#     double* Q, double* dQ,
#     double Lx, double* dLx, double Ly, double* dLy, double dt, double* ddt)
_lib.thermal_2d_vjp.restype = None
_lib.thermal_2d_vjp.argtypes = [
    ctypes.c_int,  # nx
    ctypes.c_int,  # ny
    ctypes.c_int,  # n_steps
    ctypes.POINTER(ctypes.c_double),  # T_init
    ctypes.POINTER(ctypes.c_double),  # dT_init
    ctypes.POINTER(ctypes.c_double),  # T_final
    ctypes.POINTER(ctypes.c_double),  # dT_final
    ctypes.c_double,  # k0
    ctypes.POINTER(ctypes.c_double),  # dk0
    ctypes.c_double,  # k1
    ctypes.POINTER(ctypes.c_double),  # dk1
    ctypes.c_double,  # rho
    ctypes.POINTER(ctypes.c_double),  # drho
    ctypes.c_double,  # cp
    ctypes.POINTER(ctypes.c_double),  # dcp
    ctypes.c_double,  # h_conv
    ctypes.POINTER(ctypes.c_double),  # dh_conv
    ctypes.c_double,  # T_inf
    ctypes.POINTER(ctypes.c_double),  # dT_inf
    ctypes.c_double,  # T_hot
    ctypes.POINTER(ctypes.c_double),  # dT_hot
    ctypes.POINTER(ctypes.c_double),  # Q
    ctypes.POINTER(ctypes.c_double),  # dQ
    ctypes.c_double,  # Lx
    ctypes.POINTER(ctypes.c_double),  # dLx
    ctypes.c_double,  # Ly
    ctypes.POINTER(ctypes.c_double),  # dLy
    ctypes.c_double,  # dt
    ctypes.POINTER(ctypes.c_double),  # ddt
]

# void thermal_2d_jvp(int nx, int ny, int n_steps,
#     double* T_init, double* dT_init, double* T_final, double* dT_final,
#     double k0, double dk0, double k1, double dk1,
#     double rho, double drho, double cp, double dcp,
#     double h_conv, double dh_conv, double T_inf, double dT_inf,
#     double T_hot, double dT_hot,
#     double* Q, double* dQ,
#     double Lx, double dLx, double Ly, double dLy, double dt, double ddt)
_lib.thermal_2d_jvp.restype = None
_lib.thermal_2d_jvp.argtypes = [
    ctypes.c_int,  # nx
    ctypes.c_int,  # ny
    ctypes.c_int,  # n_steps
    ctypes.POINTER(ctypes.c_double),  # T_init
    ctypes.POINTER(ctypes.c_double),  # dT_init
    ctypes.POINTER(ctypes.c_double),  # T_final
    ctypes.POINTER(ctypes.c_double),  # dT_final
    ctypes.c_double,  # k0
    ctypes.c_double,  # dk0
    ctypes.c_double,  # k1
    ctypes.c_double,  # dk1
    ctypes.c_double,  # rho
    ctypes.c_double,  # drho
    ctypes.c_double,  # cp
    ctypes.c_double,  # dcp
    ctypes.c_double,  # h_conv
    ctypes.c_double,  # dh_conv
    ctypes.c_double,  # T_inf
    ctypes.c_double,  # dT_inf
    ctypes.c_double,  # T_hot
    ctypes.c_double,  # dT_hot
    ctypes.POINTER(ctypes.c_double),  # Q
    ctypes.POINTER(ctypes.c_double),  # dQ
    ctypes.c_double,  # Lx
    ctypes.c_double,  # dLx
    ctypes.c_double,  # Ly
    ctypes.c_double,  # dLy
    ctypes.c_double,  # dt
    ctypes.c_double,  # ddt
]


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    """Get a ctypes double pointer from a contiguous float64 array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# -- Schemas ---------------------------------------------------------------


class InputSchema(BaseModel):
    """Input for a 2D transient heat conduction solver.

    Solves rho*cp*dT/dt = div(k(T)*grad(T)) + Q on a rectangular domain
    with Dirichlet (hot wall), convection, and insulated boundary conditions.
    """

    # Initial temperature field (flattened row-major, nx*ny)
    T_init: Differentiable[Array[(None,), Float64]] = Field(
        description=(
            "Initial temperature field [K]. Flattened row-major array of "
            "shape (nx*ny,). Index (i,j) maps to j*nx+i."
        ),
    )

    # Grid dimensions (not differentiable — integer-like)
    nx: int = Field(
        default=20,
        description="Number of grid points in x direction.",
        ge=3,
    )
    ny: int = Field(
        default=20,
        description="Number of grid points in y direction.",
        ge=3,
    )

    # Time integration
    n_steps: int = Field(
        default=100,
        description="Number of explicit Euler time steps.",
        ge=1,
    )
    dt: Differentiable[Float64] = Field(
        default=0.01,
        description="Time step size [s].",
        gt=0.0,
    )

    # Domain geometry
    Lx: Differentiable[Float64] = Field(
        default=0.1,
        description="Domain length in x [m].",
        gt=0.0,
    )
    Ly: Differentiable[Float64] = Field(
        default=0.05,
        description="Domain length in y [m].",
        gt=0.0,
    )

    # Material properties
    k0: Differentiable[Float64] = Field(
        default=45.0,
        description="Base thermal conductivity [W/(m*K)]. k(T) = k0 + k1*T.",
        gt=0.0,
    )
    k1: Differentiable[Float64] = Field(
        default=-0.01,
        description=(
            "Temperature coefficient of conductivity [W/(m*K^2)]. "
            "k(T) = k0 + k1*T. Negative values model metals."
        ),
    )
    rho: Differentiable[Float64] = Field(
        default=7850.0,
        description="Density [kg/m^3].",
        gt=0.0,
    )
    cp: Differentiable[Float64] = Field(
        default=460.0,
        description="Specific heat capacity [J/(kg*K)].",
        gt=0.0,
    )

    # Boundary conditions
    h_conv: Differentiable[Float64] = Field(
        default=25.0,
        description="Convective heat transfer coefficient at top boundary [W/(m^2*K)].",
        gt=0.0,
    )
    T_inf: Differentiable[Float64] = Field(
        default=293.15,
        description="Ambient temperature for convection BC [K].",
    )
    T_hot: Differentiable[Float64] = Field(
        default=373.15,
        description="Fixed temperature at bottom (Dirichlet) boundary [K].",
    )

    # Volumetric heat source (flattened row-major, nx*ny)
    Q: Differentiable[Array[(None,), Float64]] = Field(
        description=(
            "Volumetric heat source [W/m^3]. Flattened row-major array of "
            "shape (nx*ny,). Use zeros for no internal heating."
        ),
    )

    @model_validator(mode="after")
    def check_array_sizes(self) -> Self:
        """Verify T_init and Q have the correct size."""
        expected = self.nx * self.ny
        if len(self.T_init) != expected:
            raise ValueError(
                f"T_init has {len(self.T_init)} elements, expected nx*ny = {expected}."
            )
        if len(self.Q) != expected:
            raise ValueError(
                f"Q has {len(self.Q)} elements, expected nx*ny = {expected}."
            )
        return self

    @model_validator(mode="after")
    def check_stability(self) -> Self:
        """Check CFL stability for the explicit scheme.

        For temperature-dependent conductivity, use k_max = k0 + k1*T_hot
        (conservative estimate with the hottest expected temperature).
        """
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        k_max = self.k0 + self.k1 * self.T_hot
        if k_max <= 0:
            raise ValueError(
                f"Conductivity k(T_hot) = {k_max:.4f} <= 0. Increase k0 or reduce |k1|."
            )
        r = k_max * self.dt / (self.rho * self.cp) * (1.0 / (dx * dx) + 1.0 / (dy * dy))
        if r > 0.5:
            raise ValueError(
                f"CFL stability condition violated: r = {r:.4f} > 0.5. "
                f"Reduce dt, or increase grid spacing."
            )
        return self


class OutputSchema(BaseModel):
    """Output: temperature field after time integration."""

    T_final: Differentiable[Array[(None,), Float64]] = Field(
        description=(
            "Temperature field after n_steps time steps [K]. "
            "Flattened row-major array of shape (nx*ny,)."
        ),
    )


# -- Required endpoints ----------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the 2D thermal solver for n_steps explicit Euler steps."""
    T_init = np.ascontiguousarray(inputs.T_init, dtype=np.float64)
    Q = np.ascontiguousarray(inputs.Q, dtype=np.float64)
    n = inputs.nx * inputs.ny
    T_final = np.zeros(n, dtype=np.float64)

    _lib.thermal_2d_forward(
        inputs.nx,
        inputs.ny,
        inputs.n_steps,
        _as_ptr(T_init),
        _as_ptr(T_final),
        inputs.k0,
        inputs.k1,
        inputs.rho,
        inputs.cp,
        inputs.h_conv,
        inputs.T_inf,
        inputs.T_hot,
        _as_ptr(Q),
        inputs.Lx,
        inputs.Ly,
        inputs.dt,
    )

    return OutputSchema(T_final=T_final)


# -- Optional endpoints (AD via Enzyme) ------------------------------------


# All differentiable scalar parameters, in the order they appear in the wrapper
_SCALAR_PARAMS = ["k0", "k1", "rho", "cp", "h_conv", "T_inf", "T_hot", "Lx", "Ly", "dt"]


def _run_vjp(inputs: InputSchema, cotangent_T_final: np.ndarray):
    """Run Enzyme reverse-mode AD and return all gradients."""
    T_init = np.ascontiguousarray(inputs.T_init, dtype=np.float64)
    Q = np.ascontiguousarray(inputs.Q, dtype=np.float64)
    n = inputs.nx * inputs.ny

    # Shadow arrays (Enzyme accumulates gradients into these)
    dT_init = np.zeros(n, dtype=np.float64)
    T_final = np.zeros(n, dtype=np.float64)
    dT_final = np.array(cotangent_T_final, dtype=np.float64)
    dQ = np.zeros(n, dtype=np.float64)

    # Shadow scalars
    dk0 = ctypes.c_double(0.0)
    dk1 = ctypes.c_double(0.0)
    drho = ctypes.c_double(0.0)
    dcp = ctypes.c_double(0.0)
    dh_conv = ctypes.c_double(0.0)
    dT_inf = ctypes.c_double(0.0)
    dT_hot = ctypes.c_double(0.0)
    dLx = ctypes.c_double(0.0)
    dLy = ctypes.c_double(0.0)
    ddt = ctypes.c_double(0.0)

    _lib.thermal_2d_vjp(
        inputs.nx,
        inputs.ny,
        inputs.n_steps,
        _as_ptr(T_init),
        _as_ptr(dT_init),
        _as_ptr(T_final),
        _as_ptr(dT_final),
        inputs.k0,
        ctypes.byref(dk0),
        inputs.k1,
        ctypes.byref(dk1),
        inputs.rho,
        ctypes.byref(drho),
        inputs.cp,
        ctypes.byref(dcp),
        inputs.h_conv,
        ctypes.byref(dh_conv),
        inputs.T_inf,
        ctypes.byref(dT_inf),
        inputs.T_hot,
        ctypes.byref(dT_hot),
        _as_ptr(Q),
        _as_ptr(dQ),
        inputs.Lx,
        ctypes.byref(dLx),
        inputs.Ly,
        ctypes.byref(dLy),
        inputs.dt,
        ctypes.byref(ddt),
    )

    return {
        "T_init": dT_init,
        "Q": dQ,
        "k0": dk0.value,
        "k1": dk1.value,
        "rho": drho.value,
        "cp": dcp.value,
        "h_conv": dh_conv.value,
        "T_inf": dT_inf.value,
        "T_hot": dT_hot.value,
        "Lx": dLx.value,
        "Ly": dLy.value,
        "dt": ddt.value,
    }


def _run_jvp(inputs: InputSchema, tangents: dict[str, Any]):
    """Run Enzyme forward-mode AD and return output tangent."""
    T_init = np.ascontiguousarray(inputs.T_init, dtype=np.float64)
    Q = np.ascontiguousarray(inputs.Q, dtype=np.float64)
    n = inputs.nx * inputs.ny

    dT_init = np.ascontiguousarray(
        tangents.get("T_init", np.zeros(n, dtype=np.float64)),
        dtype=np.float64,
    )
    dQ = np.ascontiguousarray(
        tangents.get("Q", np.zeros(n, dtype=np.float64)),
        dtype=np.float64,
    )
    T_final = np.zeros(n, dtype=np.float64)
    dT_final = np.zeros(n, dtype=np.float64)

    _lib.thermal_2d_jvp(
        inputs.nx,
        inputs.ny,
        inputs.n_steps,
        _as_ptr(T_init),
        _as_ptr(dT_init),
        _as_ptr(T_final),
        _as_ptr(dT_final),
        inputs.k0,
        float(tangents.get("k0", 0.0)),
        inputs.k1,
        float(tangents.get("k1", 0.0)),
        inputs.rho,
        float(tangents.get("rho", 0.0)),
        inputs.cp,
        float(tangents.get("cp", 0.0)),
        inputs.h_conv,
        float(tangents.get("h_conv", 0.0)),
        inputs.T_inf,
        float(tangents.get("T_inf", 0.0)),
        inputs.T_hot,
        float(tangents.get("T_hot", 0.0)),
        _as_ptr(Q),
        _as_ptr(dQ),
        inputs.Lx,
        float(tangents.get("Lx", 0.0)),
        inputs.Ly,
        float(tangents.get("Ly", 0.0)),
        inputs.dt,
        float(tangents.get("dt", 0.0)),
    )

    return dT_final


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Reverse-mode AD via Enzyme: compute v^T @ J."""
    n = inputs.nx * inputs.ny
    cotangent_T_final = cotangent_vector.get(
        "T_final",
        np.zeros(n, dtype=np.float64),
    )
    all_grads = _run_vjp(inputs, cotangent_T_final)

    return {k: v for k, v in all_grads.items() if k in vjp_inputs}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Forward-mode AD via Enzyme: compute J @ v."""
    dT_final = _run_jvp(inputs, tangent_vector)

    result = {}
    if "T_final" in jvp_outputs:
        result["T_final"] = dT_final
    return result
