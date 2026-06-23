# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract wrapping an FMI 2.0 Model Exchange (ME) FMU via FMPy.

Unlike a Co-Simulation FMU (which integrates internally), this Tesseract exposes
the model's *right-hand side* as a single, stateless evaluation::

    dx/dt = f(t, x, u; p)

Time integration is intentionally left to the caller -- e.g. a JAX/diffrax program
that integrates over this Tesseract via ``tesseract-jax`` (see
``parameter_estimation.py``). Keeping ``apply`` a pure function of its inputs is what
makes the Tesseract differentiable and safe to call concurrently.

Differentiability hooks into FMI's native sensitivity API:
``fmi2GetDirectionalDerivative`` computes a Jacobian-vector product (J @ v) -- i.e. a
forward-mode JVP -- exactly. We therefore wire ``jacobian_vector_product`` straight to
it, assemble ``jacobian`` by sweeping unit tangents, and derive
``vector_jacobian_product`` from that Jacobian (the local VJP that diffrax's
path-level reverse-mode adjoint consumes). Inputs the FMU cannot differentiate via
directional derivatives -- the simulation time ``t`` (no value reference) and
``parameters`` (VanDerPol, like many FMUs, declares no derivative dependency on them) --
fall back to finite differences.

The schema is intentionally FMU-agnostic: states/inputs/outputs and their FMI value
references are discovered from ``modelDescription.xml`` at import. To wrap a different
FMU, drop in a new ``.fmu`` (and adjust ``FMU_PARAMETERS`` if needed) -- no endpoint
code changes required.
"""

import atexit
import os
import shutil
import threading
from ctypes import POINTER, c_double
from pathlib import Path
from typing import Any, Self

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Model
from pydantic import BaseModel, Field, model_validator

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    vjp_from_jacobian,
)

# --------------------------------------------------------------------------------------
# FMU configuration and one-time metadata loading (runs once at import).
# --------------------------------------------------------------------------------------

# Which FMU to wrap. Override with the FMU_PATH env var to point at a different bundled
# file without rebuilding the rest of the Tesseract logic.
FMU_PATH = Path(
    os.environ.get("FMU_PATH", Path(__file__).parent / "fmus" / "VanDerPol.fmu")
)

# Real parameters exposed as differentiable inputs (e.g. for parameter estimation).
# Comma-separated list of FMU variable names; override with the FMU_PARAMETERS env var.
PARAMETER_NAMES = [
    name for name in os.environ.get("FMU_PARAMETERS", "mu").split(",") if name
]

_MODEL_DESCRIPTION = read_model_description(FMU_PATH)
_UNZIP_DIR = extract(FMU_PATH)
atexit.register(shutil.rmtree, _UNZIP_DIR, ignore_errors=True)

# Map names/value references once. In FMI 2.0 each entry of ``derivatives`` is the
# der(state) variable; its ``.derivative`` points back at the state it differentiates.
_STATE_NAMES = [d.variable.derivative.name for d in _MODEL_DESCRIPTION.derivatives]
_STATE_VRS = [
    d.variable.derivative.valueReference for d in _MODEL_DESCRIPTION.derivatives
]
_DERIV_VRS = [d.variable.valueReference for d in _MODEL_DESCRIPTION.derivatives]
_INPUT_VRS = [
    v.valueReference
    for v in _MODEL_DESCRIPTION.modelVariables
    if v.causality == "input"
]
_OUTPUT_VRS = [
    v.valueReference
    for v in _MODEL_DESCRIPTION.modelVariables
    if v.causality == "output"
]

_VAR_BY_NAME = {v.name: v for v in _MODEL_DESCRIPTION.modelVariables}
_PARAM_VRS = [_VAR_BY_NAME[name].valueReference for name in PARAMETER_NAMES]
_PARAM_DEFAULTS = np.array(
    [float(_VAR_BY_NAME[name].start) for name in PARAMETER_NAMES], dtype=np.float64
)

N_STATES = len(_STATE_VRS)
N_INPUTS = len(_INPUT_VRS)
N_OUTPUTS = len(_OUTPUT_VRS)
N_PARAMS = len(_PARAM_VRS)

# FMI 2.0 forward directional derivatives give exact JVPs w.r.t. states and inputs.
# NOTE: FMI 3.0 uses the *plural* XML attribute ``providesDirectionalDerivatives``;
# fmpy may not surface it on the model-description object, so for FMI 3.0 read the raw
# XML rather than trusting this flag.
_PROVIDES_DIRECTIONAL_DERIVATIVE = bool(
    _MODEL_DESCRIPTION.modelExchange.providesDirectionalDerivative
)

# Input groups FMI directional derivatives can handle. Everything else ("t",
# "parameters") uses finite differences.
_DD_INPUT_GROUPS = {"x", "u"}

# A single FMU instance, created once at import and reused for every call. Allocating an
# instance is ~100x more expensive than resetting one (measured), and the Model Exchange
# evaluation itself is cheap once the point is set. Fixed parameters (e.g. VanDerPol's
# `mu`) can only be set in initialization mode, so each call resets and re-initializes --
# which keeps `apply` a pure function of its inputs despite the shared instance. The lock
# serializes access in case the server dispatches requests on multiple threads.
_FMU = FMU2Model(
    guid=_MODEL_DESCRIPTION.guid,
    unzipDirectory=_UNZIP_DIR,
    modelIdentifier=_MODEL_DESCRIPTION.modelExchange.modelIdentifier,
    instanceName="tesseract_me",
)
_FMU.instantiate()
_FMU_LOCK = threading.Lock()


def _free_fmu() -> None:
    """Best-effort teardown of the shared FMU instance at interpreter exit."""
    for teardown in (_FMU.terminate, _FMU.freeInstance):
        try:
            teardown()
        except Exception:
            pass


# Registered after the unzip-dir cleanup so it runs first (atexit is LIFO): free the
# instance before the directory holding its shared library is removed.
atexit.register(_free_fmu)


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------


class InputSchema(BaseModel):
    """A single right-hand-side evaluation point ``dx/dt = f(t, x, u; p)``."""

    t: Differentiable[Float64] = Field(default=0.0, description="Simulation time [s].")
    x: Differentiable[Array[(None,), Float64]] = Field(
        description=f"Continuous state vector, length {N_STATES}, ordered as {_STATE_NAMES}.",
    )
    u: Differentiable[Array[(None,), Float64]] = Field(
        default_factory=lambda: np.zeros(N_INPUTS, dtype=np.float64),
        description=f"Continuous input vector, length {N_INPUTS}.",
    )
    parameters: Differentiable[Array[(None,), Float64]] = Field(
        default_factory=lambda: _PARAM_DEFAULTS.copy(),
        description=f"Tunable real parameters {PARAMETER_NAMES}, length {N_PARAMS}.",
    )

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        for name, value, expected in (
            ("x", self.x, N_STATES),
            ("u", self.u, N_INPUTS),
            ("parameters", self.parameters, N_PARAMS),
        ):
            if value.shape != (expected,):
                raise ValueError(
                    f"{name} must have shape ({expected},), got {value.shape}."
                )
        return self


class OutputSchema(BaseModel):
    """Right-hand-side value and (informational) FMU outputs."""

    dx_dt: Differentiable[Array[(None,), Float64]] = Field(
        description="State derivatives dx/dt = f(t, x, u; p).",
    )
    y: Array[(None,), Float64] = Field(
        description=(
            f"FMU output variables, length {N_OUTPUTS}. Non-differentiable: VanDerPol "
            "(like many FMUs) declares no dependency structure for its outputs, so their "
            "directional derivatives are unavailable. For VanDerPol the outputs coincide "
            "with the states."
        ),
    )


# --------------------------------------------------------------------------------------
# FMU lifecycle helpers (one shared instance, reset + re-initialized per call)
# --------------------------------------------------------------------------------------


def _double_ptr(array: np.ndarray) -> "POINTER(c_double)":
    """Return a C ``double*`` view of a contiguous float64 array (for FMPy calls)."""
    return array.ctypes.data_as(POINTER(c_double))


def _reinit(t: float, parameters: np.ndarray) -> None:
    """Reset the shared instance and re-enter ME time mode (caller holds ``_FMU_LOCK``).

    Parameters (e.g. VanDerPol's ``mu``) have ``fixed`` variability, so they can only be
    (re)set here, in initialization mode -- hence the reset on every call.
    """
    _FMU.reset()
    _FMU.setupExperiment(startTime=float(t))
    _FMU.enterInitializationMode()
    if N_PARAMS:
        _FMU.setReal(_PARAM_VRS, [float(p) for p in parameters])
    _FMU.exitInitializationMode()
    _FMU.enterContinuousTimeMode()


def _set_point(t: float, x: np.ndarray, u: np.ndarray) -> None:
    """Set time, inputs, and continuous states on the shared instance (lock held)."""
    _FMU.setTime(float(t))
    if N_INPUTS:
        _FMU.setReal(_INPUT_VRS, [float(v) for v in u])
    states = np.ascontiguousarray(x, dtype=np.float64)
    _FMU.setContinuousStates(_double_ptr(states), N_STATES)


def _evaluate(
    t: float, x: np.ndarray, u: np.ndarray, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``dx/dt`` and outputs ``y`` at a single point."""
    with _FMU_LOCK:
        _reinit(t, parameters)
        _set_point(t, x, u)
        dx_dt = np.zeros(N_STATES, dtype=np.float64)
        _FMU.getDerivatives(_double_ptr(dx_dt), N_STATES)
        y = (
            np.array(_FMU.getReal(_OUTPUT_VRS), dtype=np.float64)
            if N_OUTPUTS
            else np.zeros(0, dtype=np.float64)
        )
    return dx_dt, y


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Evaluate the FMU right-hand side at a single point."""
    dx_dt, y = _evaluate(inputs.t, inputs.x, inputs.u, inputs.parameters)
    return OutputSchema(dx_dt=dx_dt, y=y)


def abstract_eval(abstract_inputs: Any) -> dict:
    """Return output shapes/dtypes without running the FMU (sizes are fixed at import)."""
    return {
        "dx_dt": ShapeDType(shape=(N_STATES,), dtype="float64"),
        "y": ShapeDType(shape=(N_OUTPUTS,), dtype="float64"),
    }


def jacobian(
    inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Jacobian of ``dx_dt`` w.r.t. requested inputs.

    Columns for state/input directions come from FMI directional derivatives; columns
    for ``t``/``parameters`` come from finite differences.
    """
    jac_inputs, jac_outputs = set(jac_inputs), set(jac_outputs)
    result: dict[str, dict[str, np.ndarray]] = {dy: {} for dy in jac_outputs}

    dd_inputs = (
        jac_inputs & _DD_INPUT_GROUPS if _PROVIDES_DIRECTIONAL_DERIVATIVE else set()
    )
    fd_inputs = jac_inputs - dd_inputs

    if dd_inputs and "dx_dt" in jac_outputs:
        with _FMU_LOCK:
            _reinit(inputs.t, inputs.parameters)
            _set_point(inputs.t, inputs.x, inputs.u)
            for group in dd_inputs:
                known_vrs = _STATE_VRS if group == "x" else _INPUT_VRS
                columns = []
                for j in range(len(known_vrs)):
                    seed = [0.0] * len(known_vrs)
                    seed[j] = 1.0
                    columns.append(
                        np.array(
                            _FMU.getDirectionalDerivative(_DERIV_VRS, known_vrs, seed),
                            dtype=np.float64,
                        )
                    )
                # Shape (N_STATES, len(known_vrs)) = (d output, d input).
                result["dx_dt"][group] = (
                    np.stack(columns, axis=1)
                    if columns
                    else np.zeros((N_STATES, 0), dtype=np.float64)
                )

    if fd_inputs:
        fd_jac = finite_difference_jacobian(
            apply, inputs, fd_inputs, jac_outputs, algorithm="central", eps=1e-6
        )
        for dy in jac_outputs:
            for group in fd_inputs:
                result[dy][group] = np.asarray(fd_jac[dy][group])

    return result


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Forward-mode JVP ``J @ v``, summed over input directions.

    State/input directions go straight through ``fmi2GetDirectionalDerivative``;
    ``t``/``parameters`` directions use finite differences. JVP is linear in the
    tangent, so the per-group contributions simply add.
    """
    jvp_inputs, jvp_outputs = set(jvp_inputs), set(jvp_outputs)
    out = {dy: np.zeros(N_STATES, dtype=np.float64) for dy in jvp_outputs}

    dd_inputs = (
        jvp_inputs & _DD_INPUT_GROUPS if _PROVIDES_DIRECTIONAL_DERIVATIVE else set()
    )
    fd_inputs = jvp_inputs - dd_inputs

    if dd_inputs and "dx_dt" in jvp_outputs:
        with _FMU_LOCK:
            _reinit(inputs.t, inputs.parameters)
            _set_point(inputs.t, inputs.x, inputs.u)
            for group in dd_inputs:
                known_vrs = _STATE_VRS if group == "x" else _INPUT_VRS
                seed = [float(s) for s in np.asarray(tangent_vector[group]).ravel()]
                out["dx_dt"] = out["dx_dt"] + np.array(
                    _FMU.getDirectionalDerivative(_DERIV_VRS, known_vrs, seed),
                    dtype=np.float64,
                )

    if fd_inputs:
        fd_tangent = {group: tangent_vector[group] for group in fd_inputs}
        fd_jvp = finite_difference_jvp(
            apply,
            inputs,
            fd_inputs,
            jvp_outputs,
            fd_tangent,
            algorithm="central",
            eps=1e-6,
        )
        for dy in jvp_outputs:
            out[dy] = out[dy] + np.asarray(fd_jvp[dy])

    return out


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reverse-mode VJP ``v^T @ J``.

    This local VJP is what diffrax's path-level (reverse-mode) adjoint calls when
    integrating over the Tesseract. We assemble it from the Jacobian above (built from
    forward directional derivatives + finite differences). For high state dimension,
    FMI 3.0's ``fmi3GetAdjointDerivative`` would compute this directly in one call.
    """
    return vjp_from_jacobian(
        jacobian, inputs, vjp_inputs, vjp_outputs, cotangent_vector
    )
