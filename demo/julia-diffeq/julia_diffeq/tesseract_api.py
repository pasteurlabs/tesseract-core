"""Tesseract wrapper for a Julia Lotka-Volterra ODE solver with gradient endpoints.

Demonstrates wrapping a Julia solver (DifferentialEquations.jl) as a
differentiable Tesseract. Gradients are computed via SciMLSensitivity's
adjoint methods — Julia's native AD machinery, not finite differences.

Python consumers call this Tesseract without any Julia installation.
JuliaCall handles the in-process bridge; it's an implementation detail
hidden inside the container.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64

# ---------------------------------------------------------------------------
# Julia initialization
# ---------------------------------------------------------------------------
# Julia + JuliaCall must be initialized *before* the Tesseract runtime
# redirects stderr through its TeePipe logging.  Julia's C runtime writes
# to fd 2 (stderr) on its own threads during startup; if that fd points to
# an os.pipe() the GIL can prevent the Python reader thread from draining
# the pipe, deadlocking the process.
#
# Initializing at import time is safe because tesseract_api.py is loaded by
# the CLI before the TeePipe redirect is established.  During Docker build
# the _TESSERACT_IS_BUILDING env-var skips this (Julia isn't needed then).
# ---------------------------------------------------------------------------

_jl = None


def _get_jl():
    global _jl
    if _jl is None:
        import juliacall

        _jl = juliacall.Main
        _jl.seval('include("/tesseract/julia/solve.jl")')
        _jl.seval("using .LotkaVolterraSolver")
    return _jl


_get_jl()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class InputSchema(BaseModel):
    params: Differentiable[Array[(4,), Float64]] = Field(
        description="Lotka-Volterra parameters [alpha, beta, delta, gamma]. "
        "alpha: prey birth rate, beta: predation rate, "
        "delta: predator growth rate, gamma: predator death rate.",
    )
    u0: Differentiable[Array[(2,), Float64]] = Field(
        description="Initial conditions [prey_population, predator_population].",
    )
    t_end: float = Field(
        default=20.0,
        gt=0.0,
        description="End time for the simulation.",
    )
    n_saveat: int = Field(
        default=100,
        ge=2,
        le=10000,
        description="Number of equally-spaced time points to save.",
    )


class OutputSchema(BaseModel):
    trajectory: Differentiable[Array[(None, 2), Float64]] = Field(
        description="Solution trajectory, shape (n_saveat, 2). "
        "Columns are [prey, predator].",
    )
    time: Array[(None,), Float64] = Field(
        description="Time points corresponding to trajectory rows.",
    )


# ---------------------------------------------------------------------------
# Required endpoint
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Solve the Lotka-Volterra ODE system using DifferentialEquations.jl."""
    saveat = np.linspace(0.0, inputs.t_end, inputs.n_saveat)

    jl = _get_jl()
    t_jl, u_jl = jl.LotkaVolterraSolver.solve_lotka_volterra(
        inputs.params.tolist(),
        inputs.u0.tolist(),
        (0.0, inputs.t_end),
        saveat.tolist(),
    )

    # Convert Julia arrays to numpy via collect() to avoid PythonCall wrapping issues
    t = np.array(jl.collect(t_jl))
    # Julia matrices are column-major; vec() flattens column-by-column
    u_flat = np.array(jl.collect(jl.vec(u_jl)))
    trajectory = u_flat.reshape(inputs.n_saveat, 2, order="F")

    return OutputSchema(
        trajectory=trajectory,
        time=t,
    )


# ---------------------------------------------------------------------------
# Gradient endpoints
# ---------------------------------------------------------------------------


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Compute VJP using SciMLSensitivity's interpolating adjoint method.

    This calls Julia's native adjoint sensitivity analysis — the same
    machinery Julia users rely on, now accessible from Python.
    """
    saveat = np.linspace(0.0, inputs.t_end, inputs.n_saveat)

    cotangent_traj = cotangent_vector["trajectory"]

    jl = _get_jl()
    result = jl.LotkaVolterraSolver.vjp_lotka_volterra(
        inputs.params.tolist(),
        inputs.u0.tolist(),
        (0.0, inputs.t_end),
        saveat.tolist(),
        cotangent_traj.tolist(),
        list(vjp_inputs),
    )

    # Convert Julia Dict values to numpy via collect()
    return {str(k): np.array(jl.collect(v)) for k, v in result.items()}


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    """Compute the full Jacobian by sweeping VJP with one-hot cotangents."""
    from tesseract_core.runtime.experimental import jacobian_from_vjp

    return jacobian_from_vjp(
        vector_jacobian_product,
        apply,
        inputs,
        jac_inputs,
        jac_outputs,
    )


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Compute JVP by contracting the Jacobian with the tangent vector."""
    from tesseract_core.runtime.experimental import jvp_from_jacobian

    return jvp_from_jacobian(
        jacobian,
        inputs,
        jvp_inputs,
        jvp_outputs,
        tangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Infer output shapes without running the solver."""
    from tesseract_core.runtime import ShapeDType

    n = abstract_inputs.n_saveat
    return {
        "trajectory": ShapeDType(shape=(n, 2), dtype="float64"),
        "time": ShapeDType(shape=(n,), dtype="float64"),
    }
