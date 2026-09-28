# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-timestep Burgers' equation solver Tesseract (PyTorch).

Solves one explicit Euler step of the 1D viscous Burgers' equation:

    u^{n+1} = u^n + dt * (-u * du/dx + nu * d²u/dx²)

The viscosity field nu is provided as an input — the solver does not compute it.
This clean interface (state + material field → next state) is the same contract
that a Fortran solver with an adjoint could implement. The outer time-stepping
loop and closure evaluation live in the caller, enabling per-timestep closure
calls and end-to-end gradient flow through both solver and closure.
"""

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils._pytree import tree_map

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Default grid size
N = 128

# --- Grid setup (fixed for this Tesseract) ---
DX = 1.0 / (N - 1)

to_tensor = lambda x: (
    torch.tensor(x, dtype=torch.float64)
    if isinstance(x, np.generic | np.ndarray)
    else x
)


class InputSchema(BaseModel):
    u: Differentiable[Array[(N,), Float64]] = Field(
        description="Current velocity field on the grid"
    )
    nu: Differentiable[Array[(N,), Float64]] = Field(
        description="Viscosity field at each grid point (must be positive)"
    )
    dt: float = Field(description="Time step size", default=1e-4)


class OutputSchema(BaseModel):
    u_next: Differentiable[Array[(N,), Float64]] = Field(
        description="Velocity field after one time step"
    )


def evaluate(inputs: dict) -> dict:
    """Core differentiable computation — pure torch operations."""
    u = inputs["u"]
    nu = inputs["nu"]
    dt = inputs["dt"]

    # Spatial derivatives via plain central differences. The flow stays smooth
    # and low-Reynolds here (gentle low-frequency ICs, viscosity bounded by the
    # closure's sigmoid so nu*dt/dx^2 stays within the explicit CFL limit), so no
    # shocks form and we don't need upwinding, a conservative/flux form, or a
    # stiff integrator (e.g. ETDRK). A real solver in a harder regime would.
    dudx = torch.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2 * DX)

    d2udx2 = torch.zeros_like(u)
    d2udx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (DX**2)

    # Burgers' equation: du/dt = -u * du/dx + nu * d²u/dx²
    dudt = -u * dudx + nu * d2udx2

    # Forward Euler step
    u_next = u + dt * dudt

    # Enforce boundary conditions (Dirichlet: hold boundary values)
    u_next = torch.cat([u[:1], u_next[1:-1], u[-1:]])

    return {"u_next": u_next}


def apply(inputs: InputSchema) -> OutputSchema:
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    return evaluate(tensor_inputs)


def abstract_eval(abstract_inputs: Any) -> Any:
    return {"u_next": {"shape": [N], "dtype": "float64"}}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    jvp_inputs = tuple(jvp_inputs)
    tangent_vector = {key: tangent_vector[key] for key in jvp_inputs}

    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    pos_tangent = tree_map(to_tensor, tangent_vector).values()
    pos_inputs = flatten_with_paths(tensor_inputs, jvp_inputs).values()

    filtered_pos_eval = filter_func(
        evaluate, tensor_inputs, jvp_outputs, input_paths=jvp_inputs
    )

    return torch.func.jvp(filtered_pos_eval, tuple(pos_inputs), tuple(pos_tangent))[1]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    vjp_inputs = tuple(vjp_inputs)
    cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}

    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    tensor_cotangent = tree_map(to_tensor, cotangent_vector)
    pos_inputs = flatten_with_paths(tensor_inputs, vjp_inputs).values()

    filtered_pos_func = filter_func(
        evaluate, tensor_inputs, vjp_outputs, input_paths=vjp_inputs
    )

    _, vjp_func = torch.func.vjp(filtered_pos_func, *pos_inputs)
    vjp_vals = vjp_func(tensor_cotangent)
    return dict(zip(vjp_inputs, vjp_vals, strict=True))


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    jac_inputs = tuple(jac_inputs)
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    pos_inputs = flatten_with_paths(tensor_inputs, jac_inputs).values()

    filtered_pos_eval = filter_func(
        evaluate, tensor_inputs, jac_outputs, input_paths=jac_inputs
    )

    def filtered_pos_eval_flat(*args):
        res = filtered_pos_eval(*args)
        return tuple(res[k] for k in jac_outputs)

    jac = torch.autograd.functional.jacobian(filtered_pos_eval_flat, tuple(pos_inputs))

    jac_dict = {}
    for dy, dys in zip(jac_outputs, jac, strict=True):
        jac_dict[dy] = {}
        for dx, dxs in zip(jac_inputs, dys, strict=True):
            jac_dict[dy][dx] = dxs

    return jac_dict
