# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neural viscosity closure Tesseract (PyTorch).

A small MLP that predicts spatially-varying viscosity from local flow features.
Used as a learned closure inside a PDE solver — the solver calls this Tesseract
at every timestep to get the viscosity field, and gradients flow back through
both during training.

The network weights are passed as explicit inputs (not internal state) so that
an external optimizer can differentiate through the full solver-closure pipeline.
"""

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils._pytree import tree_map

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Network architecture constants
HIDDEN_DIM = 32
N_HIDDEN_LAYERS = 2

to_tensor = lambda x: (
    torch.tensor(x, dtype=torch.float64)
    if isinstance(x, np.generic | np.ndarray)
    else x
)


class InputSchema(BaseModel):
    u: Differentiable[Array[(None,), Float64]] = Field(
        description="Velocity field at grid points"
    )
    dudx: Differentiable[Array[(None,), Float64]] = Field(
        description="Velocity gradient du/dx at grid points"
    )
    x: Array[(None,), Float64] = Field(description="Spatial coordinates of grid points")
    # Network weights as flat arrays for easy composition
    w1: Differentiable[Array[(3, HIDDEN_DIM), Float64]] = Field(
        description="First layer weights (3 input features -> hidden)"
    )
    b1: Differentiable[Array[(HIDDEN_DIM,), Float64]] = Field(
        description="First layer bias"
    )
    w2: Differentiable[Array[(HIDDEN_DIM, HIDDEN_DIM), Float64]] = Field(
        description="Second layer weights"
    )
    b2: Differentiable[Array[(HIDDEN_DIM,), Float64]] = Field(
        description="Second layer bias"
    )
    w3: Differentiable[Array[(HIDDEN_DIM, 1), Float64]] = Field(
        description="Output layer weights (hidden -> 1)"
    )
    b3: Differentiable[Array[(1,), Float64]] = Field(description="Output layer bias")


class OutputSchema(BaseModel):
    nu: Differentiable[Array[(None,), Float64]] = Field(
        description="Predicted viscosity at each grid point (always positive)"
    )


def evaluate(inputs: dict) -> dict:
    """Core differentiable computation — pure torch operations."""
    u = inputs["u"]
    dudx = inputs["dudx"]
    x = inputs["x"]

    # Stack features: [u, dudx, x] at each grid point -> (N, 3)
    features = torch.stack([u, dudx, x], dim=-1)

    # Forward pass through MLP
    h = features @ inputs["w1"] + inputs["b1"]
    h = torch.tanh(h)
    h = h @ inputs["w2"] + inputs["b2"]
    h = torch.tanh(h)
    out = h @ inputs["w3"] + inputs["b3"]

    # Sigmoid * scale to keep viscosity in a physically reasonable range.
    # Range [0, nu_max] prevents CFL violations in the explicit solver.
    nu_max = 0.05
    nu = nu_max * torch.sigmoid(out[:, 0])

    return {"nu": nu}


def apply(inputs: InputSchema) -> OutputSchema:
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    return evaluate(tensor_inputs)


def abstract_eval(abstract_inputs: Any) -> Any:
    inputs_dict = abstract_inputs.model_dump()
    n = inputs_dict["u"]["shape"][0]
    return {"nu": {"shape": [n], "dtype": "float64"}}


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
