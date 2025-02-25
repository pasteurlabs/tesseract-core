# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float32


def log_rosenbrock(x: float, y: float, a: float = 1.0, b: float = 100.0):
    """The log-rosenbrock function.

    https://en.wikipedia.org/wiki/Rosenbrock_function

    Global minimum is (x, y) = (a, a**2).
    """
    # compute Rosenbrock function
    rosenbrock = (a - x) ** 2 + b * (y - x**2) ** 2

    # ensure it's a pytorch tensor
    rosenbrock = torch.as_tensor(rosenbrock)

    # take log with small offset to prevent log(0) at (x,y) = (a, a^2)
    return torch.log(rosenbrock + 1e-5)


#
# Schemas
#


class InputSchema(BaseModel):
    x: Differentiable[Float32] = Field(description="X-value of inputs")
    y: Differentiable[Float32] = Field(description="Y-value of inputs")
    a: Float32
    b: Float32


class OutputSchema(BaseModel):
    loss: Differentiable[Float32] = Field(description="Rosenbrock loss function value")


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    loss = log_rosenbrock(
        inputs.x,
        inputs.y,
        inputs.a,
        inputs.b,
    )

    return OutputSchema(loss=loss)


#
# Optional endpoints
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    # create pytorch tensors
    inputs.x = torch.as_tensor(inputs.x)
    inputs.y = torch.as_tensor(inputs.y)
    inputs.a = torch.as_tensor(inputs.a)
    inputs.b = torch.as_tensor(inputs.b)

    # make tensors differentiable
    for key in jac_inputs:
        setattr(
            inputs,
            key,
            torch.nn.Parameter(getattr(inputs, key)),
        )

    # perform forward pass
    jac_result = {dy: {} for dy in jac_outputs}
    with torch.enable_grad():
        # do forward pass with graph
        output = log_rosenbrock(inputs.x, inputs.y, inputs.a, inputs.b)

        # do backprop to get gradients
        for dx in jac_inputs:
            grads = torch.autograd.grad(output, getattr(inputs, dx), retain_graph=True)[
                0
            ]
            for dy in jac_outputs:
                jac_result[dy][dx] = grads

    return jac_result
