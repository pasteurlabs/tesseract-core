# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float32, ShapeDType


def rosenbrock(x: float, y: float, a: float = 1.0, b: float = 100.0) -> float:
    return (a - x) ** 2 + b * (y - x**2) ** 2


#
# Schemas
#


class InputSchema(BaseModel):
    x: Differentiable[Float32] = Field(description="Scalar value x.", default=0.0)
    y: Differentiable[Float32] = Field(description="Scalar value y.", default=0.0)
    a: Float32 = Field(description="Scalar parameter a.", default=1.0)
    b: Float32 = Field(description="Scalar parameter b.", default=100.0)


class OutputSchema(BaseModel):
    result: Differentiable[Float32] = Field(
        description="Result of Rosenbrock function evaluation."
    )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Evaluates the Rosenbrock function given input values and parameters."""
    result = rosenbrock(inputs.x, inputs.y, a=inputs.a, b=inputs.b)
    return OutputSchema(result=result)


#
# Optional endpoints
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    rosenbrock_signature = ["x", "y", "a", "b"]

    jac_result = {dy: {} for dy in jac_outputs}
    for dx in jac_inputs:
        grad_func = jax.jacrev(rosenbrock, argnums=rosenbrock_signature.index(dx))
        for dy in jac_outputs:
            jac_result[dy][dx] = grad_func(inputs.x, inputs.y, inputs.a, inputs.b)

    return jac_result


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector,
):
    # NOTE: This is a naive implementation of JVP, which is not efficient.
    jac = jacobian(inputs, jvp_inputs, jvp_outputs)
    out = {}
    for dy in jvp_outputs:
        out[dy] = sum(jac[dy][dx] * tangent_vector[dx] for dx in jvp_inputs)
    return out


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    # NOTE: This is a naive implementation of VJP, which is not efficient.
    jac = jacobian(inputs, vjp_inputs, vjp_outputs)
    out = {}
    for dx in vjp_inputs:
        out[dx] = sum(jac[dy][dx] * cotangent_vector[dy] for dy in vjp_outputs)
    return out


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {"result": ShapeDType(shape=(), dtype="Float32")}
