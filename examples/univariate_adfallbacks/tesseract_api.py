# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example Tesseract demonstrating AD endpoint derivation fallbacks.

This example shows how to use jvp_from_jacobian and vjp_from_jacobian to
automatically derive the JVP and VJP endpoints from an existing Jacobian
implementation, without writing additional gradient code.

This is a variant of the univariate example (Rosenbrock function) where
the JVP and VJP endpoints are derived from the Jacobian instead of being
implemented manually.
"""

import jax
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float32, ShapeDType
from tesseract_core.runtime.experimental import jvp_from_jacobian, vjp_from_jacobian


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
    return jvp_from_jacobian(jacobian, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    return vjp_from_jacobian(
        jacobian, inputs, vjp_inputs, vjp_outputs, cotangent_vector
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {"result": ShapeDType(shape=(), dtype="Float32")}
