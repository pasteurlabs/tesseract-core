# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector normalized according to [...]"
    )
    b: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector. Needs to have the same dimensions as a."
    )
    s: int | float = Field(description="A scalar.", default=3)
    normalize: bool = Field(
        description="True if the output should be normalized, False otherwise.",
        default=False,
    )


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="Vector s·a + b"
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    result_shape = abstract_inputs.a
    assert result_shape is not None

    return {"result": result_shape}


def apply(inputs: InputSchema) -> OutputSchema:
    """Multiplies a vector `a` by `s`, and sums the result to `b`."""
    a = inputs.a
    b = inputs.b
    s = inputs.s

    assert a.ndim == 1
    assert a.shape == b.shape
    result = a * s + b

    if inputs.normalize:
        norm = np.linalg.norm(result, ord=2)
        result /= norm

    return OutputSchema(result=result)


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    a = inputs.a
    b = inputs.b
    s = inputs.s

    assert a.ndim == 1
    assert a.shape == b.shape

    n = len(inputs.a)

    partials = {}
    partials["a"] = np.eye(n) * s
    partials["b"] = np.eye(n)
    partials["s"] = a.reshape(1, 3)

    if inputs.normalize:
        result = a * s + b
        norm = np.linalg.norm(result, ord=2)
        partials["a"] = partials["a"] / norm - np.outer(result, (a + s * b)) / norm**3
        partials["b"] = partials["b"] / norm - np.outer(result, (s * a + b)) / norm**3
        partials["s"] = partials["s"] - (a + s * b) / norm**3 * (s * a * a + 2 * a * b)

    jacobian = {"result": {v: partials[v] for v in jac_inputs}}
    return jacobian


# Optional endpoints, do not remove (are used to generate complete API for docs)


def jacobian_vector_product(
    inputs: InputSchema, jvp_inputs: set[str], jvp_outputs: set[str], tangent_vector
):
    return {}


def vector_jacobian_product(
    inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector
):
    return {}
