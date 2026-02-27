# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A no-op Tesseract for benchmarking framework overhead.

This Tesseract does nothing but decode inputs and encode outputs,
making it ideal for measuring pure framework overhead without any
computation contaminating the results.
"""

from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float64


class InputSchema(BaseModel):
    """Input schema with a single array."""

    data: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    """Output schema returning the same array."""

    result: Differentiable[Array[(None,), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    """Identity function - returns input unchanged.

    This measures pure framework overhead: serialization, validation,
    HTTP transport, and deserialization.
    """
    return OutputSchema(result=inputs.data)


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict:
    """Returns identity Jacobian (derivative of identity is identity)."""
    import numpy as np

    n = inputs.data.shape[0]
    return {"result": {"data": np.eye(n)}}
