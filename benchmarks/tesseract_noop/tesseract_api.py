# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A no-op Tesseract for benchmarking framework overhead.

This Tesseract does nothing but decode inputs and encode outputs,
making it ideal for measuring pure framework overhead without any
computation contaminating the results.
"""

from pydantic import BaseModel

from tesseract_core.runtime import Array, Float64


class InputSchema(BaseModel):
    """Input schema with a single, optional array.

    The array is optional so a benchmark can send an empty payload
    (``{"inputs": {}}``) to measure the fixed per-request floor with no array
    data at all, alongside the array-carrying cases.
    """

    data: Array[(None,), Float64] | None = None


class OutputSchema(BaseModel):
    """Output schema returning the same (optional) array."""

    result: Array[(None,), Float64] | None = None


def apply(inputs: InputSchema) -> OutputSchema:
    """Identity function - returns input unchanged.

    This measures pure framework overhead: serialization, validation,
    HTTP transport, and deserialization. With an empty payload it isolates the
    fixed per-request floor (no array serialization at all).
    """
    return OutputSchema(result=inputs.data)
