# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU Tesseract that returns its results as device (PyTorch) memory.

Because every endpoint returns CUDA tensors that expose
``__cuda_array_interface__``, serving this Tesseract with
``gpu_transport="cuda_ipc"`` exercises the full CUDA IPC export path: the runtime
hands back device-memory IPC handles instead of copying results to host. This
requires a real GPU and is only built/run by the GPU end-to-end tests
(``tests/endtoend_tests/test_serving_gpu.py``).
"""

from typing import Any

import torch
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector."
    )
    b: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector, same shape as a."
    )
    s: float = Field(description="A scalar.", default=3.0)


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="Vector s * a + b, on GPU."
    )


def _partial_scales(inputs: InputSchema) -> dict[str, float]:
    """Scalar factors of d(result)/da and d(result)/db, which are multiples of the identity."""
    return {"a": inputs.s, "b": 1.0}


def apply(inputs: InputSchema) -> OutputSchema:
    """Compute ``s * a + b`` on the GPU and return device memory."""
    a = torch.as_tensor(inputs.a, device="cuda")
    b = torch.as_tensor(inputs.b, device="cuda")
    result = inputs.s * a + b
    return OutputSchema(result=result)


def abstract_eval(abstract_inputs: Any) -> dict:
    """The result has the shape and dtype of ``a``."""
    a = abstract_inputs.a
    return {"result": ShapeDType(shape=a.shape, dtype=a.dtype)}


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """Return the Jacobian blocks as device memory."""
    eye = torch.eye(len(inputs.a), dtype=torch.float32, device="cuda")
    scales = _partial_scales(inputs)
    return {"result": {name: scales[name] * eye for name in jac_inputs}}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Return the JVP as device memory."""
    scales = _partial_scales(inputs)
    result = sum(
        scales[name] * torch.as_tensor(tangent_vector[name], device="cuda")
        for name in jvp_inputs
    )
    return {"result": result}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Return the VJP as device memory."""
    cotangent = torch.as_tensor(cotangent_vector["result"], device="cuda")
    scales = _partial_scales(inputs)
    return {name: scales[name] * cotangent for name in vjp_inputs}
