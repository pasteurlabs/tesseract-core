# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU Tesseract that returns its result as device (PyTorch) memory.

Because ``apply`` returns a CUDA tensor that exposes
``__cuda_array_interface__``, serving this Tesseract with
``gpu_transport="cuda_ipc"`` exercises the full CUDA IPC export path: the runtime
hands back a device-memory IPC handle instead of copying the result to host. This
requires a real GPU and is only built/run by the GPU end-to-end tests
(``tests/endtoend_tests/test_serving_gpu.py``).
"""

import torch
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float32


class InputSchema(BaseModel):
    a: Array[(None,), Float32] = Field(description="An arbitrary vector.")
    b: Array[(None,), Float32] = Field(
        description="An arbitrary vector, same shape as a."
    )
    s: float = Field(description="A scalar.", default=3.0)


class OutputSchema(BaseModel):
    result: Array[(None,), Float32] = Field(description="Vector s * a + b, on GPU.")


def apply(inputs: InputSchema) -> OutputSchema:
    """Compute ``s * a + b`` on the GPU and return device memory."""
    a = torch.as_tensor(inputs.a, device="cuda")
    b = torch.as_tensor(inputs.b, device="cuda")
    result = inputs.s * a + b
    return OutputSchema(result=result)
