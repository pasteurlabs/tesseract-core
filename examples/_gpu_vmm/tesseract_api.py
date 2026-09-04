# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU Tesseract whose result is VMM-backed device memory (via JAX/XLA).

Identical math to ``_gpu_cuda_ipc`` (``s * a + b``), but ``apply`` computes on
JAX, so the result buffer comes from XLA's default GPU allocator -- CUDA VMM,
single-segment. The JAX array is adopted into CuPy zero-copy via DLPack (same
device pointer) so it exposes ``__cuda_array_interface__``, which is how the
runtime recognises an on-device result to export; the underlying memory stays
VMM-backed. Serving this with ``output_format="json+cuda_ipc"`` therefore
exercises the VMM export path (``cuMemExportToShareableHandle`` + fd passing)
rather than legacy ``cudaIpcGetMemHandle``. Built/run only by the GPU
end-to-end tests (``tests/endtoend_tests/test_serving_gpu.py``).
"""

import cupy as cp
import jax
import jax.numpy as jnp
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
    """Compute ``s * a + b`` on the GPU (JAX) and return VMM-backed device memory."""
    a = jnp.asarray(inputs.a)
    b = jnp.asarray(inputs.b)
    result = inputs.s * a + b
    jax.block_until_ready(result)
    # Adopt the JAX (VMM-backed) buffer into CuPy zero-copy so it carries
    # __cuda_array_interface__ for the runtime's cuda_ipc encoder. Same pointer,
    # still VMM memory -> exported via the VMM path, not legacy IPC.
    return OutputSchema(result=cp.from_dlpack(result))
