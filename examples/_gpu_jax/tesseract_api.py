# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU Tesseract that returns its result as device (JAX) memory.

JAX exposes its device buffers only through DLPack, never
``__cuda_array_interface__``. Serving this Tesseract with
``gpu_transport="cuda_ipc"`` therefore exercises the DLPack metadata path of the
CUDA IPC export: the runtime reads the device pointer, shape, and dtype off the
DLPack capsule and hands back an IPC handle instead of copying the result to
host. This requires a real GPU and is only built/run by the GPU end-to-end tests
(``tests/endtoend_tests/test_serving_gpu.py``).
"""

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
    """Compute ``s * a + b`` on the GPU and return device memory."""
    import jax

    a = jnp.asarray(inputs.a)
    b = jnp.asarray(inputs.b)
    result = inputs.s * a + b
    # TEMP diagnostic: surface the actual JAX backend/device in CI.
    raise RuntimeError(
        f"JAX_DIAG backend={jax.default_backend()} "
        f"devices={jax.devices()} result_device={result.devices()}"
    )
    return OutputSchema(result=result)
