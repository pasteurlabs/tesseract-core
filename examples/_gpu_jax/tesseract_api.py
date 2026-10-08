# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU Tesseract that returns its results as device (JAX) memory.

Every endpoint returns single-device JAX arrays, which expose
``__cuda_array_interface__``. Served with ``gpu_transport="cuda_ipc"``, the
runtime therefore hands back device-memory IPC handles instead of copying
results to host. This requires a real GPU and is only built/run by the GPU
end-to-end tests (``tests/endtoend_tests/test_serving_gpu.py``).
"""

from typing import Any

import jax.numpy as jnp
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
    a = jnp.asarray(inputs.a)
    b = jnp.asarray(inputs.b)
    result = inputs.s * a + b
    return OutputSchema(result=result)


def abstract_eval(abstract_inputs: Any) -> dict:
    """The result has the shape and dtype of ``a``."""
    a = abstract_inputs.a
    return {"result": ShapeDType(shape=a.shape, dtype=a.dtype)}


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """Return the Jacobian blocks as device memory."""
    eye = jnp.eye(len(inputs.a), dtype=jnp.float32)
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
        scales[name] * jnp.asarray(tangent_vector[name]) for name in jvp_inputs
    )
    return {"result": result}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Return the VJP as device memory."""
    cotangent = jnp.asarray(cotangent_vector["result"])
    scales = _partial_scales(inputs)
    return {name: scales[name] * cotangent for name in vjp_inputs}
