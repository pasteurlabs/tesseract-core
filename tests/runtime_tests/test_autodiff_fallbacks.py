# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for autodiff fallback helpers in tesseract_core.runtime.experimental.

Each test fixture defines a mock Tesseract API module that explicitly calls the
experimental helpers to derive missing autodiff endpoints — mirroring how a user
would opt in to this functionality inside their own tesseract_api.py.
"""

from types import ModuleType

import numpy as np
import pytest
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.core import create_endpoints
from tesseract_core.runtime.experimental import (
    jacobian_from_jvp,
    jacobian_from_vjp,
    jvp_from_jacobian,
    vjp_from_jacobian,
)

# A fixed 2×3 linear map: f(x) = A @ x, Jacobian is A everywhere.
_A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

_linear_input = {"x": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
_linear_tangent = {"x": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
_linear_cotangent = {"y": np.array([1.0, 0.0], dtype=np.float32)}


def _find_endpoint(endpoint_list, endpoint_name):
    for endpoint in endpoint_list:
        if endpoint.__name__ == endpoint_name:
            input_schema = endpoint.__annotations__.get("payload", None)
            output_schema = endpoint.__annotations__.get("return", None)
            return endpoint, input_schema, output_schema
    raise ValueError(f"Endpoint {endpoint_name} not found.")


# ---------------------------------------------------------------------------
# Fixture: jacobian defined analytically; jvp and vjp use experimental helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def module_jac_with_derived_jvp_vjp():
    """Module defines jacobian analytically and derives jvp/vjp via experimental helpers."""

    class _InputSchema(BaseModel):
        x: Differentiable[Array[(3,), Float32]]

    class _OutputSchema(BaseModel):
        y: Differentiable[Array[(2,), Float32]]

    class LinearModule(ModuleType):
        @property
        def InputSchema(self):
            return _InputSchema

        @property
        def OutputSchema(self):
            return _OutputSchema

        def apply(self, inputs):
            return _OutputSchema(y=_A @ np.asarray(inputs.x))

        def jacobian(self, inputs, jac_inputs, jac_outputs):
            return {"y": {"x": _A.copy()}}

        def jacobian_vector_product(self, inputs, jvp_inputs, jvp_outputs, tangent_vector):
            return jvp_from_jacobian(self.jacobian, inputs, jvp_inputs, jvp_outputs, tangent_vector)

        def vector_jacobian_product(self, inputs, vjp_inputs, vjp_outputs, cotangent_vector):
            return vjp_from_jacobian(self.jacobian, inputs, vjp_inputs, vjp_outputs, cotangent_vector)

    return LinearModule("LinearModule")


def test_jvp_via_experimental_helper(module_jac_with_derived_jvp_vjp):
    endpoints = create_endpoints(module_jac_with_derived_jvp_vjp)
    endpoint_func, EndpointSchema, _ = _find_endpoint(endpoints, "jacobian_vector_product")

    payload = EndpointSchema.model_validate(
        {
            "inputs": _linear_input,
            "jvp_inputs": {"x"},
            "jvp_outputs": {"y"},
            "tangent_vector": _linear_tangent,
        }
    )
    result = endpoint_func(payload)
    expected = _A @ _linear_tangent["x"]
    np.testing.assert_allclose(np.asarray(result.root["y"]), expected)


def test_vjp_via_experimental_helper(module_jac_with_derived_jvp_vjp):
    endpoints = create_endpoints(module_jac_with_derived_jvp_vjp)
    endpoint_func, EndpointSchema, _ = _find_endpoint(endpoints, "vector_jacobian_product")

    payload = EndpointSchema.model_validate(
        {
            "inputs": _linear_input,
            "vjp_inputs": {"x"},
            "vjp_outputs": {"y"},
            "cotangent_vector": _linear_cotangent,
        }
    )
    result = endpoint_func(payload)
    expected = _A.T @ _linear_cotangent["y"]
    np.testing.assert_allclose(np.asarray(result.root["x"]), expected)


# ---------------------------------------------------------------------------
# Fixture: jvp defined analytically; jacobian derived via experimental helper
# ---------------------------------------------------------------------------


@pytest.fixture
def module_jvp_with_derived_jacobian():
    """Module defines jvp analytically and derives jacobian via experimental helper."""

    class _InputSchema(BaseModel):
        x: Differentiable[Array[(3,), Float32]]

    class _OutputSchema(BaseModel):
        y: Differentiable[Array[(2,), Float32]]

    class LinearModule(ModuleType):
        @property
        def InputSchema(self):
            return _InputSchema

        @property
        def OutputSchema(self):
            return _OutputSchema

        def apply(self, inputs):
            return _OutputSchema(y=_A @ np.asarray(inputs.x))

        def jacobian_vector_product(self, inputs, jvp_inputs, jvp_outputs, tangent_vector):
            t = np.asarray(tangent_vector["x"])
            return {"y": _A @ t}

        def jacobian(self, inputs, jac_inputs, jac_outputs):
            return jacobian_from_jvp(
                self.jacobian_vector_product, self.apply, inputs, jac_inputs, jac_outputs
            )

    return LinearModule("LinearModule")


def test_jacobian_derived_from_jvp_via_experimental_helper(module_jvp_with_derived_jacobian):
    endpoints = create_endpoints(module_jvp_with_derived_jacobian)
    endpoint_func, EndpointSchema, _ = _find_endpoint(endpoints, "jacobian")

    payload = EndpointSchema.model_validate(
        {
            "inputs": _linear_input,
            "jac_inputs": {"x"},
            "jac_outputs": {"y"},
        }
    )
    result = endpoint_func(payload)
    np.testing.assert_allclose(np.asarray(result.root["y"]["x"]), _A, atol=1e-6)


# ---------------------------------------------------------------------------
# Fixture: vjp defined analytically; jacobian derived via experimental helper
# ---------------------------------------------------------------------------


@pytest.fixture
def module_vjp_with_derived_jacobian():
    """Module defines vjp analytically and derives jacobian via experimental helper."""

    class _InputSchema(BaseModel):
        x: Differentiable[Array[(3,), Float32]]

    class _OutputSchema(BaseModel):
        y: Differentiable[Array[(2,), Float32]]

    class LinearModule(ModuleType):
        @property
        def InputSchema(self):
            return _InputSchema

        @property
        def OutputSchema(self):
            return _OutputSchema

        def apply(self, inputs):
            return _OutputSchema(y=_A @ np.asarray(inputs.x))

        def vector_jacobian_product(self, inputs, vjp_inputs, vjp_outputs, cotangent_vector):
            v = np.asarray(cotangent_vector["y"])
            return {"x": _A.T @ v}

        def jacobian(self, inputs, jac_inputs, jac_outputs):
            return jacobian_from_vjp(
                self.vector_jacobian_product, self.apply, inputs, jac_inputs, jac_outputs
            )

    return LinearModule("LinearModule")


def test_jacobian_derived_from_vjp_via_experimental_helper(module_vjp_with_derived_jacobian):
    endpoints = create_endpoints(module_vjp_with_derived_jacobian)
    endpoint_func, EndpointSchema, _ = _find_endpoint(endpoints, "jacobian")

    payload = EndpointSchema.model_validate(
        {
            "inputs": _linear_input,
            "jac_inputs": {"x"},
            "jac_outputs": {"y"},
        }
    )
    result = endpoint_func(payload)
    np.testing.assert_allclose(np.asarray(result.root["y"]["x"]), _A, atol=1e-6)
