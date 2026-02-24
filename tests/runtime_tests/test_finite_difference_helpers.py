# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the finite_difference_* helper functions."""

import numpy as np
import pytest
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)


class SimpleInputSchema(BaseModel):
    a: Differentiable[Array[(3,), Float64]] = Field(description="Vector a")
    b: Differentiable[Array[(3,), Float64]] = Field(description="Vector b")
    s: Differentiable[Float64] = Field(description="Scalar", default=1.0)


class SimpleOutputSchema(BaseModel):
    result: Differentiable[Array[(3,), Float64]] = Field(description="Result")


def linear_apply(inputs: SimpleInputSchema) -> SimpleOutputSchema:
    """Linear function: result = s * a + b."""
    result = inputs.a * inputs.s + inputs.b
    return SimpleOutputSchema(result=result)


def nonlinear_apply(inputs: SimpleInputSchema) -> SimpleOutputSchema:
    """Non-linear function: result = (s * a + b) / |s * a + b|."""
    result = inputs.a * inputs.s + inputs.b
    norm = np.linalg.norm(result)
    result = result / norm
    return SimpleOutputSchema(result=result)


@pytest.fixture
def simple_inputs():
    return SimpleInputSchema(
        a=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        b=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        s=np.float64(2.0),
    )


class TestFiniteDifferenceJacobian:
    def test_linear_jacobian_central(self, simple_inputs):
        """Test Jacobian computation with central differences on a linear function."""
        jac = finite_difference_jacobian(
            linear_apply,
            simple_inputs,
            jac_inputs={"a", "b", "s"},
            jac_outputs={"result"},
            algorithm="central",
            eps=1e-6,
        )

        # For result = s * a + b:
        # d(result)/d(a) = s * I = 2 * I
        # d(result)/d(b) = I
        # d(result)/d(s) = a
        expected_jac_a = np.eye(3) * 2.0
        expected_jac_b = np.eye(3)
        expected_jac_s = np.array([1.0, 2.0, 3.0])

        assert np.allclose(jac["result"]["a"], expected_jac_a, atol=1e-6)
        assert np.allclose(jac["result"]["b"], expected_jac_b, atol=1e-6)
        assert np.allclose(jac["result"]["s"], expected_jac_s, atol=1e-6)

    def test_linear_jacobian_forward(self, simple_inputs):
        """Test Jacobian computation with forward differences."""
        jac = finite_difference_jacobian(
            linear_apply,
            simple_inputs,
            jac_inputs={"a"},
            jac_outputs={"result"},
            algorithm="forward",
            eps=1e-6,
        )

        expected_jac_a = np.eye(3) * 2.0
        assert np.allclose(jac["result"]["a"], expected_jac_a, atol=1e-5)

    def test_jacobian_stochastic(self, simple_inputs):
        """Test Jacobian computation with stochastic algorithm (SPSA)."""
        jac = finite_difference_jacobian(
            linear_apply,
            simple_inputs,
            jac_inputs={"a"},
            jac_outputs={"result"},
            algorithm="stochastic",
            eps=1e-4,
            num_samples=500,
            seed=42,
        )

        expected_jac_a = np.eye(3) * 2.0
        # Stochastic has higher variance, so use looser tolerance
        assert np.allclose(jac["result"]["a"], expected_jac_a, atol=0.1)

    def test_nonlinear_jacobian(self, simple_inputs):
        """Test Jacobian computation on a non-linear function."""
        jac = finite_difference_jacobian(
            nonlinear_apply,
            simple_inputs,
            jac_inputs={"a", "s"},
            jac_outputs={"result"},
            algorithm="central",
            eps=1e-6,
        )

        # Verify shapes
        assert jac["result"]["a"].shape == (3, 3)
        assert jac["result"]["s"].shape == (3,)

        # Verify against numerical check with different eps
        jac_check = finite_difference_jacobian(
            nonlinear_apply,
            simple_inputs,
            jac_inputs={"a", "s"},
            jac_outputs={"result"},
            algorithm="central",
            eps=1e-5,
        )
        assert np.allclose(jac["result"]["a"], jac_check["result"]["a"], rtol=0.01)

    def test_partial_inputs_outputs(self, simple_inputs):
        """Test computing Jacobian for only a subset of inputs/outputs."""
        jac = finite_difference_jacobian(
            linear_apply,
            simple_inputs,
            jac_inputs={"a"},
            jac_outputs={"result"},
            algorithm="central",
        )

        assert set(jac.keys()) == {"result"}
        assert set(jac["result"].keys()) == {"a"}


class TestFiniteDifferenceJVP:
    def test_jvp_single_direction(self, simple_inputs):
        """Test JVP with a single tangent direction."""
        tangent = {"a": np.array([1.0, 0.0, 0.0], dtype=np.float64)}
        jvp = finite_difference_jvp(
            linear_apply,
            simple_inputs,
            jvp_inputs={"a"},
            jvp_outputs={"result"},
            tangent_vector=tangent,
            algorithm="central",
            eps=1e-6,
        )

        # JVP with tangent [1,0,0] on 'a' should give [s, 0, 0] = [2, 0, 0]
        expected = np.array([2.0, 0.0, 0.0])
        assert np.allclose(jvp["result"], expected, atol=1e-6)

    def test_jvp_multiple_inputs(self, simple_inputs):
        """Test JVP with multiple inputs in tangent vector."""
        tangent = {
            "a": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "b": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        }
        jvp = finite_difference_jvp(
            linear_apply,
            simple_inputs,
            jvp_inputs={"a", "b"},
            jvp_outputs={"result"},
            tangent_vector=tangent,
            algorithm="central",
            eps=1e-6,
        )

        # JVP = s * tangent_a + tangent_b = 2*[1,1,1] + [1,0,0] = [3, 2, 2]
        expected = np.array([3.0, 2.0, 2.0])
        assert np.allclose(jvp["result"], expected, atol=1e-6)

    def test_jvp_forward_algorithm(self, simple_inputs):
        """Test JVP with forward differences."""
        tangent = {"a": np.array([0.0, 1.0, 0.0], dtype=np.float64)}
        jvp = finite_difference_jvp(
            linear_apply,
            simple_inputs,
            jvp_inputs={"a"},
            jvp_outputs={"result"},
            tangent_vector=tangent,
            algorithm="forward",
            eps=1e-6,
        )

        expected = np.array([0.0, 2.0, 0.0])
        assert np.allclose(jvp["result"], expected, atol=1e-5)


class TestFiniteDifferenceVJP:
    def test_vjp_single_cotangent(self, simple_inputs):
        """Test VJP with a cotangent vector."""
        cotangent = {"result": np.array([1.0, 1.0, 1.0], dtype=np.float64)}
        vjp = finite_difference_vjp(
            linear_apply,
            simple_inputs,
            vjp_inputs={"a", "s"},
            vjp_outputs={"result"},
            cotangent_vector=cotangent,
            algorithm="central",
            eps=1e-6,
        )

        # VJP for a: cotangent @ d(result)/d(a) = [1,1,1] @ (2*I) = [2, 2, 2]
        # VJP for s: cotangent @ d(result)/d(s) = [1,1,1] @ a = 1+2+3 = 6
        expected_vjp_a = np.array([2.0, 2.0, 2.0])
        expected_vjp_s = 6.0

        assert np.allclose(vjp["a"], expected_vjp_a, atol=1e-6)
        assert np.allclose(vjp["s"], expected_vjp_s, atol=1e-6)

    def test_vjp_partial_outputs(self, simple_inputs):
        """Test VJP with specific output selection."""
        cotangent = {"result": np.array([1.0, 0.0, 0.0], dtype=np.float64)}
        vjp = finite_difference_vjp(
            linear_apply,
            simple_inputs,
            vjp_inputs={"b"},
            vjp_outputs={"result"},
            cotangent_vector=cotangent,
            algorithm="central",
            eps=1e-6,
        )

        # VJP for b: cotangent @ d(result)/d(b) = [1,0,0] @ I = [1, 0, 0]
        expected_vjp_b = np.array([1.0, 0.0, 0.0])
        assert np.allclose(vjp["b"], expected_vjp_b, atol=1e-6)


class TestNestedSchema:
    """Test with nested Pydantic schemas."""

    def test_nested_schema_jacobian(self):
        class NestedInput(BaseModel):
            x: Differentiable[Array[(2,), Float64]]

        class NestedInputSchema(BaseModel):
            inner: NestedInput
            scale: Differentiable[Float64] = 1.0

        class NestedOutputSchema(BaseModel):
            y: Differentiable[Array[(2,), Float64]]

        def nested_apply(inputs: NestedInputSchema) -> NestedOutputSchema:
            return NestedOutputSchema(y=inputs.inner.x * inputs.scale)

        inputs = NestedInputSchema(
            inner=NestedInput(x=np.array([1.0, 2.0])),
            scale=np.float64(3.0),
        )

        jac = finite_difference_jacobian(
            nested_apply,
            inputs,
            jac_inputs={"inner.x", "scale"},
            jac_outputs={"y"},
            algorithm="central",
            eps=1e-6,
        )

        # d(y)/d(inner.x) = scale * I = 3 * I
        expected_jac_x = np.eye(2) * 3.0
        assert np.allclose(jac["y"]["inner.x"], expected_jac_x, atol=1e-6)

        # d(y)/d(scale) = inner.x = [1, 2]
        expected_jac_scale = np.array([1.0, 2.0])
        assert np.allclose(jac["y"]["scale"], expected_jac_scale, atol=1e-6)
