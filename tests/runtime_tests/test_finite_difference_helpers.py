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

        # Verify against analytic Jacobian
        expected_jac_a = np.eye(3) * 2.0
        assert np.allclose(jac["result"]["a"], expected_jac_a, atol=0.1)

    def test_jacobian_stochastic_convergence(self):
        """Test that stochastic Jacobian error decreases with more samples.

        Uses a higher-dimensional input so that the stochastic algorithm doesn't
        fall back to elementwise (which would give exact results regardless of
        num_samples).
        """

        class HighDimInput(BaseModel):
            x: Differentiable[Array[(20,), Float64]]

        class HighDimOutput(BaseModel):
            y: Differentiable[Array[(20,), Float64]]

        def square_apply(inputs: HighDimInput) -> HighDimOutput:
            return HighDimOutput(y=inputs.x**2)

        inputs = HighDimInput(x=np.arange(1.0, 21.0, dtype=np.float64))
        # Analytic Jacobian: d(x^2)/dx = diag(2*x)
        expected_jac = np.diag(2.0 * np.arange(1.0, 21.0))

        sample_counts = [5, 50, 500]
        errors = []

        for n in sample_counts:
            jac = finite_difference_jacobian(
                square_apply,
                inputs,
                jac_inputs={"x"},
                jac_outputs={"y"},
                algorithm="stochastic",
                eps=1e-4,
                num_samples=n,
                seed=42,
            )
            error = np.max(np.abs(jac["y"]["x"] - expected_jac))
            errors.append(error)

        # Error should decrease as num_samples increases
        assert errors[-1] < errors[0], (
            f"Error did not decrease with more samples: {list(zip(sample_counts, errors, strict=True))}"
        )

    def test_nonlinear_jacobian(self, simple_inputs):
        """Test Jacobian computation on a non-linear function against analytic Jacobian."""
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

        # Compute analytic Jacobian for result = v / |v| where v = s*a + b
        a = simple_inputs.a
        b = simple_inputs.b
        s = simple_inputs.s
        v = s * a + b
        norm_v = np.linalg.norm(v)
        # d(v/|v|)/dv = (I - v v^T / |v|^2) / |v|
        identity = np.eye(3)
        analytic_jac_v = (identity - np.outer(v, v) / norm_v**2) / norm_v
        # d(result)/d(a) = d(result)/d(v) * d(v)/d(a) = analytic_jac_v * s
        analytic_jac_a = analytic_jac_v * s
        # d(result)/d(s) = d(result)/d(v) * d(v)/d(s) = analytic_jac_v @ a
        analytic_jac_s = analytic_jac_v @ a

        assert np.allclose(jac["result"]["a"], analytic_jac_a, atol=1e-5)
        assert np.allclose(jac["result"]["s"], analytic_jac_s, atol=1e-5)

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


class ScaleSpreadInputSchema(BaseModel):
    """Inputs whose nominal values sit nine orders of magnitude apart."""

    big: Differentiable[Float64] = Field(description="Nominally 5.0")
    small: Differentiable[Float64] = Field(description="Nominally 2.6e-5")


class ScaleSpreadOutputSchema(BaseModel):
    from_big: Differentiable[Float64] = Field(description="big ** 3")
    from_small: Differentiable[Float64] = Field(description="sqrt(small)")


def scale_spread_apply(inputs: ScaleSpreadInputSchema) -> ScaleSpreadOutputSchema:
    """Depends on each input through a function only defined for positive values."""
    return ScaleSpreadOutputSchema(
        from_big=inputs.big**3, from_small=np.sqrt(inputs.small)
    )


BIG = 5.0
SMALL = 2.6e-5


@pytest.fixture
def scale_spread_inputs():
    return ScaleSpreadInputSchema(big=np.float64(BIG), small=np.float64(SMALL))


class TestPerInputEps:
    """A step size chosen for one input can be wrong for another.

    ``small`` is 2.6e-5, so the default ``eps`` of 1e-4 does not perturb it, it
    replaces it: ``small - eps`` is -7.4e-5 and ``sqrt`` of that is undefined.
    ``big`` at 5.0 is differenced accurately by that same step.
    """

    def test_scalar_eps_cannot_serve_both_scales(self, scale_spread_inputs):
        """The step reaches outside the domain, so the derivative comes back NaN."""
        with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
            jac = finite_difference_jacobian(
                scale_spread_apply,
                scale_spread_inputs,
                jac_inputs={"big", "small"},
                jac_outputs={"from_big", "from_small"},
                eps=1e-4,
            )
        assert np.allclose(jac["from_big"]["big"], 3 * BIG**2)
        assert np.isnan(jac["from_small"]["small"])

    def test_per_input_eps_recovers_both_derivatives(self, scale_spread_inputs):
        jac = finite_difference_jacobian(
            scale_spread_apply,
            scale_spread_inputs,
            jac_inputs={"big", "small"},
            jac_outputs={"from_big", "from_small"},
            eps={"big": 1e-4, "small": 1e-9},
        )
        assert np.allclose(jac["from_big"]["big"], 3 * BIG**2, rtol=1e-6)
        assert np.allclose(jac["from_small"]["small"], 0.5 / np.sqrt(SMALL), rtol=1e-6)

    @pytest.mark.parametrize("algorithm", ["central", "forward", "stochastic"])
    def test_scalar_and_uniform_dict_agree_exactly(self, simple_inputs, algorithm):
        """A dict of equal values must be the scalar path, bit for bit.

        This is what keeps the default cost of the stochastic and JVP algorithms
        unchanged: one step size means one perturbation group.
        """
        kwargs = {
            "jac_inputs": {"a", "b", "s"},
            "jac_outputs": {"result"},
            "algorithm": algorithm,
            "num_samples": 4,
            "seed": 7,
        }
        from_scalar = finite_difference_jacobian(
            nonlinear_apply, simple_inputs, eps=1e-6, **kwargs
        )
        from_dict = finite_difference_jacobian(
            nonlinear_apply,
            simple_inputs,
            eps={"a": 1e-6, "b": 1e-6, "s": 1e-6},
            **kwargs,
        )
        for in_path in ("a", "b", "s"):
            np.testing.assert_array_equal(
                from_scalar["result"][in_path], from_dict["result"][in_path]
            )

    def test_jvp_sums_over_step_size_groups(self, scale_spread_inputs):
        """A JVP is linear in the tangent, so disjoint groups add.

        Two distinct step sizes means two directional derivatives rather than
        one, and their sum has to be the whole JVP.
        """
        jvp = finite_difference_jvp(
            scale_spread_apply,
            scale_spread_inputs,
            jvp_inputs={"big", "small"},
            jvp_outputs={"from_big", "from_small"},
            tangent_vector={"big": np.float64(1.0), "small": np.float64(1.0)},
            eps={"big": 1e-4, "small": 1e-9},
        )
        assert np.allclose(jvp["from_big"], 3 * BIG**2, rtol=1e-6)
        assert np.allclose(jvp["from_small"], 0.5 / np.sqrt(SMALL), rtol=1e-6)

    def test_vjp_uses_each_inputs_own_step(self, scale_spread_inputs):
        vjp = finite_difference_vjp(
            scale_spread_apply,
            scale_spread_inputs,
            vjp_inputs={"big", "small"},
            vjp_outputs={"from_big", "from_small"},
            cotangent_vector={
                "from_big": np.float64(1.0),
                "from_small": np.float64(1.0),
            },
            eps={"big": 1e-4, "small": 1e-9},
        )
        assert np.allclose(vjp["big"], 3 * BIG**2, rtol=1e-6)
        assert np.allclose(vjp["small"], 0.5 / np.sqrt(SMALL), rtol=1e-6)

    @pytest.mark.parametrize(
        "eps", [{"big": 1e-4}, {"big": 1e-4, "small": 1e-9, "medium": 1e-6}]
    )
    def test_dict_eps_must_match_the_differentiated_paths(
        self, scale_spread_inputs, eps
    ):
        """A path that is misspelled or forgotten silently changes the step size.

        Nothing downstream can detect that, so it is rejected up front.
        """
        with pytest.raises(KeyError):
            finite_difference_jacobian(
                scale_spread_apply,
                scale_spread_inputs,
                jac_inputs={"big", "small"},
                jac_outputs={"from_big", "from_small"},
                eps=eps,
            )
