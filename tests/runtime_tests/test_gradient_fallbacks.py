# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gradient fallback helpers in tesseract_core.runtime.experimental."""

import numpy as np
import pytest

from tesseract_core.runtime import ShapeDType
from tesseract_core.runtime.experimental import (
    jacobian_from_jvp,
    jacobian_from_vjp,
    jvp_from_jacobian,
    vjp_from_jacobian,
)


def _make_case(x_val, tangent_val, cotangent_val, f_fn, J_fn):
    """Build a (inputs, tangent, cotangent, apply, abstract_eval, jacobian, vjp, jvp) tuple."""
    out_shape = f_fn(x_val).shape
    inputs = {"x": x_val}
    tangent = {"x": tangent_val}
    cotangent = {"y": cotangent_val}

    def apply_fn(inputs):
        return {"y": f_fn(inputs["x"])}

    def abstract_eval_fn(inputs):
        return {"y": ShapeDType(shape=out_shape, dtype="float64")}

    def jacobian_fn(inputs, jac_inputs, jac_outputs):
        return {"y": {"x": J_fn(inputs["x"])}}

    def vjp_fn(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
        return {"x": J_fn(inputs["x"]).T @ cotangent_vector["y"]}

    def jvp_fn(inputs, jvp_inputs, jvp_outputs, tangent_vector):
        return {"y": J_fn(inputs["x"]) @ tangent_vector["x"]}

    return (
        inputs,
        tangent,
        cotangent,
        apply_fn,
        abstract_eval_fn,
        jacobian_fn,
        vjp_fn,
        jvp_fn,
    )


# Case 1 - wide Jacobian: f: R^4 -> R^3
#   f(x) = [x0*x1 + x2^2,  exp(x0) - x1*x2,  x0^2*x3 + x1]
#   J(x) = [[x1,       x0,   2*x2,  0   ],
#            [exp(x0), -x2,  -x1,   0   ],
#            [2*x0*x3,  1,    0,    x0^2]]
def _f1(x):
    return np.array(
        [x[0] * x[1] + x[2] ** 2, np.exp(x[0]) - x[1] * x[2], x[0] ** 2 * x[3] + x[1]]
    )


def _J1(x):
    return np.array(
        [
            [x[1], x[0], 2 * x[2], 0.0],
            [np.exp(x[0]), -x[2], -x[1], 0.0],
            [2 * x[0] * x[3], 1.0, 0.0, x[0] ** 2],
        ]
    )


# Case 2 - tall Jacobian: g: R^2 -> R^4
#   g(x) = [sin(x0)*x1,  x0^2 + cos(x1),  exp(x0*x1),  x0 - x1^2]
#   J(x) = [[cos(x0)*x1,      sin(x0)       ],
#            [2*x0,           -sin(x1)       ],
#            [x1*exp(x0*x1),   x0*exp(x0*x1)],
#            [1,              -2*x1          ]]
def _f2(x):
    return np.array(
        [
            np.sin(x[0]) * x[1],
            x[0] ** 2 + np.cos(x[1]),
            np.exp(x[0] * x[1]),
            x[0] - x[1] ** 2,
        ]
    )


def _J2(x):
    e = np.exp(x[0] * x[1])
    return np.array(
        [
            [np.cos(x[0]) * x[1], np.sin(x[0])],
            [2 * x[0], -np.sin(x[1])],
            [x[1] * e, x[0] * e],
            [1.0, -2 * x[1]],
        ]
    )


_CASES = [
    pytest.param(
        *_make_case(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, -1.0, 0.5, 2.0]),
            np.array([1.0, -1.0, 0.5]),
            _f1,
            _J1,
        ),
        id="wide-R4-R3",
    ),
    pytest.param(
        *_make_case(
            np.array([0.5, 1.2]),
            np.array([1.0, -0.5]),
            np.array([1.0, -1.0, 0.5, 0.3]),
            _f2,
            _J2,
        ),
        id="tall-R2-R4",
    ),
]


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
def test_jvp_from_jacobian(
    inputs, tangent, cotangent, apply_fn, abstract_eval_fn, jacobian_fn, vjp_fn, jvp_fn
):
    result = jvp_from_jacobian(jacobian_fn, inputs, {"x"}, {"y"}, tangent)
    np.testing.assert_allclose(result["y"], jvp_fn(inputs, {"x"}, {"y"}, tangent)["y"])


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
def test_vjp_from_jacobian(
    inputs, tangent, cotangent, apply_fn, abstract_eval_fn, jacobian_fn, vjp_fn, jvp_fn
):
    result = vjp_from_jacobian(jacobian_fn, inputs, {"x"}, {"y"}, cotangent)
    np.testing.assert_allclose(
        result["x"], vjp_fn(inputs, {"x"}, {"y"}, cotangent)["x"]
    )


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
@pytest.mark.parametrize("eval_fn_name", ["apply_fn", "abstract_eval_fn"])
def test_jacobian_from_vjp(
    inputs,
    tangent,
    cotangent,
    apply_fn,
    abstract_eval_fn,
    jacobian_fn,
    vjp_fn,
    jvp_fn,
    eval_fn_name,
):
    eval_fn = apply_fn if eval_fn_name == "apply_fn" else abstract_eval_fn
    jac = jacobian_from_vjp(vjp_fn, eval_fn, inputs, {"x"}, {"y"})
    np.testing.assert_allclose(
        jac["y"]["x"], jacobian_fn(inputs, {"x"}, {"y"})["y"]["x"], rtol=1e-10
    )


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
def test_jacobian_from_jvp(
    inputs, tangent, cotangent, apply_fn, abstract_eval_fn, jacobian_fn, vjp_fn, jvp_fn
):
    jac = jacobian_from_jvp(jvp_fn, inputs, {"x"}, {"y"})
    np.testing.assert_allclose(
        jac["y"]["x"], jacobian_fn(inputs, {"x"}, {"y"})["y"]["x"], rtol=1e-10
    )


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
@pytest.mark.parametrize("eval_fn_name", ["apply_fn", "abstract_eval_fn"])
def test_vjp_to_jvp_via_jacobian(
    inputs,
    tangent,
    cotangent,
    apply_fn,
    abstract_eval_fn,
    jacobian_fn,
    vjp_fn,
    jvp_fn,
    eval_fn_name,
):
    # VJP -> jacobian_from_vjp -> jvp_from_jacobian should agree with a direct JVP call.
    eval_fn = apply_fn if eval_fn_name == "apply_fn" else abstract_eval_fn
    jac_fn = lambda inputs, jac_inputs, jac_outputs: jacobian_from_vjp(
        vjp_fn, eval_fn, inputs, jac_inputs, jac_outputs
    )
    result = jvp_from_jacobian(jac_fn, inputs, {"x"}, {"y"}, tangent)
    np.testing.assert_allclose(
        result["y"], jvp_fn(inputs, {"x"}, {"y"}, tangent)["y"], rtol=1e-10
    )


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
def test_jvp_to_vjp_via_jacobian(
    inputs, tangent, cotangent, apply_fn, abstract_eval_fn, jacobian_fn, vjp_fn, jvp_fn
):
    # JVP -> jacobian_from_jvp -> vjp_from_jacobian should agree with a direct VJP call.
    jac_fn = lambda inputs, jac_inputs, jac_outputs: jacobian_from_jvp(
        jvp_fn, inputs, jac_inputs, jac_outputs
    )
    result = vjp_from_jacobian(jac_fn, inputs, {"x"}, {"y"}, cotangent)
    np.testing.assert_allclose(
        result["x"], vjp_fn(inputs, {"x"}, {"y"}, cotangent)["x"], rtol=1e-10
    )
