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


# Case 3 - two inputs and two outputs of differing rank, so the probe sweeps
# have to keep one-hot seeds and results matched up across keys and axes:
#   a: (2, 2), b: (3,)  ->  u: (2,), v: (2, 2)
#   u[i]    = sum_j a[i,j]^2 + b[i]
#   v[i,j]  = a[i,j]*b[2] + sin(b[j])
def _f3(a, b):
    return {
        "u": (a**2).sum(axis=1) + b[:2],
        "v": a * b[2] + np.sin(b[:2])[None, :],
    }


def _J3(a, b):
    du_da = np.zeros((2, 2, 2))
    du_db = np.zeros((2, 3))
    dv_da = np.zeros((2, 2, 2, 2))
    dv_db = np.zeros((2, 2, 3))
    for i in range(2):
        du_da[i, i, :] = 2 * a[i, :]
        du_db[i, i] = 1.0
        for j in range(2):
            dv_da[i, j, i, j] = b[2]
            dv_db[i, j, 2] = a[i, j]
            dv_db[i, j, j] += np.cos(b[j])
    return {"u": {"a": du_da, "b": du_db}, "v": {"a": dv_da, "b": dv_db}}


def _make_multi_case():
    """Build the same 8-tuple as _make_case for the multi-key, multi-rank case.

    The endpoints only answer for the keys they are asked about, which is what
    a Tesseract does, so a helper that probes with a seed for one key while
    requesting another is a KeyError rather than a silent wrong answer.
    """
    a = np.array([[0.5, -1.2], [2.0, 0.3]])
    b = np.array([0.7, -0.4, 1.1])
    inputs = {"a": a, "b": b}
    tangent = {
        "a": np.array([[1.0, -0.5], [0.25, 2.0]]),
        "b": np.array([1.0, -1.0, 0.5]),
    }
    cotangent = {"u": np.array([1.0, -2.0]), "v": np.array([[0.5, 1.5], [-1.0, 0.25]])}

    def apply_fn(inputs):
        return _f3(np.asarray(inputs["a"]), np.asarray(inputs["b"]))

    def abstract_eval_fn(inputs):
        return {
            "u": ShapeDType(shape=(2,), dtype="float64"),
            "v": ShapeDType(shape=(2, 2), dtype="float64"),
        }

    def jacobian_fn(inputs, jac_inputs, jac_outputs):
        full = _J3(np.asarray(inputs["a"]), np.asarray(inputs["b"]))
        return {dy: {dx: full[dy][dx] for dx in jac_inputs} for dy in jac_outputs}

    def vjp_fn(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
        full = _J3(np.asarray(inputs["a"]), np.asarray(inputs["b"]))
        out = {}
        for dx in vjp_inputs:
            total = np.zeros(np.shape(inputs[dx]))
            for dy in vjp_outputs:
                ct = np.asarray(cotangent_vector[dy])
                total = total + np.tensordot(ct, full[dy][dx], axes=ct.ndim)
            out[dx] = total
        return out

    def jvp_fn(inputs, jvp_inputs, jvp_outputs, tangent_vector):
        full = _J3(np.asarray(inputs["a"]), np.asarray(inputs["b"]))
        out = {}
        for dy in jvp_outputs:
            total = None
            for dx in jvp_inputs:
                t = np.asarray(tangent_vector[dx])
                term = np.tensordot(full[dy][dx], t, axes=t.ndim)
                total = term if total is None else total + term
            out[dy] = total
        return out

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
    pytest.param(*_make_multi_case(), id="multi-key-multi-rank"),
]


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
def test_jvp_from_jacobian(
    inputs, tangent, cotangent, apply_fn, abstract_eval_fn, jacobian_fn, vjp_fn, jvp_fn
):
    jac_inputs, jac_outputs = set(inputs), set(cotangent)
    result = jvp_from_jacobian(jacobian_fn, inputs, jac_inputs, jac_outputs, tangent)
    expected = jvp_fn(inputs, jac_inputs, jac_outputs, tangent)
    for dy in jac_outputs:
        np.testing.assert_allclose(result[dy], expected[dy])


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
def test_vjp_from_jacobian(
    inputs, tangent, cotangent, apply_fn, abstract_eval_fn, jacobian_fn, vjp_fn, jvp_fn
):
    jac_inputs, jac_outputs = set(inputs), set(cotangent)
    result = vjp_from_jacobian(jacobian_fn, inputs, jac_inputs, jac_outputs, cotangent)
    expected = vjp_fn(inputs, jac_inputs, jac_outputs, cotangent)
    for dx in jac_inputs:
        np.testing.assert_allclose(result[dx], expected[dx])


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
    jac_inputs, jac_outputs = set(inputs), set(cotangent)
    jac = jacobian_from_vjp(vjp_fn, eval_fn, inputs, jac_inputs, jac_outputs)
    expected = jacobian_fn(inputs, jac_inputs, jac_outputs)
    for dy in jac_outputs:
        for dx in jac_inputs:
            np.testing.assert_allclose(jac[dy][dx], expected[dy][dx], rtol=1e-10)


@pytest.mark.parametrize(
    "inputs,tangent,cotangent,apply_fn,abstract_eval_fn,jacobian_fn,vjp_fn,jvp_fn",
    _CASES,
)
def test_jacobian_from_jvp(
    inputs, tangent, cotangent, apply_fn, abstract_eval_fn, jacobian_fn, vjp_fn, jvp_fn
):
    jac_inputs, jac_outputs = set(inputs), set(cotangent)
    jac = jacobian_from_jvp(jvp_fn, inputs, jac_inputs, jac_outputs)
    expected = jacobian_fn(inputs, jac_inputs, jac_outputs)
    for dy in jac_outputs:
        for dx in jac_inputs:
            np.testing.assert_allclose(jac[dy][dx], expected[dy][dx], rtol=1e-10)


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
    jac_inputs, jac_outputs = set(inputs), set(cotangent)
    result = jvp_from_jacobian(jac_fn, inputs, jac_inputs, jac_outputs, tangent)
    expected = jvp_fn(inputs, jac_inputs, jac_outputs, tangent)
    for dy in jac_outputs:
        np.testing.assert_allclose(result[dy], expected[dy], rtol=1e-10)


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
    jac_inputs, jac_outputs = set(inputs), set(cotangent)
    result = vjp_from_jacobian(jac_fn, inputs, jac_inputs, jac_outputs, cotangent)
    expected = vjp_fn(inputs, jac_inputs, jac_outputs, cotangent)
    for dx in jac_inputs:
        np.testing.assert_allclose(result[dx], expected[dx], rtol=1e-10)
