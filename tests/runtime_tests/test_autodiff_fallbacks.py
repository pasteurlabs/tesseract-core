# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for autodiff fallback helpers in tesseract_core.runtime.experimental."""

import numpy as np

from tesseract_core.runtime.experimental import (
    jvp_from_jacobian,
    vjp_from_jacobian,
)

# A fixed 2x3 linear map: f(x) = A @ x, Jacobian is A everywhere.
_A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


def _jacobian_fn(inputs, jac_inputs, jac_outputs):
    return {"y": {"x": _A.copy()}}


_inputs = object()  # opaque; jacobian_fn ignores it
_jvp_inputs = {"x"}
_jvp_outputs = {"y"}
_tangent = {"x": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
_cotangent = {"y": np.array([1.0, 0.0], dtype=np.float32)}


def test_jvp_from_jacobian():
    result = jvp_from_jacobian(
        _jacobian_fn, _inputs, _jvp_inputs, _jvp_outputs, _tangent
    )
    np.testing.assert_allclose(result["y"], _A @ _tangent["x"])


def test_vjp_from_jacobian():
    result = vjp_from_jacobian(
        _jacobian_fn, _inputs, _jvp_inputs, _jvp_outputs, _cotangent
    )
    np.testing.assert_allclose(result["x"], _A.T @ _cotangent["y"])
