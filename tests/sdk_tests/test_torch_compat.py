# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from tesseract_core import Tesseract
from tesseract_core.torch_compat import apply_tesseract

UNIVARIATE_API = (
    Path(__file__).parent.parent.parent / "examples" / "univariate" / "tesseract_api.py"
)

MESHSTATS_API = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "meshstats_finitediff"
    / "tesseract_api.py"
)


@pytest.fixture(scope="module")
def univariate():
    return Tesseract.from_tesseract_api(UNIVARIATE_API)


@pytest.fixture(scope="module")
def meshstats():
    return Tesseract.from_tesseract_api(MESHSTATS_API)


class TestFlatSchema:
    """Tests with the univariate Tesseract (flat scalar input/output schema).

    univariate computes the Rosenbrock function: result = (a - x)^2 + b*(y - x^2)^2
    Default: a=1, b=100. Differentiable inputs: x, y.
    """

    def test_forward_pass(self, univariate):
        x = torch.tensor(1.0)
        y = torch.tensor(1.0)

        result = apply_tesseract(univariate, {"x": x, "y": y})

        assert "result" in result
        # (1-1)^2 + 100*(1-1)^2 = 0
        assert torch.allclose(result["result"], torch.tensor(0.0))

    def test_backward_pass(self, univariate):
        x = torch.tensor(0.0, requires_grad=True)
        y = torch.tensor(0.0, requires_grad=True)

        result = apply_tesseract(univariate, {"x": x, "y": y})
        result["result"].backward()

        # f = (1-x)^2 + 100*(y-x^2)^2 at (0,0): f=1
        # df/dx = -2(1-x) - 400*x*(y-x^2) = -2 at (0,0)
        # df/dy = 200*(y-x^2) = 0 at (0,0)
        assert torch.allclose(x.grad, torch.tensor(-2.0))
        assert torch.allclose(y.grad, torch.tensor(0.0))

    def test_autograd_grad(self, univariate):
        x = torch.tensor(0.0, requires_grad=True)
        y = torch.tensor(0.0, requires_grad=True)

        result = apply_tesseract(univariate, {"x": x, "y": y})

        (grad_x,) = torch.autograd.grad(result["result"], x, retain_graph=True)
        (grad_y,) = torch.autograd.grad(result["result"], y)

        assert torch.allclose(grad_x, torch.tensor(-2.0))
        assert torch.allclose(grad_y, torch.tensor(0.0))

    def test_forward_mode_jvp(self, univariate):
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        tangent_x = torch.tensor(1.0)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, tangent_x)
            result = apply_tesseract(univariate, {"x": x_dual, "y": y})
            primal, tangent_out = fwAD.unpack_dual(result["result"])

        assert torch.allclose(primal, torch.tensor(1.0))
        # tangent = df/dx * tangent_x = -2 * 1 = -2
        assert torch.allclose(tangent_out, torch.tensor(-2.0))

    def test_gradients_match_native_pytorch(self, univariate):
        """Verify gradients match PyTorch's own autodiff for the same math."""
        x = torch.tensor(3.0, requires_grad=True)
        y = torch.tensor(5.0, requires_grad=True)

        result = apply_tesseract(univariate, {"x": x, "y": y})
        result["result"].backward()

        # Reference: native PyTorch
        x_ref = torch.tensor(3.0, requires_grad=True)
        y_ref = torch.tensor(5.0, requires_grad=True)
        a, b = 1.0, 100.0
        y_native = (a - x_ref) ** 2 + b * (y_ref - x_ref**2) ** 2
        y_native.backward()

        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(y.grad, y_ref.grad)


class TestNestedSchema:
    """Tests with meshstats_finitediff (nested input/output schema with VJP/JVP)."""

    def _make_inputs(self, requires_grad: bool = False):
        points = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        return {
            "mesh": {
                "n_points": 3,
                "n_cells": 1,
                "points": points,
                "num_points_per_cell": np.array([3.0], dtype=np.float64),
                "cell_connectivity": np.array([0, 1, 2], dtype=np.int32),
                "cell_data": {},
                "point_data": {},
            }
        }, points

    def test_forward_pass(self, meshstats):
        inputs, _ = self._make_inputs()
        result = apply_tesseract(meshstats, inputs)

        assert "statistics" in result
        stats = result["statistics"]
        assert "barycenter" in stats
        assert "first_point_coordinates" in stats

        assert torch.allclose(
            stats["first_point_coordinates"],
            torch.tensor([1.0, 2.0, 3.0]),
        )
        assert torch.allclose(
            stats["barycenter"],
            torch.tensor([4.0, 5.0, 6.0]),
        )

    def test_output_types(self, meshstats):
        inputs, _ = self._make_inputs()
        result = apply_tesseract(meshstats, inputs)

        assert isinstance(result["statistics"]["barycenter"], torch.Tensor)
        assert isinstance(result["statistics"]["first_point_coordinates"], torch.Tensor)

    def test_backward_pass(self, meshstats):
        inputs, points = self._make_inputs(requires_grad=True)
        result = apply_tesseract(meshstats, inputs)

        # barycenter = mean(points, axis=0), so d(sum(barycenter))/d(points) = 1/n
        result["statistics"]["barycenter"].sum().backward()

        n = 3
        expected_grad = torch.full_like(points, 1.0 / n)
        assert torch.allclose(points.grad, expected_grad, atol=1e-4)

    def test_forward_mode_jvp(self, meshstats):
        points = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float32,
        )
        # Tangent: perturb only the first point's x-coordinate
        tangent_points = torch.zeros_like(points)
        tangent_points[0, 0] = 1.0

        with fwAD.dual_level():
            points_dual = fwAD.make_dual(points, tangent_points)
            result = apply_tesseract(
                meshstats,
                {
                    "mesh": {
                        "n_points": 3,
                        "n_cells": 1,
                        "points": points_dual,
                        "num_points_per_cell": np.array([3.0], dtype=np.float64),
                        "cell_connectivity": np.array([0, 1, 2], dtype=np.int32),
                        "cell_data": {},
                        "point_data": {},
                    },
                },
            )
            _, tangent_bary = fwAD.unpack_dual(result["statistics"]["barycenter"])
            _, tangent_fpc = fwAD.unpack_dual(
                result["statistics"]["first_point_coordinates"]
            )

        assert tangent_bary is not None
        assert tangent_fpc is not None
        # barycenter = mean(points, axis=0) -> tangent = [1/3, 0, 0]
        assert torch.allclose(
            tangent_bary, torch.tensor([1.0 / 3, 0.0, 0.0]), atol=1e-4
        )
        # first_point_coordinates = points[0] -> tangent = [1, 0, 0]
        assert torch.allclose(tangent_fpc, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
