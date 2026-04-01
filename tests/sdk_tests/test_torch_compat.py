# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from tesseract_core import Tesseract
from tesseract_core.torch_compat import apply_tesseract

QUADRATIC_API = (
    Path(__file__).parent.parent.parent
    / "demo"
    / "torch-primitive"
    / "quadratic_tesseract"
    / "tesseract_api.py"
)

MESHSTATS_API = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "meshstats_finitediff"
    / "tesseract_api.py"
)


@pytest.fixture(scope="module")
def quadratic():
    return Tesseract.from_tesseract_api(QUADRATIC_API)


@pytest.fixture(scope="module")
def meshstats():
    return Tesseract.from_tesseract_api(MESHSTATS_API)


class TestFlatSchema:
    """Tests with the quadratic Tesseract (flat input/output schema)."""

    def test_forward_pass(self, quadratic):
        x = torch.tensor([1.0, 2.0, 3.0])
        A = torch.eye(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float32)

        result = apply_tesseract(quadratic, {"x": x, "A": A, "b": b})

        assert "y" in result
        assert torch.allclose(result["y"], torch.tensor([1.0, 4.0, 9.0]))

    def test_backward_pass(self, quadratic):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        A = torch.eye(3, dtype=torch.float32, requires_grad=True)
        b = torch.zeros(3, dtype=torch.float32, requires_grad=True)

        result = apply_tesseract(quadratic, {"x": x, "A": A, "b": b})
        result["y"].sum().backward()

        assert torch.allclose(x.grad, torch.tensor([2.0, 4.0, 6.0]))
        assert torch.allclose(b.grad, torch.ones(3))
        # dy/dA = outer(ones, x^2) since cotangent is ones
        expected_A_grad = torch.tensor([[1.0, 4.0, 9.0]] * 3)
        assert torch.allclose(A.grad, expected_A_grad)

    def test_autograd_grad(self, quadratic):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        result = apply_tesseract(
            quadratic,
            {
                "x": x,
                "A": np.eye(3, dtype=np.float32),
                "b": np.zeros(3, dtype=np.float32),
            },
        )
        y = result["y"]

        (grad_y0,) = torch.autograd.grad(y[0], x, retain_graph=True)
        (grad_y1,) = torch.autograd.grad(y[1], x, retain_graph=True)
        (grad_y2,) = torch.autograd.grad(y[2], x)

        assert torch.allclose(grad_y0, torch.tensor([2.0, 0.0, 0.0]))
        assert torch.allclose(grad_y1, torch.tensor([0.0, 4.0, 0.0]))
        assert torch.allclose(grad_y2, torch.tensor([0.0, 0.0, 6.0]))

    def test_forward_mode_jvp(self, quadratic):
        x = torch.tensor([1.0, 2.0, 3.0])
        A = torch.eye(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float32)
        tangent_x = torch.tensor([1.0, 0.0, 0.0])

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, tangent_x)
            result = apply_tesseract(quadratic, {"x": x_dual, "A": A, "b": b})
            primal, tangent_out = fwAD.unpack_dual(result["y"])

        assert torch.allclose(primal, torch.tensor([1.0, 4.0, 9.0]))
        assert torch.allclose(tangent_out, torch.tensor([2.0, 0.0, 0.0]))

    def test_gradients_match_native_pytorch(self, quadratic):
        """Verify gradients match PyTorch's own autodiff for the same math."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        A = torch.eye(3, dtype=torch.float32, requires_grad=True)
        b = torch.zeros(3, dtype=torch.float32, requires_grad=True)

        result = apply_tesseract(quadratic, {"x": x, "A": A, "b": b})
        result["y"].sum().backward()

        # Reference: native PyTorch
        x_ref = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        A_ref = torch.eye(3, dtype=torch.float32, requires_grad=True)
        b_ref = torch.zeros(3, dtype=torch.float32, requires_grad=True)
        y_ref = A_ref @ (x_ref**2) + b_ref
        y_ref.sum().backward()

        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(A.grad, A_ref.grad)
        assert torch.allclose(b.grad, b_ref.grad)


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
