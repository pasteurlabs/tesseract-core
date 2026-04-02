# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from tesseract_core import Tesseract

try:
    import torch
except ImportError:
    HAS_TORCH = False
else:
    import torch.autograd.forward_ad as fwAD

    from tesseract_core.torch_compat import apply_tesseract

    HAS_TORCH = True


MESHSTATS_API = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "meshstats_finitediff"
    / "tesseract_api.py"
)


@pytest.fixture(scope="module", autouse=True)
def needs_torch():
    if not HAS_TORCH:
        pytest.fail("PyTorch is not installed.")


@pytest.fixture(scope="module")
def meshstats():
    return Tesseract.from_tesseract_api(MESHSTATS_API)


def _make_inputs(requires_grad: bool = False):
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


def test_forward_pass(meshstats):
    inputs, _ = _make_inputs()
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


def test_output_types(meshstats):
    inputs, _ = _make_inputs()
    result = apply_tesseract(meshstats, inputs)

    assert isinstance(result["statistics"]["barycenter"], torch.Tensor)
    assert isinstance(result["statistics"]["first_point_coordinates"], torch.Tensor)


def test_backward_pass(meshstats):
    inputs, points = _make_inputs(requires_grad=True)
    result = apply_tesseract(meshstats, inputs)

    # barycenter = mean(points, axis=0), so d(sum(barycenter))/d(points) = 1/n
    result["statistics"]["barycenter"].sum().backward()

    n = 3
    expected_grad = torch.full_like(points, 1.0 / n)
    assert torch.allclose(points.grad, expected_grad, atol=1e-4)


def test_autograd_grad(meshstats):
    inputs, points = _make_inputs(requires_grad=True)
    result = apply_tesseract(meshstats, inputs)

    (grad_points,) = torch.autograd.grad(
        result["statistics"]["barycenter"].sum(), points
    )

    n = 3
    expected_grad = torch.full_like(points, 1.0 / n)
    assert torch.allclose(grad_points, expected_grad, atol=1e-4)


def test_forward_mode_jvp(meshstats):
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
    assert torch.allclose(tangent_bary, torch.tensor([1.0 / 3, 0.0, 0.0]), atol=1e-4)
    # first_point_coordinates = points[0] -> tangent = [1, 0, 0]
    assert torch.allclose(tangent_fpc, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
