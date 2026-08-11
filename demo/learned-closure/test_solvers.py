"""Smoke tests for the learned closure demo.

Covers the solver Tesseract (forward, VJP) and the closure + solver composition.
The solver is loaded in-process via ``Tesseract.from_tesseract_api`` so the tests
run without Docker; the notebook serves the same solver over HTTP via
``Tesseract.from_image``.
"""

import sys

sys.path.insert(0, "burgers_solver")

import burgers_solver.tesseract_api as solver_api
import numpy as np
import torch
import torch.nn as nn
from tesseract_torch import apply_tesseract

from tesseract_core import Tesseract

torch.set_default_dtype(torch.float64)

SOLVER_API_PATH = "burgers_solver/tesseract_api.py"

N = 128
DX = 1.0 / (N - 1)
X_GRID = torch.linspace(0.0, 1.0, N)


class ViscosityNet(nn.Module):
    """MLP closure: local flow features (u, du/dx, x) -> viscosity nu."""

    def __init__(self, hidden_dim=32, nu_max=0.05):
        super().__init__()
        self.nu_max = nu_max
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, u, dudx, x):
        features = torch.stack([u, dudx, x], dim=-1)
        out = self.net(features)[:, 0]
        return self.nu_max * torch.sigmoid(out)


def _make_initial_condition():
    """Smooth initial condition: a sine wave."""
    return torch.sin(2 * np.pi * X_GRID)


def test_closure_forward():
    print("=== Neural viscosity closure forward pass ===")
    torch.manual_seed(0)
    closure = ViscosityNet()
    u0 = _make_initial_condition()
    dudx = torch.gradient(u0, spacing=(DX,))[0]

    with torch.no_grad():
        nu = closure(u0, dudx, X_GRID)

    print(f"  Shape: {nu.shape}, range: [{float(nu.min()):.4f}, {float(nu.max()):.4f}]")
    assert nu.shape == (N,)
    assert torch.all(nu > 0), "Viscosity must be positive"
    print("  PASSED")


def test_solver_single_step():
    print("\n=== Solver single timestep ===")
    u0 = _make_initial_condition()
    nu = torch.full((N,), 0.01)
    dt = 1e-4

    inputs = solver_api.InputSchema(u=u0, nu=nu, dt=dt)
    out = solver_api.apply(inputs)
    u_next = out["u_next"]

    print(f"  Shape: {u_next.shape}")
    print(f"  Max change: {float(torch.max(torch.abs(u_next - u0))):.6e}")
    assert u_next.shape == (N,)
    assert torch.all(torch.isfinite(u_next)), "Solution contains NaN or Inf"
    # Boundary values should be preserved
    assert float(u_next[0]) == float(u0[0]), "Left BC violated"
    assert float(u_next[-1]) == float(u0[-1]), "Right BC violated"
    print("  PASSED")


def test_solver_gradient():
    print("\n=== Solver gradient (VJP w.r.t. nu field) ===")
    u0 = _make_initial_condition()
    nu = torch.full((N,), 0.01, requires_grad=True)
    dt = 1e-4

    tensor_inputs = {
        "u": u0.clone(),
        "nu": nu,
        "dt": torch.tensor(dt),
    }
    out = solver_api.evaluate(tensor_inputs)
    loss = torch.mean(out["u_next"] ** 2)
    loss.backward()

    grad_nu = nu.grad
    print(
        f"  Gradient shape: {grad_nu.shape}, norm: {float(torch.linalg.norm(grad_nu)):.6e}"
    )
    assert grad_nu.shape == (N,)
    assert torch.all(torch.isfinite(grad_nu))
    print("  PASSED")


def _solve_with_closure(u0, closure, solver_tess, dt, n_steps):
    u = u0
    for _step in range(n_steps):
        dudx = torch.zeros_like(u)
        dudx[1:-1] = (u[2:] - u[:-2]) / (2 * DX)
        nu = closure(u, dudx, X_GRID)
        solver_out = apply_tesseract(solver_tess, {"u": u, "nu": nu, "dt": dt})
        u = solver_out["u_next"]
    return u


def test_composition_forward():
    """Outer loop: plain torch closure + solver Tesseract via apply_tesseract."""
    print("\n=== Composed forward pass (closure + solver Tesseract) ===")
    solver_tess = Tesseract.from_tesseract_api(SOLVER_API_PATH)

    torch.manual_seed(42)
    closure = ViscosityNet()
    u0 = _make_initial_condition()

    with torch.no_grad():
        u = _solve_with_closure(u0, closure, solver_tess, dt=1e-4, n_steps=50)

    print(f"  Shape: {u.shape}")
    print(f"  Range: [{float(u.min()):.4f}, {float(u.max()):.4f}]")
    assert u.shape == (N,)
    assert torch.all(torch.isfinite(u)), "Solution contains NaN or Inf"
    print("  PASSED")


def test_composition_gradient():
    """End-to-end gradient: loss -> solver VJP -> network weights."""
    print("\n=== End-to-end gradient (closure + solver Tesseract) ===")
    solver_tess = Tesseract.from_tesseract_api(SOLVER_API_PATH)

    torch.manual_seed(42)
    closure = ViscosityNet()
    u0 = _make_initial_condition()
    target = 0.9 * u0
    n_steps = 20

    def run_forward():
        u = _solve_with_closure(
            u0.clone(), closure, solver_tess, dt=1e-4, n_steps=n_steps
        )
        return torch.mean((u - target) ** 2)

    # AD gradient on one weight element of the first layer
    closure.zero_grad()
    loss = run_forward()
    loss.backward()
    w = closure.net[0].weight
    idx = (0, 0)
    ad_val = float(w.grad[idx])

    # Finite difference check on the same element
    eps = 1e-5
    with torch.no_grad():
        orig = w[idx].item()
        w[idx] = orig + eps
        l_plus = float(run_forward())
        w[idx] = orig - eps
        l_minus = float(run_forward())
        w[idx] = orig
    fd = (l_plus - l_minus) / (2 * eps)

    rel_err = abs(ad_val - fd) / (abs(fd) + 1e-30)
    print(f"  AD: {ad_val:.6e}, FD: {fd:.6e}, Rel error: {rel_err:.2e}")
    assert rel_err < 1e-2, f"Gradient error too large: {rel_err}"
    print("  PASSED")


if __name__ == "__main__":
    test_closure_forward()
    test_solver_single_step()
    test_solver_gradient()
    test_composition_forward()
    test_composition_gradient()
    print("\nAll smoke tests passed.")
