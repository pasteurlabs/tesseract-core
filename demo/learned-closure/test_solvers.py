"""Smoke tests for the learned closure demo (PyTorch version).

Tests the composition pattern: an outer loop calls the closure Tesseract to get
a viscosity field, then calls the solver Tesseract to step forward. Gradients
flow end-to-end through both Tesseracts via apply_tesseract / torch.autograd.

This is the same pattern that would work with a Fortran solver Tesseract backed
by Enzyme or a hand-written adjoint — the solver just needs apply + VJP with
the interface (u, nu_field, dt) -> u_next.
"""

import sys

sys.path.insert(0, "neural_viscosity")
sys.path.insert(0, "burgers_solver")

import burgers_solver.tesseract_api as solver_api
import neural_viscosity.tesseract_api as closure_api
import numpy as np
import torch
from tesseract_torch import apply_tesseract

from tesseract_core import Tesseract

CLOSURE_API_PATH = "neural_viscosity/tesseract_api.py"
SOLVER_API_PATH = "burgers_solver/tesseract_api.py"

N = 128
DX = 1.0 / (N - 1)
X_GRID = torch.linspace(0.0, 1.0, N, dtype=torch.float64)


def _make_closure_params(seed=0):
    """Initialize random closure network weights."""
    rng = torch.Generator().manual_seed(seed)
    w1 = torch.randn(3, 32, dtype=torch.float64, generator=rng) * np.sqrt(2.0 / 3)
    b1 = torch.zeros(32, dtype=torch.float64)
    w2 = torch.randn(32, 32, dtype=torch.float64, generator=rng) * np.sqrt(2.0 / 32)
    b2 = torch.zeros(32, dtype=torch.float64)
    w3 = torch.randn(32, 1, dtype=torch.float64, generator=rng) * np.sqrt(2.0 / 32)
    b3 = torch.zeros(1, dtype=torch.float64)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def _make_initial_condition():
    """Smooth initial condition: a sine wave."""
    u0 = torch.sin(2 * np.pi * X_GRID)
    return u0


def test_closure_forward():
    print("=== Neural viscosity closure forward pass ===")
    params = _make_closure_params(seed=0)
    u0 = _make_initial_condition()
    dudx = torch.gradient(u0, spacing=(DX,))[0]

    inputs = closure_api.InputSchema(u=u0, dudx=dudx, x=X_GRID, **params)
    out = closure_api.apply(inputs)
    nu = out["nu"]

    print(f"  Shape: {nu.shape}, range: [{float(nu.min()):.4f}, {float(nu.max()):.4f}]")
    assert nu.shape == (N,)
    assert torch.all(nu > 0), "Viscosity must be positive"
    print("  PASSED")


def test_solver_single_step():
    print("\n=== Solver single timestep ===")
    u0 = _make_initial_condition()
    nu = torch.full((N,), 0.01, dtype=torch.float64)
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
    nu = torch.full((N,), 0.01, dtype=torch.float64, requires_grad=True)
    dt = 1e-4

    tensor_inputs = {
        "u": u0.clone(),
        "nu": nu,
        "dt": torch.tensor(dt, dtype=torch.float64),
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


def test_composition_forward():
    """Outer loop calling closure + solver via apply_tesseract."""
    print("\n=== Composed forward pass (closure + solver via apply_tesseract) ===")
    closure_tess = Tesseract.from_tesseract_api(CLOSURE_API_PATH)
    solver_tess = Tesseract.from_tesseract_api(SOLVER_API_PATH)

    params = _make_closure_params(seed=42)
    u = _make_initial_condition()
    dt = 1e-4
    n_steps = 50

    for _step in range(n_steps):
        dudx = torch.gradient(u, spacing=(DX,))[0]
        closure_out = apply_tesseract(
            closure_tess, {"u": u, "dudx": dudx, "x": X_GRID, **params}
        )
        nu = closure_out["nu"]
        solver_out = apply_tesseract(solver_tess, {"u": u, "nu": nu, "dt": dt})
        u = solver_out["u_next"]

    print(f"  Shape: {u.shape}")
    print(f"  Range: [{float(u.min()):.4f}, {float(u.max()):.4f}]")
    assert u.shape == (N,)
    assert torch.all(torch.isfinite(u)), "Solution contains NaN or Inf"
    print("  PASSED")


def test_composition_gradient():
    """End-to-end gradient through solver + closure via apply_tesseract."""
    print("\n=== End-to-end gradient (closure + solver via apply_tesseract) ===")
    closure_tess = Tesseract.from_tesseract_api(CLOSURE_API_PATH)
    solver_tess = Tesseract.from_tesseract_api(SOLVER_API_PATH)

    params = _make_closure_params(seed=42)
    u0 = _make_initial_condition()
    target = 0.9 * u0
    dt = 1e-4
    n_steps = 20

    # Make w1 require grad for end-to-end differentiation
    w1 = params["w1"].clone().requires_grad_(True)

    def run_forward(w1_val):
        u = u0.clone()
        p = {**params, "w1": w1_val}
        for _step in range(n_steps):
            dudx = torch.gradient(u, spacing=(DX,))[0]
            closure_out = apply_tesseract(
                closure_tess, {"u": u, "dudx": dudx, "x": X_GRID, **p}
            )
            nu = closure_out["nu"]
            solver_out = apply_tesseract(solver_tess, {"u": u, "nu": nu, "dt": dt})
            u = solver_out["u_next"]
        return torch.mean((u - target) ** 2)

    # AD gradient
    loss = run_forward(w1)
    (grad_ad,) = torch.autograd.grad(loss, w1)

    # Finite difference check on one element
    eps = 1e-5
    idx = (0, 0)
    ad_val = float(grad_ad[idx])

    w1_plus = w1.detach().clone()
    w1_plus[idx] += eps
    w1_minus = w1.detach().clone()
    w1_minus[idx] -= eps
    fd = (float(run_forward(w1_plus)) - float(run_forward(w1_minus))) / (2 * eps)

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
