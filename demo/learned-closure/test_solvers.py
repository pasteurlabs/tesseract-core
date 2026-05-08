"""Smoke tests for the learned closure demo.

Tests the composition pattern: an outer loop calls the closure Tesseract to get
a viscosity field, then calls the solver Tesseract to step forward. Gradients
flow end-to-end through both Tesseracts via apply_tesseract / jax.grad.

This is the same pattern that would work with a Fortran solver Tesseract backed
by Enzyme or a hand-written adjoint — the solver just needs apply + VJP with
the interface (u, nu_field, dt) -> u_next.
"""

import sys

sys.path.insert(0, "neural_viscosity")
sys.path.insert(0, "burgers_solver")

import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract

from tesseract_core import Tesseract

jax.config.update("jax_enable_x64", True)

import burgers_solver.tesseract_api as solver_api  # noqa: E402
import neural_viscosity.tesseract_api as closure_api  # noqa: E402

CLOSURE_API_PATH = "neural_viscosity/tesseract_api.py"
SOLVER_API_PATH = "burgers_solver/tesseract_api.py"

N = 128
DX = 1.0 / (N - 1)
X_GRID = jnp.linspace(0.0, 1.0, N)


def _make_closure_params(key):
    """Initialize random closure network weights."""
    keys = jax.random.split(key, 6)
    w1 = jax.random.normal(keys[0], (3, 32)) * jnp.sqrt(2.0 / 3)
    b1 = jnp.zeros(32)
    w2 = jax.random.normal(keys[1], (32, 32)) * jnp.sqrt(2.0 / 32)
    b2 = jnp.zeros(32)
    w3 = jax.random.normal(keys[2], (32, 1)) * jnp.sqrt(2.0 / 32)
    b3 = jnp.zeros(1)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def _make_initial_condition():
    """Smooth initial condition: a sine wave."""
    u0 = jnp.sin(2 * jnp.pi * X_GRID)
    return u0


def test_closure_forward():
    print("=== Neural viscosity closure forward pass ===")
    key = jax.random.PRNGKey(0)
    params = _make_closure_params(key)
    u0 = _make_initial_condition()
    dudx = jnp.gradient(u0, DX)

    inputs = closure_api.InputSchema(u=u0, dudx=dudx, x=X_GRID, **params)
    out = closure_api.apply(inputs)
    nu = out["nu"]

    print(f"  Shape: {nu.shape}, range: [{float(nu.min()):.4f}, {float(nu.max()):.4f}]")
    assert nu.shape == (N,)
    assert jnp.all(nu > 0), "Viscosity must be positive"
    print("  PASSED")


def test_solver_single_step():
    print("\n=== Solver single timestep ===")
    u0 = _make_initial_condition()
    nu = jnp.full(N, 0.01)  # constant viscosity
    dt = 1e-4

    inputs = solver_api.InputSchema(u=u0, nu=nu, dt=dt)
    out = solver_api.apply(inputs)
    u_next = out["u_next"]

    print(f"  Shape: {u_next.shape}")
    print(f"  Max change: {float(jnp.max(jnp.abs(u_next - u0))):.6e}")
    assert u_next.shape == (N,)
    assert jnp.all(jnp.isfinite(u_next)), "Solution contains NaN or Inf"
    # Boundary values should be preserved
    assert float(u_next[0]) == float(u0[0]), "Left BC violated"
    assert float(u_next[-1]) == float(u0[-1]), "Right BC violated"
    print("  PASSED")


def test_solver_gradient():
    print("\n=== Solver gradient (VJP w.r.t. nu field) ===")
    u0 = _make_initial_condition()
    nu = jnp.full(N, 0.01)
    dt = 1e-4

    def loss_fn(nu_field):
        out = solver_api.apply_jit({"u": u0, "nu": nu_field, "dt": dt})
        return jnp.mean(out["u_next"] ** 2)

    grad_nu = jax.grad(loss_fn)(nu)
    print(
        f"  Gradient shape: {grad_nu.shape}, norm: {float(jnp.linalg.norm(grad_nu)):.6e}"
    )
    assert grad_nu.shape == (N,)
    assert jnp.all(jnp.isfinite(grad_nu))
    print("  PASSED")


def test_composition_forward():
    """Outer loop calling closure + solver via apply_tesseract."""
    print("\n=== Composed forward pass (closure + solver via apply_tesseract) ===")
    closure_tess = Tesseract.from_tesseract_api(CLOSURE_API_PATH)
    solver_tess = Tesseract.from_tesseract_api(SOLVER_API_PATH)

    key = jax.random.PRNGKey(42)
    params = _make_closure_params(key)
    u = _make_initial_condition()
    dt = 1e-4
    n_steps = 50

    for _step in range(n_steps):
        dudx = jnp.gradient(u, DX)
        closure_out = apply_tesseract(
            closure_tess, {"u": u, "dudx": dudx, "x": X_GRID, **params}
        )
        nu = closure_out["nu"]
        solver_out = apply_tesseract(solver_tess, {"u": u, "nu": nu, "dt": dt})
        u = solver_out["u_next"]

    print(f"  Shape: {u.shape}")
    print(f"  Range: [{float(u.min()):.4f}, {float(u.max()):.4f}]")
    assert u.shape == (N,)
    assert jnp.all(jnp.isfinite(u)), "Solution contains NaN or Inf"
    print("  PASSED")


def test_composition_gradient():
    """End-to-end gradient through solver + closure via apply_tesseract."""
    print("\n=== End-to-end gradient (closure + solver via apply_tesseract) ===")
    closure_tess = Tesseract.from_tesseract_api(CLOSURE_API_PATH)
    solver_tess = Tesseract.from_tesseract_api(SOLVER_API_PATH)

    key = jax.random.PRNGKey(42)
    params = _make_closure_params(key)
    u0 = _make_initial_condition()
    target = 0.9 * u0
    dt = 1e-4
    n_steps = 20

    def loss_fn(w1):
        u = u0
        p = {**params, "w1": w1}
        for _step in range(n_steps):
            dudx = jnp.gradient(u, DX)
            closure_out = apply_tesseract(
                closure_tess, {"u": u, "dudx": dudx, "x": X_GRID, **p}
            )
            nu = closure_out["nu"]
            solver_out = apply_tesseract(solver_tess, {"u": u, "nu": nu, "dt": dt})
            u = solver_out["u_next"]
        return jnp.mean((u - target) ** 2)

    # AD gradient
    grad_ad = jax.grad(loss_fn)(params["w1"])

    # Finite difference check on one element
    eps = 1e-5
    idx = (0, 0)
    w1_plus = params["w1"].at[idx].add(eps)
    w1_minus = params["w1"].at[idx].add(-eps)
    fd = (loss_fn(w1_plus) - loss_fn(w1_minus)) / (2 * eps)

    rel_err = abs(float(grad_ad[idx]) - float(fd)) / (abs(float(fd)) + 1e-30)
    print(
        f"  AD: {float(grad_ad[idx]):.6e}, FD: {float(fd):.6e}, Rel error: {rel_err:.2e}"
    )
    assert rel_err < 1e-2, f"Gradient error too large: {rel_err}"
    print("  PASSED")


if __name__ == "__main__":
    test_closure_forward()
    test_solver_single_step()
    test_solver_gradient()
    test_composition_forward()
    test_composition_gradient()
    print("\nAll smoke tests passed.")
