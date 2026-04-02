"""Smoke tests for multiphysics demo solvers."""

import sys

sys.path.insert(0, "thermal_solver")
sys.path.insert(0, "structural_solver")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import structural_solver.tesseract_api as structural_api  # noqa: E402
import thermal_solver.tesseract_api as thermal_api  # noqa: E402


def test_thermal_forward():
    print("=== Thermal solver forward pass ===")
    from thermal_solver.tesseract_api import InputSchema as ThermalInput

    inputs = ThermalInput(
        source_x=0.5,
        source_y=0.5,
        source_intensity=10.0,
        displacement=jnp.zeros((30, 30, 2), dtype=jnp.float32),
    )
    out = thermal_api.apply(inputs)
    temp = out["temperature"]
    print(
        f"  Shape: {temp.shape}, range: [{float(temp.min()):.4f}, {float(temp.max()):.4f}]"
    )
    assert temp.shape == (30, 30)
    assert float(temp.max()) > 0
    print("  PASSED")
    return temp


def test_structural_forward(temperature):
    print("\n=== Structural solver forward pass ===")
    from structural_solver.tesseract_api import InputSchema as StructuralInput

    inputs = StructuralInput(temperature=temperature)
    out = structural_api.apply(inputs)
    print(f"  Displacement: {out['displacement'].shape}, Stress: {out['stress'].shape}")
    print(f"  Objective: {float(out['objective']):.6e}")
    assert out["displacement"].shape == (30, 30, 2)
    assert out["stress"].shape == (30, 30, 3)
    print("  PASSED")


def test_one_way_gradient():
    print("\n=== One-way coupled gradient ===")

    def pipeline(source_x, source_y):
        thermal_out = thermal_api.apply_jit(
            {
                "source_x": source_x,
                "source_y": source_y,
                "source_intensity": jnp.float32(10.0),
                "source_width": 0.1,
                "displacement": jnp.zeros((30, 30, 2), dtype=jnp.float32),
                "conductivity": 1.0,
                "boundary_temp": 0.0,
            }
        )
        structural_out = structural_api.apply_jit(
            {
                "temperature": thermal_out["temperature"],
                "youngs_modulus": 200.0,
                "poissons_ratio": 0.3,
                "thermal_expansion": 1e-3,
            }
        )
        return structural_out["objective"]

    grad_fn = jax.grad(pipeline, argnums=(0, 1))
    sx, sy = jnp.float32(0.3), jnp.float32(0.7)
    grads = grad_fn(sx, sy)

    eps = jnp.float32(1e-4)
    fd_x = (pipeline(sx + eps, sy) - pipeline(sx - eps, sy)) / (2 * eps)
    fd_y = (pipeline(sx, sy + eps) - pipeline(sx, sy - eps)) / (2 * eps)

    rel_err_x = abs(float(grads[0]) - float(fd_x)) / (abs(float(fd_x)) + 1e-30)
    rel_err_y = abs(float(grads[1]) - float(fd_y)) / (abs(float(fd_y)) + 1e-30)
    print(f"  Rel error: x={rel_err_x:.2e}, y={rel_err_y:.2e}")
    assert max(rel_err_x, rel_err_y) < 1e-2, "Gradient error too large"
    print("  PASSED")


def test_two_way_gradient():
    print("\n=== Two-way coupled gradient (lax.scan) ===")

    def coupled_pipeline(source_x, source_y):
        temp = jnp.zeros((30, 30), dtype=jnp.float32)
        disp = jnp.zeros((30, 30, 2), dtype=jnp.float32)

        def coupling_step(carry, _):
            _temp, disp = carry
            thermal_out = thermal_api.apply_jit(
                {
                    "source_x": source_x,
                    "source_y": source_y,
                    "source_intensity": jnp.float32(10.0),
                    "source_width": 0.1,
                    "displacement": disp,
                    "conductivity": 1.0,
                    "boundary_temp": 0.0,
                }
            )
            structural_out = structural_api.apply_jit(
                {
                    "temperature": thermal_out["temperature"],
                    "youngs_modulus": 200.0,
                    "poissons_ratio": 0.3,
                    "thermal_expansion": 1e-3,
                }
            )
            return (
                thermal_out["temperature"],
                structural_out["displacement"],
            ), structural_out["objective"]

        _, objectives = jax.lax.scan(coupling_step, (temp, disp), None, length=3)
        return objectives[-1]

    grad_fn = jax.grad(coupled_pipeline, argnums=(0, 1))
    sx, sy = jnp.float32(0.3), jnp.float32(0.7)
    grads = grad_fn(sx, sy)

    eps = jnp.float32(1e-4)
    fd_x = (coupled_pipeline(sx + eps, sy) - coupled_pipeline(sx - eps, sy)) / (2 * eps)
    fd_y = (coupled_pipeline(sx, sy + eps) - coupled_pipeline(sx, sy - eps)) / (2 * eps)

    rel_err_x = abs(float(grads[0]) - float(fd_x)) / (abs(float(fd_x)) + 1e-30)
    rel_err_y = abs(float(grads[1]) - float(fd_y)) / (abs(float(fd_y)) + 1e-30)
    print(f"  Rel error: x={rel_err_x:.2e}, y={rel_err_y:.2e}")
    assert max(rel_err_x, rel_err_y) < 1e-2, "Gradient error too large"
    print("  PASSED")


def test_implicit_differentiation():
    print("\n=== Implicit differentiation vs unrolled ===")

    def G(temp_disp, params):
        _temp, disp = temp_disp
        sx, sy, q = params
        thermal_out = thermal_api.apply_jit(
            {
                "source_x": sx,
                "source_y": sy,
                "source_intensity": q,
                "source_width": jnp.float32(0.15),
                "displacement": disp,
                "conductivity": 1.0,
                "boundary_temp": 0.0,
            }
        )
        structural_out = structural_api.apply_jit(
            {
                "temperature": thermal_out["temperature"],
                "youngs_modulus": 200.0,
                "poissons_ratio": 0.3,
                "thermal_expansion": 1e-3,
            }
        )
        return (thermal_out["temperature"], structural_out["displacement"])

    N_ITERS = 3
    SENSORS = [(8, 8), (8, 22), (22, 8), (22, 22)]
    TARGETS = [
        jnp.float32(0.01),
        jnp.float32(0.02),
        jnp.float32(0.02),
        jnp.float32(0.05),
    ]

    def make_loss(temp):
        loss = jnp.float32(0.0)
        for (si, sj), target in zip(SENSORS, TARGETS, strict=False):
            loss = loss + (temp[si, sj] - target) ** 2
        return loss

    # Unrolled version
    def unrolled(params):
        sx, sy, log_q = params[0], params[1], params[2]
        q = jnp.exp(log_q)
        temp = jnp.zeros((30, 30), dtype=jnp.float32)
        disp = jnp.zeros((30, 30, 2), dtype=jnp.float32)

        def step(carry, _):
            return G(carry, (sx, sy, q)), None

        (final_temp, _), _ = jax.lax.scan(step, (temp, disp), None, length=N_ITERS)
        return make_loss(final_temp)

    # Implicit version
    def implicit(params):
        sx, sy, log_q = params[0], params[1], params[2]
        q = jnp.exp(log_q)

        @jax.custom_vjp
        def solve(p):
            temp = jnp.zeros((30, 30), dtype=jnp.float32)
            disp = jnp.zeros((30, 30, 2), dtype=jnp.float32)

            def step(carry, _):
                return G(carry, p), None

            (ft, fd), _ = jax.lax.scan(step, (temp, disp), None, length=N_ITERS)
            return (ft, fd)

        def fwd(p):
            fp = solve(p)
            return fp, (fp, p)

        def bwd(res, g):
            (ft, fd), p = res
            vt, vd = g

            def G_state(td):
                return G(td, p)

            def adj_step(lam, _):
                _, vjp_fn = jax.vjp(G_state, (ft, fd))
                dGT = vjp_fn(lam)[0]
                return (vt + dGT[0], vd + dGT[1]), None

            (lt, ld), _ = jax.lax.scan(adj_step, (vt, vd), None, length=20)

            def G_params(pp):
                return G((ft, fd), pp)

            _, vjp_p = jax.vjp(G_params, p)
            return (vjp_p((lt, ld))[0],)

        solve.defvjp(fwd, bwd)
        final_temp, _ = solve((sx, sy, q))
        return make_loss(final_temp)

    p0 = jnp.array([0.3, 0.7, jnp.log(8.0)], dtype=jnp.float32)
    grad_u = jax.grad(unrolled)(p0)
    grad_i = jax.grad(implicit)(p0)

    for i, name in enumerate(["sx", "sy", "log_q"]):
        rel = abs(float(grad_i[i]) - float(grad_u[i])) / (abs(float(grad_u[i])) + 1e-30)
        print(
            f"  {name}: unrolled={float(grad_u[i]):.6e} implicit={float(grad_i[i]):.6e} rel_err={rel:.2e}"
        )
        assert rel < 1e-3, f"Implicit gradient for {name} deviates too much: {rel:.2e}"
    print("  PASSED")


if __name__ == "__main__":
    temp = test_thermal_forward()
    test_structural_forward(temp)
    test_one_way_gradient()
    test_two_way_gradient()
    test_implicit_differentiation()
    print("\nAll smoke tests passed.")
