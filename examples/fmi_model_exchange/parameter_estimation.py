# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimate the VanDerPol parameter ``mu`` by differentiating through a diffrax solve.

This is a *host-side* demo of the headline use case for the Model Exchange Tesseract:
the Tesseract provides the right-hand side ``dx/dt = f(t, x; mu)`` (and its
sensitivities) for a single point, while the time integration -- and the
differentiation *through* that integration -- happens in JAX/diffrax via
``tesseract-jax``.

Pipeline:
  1. Integrate the FMU's RHS with a known "true" mu to synthesize noisy observations.
  2. Starting from a wrong guess, minimize the trajectory MSE w.r.t. mu using
     ``jax.grad``. The gradient flows: diffrax's reverse-mode adjoint over the solve
     -> the Tesseract's vector_jacobian_product at each step -> FMI directional
     derivatives (d/dx, exact) + finite differences (d/dmu).

This script is NOT part of the container and is NOT run in CI. Run it on the host:

    pip install tesseract-jax diffrax jax
    tesseract build examples/fmi_model_exchange
    python examples/fmi_model_exchange/parameter_estimation.py

Performance: every vector-field evaluation is a separate (synchronous) round-trip to
the served Tesseract, and an adaptive solve does many of them per step -- so wall-clock
is dominated by the round-trip *count*, not the FMU work. The ``--rtol``/``--atol``,
``--t-end``/``--n-points``, and ``--steps`` knobs trade accuracy for speed; loosening
tolerances is the biggest lever. This is illustrative, not a fast solver.
"""

import argparse
import time

import diffrax
import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract

from tesseract_core import Tesseract

jax.config.update("jax_enable_x64", True)


def make_solver(tesseract: Tesseract):
    """Build a function ``solve(mu, x0, ts) -> trajectory`` that integrates the FMU RHS."""

    def vector_field(t, x, args):
        mu = args
        # VanDerPol has no continuous inputs, so `u` is omitted -- the Tesseract fills
        # it from its (empty) default. Passing a length-0 array over the wire would fail
        # to serialize, and we never differentiate w.r.t. `u` here anyway.
        outputs = apply_tesseract(
            tesseract,
            {
                "t": jnp.asarray(t, dtype=jnp.float64),
                "x": x,
                "parameters": mu,
            },
        )
        return outputs["dx_dt"]

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()

    def solve(mu, x0, ts, rtol, atol):
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=0.1,
            y0=x0,
            args=mu,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=100,
        )

    return solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="fmi-model-exchange",
        help="Built Tesseract image tag (default: fmi-model-exchange).",
    )
    parser.add_argument("--true-mu", type=float, default=1.0)
    parser.add_argument("--initial-mu", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument("--noise", type=float, default=2e-2)
    parser.add_argument("--grad-tol", type=float, default=1e-4)
    # Performance knobs: this workload is dominated by the number of sequential
    # round-trips to the Tesseract (adaptive solver steps x stages x 2 for the
    # backward pass), each ~a few ms. Looser tolerances and a shorter horizon cut
    # the step count roughly linearly -- the single biggest lever on wall-clock.
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--n-points", type=int, default=60)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    ts = jnp.linspace(0.0, args.t_end, args.n_points)
    x0 = jnp.array([2.0, 0.0])

    with Tesseract.from_image(args.image) as tesseract:
        solve = make_solver(tesseract)

        # 1. Synthesize noisy observations from the true mu.
        true_traj = solve(jnp.array([args.true_mu]), x0, ts, args.rtol, args.atol).ys
        key = jax.random.PRNGKey(0)
        observations = true_traj + args.noise * jax.random.normal(key, true_traj.shape)

        # 2. Fit mu by gradient descent through the ODE solve. `has_aux` carries the
        # solver step counts (solution.stats) out of the jitted forward solve so we can
        # report the work done each iteration.
        def loss(mu):
            sol = solve(mu, x0, ts, args.rtol, args.atol)
            return jnp.mean((sol.ys - observations) ** 2), sol.stats

        loss_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

        mu = jnp.array([args.initial_mu])
        print(f"true mu = {args.true_mu}, initial mu = {float(mu[0]):.4f}")
        total_elapsed = 0.0
        for step in range(args.steps):
            # Block on the results so JAX's async dispatch doesn't skew the timing.
            # Step 0 also pays one-time JIT compilation.
            t0 = time.perf_counter()
            (value, stats), grad = jax.block_until_ready(loss_and_grad(mu))
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed
            mu = mu - args.lr * grad
            # solution.stats reflects the forward solve; Tsit5 does ~6 RHS evals/step.
            accepted, total = int(stats["num_accepted_steps"]), int(stats["num_steps"])
            print(
                f"step {step:3d}  loss={float(value):.6e}  mu={float(mu[0]):.5f}  "
                f"grad={float(grad[0]):+.4e}  fwd_steps={accepted}/{total} "
                f"(~{total * 6} fwd evals)  t={elapsed:.2f}s (total {total_elapsed:.1f}s)"
            )
            # Each step is hundreds of Tesseract round-trips, so stop once converged
            # rather than burning iterations at the minimum.
            if abs(float(grad[0])) < args.grad_tol:
                print(f"converged (|grad| < {args.grad_tol:g})")
                break

        print(f"\nrecovered mu = {float(mu[0]):.5f}  (true {args.true_mu})")


if __name__ == "__main__":
    main()
