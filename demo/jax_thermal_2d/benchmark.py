#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: Enzyme (Fortran) vs JAX thermal 2D solver.

Compares wall-clock time for forward pass and VJP across grid sizes.
Both images must be built before running:

    tesseract build examples/enzyme_thermal_2d
    tesseract build examples/jax_thermal_2d

NOTE: For a fair comparison, both images must run on the same architecture.
The Enzyme image requires linux/amd64 (LFortran + Enzyme are x86-only),
so on ARM machines (Apple Silicon) it runs under Rosetta emulation.
Run this script on an x86-64 machine for meaningful results.
"""

import time

import numpy as np

from tesseract_core import Tesseract

IMAGES = {
    "Enzyme": "enzyme-thermal-2d:latest",
    "JAX": "jax-thermal-2d:latest",
}

GRID_SIZES = [
    (30, 30, 50),
    (100, 100, 50),
    (200, 200, 50),
    (300, 300, 50),
]

N_WARMUP = 5
N_TRIALS = 15


def make_inputs(nx, ny, n_steps):
    n = nx * ny
    dx = 0.1 / (nx - 1)
    dy = 0.05 / (ny - 1)
    k_max = 45.0 + (-0.01) * 373.15
    dt_max = 0.5 * 7850.0 * 460.0 / (k_max * (1 / dx**2 + 1 / dy**2))
    dt = 0.9 * dt_max

    return {
        "T_init": np.full(n, 293.15),
        "Q": np.zeros(n),
        "nx": nx,
        "ny": ny,
        "n_steps": n_steps,
        "k0": 45.0,
        "k1": -0.01,
        "rho": 7850.0,
        "cp": 460.0,
        "h_conv": 25.0,
        "T_inf": 293.15,
        "T_hot": 373.15,
        "Lx": 0.1,
        "Ly": 0.05,
        "dt": dt,
    }


def benchmark_image(image, inputs):
    n = inputs["nx"] * inputs["ny"]
    cotangent = np.full(n, 1.0 / n)
    vjp_kwargs = dict(
        vjp_inputs=["k0", "k1", "T_init"],
        vjp_outputs=["T_final"],
        cotangent_vector={"T_final": cotangent},
    )

    with Tesseract.from_image(image) as t:
        # Warmup (includes JIT compilation for JAX)
        for _ in range(N_WARMUP):
            t.apply(inputs=inputs)
            t.vector_jacobian_product(inputs=inputs, **vjp_kwargs)

        # Benchmark forward
        times_fwd = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            t.apply(inputs=inputs)
            times_fwd.append(time.perf_counter() - t0)

        # Benchmark VJP
        times_vjp = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            t.vector_jacobian_product(inputs=inputs, **vjp_kwargs)
            times_vjp.append(time.perf_counter() - t0)

    return np.median(times_fwd), np.median(times_vjp)


def main():
    print("Thermal 2D Benchmark: Enzyme (Fortran + LLVM AD) vs JAX (XLA JIT)")
    print(f"Warmup: {N_WARMUP}, Trials: {N_TRIALS} (median reported)")
    print("=" * 75)

    # Correctness check first
    print("\nCorrectness check (30x30, 50 steps)...")
    inputs_check = make_inputs(30, 30, 50)
    results = {}
    for label, image in IMAGES.items():
        with Tesseract.from_image(image) as t:
            r = t.apply(inputs=inputs_check)
            results[label] = np.array(r["T_final"])
    diff = np.max(np.abs(results["Enzyme"] - results["JAX"]))
    print(f"  Forward max abs diff: {diff:.2e}")

    # Benchmark
    print(
        f"\n{'Grid':>10s} {'DOFs':>8s} | {'Enzyme fwd':>11s} {'Enzyme VJP':>11s} "
        f"{'JAX fwd':>11s} {'JAX VJP':>11s} | {'VJP/fwd (E)':>11s} {'VJP/fwd (J)':>11s}"
    )
    print("-" * 100)

    for nx, ny, n_steps in GRID_SIZES:
        inputs = make_inputs(nx, ny, n_steps)
        n = nx * ny

        timings = {}
        for label, image in IMAGES.items():
            fwd, vjp = benchmark_image(image, inputs)
            timings[label] = (fwd, vjp)

        e_fwd, e_vjp = timings["Enzyme"]
        j_fwd, j_vjp = timings["JAX"]
        print(
            f"{nx}x{ny:>3d} {n_steps:>3d}st "
            f"{n:>7,d} | "
            f"{e_fwd * 1000:>8.1f} ms {e_vjp * 1000:>8.1f} ms "
            f"{j_fwd * 1000:>8.1f} ms {j_vjp * 1000:>8.1f} ms | "
            f"{e_vjp / e_fwd:>8.2f}x   {j_vjp / j_fwd:>8.2f}x"
        )


if __name__ == "__main__":
    main()
