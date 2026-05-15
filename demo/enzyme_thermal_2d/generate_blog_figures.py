#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate figures for the Enzyme AD blog post.

Usage:
    tesseract build demo/enzyme_thermal_2d/
    python demo/enzyme_thermal_2d/generate_blog_figures.py

Outputs PNG files to demo/enzyme_thermal_2d/figures/.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from tesseract_core import Tesseract

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# ── Shared parameters ────────────────────────────────────────────────────────

nx, ny = 30, 30
n = nx * ny
n_steps = 500
dt = 0.05
Lx, Ly = 0.1, 0.05
rho = 7850.0
cp = 460.0
h_conv = 25.0
T_inf = 293.15
T_hot = 373.15
Q = np.zeros(n)
T_init_uniform = np.full(n, T_inf)

k0_true = 45.0
k1_true = -0.02

rng = np.random.default_rng(42)

plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "font.size": 11,
        "axes.titlesize": 12,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def make_inputs(k0, k1, T_init=None, n_steps_=None, dt_=None):
    return {
        "T_init": (T_init if T_init is not None else T_init_uniform),
        "Q": Q,
        "nx": nx,
        "ny": ny,
        "n_steps": n_steps_ or n_steps,
        "k0": float(k0),
        "k1": float(k1),
        "rho": rho,
        "cp": cp,
        "h_conv": h_conv,
        "T_inf": T_inf,
        "T_hot": T_hot,
        "Lx": Lx,
        "Ly": Ly,
        "dt": dt_ or dt,
    }


# ── Figure 1: FD vs Enzyme convergence ───────────────────────────────────────


def generate_fd_convergence(t):
    print("Generating FD vs Enzyme convergence plot...")

    inputs = make_inputs(k0_true, k1_true)

    cotangent = np.ones(n, dtype=np.float64)
    vjp = t.vector_jacobian_product(
        inputs=inputs,
        vjp_inputs=["k0", "k1", "h_conv"],
        vjp_outputs=["T_final"],
        cotangent_vector={"T_final": cotangent},
    )
    enzyme_dk0 = vjp["k0"]
    enzyme_dk1 = vjp["k1"]
    enzyme_dh = vjp["h_conv"]

    print(f"  Enzyme (VJP) dk0 = {enzyme_dk0:.10e}")
    print(f"  Enzyme (VJP) dk1 = {enzyme_dk1:.10e}")
    print(f"  Enzyme (VJP) dh  = {enzyme_dh:.10e}")

    epsilons = np.logspace(-1, -12, 24)
    fd_errors_k0 = []
    fd_errors_k1 = []
    fd_errors_h = []

    for i, eps in enumerate(epsilons):
        if (i + 1) % 6 == 0:
            print(f"  FD step {i + 1}/{len(epsilons)}...")

        T_plus = np.array(
            t.apply(inputs=make_inputs(k0_true + eps, k1_true))["T_final"]
        )
        T_minus = np.array(
            t.apply(inputs=make_inputs(k0_true - eps, k1_true))["T_final"]
        )
        fd_dk0 = np.sum(T_plus - T_minus) / (2 * eps)
        fd_errors_k0.append(abs(fd_dk0 - enzyme_dk0) / (abs(enzyme_dk0) + 1e-30))

        T_plus = np.array(
            t.apply(inputs=make_inputs(k0_true, k1_true + eps))["T_final"]
        )
        T_minus = np.array(
            t.apply(inputs=make_inputs(k0_true, k1_true - eps))["T_final"]
        )
        fd_dk1 = np.sum(T_plus - T_minus) / (2 * eps)
        fd_errors_k1.append(abs(fd_dk1 - enzyme_dk1) / (abs(enzyme_dk1) + 1e-30))

        inp_p = make_inputs(k0_true, k1_true)
        inp_m = make_inputs(k0_true, k1_true)
        inp_p["h_conv"] = h_conv + eps
        inp_m["h_conv"] = h_conv - eps
        T_plus = np.array(t.apply(inputs=inp_p)["T_final"])
        T_minus = np.array(t.apply(inputs=inp_m)["T_final"])
        fd_dh = np.sum(T_plus - T_minus) / (2 * eps)
        fd_errors_h.append(abs(fd_dh - enzyme_dh) / (abs(enzyme_dh) + 1e-30))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.loglog(
        epsilons,
        fd_errors_k0,
        "o-",
        ms=4,
        label="$\\partial / \\partial k_0$",
        color="#1971c2",
    )
    ax.loglog(
        epsilons,
        fd_errors_k1,
        "s-",
        ms=4,
        label="$\\partial / \\partial k_1$",
        color="#e8590c",
    )
    ax.loglog(
        epsilons,
        fd_errors_h,
        "^-",
        ms=4,
        label="$\\partial / \\partial h_{\\mathrm{conv}}$",
        color="#2f9e44",
    )
    ax.axhline(1e-15, color="gray", ls=":", alpha=0.5, label="Machine precision")

    ax.set_xlabel("Finite difference step size $\\epsilon$")
    ax.set_ylabel("Relative error vs. Enzyme gradient")
    ax.set_title(
        "Enzyme provides exact gradients; finite differences have a sweet spot"
    )
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(1e-17, 1e1)

    fig.savefig(FIGURE_DIR / "fd_convergence.png")
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fd_convergence.png'}")


# ── Figure 2: Part 1 convergence (3-panel + 4-panel) ─────────────────────────


def generate_part1_convergence(t):
    print("Generating Part 1 convergence plots...")

    result_true = t.apply(inputs=make_inputs(k0_true, k1_true))
    T_true = np.array(result_true["T_final"]).reshape(ny, nx)

    sensor_ix = [7, 15, 22]
    sensor_jy = [7, 15, 22]
    sensor_indices = np.array([jy * nx + ix for jy in sensor_jy for ix in sensor_ix])
    sensor_coords = [(ix, jy) for jy in sensor_jy for ix in sensor_ix]

    noise_std = 0.5
    T_obs = T_true.flatten()[sensor_indices] + rng.normal(
        0, noise_std, len(sensor_indices)
    )

    k0_init, k1_init = 60.0, 0.01
    history = {"k0": [k0_init], "k1": [k1_init], "loss": []}

    def obj_grad(params):
        k0, k1 = float(params[0]), float(params[1])
        inputs = make_inputs(k0, k1)
        result = t.apply(inputs=inputs)
        T_pred = np.array(result["T_final"])
        residuals = T_pred[sensor_indices] - T_obs
        loss = 0.5 * np.sum(residuals**2)

        cotangent = np.zeros(n, dtype=np.float64)
        cotangent[sensor_indices] = residuals
        vjp = t.vector_jacobian_product(
            inputs=inputs,
            vjp_inputs=["k0", "k1"],
            vjp_outputs=["T_final"],
            cotangent_vector={"T_final": cotangent},
        )
        return loss, np.array([vjp["k0"], vjp["k1"]])

    loss0, _ = obj_grad([k0_init, k1_init])
    history["loss"].append(loss0)
    print(f"  Initial loss: {loss0:.4f}")

    def callback(params):
        k0, k1 = float(params[0]), float(params[1])
        loss, _ = obj_grad(params)
        history["k0"].append(k0)
        history["k1"].append(k1)
        history["loss"].append(loss)
        print(f"    k0={k0:.4f}, k1={k1:.6f}, loss={loss:.6f}")

    result_opt = minimize(
        fun=obj_grad,
        x0=[k0_init, k1_init],
        method="L-BFGS-B",
        jac=True,
        bounds=[(5.0, 80.0), (-0.08, 0.08)],
        callback=callback,
        options={"maxiter": 50, "ftol": 1e-12, "gtol": 1e-8},
    )
    k0_opt, k1_opt = float(result_opt.x[0]), float(result_opt.x[1])

    # ── 3-panel convergence ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].semilogy(history["loss"], "k.-", linewidth=1.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss (sum of squared residuals)")
    axes[0].set_title("Convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["k0"], "b.-", linewidth=1.5, label="$k_0$ estimate")
    axes[1].axhline(
        k0_true, color="b", ls="--", alpha=0.5, label=f"$k_0$ true = {k0_true}"
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("$k_0$ [W/(m K)]")
    axes[1].set_title("Base conductivity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["k1"], "r.-", linewidth=1.5, label="$k_1$ estimate")
    axes[2].axhline(
        k1_true, color="r", ls="--", alpha=0.5, label=f"$k_1$ true = {k1_true}"
    )
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("$k_1$ [W/(m K$^2$)]")
    axes[2].set_title("Temperature coefficient")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.savefig(FIGURE_DIR / "part1_convergence.png")
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'part1_convergence.png'}")

    # ── 4-panel temperature fields ──
    T_opt_field = np.array(
        t.apply(inputs=make_inputs(k0_opt, k1_opt))["T_final"]
    ).reshape(ny, nx)
    T_init_field = np.array(
        t.apply(inputs=make_inputs(k0_init, k1_init))["T_final"]
    ).reshape(ny, nx)

    vmin = min(T_true.min(), T_opt_field.min(), T_init_field.min())
    vmax = max(T_true.max(), T_opt_field.max(), T_init_field.max())
    extent = [0, Lx * 1e3, 0, Ly * 1e3]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = [
        f"Initial guess\n$k_0$={k0_init}, $k_1$={k1_init}",
        f"Recovered\n$k_0$={k0_opt:.2f}, $k_1$={k1_opt:.4f}",
        f"Ground truth\n$k_0$={k0_true}, $k_1$={k1_true}",
        None,
    ]
    fields = [T_init_field, T_opt_field, T_true, np.abs(T_opt_field - T_true)]
    cmaps = ["hot", "hot", "hot", "Blues"]

    for i, (ax, field, cmap) in enumerate(zip(axes, fields, cmaps, strict=True)):
        if i < 3:
            im = ax.imshow(
                field,
                origin="lower",
                cmap=cmap,
                extent=extent,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            im = ax.imshow(
                field, origin="lower", cmap=cmap, extent=extent, aspect="auto"
            )
            ax.set_title(f"|Recovered - Truth|\nmax error: {field.max():.3f} K")
        if titles[i]:
            ax.set_title(titles[i])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        for ix, jy in sensor_coords:
            x_mm = ix / (nx - 1) * Lx * 1e3
            y_mm = jy / (ny - 1) * Ly * 1e3
            ax.plot(x_mm, y_mm, "ws", ms=5, markeredgecolor="blue", markeredgewidth=1)

    plt.colorbar(im, ax=axes[:3].tolist(), label="Temperature [K]", shrink=0.9)
    plt.colorbar(axes[3].images[0], ax=axes[3], label="Error [K]", shrink=0.9)
    fig.savefig(FIGURE_DIR / "part1_temperature_fields.png")
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'part1_temperature_fields.png'}")
    print(
        f"  Recovered: k0={k0_opt:.4f}, k1={k1_opt:.6f} in {result_opt.nit} iterations"
    )


# ── Figure 3: Part 2 forensics (6-panel) ─────────────────────────────────────


def generate_part2_forensics(t):
    print("Generating Part 2 forensics plots...")

    n_steps_p2 = 100
    dt_p2 = 0.05
    k0_p2 = 45.0
    k1_p2 = -0.01

    x_coord = np.linspace(0, Lx, nx)
    y_coord = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_coord, y_coord)
    X_flat, Y_flat = X.flatten(), Y.flatten()

    T_init_true = (
        T_inf
        + 40.0 * np.exp(-((X_flat - 0.04) ** 2 + (Y_flat - 0.025) ** 2) / 0.015**2)
        + 25.0 * np.exp(-((X_flat - 0.08) ** 2 + (Y_flat - 0.035) ** 2) / 0.01**2)
    )

    def make_inputs_p2(T_init_field):
        return {
            "T_init": T_init_field.astype(np.float64),
            "Q": Q,
            "nx": nx,
            "ny": ny,
            "n_steps": n_steps_p2,
            "k0": k0_p2,
            "k1": k1_p2,
            "rho": rho,
            "cp": cp,
            "h_conv": h_conv,
            "T_inf": T_inf,
            "T_hot": T_hot,
            "Lx": Lx,
            "Ly": Ly,
            "dt": dt_p2,
        }

    result_true_p2 = t.apply(inputs=make_inputs_p2(T_init_true))
    T_final_true_p2 = np.array(result_true_p2["T_final"])

    sensor_ix_p2 = np.linspace(3, nx - 4, 10, dtype=int)
    sensor_jy_p2 = np.linspace(3, ny - 4, 10, dtype=int)
    sensor_grid = np.array([jy * nx + ix for jy in sensor_jy_p2 for ix in sensor_ix_p2])

    noise_std_p2 = 0.3
    T_obs_p2 = T_final_true_p2[sensor_grid] + rng.normal(
        0, noise_std_p2, len(sensor_grid)
    )

    T_init_guess = np.full(n, T_inf)
    loss_history = []

    # Tikhonov regularization: penalize deviation from prior (uniform T_inf)
    # This stabilizes the ill-posed inverse problem (900 unknowns, 100 observations)
    alpha_reg = 0.001

    def obj_grad_p2(T_init_vec):
        inputs = make_inputs_p2(T_init_vec)
        result = t.apply(inputs=inputs)
        T_pred = np.array(result["T_final"])
        residuals = T_pred[sensor_grid] - T_obs_p2
        data_loss = 0.5 * np.sum(residuals**2)
        reg_loss = 0.5 * alpha_reg * np.sum((T_init_vec - T_inf) ** 2)
        loss = data_loss + reg_loss

        cotangent = np.zeros(n, dtype=np.float64)
        cotangent[sensor_grid] = residuals
        vjp = t.vector_jacobian_product(
            inputs=inputs,
            vjp_inputs=["T_init"],
            vjp_outputs=["T_final"],
            cotangent_vector={"T_final": cotangent},
        )
        grad = np.array(vjp["T_init"]) + alpha_reg * (T_init_vec - T_inf)
        return loss, grad

    loss0, _ = obj_grad_p2(T_init_guess)
    loss_history.append(loss0)
    print(f"  Initial loss: {loss0:.2f}")

    iter_count = [0]
    t_start = time.time()

    def callback_p2(x):
        iter_count[0] += 1
        if iter_count[0] % 10 == 0:
            loss, _ = obj_grad_p2(x)
            loss_history.append(loss)
            elapsed = time.time() - t_start
            print(
                f"    iter {iter_count[0]:3d}: loss={loss:.4f}, elapsed={elapsed:.1f}s"
            )

    result_p2 = minimize(
        fun=obj_grad_p2,
        x0=T_init_guess,
        method="L-BFGS-B",
        jac=True,
        bounds=[(250.0, 450.0)] * n,
        callback=callback_p2,
        options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-10},
    )
    elapsed_total = time.time() - t_start
    loss_final, _ = obj_grad_p2(result_p2.x)
    loss_history.append(loss_final)
    T_init_recovered = result_p2.x

    corr = np.corrcoef(T_init_true, T_init_recovered)[0, 1]
    print(f"  Optimization: {result_p2.nit} iterations, {elapsed_total:.1f}s")
    print(f"  Loss: {loss0:.2f} -> {loss_final:.4f}")
    print(f"  Correlation: {corr:.4f}")

    # Cost comparison
    n_fev = result_p2.nfev
    print("\n  Cost comparison:")
    print(
        f"    FD: {n + 1} forward solves/iter = ~{(n + 1) * elapsed_total / n_fev:.1f}s/iter"
    )
    print(f"    VJP: 2 solves/iter (fwd+rev) = ~{2 * elapsed_total / n_fev:.2f}s/iter")
    print(f"    Speedup: ~{(n + 1) / 2:.0f}x")

    # ── 6-panel figure ──
    extent = [0, Lx * 1e3, 0, Ly * 1e3]
    vmin_init = min(T_init_true.min(), T_init_recovered.min(), T_inf)
    vmax_init = max(T_init_true.max(), T_init_recovered.max())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(
        T_init_guess.reshape(ny, nx),
        origin="lower",
        cmap="hot",
        extent=extent,
        aspect="auto",
        vmin=vmin_init,
        vmax=vmax_init,
    )
    axes[0, 0].set_title("Starting guess\n(uniform ambient)")

    axes[0, 1].imshow(
        T_init_recovered.reshape(ny, nx),
        origin="lower",
        cmap="hot",
        extent=extent,
        aspect="auto",
        vmin=vmin_init,
        vmax=vmax_init,
    )
    axes[0, 1].set_title(f"Recovered $T_0$\n(corr={corr:.3f})")

    im_true = axes[0, 2].imshow(
        T_init_true.reshape(ny, nx),
        origin="lower",
        cmap="hot",
        extent=extent,
        aspect="auto",
        vmin=vmin_init,
        vmax=vmax_init,
    )
    axes[0, 2].set_title("True $T_0$\n(two Gaussian hot spots)")

    plt.colorbar(im_true, ax=axes[0, :].tolist(), label="Temperature [K]", shrink=0.85)

    for ax in axes[0, :]:
        for jy_idx in sensor_jy_p2:
            for ix_idx in sensor_ix_p2:
                x_mm = ix_idx / (nx - 1) * Lx * 1e3
                y_mm = jy_idx / (ny - 1) * Ly * 1e3
                ax.plot(x_mm, y_mm, ".", color="cyan", ms=2, alpha=0.5)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    error_p2 = np.abs(T_init_recovered - T_init_true).reshape(ny, nx)
    im_err = axes[1, 0].imshow(
        error_p2, origin="lower", cmap="Blues", extent=extent, aspect="auto"
    )
    axes[1, 0].set_title(
        f"|Recovered - True|\nmax={error_p2.max():.1f} K, mean={error_p2.mean():.1f} K"
    )
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("y [mm]")
    plt.colorbar(im_err, ax=axes[1, 0], label="Error [K]", shrink=0.85)

    axes[1, 1].scatter(T_init_true, T_init_recovered, s=3, alpha=0.5, c="steelblue")
    lims = [vmin_init - 5, vmax_init + 5]
    axes[1, 1].plot(lims, lims, "k--", alpha=0.5, label="perfect recovery")
    axes[1, 1].set_xlim(lims)
    axes[1, 1].set_ylim(lims)
    axes[1, 1].set_xlabel("True $T_0$ [K]")
    axes[1, 1].set_ylabel("Recovered $T_0$ [K]")
    axes[1, 1].set_title("True vs. recovered (per grid cell)")
    axes[1, 1].legend()
    axes[1, 1].set_aspect("equal")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].semilogy(loss_history, "k.-", linewidth=1.5)
    axes[1, 2].set_xlabel("Checkpoint")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title(f"Convergence ({result_p2.nit} L-BFGS iterations)")
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(
        "Recovering a 900-element initial temperature field from 100 sensors",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(FIGURE_DIR / "part2_forensics.png")
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'part2_forensics.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Output directory: {FIGURE_DIR}")
    print()

    with Tesseract.from_image("enzyme-thermal-2d:latest") as t:
        generate_fd_convergence(t)
        print()
        generate_part1_convergence(t)
        print()
        generate_part2_forensics(t)

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
