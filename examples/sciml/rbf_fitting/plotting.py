# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from matplotlib import pyplot as plt


def gaussian_rbf(x: float, c: float, length_scale: float) -> float:
    return jnp.exp(-((x - c) ** 2) / (2 * length_scale**2))


def plot_rbf(
    x_centers: jnp.ndarray,
    weights: jnp.ndarray,
    length_scale: float,
    x_target: jnp.ndarray,
    y_target: jnp.ndarray,
    title: str | None = None,
    ground_truth_func: Callable | None = None,
):
    # Initialize the RBF expansion result to zero
    x_plot = jnp.linspace(0, 1, 100)
    y_hat = 0
    # Compute the RBF expansion
    for coeff, center in zip(weights, x_centers, strict=False):
        y_hat += coeff * gaussian_rbf(x_plot, center, length_scale)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_plot, y_hat, label="RBF interpolation")
    ax.scatter(x_target, y_target, c="r", label="Target points")

    if ground_truth_func is not None:
        y_truth = ground_truth_func(x_plot)
        ax.plot(x_plot, y_truth, c="g", label="Ground truth")

    if title:
        ax.set_title(title)

    ax.legend()

    return fig, ax


def plot_loss(loss_history: Sequence[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history)
    ax.set_title("Optimisation loss history")
    ax.set_xlabel("Iteration")
