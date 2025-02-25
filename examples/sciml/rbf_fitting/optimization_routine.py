# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""This is an optimisation scripts for the RBF fitting example."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import random
from plotting import plot_loss, plot_rbf

from tesseract_core.sdk.tesseract import Tesseract

# JAX random keys
key = random.PRNGKey(42)
subkeys = random.split(key, 2)


# Initialize the RBF interpolation problem
def ground_truth(x):
    return (
        10 * jnp.exp(-((x - 0.25) ** 2) / 0.3**2)
        + 4 * jnp.sin(6 * jnp.pi * x)
        - 5 * x**4
    )


# Initialize the parameters
n_centers = 20
n_target = 50
x_centers = jnp.linspace(0.0, 1.0, n_centers)
weights_0 = random.normal(subkeys[0], (n_centers,))
length_scale = 0.05
x_target = jnp.linspace(0, 1, n_target)
y_target = ground_truth(x_target) + 0.75 * random.normal(subkeys[1], (n_target,))

inputs = {
    "x_centers": x_centers.tolist(),
    "weights": weights_0.tolist(),
    "length_scale": length_scale,
    "x_target": x_target.tolist(),
    "y_target": y_target.tolist(),
}

diff_inputs = ["weights"]
diff_outputs = ["mse"]

# Initialize optimizer
optimizer = optax.adam(learning_rate=0.5)
opt_state = optimizer.init(jnp.array(inputs["weights"]))

# Optimize the RBF interpolation
loss_history = []
max_iterations = 100
print(f"Starting the optimization process for {max_iterations} iterations.")

# Initialize Tesseract client
with Tesseract.from_image(image="rbf_fitting") as tess:
    for n_iteration in range(max_iterations):
        if n_iteration % 10 == 0:
            print(f" ---- iteration {n_iteration} / {max_iterations}")

        # Compute loss
        apply_response = tess.apply(inputs)
        loss = apply_response["mse"]
        loss_history.append(loss)

        # Compute gradients
        # Option 1: Use Jacobian
        # grad_weights = tess.jacobian(
        #     inputs,
        #     diff_inputs,
        #     diff_outputs,
        # )["mse"]["weights"]
        # Option 2: Use VJP
        grad_weights = tess.vector_jacobian_product(
            inputs, diff_inputs, diff_outputs, {"mse": 1.0}
        )["weights"]

        # Update weights
        weights = jnp.array(inputs["weights"])
        updates, opt_state = optimizer.update(grad_weights, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        inputs["weights"] = weights.tolist()

print("Optimisation completed!")

final_weights = jnp.array(inputs["weights"])

print("Plotting the results...")
plot_rbf(
    x_centers,
    weights_0,
    length_scale,
    x_target,
    y_target,
    title="RBF interpolation with random coefficients",
    ground_truth_func=ground_truth,
)
plt.show()
plot_rbf(
    x_centers,
    final_weights,
    length_scale,
    x_target,
    y_target,
    title="RBF interpolation with optimized coefficients",
    ground_truth_func=ground_truth,
)
plt.show()
plot_loss(loss_history)
plt.show()
