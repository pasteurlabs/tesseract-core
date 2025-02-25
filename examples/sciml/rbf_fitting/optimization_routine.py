# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""This is an optimisation scripts for the RBF fitting example."""

import argparse

import jax.numpy as jnp
import optax
from jax import random
from matplotlib import pyplot as plt
from plotting import plot_loss, plot_rbf
from tesseract_client import Client

parser = argparse.ArgumentParser(description="RBF tesseract optimisation script")
parser.add_argument(
    "-p", "--port", help="Port at which RBF tesseract is being served", required=True
)
args = vars(parser.parse_args())

if "port" not in args:
    raise ValueError("Port at which RBF tesseract is being served is required.")

port = int(args["port"])

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

jac_inputs = ["weights", "length_scale"]
jac_outputs = ["mse"]

# Initialize Tesseract client
client = Client(host="127.0.0.1", port=port)

# Initialize optimizer
optimizer = optax.adam(learning_rate=0.5)
opt_state = optimizer.init(jnp.array(inputs["weights"]))

# Optimize the RBF interpolation
loss_history = []
max_iterations = 100
print(f"Starting the optimization process for {max_iterations} iterations.")
for n_iteration in range(max_iterations):
    if n_iteration % 10 == 0:
        print(f" ---- iteration {n_iteration} / {max_iterations}")

    # Compute loss
    apply_response = client.request("apply", method="POST", payload={"inputs": inputs})
    loss = apply_response["mse"]["data"]["buffer"]
    loss_history.append(loss)

    # Compute gradients
    jacobian_response = client.request(
        "jacobian",
        method="POST",
        payload={
            "inputs": inputs,
            "jac_inputs": jac_inputs,
            "jac_outputs": jac_outputs,
        },
    )

    buffer = jacobian_response["weights"]["mse"]["data"]["buffer"]
    grad_weights = jnp.array(buffer, dtype=jnp.float32)

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
