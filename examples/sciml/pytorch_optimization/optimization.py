# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tesseract_api import log_rosenbrock
from tesseract_client import Client

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", help="Port of tesseract server", required=True)
parser.add_argument("-n", "--niter", help="The number of iterations", default=300)
parser.add_argument(
    "-l",
    "--lr",
    help="The learning rate",
    default=1e-2,
)
parser.add_argument("-s", "--seed", help="The random seed", default=100)
parser.add_argument(
    "--jac_inputs",
    nargs="+",
    help="The variables to optimize, e.g. 'x y'",
    default=["x", "y"],
)
parser.add_argument(
    "--opt", help="The optimizer to use, ['SGD', 'Adam']", default="Adam"
)

args = vars(parser.parse_args())

port = int(args["port"])

torch.manual_seed(int(args["seed"]))

# initiate starting x, y
# note: even if only optimizing x (or y) we treat both as differentiable
# but only update the gradients of the optimizing parameter
x0 = torch.nn.Parameter(torch.randn(1) * 2)
y0 = torch.nn.Parameter(torch.randn(1) * 2)
jac_inputs = args["jac_inputs"]

# this defines the rosenbrock shape, global minimum at (x, y) = (a, a^2)
a = 1.0
b = 100.0

inputs = {
    "x": x0.detach().item(),
    "y": y0.detach().item(),
    "a": a,
    "b": b,
}

# Initialize Tesseract client
client = Client(host="127.0.0.1", port=port)

# setup optimizer: either Adam or SGD (default)
assert args["opt"] in ["SGD", "Adam"], "Only Adam or SGD optimizer supported."
opt = getattr(torch.optim, args["opt"])((x0, y0), lr=float(args["lr"]))

# perform iterations
losses = []
x = []
y = []
for _i in range(int(args["niter"])):
    # zero out the gradients
    opt.zero_grad()

    # get loss
    output = client.request("apply", method="POST", payload={"inputs": inputs})
    loss = output["loss"]["data"]["buffer"]

    # append values
    losses.append(loss)
    x.append(inputs["x"])
    y.append(inputs["y"])

    # get gradients
    jacobian_response = client.request(
        "jacobian",
        method="POST",
        payload={
            "inputs": inputs,
            "jac_inputs": jac_inputs,
            "jac_outputs": ["loss"],
        },
    )

    # insert gradients into pytorch tensors where optimizer expects them
    if "x" in jac_inputs:
        x0.grad = torch.as_tensor(
            jacobian_response["loss"]["x"]["data"]["buffer"]
        ).unsqueeze(0)
    if "y" in jac_inputs:
        y0.grad = torch.as_tensor(
            jacobian_response["loss"]["y"]["data"]["buffer"]
        ).unsqueeze(0)

    # make optimization step
    opt.step()

    # update inputs
    inputs["x"] = x0.detach().item()
    inputs["y"] = y0.detach().item()

# plotting
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), indexing="xy")

plt.imshow(log_rosenbrock(X, Y, a, b), extent=[-3, 3, 3, -3], cmap="Blues_r")
plt.colorbar()
plt.scatter(x, y, c=np.arange(len(x)))
plt.scatter([a], [a**2], marker="*", s=500, color="r")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(losses)
axes[0].set_title("loss")
axes[0].set_xlabel("iteration")
axes[1].plot(x)
axes[1].set_title("X")
axes[1].set_xlabel("iteration")
axes[1].axhline(a, c="r", lw=3, alpha=0.5)
axes[2].plot(y)
axes[2].set_title("Y")
axes[2].set_xlabel("iteration")
axes[2].axhline(a**2, c="r", lw=3, alpha=0.5)
plt.show()
