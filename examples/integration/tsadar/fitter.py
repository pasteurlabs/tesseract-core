import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import ndarray
from scipy.optimize import minimize

from tesseract_core import Tesseract

tesseract_url = "http://localhost:8000"  # Change this to the correct address

tsadar = Tesseract(url=tesseract_url)

# Sample random true parameters
rng = np.random.default_rng(2408)
true_ne = rng.uniform(0.1, 0.9)
true_Te = rng.uniform(0.1, 1.5)
true_amp1 = rng.uniform(0.1, 3.5)
true_amp2 = rng.uniform(0.1, 3.5)
true_lam = rng.uniform(523, 528)

diff_wrt_to = ["ne", "Te", "amp1", "amp2", "lam"]
true_parameters = {
    "ne": true_ne,
    "Te": true_Te,
    "amp1": true_amp1,
    "amp2": true_amp2,
    "lam": true_lam,
}
true_electron_spectrum = tsadar.apply(true_parameters)["electron_spectrum"]


def to_numpy(x: dict[str, float]) -> np.ndarray:
    """Convert the parameter dictionary to a numpy array."""
    return np.array([x[k] for k in diff_wrt_to])


def to_dict(params: ndarray) -> dict:
    """Convert the numpy array to a parameter dictionary."""
    return {
        "ne": params[0],
        "Te": params[1],
        "amp1": params[2],
        "amp2": params[3],
        "lam": params[4],
    }


def mse(pred: ndarray, true: ndarray) -> float:
    """Mean Squared Error."""
    mse = np.mean(np.square(pred - true))
    return mse


def jacobian(parameters: np.ndarray, true_electron_spectrum: np.ndarray) -> np.ndarray:
    """Compute the gradient of the MSE loss function with respect to the parameters."""
    # Compute the gradient
    jacobian = tsadar.jacobian(to_dict(parameters), jac_inputs, jac_outputs)[
        "electron_spectrum"
    ]

    # Compute the primal
    electron_spectrum = tsadar.apply(to_dict(parameters))["electron_spectrum"]

    # Propagate the gradient through the model by differentiating the mse function
    error = electron_spectrum - true_electron_spectrum
    grad = {}
    for k in diff_wrt_to:
        grad[k] = 2 * np.mean(jacobian[k] * error)

    return to_numpy(grad)


# create an initial guess for the parameters
this_rng = np.random.default_rng(363)
init_ne = this_rng.uniform(0.1, 0.9)
init_Te = this_rng.uniform(0.1, 1.5)
init_amp1 = this_rng.uniform(0.1, 3.5)
init_amp2 = this_rng.uniform(0.1, 3.5)
init_lam = this_rng.uniform(523, 528)
parameters = {
    "ne": init_ne,
    "Te": init_Te,
    "amp1": init_amp1,
    "amp2": init_amp2,
    "lam": init_lam,
}
parameters = to_numpy(parameters)

# define the inputs and outputs for the Jacobian endpoint
jac_inputs = ["ne", "Te", "amp1", "amp2", "lam"]
jac_outputs = ["electron_spectrum"]

trajectory = []

electron_spectrum = tsadar.apply(to_dict(parameters))["electron_spectrum"]

trajectory.append(electron_spectrum)


def callback(xk):
    electron_spectrum = tsadar.apply(to_dict(xk))["electron_spectrum"]
    trajectory.append(electron_spectrum)
    print(f"loss: {mse(electron_spectrum, true_electron_spectrum)}")


res = minimize(
    lambda x: mse(
        tsadar.apply(to_dict(x))["electron_spectrum"], true_electron_spectrum
    ),
    jac=lambda x: jacobian(x, true_electron_spectrum),
    x0=parameters,
    method="L-BFGS-B",
    options={"maxiter": 200, "maxls": 10},
    callback=callback,
)

n = len(trajectory)

optim_steps = np.linspace(0, n, n + 1)

# repeat last trajectory point
for _ in range(10):
    trajectory.append(trajectory[-1])
    optim_steps = np.append(optim_steps, n)
fig, ax = plt.subplots()


def update(i):
    ax.clear()
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Optimization step {int(optim_steps[i])}")

    ax.plot(trajectory[i], label="Fit")
    ax.plot(true_electron_spectrum, label="True")
    ax.legend()
    ax.grid()


ani = FuncAnimation(fig, update, frames=len(trajectory), repeat=False)
ani.save("fit_trajectory.gif", writer="imagemagick", fps=3)
