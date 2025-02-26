import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def animate(trajectory : list[np.ndarray], true_electron_spectrum: np.ndarray):
    
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
