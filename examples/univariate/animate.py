import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def make_animation(xs, ys):
    """Animate optimization trajectory on a countour of the rosenbrock function."""
    # Generate a grid of points for the contour plot
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-0.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.logspace(-2.5, 3.5, 10)
    ax.contourf(X, Y, Z, levels=levels, norm=LogNorm(), cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(1, 1, "*", markersize=15, label="Global Minimum", color="black")
    (line,) = ax.plot([], [], ".-", markersize=5, label="Trajectory", color="black")
    ax.legend()
    plt.tight_layout()

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(xs[: frame + 1], ys[: frame + 1])
        return (line,)

    return animation.FuncAnimation(
        fig, update, frames=len(xs), init_func=init, interval=300, blit=True
    )
