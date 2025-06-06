import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import Normalize


def make_animation(constraints,
                   sols,
                   x_true,
                   Q, q):
    num_points = 100
    xlim = (-2, 0)
    x1 = np.linspace(xlim[0], xlim[1], num_points)
    x2 = np.linspace(xlim[0], xlim[1], num_points)
    X1, X2 = np.meshgrid(x1, x2)
    batched_x = np.stack([X1.flatten(), X2.flatten()], axis=-1)
    batched_obj = 0.5 * np.einsum('ij, ji->i',(batched_x @ Q), batched_x.T) + batched_x @ q
    Z = batched_obj.reshape(X1.shape)

    x_0 = np.linspace(xlim[0], xlim[1], num_points)

    fig, ax = plt.subplots(figsize=(8, 6))
    (cline,) = ax.plot([], [], color="black",
                        path_effects=[patheffects.withTickedStroke(spacing=7, angle=135)],
                        label='current constraint')

    levels = np.linspace(-1, 1, 50)
    ax.contourf(X1, X2, Z, levels=levels, cmap="viridis", norm=Normalize(vmin=Z.min(), vmax=Z.max()))
    ax.scatter(sols[0][0], sols[0][1], marker="o", s=100, color="black", label="current solution")
    ax.scatter(
        x_true[0], x_true[1], marker="*", s=200, color="r", label="target solution"
    )
    ax.legend()
    ax.grid()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    def init():
        cline.set_data([], [])
        return (cline, )
    def update(frame):
        G, h = constraints[frame]
        x_1 = (h - G[:, 0] * x_0) / G[:, 1]
        cline.set_data(x_0[np.where((x_1 > xlim[0]) & (x_1 < xlim[1]))],
                      x_1[np.where((x_1 > xlim[0]) & (x_1 < xlim[1]))])
        ax.scatter(sols[frame][0], sols[frame][1], marker="o", s=100, color="black")
        return (cline,)

    return animation.FuncAnimation(
        fig, update, frames=len(constraints), init_func=init, interval=100, blit=True, repeat=False
    )
