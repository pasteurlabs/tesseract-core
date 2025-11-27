from collections.abc import Sequence
from typing import TypeVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mmapy
import numpy as np
import pyvista as pv
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_mesh(
    mesh: dict,
    bounds: Sequence[float],
    save_path: str | None = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot a 3D triangular mesh with boundary conditions visualization.

    Args:
        mesh: Dictionary containing 'points' and 'faces' arrays.
        save_path: Optional path to save the plot as an image file.
        bounds: bounds of the 3D space.
        figsize: size of the matplotlib figure.
    """
    Lx = bounds[0]
    Ly = bounds[1]
    Lz = bounds[2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        mesh["points"][:, 0],
        mesh["points"][:, 1],
        mesh["points"][:, 2],
        triangles=mesh["faces"],
        alpha=0.7,
        antialiased=True,
        color="lightblue",
        edgecolor="black",
    )

    ax.set_xlim(-Lx / 2, Lx / 2)
    ax.set_ylim(-Ly / 2, Ly / 2)
    ax.set_zlim(-Lz / 2, Lz / 2)

    # set equal aspect ratio
    ax.set_box_aspect(
        (
            (Lx) / (Ly),
            1,
            (Lz) / (Ly),
        )
    )

    ax.set_zticks([])

    # x axis label
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Tighten layout to reduce whitespace
    plt.subplots_adjust(
        left=0.05,
        right=0.95,  # Adjust as needed
        bottom=0.05,
        top=0.95,  # Adjust as needed
        wspace=0.1,
        hspace=0.1,
    )

    if save_path:
        # avoid showing the plot in notebook
        plt.savefig(save_path)
        plt.close(fig)


def plot_grid_slice(field_slice, extent, ax, title, xlabel, ylabel):
    im = ax.imshow(field_slice.T, extent=extent, origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, orientation="vertical")
    return im


def plot_grid(field, Lx, Ly, Lz, Nx, Ny, Nz, title="SDF"):
    _, axs = plt.subplots(1, 3, figsize=(15, 5))

    plot_grid_slice(
        field[Nx // 2, :, :],
        extent=(-Ly / 2, Ly / 2, -Lz / 2, Lz / 2),
        ax=axs[0],
        title=f"{title} slice at x=0",
        xlabel="y",
        ylabel="z",
    )
    plot_grid_slice(
        field[:, Ny // 2, :],
        extent=(-Lx / 2, Lx / 2, -Lz / 2, Lz / 2),
        ax=axs[1],
        title=f"{title} slice at y=0",
        xlabel="x",
        ylabel="z",
    )
    plot_grid_slice(
        field[:, :, Nz // 2],
        extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
        ax=axs[2],
        title=f"{title} slice at z=0",
        xlabel="x",
        ylabel="y",
    )


T = TypeVar("T")


def stop_grads_int(x: T) -> T:
    """Stops gradient computation.

    We cannot use jax.lax.stop_gradient directly because Tesseract meshes are
    nested dictionaries with arrays and integers, and jax.lax.stop_gradient
    does not support integers.

    Args:
        x: Input value.

    Returns:
        Value with stopped gradients.
    """

    def stop(x):
        return jax._src.ad_util.stop_gradient_p.bind(x)

    return jax.tree_util.tree_map(stop, x)


def hex_to_pyvista(
    pts: jax.typing.ArrayLike, faces: jax.typing.ArrayLike, cell_data: dict
) -> pv.UnstructuredGrid:
    """Convert hex mesh defined by points and faces into a PyVista UnstructuredGrid.

    Args:
        pts: Array of point coordinates, shape (N, 3).
        faces: Array of hexahedral cell connectivity, shape (M, 8).
        cell_data: additional cell center data.

    Returns:
        PyVista mesh representing the hexahedral grid.
    """
    pts = np.array(pts)
    faces = np.array(faces)

    # Define the cell type for hexahedrons (VTK_HEXAHEDRON = 12)
    cell_type = pv.CellType.HEXAHEDRON
    cell_types = np.array([cell_type] * faces.shape[0], dtype=np.uint8)

    # Prepare the cells array: [number_of_points, i0, i1, i2, i3, i4, i5, i6, i7]
    n_cells = faces.shape[0]
    cells = np.empty((n_cells, 9), dtype=np.int64)
    cells[:, 0] = 8  # Each cell has 8 points
    cells[:, 1:9] = faces

    # Flatten the cells array for PyVista
    cells = cells.flatten()

    mesh = pv.UnstructuredGrid(cells, cell_types, pts)

    # Add cell data
    for name, data in cell_data.items():
        mesh.cell_data[name] = data

    return mesh


def hex_grid(
    Lx: float, Ly: float, Lz: float, Nx: int, Ny: int, Nz: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Creates a hex mesh with Nx * Ny * Nz points.

    This is (Nx-1) * (Ny-1) * (Nz-1) cells
    """
    xs = jnp.linspace(-Lx / 2, Lx / 2, Nx)
    ys = jnp.linspace(-Ly / 2, Ly / 2, Ny)
    zs = jnp.linspace(-Lz / 2, Lz / 2, Nz)

    xs, ys, zs = jnp.meshgrid(xs, ys, zs, indexing="ij")

    pts = jnp.stack((xs, ys, zs), -1)

    points_inds = jnp.arange(Nx * Ny * Nz)
    points_inds_xyz = points_inds.reshape(Nx, Ny, Nz)
    inds1 = points_inds_xyz[:-1, :-1, :-1]
    inds2 = points_inds_xyz[1:, :-1, :-1]
    inds3 = points_inds_xyz[1:, 1:, :-1]
    inds4 = points_inds_xyz[:-1, 1:, :-1]
    inds5 = points_inds_xyz[:-1, :-1, 1:]
    inds6 = points_inds_xyz[1:, :-1, 1:]
    inds7 = points_inds_xyz[1:, 1:, 1:]
    inds8 = points_inds_xyz[:-1, 1:, 1:]

    cells = jnp.stack(
        (inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), axis=-1
    ).reshape(-1, 8)

    return pts.reshape(-1, 3), cells


class MMAOptimizer:
    """A wrapper for the MMA optimizer from mmapy.

    Source is github.com/arjendeetman/GCMMA-MMA-Python.
    mmapy is a pretty barebones implementation of MMA in python. It should work for now.
    Alternatives to consider:
    - github.com/LLNL/pyMMAopt
    - pyopt.org/reference/optimizers.mma.html.

    """

    def __init__(
        self,
        x_init: jax.typing.ArrayLike,
        x_min: jax.typing.ArrayLike,
        x_max: jax.typing.ArrayLike,
        num_constraints: jax.typing.ArrayLike,
        constraint_scale: jax.typing.ArrayLike = 1000.0,
        x_update_limit: jax.typing.ArrayLike = 0.1,
    ) -> None:
        self.n = x_init.shape[0]
        self.m = num_constraints
        self.__check_input_sizes(x_init, x_min, x_max)

        # follow the original MMA variable names...
        self.asyinit = 0.5
        self.asyincr = 1.2
        self.asydecr = 0.7
        self.objective_scale: float = 100.0
        self.objective_scale_factor: float = 1.0

        self.eeen = np.ones((self.n, 1))
        self.eeem = np.ones((self.m, 1))
        self.zeron = np.zeros((self.n, 1))
        self.zerom = np.zeros((self.m, 1))

        self.xval = x_init
        self.xold1 = self.xval.copy()
        self.xold2 = self.xval.copy()
        self.x_min = x_min
        self.x_max = x_max
        self.low = self.x_min.copy()
        self.upp = self.x_max.copy()
        self.c = constraint_scale + self.zerom.copy()
        self.d = self.zerom.copy()
        self.a0 = 1
        self.a = self.zerom.copy()
        self.move = x_update_limit

    def calculate_next_x(
        self,
        objective_value: jax.typing.ArrayLike,
        objective_gradient: jax.typing.ArrayLike,
        constraint_values: jax.typing.ArrayLike,
        constraint_gradients: jax.typing.ArrayLike,
        iteration: int,
        x: jax.typing.ArrayLike,
        x_min: jax.typing.ArrayLike = None,
        x_max: jax.typing.ArrayLike = None,
    ) -> jax.typing.ArrayLike:
        if iteration < 1:
            raise Exception("The MMA problem expects an iteration count >= 1.")

        # The MMA problem works best with an objective scaled around [1, 100]
        if iteration == 1:
            self.objective_scale_factor = np.abs(self.objective_scale / objective_value)
        objective_value *= self.objective_scale_factor
        objective_gradient = (
            np.asarray(objective_gradient) * self.objective_scale_factor
        )

        # the bounds dont necessarily change every iteration
        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max

        # verify indputs
        x = self.__preprocess_jnp_array(x)
        x_min = self.__preprocess_jnp_array(x_min)
        x_max = self.__preprocess_jnp_array(x_max)
        constraint_values = self.__preprocess_jnp_array(constraint_values)
        objective_gradient = self.__preprocess_jnp_array(objective_gradient)
        self.__check_input_sizes(
            x,
            x_min,
            x_max,
            objective_gradient=objective_gradient,
            constraint_values=constraint_values,
            constraint_gradients=constraint_gradients,
        )

        # calculate the next iteration of x
        xmma, _ymma, _zmma, _lam, _xsi, _eta, _mu, _zet, _s, low, upp = mmapy.mmasub(
            self.m,
            self.n,
            iteration,
            x,
            x_min,
            x_max,
            self.xold1,
            self.xold2,
            objective_value,
            objective_gradient,
            constraint_values,
            constraint_gradients,
            self.low,
            self.upp,
            self.a0,
            self.a,
            self.c,
            self.d,
            move=self.move,
            asyinit=self.asyinit,
            asyincr=self.asyincr,
            asydecr=self.asydecr,
        )
        # update internal copies for mma
        self.xold2 = self.xold1.copy()
        self.xold1 = self.xval.copy()
        self.xval = xmma.copy()
        self.low = low
        self.upp = upp

        return xmma

    def __preprocess_jnp_array(
        self,
        x: jax.typing.ArrayLike,
    ) -> np.array:
        np_x = np.array(x)
        if len(np_x.shape) == 1:
            np_x = np_x[:, None]
        return np_x

    def __check_input_sizes(
        self,
        x: jax.typing.ArrayLike,
        x_min: jax.typing.ArrayLike,
        x_max: jax.typing.ArrayLike,
        objective_gradient: jax.typing.ArrayLike = None,
        constraint_values: jax.typing.ArrayLike = None,
        constraint_gradients: jax.typing.ArrayLike = None,
    ) -> None:
        def check_shape(shape: tuple, expected_shape: tuple, name: str) -> None:
            if (len(shape) == 1) or (
                shape[0] != expected_shape[0] or shape[1] != expected_shape[1]
            ):
                raise TypeError(
                    f"MMAError: The '{name}' was expected to have shape {expected_shape} but has shape {shape}."
                )

        check_shape(x.shape, (self.n, 1), "parameter vector")
        check_shape(x_min.shape, (self.n, 1), "parameter minimum bound vector")
        check_shape(x_max.shape, (self.n, 1), "parameter maximum bound vector")
        if objective_gradient is not None:
            check_shape(objective_gradient.shape, (self.n, 1), "objective gradient")
        if constraint_values is not None:
            check_shape(constraint_values.shape, (self.m, 1), "constraint values")
        if constraint_gradients is not None:
            check_shape(
                constraint_gradients.shape, (self.m, self.n), "constraint gradients"
            )
