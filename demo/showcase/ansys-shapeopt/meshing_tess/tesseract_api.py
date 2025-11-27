from typing import Any

import jax
import jax.numpy as jnp

# import numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

#
# Schemata
#


class InputSchema(BaseModel):
    """Input schema for hexahedral mesh generation and field interpolation."""

    field_values: Differentiable[
        Array[
            (None, None, None),
            Float32,
        ]
    ] = Field(
        description=("Values defined on a regular grid that are to be differentiated.")
    )
    sizing_field: Array[
        (None, None, None),
        Float32,
    ] = Field(
        description=(
            "Sizing field values defined on a regular grid for mesh adaptation."
        )
    )
    domain_size: tuple[float, float, float] = Field(
        description=("Size of the domain in x, y, z directions.")
    )

    max_points: int = Field(
        default=10000,
        description=("Maximum number of points in the output hex mesh. "),
    )

    max_cells: int = Field(
        default=10000,
        description=("Maximum number of hexahedral cells in the output hex mesh. "),
    )

    max_subdivision_levels: int = Field(
        default=5,
        description=("Maximum number of subdivision levels for the hex mesh. "),
    )


class HexMesh(BaseModel):
    """Hexagonal mesh representation."""

    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    faces: Array[(None, 8), Int32] = Field(
        description="Array of hexahedral faces defined by indices into the points array."
    )
    n_points: Int32 = Field(
        default=0, description="Number of valid points in the points array."
    )
    n_faces: Int32 = Field(
        default=0, description="Number of valid faces in the faces array."
    )


class OutputSchema(BaseModel):
    """Output schema for hexahedral mesh generation and field interpolation."""

    mesh: HexMesh = Field(description="Hexagonal mesh representation of the geometry")
    mesh_cell_values: Differentiable[
        Array[
            (None,),
            Float32,
        ]
    ] = Field(description="Cell-centered values defined on the hexahedral mesh.")


#
# Helper functions
#


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


def vectorized_subdivide_hex_mesh(
    hex_cells: jnp.ndarray,  # (n_hex, 8)
    pts_coords: jnp.ndarray,  # (n_points, 3)
    mask: jnp.ndarray,  # (n_hex,) boolean array indicating which hexes to subdivide
    split_x: bool = True,
    split_y: bool = True,
    split_z: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized subdivision of HEX8 mesh.

    This method introduces duplicates of points that should later be merged.

    Hexahedron is constructed as follows:

          7 -------- 6
         /|         /|
        4 -------- 5 |
        | |        | |
        | 3 -------|-2
        |/         |/
        0 -------- 1

    Axis orientation:

        z  y
        | /
        |/____ x

    """
    # compute sizes
    n_hex_to_subdiv = mask.sum()
    n_hex_each_hex = (split_x + 1) * (split_y + 1) * (split_z + 1)
    n_points_per_hex = 8
    # 8 corners per new hex, 8 new hexes per old hex
    n_new_pts = n_points_per_hex * n_hex_each_hex * n_hex_to_subdiv
    n_new_cells = n_hex_each_hex * n_hex_to_subdiv

    new_pts_coords = jnp.zeros((n_new_pts, 3), dtype=pts_coords.dtype)
    new_hex_cells = jnp.zeros((n_new_cells, 8), dtype=hex_cells.dtype)

    # get sizes of hexes to subdivide
    hex_sizes = jnp.abs(pts_coords[hex_cells[mask, 6]] - pts_coords[hex_cells[mask, 0]])
    # Ceneter points of shape (n_hex_to_subdiv, 3)
    center_points = jnp.mean(pts_coords[hex_cells[mask]], axis=1)

    # Build cell offset tensor
    # that is the offset of a hex center to each of the new hex centers
    cell_offsets = jnp.zeros((1, n_hex_each_hex, 3), dtype=jnp.float32)
    index = 0
    for ix in range(split_x + 1):
        for iy in range(split_y + 1):
            for iz in range(split_z + 1):
                cell_offsets = cell_offsets.at[0, index].set(
                    jnp.array(
                        [
                            (-0.25 + ix * 0.5) if split_x else 0.0,
                            (-0.25 + iy * 0.5) if split_y else 0.0,
                            (-0.25 + iz * 0.5) if split_z else 0.0,
                        ]
                    ).T
                )
                index += 1

    # We now repeat the cell offsets and scale them by the corresponding hex sizes
    # Hence we have a cell_offset tensor of shape (n_hex_to_subdiv, n_hex_each_hex, 3)
    cell_offsets = cell_offsets.repeat(n_hex_to_subdiv, axis=0) * hex_sizes.reshape(
        (n_hex_to_subdiv, 1, 3)
    ).repeat(n_hex_each_hex, axis=1)

    # Build point offset tensor
    # that is the offset of a hex center to each of the new hex points
    offset_x = 0.25 if split_x else 0.5
    offset_y = 0.25 if split_y else 0.5
    offset_z = 0.25 if split_z else 0.5
    point_offsets = jnp.array(
        [
            [-offset_x, -offset_y, -offset_z],
            [offset_x, -offset_y, -offset_z],
            [offset_x, offset_y, -offset_z],
            [-offset_x, offset_y, -offset_z],
            [-offset_x, -offset_y, offset_z],
            [offset_x, -offset_y, offset_z],
            [offset_x, offset_y, offset_z],
            [-offset_x, offset_y, offset_z],
        ]
    )

    # Repeat the point offsets and scale them by the corresponding hex sizes
    # -> point_offset tensor of shape (n_hex_to_subdiv, n_points_per_hex, 3)
    point_offsets = point_offsets.reshape((1, n_points_per_hex, 3)).repeat(
        n_hex_to_subdiv, axis=0
    ) * hex_sizes.reshape((n_hex_to_subdiv, 1, 3)).repeat(n_points_per_hex, axis=1)

    # Repeat the two offsets at an additional axis to get all combinations
    cell_offsets = cell_offsets.reshape((n_hex_to_subdiv, n_hex_each_hex, 1, 3)).repeat(
        n_points_per_hex, axis=2
    )
    point_offsets = point_offsets.reshape(
        (n_hex_to_subdiv, 1, n_points_per_hex, 3)
    ).repeat(n_hex_each_hex, axis=1)

    # Compute total offset relative to old hex center
    # -> (n_hex_to_subdiv, n_hex_each_hex, n_points_per_hex, 3)
    total_offsets = cell_offsets + point_offsets

    # lets reshape the center points to broadcast
    center_points = (
        center_points.reshape((n_hex_to_subdiv, 1, 1, 3))
        .repeat(n_hex_each_hex, axis=1)
        .repeat(n_points_per_hex, axis=2)
    )

    # Directly compute new point coordinates and reshape
    new_pts_coords = (center_points + total_offsets).reshape((n_new_pts, 3))
    # Compute new hex cell indices
    new_hex_cells = jnp.arange(n_new_pts, dtype=jnp.int32).reshape(
        (n_new_cells, n_points_per_hex)
    )

    def reindex_and_mask(
        coords: jnp.ndarray, cells: jnp.ndarray, keep_mask: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Reindex points and cells based on mask."""
        # map mask to points
        point_mask = jnp.zeros(coords.shape[0], dtype=jnp.float32)
        point_mask = point_mask.at[cells.flatten()].add(keep_mask.repeat(8))

        # Reindex new points and cells based on mask
        index_offset = jnp.cumsum(jnp.logical_not(point_mask))
        cells = cells - index_offset.at[cells.flatten()].get().reshape(cells.shape)

        # apply mask to keep only subdivided hexes
        coords = coords.at[point_mask > 0].get()
        cells = cells.at[keep_mask].get()

        return coords, cells

    old_pts_coords, old_hex_cells = reindex_and_mask(
        pts_coords, hex_cells, jnp.logical_not(mask)
    )

    old_hex_cells = old_hex_cells + new_pts_coords.shape[0]

    combined_pts_coords = jnp.vstack([new_pts_coords, old_pts_coords])
    combined_hex_cells = jnp.vstack([new_hex_cells, old_hex_cells])

    return combined_pts_coords, combined_hex_cells


def remove_duplicate_points(
    pts_coords: jnp.ndarray, hex_cells: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Remove duplicate points from the mesh and update hex cell indices."""
    # TODO: remove rounding after removing duplicate points
    pts_coords = jnp.round(pts_coords, decimals=4)
    _, indices, inverse_indices = jnp.unique(
        pts_coords, axis=0, return_index=True, return_inverse=True
    )

    pts_coords = pts_coords[indices]

    hex_cells = inverse_indices[hex_cells]

    return pts_coords, hex_cells


def recursive_subdivide_hex_mesh(
    hex_cells: jnp.ndarray,
    pts_coords: jnp.ndarray,
    sizing_field: jnp.ndarray,
    levels: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recursively (unrolled) subdivide HEX8 mesh.

    Args:
        hex_cells: Initial hexahedral cells.
        pts_coords: Initial points coordinates.
        sizing_field: Sizing field values on a regular grid.
        levels: Maximum number of subdivision levels.
        Lx: Length of the domain in x direction.
        Ly: Length of the domain in y direction.
        Lz: Length of the domain in z direction.

    Returns:
        Subdivided points and hex cells.
    """
    xs = jnp.linspace(-Lx / 2, Lx / 2, sizing_field.shape[0])
    ys = jnp.linspace(-Ly / 2, Ly / 2, sizing_field.shape[1])
    zs = jnp.linspace(-Lz / 2, Lz / 2, sizing_field.shape[2])

    interpolator = RegularGridInterpolator(
        (xs, ys, zs), sizing_field, method="nearest", bounds_error=False, fill_value=-1
    )

    for i in range(levels):
        voxel_sizes = jnp.abs(pts_coords[hex_cells[:, 6]] - pts_coords[hex_cells[:, 0]])

        # voxel_center_points = jnp.mean(pts_coords[hex_cells], axis=1)
        sizing_values_pts = interpolator(pts_coords)
        voxel_sizing_min = jnp.min(sizing_values_pts[hex_cells], axis=1)

        subdivision_mask = jnp.max(voxel_sizes, axis=-1) > voxel_sizing_min

        if not jnp.any(subdivision_mask):
            print(f"No more subdivisions needed at level {i}.")
            break

        split_x = (
            voxel_sizes[subdivision_mask, :].max()
            / voxel_sizes[subdivision_mask, 0].mean()
            < 2
        )
        split_y = (
            voxel_sizes[subdivision_mask, :].max()
            / voxel_sizes[subdivision_mask, 1].mean()
            < 2
        )
        split_z = (
            voxel_sizes[subdivision_mask, :].max()
            / voxel_sizes[subdivision_mask, 2].mean()
            < 2
        )

        pts_coords, hex_cells = vectorized_subdivide_hex_mesh(
            hex_cells,
            pts_coords,
            subdivision_mask,
            split_x=split_x,
            split_y=split_y,
            split_z=split_z,
        )

        pts_coords, hex_cells = remove_duplicate_points(pts_coords, hex_cells)

    return pts_coords, hex_cells


# @lru_cache(maxsize=1)
def generate_mesh(
    Lx: float,
    Ly: float,
    Lz: float,
    sizing_field: jnp.ndarray,
    max_levels: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate adapted HEX8 mesh based on sizing field.

    Args:
        Lx: Length of the domain in x direction.
        Ly: Length of the domain in y direction.
        Lz: Length of the domain in z direction.
        sizing_field: Sizing field values on a regular grid.
        max_levels: Maximum number of subdivision levels.

    Returns:
        points: (n_points, 3) array of vertex positions.
        hex_cells: (n_hex, 8) array of hexahedron cell indices.
    """
    # get largest cell size
    max_size = jnp.max(sizing_field)

    Nx = max(int(Lx / max_size), 2)
    Ny = max(int(Ly / max_size), 2)
    Nz = max(int(Lz / max_size), 2)

    initial_pts, initial_hex_cells = hex_grid(Lx, Ly, Lz, Nx, Ny, Nz)

    pts, cells = recursive_subdivide_hex_mesh(
        initial_hex_cells,
        initial_pts,
        sizing_field,
        levels=max_levels,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
    )

    return pts, cells


def compute_integral_volume(grid):
    """Computes the integral volume (3D cumulative sum) of the grid.

    Args:
        grid: grid values
    """
    # We pad with one layer of zeros on the 'left' of every dimension.
    # This handles the boundary condition where a hex starts at index 0.
    # Cumulative sum along Depth, Height, and Width
    integral = jnp.cumsum(grid, axis=-1)
    integral = jnp.cumsum(integral, axis=-2)
    integral = jnp.cumsum(integral, axis=-3)

    # Pad with zeros at the beginning of each spatial dimension
    padding = [(0, 0)] * (grid.ndim - 3) + [(1, 0), (1, 0), (1, 0)]
    integral_padded = jnp.pad(integral, padding, mode="constant", constant_values=0)

    return integral_padded


def apply_fn(inputs: dict) -> dict:
    """Compute the compliance of the structure given a density field.

    Args:
        inputs: Dictionary containing input parameters and density field.

    Returns:
        Dictionary containing the compliance of the structure.
    """
    Lx = inputs["domain_size"][0]
    Ly = inputs["domain_size"][1]
    Lz = inputs["domain_size"][2]

    field_values = inputs["field_values"]
    max_points = inputs["max_points"]
    max_cells = inputs["max_cells"]
    sizing_field = inputs["sizing_field"]
    max_levels = inputs["max_subdivision_levels"]

    # no stop grads
    pts, cells = generate_mesh(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        sizing_field=sizing_field,
        max_levels=max_levels,
    )

    print("Done building mesh")

    pts_padded = jnp.zeros((max_points, 3), dtype=pts.dtype)
    pts_padded = pts_padded.at[: pts.shape[0], :].set(pts)
    cells_padded = jnp.zeros((max_cells, 8), dtype=cells.dtype)
    cells_padded = cells_padded.at[: cells.shape[0], :].set(cells)

    # Integral volumes trick for O(N) hex mesh integration
    # This works ONLY because all hexes are axis aligned

    # helper function to find indices on grid of mesh points
    def discretize(coord):
        coord = coord + jnp.array([Lx / 2, Ly / 2, Lz / 2])
        coord = coord / jnp.array([Lx, Ly, Lz])
        coord = coord * jnp.array([field_values.shape])
        return jnp.floor(coord).astype(jnp.int32)

    # discretized coordinates / indices
    coords_disc = jax.vmap(discretize, in_axes=0)(pts)[:, 0]

    # volume integral
    integral = compute_integral_volume(field_values)

    # obtain integral values at corresponding points
    ind = coords_disc[cells[:, 0]]
    cell_000 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    ind = coords_disc[cells[:, 1]]
    cell_100 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    ind = coords_disc[cells[:, 2]]
    cell_110 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    ind = coords_disc[cells[:, 3]]
    cell_010 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    ind = coords_disc[cells[:, 4]]
    cell_001 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    ind = coords_disc[cells[:, 5]]
    cell_101 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    ind = coords_disc[cells[:, 6]]
    cell_111 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    ind = coords_disc[cells[:, 7]]
    cell_011 = integral[ind[:, 0], ind[:, 1], ind[:, 2]]

    # sum up
    total_sum = (
        cell_111
        - cell_011
        - cell_101
        - cell_110
        + cell_001
        + cell_010
        + cell_100
        - cell_000
    )

    # Obtain discrete volume (number of grid points assigned to each hex)
    volume = jnp.prod(
        jnp.abs(coords_disc[cells[:, 6]] - coords_disc[cells[:, 0]]), axis=-1
    )
    volume = jnp.maximum(volume, 1.0)

    # Normalize sum by dividing through volume
    cell_values = total_sum / volume

    cell_values_padded = jnp.zeros((max_cells,), dtype=jnp.float32)
    cell_values_padded = cell_values_padded.at[: cell_values.shape[0]].set(cell_values)

    return {
        "mesh": {
            "points": pts_padded.astype(jnp.float32),
            "faces": cells_padded.astype(jnp.int32),
            "n_points": pts.shape[0],
            "n_faces": cells.shape[0],
        },
        "mesh_cell_values": cell_values_padded.astype(jnp.float32),
    }


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Compute the compliance of the structure given a density field."""
    return apply_fn(inputs.model_dump())


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute vector-Jacobian product for specified inputs and outputs.

    Args:
        inputs: InputSchema instance containing input parameters and density field.
        vjp_inputs: Set of input variable names for which to compute gradients.
        vjp_outputs: Set of output variable names with respect to which to compute gradients.
        cotangent_vector: Dictionary containing cotangent vectors for the specified outputs.

    Returns:
        Dictionary containing the vector-Jacobian product for the specified inputs.
    """
    assert vjp_inputs == {"field_values"}
    assert vjp_outputs == {"mesh_cell_values"}

    inputs = inputs.model_dump()

    filtered_apply = filter_func(apply_fn, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    out = vjp_func(cotangent_vector)[0]
    return out


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    """Calculate output shape of apply from the shape of its inputs."""
    return {
        "mesh_cell_values": ShapeDType(
            shape=(abstract_inputs.max_cells,),
            dtype="float32",
        ),
        "mesh": {
            "points": ShapeDType(
                shape=(abstract_inputs.max_points, 3), dtype="float32"
            ),
            "faces": ShapeDType(shape=(abstract_inputs.max_cells, 8), dtype="int32"),
            "n_points": ShapeDType(shape=(), dtype="int32"),
            "n_faces": ShapeDType(shape=(), dtype="int32"),
        },
    }
