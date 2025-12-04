import numpy as np
import pyvista as pv
import trimesh
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float32

#
# Schemata
#


class InputSchema(BaseModel):
    """Input schema for bar geometry design and SDF generation."""

    differentiable_parameters: list[
        Array[
            (None,),
            Float32,
        ]
    ] = Field(
        description=(
            "Vertex positions of the bar geometry. "
            "The shape is (num_bars, num_vertices, 3), where num_bars is the number of bars "
            "and num_vertices is the number of vertices per bar. The last dimension represents "
            "the x, y, z coordinates of each vertex."
        )
    )

    non_differentiable_parameters: list[
        Array[
            (None,),
            Float32,
        ]
    ] = Field(description="Flattened array of non-differentiable geometry parameters.")

    static_parameters: list[list[int]] = Field(
        description=(
            "List of integers used to construct the geometry."
            " The first integer is the number of bars, and the second integer is the number of vertices per bar."
        )
    )

    string_parameters: list[str] = Field(
        description="List of string parameters for geometry construction."
    )


class TriangularMesh(BaseModel):
    """Triangular mesh representation with fixed-size arrays."""

    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    faces: Array[(None, 3), Float32] = Field(
        description="Array of triangular faces defined by indices into the points array."
    )


class OutputSchema(BaseModel):
    """Output schema for generated geometry."""

    meshes: list[TriangularMesh] = Field(
        description="Triangular meshes representing the geometries"
    )


def pyvista_to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    """Convert a pyvista mesh to a trimesh style polygon mesh."""
    points = mesh.points
    points_per_face = mesh.faces[0]
    n_faces = mesh.faces.shape[0] // (points_per_face + 1)

    faces = mesh.faces.reshape(n_faces, (points_per_face + 1))[:, 1:]

    return trimesh.Trimesh(vertices=points, faces=faces)


def build_geometries(
    differentiable_parameters: list[np.ndarray],
    non_differentiable_parameters: list[np.ndarray],
    static_parameters: list[list[int]],
    string_parameters: list[str],
) -> list[trimesh.Trimesh]:
    """Build a pyvista geometry from the parameters.

    The parameters are expected to be of shape (n_chains, n_edges_per_chain + 1, 3),
    """
    n_geometrics = len(differentiable_parameters)
    geometries = []
    for i in range(n_geometrics):
        n_chains = static_parameters[i][0]
        n_vertices_per_chain = static_parameters[i][1]
        geometry = []

        params = differentiable_parameters[i].reshape(
            (n_chains, n_vertices_per_chain, 3)
        )

        radius = non_differentiable_parameters[i][0]

        for chain in range(n_chains):
            tube = pv.Spline(points=params[chain]).tube(
                radius=radius, capping=True, n_sides=30
            )
            tube = tube.triangulate()
            tube = pyvista_to_trimesh(tube)
            geometry.append(tube)

        # convert each geometry in a trimesh style mesh and combine them
        mesh = geometry[0]

        for geom in geometry[1:]:
            mesh = mesh.union(geom)

        geometries.append(mesh)

    return geometries


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Generate mesh and SDF from bar geometry parameters.

    Args:
        inputs: Input schema containing bar geometry parameters.

    Returns:
        Output schema with generated mesh and SDF field.
    """
    meshes = build_geometries(
        differentiable_parameters=inputs.differentiable_parameters,
        non_differentiable_parameters=inputs.non_differentiable_parameters,
        static_parameters=inputs.static_parameters,
        string_parameters=inputs.string_parameters,
    )

    return OutputSchema(
        meshes=[
            TriangularMesh(
                points=mesh.vertices.astype(np.float32),
                faces=mesh.faces.astype(np.int32),
            )
            for mesh in meshes
        ],
    )
