import os
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import meshio
from jax_fem.generate_mesh import Mesh

# Import JAX-FEM specific modules
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, "data")
#
# Schemata
#


class HexMesh(BaseModel):
    """Hexahedral mesh representation."""

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


class InputSchema(BaseModel):
    """Input schema for topology optimization using hexahedral mesh."""

    rho: Differentiable[
        Array[
            (
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="2D density field for topology optimization")
    van_neumann_mask: Array[(None,), Int32] = Field(
        description="Mask for van Neumann boundary conditions",
    )
    van_neumann_values: Array[(None, None), Float32] = Field(
        description="Values for van Neumann boundary conditions",
    )
    dirichlet_mask: Array[(None,), Int32] = Field(
        description="Mask for Dirichlet boundary conditions",
    )
    dirichlet_values: Array[(None,), Float32] = Field(
        description="Values for Dirichlet boundary conditions",
    )
    hex_mesh: HexMesh = Field(
        description="Hexahedral mesh representation of the geometry",
    )


class OutputSchema(BaseModel):
    """Output schema for topology optimization using hexahedral mesh."""

    compliance: Differentiable[
        Array[
            (),
            Float32,
        ]
    ] = Field(description="Compliance of the structure, a measure of stiffness")


#     displacement: Array[
#         (None, 3),
#         Float32,
#     ] = Field(description="Nodal displacement field")

#
# Helper functions
#


# Define constitutive relationship
# Adapted from JAX-FEM
# https://github.com/deepmodeling/jax-fem/blob/1bdbf060bb32951d04ed9848c238c9a470fee1b4/demos/topology_optimization/example.py
class Elasticity(Problem):
    """Linear elasticity problem with custom constitutive law."""

    def custom_init(self, van_neumann_value_fns: list[Callable]) -> None:
        """Initialize custom problem parameters.

        Args:
            van_neumann_value_fns: List of functions for van Neumann boundary conditions.
        """
        self.fe = self.fes[0]
        self.fe.flex_inds = jnp.arange(len(self.fe.cells))

        self.van_neumann_value_fns = van_neumann_value_fns

    def get_tensor_map(self) -> Callable:
        """Get the stress-strain constitutive relationship tensor map.

        Returns:
            Callable that computes stress from strain gradient and density.
        """

        def stress(u_grad, theta):
            Emax = 70.0e3
            Emin = 1e-3 * Emax
            penal = 3.0

            E = Emin + (Emax - Emin) * theta[0] ** penal

            nu = 0.3
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

            epsilon = 0.5 * (u_grad + u_grad.T)

            sigma = lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2.0 * mu * epsilon
            return sigma

        return stress

    def get_surface_maps(self) -> list[Callable]:
        """Get surface traction boundary condition functions.

        Returns:
            List of van Neumann boundary condition value functions.
        """
        return self.van_neumann_value_fns

    def set_params(self, params: jnp.ndarray) -> None:
        """Set density parameters for topology optimization.

        Args:
            params: Density field array for the flexible elements.
        """
        # Override base class method.
        full_params = jnp.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = jnp.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol: jnp.ndarray) -> jnp.ndarray:
        """Compute structural compliance via surface integral.

        Args:
            sol: Solution displacement field.

        Returns:
            Compliance value (scalar).
        """
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        u_face = (
            sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :]
            * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        )
        u_face = jnp.sum(u_face, axis=2)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
        val = jnp.sum(traction * u_face * nanson_scale[:, :, None])
        return val


# Memoize the setup function to avoid expensive recomputation
# @lru_cache(maxsize=1)
def setup(
    pts: jnp.ndarray = None,
    cells: jnp.ndarray = None,
    dirichlet_mask: jnp.ndarray = None,
    dirichlet_values: jnp.ndarray = None,
    van_neumann_mask: jnp.ndarray = None,
    van_neumann_values: jnp.ndarray = None,
) -> tuple[Elasticity, Callable]:
    """Setup the elasticity problem and its differentiable solver.

    Args:
        pts: Optional array of mesh vertex positions for custom mesh.
        cells: Optional array of hexahedral cell definitions for custom mesh.
        dirichlet_mask: Mask array for Dirichlet boundary conditions.
        dirichlet_values: Values array for Dirichlet boundary conditions.
        van_neumann_mask: Mask array for van Neumann boundary conditions.
        van_neumann_values: Values array for van Neumann boundary conditions.

    Returns:
        Tuple of (problem, fwd_pred) where problem is the configured Elasticity
        problem instance and fwd_pred is the differentiable forward solver.
    """
    ele_type = "HEX8"
    meshio_mesh = meshio.Mesh(points=pts, cells={"hexahedron": cells})
    mesh = Mesh(pts, meshio_mesh.cells_dict["hexahedron"])

    def bc_factory(
        masks: jnp.ndarray,
        values: jnp.ndarray,
        is_van_neumann: bool = False,
    ) -> tuple[list[Callable], list[Callable]]:
        location_functions = []
        value_functions = []

        for i in range(values.shape[0]):
            # Create a factory that captures the current value of i
            def make_location_fn(idx):
                def location_fn(point, index):
                    return (
                        jnp.sum(
                            jax.lax.dynamic_index_in_dim(
                                masks, index, 0, keepdims=False
                            )
                        )
                        == idx + 1
                    ).astype(jnp.bool_)

                return location_fn

            def make_value_fn(idx):
                def value_fn(point):
                    return values[idx]

                return value_fn

            def make_value_fn_vn(idx):
                def value_fn_vn(u, x):
                    return values[idx]

                return value_fn_vn

            location_functions.append(make_location_fn(i))
            value_functions.append(
                make_value_fn_vn(i) if is_van_neumann else make_value_fn(i)
            )

        return location_functions, value_functions

    dirichlet_mask = jnp.array(dirichlet_mask)
    van_neumann_mask = jnp.array(van_neumann_mask)

    dirichlet_location_fns, dirichlet_value_fns = bc_factory(
        dirichlet_mask, dirichlet_values
    )

    van_neumann_locations, van_neumann_value_fns = bc_factory(
        van_neumann_mask, van_neumann_values, is_van_neumann=True
    )

    dirichlet_bc_info = [dirichlet_location_fns * 3, [0, 1, 2], dirichlet_value_fns * 3]

    location_fns = van_neumann_locations

    # Define forward problem
    problem = Elasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        additional_info=(van_neumann_value_fns,),
        # additional_info=([0.1],),
    )

    # Apply the automatic differentiation wrapper
    # This is a critical step that makes the problem solver differentiable
    fwd_pred = ad_wrapper(
        problem,
        solver_options={"umfpack_solver": {}},
        adjoint_solver_options={"umfpack_solver": {}},
    )
    return problem, fwd_pred


def apply_fn(inputs: dict) -> dict:
    """Compute the compliance of the structure given a density field.

    Args:
        inputs: Dictionary containing input parameters and density field.

    Returns:
        Dictionary containing the compliance of the structure.
    """
    # no stop grads
    problem, fwd_pred = setup(
        pts=inputs["hex_mesh"]["points"][: inputs["hex_mesh"]["n_points"]],
        cells=inputs["hex_mesh"]["faces"][: inputs["hex_mesh"]["n_faces"]],
        dirichlet_mask=inputs["dirichlet_mask"],
        dirichlet_values=inputs["dirichlet_values"],
        van_neumann_mask=inputs["van_neumann_mask"],
        van_neumann_values=inputs["van_neumann_values"],
    )

    rho = inputs["rho"][: inputs["hex_mesh"]["n_faces"]]

    sol_list = fwd_pred(rho)
    compliance = problem.compute_compliance(sol_list[0])

    return {"compliance": compliance.astype(jnp.float32)}


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
    assert vjp_inputs == {"rho"}
    assert vjp_outputs == {"compliance"}

    inputs = inputs.model_dump()

    filtered_apply = filter_func(apply_fn, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    out = vjp_func(cotangent_vector)[0]
    return out


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Calculate output shape of apply from the shape of its inputs."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
