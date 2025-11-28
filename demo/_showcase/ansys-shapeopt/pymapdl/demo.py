import os

import numpy as np

from tesseract_core import Tesseract

# Load the MAPDL server's address from your environment
# The MAPDL server is typically ran on a machine with ANSYS tools via
# usr/ansys_inc/v241/ansys/bin/ansys241 -port 50050  -grpc
host = os.getenv("MAPDL_HOST")
if host is None:
    raise ValueError(
        "Unable to read $MAPDL_HOST from the environment. "
        + "Use 'export MAPDL_HOST=X.X.X.X' for local IP address of your MAPDL Instance."
    )
port = os.getenv("MAPDL_PORT")
if port is None:
    raise ValueError(
        "Unable to read $MAPDL_PORT from the environment. "
        + "Use 'export MAPDL_PORT=X' for the port of your MAPDL Instance."
    )


def create_hex_mesh(Lx: float, Ly: float, Lz: float, Nx: int, Ny: int, Nz: int) -> dict:
    """Generate a structured hexahedral mesh for a rectangular domain.

    Args:
        Lx (float): Domain length in x-direction
        Ly (float): Domain length in y-direction
        Lz (float): Domain length in z-direction
        Nx (int): Number of elements in x-direction
        Ny (int): Number of elements in y-direction
        Nz (int): Number of elements in z-direction

    Returns:
        dict: Dictionary containing:
            - points (ndarray): Array of shape ((Nx+1)*(Ny+1)*(Nz+1), 3) with vertex coordinates
            - elems (ndarray): Array of shape (Nx*Ny*Nz, 8) with element connectivity
            - n_points (int): Total number of nodes
            - n_elems (int): Total number of elements
    """
    # Generate structured grid points
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    z = np.linspace(0, Lz, Nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack(
        [X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]
    )

    # Generate hexahedral connectivity in VTK convention
    n_elements = Nx * Ny * Nz
    connectivity = np.zeros((n_elements, 8), dtype=np.int32)

    elem_idx = 0
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                # Calculate node indices for this element
                # Node index = i + j*(Nx+1) + k*(Nx+1)*(Ny+1)
                n0 = i + j * (Nx + 1) + k * (Nx + 1) * (Ny + 1)
                n1 = n0 + 1
                n2 = n0 + (Nx + 1) + 1
                n3 = n0 + (Nx + 1)
                n4 = n0 + (Nx + 1) * (Ny + 1)
                n5 = n4 + 1
                n6 = n4 + (Nx + 1) + 1
                n7 = n4 + (Nx + 1)

                connectivity[elem_idx] = [n0, n1, n2, n3, n4, n5, n6, n7]
                elem_idx += 1

    hex_mesh = {
        "points": points,
        "elems": connectivity,
        "n_points": points.shape[0],
        "n_elems": n_elements,
    }
    return hex_mesh


def cantilever_bc(
    Lx: float, Ly: float, Lz: float, Nx: int, Ny: int, Nz: int, hex_mesh: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Define boundary conditions for a cantilever beam test case."""
    # Create a dirichlet_mask of nodes indices associated with dirichlet condition
    # dirichlet condition (select nodes at x=0)
    on_lhs = hex_mesh["points"][:, 0] <= 0
    dirichlet_mask = np.where(on_lhs)[0]  # size (num_dirichlet_nodes,)
    dirichlet_values = np.zeros(dirichlet_mask.shape[0])

    # von Neumann condition (select nodes at x=Lx with constraints on y and z)
    x_lim = Lx
    y_min = 0
    y_max = 0.2 * Ly
    z_min = 0.4 * Lz
    z_max = 0.6 * Lz
    von_neumann = np.logical_and(
        hex_mesh["points"][:, 0] >= x_lim,
        np.logical_and(
            np.logical_and(
                hex_mesh["points"][:, 1] >= y_min, hex_mesh["points"][:, 1] <= y_max
            ),
            np.logical_and(
                hex_mesh["points"][:, 2] >= z_min, hex_mesh["points"][:, 2] <= z_max
            ),
        ),
    )
    # A (num_von_neumann_nodes, n_dof) array
    von_neumann_mask = np.where(von_neumann)[0]
    von_neumann_values = np.array([0, 0.0, 0.001 / len(von_neumann_mask)]) + np.zeros(
        (von_neumann_mask.shape[0], 3)
    )
    return dirichlet_mask, dirichlet_values, von_neumann_mask, von_neumann_values


def sample_rho(hex_mesh: dict) -> np.ndarray:
    """Create a uniform density field for topology optimization."""
    # Create a test density field varying from 0 to 1
    n_elem = hex_mesh["n_elems"]
    rho = (np.arange(0, n_elem, 1) / n_elem).reshape((n_elem, 1))
    rho = 0.5 * np.ones((n_elem, 1))
    return rho


def main(tess_pymapdl: Tesseract) -> None:
    """Run the ANSYS MAPDL integration demo."""
    # Set the domain dimensions
    Lx, Ly, Lz = 3, 2, 1

    # Set the number of elements in each dimension
    Nx, Ny, Nz = 6, 4, 2

    hex_mesh = create_hex_mesh(Lx, Ly, Lz, Nx, Ny, Nz)
    dirichlet_mask, dirichlet_values, von_neumann_mask, von_neumann_values = (
        cantilever_bc(Lx, Ly, Lz, Nx, Ny, Nz, hex_mesh)
    )
    rho = sample_rho(hex_mesh)

    inputs = {
        "dirichlet_mask": dirichlet_mask,
        "dirichlet_values": dirichlet_values,
        "von_neumann_mask": von_neumann_mask,
        "von_neumann_values": von_neumann_values,
        "hex_mesh": dict(hex_mesh),
        "host": host,
        "port": port,
        "rho": rho,
        "E0": 1.0,
        "rho_min": 1e-6,
        "log_level": "DEBUG",
        "vtk_output": "pymapdl_simp_compliance.vtk",
    }

    outputs = tess_pymapdl.apply(inputs)

    # Verify relationship between compliance and strain energy
    strain_energy = outputs["strain_energy"]
    compliance = outputs["compliance"]
    total_strain_energy = np.sum(strain_energy)
    print(f"Compliance: {compliance:.6e}")
    print(f"Total Strain Energy: {total_strain_energy:.6e}")
    print(f"0.5 * Compliance: {0.5 * compliance:.6e}")
    print(f"Ratio (should be ~1.0): {total_strain_energy / (0.5 * compliance):.6f}")

    # a sample backwards pass
    vjp = tess_pymapdl.vector_jacobian_product(
        inputs, ["rho"], ["compliance"], {"compliance": 1.0}
    )
    sensitivity = vjp["rho"]
    print(f"The first five components of the sensitivity are\n{sensitivity[0:5]}\n...")


if __name__ == "__main__":
    # Initialize the Tesseract from image
    tess_pymapdl = Tesseract.from_image("pymapdl_tess")
    tess_pymapdl.serve()

    # Initialize Tesseract from api (if desired)
    # try:
    #     tess_pymapdl = Tesseract.from_tesseract_api("./tesseract_api.py")
    # except RuntimeError as e:
    #     raise RuntimeError(
    #         "Unable to load tesseract from api. "
    #         "Ensure that you have installed the build requirements using 'pip install -r tesseract_requirements.txt'"
    #     ) from e

    main(tess_pymapdl)

    tess_pymapdl.teardown()
