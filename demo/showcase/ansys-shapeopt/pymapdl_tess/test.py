import os

import numpy as np
import pyvista as pv
from tesseract_api import HexMesh
from tesseract_core import Tesseract

# load the MAPDL server's address from your environment
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


# Initialize Tesseract from api (for testing)
try:
    tess_simp_compliance = Tesseract.from_tesseract_api("./tesseract_api.py")
except RuntimeError as e:
    raise RuntimeError(
        "Unable to load tesseract from api. "
        "Ensure that you have installed the build requirements using 'pip install -r tesseract_requirements.txt'"
    ) from e


def mesh_from_pyvista(Lx, Ly, Lz, Nx, Ny, Nz):
    grid = pv.ImageData(
        dimensions=np.array((Nx, Ny, Nz)) + 1,
        origin=(0, 0, 0),
        spacing=(Lx / Nx, Ly / Ny, Lz / Nz),  # These are the cell sizes along each axis
    )
    # repeated casts will eventually expose cell_connectivitiy
    mesh = (
        grid.cast_to_structured_grid()
        .cast_to_explicit_structured_grid()
        .cast_to_unstructured_grid()
    )

    hex_mesh = HexMesh(
        points=mesh.points,
        faces=mesh.cell_connectivity.reshape((Nx * Ny * Nz, 8)),
        n_points=mesh.points.shape[0],
        n_faces=Nx * Ny * Nz,
    )
    return hex_mesh


def cantilever_bc(Lx, Ly, Lz, Nx, Ny, Nz, hex_mesh):
    # Create a dirichlet_mask of nodes indices associated with diricelet condition
    # dirichlet condition (select nodes at x=0)
    on_lhs = hex_mesh.points[:, 0] <= 0
    dirichlet_mask = np.where(on_lhs)[0]  # size (num_dirichlet_nodes,)
    dirichlet_values = np.zeros(dirichlet_mask.shape[0])

    # von Neumann condition (select nodes at x=Lx with constraints on y and z)
    x_lim = Lx
    y_min = 0
    y_max = 0.2 * Ly
    z_min = 0.4 * Lz
    z_max = 0.6 * Lz
    von_neumann = np.logical_and(
        hex_mesh.points[:, 0] >= x_lim,
        np.logical_and(
            np.logical_and(
                hex_mesh.points[:, 1] >= y_min, hex_mesh.points[:, 1] <= y_max
            ),
            np.logical_and(
                hex_mesh.points[:, 2] >= z_min, hex_mesh.points[:, 2] <= z_max
            ),
        ),
    )
    # A (num_von_neumann_nodes, n_dof) array
    von_neumann_mask = np.where(von_neumann)[0]
    von_neumann_values = np.array([0, 0.0, 0.1 / len(von_neumann_mask)]) + np.zeros(
        (von_neumann_mask.shape[0], 3)
    )
    return dirichlet_mask, dirichlet_values, von_neumann_mask, von_neumann_values


def sample_rho(hex_mesh):
    # Create a test density field varying from 0 to 1
    n_elem = hex_mesh.n_faces
    rho = (np.arange(0, n_elem, 1) / n_elem).reshape((n_elem, 1))
    rho = 0.5 * np.ones((n_elem, 1))
    return rho


def main():
    Lx, Ly, Lz = 3, 2, 1
    Nx, Ny, Nz = 60, 40, 20
    Nx, Ny, Nz = 6, 4, 2
    num_tests = 0  # set to 0 if you don't want to run this check
    run_central_difference = True

    hex_mesh = mesh_from_pyvista(Lx, Ly, Lz, Nx, Ny, Nz)
    dirichlet_mask, dirichlet_values, von_neumann_mask, von_neumann_values = (
        cantilever_bc(Lx, Ly, Lz, Nx, Ny, Nz, hex_mesh)
    )
    rho = sample_rho(hex_mesh)

    inputs = {
        "dirichlet_mask": dirichlet_mask,
        "dirichlet_values": dirichlet_values,
        "van_neumann_mask": von_neumann_mask,
        "van_neumann_values": von_neumann_values,
        "hex_mesh": dict(hex_mesh),
        "host": host,
        "port": port,
        "rho": rho,
        "E0": 1.0,
        "rho_min": 1e-6,
        "log_level": "DEBUG",
        "vtk_output": "mesh_density.vtk",
    }

    outputs = tess_simp_compliance.apply(inputs)

    # Verify relationship between compliance and strain energy
    # For static analysis: Total Strain Energy = 0.5 * Compliance
    strain_energy = outputs["strain_energy"]
    compliance = outputs["compliance"]
    total_strain_energy = np.sum(strain_energy)
    print(f"\nCompliance: {compliance:.6e}")
    print(f"Total Strain Energy: {total_strain_energy:.6e}")
    print(f"0.5 * Compliance: {0.5 * compliance:.6e}")
    print(f"Ratio (should be ~1.0): {total_strain_energy / (0.5 * compliance):.6f}")

    # a sample backwards pass
    f0 = outputs["compliance"]
    vjp = tess_simp_compliance.vector_jacobian_product(
        inputs, {"rho"}, {"compliance"}, {"compliance": 1.0}
    )
    sensitivity = vjp["rho"]

    # Finite difference check
    FD_delta = 1.0e-4
    FD_sensitivity = 0 * sensitivity
    for i in range(num_tests):
        print(i)
        inputs["rho"][i] += FD_delta
        outputs = tess_simp_compliance.apply(inputs)
        fupp = outputs["compliance"]

        if run_central_difference:
            inputs["rho"][i] -= 2.0 * FD_delta
            outputs = tess_simp_compliance.apply(inputs)
            fdown = outputs["compliance"]
            FD_sensitivity[i] = (fupp - fdown) / FD_delta / 2.0
            inputs["rho"][i] += FD_delta

        else:
            FD_sensitivity[i] = (fupp - f0) / FD_delta
            inputs["rho"][i] -= FD_delta

    if num_tests > 0:
        sens = sensitivity[0:num_tests]
        FD_sens = FD_sensitivity[0:num_tests]
        print(sens)
        print(FD_sens)
        errors = sens - FD_sens
        print(errors)
        rel_abs_error = np.abs(errors / sens)
        if run_central_difference:
            print(f"Relative error should be near O({FD_delta})")
        else:
            print(f"Relative error should be O({FD_delta * 10})")
        print(rel_abs_error)


if __name__ == "__main__":
    main()
