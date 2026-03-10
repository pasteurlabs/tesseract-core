# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# create a temp directory to work in
import logging
import os
import tempfile
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import numpy as np
import pyvista as pv
from ansys.mapdl.core import Mapdl
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType

# Set up module logger
logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def log_timing(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to log the wall time of method execution."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        method_name = func.__name__
        start_time = time.time()
        logger.info(f"Starting {method_name}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Completed {method_name} in {elapsed:.4f} seconds")
        return result

    return wrapper


class HexMesh(BaseModel):
    """Hexahedral mesh representation with 8-node elements."""

    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    elems: Array[(None, 8), Int32] = Field(
        description="Array of hexahedral elems defined by indices into the points array."
    )
    n_points: Int32 = Field(
        default=0, description="Number of valid points in the points array."
    )
    n_elems: Int32 = Field(
        default=0, description="Number of valid elems in the elems array."
    )


class InputSchema(BaseModel):
    """Input specification for SIMP compliance computation via ANSYS MAPDL.

    Defines all inputs required to run a static structural analysis with SIMP
    material interpolation. Requires a running ANSYS MAPDL server accessible
    via gRPC protocol.

    Example MAPDL server startup:
        Linux: /usr/ansys_inc/v241/ansys/bin/ansys241 -grpc -port 50052
        Windows: "C:/Program Files/ANSYS Inc/v241/ansys/bin/winx64/ANSYS241.exe" -grpc -port 50052

    The density field (rho) is marked as Differentiable, enabling gradient
    computation via vector-Jacobian products for topology optimization.
    """

    host: str = Field(description="The IP of your MAPDL grpc server.")
    port: str = Field(description="The port of your MAPDL grpc server.")
    rho: Differentiable[
        Array[
            (
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="2D density field for topology optimization")
    von_neumann_mask: Array[(None,), Int32] = Field(
        description="Mask for von Neumann boundary conditions",
    )
    von_neumann_values: Array[(None, None), Float32] = Field(
        description="Values for von Neumann boundary conditions",
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

    E0: float = Field(
        default=1.0,
        description="Base Young's modulus in Pa for SIMP material interpolation.",
    )

    rho_min: float = Field(
        default=1e-6,
        description="Minimum density value to avoid singular stiffness matrix in SIMP.",
    )

    p: float = Field(
        default=3.0,
        description="SIMP penalty parameter for material interpolation (default: 3.0).",
    )

    log_level: str = Field(
        default="WARNING",
        description="Logging level for output messages (DEBUG, INFO, WARNING, ERROR).",
    )

    vtk_output: str | None = Field(
        default=None,
        description="The path to write the results in VTK format.",
    )


class OutputSchema(BaseModel):
    """Output specification from SIMP compliance analysis.

    Contains the results of a static structural analysis including compliance
    (objective function for minimum compliance optimization), element-wise
    strain energy, and sensitivity information for gradient-based optimization.

    The compliance is marked as Differentiable to support automatic differentiation
    through the ANSYS solver using cached sensitivities from the forward pass.
    """

    compliance: Differentiable[
        Array[
            (),
            Float32,
        ]
    ] = Field(
        description="Compliance of the structure (scalar), a measure of structural flexibility (inverse of stiffness)"
    )
    strain_energy: Array[(None,), Float32] = Field(
        description="Element-wise strain energy array of shape (n_elements,)"
    )
    sensitivity: Array[(None,), Float32] = Field(
        description="Derivative of compliance with respect to each density variable: shape (n_elements,)"
    )


class SIMPElasticity:
    """SIMP-based topology optimization solver using ANSYS MAPDL."""

    def __init__(self, inputs: InputSchema, mapdl: Mapdl) -> None:
        """Initialize the SIMP elasticity solver.

        Args:
            inputs: Input parameters for the simulation
            mapdl: Active MAPDL instance
        """
        # Store inputs
        self.inputs = inputs
        self.mapdl = mapdl

        # Extract input parameters
        self.rho = inputs.rho
        self.dirichlet_mask = inputs.dirichlet_mask
        self.dirichlet_values = inputs.dirichlet_values
        self.von_neumann_mask = inputs.von_neumann_mask
        self.von_neumann_values = inputs.von_neumann_values
        self.hex_mesh = inputs.hex_mesh
        if self.hex_mesh.n_elems == 0:
            self.hex_mesh.n_elems = self.hex_mesh.elems.shape[0]
        if self.hex_mesh.n_points == 0:
            self.hex_mesh.n_points = self.hex_mesh.points.shape[0]
        self.E0 = inputs.E0
        self.rho_min = inputs.rho_min
        self.p = inputs.p
        self.log_level = inputs.log_level
        self.vtk_output = inputs.vtk_output

        # Configure logger
        logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        # Add console handler if not already present
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Initialize result storage
        self.element_numbers = None  # Store actual MAPDL element numbers
        self.n_elements = None
        self.node_numbers = None  # Store actual MAPDL node numbers
        self.n_nodes = None
        self.displacement_constraints = None
        self.nodal_displacement = None
        self.nodal_force = None
        self.strain_energy = None
        self.compliance = None
        self.sensitivity = None
        self.pvmesh = None

    @log_timing
    def solve(self) -> OutputSchema:
        """Run the complete SIMP analysis workflow."""
        logger.info("Starting SIMP elasticity analysis...")

        self._create_mesh()
        self._define_simp_materials()
        self._assign_materials_to_elements()
        self._apply_boundary_conditions()
        self._run_analysis()
        self._extract_displacement_constraints()
        self._extract_nodal_displacement()
        self._extract_nodal_force()
        self._extract_strain_energy()
        self._calculate_compliance()
        self._calculate_sensitivity()
        if self.vtk_output:
            self._create_pvmesh()

        logger.info("SIMP analysis complete!")
        logger.debug(f"MAPDL status: {self.mapdl}")

        return self._build_output_schema()

    @log_timing
    def _define_simp_materials(self) -> None:
        """Define materials using SIMP approach with batch commands for efficiency."""
        # Flatten rho array and ensure it matches the number of elements
        rho_flat = np.array(self.rho).flatten()

        if len(rho_flat) != self.n_elements:
            raise ValueError(
                f"Density field size {len(rho_flat)} does not match "
                f"number of elements {self.n_elements}"
            )

        # Apply minimum density constraint to avoid singular stiffness matrix
        # SIMP formula: E = E0 * rho^p
        E_values = self.E0 * (self.rho_min + (1 - self.rho_min) * (rho_flat**self.p))
        dens_values = self.rho_min + (1 - self.rho_min) * rho_flat

        # Build all commands as a list for batch execution
        commands = []
        for i in range(len(rho_flat)):
            mat_id = i + 1
            commands.append(f"MP,EX,{mat_id},{E_values[i]}")
            commands.append(f"MP,DENS,{mat_id},{dens_values[i]}")
            commands.append(f"MP,NUXY,{mat_id},0.3")
        self.mapdl.input_strings(commands)

    @log_timing
    def _define_element(self) -> None:
        """Define element type for structural analysis.

        Sets element type 1 to SOLID185, a 8-node 3D structural solid element
        with linear displacement behavior.

        Note: SOLID186, a 20-node 3D structural solid element
        with quadratic displacement behavior, would be an improvement.
        """
        self.mapdl.et(1, "SOLID185")

    @log_timing
    def _assign_materials_to_elements(self) -> None:
        """Assign materials to elements after meshing using batch commands.

        For a structured mesh with Nx x Ny x Nz elements, this assigns
        material ID to each element based on the stored element numbers.
        Material i+1 is assigned to the element at position i in the element_numbers array.
        """
        # Get all element numbers from the mesh
        self.mapdl.allsel()

        # Use actual element numbers instead of assuming sequential 1, 2, 3, ...
        commands = []
        for i in range(self.n_elements):
            elem_id = self.element_numbers[i]  # Use actual element number from mesh
            mat_id = i + 1  # Material index corresponds to density array index
            commands.append(f"EMODIF,{elem_id},MAT,{mat_id}")

        self.mapdl.input_strings(commands)

    @log_timing
    def _create_mesh(self) -> None:
        """Create hexahedral mesh directly using MAPDL commands.

        Creates nodes and elements from hex_mesh using direct MAPDL commands (N and E).
        This ensures perfect 1:1 correspondence: node i+1 = points[i], element i+1 = elems[i].
        """
        self.mapdl.prep7()
        self._define_element()

        logger.info(f"Creating {self.hex_mesh.n_points} nodes...")

        # Create nodes using batch commands for efficiency
        node_commands = []
        for i in range(self.hex_mesh.n_points):
            x, y, z = self.hex_mesh.points[i]
            # Node number i+1 corresponds to point index i
            node_commands.append(f"N,{i + 1},{x},{y},{z}")

        self.mapdl.input_strings(node_commands)

        logger.info(f"Creating {self.hex_mesh.n_elems} hexahedral elements...")

        # Create elements using batch commands
        element_commands = []
        for i in range(self.hex_mesh.n_elems):
            # Get node indices for this element (0-based)
            node_indices = self.hex_mesh.elems[i]
            # Convert to node numbers (1-based)
            node_nums = [int(idx + 1) for idx in node_indices]
            # Element number i+1 corresponds to face index i
            element_commands.append(f"E,{','.join(map(str, node_nums))}")

        self.mapdl.input_strings(element_commands)

        # Verify the mesh was created correctly
        self.element_numbers = self.mapdl.mesh.enum
        self.node_numbers = self.mapdl.mesh.nnum
        self.n_elements = len(self.element_numbers)
        self.n_nodes = len(self.node_numbers)

        logger.info(f"Mesh created: {self.n_nodes} nodes, {self.n_elements} elements")

        # Validate counts
        if self.n_nodes != self.hex_mesh.n_points:
            raise ValueError(
                f"Node count mismatch: created {self.n_nodes}, expected {self.hex_mesh.n_points}"
            )

        if self.n_elements != self.hex_mesh.n_elems:
            raise ValueError(
                f"Element count mismatch: created {self.n_elements}, expected {self.hex_mesh.n_elems}"
            )

        # Verify sequential numbering (should always be true with direct creation)
        is_sequential_nodes = np.array_equal(
            self.node_numbers, np.arange(1, self.n_nodes + 1)
        )
        is_sequential_elements = np.array_equal(
            self.element_numbers, np.arange(1, self.n_elements + 1)
        )

        if not is_sequential_nodes or not is_sequential_elements:
            logger.warning(
                f"Non-sequential numbering detected: nodes={is_sequential_nodes}, elements={is_sequential_elements}"
            )
        else:
            logger.info(
                "Verified: Node and element numbering is sequential (1, 2, 3, ...)"
            )

    @log_timing
    def _apply_boundary_conditions(self) -> None:
        """Apply boundary conditions and forces.

        Since nodes are created with direct correspondence (node i+1 = points[i]),
        we can directly reference node numbers in D and F commands.
        """
        commands = []

        # Build Dirichlet boundary condition commands
        n_dirichlet = len(self.dirichlet_mask)
        logger.info(f"Applying Dirichlet BCs to {n_dirichlet} nodes")

        for i, x in zip(self.dirichlet_mask, self.dirichlet_values, strict=True):
            node = i + 1
            commands.append(f"D,{node},ALL,{x}")

        # Build von Neumann boundary condition (force) commands
        n_von_neumann = len(self.von_neumann_mask)
        logger.info(f"Applying von Neumann BCs to {n_von_neumann} nodes")

        for i, f in zip(self.von_neumann_mask, self.von_neumann_values, strict=True):
            node = i + 1
            if f[0] != 0:
                commands.append(f"F,{node},FX,{f[0]}")
            if f[1] != 0:
                commands.append(f"F,{node},FY,{f[1]}")
            if f[2] != 0:
                commands.append(f"F,{node},FZ,{f[2]}")

        logger.debug(f"Executing {len(commands)} boundary condition commands")
        self.mapdl.input_strings(commands)

    @log_timing
    def _run_analysis(self) -> None:
        """Run static structural analysis."""
        self.mapdl.slashsolu()
        self.mapdl.allsel()  # making sure all nodes and elements are selected.
        self.mapdl.antype("STATIC")
        self.mapdl.ematwrite("YES")
        output = self.mapdl.solve()
        self.mapdl.save("my_analysis")
        self.mapdl.finish()

        logger.debug(f"Analysis output: {output}")

    @log_timing
    def _extract_strain_energy(self) -> None:
        """Extract strain energy for all elements and save to file."""
        logger.debug("Extracting strain energy data...")

        # Load first load step, first substep
        self.mapdl.post1()
        self.mapdl.set(1, 1)

        # Create element table for strain energy
        self.mapdl.etable("SENE", "SENE")

        # Create array parameter and populate with strain energy values
        self.mapdl.run(f"*DIM,strain_e,ARRAY,{self.n_elements},1")
        self.mapdl.run("*VGET,strain_e,ELEM,1,ETAB,SENE, , ,2")

        # Write strain energy to text file
        with self.mapdl.non_interactive:
            self.mapdl.run("*CFOPEN,strain_energy,txt")
            self.mapdl.run("*VWRITE,strain_e(1,1)")
            self.mapdl.run("(E15.7)")
            self.mapdl.run("*CFCLOS")

        # Download the file from the MAPDL working directory
        self.mapdl.download("strain_energy.txt", ".")
        strain_energy_sorted = np.loadtxt("strain_energy.txt")
        sorted_element_numbers = np.sort(self.element_numbers)
        reorder_indices = np.searchsorted(sorted_element_numbers, self.element_numbers)
        self.strain_energy = strain_energy_sorted[reorder_indices]

        logger.debug(
            f"Strain energy reordered: first 5 values = {self.strain_energy[:5]}"
        )

    @log_timing
    def _extract_nodal_displacement(self) -> None:
        """Extract nodal displacements from ANSYS MAPDL results."""
        nodal_displacement = np.zeros((self.n_nodes, 3))
        nnum, disp = self.mapdl.result.nodal_displacement(0)  # 0 for first result set
        # populate nodal_displacement using vectorized NumPy indexing
        nodal_displacement[nnum - 1] = disp[:, 0:3]
        self.nodal_displacement = nodal_displacement

    @log_timing
    def _extract_nodal_force(self) -> None:
        """Extract nodal reaction and applied forces from ANSYS MAPDL results."""
        nodal_force = np.zeros((self.n_nodes, 3))
        # populate force using vectorized NumPy advanced indexing
        nnum, dof_idx, f = self.mapdl.result.nodal_input_force(0)
        node_indices = nnum - 1
        dof_indices = dof_idx - 1
        nodal_force[node_indices, dof_indices] = f
        self.nodal_force = nodal_force

    @log_timing
    def _extract_displacement_constraints(self) -> None:
        displacement_constraints = np.zeros((self.n_nodes, 3))
        # Get a list of all nodal constraints and process with optimized mapping
        nodal_constraints = self.mapdl.get_nodal_constrains()

        constraint_map = {"UX": 0, "UY": 1, "UZ": 2}
        for idx, field, _, _ in nodal_constraints:
            dof = constraint_map.get(field)
            if dof is not None:
                node_num = int(idx) - 1
                displacement_constraints[node_num, dof] = 1.0
        self.displacement_constraints = displacement_constraints

    @log_timing
    def _calculate_compliance(self) -> None:
        """Calculate compliance from nodal forces and displacements."""
        self.compliance = np.dot(
            self.nodal_force.flatten(), self.nodal_displacement.flatten()
        )

    @log_timing
    def _calculate_sensitivity(self) -> None:
        """Calculate sensitivity of compliance with respect to density.

        For this special case, the adjoint solution is equal to -U,
        so the sensitivity is equal to:
             - U_e^T * (dK_e / drho) * U_e
              = - (p/rho) * U_e^T * K_e * U_e
              = - 2 * (p/rho) * strain_energy
        """
        inverse_rho = np.nan_to_num(1 / self.rho.flatten(), nan=0.0)
        self.sensitivity = -2.0 * self.p * inverse_rho * self.strain_energy.flatten()

        # Cache sensitivity in temporary directory for use in VJP
        sensitivity_path = os.path.join(tempfile.gettempdir(), "sensitivity.npy")
        np.save(sensitivity_path, self.sensitivity)

    @log_timing
    def _create_pvmesh(self) -> None:
        """Create PyVista grid with analysis results."""
        logger.debug("Creating PyVista results grid...")

        # Enter POST1 to access solution results
        self.mapdl.post1()
        self.mapdl.set(1, 1)

        # Try to get mesh directly from result object to avoid disk I/O
        # The result.grid property provides direct access to PyVista mesh
        try:
            self.pvmesh = self.mapdl.result.grid
            logger.debug("Successfully loaded mesh directly from result.grid")
        except (AttributeError, Exception) as e:
            # Fallback to file-based approach if direct access fails
            logger.debug(f"Direct grid access failed ({e}), using file-based approach")
            self.mapdl.download_result(".")

            show_progress = logger.level <= logging.INFO
            self.mapdl.result.save_as_vtk("file.vtk", progress_bar=show_progress)

            self.pvmesh = pv.read("file.vtk")

        # add attributes
        rho_flat = np.array(self.rho).flatten()
        self.pvmesh.cell_data["density"] = self.__convert_celldata_for_pv(rho_flat)
        self.pvmesh.cell_data["strain_energy"] = self.__convert_celldata_for_pv(
            self.strain_energy
        )
        self.pvmesh.cell_data["sensitivity"] = self.__convert_celldata_for_pv(
            self.sensitivity
        )
        self.pvmesh.point_data["displacement_constraints"] = (
            self.displacement_constraints
        )
        self.pvmesh.point_data["nodal_displacement"] = self.nodal_displacement
        self.pvmesh.point_data["nodal_force"] = self.nodal_force

        # Export to VTK
        self.pvmesh.save(self.vtk_output)

        logger.info(f"Exported results to {self.vtk_output}")

    def __convert_celldata_for_pv(self, elem_array: np.array) -> np.array:
        """Convert element data array to match PyVista mesh cell ordering."""
        # PyVista cells may be ordered differently than our element_numbers array
        # Get PyVista cell to MAPDL element number mapping
        pv_elem_nums = self.pvmesh.cell_data["ansys_elem_num"]

        # Create mapping: for each PyVista cell, find the corresponding density
        array_for_pv = np.zeros(len(pv_elem_nums))
        for pv_idx, mapdl_elem_num in enumerate(pv_elem_nums):
            # Find where this MAPDL element number appears in self.element_numbers
            array_idx = np.where(self.element_numbers == mapdl_elem_num)[0][0]
            array_for_pv[pv_idx] = elem_array[array_idx]
        return array_for_pv

    def _build_output_schema(self) -> OutputSchema:
        """Build and return the output schema."""
        return OutputSchema(
            compliance=self.compliance,
            strain_energy=self.strain_energy.flatten(),
            sensitivity=self.sensitivity,
        )


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the pymapdl Tesseract for SIMP-Elasticity analysis."""
    # Initialize MAPDL
    mapdl = Mapdl(inputs.host, port=inputs.port)
    mapdl.clear()

    # Create solver instance and run analysis
    solver = SIMPElasticity(inputs, mapdl)
    return solver.solve()


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute vector-Jacobian product for backpropagation through ANSYS solve."""
    gradients = {}
    assert vjp_inputs == {"rho"}
    assert vjp_outputs == {"compliance"}

    # Load the cached sensitivity (∂compliance/∂rho) from temporary directory
    # This was computed and saved in _calculate_sensitivity()
    sensitivity_path = os.path.join(tempfile.gettempdir(), "sensitivity.npy")
    sensitivity = np.load(sensitivity_path)

    # Clean up the temporary file after loading
    try:
        os.unlink(sensitivity_path)
    except FileNotFoundError:
        pass  # Already deleted, no problem

    grad_rho_flat = cotangent_vector["compliance"] * sensitivity

    # Reshape to match input rho shape
    gradients["rho"] = grad_rho_flat.reshape(inputs.rho.shape)

    return gradients


def abstract_eval(abstract_inputs: Any) -> dict:
    """Calculate output shapes and dtypes without executing the forward pass."""
    n_elems = abstract_inputs.hex_mesh.elems.shape[0]
    return {
        "compliance": ShapeDType(shape=(), dtype="float32"),
        "strain_energy": ShapeDType(shape=(n_elems,), dtype="float32"),
        "sensitivity": ShapeDType(shape=(n_elems,), dtype="float32"),
    }
