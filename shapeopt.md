# Parametric Shape Optimization of Rocket Fins with Ansys SpaceClaim and PyAnsys

Grid fins are lattice-like structures on multi-stage rockets that provide steering control across a wide range of speeds. For example during SpaceX Super Heavy booster re-entry, grid fins experience high dynamic pressure, much higher than during ascent. At this critical flight stage, the fins must maintain structural rigidity under maximum loading to preserve aerodynamic characteristics and enable precise trajectory control back to the landing pad.

| Grid Fin Example | SpaceX Super Heavy Rocket    |
| ------------- | ------------- |
| ![grid](imgs/grid_fin_example.png) | ![star](imgs/superheavy.png) |

This case study demonstrates a gradient-based optimization workflow combining Ansys tools with Tesseract-driven differentiable programming. The goal is to maximize grid fin stiffness while maintaining a fixed mass constraint of 8 bars. Higher stiffness reduces deformation during Max-Q, keeping lift and drag coefficients consistent for reliable aerodynamic control.

Each bar is defined by start and end angular positions, giving us 16 design parameters to optimize. Below are two example starting configurations:

| Grid initial conditions | Random initial conditions    |
| ------------- | ------------- |
| ![grid](imgs/grid_surface.png) | ![star](imgs/rnd_surface.png) |

The simulation uses fixed boundary conditions at the knuckles (where the fin attaches to the rocket) and an out-of-plane load at the fin tip. This load placement approximates the aerodynamic forces while decoupling them from the bar geometry during optimization:

![BCs](imgs/boundary_conditions.png)

The simulation uses a linear elastic finite element solver with small deformation assumptions. To maximize stiffness, we minimize compliance, which is the inverse measure of structural rigidity.

## Workflow

This workflow demonstrates an end-to-end gradient-based optimization connecting Ansys SpaceClaim for parametric geometry generation with PyAnsys for finite element analysis. Tesseract acts as glue between components, by packaging each component into a unified interface with built-in differentiation support. By composing Tesseracts with [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), we can leverage automatic differentiation across the entire pipeline:

![Workflow](imgs/workflow_1.png)


The workflow uses three Tesseract components:

- **Ansys SpaceClaim Tesseract**: Takes design parameters and generates the grid fin geometry through a SpaceClaim script, returning a triangular surface mesh (vertices and faces). Described in more detail in the [Tesseract docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/ansys_integration/spaceclaim_tess.html).

- **SDF and Finite Difference Tesseract**: Converts the surface mesh into a signed distance field (SDF) on a regular grid. Additionally computes gradients with respect to design parameters using finite differences. Takes another Tesseract (here: SpaceClaim Tesseract) as input, which makes it a higher-order Tesseract. This Tesseract can work with any mesh generator Tesseract that matches the expected interface.

- **PyMAPDL Tesseract**: Takes a hex mesh and boundary conditions as inputs, then solves the linear elasticity problem using PyMAPDL. Returns strain energy per cell and total compliance, with full gradient support for optimization via an analytical adjoint. Described in more detail in the [Tesseract docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/ansys_integration/pymapdl_tess.html).

Between these Tesseracts, standard Python code handles hex mesh generation, boundary condition setup, and density derivation from the SDF. The hex mesh generation and boundary condition setup operations don't require differentiation since we optimize with respect to field quantities on mesh cells, not the mesh structure itself. The density function uses a sigmoid-like mapping and is differentiated with [JAX](github.com/google/jax)'s automatic differentiation.


## Optimization

We first compare the two initial configurations. The regular grid (compliance: 61.9) significantly outperforms the random arrangement (compliance: 87.0). The plots below show strain energy and compliance sensitivities with respect to density. Negative sensitivity values indicate where adding material would reduce compliance. Note the tendency to thicken bars along their length, though this isn't achievable under our angular parametrization.

![Workflow](imgs/sim_comparision.png)

We run gradient-based optimization using Adam (learning rate: 0.01, 80 iterations) on both initial conditions.

| Grid IC | Random IC |
| ------------- | ------------- |
| ![grid](imgs/mesh_grid_adam.gif) | ![star](imgs/mesh_rnd_adam.gif) |

Both runs converge to similar asymmetric solutions. The optimizer concentrates material near the knuckle attachments to maximize local stiffness, consistent with the strain energy distributions showing highest concentrations at the fixed boundaries. The emergence of a grid-like structure from random initial conditions suggests the solver finds a near-optimal topology.

However, the resulting geometries lack symmetry and would be difficult to manufacture. While increasing the number of optimization iterations might improve symmetry, explicitly enforcing symmetry constraints on the parameters would be more effective.

The compliance evolution for both initial conditions is shown below:

![loss](imgs/conv_rnd.png)

Both configurations converge to similar final compliance values. The optimization reveals three structural behaviors for optimal load paths under these boundary conditions:

- **Emergent Orthogonality**: Regardless of initialization, the topology settles into roughly equal numbers of lateral and longitudinal members. The random initialization is particularly revealing, where bars initially spanning 180 degrees reorganize into a nearly orthogonal pattern.

- **Diagonal Lateral Load Paths**: Lateral bars orient diagonally relative to the opposing knuckle, creating direct load paths that efficiently transfer tip moments to the fixed boundary.

- **Root Reinforcement**: Longitudinal bars align vertically and cluster near the knuckles. Concentrating material at the fixed boundary stiffens the fin root where strain energy gradients are highest.

## Results

The optimized designs achieve higher stiffness at constant mass through non-uniform bar distributions. Lower compliance translates to reduced deformation under load, maintaining consistent aerodynamic coefficients during re-entry. However, the asymmetric topologies present manufacturing challenges and would provide unequal control authority in different flight directions.

The final step in shape optimization is translating computational insights into manufacturable designs that satisfy practical constraints the optimizer didn't account for. We interpret the three structural behaviors (Emergent Orthogonality, Diagonal Lateral Load Paths, Root Reinforcement) into a symmetric, manufacturable geometry:

| Final Geometry | Comparison with optimization results |
| ------------- | ------------- |
| ![manual_result](imgs/surface_radial_manual.png) | ![conv_with_manual](imgs/conv_with_manual.png) |

Running this geometry through the Tesseract pipeline yields a compliance of 49.8, that is, a design that is 24% stiffer than the original grid and 75% stiffer than random bars. While not matching the fully optimized result, this design balances performance with manufacturability and symmetric aerodynamic characteristics.

This demonstrates how gradient-based optimization with the Tesseract ecosystem and Ansys software can guide practical engineering decisions, even when the final design incorporates constraints beyond the optimization problem.

## Why Tesseract?

This case study demonstrates several capabilities that make Tesseract practical for simulation-driven design workflows:

- **Composability**: Each component (geometry generation, meshing, FEM) is independently packaged. You can swap SpaceClaim for another CAD tool or replace PyMAPDL with a different solver without rewriting the pipeline. We've validated this by running the same workflow with [PyVista](https://docs.pyvista.org/) geometry and the [JAX-FEM](https://github.com/deepmodeling/jax-fem) solver.

![Workflow](imgs/workflow_2.png)

- **Gradient Support**: Tesseract's differentiation interface connects tools not originally designed for gradient-based optimization. This workflow combines analytic adjoints (PyMAPDL), finite differences (SDF conversion), and automatic differentiation (JAX glue code) into a complete gradient chain for the optimization problem.

- **Heterogeneous Compute**: Tesseract integrates across different operating systems and environments. This workflow runs PyAnsys tools on Windows (with specific licenses and installation requirements) while executing optimization logic and Tesseract orchestration on Linux.

- **Dependency Management**: Setting up workflows with multiple commercial and open-source packages typically creates dependency conflicts. Each Tesseract is self-contained with its own environment, isolating Python packages and system requirements (like OpenGL libraries for meshing). This makes workflows reproducible and eliminates version conflicts.

- **Team Collaboration**: Tesseract uses a contract-first approach where each component's inputs and outputs are defined upfront through schemas. Different engineers can develop components independently against these interfaces, reducing integration issues when combining work.

This approach generalizes beyond structural optimization to virtually any workflow involving simulation tools, multiphysics coupling, design exploration, or inverse problems.
