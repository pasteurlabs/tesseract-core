# Wrapping MAPDL as a Tesseract

This example wraps Ansys MAPDL as a differentiable Tesseract that can e.g. be used for SIMP (Solid Isotropic Material with Penalization) topology optimization.

The Tesseract computes structural compliance using finite element analysis and provides analytical sensitivities for gradient-based optimization.

```{seealso}
The full code for this Tesseract can be found under `demo/_showcase/ansys-shapeopt/pymapdl` in the [Tesseract Core repository](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/_showcase/ansys-shapeopt/pymapdl).
```

## Prerequisites

This example requires a running ANSYS MAPDL server accessible via gRPC, e.g. via:

```bash
# Linux
/usr/ansys_inc/v241/ansys/bin/ansys241 -grpc -port 50052

# Windows
"C:/Program Files/ANSYS Inc/v241/ansys/bin/winx64/ANSYS241.exe" -grpc -port 50052
```

## Tesseract definition

### Core functionality --- schemas and `apply` function

The Tesseract accepts a hexahedral mesh, density field, and boundary conditions as inputs. The density field `rho` is marked as `Differentiable` to enable gradient computation:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: HexMesh
:language: python
```

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: InputSchema
:language: python
```

The outputs include compliance (the optimization objective), element-wise strain energy, and sensitivity values:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

The `apply` function connects to MAPDL, creates the mesh, assigns SIMP-interpolated materials based on density, applies boundary conditions, solves the static analysis, and extracts results:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: apply
:language: python
```

### SIMP material interpolation

The solver creates a unique material for each element with properties scaled by the density field using the SIMP formula:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: SIMPElasticity._define_simp_materials
:language: python
```

This interpolation scheme penalizes intermediate densities, encouraging binary (solid/void) designs in topology optimization.

### Vector-Jacobian product endpoint

```{seealso}
For more information on differentiable programming in general and vector-Jacobian products specifically, see [](../../introduction/differentiable-programming.md).
```

For compliance minimization, the adjoint solution equals the negative displacement field. This allows computing sensitivities analytically without an explicit adjoint solve:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: SIMPElasticity._calculate_sensitivity
:language: python
```

The cached sensitivity is loaded during the VJP computation and multiplied by the upstream gradient:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: vector_jacobian_product
:language: python
```

Returning VJPs allows us to avoid materializing full Jacobian matrices, and allows for memory-efficient backpropagation in complex Tesseract pipelines.

### Abstract evaluation

The `abstract_eval` function computes output shapes based on input shapes without running the Ansys solver, enabling static analysis and memory pre-allocation:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/tesseract_api.py
:pyobject: abstract_eval
:language: python
```

## Demo script

This demo shows a complete workflow for a cantilever beam problem. It requires you set environment variables pointing to the MAPDL server:

```bash
export MAPDL_HOST=192.168.1.100
export MAPDL_PORT=50052
```

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/pymapdl/demo.py
:pyobject: main
:language: python
```

The script verifies correctness by checking that total strain energy equals half the compliance, which holds for linear elastic problems.

```{figure} ../../../img/pymapdl_simp_compliance.png
:alt: Cantilever beam sensitivity analysis

Element-wise sensitivity gradient (∂compliance/∂ρ) for the cantilever beam test case.
```
