# Wrapping Compiled Code (Fortran Example)

## Context

Many industries---aerospace, energy, automotive---rely on simulation codes written in compiled languages like Fortran that have been developed and validated over decades. These legacy assets represent significant investment and domain expertise. Tesseract provides a straightforward path to wrap such codes, making them accessible through a modern API while preserving their proven numerical implementations.

This example demonstrates how to wrap a Fortran heat equation solver as a Tesseract using subprocess-based integration. The pattern shown here applies to any compiled executable, whether Fortran, C, C++, or other languages.

```{figure} /img/fortran_heat_evolution.gif
:alt: Heat equation evolution
:width: 600px

Animation of the 1D heat equation solution showing temperature diffusing from a hot boundary (left, 100K) through an initially cold material toward a cold boundary (right, 0K).
```

## Example Tesseract (examples/fortran_heat)

### The Fortran solver

The solver implements a 1D transient heat equation using explicit finite differences:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

where $T(x,t)$ is temperature and $\alpha$ is thermal diffusivity. The Fortran code reads parameters from a text file and writes results to a binary file:

```{literalinclude} ../../../../examples/fortran_heat/fortran/heat_solver.f90
:language: fortran
:lines: 4-25
```

### Input and output schemas

The `InputSchema` defines the simulation parameters with physical descriptions and validation. Notably, it includes a CFL stability check---the explicit finite difference scheme requires $r = \alpha \Delta t / \Delta x^2 \leq 0.5$ for numerical stability:

```{literalinclude} ../../../../examples/fortran_heat/tesseract_api.py
:pyobject: InputSchema
:language: python
```

The `OutputSchema` returns the full temperature history along with convenience fields:

```{literalinclude} ../../../../examples/fortran_heat/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

### Subprocess integration

The `apply` function writes input parameters to a temporary file, invokes the compiled Fortran executable, and reads back the binary results. Output from the solver is streamed in real-time to the parent process:

```{literalinclude} ../../../../examples/fortran_heat/tesseract_api.py
:pyobject: apply
:language: python
```

### Build configuration

The `tesseract_config.yaml` installs gfortran and compiles the solver during image build:

```{literalinclude} ../../../../examples/fortran_heat/tesseract_config.yaml
:language: yaml
```

Key points:

- **`extra_packages`**: Installs the Fortran compiler via apt-get
- **`package_data`**: Copies the Fortran source into the container
- **`custom_build_steps`**: Compiles the code during image build, so the executable is ready at runtime

## Results

The solver produces physically meaningful results. Temperature diffuses from the hot left boundary through the material:

```{figure} /img/fortran_heat_profiles.png
:alt: Temperature profiles at different times
:width: 600px

Temperature profiles at different simulation times, showing heat diffusing from the hot boundary (100K at x=0) toward the cold boundary (0K at x=1).
```

## Adapting this pattern

To wrap your own Fortran (or other compiled) code:

1. **Define the I/O protocol**: Decide how Python and the executable will exchange data (text files, binary files, stdin/stdout).
2. **Create schemas**: Map your solver's parameters to Pydantic models with validation.
3. **Configure the build**: Add compiler installation and compilation steps to `tesseract_config.yaml`.
4. **Implement `apply()`**: Write the glue code that marshals data between Python and the executable.

This pattern works for any executable, including MATLAB scripts, Julia programs, or proprietary simulation tools, as long as they can be invoked from the command line.
